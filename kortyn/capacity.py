"""Kortyn capacity tool: how many robots can one GPU serve, and with what expected success?

One command, three inputs:

* a **dose-response curve** (format ``kortyn-dose-response-v1``): success of the policy per task at fixed
  end-to-end latencies, as counts (successes and episodes per latency level);
* a **network profile**: a Kortyn network scenario (built-in name such as ``buero_wlan`` - office Wi-Fi, RTT
  20 +/- 10 ms - or a scenario YAML file) with the Brain's return-trip margin, optionally one Wi-Fi medium
  shared by all robots;
* a **GPU measurement** (format ``kortyn-gpu-benchmark-v2`` from ``notebooks/gpu_benchmark.ipynb``, or a
  ``kortyn-batch-service-v1`` table), or an assumed service time ``s(b) = s1 * (1 + beta * (b - 1))``.

The tool simulates fleets of 1, 2, 3, ... robots sharing that GPU in simulation time - the queue model of
:mod:`kortyn.fleet`: the robots' request rules, the network, the Brain's batching scheduler - predicts every
robot's success from the curve at the end-to-end latencies its requests saw, and reports per scheduling
strategy (and GPU configuration):

* **capacity** (the criterion of phases 4-5): the largest fleet for which it and every smaller fleet keep at
  most 1 % of the plans late *and* the worst robot at most ``tolerance_pp`` below a single robot. The
  conservative capacity uses the 97.5-% quantile of a parametric bootstrap of the curve (successes redrawn
  binomially per level, curve refitted);
* **latency capacity** (optional ``latency_slo``): the largest fleet for which it and every smaller fleet keep
  the pooled end-to-end latency quantile (e.g. p99) within a bound and admit every robot;
* the expected success of the fleet (mean robot and worst robot) and the latency quantiles at every size.

The curve model is the one of phases 4-5 (:mod:`kortyn.dose_curve`): per task a log-logistic decay over the
effective latency (latencies act in whole control ticks), an isotonic fallback if the fit is poor, linear
decay to 0 beyond the latency budget; tasks weighted equally.

    kortyn-capacity --config examples/capacity/example.yaml
    kortyn-capacity --config examples/capacity/example.yaml --gpu results/phase5/gpu/<file>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from kortyn import dose_curve as dc
from kortyn.config import ConfigError, decode, load_scenario
from kortyn.fleet import (
    HISTOGRAM_BIN_MS,
    MISS_SHARE_LIMIT,
    BatchingConfig,
    EpisodeLengthConfig,
    FleetAgentConfig,
    FleetSimConfig,
    GpuConfig,
    NetworkChoice,
    RunKey,
    RunResult,
    StrategyConfig,
    load_episode_lengths,
    run_fleet,
)
from kortyn.fleet_brain import BatchModelError
from kortyn.fleet_predict import PredictionError
from kortyn.gpu_benchmark import V2_FORMAT, GpuBenchmark, GpuBenchmarkError

LOG = logging.getLogger("kortyn.capacity")

DOSE_FORMAT = "kortyn-dose-response-v1"
MEASURED_BATCHING = "gemessen"  # batching label of a measured GPU (the measurement holds the batching gain)
ALL_VALID = "all-valid"

Floats = npt.NDArray[np.float64]


class CapacityError(RuntimeError):
    """Invalid inputs of the capacity tool, reported without a traceback."""


def _default_strategies() -> dict[str, StrategyConfig]:
    """The strategies of phases 3-5: FIFO, FIFO + batching, EDF + batching, EDF + batching + admission."""
    return {
        "a": StrategyConfig(description="FIFO", scheduler="fifo"),
        "b": StrategyConfig(
            description="FIFO + batching", scheduler="fifo_batch", max_batch=16, max_wait_ms=25.0
        ),
        "d": StrategyConfig(description="EDF + batching", scheduler="edf_batch", max_batch=16),
        "e": StrategyConfig(
            description="EDF + batching + admission control",
            scheduler="edf_batch",
            max_batch=16,
            admission=True,
        ),
    }


# --------------------------------------------------------------------------------------------- config
@dataclass
class GpuSpec:
    """A measurement file or an assumed affine service time: exactly one of ``measurement`` and ``s1_ms``."""

    measurement: str | None = None
    configuration: str | None = None  # v2 file: a configuration name, "all-valid", or None (fastest valid)
    s1_ms: float | None = None
    beta: float = 0.15
    service_cv: float = 0.05
    label: str = "gpu"

    def __post_init__(self) -> None:
        if (self.measurement is None) == (self.s1_ms is None):
            raise ConfigError("gpu needs exactly one of 'measurement' (file) and 's1_ms' (assumption)")
        if self.s1_ms is not None and self.s1_ms <= 0:
            raise ConfigError("gpu s1_ms must be > 0")
        if not 0.0 <= self.beta <= 1.0 or self.service_cv < 0:
            raise ConfigError("gpu beta must be in [0, 1] and service_cv >= 0")
        if not self.label:
            raise ConfigError("gpu label must not be empty")


@dataclass
class NetworkSpec:
    scenario: str = "buero_wlan"  # built-in scenario name or a scenario YAML file
    margin_ms: float = 15.0
    wlan_mbit_s: float | None = None
    request_bytes: int = 0
    response_bytes: int = 0


@dataclass
class LatencySlo:
    """Latency target: the pooled ``quantile`` of the end-to-end latency at most ``max_ms``."""

    quantile: float = 0.99
    max_ms: float = 150.0
    robots: int | None = None  # optional verdict: is the latency capacity at least this many robots?

    def __post_init__(self) -> None:
        if not 0.5 <= self.quantile < 1.0:
            raise ConfigError("latency_slo.quantile must be in [0.5, 1)")
        bins = self.max_ms / HISTOGRAM_BIN_MS
        if self.max_ms <= 0 or abs(bins - round(bins)) > 1e-9:
            raise ConfigError(f"latency_slo.max_ms must be a positive multiple of {HISTOGRAM_BIN_MS:g} ms")
        if self.robots is not None and self.robots < 1:
            raise ConfigError("latency_slo.robots must be >= 1")


@dataclass
class CriteriaSpec:
    tolerance_pp: float = 3.0
    latency_slo: LatencySlo | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.tolerance_pp < 100.0:
            raise ConfigError("criteria.tolerance_pp must be in (0, 100)")


@dataclass
class FleetSpec:
    max_robots: int = 32
    replications: int = 5
    horizon_s: float = 630.0
    window_start_s: float = 30.0
    start_spread_s: float = 10.0
    seed: int = 3000
    batch_limit: int = 32

    def __post_init__(self) -> None:
        if self.max_robots < 1 or self.replications < 1 or self.batch_limit < 1:
            raise ConfigError("fleet needs max_robots, replications and batch_limit >= 1")
        if not 0.0 <= self.window_start_s < self.horizon_s or self.start_spread_s < 0:
            raise ConfigError("fleet needs 0 <= window_start_s < horizon_s and start_spread_s >= 0")


@dataclass
class CapacityConfig:
    dose_response: str
    gpu: GpuSpec
    network: NetworkSpec = field(default_factory=NetworkSpec)
    agent: FleetAgentConfig = field(default_factory=FleetAgentConfig)
    episode_lengths: EpisodeLengthConfig = field(default_factory=lambda: EpisodeLengthConfig(values=[520]))
    strategies: dict[str, StrategyConfig] = field(default_factory=_default_strategies)
    criteria: CriteriaSpec = field(default_factory=CriteriaSpec)
    fleet: FleetSpec = field(default_factory=FleetSpec)
    bootstrap: int = 200
    bootstrap_seed: int = 4242
    output: str | None = None

    def __post_init__(self) -> None:
        if not self.strategies:
            raise ConfigError("at least one strategy")
        if self.bootstrap < 0:
            raise ConfigError("bootstrap must be >= 0 (0: no conservative capacity)")


def load_capacity_config(path: Path | str) -> CapacityConfig:
    import yaml

    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read capacity config {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    return decode(CapacityConfig, data, str(source))


# --------------------------------------------------------------------------------------------- curve
@dataclass(frozen=True)
class TaskLevels:
    """Counts of one task: successes out of episodes at fixed end-to-end latencies."""

    name: str
    latency_ms: tuple[float, ...]
    successes: tuple[int, ...]
    episodes: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.latency_ms)
        if n == 0 or len(self.successes) != n or len(self.episodes) != n:
            raise CapacityError(f"task {self.name!r}: need matching latency, success and episode counts")
        for latency, wins, trials in zip(self.latency_ms, self.successes, self.episodes, strict=True):
            if latency < 0 or not math.isfinite(latency) or trials < 1 or not 0 <= wins <= trials:
                raise CapacityError(f"task {self.name!r}: invalid level ({latency} ms, {wins}/{trials})")


@dataclass(frozen=True)
class DoseResponseFile:
    source: str
    description: str
    spec: dc.CurveSpec
    tasks: tuple[TaskLevels, ...]


def read_dose_response(path: Path | str) -> DoseResponseFile:
    """A ``kortyn-dose-response-v1`` file."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapacityError(f"cannot read the dose-response curve {source}: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != DOSE_FORMAT:
        raise CapacityError(f"{source}: format must be {DOSE_FORMAT!r}")
    try:
        spec = dc.CurveSpec(
            tick_ms=float(data.get("tick_ms", 50.0)),
            fit_max_ms=float(data.get("fit_max_ms", 1250.0)),
            zero_from_ms=float(data.get("zero_from_ms", 1500.0)),
            gof_alpha=float(data.get("gof_alpha", 0.01)),
        )
        tasks = tuple(
            TaskLevels(
                name=str(task["name"]),
                latency_ms=tuple(float(level["latency_ms"]) for level in task["levels"]),
                successes=tuple(int(level["successes"]) for level in task["levels"]),
                episodes=tuple(int(level["episodes"]) for level in task["levels"]),
            )
            for task in data["tasks"]
        )
    except (KeyError, TypeError, ValueError, PredictionError) as exc:
        raise CapacityError(f"{source}: malformed dose-response curve: {exc}") from exc
    if not tasks:
        raise CapacityError(f"{source}: no task")
    return DoseResponseFile(str(source), str(data.get("description", "")), spec, tasks)


def _fit_counts(
    task: TaskLevels, spec: dc.CurveSpec, successes: Sequence[float]
) -> tuple[dc.TaskFunction, str]:
    """The phase-4 rule on counts: log-logistic fit, isotonic fallback if the deviance test rejects it."""
    x = dc.effective_latency_ms(task.latency_ms, spec.tick_ms)
    pooled: dict[float, list[float]] = {}
    for latency, wins, trials in zip(x, successes, task.episodes, strict=True):
        if latency <= spec.fit_max_ms + 1e-9:
            entry = pooled.setdefault(float(latency), [0.0, 0.0])
            entry[0] += float(wins)
            entry[1] += float(trials)
    if len(pooled) < 4:
        raise CapacityError(f"task {task.name!r}: the curve needs at least four levels inside the budget")
    xs = np.array(sorted(pooled), dtype=float)
    ws = np.array([pooled[float(v)][0] for v in xs])
    ts = np.array([pooled[float(v)][1] for v in xs])
    curve, _ = dc.fit_task_curve(xs, ws, ts)
    if dc.goodness_of_fit(curve, xs, ws, ts)[2] < spec.gof_alpha:
        return dc.isotonic_task_curve(xs, ws, ts), "isotonic"
    return curve, "loglogistic"


def fit_curve(dose: DoseResponseFile) -> dc.DoseCurve:
    tasks: dict[int, dc.TaskFunction] = {}
    families: dict[int, str] = {}
    try:
        for index, task in enumerate(dose.tasks):
            tasks[index], families[index] = _fit_counts(task, dose.spec, task.successes)
    except PredictionError as exc:
        raise CapacityError(f"{dose.source}: {exc}") from exc
    return dc.DoseCurve(spec=dose.spec, tasks=tasks, families=families)


def bootstrap_curves(dose: DoseResponseFile, draws: int, seed: int) -> list[dc.DoseCurve]:
    """Parametric bootstrap: successes redrawn binomially at every level, every task refitted."""
    rng = np.random.default_rng(seed)
    curves: list[dc.DoseCurve] = []
    for _ in range(draws):
        tasks: dict[int, dc.TaskFunction] = {}
        families: dict[int, str] = {}
        for index, task in enumerate(dose.tasks):
            redrawn = [
                float(rng.binomial(trials, wins / trials))
                for wins, trials in zip(task.successes, task.episodes, strict=True)
            ]
            try:
                tasks[index], families[index] = _fit_counts(task, dose.spec, redrawn)
            except PredictionError as exc:
                raise CapacityError(f"{dose.source}: bootstrap refit failed: {exc}") from exc
        curves.append(dc.DoseCurve(spec=dose.spec, tasks=tasks, families=families))
    return curves


def bin_values(curve: dc.DoseCurve, bins: int) -> Floats:
    """The pooled curve at the centres of the 10-ms latency bins of the fleet histograms."""
    centres = (np.arange(bins, dtype=float) + 0.5) * HISTOGRAM_BIN_MS
    return np.asarray(curve.value(centres), dtype=float)


# --------------------------------------------------------------------------------------------- fleet
@dataclass(frozen=True)
class GpuVariant:
    """One service model to simulate: a measured configuration or the assumed affine profile."""

    name: str
    gpu: GpuConfig
    batching: str
    description: str


def gpu_variants(spec: GpuSpec) -> list[GpuVariant]:
    if spec.s1_ms is not None:
        text = f"assumed s(b) = {spec.s1_ms:g} ms x (1 + {spec.beta:g} (b - 1))"
        return [GpuVariant(spec.label, GpuConfig(description=text, s1_ms=spec.s1_ms), "annahme", text)]
    assert spec.measurement is not None  # checked by GpuSpec
    path = Path(spec.measurement)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapacityError(f"cannot read the GPU measurement {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != V2_FORMAT:
        if spec.configuration not in (None, ALL_VALID):
            raise CapacityError(f"{path}: only a {V2_FORMAT} file has configurations")
        return [
            GpuVariant(
                spec.label, GpuConfig(table=str(path)), MEASURED_BATCHING, f"measured table {path.name}"
            )
        ]
    try:
        benchmark = GpuBenchmark.from_dict(data, str(path))
        if spec.configuration == ALL_VALID:
            chosen = benchmark.valid_configurations()
            if not chosen:
                raise CapacityError(f"{path}: no numerically valid configuration with enough measurements")
        elif spec.configuration is not None:
            chosen = [benchmark.configuration(spec.configuration)]
        else:
            chosen = [benchmark.fastest_valid()]
    except GpuBenchmarkError as exc:
        raise CapacityError(str(exc)) from exc
    return [
        GpuVariant(
            f"{spec.label}/{c.name}",
            GpuConfig(description=f"{benchmark.gpu}, {c.name}", table=str(path), configuration=c.name),
            MEASURED_BATCHING,
            f"{benchmark.gpu} ({benchmark.platform}), configuration {c.name}, median at batch 1 "
            f"{c.median_ms(1):.1f} ms"
            if c.median_ms(1) is not None
            else f"{benchmark.gpu}, {c.name}",
        )
        for c in chosen
    ]


def fleet_config(config: CapacityConfig, variants: Sequence[GpuVariant]) -> FleetSimConfig:
    network_name = config.network.scenario
    return FleetSimConfig(
        description="kortyn capacity tool",
        agent=config.agent,
        episode_lengths=config.episode_lengths,
        gpus={variant.name: variant.gpu for variant in variants},
        batching={
            "annahme": BatchingConfig(description="assumption", beta=config.gpu.beta),
            MEASURED_BATCHING: BatchingConfig(description="measured", beta=1.0),
        },
        strategies=dict(config.strategies),
        networks={
            network_name: NetworkChoice(
                scenario=network_name,
                margin_ms=config.network.margin_ms,
                wlan_mbit_s=config.network.wlan_mbit_s,
                request_bytes=config.network.request_bytes,
                response_bytes=config.network.response_bytes,
            )
        },
        fleet_sizes=[1],
        replications=config.fleet.replications,
        start_spread_s=config.fleet.start_spread_s,
        window_start_s=config.fleet.window_start_s,
        horizon_s=config.fleet.horizon_s,
        service_cv=config.gpu.service_cv,
        batch_limit=config.fleet.batch_limit,
        seed=config.fleet.seed,
        results_path="",
    )


def histograms(result: RunResult, bins: int) -> list[tuple[bool, Floats]]:
    """Per robot: admitted, and the dense 10-ms latency histogram of its requests in the window."""
    rows: list[tuple[bool, Floats]] = []
    for robot in result.robots:
        dense = np.zeros(bins, dtype=float)
        for index, count in robot.histogram.items():
            if int(index) >= bins:
                raise CapacityError("a latency beyond the evaluated range (raise the bin count)")
            dense[int(index)] += float(count)
        rows.append((robot.admitted, dense))
    return rows


# --------------------------------------------------------------------------------------------- evaluation
@dataclass
class SizeRow:
    robots: int
    replications: int
    requests: int
    misses: int
    miss_share: float
    admitted_mean: float
    mean_batch: float
    busy_share: float
    latency_p50_ms: float | None  # upper edge of the 10-ms bin holding the pooled quantile
    latency_p99_ms: float | None
    slo_share: float | None  # share of requests below latency_slo.max_ms
    fleet_mean: float
    worst: float
    single: float
    loss: float
    loss_upper: float
    passes: bool
    passes_conservative: bool
    passes_slo: bool | None


def pooled_quantile_ms(histogram: Floats, q: float) -> float | None:
    """Upper edge of the 10-ms bin in which the pooled ``q`` quantile lies (conservative)."""
    total = histogram.sum()
    if total <= 0:
        return None
    cumulative = np.cumsum(histogram) / total
    index = int(np.searchsorted(cumulative, q - 1e-12))
    return float((index + 1) * HISTOGRAM_BIN_MS)


@dataclass
class _SizeRuns:
    robots: int
    results: list[RunResult]
    predictions: list[Floats]  # per replication: point prediction per robot
    draws: list[Floats]  # per replication: (draws, robots)


def predict(result: RunResult, bins: int, point: Floats, draws: Floats) -> tuple[Floats, Floats, Floats]:
    """Per robot point prediction and bootstrap draws (rejected or silent robots: 0); pooled histogram."""
    rows = histograms(result, bins)
    points = np.zeros(len(rows))
    sampled = np.zeros((draws.shape[0], len(rows)))
    pooled = np.zeros(bins)
    for index, (admitted, histogram) in enumerate(rows):
        pooled += histogram
        total = histogram.sum()
        if not admitted or total <= 0:
            continue
        weights = histogram / total
        points[index] = float(point @ weights)
        sampled[:, index] = draws @ weights
    return points, sampled, pooled


def evaluate_sizes(
    runs: Mapping[int, list[RunResult]],
    point: Floats,
    draws: Floats,
    tolerance: float,
    slo: LatencySlo | None,
) -> list[SizeRow]:
    """Criteria per fleet size; ``runs`` must hold the single robot (size 1)."""
    if 1 not in runs:
        raise CapacityError("the evaluation needs the single-robot runs")
    bins = point.size
    single_points: list[float] = []
    single_draws: list[Floats] = []
    for result in runs[1]:
        points, sampled, _ = predict(result, bins, point, draws)
        single_points.append(float(points[0]))
        single_draws.append(sampled[:, 0])
    single = float(np.mean(single_points))
    single_draw = np.mean(single_draws, axis=0)
    rows: list[SizeRow] = []
    for robots in sorted(runs):
        worst_points, worst_draws, all_points = [], [], []
        pooled = np.zeros(bins)
        for result in runs[robots]:
            points, sampled, histogram = predict(result, bins, point, draws)
            worst_points.append(float(points.min()))
            worst_draws.append(sampled.min(axis=1))
            all_points.append(points)
            pooled += histogram
        results = runs[robots]
        requests = sum(r.requests for r in results)
        misses = sum(r.misses for r in results)
        share = misses / requests if requests else 1.0
        worst = float(np.mean(worst_points))
        loss = single - worst
        loss_upper = float(np.percentile(single_draw - np.mean(worst_draws, axis=0), 97.5))
        admitted_all = all(r.admitted == robots for r in results)
        slo_share: float | None = None
        passes_slo: bool | None = None
        if slo is not None:
            below = round(slo.max_ms / HISTOGRAM_BIN_MS)
            total = pooled.sum()
            slo_share = float(pooled[:below].sum() / total) if total > 0 else 0.0
            passes_slo = admitted_all and total > 0 and slo_share >= slo.quantile - 1e-12
        criterion_1 = requests > 0 and share <= MISS_SHARE_LIMIT
        rows.append(
            SizeRow(
                robots=robots,
                replications=len(results),
                requests=requests,
                misses=misses,
                miss_share=share,
                admitted_mean=float(np.mean([r.admitted for r in results])),
                mean_batch=float(np.mean([r.mean_batch for r in results])),
                busy_share=float(np.mean([r.busy_share for r in results])),
                latency_p50_ms=pooled_quantile_ms(pooled, 0.5),
                latency_p99_ms=pooled_quantile_ms(pooled, 0.99),
                slo_share=slo_share,
                fleet_mean=float(np.mean(np.concatenate(all_points))),
                worst=worst,
                single=single,
                loss=loss,
                loss_upper=loss_upper,
                passes=criterion_1 and loss <= tolerance + 1e-12,
                passes_conservative=criterion_1 and loss_upper <= tolerance + 1e-12,
                passes_slo=passes_slo,
            )
        )
    return rows


def capacity_of(rows: Sequence[SizeRow], attribute: str) -> int:
    """Largest size for which this size and every smaller one pass (sizes 1, 2, 3, ... without gaps)."""
    capacity = 0
    for expected, row in enumerate(sorted(rows, key=lambda r: r.robots), start=1):
        if row.robots != expected or not bool(getattr(row, attribute)):
            break
        capacity = row.robots
    return capacity


@dataclass
class StrategyResult:
    gpu: str
    gpu_description: str
    strategy: str
    capacity: int
    capacity_conservative: int
    latency_capacity: int | None
    rows: list[SizeRow]


def scan_strategy(
    config: CapacityConfig,
    fleet: FleetSimConfig,
    variant: GpuVariant,
    strategy: str,
    lengths: tuple[int, ...],
    point: Floats,
    draws: Floats,
) -> StrategyResult:
    """Fleet sizes 1, 2, 3, ... until every criterion has failed once (or ``max_robots``)."""
    tolerance = config.criteria.tolerance_pp / 100.0
    slo = config.criteria.latency_slo
    runs: dict[int, list[RunResult]] = {}
    rows: list[SizeRow] = []
    for robots in range(1, config.fleet.max_robots + 1):
        runs[robots] = [
            run_fleet(
                fleet,
                RunKey(
                    config.network.scenario, variant.name, variant.batching, strategy, robots, replication
                ),
                lengths,
            )
            for replication in range(config.fleet.replications)
        ]
        rows = evaluate_sizes(runs, point, draws, tolerance, slo)
        done = [
            capacity_of(rows, "passes") < robots,
            capacity_of(rows, "passes_conservative") < robots,
            slo is None or capacity_of(rows, "passes_slo") < robots,
        ]
        if all(done):
            break
    return StrategyResult(
        gpu=variant.name,
        gpu_description=variant.description,
        strategy=strategy,
        capacity=capacity_of(rows, "passes"),
        capacity_conservative=capacity_of(rows, "passes_conservative"),
        latency_capacity=None if slo is None else capacity_of(rows, "passes_slo"),
        rows=rows,
    )


def latency_bins(config: CapacityConfig) -> int:
    """Histogram bins to evaluate: every latency up to the robots' request timeout."""
    return math.ceil(config.agent.request_timeout_s * 1000.0 / HISTOGRAM_BIN_MS) + 2


def run_tool(config: CapacityConfig) -> dict[str, Any]:
    dose = read_dose_response(config.dose_response)
    curve = fit_curve(dose)
    bins = latency_bins(config)
    point = bin_values(curve, bins)
    draws = (
        np.array(
            [bin_values(c, bins) for c in bootstrap_curves(dose, config.bootstrap, config.bootstrap_seed)]
        )
        if config.bootstrap > 0
        else point[None, :]
    )
    try:
        load_scenario(config.network.scenario)
        variants = gpu_variants(config.gpu)
        fleet = fleet_config(config, variants)
        lengths = load_episode_lengths(config.episode_lengths)
        results = [
            scan_strategy(config, fleet, variant, strategy, lengths, point, draws)
            for variant in variants
            for strategy in config.strategies
        ]
    except (ConfigError, BatchModelError) as exc:
        raise CapacityError(str(exc)) from exc
    slo = config.criteria.latency_slo
    best = max(results, key=lambda r: (r.capacity, r.capacity_conservative, -len(r.strategy)))
    document: dict[str, Any] = {
        "inputs": {
            "dose_response": dose.source,
            "dose_response_description": dose.description,
            "gpu": [
                asdict(variant.gpu) | {"name": variant.name, "description": variant.description}
                for variant in variants
            ],
            "network": asdict(config.network),
            "agent": asdict(config.agent),
            "fleet": asdict(config.fleet),
            "tolerance_pp": config.criteria.tolerance_pp,
            "miss_share_limit": MISS_SHARE_LIMIT,
            "latency_slo": None if slo is None else asdict(slo),
            "bootstrap": config.bootstrap,
        },
        "curve": {
            "families": {dose.tasks[i].name: family for i, family in curve.families.items()},
            "success_at_0ms": float(curve.value(0.0)),
            "tolerable_latency_ms": curve.tolerable_latency(config.criteria.tolerance_pp / 100.0)[0],
        },
        "results": [asdict(r) for r in results],
        "best": {"gpu": best.gpu, "strategy": best.strategy, "capacity": best.capacity},
    }
    if slo is not None:
        best_latency = max(results, key=lambda r: (r.latency_capacity or 0, -len(r.strategy)))
        document["latency"] = {
            "gpu": best_latency.gpu,
            "strategy": best_latency.strategy,
            "latency_capacity": best_latency.latency_capacity,
            "target_robots": slo.robots,
            "meets_target": None
            if slo.robots is None
            else (best_latency.latency_capacity or 0) >= slo.robots,
        }
    return document


def report_lines(document: Mapping[str, Any]) -> list[str]:
    """A readable summary: one block per GPU configuration and strategy."""
    inputs = document["inputs"]
    slo = inputs["latency_slo"]
    lines = [
        f"Dose-response curve: {inputs['dose_response']} (success at 0 ms "
        f"{100 * document['curve']['success_at_0ms']:.1f} %)",
        f"Network: {inputs['network']['scenario']}, tolerance {inputs['tolerance_pp']:g} pp, "
        f"at most {100 * inputs['miss_share_limit']:g} % late plans"
        + ("" if slo is None else f", latency target p{100 * slo['quantile']:g} <= {slo['max_ms']:g} ms"),
    ]
    for result in document["results"]:
        lines.append("")
        lines.append(f"{result['gpu']} - {result['gpu_description']}")
        latency = (
            "" if result["latency_capacity"] is None else f", latency capacity {result['latency_capacity']}"
        )
        lines.append(
            f"  strategy {result['strategy']}: capacity {result['capacity']} robots "
            f"(conservative {result['capacity_conservative']}){latency}"
        )
        lines.append("    robots  p50 ms  p99 ms  late %  mean robot  worst robot  passes")
        for row in result["rows"]:
            flags = ("C" if row["passes"] else "-") + (
                "" if row["passes_slo"] is None else ("L" if row["passes_slo"] else "-")
            )
            p50 = "-" if row["latency_p50_ms"] is None else f"{row['latency_p50_ms']:.0f}"
            p99 = "-" if row["latency_p99_ms"] is None else f"{row['latency_p99_ms']:.0f}"
            lines.append(
                f"    {row['robots']:>6}  {p50:>6}  {p99:>6}  {100 * row['miss_share']:>6.2f}  "
                f"{100 * row['fleet_mean']:>9.1f} %  {100 * row['worst']:>10.1f} %  {flags:>6}"
            )
    best = document["best"]
    lines.append("")
    lines.append(f"Best: {best['gpu']}, strategy {best['strategy']}: {_robots(best['capacity'])} per GPU")
    if "latency" in document:
        latency = document["latency"]
        verdict = (
            ""
            if latency["meets_target"] is None
            else (
                f" -> target of {_robots(latency['target_robots'])} "
                + ("met" if latency["meets_target"] else "NOT met")
            )
        )
        lines.append(
            f"Latency: {_robots(latency['latency_capacity'])} within the target "
            f"({latency['gpu']}, strategy {latency['strategy']}){verdict}"
        )
    return lines


def _robots(count: int) -> str:
    return f"{count} robot" if count == 1 else f"{count} robots"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Robots per GPU and expected success from a dose-response curve, a network profile and a "
        "GPU measurement (Kortyn)."
    )
    parser.add_argument(
        "--config", required=True, help="capacity config (YAML), e.g. examples/capacity/example.yaml"
    )
    parser.add_argument("--gpu", default=None, help="GPU measurement file (overrides gpu.measurement)")
    parser.add_argument("--configuration", default=None, help="configuration of a v2 file, or all-valid")
    parser.add_argument("--out", default=None, help="result JSON (overrides output)")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_capacity_config(args.config)
        if args.gpu is not None:
            config.gpu = GpuSpec(
                measurement=args.gpu,
                configuration=args.configuration or config.gpu.configuration,
                label=Path(args.gpu).stem,  # not the config's label: that names the GPU it replaces
            )
        elif args.configuration is not None:
            config.gpu.configuration = args.configuration
        document = run_tool(config)
    except (ConfigError, CapacityError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("\n".join(report_lines(document)))
    out = args.out or config.output
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=1), encoding="utf-8")
        print(f"\nwritten {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
