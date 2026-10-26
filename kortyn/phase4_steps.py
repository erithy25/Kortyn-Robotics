"""Phase-4 evaluation steps (pre-registered in reports/phase4_preregistration.md).

* ``curves``: the dose-response conditions (without and with RTC) on the same seeds, fitted with the smooth
  monotone curve of :mod:`kortyn.dose_curve` and one shared set of bootstrap multiplicities, so every
  difference between the conditions is paired. Reports per condition the fit, the band, the tolerable latency
  ``L*`` (point and conservative), and between the conditions H1 (mean gain within the budget) and H2
  (``L*`` difference); further the check R-RTC (RTC at 0 ms against the synchronous reference ``sync25``).
* ``retest``: the phase-3 validation episodes predicted again with the new curve without RTC (each episode:
  the curve of its task over its own measured latencies, and over the stage-1 latencies of its robot), the
  pooled bias against the phase-3 rule on the same episodes (cluster bootstrap over seeds).

    python -m kortyn.phase4_steps curves --config kortyn/experiments/phase4/analysis.yaml
    python -m kortyn.phase4_steps retest --config kortyn/experiments/phase4/analysis.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from kortyn import dose_curve as dc
from kortyn.config import ConfigError, LiberoEnvConfig, decode, load_config
from kortyn.fleet_predict import DoseResponse, PredictionError
from kortyn.libero_env import episode_slot
from kortyn.phase3_steps import StepError, load_episodes, paired_comparison, successes

LOG = logging.getLogger("kortyn.phase4_steps")

Floats = npt.NDArray[np.float64]

R_RTC_MARGIN = 0.15  # lower bound of the 95 % interval of (RTC@0 - sync25) must be >= -15 pp
BIAS_GONE = 0.025  # |bias| <= 2.5 pp: "verschwunden"
PHASE3_BIAS = -0.074  # phase-3 rule on the informative validation episodes (own latencies)


# --------------------------------------------------------------------------------------------- config
@dataclass
class LevelSpec:
    """One fixed-latency run of a dose-response condition."""

    latency_ms: float
    run: str  # directory with episodes.jsonl

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ConfigError(f"level latency must be >= 0, got {self.latency_ms}")


@dataclass
class RetestConfig:
    """The phase-3 validation (stage-1 results, rollouts) re-predicted with the phase-4 curve."""

    fleet: str = "kortyn/experiments/phase3/fleet.yaml"
    results_dir: str = "results/phase3/validation"
    informative: list[str] = field(default_factory=lambda: ["a_n2", "e_n2", "e_n4"])
    draws: int = 10_000


@dataclass
class AnalysisConfig:
    experiment: str = "kortyn/experiments/phase3/libero_main.yaml"  # episode layout: task of every seed
    bootstrap: int = 2000
    seed: int = 4242
    tolerance: float = 0.03
    tick_ms: float = 50.0
    fit_max_ms: float = 1250.0
    zero_from_ms: float = 1500.0
    gof_alpha: float = 0.01
    without: str = "ohne_rtc"  # condition names
    with_rtc: str = "mit_rtc"
    conditions: dict[str, list[LevelSpec]] = field(default_factory=dict)
    reference_sync: str = "results/phase3/libero/smolvla_libero/sync25"
    retest: RetestConfig = field(default_factory=RetestConfig)
    output: str = "results/phase4/curves.json"
    retest_output: str = "results/phase4/retest.json"

    def __post_init__(self) -> None:
        if self.bootstrap < 100 or not 0 < self.tolerance < 1:
            raise ConfigError("need bootstrap >= 100 and a tolerance in (0, 1)")
        if not self.conditions:
            raise ConfigError("the analysis needs at least one condition")

    @property
    def spec(self) -> dc.CurveSpec:
        return dc.CurveSpec(
            tick_ms=self.tick_ms,
            fit_max_ms=self.fit_max_ms,
            zero_from_ms=self.zero_from_ms,
            gof_alpha=self.gof_alpha,
        )


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    import yaml

    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read analysis config {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    return decode(AnalysisConfig, data, str(source))


def libero_layout(experiment: str) -> LiberoEnvConfig:
    settings = load_config(experiment).env.libero
    if settings is None:
        raise ConfigError(f"{experiment} is not a LIBERO experiment")
    return settings


# --------------------------------------------------------------------------------------------- data
def load_condition(levels: Sequence[LevelSpec], libero: LiberoEnvConfig) -> DoseResponse:
    """Final attempt per seed at every level; all levels must hold the same seeds."""
    if len(levels) < 2:
        raise StepError("a condition needs at least two levels")
    ordered = sorted(levels, key=lambda level: level.latency_ms)
    per_level = [successes(load_episodes(Path(level.run))) for level in ordered]
    seeds = sorted(set.intersection(*(set(level) for level in per_level)))
    missing = sorted(set.union(*(set(level) for level in per_level)) - set(seeds))
    if missing:
        raise StepError(f"seeds {missing[:5]} are missing in some levels")
    outcomes = np.array([[float(level[seed]) for seed in seeds] for level in per_level])
    return DoseResponse(
        levels_ms=np.array([level.latency_ms for level in ordered], dtype=float),
        seeds=tuple(seeds),
        groups=tuple(episode_slot(libero, seed)[0] for seed in seeds),
        outcomes=outcomes,
    )


def fit_conditions(
    doses: Mapping[str, DoseResponse], spec: dc.CurveSpec, draws: int, seed: int
) -> dict[str, dc.BootstrapFit]:
    """Bootstrap fits of all conditions with one shared set of seed multiplicities (paired)."""
    if not doses:
        raise StepError("no condition to fit")
    first = next(iter(doses.values()))
    for name, dose in doses.items():
        if dose.seeds != first.seeds or dose.groups != first.groups:
            raise StepError(f"condition {name!r} does not hold the same seeds as the others")
    multiplicities = first.bootstrap_weights(draws, seed)
    return {name: dc.bootstrap_fit(dose, spec, multiplicities) for name, dose in doses.items()}


# --------------------------------------------------------------------------------------------- summaries
def level_rates(dose: DoseResponse) -> list[dict[str, Any]]:
    """Measured success per level: pooled (tasks weighted equally) and per task."""
    rows = []
    groups = np.asarray(dose.groups)
    for index, latency in enumerate(dose.levels_ms):
        per_task = {str(t): float(dose.outcomes[index, groups == t].mean()) for t in dose.group_ids}
        rows.append(
            {
                "latency_ms": float(latency),
                "effective_ms": float(dc.effective_latency_ms(latency)),
                "pooled": float(np.mean(list(per_task.values()))),
                "tasks": per_task,
            }
        )
    return rows


def condition_summary(dose: DoseResponse, fit: dc.BootstrapFit, tolerance: float) -> dict[str, Any]:
    spec = fit.point.spec
    grid = np.arange(0.0, spec.zero_from_ms + spec.tick_ms, spec.tick_ms)
    low, high = fit.band(grid)
    x_grid = np.linspace(spec.tick_ms, spec.fit_max_ms, 120)
    x_low, x_high = fit.band_x(x_grid)
    tolerable = fit.tolerable(tolerance)
    return {
        "levels": level_rates(dose),
        "fit": {str(task): entry for task, entry in dc.fit_report(dose, fit.point).items()},
        "lstar": asdict(tolerable),
        "success_at_lstar": float(fit.point.value(tolerable.point_ms)),
        "curve": {
            "latency_ms": grid.tolist(),
            "point": fit.point.value(grid).tolist(),
            "low": low.tolist(),
            "high": high.tolist(),
        },
        "smooth": {
            "x_ms": x_grid.tolist(),
            "point": fit.point.value_x(x_grid).tolist(),
            "low": x_low.tolist(),
            "high": x_high.tolist(),
        },
    }


def rtc_effect(without: dc.BootstrapFit, with_rtc: dc.BootstrapFit, tolerance: float) -> dict[str, Any]:
    """H1 (mean gain over the budget ticks) and H2 (``L*`` difference), paired over the bootstrap draws."""
    spec = without.point.spec
    grid = spec.tick_ms * np.arange(1, spec.budget_ticks + 1, dtype=float)
    gain, (low, high) = dc.mean_success_difference(without, with_rtc, grid)
    h1 = "bestätigt" if low > 0 else ("RTC schadet" if high < 0 else "nicht nachweisbar")
    difference, (d_low, d_high) = dc.paired_tolerable_difference(without, with_rtc, tolerance)
    h2 = "nachgewiesen" if d_low > 0 else "nicht nachgewiesen"
    return {
        "h1": {"gain": gain, "interval": [low, high], "verdict": h1},
        "h2": {"difference_ms": difference, "interval_ms": [d_low, d_high], "verdict": h2},
    }


def r_rtc_check(rtc_zero: Path, sync25: Path, seed: int = 0) -> dict[str, Any]:
    """R-RTC: McNemar p >= 0.05 and the lower bound of (RTC@0 - sync25) >= -15 pp."""
    comparison = paired_comparison(
        successes(load_episodes(sync25)), successes(load_episodes(rtc_zero)), seed=seed
    )
    passes = comparison.p_value >= 0.05 and comparison.interval[0] >= -R_RTC_MARGIN - 1e-12
    if passes:
        verdict = "erfüllt"
    elif comparison.p_value < 0.05 and comparison.difference > 0:
        verdict = "nicht entsprechend (besser)"
    else:
        verdict = "nicht erfüllt"
    return {**asdict(comparison), "passes": passes, "verdict": verdict}


def curves_document(config: AnalysisConfig, names: Sequence[str]) -> dict[str, Any]:
    libero = libero_layout(config.experiment)
    doses = {name: load_condition(config.conditions[name], libero) for name in names}
    fits = fit_conditions(doses, config.spec, config.bootstrap, config.seed)
    document: dict[str, Any] = {
        "spec": asdict(config.spec),
        "bootstrap": config.bootstrap,
        "seed": config.seed,
        "tolerance": config.tolerance,
        "seeds": len(next(iter(doses.values())).seeds),
        "tasks": next(iter(doses.values())).group_ids,
        "conditions": {name: condition_summary(doses[name], fits[name], config.tolerance) for name in names},
    }
    if config.without in fits and config.with_rtc in fits:
        document["rtc_effect"] = rtc_effect(fits[config.without], fits[config.with_rtc], config.tolerance)
        zero = min(config.conditions[config.with_rtc], key=lambda level: level.latency_ms)
        if zero.latency_ms == 0.0:
            document["r_rtc"] = r_rtc_check(Path(zero.run), Path(config.reference_sync))
    return document


# --------------------------------------------------------------------------------------------- re-test
@dataclass(frozen=True)
class EpisodePrediction:
    case: str
    seed: int
    task: int
    measured: float
    own: float  # task curve over the episode's own measured latencies
    stage1: float  # task curve over the stage-1 latencies of its robot (replication 0)
    phase3_own: float  # phase-3 rule (per-seed outcomes, linear in L) over the own latencies
    phase3_stage1: float


def cluster_mean_interval(
    values: Sequence[float], clusters: Sequence[int], draws: int, seed: int
) -> tuple[float, tuple[float, float]]:
    """Mean and 95-% interval, resampling whole clusters (seeds shared by several cases) with replacement."""
    data = np.asarray(values, dtype=float)
    keys = np.asarray(clusters)
    if data.size == 0 or data.shape != keys.shape:
        raise StepError("need matching, non-empty values and clusters")
    unique = np.unique(keys)
    sums = np.array([data[keys == key].sum() for key in unique])
    counts = np.array([float(np.sum(keys == key)) for key in unique])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, unique.size, size=(draws, unique.size))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(data.mean()), (float(low), float(high))


def bias_verdict(bias: float, phase3_bias: float = PHASE3_BIAS) -> str:
    if abs(bias) <= BIAS_GONE + 1e-12:
        return "verschwunden"
    if abs(bias) <= 0.5 * abs(phase3_bias) + 1e-12:
        return "verkleinert"
    return "nicht erklärt"


def retest_predictions(config: AnalysisConfig, curve: dc.DoseCurve) -> list[EpisodePrediction]:
    """Every counted phase-3 validation episode, predicted with the new curve and with the phase-3 rule."""
    from kortyn.fleet import load_fleet_config
    from kortyn.fleet_predict import (
        episode_prediction,
        histogram_latencies,
        level_weights,
        load_dose_response,
        prediction_config,
    )
    from kortyn.fleet_rollout import _batching_key, _validation, episode_latencies
    from kortyn.metrics import read_jsonl

    fleet = load_fleet_config(config.retest.fleet)
    validation = _validation(fleet)
    phase3_dose = load_dose_response(prediction_config(fleet))
    libero = libero_layout(config.experiment)
    stage1: dict[tuple[str, int], list[tuple[list[float], list[float]]]] = {}
    for record in read_jsonl(Path(fleet.results_path)):
        key = record["key"]
        if (
            key["network"] != validation.network
            or key["gpu"] != validation.gpu
            or int(key["replication"]) != 0
        ):
            continue
        robots = []
        for robot in record["robots"]:
            centres, counts = histogram_latencies(robot["histogram"]) if robot["histogram"] else ([], [])
            robots.append((centres, counts))
        stage1[(f"{key['batching']}/{key['strategy']}", int(key["robots"]))] = robots
    predictions: list[EpisodePrediction] = []
    root = Path(config.retest.results_dir)
    for case_dir in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
        strategy, robots_n = str(meta["strategy"]), int(meta["robots"])
        batching = _batching_key(fleet, validation, strategy)
        case_robots = stage1.get((f"{batching}/{strategy}", robots_n))
        if case_robots is None:
            raise StepError(f"stage 1 has no run for {batching}/{strategy} with {robots_n} robots")
        latencies = episode_latencies(case_dir / "requests.jsonl")
        for episode in read_jsonl(case_dir / "episodes.jsonl"):
            if not episode["counted"]:
                continue
            seed, robot = int(episode["seed"]), int(episode["robot"])
            task = episode_slot(libero, seed)[0]
            own = latencies[(robot, int(episode["robot_episode"]))]
            centres, counts = case_robots[robot]
            if not own or not centres:
                raise StepError(f"{case_dir.name}: episode of robot {robot} has no latencies")
            ones = [1.0] * len(own)
            predictions.append(
                EpisodePrediction(
                    case=case_dir.name,
                    seed=seed,
                    task=task,
                    measured=float(bool(episode["success"])),
                    own=curve.mean_over(own, ones, task=task),
                    stage1=curve.mean_over(centres, counts, task=task),
                    phase3_own=episode_prediction(
                        phase3_dose, seed, level_weights(phase3_dose.levels_ms, own, ones)
                    ),
                    phase3_stage1=episode_prediction(
                        phase3_dose, seed, level_weights(phase3_dose.levels_ms, centres, counts)
                    ),
                )
            )
    return predictions


def retest_document(config: AnalysisConfig, predictions: Sequence[EpisodePrediction]) -> dict[str, Any]:
    retest = config.retest
    cases: dict[str, Any] = {}
    for case in sorted({p.case for p in predictions}):
        rows = [p for p in predictions if p.case == case]
        measured = float(np.mean([p.measured for p in rows]))
        cases[case] = {
            "episodes": len(rows),
            "measured": measured,
            **{
                f"predicted_{name}": float(np.mean([getattr(p, name) for p in rows]))
                for name in ("own", "stage1", "phase3_own", "phase3_stage1")
            },
        }
    informative = [p for p in predictions if p.case in retest.informative]
    pooled: dict[str, Any] = {}
    for name in ("own", "stage1", "phase3_own", "phase3_stage1"):
        bias, interval = cluster_mean_interval(
            [p.measured - getattr(p, name) for p in informative],
            [p.seed for p in informative],
            retest.draws,
            config.seed,
        )
        pooled[name] = {"bias": bias, "interval": list(interval)}
    phase3 = pooled["phase3_own"]["bias"]
    return {
        "informative_cases": retest.informative,
        "episodes": len(informative),
        "cases": cases,
        "pooled": pooled,
        "verdict": bias_verdict(pooled["own"]["bias"], phase3),
        "verdict_stage1": bias_verdict(pooled["stage1"]["bias"], pooled["phase3_stage1"]["bias"]),
    }


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-4 evaluation steps.")
    parser.add_argument("step", choices=["curves", "retest"])
    parser.add_argument("--config", default="kortyn/experiments/phase4/analysis.yaml")
    parser.add_argument(
        "--conditions", default=None, help="comma-separated condition names (default: all configured)"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        config = load_analysis_config(args.config)
        if args.step == "curves":
            names = list(config.conditions) if args.conditions is None else args.conditions.split(",")
            unknown = [name for name in names if name not in config.conditions]
            if unknown:
                raise ConfigError(f"unknown conditions {unknown}; configured: {list(config.conditions)}")
            document = curves_document(config, names)
            out = Path(config.output)
            for name, summary in document["conditions"].items():
                lstar = summary["lstar"]
                LOG.info(
                    "%s: L* %.0f ms (conservative %.0f, upper %.0f%s)",
                    name,
                    lstar["point_ms"],
                    lstar["conservative_ms"],
                    lstar["upper_ms"],
                    ", censored" if lstar["censored"] else "",
                )
            if "rtc_effect" in document:
                effect = document["rtc_effect"]
                LOG.info("H1: %s, H2: %s", effect["h1"]["verdict"], effect["h2"]["verdict"])
        else:
            libero = libero_layout(config.experiment)
            dose = load_condition(config.conditions[config.without], libero)
            curve = dc.fit_dose_curve(dose, config.spec)
            document = retest_document(config, retest_predictions(config, curve))
            out = Path(config.retest_output)
            LOG.info(
                "re-test: pooled bias %+.1f pp (phase-3 rule %+.1f pp): %s",
                100 * document["pooled"]["own"]["bias"],
                100 * document["pooled"]["phase3_own"]["bias"],
                document["verdict"],
            )
    except (ConfigError, StepError, PredictionError, KeyError) as exc:
        LOG.error("%s", exc)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    LOG.info("written %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
