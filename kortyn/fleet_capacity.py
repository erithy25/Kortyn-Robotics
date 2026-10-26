"""Phase 4, stage 2: capacity of one Brain with the smooth dose-response curves (3-pp and 1-% criteria).

Reads the stage-1 queue simulation (latency histograms per robot, :mod:`kortyn.fleet`) and, per GPU profile,
the dose-response condition that applies (without RTC: the curve without RTC; a profile with the RTC
overhead: the RTC curve). Pre-registered in reports/phase4_preregistration.md, section 7.3:

* **robot:** prediction = mean of the pooled curve over the latencies of its requests in the window (tick and
  tail rule of :mod:`kortyn.dose_curve`); a robot that admission control rejected predicts 0;
* **worst robot:** per replication the robot with the lowest prediction, averaged over the replications;
* **criterion 1:** at most 1 % of all requests of all replications miss their budget;
* **criterion 2:** the worst robot is at most ``tolerance`` (3 pp) below the single robot (``n = 1`` of the
  same configuration) - point prediction; the conservative variant takes the 97.5-% bootstrap quantile of
  (single - worst);
* **capacity:** the largest scanned size for which this size and every smaller one pass.

The curves are evaluated once on the 10-ms histogram bins (point fit and every bootstrap draw), so a robot's
prediction is a weighted sum of its histogram.

    python -m kortyn.fleet_capacity --analysis kortyn/experiments/phase4/analysis.yaml \\
        --stage1 results/phase4/fleet/scan.jsonl --curve rtx3060ti_schaetzung=ohne_rtc \\
        --curve rtx3060ti_schaetzung_rtc=mit_rtc --out results/phase4/fleet/capacity.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from kortyn import dose_curve as dc
from kortyn.config import ConfigError
from kortyn.fleet import HISTOGRAM_BIN_MS
from kortyn.fleet_predict import MISS_SHARE_LIMIT, PredictionError
from kortyn.metrics import read_jsonl

LOG = logging.getLogger("kortyn.fleet_capacity")

Floats = npt.NDArray[np.float64]
ConfigKey = tuple[str, str, str, str]  # network, gpu, batching, strategy


# --------------------------------------------------------------------------------------------- curves on bins
@dataclass(frozen=True)
class BinCurves:
    """Pooled curve at the centres of the latency histogram bins: point fit and every bootstrap draw."""

    point: Floats  # (bins,)
    draws: Floats  # (draws, bins)

    @property
    def bins(self) -> int:
        return int(self.point.size)


def bin_curves(fit: dc.BootstrapFit, bins: int) -> BinCurves:
    if bins < 1:
        raise PredictionError("need at least one histogram bin")
    centres = (np.arange(bins, dtype=float) + 0.5) * HISTOGRAM_BIN_MS
    return BinCurves(
        point=np.asarray(fit.point.value(centres), dtype=float),
        draws=np.array([draw.value(centres) for draw in fit.draws], dtype=float),
    )


# --------------------------------------------------------------------------------------------- stage 1
@dataclass(frozen=True)
class Robot:
    admitted: bool
    histogram: Floats  # request counts per 10-ms bin (empty: no request in the window)


@dataclass(frozen=True)
class Run:
    key: ConfigKey
    robots: int
    replication: int
    requests: int
    misses: int
    mean_batch: float
    busy_share: float
    fleet: tuple[Robot, ...]
    medium_busy_share: float | None = None  # shared WLAN medium (phase 4, section 7.4)


def read_runs(path: Path) -> list[Run]:
    """Stage-1 records with dense histograms (bins up to the largest latency in the file)."""
    if not path.exists():
        raise PredictionError(f"stage-1 results {path} do not exist")
    records = list(read_jsonl(path))
    largest = max(
        (int(b) for record in records for robot in record["robots"] for b in robot["histogram"]), default=0
    )
    runs: list[Run] = []
    for record in records:
        key = record["key"]
        fleet = []
        for robot in record["robots"]:
            histogram = np.zeros(largest + 1, dtype=float)
            for index, count in robot["histogram"].items():
                histogram[int(index)] = float(count)
            fleet.append(Robot(admitted=bool(robot["admitted"]), histogram=histogram))
        runs.append(
            Run(
                key=(key["network"], key["gpu"], key["batching"], key["strategy"]),
                robots=int(key["robots"]),
                replication=int(key["replication"]),
                requests=int(record["requests"]),
                misses=int(record["misses"]),
                mean_batch=float(record["mean_batch"]),
                busy_share=float(record["busy_share"]),
                fleet=tuple(fleet),
                medium_busy_share=_optional_float(record.get("medium_busy_share")),
            )
        )
    return runs


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def predictions(run: Run, curves: BinCurves) -> tuple[Floats, Floats]:
    """Per robot: point prediction (robots,) and draws (draws, robots); rejected or silent robots get 0."""
    points = np.zeros(len(run.fleet))
    draws = np.zeros((curves.draws.shape[0], len(run.fleet)))
    for index, robot in enumerate(run.fleet):
        total = robot.histogram.sum()
        if not robot.admitted or total <= 0:
            continue
        if robot.histogram.size > curves.bins:
            raise PredictionError("histogram longer than the evaluated curve")
        weights = robot.histogram / total
        points[index] = float(curves.point[: weights.size] @ weights)
        draws[:, index] = curves.draws[:, : weights.size] @ weights
    return points, draws


# --------------------------------------------------------------------------------------------- criteria
@dataclass
class SizeResult:
    robots: int
    replications: int
    requests: int
    misses: int
    miss_share: float
    criterion_1: bool
    admitted_mean: float
    mean_batch: float
    busy_share: float
    fleet_mean: float  # mean prediction over all robots and replications
    worst: float
    single: float
    loss: float  # single - worst (point)
    loss_upper: float  # 97.5-% bootstrap quantile of single - worst
    criterion_2: bool
    criterion_2_conservative: bool
    medium_busy_share: float | None = None  # mean over replications; None without a shared WLAN medium

    @property
    def passes(self) -> bool:
        return self.criterion_1 and self.criterion_2

    @property
    def passes_conservative(self) -> bool:
        return self.criterion_1 and self.criterion_2_conservative


def evaluate_size(
    runs: Sequence[Run], single_runs: Sequence[Run], curves: BinCurves, tolerance: float
) -> SizeResult:
    if not runs or not single_runs:
        raise PredictionError("need runs of the size and of the single robot")
    requests = sum(run.requests for run in runs)
    misses = sum(run.misses for run in runs)
    share = misses / requests if requests else 1.0
    worst_points, worst_draws, all_points = [], [], []
    for run in runs:
        points, draws = predictions(run, curves)
        worst_points.append(points.min())
        worst_draws.append(draws.min(axis=1))
        all_points.append(points)
    single_points, single_draws = [], []
    for run in single_runs:
        points, draws = predictions(run, curves)
        single_points.append(points[0])
        single_draws.append(draws[:, 0])
    worst = float(np.mean(worst_points))
    single = float(np.mean(single_points))
    loss_draws = np.mean(single_draws, axis=0) - np.mean(worst_draws, axis=0)
    loss_upper = float(np.percentile(loss_draws, 97.5))
    return SizeResult(
        robots=runs[0].robots,
        replications=len(runs),
        requests=requests,
        misses=misses,
        miss_share=share,
        criterion_1=requests > 0 and share <= MISS_SHARE_LIMIT,
        admitted_mean=float(np.mean([sum(r.admitted for r in run.fleet) for run in runs])),
        mean_batch=float(np.mean([run.mean_batch for run in runs])),
        busy_share=float(np.mean([run.busy_share for run in runs])),
        fleet_mean=float(np.mean(np.concatenate(all_points))),
        worst=worst,
        single=single,
        loss=single - worst,
        loss_upper=loss_upper,
        criterion_2=single - worst <= tolerance + 1e-12,
        criterion_2_conservative=loss_upper <= tolerance + 1e-12,
        medium_busy_share=_mean_or_none([run.medium_busy_share for run in runs]),
    )


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present and len(present) == len(values) else None


def capacity_of(sizes: Sequence[SizeResult], conservative: bool = False) -> int:
    """Largest size for which this size and every smaller one pass (sizes sorted ascending)."""
    capacity = 0
    for result in sorted(sizes, key=lambda s: s.robots):
        if not (result.passes_conservative if conservative else result.passes):
            break
        capacity = result.robots
    return capacity


@dataclass
class CapacityResult:
    key: ConfigKey
    condition: str
    capacity: int
    capacity_conservative: int
    sizes: list[SizeResult]


def evaluate(
    runs: Iterable[Run], curves: Mapping[str, BinCurves], gpu_condition: Mapping[str, str], tolerance: float
) -> list[CapacityResult]:
    grouped: dict[ConfigKey, dict[int, list[Run]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        grouped[run.key][run.robots].append(run)
    results = []
    for key in sorted(grouped):
        condition = gpu_condition.get(key[1])
        if condition is None:
            continue
        by_size = grouped[key]
        if 1 not in by_size:
            raise PredictionError(f"configuration {key} lacks the single-robot runs (n = 1)")
        sizes = [evaluate_size(by_size[n], by_size[1], curves[condition], tolerance) for n in sorted(by_size)]
        results.append(
            CapacityResult(
                key=key,
                condition=condition,
                capacity=capacity_of(sizes),
                capacity_conservative=capacity_of(sizes, conservative=True),
                sizes=sizes,
            )
        )
    return results


def result_json(results: Sequence[CapacityResult], tolerance: float) -> dict[str, Any]:
    return {
        "tolerance": tolerance,
        "miss_share_limit": MISS_SHARE_LIMIT,
        "configurations": [
            {
                "network": r.key[0],
                "gpu": r.key[1],
                "batching": r.key[2],
                "strategy": r.key[3],
                "condition": r.condition,
                "capacity": r.capacity,
                "capacity_conservative": r.capacity_conservative,
                "sizes": [asdict(s) for s in r.sizes],
            }
            for r in results
        ],
    }


# --------------------------------------------------------------------------------------------- CLI
def parse_curve_map(items: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        gpu, sep, condition = item.partition("=")
        if not sep or not gpu or not condition:
            raise ConfigError(f"--curve takes GPU=CONDITION, got {item!r}")
        mapping[gpu] = condition
    return mapping


def main(argv: list[str] | None = None) -> int:
    from kortyn.phase4_steps import fit_conditions, libero_layout, load_analysis_config, load_condition

    parser = argparse.ArgumentParser(description="Phase-4 capacity from stage-1 latencies and smooth curves.")
    parser.add_argument("--analysis", default="kortyn/experiments/phase4/analysis.yaml")
    parser.add_argument("--stage1", required=True, help="stage-1 results (scan or grid)")
    parser.add_argument("--curve", action="append", required=True, help="GPU profile = condition name")
    parser.add_argument("--out", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        config = load_analysis_config(args.analysis)
        gpu_condition = parse_curve_map(args.curve)
        names = sorted(set(gpu_condition.values()))
        unknown = [name for name in names if name not in config.conditions]
        if unknown:
            raise ConfigError(f"unknown conditions {unknown}")
        libero = libero_layout(config.experiment)
        doses = {name: load_condition(config.conditions[name], libero) for name in names}
        fits = fit_conditions(doses, config.spec, config.bootstrap, config.seed)
        runs = read_runs(Path(args.stage1))
        bins = max((run.fleet[0].histogram.size for run in runs if run.fleet), default=1)
        curves = {name: bin_curves(fit, bins) for name, fit in fits.items()}
        results = evaluate(runs, curves, gpu_condition, config.tolerance)
    except (ConfigError, PredictionError) as exc:
        LOG.error("%s", exc)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result_json(results, config.tolerance), indent=1), encoding="utf-8")
    for r in results:
        LOG.info(
            "%s/%s/%s/%s (%s): capacity %d (conservative %d)",
            *r.key,
            r.condition,
            r.capacity,
            r.capacity_conservative,
        )
    LOG.info("written %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
