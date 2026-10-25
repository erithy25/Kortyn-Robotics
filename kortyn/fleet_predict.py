"""Fleet prediction (phase 3, stage 2): success of every robot from its latency distribution, and capacity.

Stage 1 (:mod:`kortyn.fleet`) yields, per robot, a histogram of the end-to-end latencies of its requests.
Step 2 of phase 3 measured the dose-response curve: success of single-robot rollouts at fixed latencies
``L_1 < ... < L_K`` on the same seeds. This module joins the two, as pre-registered in
``reports/phase3.md`` (sections 0.5 and 0.6):

* **Curve.** ``f(L)`` is the success rate at each level, the tasks weighted equally, linearly interpolated
  between levels; beyond the last level it stays at the last value (reported as extrapolation).
* **Robot.** Its prediction is the mean of ``f`` over the latencies of its requests in the window. A robot
  the admission control rejected is not served: prediction 0.
* **Uncertainty.** A paired bootstrap over the seeds of the curve (resampled within each task) gives every
  quantity below a distribution; all of them use the same draws.
* **Capacity.** For a configuration (network, GPU, batching, strategy) and fleet size ``n``:
  (1) at most 1 % of all requests miss their budget; (2) the worst robot (per replication the robot with
  the lowest prediction, averaged over replications) is not significantly below a single robot on the same
  hardware (upper bound of the 95 % interval of the difference not below 0). Capacity is the largest
  ``n`` for which (1) and (2) hold for ``n`` and every smaller size of the grid.

The validation prediction of one rollout episode (seed ``s``, robot ``r``) uses that seed's own outcomes
``y_s(L)``, interpolated the same way, averaged over the latencies of robot ``r`` in stage 1.

    python -m kortyn.fleet_predict --config kortyn/experiments/phase3/fleet.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from kortyn.config import ConfigError, load_config
from kortyn.fleet import HISTOGRAM_BIN_MS, FleetSimConfig, PredictionConfig, load_fleet_config
from kortyn.libero_env import episode_slot
from kortyn.metrics import read_jsonl

LOG = logging.getLogger("kortyn.fleet_predict")

Floats = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]

MISS_SHARE_LIMIT = 0.01  # capacity criterion (1): at most 1 % of the requests miss their budget
CONFIDENCE = 0.95  # capacity criterion (2): 95 % interval of the worst-minus-single difference


class PredictionError(ValueError):
    """Inconsistent dose-response or stage-1 data."""


# --------------------------------------------------------------------------------------------- curve
@dataclass(frozen=True)
class DoseResponse:
    """Per-seed outcomes at fixed latencies: ``outcomes[k, i]`` = success of seed ``seeds[i]`` at level k."""

    levels_ms: Floats  # increasing, shape (K,)
    seeds: tuple[int, ...]
    groups: tuple[int, ...]  # task of each seed (equal-weight strata)
    outcomes: Floats  # shape (K, S), values 0/1

    def __post_init__(self) -> None:
        levels = np.asarray(self.levels_ms, dtype=float)
        if levels.ndim != 1 or levels.size < 2 or np.any(np.diff(levels) <= 0):
            raise PredictionError("levels must be at least two strictly increasing latencies")
        if self.outcomes.shape != (levels.size, len(self.seeds)) or len(self.groups) != len(self.seeds):
            raise PredictionError("outcomes must have shape (levels, seeds) and one group per seed")
        if not self.seeds:
            raise PredictionError("dose-response without seeds")

    @property
    def group_ids(self) -> list[int]:
        return sorted(set(self.groups))

    def _group_masks(self) -> list[Mask]:
        groups = np.asarray(self.groups)
        return [groups == group for group in self.group_ids]

    def curve(self) -> Floats:
        """``f`` at every level: success rate per task, tasks weighted equally."""
        rates = [self.outcomes[:, mask].mean(axis=1) for mask in self._group_masks()]
        return np.mean(rates, axis=0)  # type: ignore[no-any-return]

    def bootstrap_weights(self, draws: int, seed: int) -> Floats:
        """Resampling multiplicities, shape (draws, S): seeds drawn with replacement within each task."""
        rng = np.random.default_rng(seed)
        weights = np.zeros((draws, len(self.seeds)), dtype=float)
        for mask in self._group_masks():
            members = np.flatnonzero(mask)
            picks = rng.integers(0, members.size, size=(draws, members.size))
            for column, member in enumerate(members):
                weights[:, member] = (picks == column).sum(axis=1)
        return weights

    def curves(self, weights: Floats) -> Floats:
        """``f`` per bootstrap draw, shape (draws, K), from multiplicities of :meth:`bootstrap_weights`."""
        rates = []
        for mask in self._group_masks():
            w = weights[:, mask]  # (B, S_g)
            rates.append(w @ self.outcomes[:, mask].T / w.sum(axis=1, keepdims=True))  # (B, K)
        return np.mean(rates, axis=0)  # type: ignore[no-any-return]

    def seed_index(self, seed: int) -> int:
        try:
            return self.seeds.index(seed)
        except ValueError as exc:
            raise PredictionError(f"seed {seed} is not part of the dose-response runs") from exc


def level_weights(levels_ms: Floats, latencies_ms: Sequence[float], counts: Sequence[float]) -> Floats:
    """Weights ``a`` (sum 1) with ``mean over requests of f(L) = a @ f`` for linear interpolation.

    Latencies below the first level use the first level, above the last level the last one (extrapolation).
    """
    levels = np.asarray(levels_ms, dtype=float)
    values = np.asarray(latencies_ms, dtype=float)
    weight = np.asarray(counts, dtype=float)
    if values.shape != weight.shape or values.size == 0 or np.any(weight < 0) or weight.sum() <= 0:
        raise PredictionError("need matching, non-negative counts with a positive total")
    clipped = np.clip(values, levels[0], levels[-1])
    upper = np.clip(np.searchsorted(levels, clipped, side="right"), 1, levels.size - 1)
    lower = upper - 1
    span = levels[upper] - levels[lower]
    fraction = (clipped - levels[lower]) / span
    result = np.zeros(levels.size, dtype=float)
    np.add.at(result, lower, weight * (1.0 - fraction))
    np.add.at(result, upper, weight * fraction)
    return result / weight.sum()  # type: ignore[no-any-return]


def histogram_latencies(histogram: Mapping[Any, int]) -> tuple[list[float], list[float]]:
    """Bin centres (ms) and counts of a stage-1 latency histogram (keys: bin index, as int or str)."""
    centres = [(int(key) + 0.5) * HISTOGRAM_BIN_MS for key in histogram]
    counts = [float(count) for count in histogram.values()]
    return centres, counts


def load_dose_response(config: PredictionConfig) -> DoseResponse:
    """Outcomes of the dose-response runs (final attempt per seed); tasks from the LIBERO layout."""
    experiment = load_config(config.experiment)
    libero = experiment.env.libero  # LIBERO: tasks from the episode layout; other envs: one task
    root = Path(experiment.experiment.results_dir) / config.profile
    ordered = sorted(config.levels, key=lambda level: level.latency_ms)
    per_level: list[dict[int, bool]] = []
    for level in ordered:
        path = root / level.run / "episodes.jsonl"
        if not path.exists():
            raise PredictionError(f"dose-response run {path} does not exist")
        final: dict[int, dict[str, Any]] = {}
        for record in read_jsonl(path):
            seed = int(record["seed"])
            if seed not in final or int(record["attempt"]) >= int(final[seed]["attempt"]):
                final[seed] = record
        per_level.append({seed: bool(record["success"]) for seed, record in final.items()})
    seeds = sorted(set.intersection(*(set(level) for level in per_level)))
    missing = sorted(set.union(*(set(level) for level in per_level)) - set(seeds))
    if missing:
        raise PredictionError(f"seeds {missing[:5]}... are missing in some dose-response levels")
    outcomes = np.array([[float(level[seed]) for seed in seeds] for level in per_level])
    groups = tuple(episode_slot(libero, seed)[0] if libero is not None else 0 for seed in seeds)
    return DoseResponse(
        levels_ms=np.array([level.latency_ms for level in ordered]),
        seeds=tuple(seeds),
        groups=groups,
        outcomes=outcomes,
    )


# --------------------------------------------------------------------------------------------- stage 1
ConfigKey = tuple[str, str, str, str]  # network, gpu, batching, strategy


@dataclass
class RobotRecord:
    admitted: bool
    requests: int
    misses: int
    weights: Floats | None  # level weights of its latencies (None: no request in the window)
    extrapolated: int  # requests beyond the last level


@dataclass
class RunRecord:
    key: ConfigKey
    robots: int
    replication: int
    requests: int
    misses: int
    admitted: int
    mean_batch: float
    busy_share: float
    robot_records: list[RobotRecord]


def read_stage1(path: Path, levels_ms: Floats) -> list[RunRecord]:
    """Stage-1 results with the level weights of every robot."""
    if not path.exists():
        raise PredictionError(f"stage-1 results {path} do not exist")
    records: list[RunRecord] = []
    last = float(levels_ms[-1])
    for data in read_jsonl(path):
        key = data["key"]
        robots: list[RobotRecord] = []
        for robot in data["robots"]:
            weights = None
            extrapolated = 0
            if robot["histogram"]:
                centres, counts = histogram_latencies(robot["histogram"])
                weights = level_weights(levels_ms, centres, counts)
                extrapolated = int(sum(c for x, c in zip(centres, counts, strict=True) if x > last))
            robots.append(
                RobotRecord(
                    admitted=bool(robot["admitted"]),
                    requests=int(robot["requests"]),
                    misses=int(robot["misses"]),
                    weights=weights,
                    extrapolated=extrapolated,
                )
            )
        records.append(
            RunRecord(
                key=(key["network"], key["gpu"], key["batching"], key["strategy"]),
                robots=int(key["robots"]),
                replication=int(key["replication"]),
                requests=int(data["requests"]),
                misses=int(data["misses"]),
                admitted=int(data["admitted"]),
                mean_batch=float(data["mean_batch"]),
                busy_share=float(data["busy_share"]),
                robot_records=robots,
            )
        )
    return records


def robot_predictions(run: RunRecord, curves: Floats) -> Floats:
    """Predicted success of every robot of a run, shape (draws, robots); rejected or silent robots get 0."""
    columns = []
    for robot in run.robot_records:
        if robot.admitted and robot.weights is not None:
            columns.append(curves @ robot.weights)
        else:
            columns.append(np.zeros(curves.shape[0]))
    return np.stack(columns, axis=1)


# --------------------------------------------------------------------------------------------- capacity
@dataclass
class SizeResult:
    """Both criteria for one configuration and fleet size."""

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
    worst_ci: tuple[float, float]
    single: float
    difference: float  # worst - single
    difference_ci: tuple[float, float]
    criterion_2: bool
    extrapolated_share: float

    @property
    def passes(self) -> bool:
        return self.criterion_1 and self.criterion_2


@dataclass
class CapacityResult:
    key: ConfigKey
    capacity: int  # 0 if even one robot fails
    sizes: list[SizeResult]


def _interval(values: Floats, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


def evaluate_size(
    runs: Sequence[RunRecord],
    single_runs: Sequence[RunRecord],
    curve: Floats,
    curves: Floats,
    confidence: float = CONFIDENCE,
) -> SizeResult:
    """Criteria (1) and (2) for the replications of one configuration and size (all with equal ``robots``)."""
    if not runs or not single_runs:
        raise PredictionError("need runs of the size and of the single robot")
    requests = sum(run.requests for run in runs)
    misses = sum(run.misses for run in runs)
    miss_share = misses / requests if requests else 1.0
    point = curve[np.newaxis, :]
    worst_draws = np.mean([robot_predictions(run, curves).min(axis=1) for run in runs], axis=0)
    worst_point = float(np.mean([robot_predictions(run, point).min() for run in runs]))
    single_draws = np.mean([robot_predictions(run, curves)[:, 0] for run in single_runs], axis=0)
    single_point = float(np.mean([robot_predictions(run, point)[0, 0] for run in single_runs]))
    difference_ci = _interval(worst_draws - single_draws, confidence)
    fleet_mean = float(np.mean(np.concatenate([robot_predictions(run, point)[0] for run in runs])))
    total = sum(r.requests for run in runs for r in run.robot_records)
    extrapolated = sum(r.extrapolated for run in runs for r in run.robot_records)
    return SizeResult(
        robots=runs[0].robots,
        replications=len(runs),
        requests=requests,
        misses=misses,
        miss_share=miss_share,
        criterion_1=requests > 0 and miss_share <= MISS_SHARE_LIMIT,
        admitted_mean=float(np.mean([run.admitted for run in runs])),
        mean_batch=float(np.mean([run.mean_batch for run in runs])),
        busy_share=float(np.mean([run.busy_share for run in runs])),
        fleet_mean=fleet_mean,
        worst=worst_point,
        worst_ci=_interval(worst_draws, confidence),
        single=single_point,
        difference=worst_point - single_point,
        difference_ci=difference_ci,
        criterion_2=difference_ci[1] >= 0.0,
        extrapolated_share=extrapolated / total if total else 0.0,
    )


def capacity_of(sizes: Sequence[SizeResult]) -> int:
    """Largest size for which this size and every smaller one pass (sizes sorted ascending)."""
    capacity = 0
    for result in sorted(sizes, key=lambda s: s.robots):
        if not result.passes:
            break
        capacity = result.robots
    return capacity


def evaluate(
    records: Iterable[RunRecord],
    dose: DoseResponse,
    draws: int,
    seed: int,
    sizes: Iterable[int] | None = None,
) -> list[CapacityResult]:
    """Capacity of every configuration in the stage-1 records (optionally only the given fleet sizes)."""
    curve = dose.curve()
    curves = dose.curves(dose.bootstrap_weights(draws, seed))
    grouped: dict[ConfigKey, dict[int, list[RunRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.key][record.robots].append(record)
    allowed = None if sizes is None else set(sizes)
    results: list[CapacityResult] = []
    for key in sorted(grouped):
        by_size = grouped[key]
        if 1 not in by_size:
            raise PredictionError(f"configuration {key} lacks the single-robot runs (n = 1)")
        evaluated = [
            evaluate_size(by_size[n], by_size[1], curve, curves)
            for n in sorted(by_size)
            if allowed is None or n in allowed
        ]
        results.append(CapacityResult(key=key, capacity=capacity_of(evaluated), sizes=evaluated))
    return results


# --------------------------------------------------------------------------------------------- validation
def episode_prediction(dose: DoseResponse, seed: int, weights: Floats) -> float:
    """Prediction for one rollout episode: the seed's own outcomes ``y_s`` averaged with level weights."""
    return float(weights @ dose.outcomes[:, dose.seed_index(seed)])


def paired_difference_interval(
    measured: Sequence[float], predicted: Sequence[float], confidence: float, draws: int, seed: int
) -> tuple[float, tuple[float, float]]:
    """Mean of measured - predicted over episodes and its percentile-bootstrap interval."""
    differences = np.asarray(measured, dtype=float) - np.asarray(predicted, dtype=float)
    if differences.size == 0 or len(measured) != len(predicted):
        raise PredictionError("need matching, non-empty measured and predicted episodes")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, differences.size, size=(draws, differences.size))
    means = differences[picks].mean(axis=1)
    return float(differences.mean()), _interval(means, confidence)


# --------------------------------------------------------------------------------------------- output
def result_json(results: Sequence[CapacityResult], dose: DoseResponse) -> dict[str, Any]:
    return {
        "levels_ms": [float(x) for x in dose.levels_ms],
        "curve": [float(x) for x in dose.curve()],
        "seeds": len(dose.seeds),
        "tasks": dose.group_ids,
        "configurations": [
            {
                "network": r.key[0],
                "gpu": r.key[1],
                "batching": r.key[2],
                "strategy": r.key[3],
                "capacity": r.capacity,
                "sizes": [
                    {
                        "robots": s.robots,
                        "replications": s.replications,
                        "requests": s.requests,
                        "misses": s.misses,
                        "miss_share": s.miss_share,
                        "criterion_1": s.criterion_1,
                        "admitted_mean": s.admitted_mean,
                        "mean_batch": s.mean_batch,
                        "busy_share": s.busy_share,
                        "fleet_mean": s.fleet_mean,
                        "worst": s.worst,
                        "worst_ci": list(s.worst_ci),
                        "single": s.single,
                        "difference": s.difference,
                        "difference_ci": list(s.difference_ci),
                        "criterion_2": s.criterion_2,
                        "extrapolated_share": s.extrapolated_share,
                    }
                    for s in r.sizes
                ],
            }
            for r in results
        ],
    }


def prediction_config(config: FleetSimConfig) -> PredictionConfig:
    if config.prediction is None:
        raise ConfigError("the fleet config has no 'prediction' section")
    return config.prediction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-3 capacity from stage-1 latencies and dose-response.")
    parser.add_argument("--config", required=True, help="fleet config with a 'prediction' section")
    parser.add_argument(
        "--stage1", default=None, help="stage-1 results (default: results_path of the config)"
    )
    parser.add_argument("--sizes", default=None, help="comma-separated fleet sizes (default: all stored)")
    parser.add_argument("--out", default=None, help="output JSON (default: prediction.output)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        config = load_fleet_config(args.config)
        settings = prediction_config(config)
        dose = load_dose_response(settings)
        records = read_stage1(Path(args.stage1 or config.results_path), dose.levels_ms)
        sizes = None if args.sizes is None else [int(s) for s in args.sizes.split(",") if s.strip()]
        results = evaluate(records, dose, settings.bootstrap, settings.seed, sizes)
    except (ConfigError, PredictionError) as exc:
        LOG.error("%s", exc)
        return 2
    out = Path(args.out or settings.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result_json(results, dose), indent=1), encoding="utf-8")
    for result in results:
        LOG.info("%s: capacity %d", "/".join(result.key), result.capacity)
    LOG.info("written %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
