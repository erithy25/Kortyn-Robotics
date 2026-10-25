"""Fleet prediction (phase 3, stage 2): interpolation, equal task weights, bootstrap, criteria, capacity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kortyn.config import ConfigError
from kortyn.fleet import (
    BatchingConfig,
    EpisodeLengthConfig,
    FleetAgentConfig,
    FleetSimConfig,
    GpuConfig,
    LevelRun,
    NetworkChoice,
    PredictionConfig,
    RunKey,
    StrategyConfig,
    run_fleet,
)
from kortyn.fleet_predict import (
    DoseResponse,
    PredictionError,
    RobotRecord,
    RunRecord,
    capacity_of,
    episode_prediction,
    evaluate,
    evaluate_size,
    level_weights,
    load_dose_response,
    main,
    paired_difference_interval,
    read_stage1,
    robot_predictions,
)

LEVELS = np.array([0.0, 1000.0, 2000.0])
BASE_CONFIG = Path(__file__).resolve().parents[1] / "kortyn" / "experiments" / "phase3" / "libero_base.yaml"


def dose(outcomes: list[list[float]], groups: list[int]) -> DoseResponse:
    data = np.array(outcomes, dtype=float)
    return DoseResponse(
        levels_ms=LEVELS[: data.shape[0]],
        seeds=tuple(range(1000, 1000 + data.shape[1])),
        groups=tuple(groups),
        outcomes=data,
    )


# --------------------------------------------------------------------------------------------- weights
def test_level_weights_interpolate_linearly() -> None:
    np.testing.assert_allclose(level_weights(LEVELS, [250.0], [1]), [0.75, 0.25, 0.0])
    np.testing.assert_allclose(level_weights(LEVELS, [1000.0], [1]), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(level_weights(LEVELS, [500.0, 1500.0], [3, 1]), [0.375, 0.5, 0.125])


def test_level_weights_hold_the_end_values_outside_the_levels() -> None:
    np.testing.assert_allclose(level_weights(LEVELS, [5000.0], [2]), [0.0, 0.0, 1.0])  # extrapolation
    np.testing.assert_allclose(level_weights(LEVELS[1:], [10.0], [1]), [1.0, 0.0])  # below the first level


@pytest.mark.parametrize(("latencies", "counts"), [([], []), ([1.0], [0]), ([1.0], [-1]), ([1.0, 2.0], [1])])
def test_level_weights_reject_invalid_counts(latencies: list[float], counts: list[float]) -> None:
    with pytest.raises(PredictionError):
        level_weights(LEVELS, latencies, counts)


# --------------------------------------------------------------------------------------------- curve
def test_curve_weights_tasks_equally() -> None:
    # task 0: 3 seeds all succeed at 0 ms; task 1: 1 seed fails -> equal weights give 0.5, not 0.75
    response = dose([[1, 1, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]], groups=[0, 0, 0, 1])
    np.testing.assert_allclose(response.curve(), [0.5, 1 / 3, 0.0])


def test_bootstrap_resamples_within_tasks_and_unit_weights_give_the_curve() -> None:
    response = dose([[1, 0, 1, 0, 1], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]], groups=[0, 0, 0, 1, 1])
    weights = response.bootstrap_weights(200, seed=1)
    assert weights.shape == (200, 5)
    np.testing.assert_allclose(weights[:, :3].sum(axis=1), 3)  # three draws among task 0's seeds
    np.testing.assert_allclose(weights[:, 3:].sum(axis=1), 2)
    np.testing.assert_allclose(response.curves(np.ones((1, 5)))[0], response.curve())
    assert np.array_equal(weights, response.bootstrap_weights(200, seed=1))  # reproducible


def test_dose_response_rejects_inconsistent_data() -> None:
    with pytest.raises(PredictionError):
        DoseResponse(np.array([0.0, 0.0]), (1,), (0,), np.zeros((2, 1)))
    with pytest.raises(PredictionError):
        DoseResponse(np.array([0.0, 1.0]), (1, 2), (0,), np.zeros((2, 2)))
    with pytest.raises(PredictionError, match="not part"):
        dose([[1], [0]], [0]).seed_index(5)


# --------------------------------------------------------------------------------------------- criteria
def robot(latency_ms: float, requests: int = 100, misses: int = 0, admitted: bool = True) -> RobotRecord:
    weights = level_weights(LEVELS, [latency_ms], [1]) if admitted else None
    return RobotRecord(admitted, requests if admitted else 0, misses, weights, int(latency_ms > LEVELS[-1]))


def run(robots: list[RobotRecord], replication: int = 0) -> RunRecord:
    return RunRecord(
        key=("net", "gpu", "batch", "d"),
        robots=len(robots),
        replication=replication,
        requests=sum(r.requests for r in robots),
        misses=sum(r.misses for r in robots),
        admitted=sum(r.admitted for r in robots),
        mean_batch=1.0,
        busy_share=0.5,
        robot_records=robots,
    )


@pytest.fixture
def flat_then_falling() -> DoseResponse:
    # 40 seeds per task: 80 % success up to 1 000 ms, 20 % at 2 000 ms
    good = [1.0] * 32 + [0.0] * 8
    bad = [1.0] * 8 + [0.0] * 32
    return dose([good * 2, good * 2, bad * 2], groups=[0] * 40 + [1] * 40)


def test_rejected_robots_are_predicted_to_fail(flat_then_falling: DoseResponse) -> None:
    curves = flat_then_falling.curve()[np.newaxis, :]
    predictions = robot_predictions(run([robot(500.0), robot(0.0, admitted=False)]), curves)
    np.testing.assert_allclose(predictions, [[0.8, 0.0]])


def test_robots_below_the_budget_pass_and_late_robots_fail(flat_then_falling: DoseResponse) -> None:
    curve = flat_then_falling.curve()
    curves = flat_then_falling.curves(flat_then_falling.bootstrap_weights(2000, seed=3))
    single = [run([robot(300.0)], replication=r) for r in range(3)]
    fine = evaluate_size([run([robot(400.0), robot(900.0)], r) for r in range(3)], single, curve, curves)
    assert fine.criterion_1
    assert fine.criterion_2
    assert fine.passes
    assert fine.worst == pytest.approx(0.8)
    assert fine.difference == pytest.approx(0.0)
    late = evaluate_size([run([robot(400.0), robot(2000.0)], r) for r in range(3)], single, curve, curves)
    assert late.worst == pytest.approx(0.2)
    assert late.difference_ci[1] < 0
    assert not late.criterion_2
    missing = evaluate_size([run([robot(400.0, misses=2)], r) for r in range(3)], single, curve, curves)
    assert missing.miss_share == pytest.approx(0.02)
    assert not missing.criterion_1


def test_extrapolated_requests_are_counted(flat_then_falling: DoseResponse) -> None:
    curve = flat_then_falling.curve()
    single = [run([robot(300.0)])]
    result = evaluate_size([run([robot(300.0), robot(3000.0)])], single, curve, curve[np.newaxis, :])
    assert result.extrapolated_share == pytest.approx(1 / 200)


def test_capacity_needs_every_smaller_size_to_pass(flat_then_falling: DoseResponse) -> None:
    curve = flat_then_falling.curve()
    curves = curve[np.newaxis, :]
    single = [run([robot(300.0)])]
    sizes = {
        1: [run([robot(300.0)])],
        2: [run([robot(300.0)] * 2)],
        4: [run([robot(300.0), robot(300.0, misses=5), robot(300.0), robot(300.0)])],  # fails (1)
        8: [run([robot(300.0)] * 8)],  # passes again, but 4 failed
    }
    results = [evaluate_size(sizes[n], single, curve, curves) for n in sorted(sizes)]
    assert [r.passes for r in results] == [True, True, False, True]
    assert capacity_of(results) == 2
    assert capacity_of(results[2:3]) == 0


def test_evaluate_needs_single_robot_runs(flat_then_falling: DoseResponse) -> None:
    with pytest.raises(PredictionError, match="n = 1"):
        evaluate([run([robot(300.0)] * 2)], flat_then_falling, draws=100, seed=1)


# --------------------------------------------------------------------------------------------- validation
def test_episode_prediction_uses_the_seeds_own_outcomes() -> None:
    response = dose([[1, 1], [1, 0], [0, 0]], groups=[0, 1])
    weights = level_weights(LEVELS, [500.0], [1])
    assert episode_prediction(response, 1000, weights) == pytest.approx(1.0)
    assert episode_prediction(response, 1001, weights) == pytest.approx(0.5)


def test_paired_difference_interval() -> None:
    mean, (low, high) = paired_difference_interval([1, 0, 1, 0], [1, 0, 1, 0], 0.9875, 1000, seed=1)
    assert mean == 0.0
    assert (low, high) == (0.0, 0.0)
    mean, (low, high) = paired_difference_interval([1, 1, 1, 1], [0.5] * 4, 0.9875, 1000, seed=1)
    assert mean == pytest.approx(0.5)
    assert low == pytest.approx(0.5)
    with pytest.raises(PredictionError):
        paired_difference_interval([], [], 0.95, 100, seed=1)


# --------------------------------------------------------------------------------------------- end to end
def write_dose_runs(tmp_path: Path, levels: dict[int, list[bool]]) -> PredictionConfig:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"extends: {BASE_CONFIG}\n"
        "env: {libero: {task_ids: [3, 7], episodes_per_task: 2}}\n"
        f"experiment: {{results_dir: {tmp_path / 'results'}, episodes: 4}}\n",
        encoding="utf-8",
    )
    runs = []
    for latency, successes in levels.items():
        label = f"fest_{latency}ms"
        run_dir = tmp_path / "results" / "smolvla_libero" / label
        run_dir.mkdir(parents=True)
        rows = [{"seed": 1000 + i, "attempt": 0, "success": s, "steps": 300} for i, s in enumerate(successes)]
        rows.append({"seed": 1000, "attempt": 1, "success": not successes[0], "steps": 300})  # final attempt
        (run_dir / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        runs.append(LevelRun(latency_ms=float(latency), run=label))
    return PredictionConfig(
        experiment=str(experiment), levels=runs, bootstrap=200, output=str(tmp_path / "c.json")
    )


def test_load_dose_response_maps_seeds_to_tasks(tmp_path: Path) -> None:
    config = write_dose_runs(tmp_path, {1000: [True, False, True, True], 0: [True, True, True, False]})
    response = load_dose_response(config)
    np.testing.assert_allclose(response.levels_ms, [0.0, 1000.0])
    assert response.seeds == (1000, 1001, 1002, 1003)
    assert response.groups == (3, 3, 7, 7)
    np.testing.assert_allclose(response.outcomes, [[0, 1, 1, 0], [0, 0, 1, 1]])  # final attempt of seed 1000


def test_load_dose_response_rejects_missing_runs_and_seeds(tmp_path: Path) -> None:
    config = write_dose_runs(tmp_path, {0: [True] * 4, 1000: [True] * 4})
    path = Path(tmp_path / "results" / "smolvla_libero" / "fest_1000ms" / "episodes.jsonl")
    path.write_text(json.dumps({"seed": 1000, "attempt": 0, "success": True, "steps": 1}), encoding="utf-8")
    with pytest.raises(PredictionError, match="missing"):
        load_dose_response(config)
    path.unlink()
    with pytest.raises(PredictionError, match="does not exist"):
        load_dose_response(config)


def test_prediction_config_validation() -> None:
    with pytest.raises(ConfigError):
        PredictionConfig(experiment="x.yaml", levels=[LevelRun(0.0, "a")])
    with pytest.raises(ConfigError):
        PredictionConfig(experiment="x.yaml", levels=[LevelRun(0.0, "a"), LevelRun(0.0, "b")])
    with pytest.raises(ConfigError):
        LevelRun(-1.0, "a")


def test_stage1_results_to_capacity_end_to_end(tmp_path: Path) -> None:
    prediction = write_dose_runs(
        tmp_path, {0: [True, True, True, False], 1000: [True, True, True, False], 2000: [False] * 4}
    )
    config = FleetSimConfig(
        agent=FleetAgentConfig(),
        episode_lengths=EpisodeLengthConfig(values=[300, 520]),
        gpus={"gpu": GpuConfig(s1_ms=300.0)},
        batching={"keiner": BatchingConfig(beta=1.0)},
        strategies={"a": StrategyConfig(scheduler="fifo")},
        networks={"buero_wlan": NetworkChoice(scenario="buero_wlan", margin_ms=15.0)},
        fleet_sizes=[1, 2, 8],
        replications=1,
        horizon_s=90.0,
        window_start_s=10.0,
        results_path=str(tmp_path / "stage1.jsonl"),
        prediction=prediction,
    )
    with Path(config.results_path).open("w", encoding="utf-8") as out:
        for size in config.fleet_sizes:
            result = run_fleet(config, RunKey("buero_wlan", "gpu", "keiner", "a", size, 0), (300, 520))
            out.write(json.dumps(result.to_json()) + "\n")
    response = load_dose_response(prediction)
    records = read_stage1(Path(config.results_path), response.levels_ms)
    assert [r.robots for r in records] == [1, 2, 8]
    results = evaluate(records, response, draws=200, seed=1)
    assert len(results) == 1
    sizes = {s.robots: s for s in results[0].sizes}
    assert sizes[1].criterion_1
    assert sizes[2].criterion_1  # 2 x 0.3 s per 1.25 s: no misses
    assert not sizes[8].criterion_1  # 8 x 0.3 s per 1.25 s: overloaded FIFO
    assert results[0].capacity == 2
    # the CLI writes the same capacity
    config_path = tmp_path / "fleet.yaml"
    config_path.write_text(
        json.dumps(
            {
                "gpus": {"gpu": {"s1_ms": 300.0}},
                "batching": {"keiner": {"beta": 1.0}},
                "strategies": {"a": {"scheduler": "fifo"}},
                "networks": {"buero_wlan": {"scenario": "buero_wlan", "margin_ms": 15.0}},
                "episode_lengths": {"values": [300, 520]},
                "results_path": config.results_path,
                "prediction": {
                    "experiment": prediction.experiment,
                    "levels": [{"latency_ms": lv.latency_ms, "run": lv.run} for lv in prediction.levels],
                    "bootstrap": 200,
                    "output": prediction.output,
                },
            }
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(config_path), "--log-level", "WARNING"]) == 0
    written = json.loads(Path(prediction.output).read_text(encoding="utf-8"))
    assert written["configurations"][0]["capacity"] == 2
    assert written["tasks"] == [3, 7]
