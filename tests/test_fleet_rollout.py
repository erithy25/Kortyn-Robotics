"""Validation rollouts (phase 3, stage 3): real robots sharing one batching Brain, end to end with PushT."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kortyn.config import DEFAULT_BASE_CONFIG, ConfigError
from kortyn.fleet import (
    BatchingConfig,
    EpisodeLengthConfig,
    FleetAgentConfig,
    FleetSimConfig,
    GpuConfig,
    LevelRun,
    NetworkChoice,
    PredictionConfig,
    StrategyConfig,
    ValidationConfig,
    grid,
    run_grid,
)
from kortyn.fleet_predict import load_dose_response
from kortyn.fleet_rollout import ValidationCase, assign_seeds, evaluate_case, run_case
from kortyn.metrics import read_jsonl
from kortyn.run_virtual import main as run_virtual_main
from kortyn.virtual_agent import deterministic_view
from tests.test_run_virtual import pusht_tiny_checkpoint  # noqa: F401  (module-scoped fixture)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="forked inference workers: Linux only")


def test_assign_seeds_round_robin() -> None:
    seeds = list(range(1000, 1010))
    assert assign_seeds(seeds, 3, 10) == [
        [1000, 1003, 1006, 1009],
        [1001, 1004, 1007, 1000],  # ceil(10 / 3) = 4 each: the last ones wrap around
        [1002, 1005, 1008, 1001],
    ]
    assert assign_seeds(seeds, 1, 2) == [[1000, 1001]]
    with pytest.raises(ConfigError):
        assign_seeds(seeds, 0, 2)


def rounded(value: object) -> object:
    """Floats to 9 significant digits, recursively (PushT physics differs in the last bits between runs)."""
    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def write_experiment(tmp_path: Path, checkpoint: Path) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        f"""
extends: {DEFAULT_BASE_CONFIG}
brain:
  checkpoint: {{repo_id: test/tiny-pusht, revision: "0", local_dir: {checkpoint}}}
agent:
  start_mode: cold
env:
  max_episode_steps: 40
hardware_profiles:
  fest_0ms: {{service: {{kalibriert: {{kind: constant, value_ms: 0}}}}}}
  fest_800ms: {{service: {{kalibriert: {{kind: constant, value_ms: 800}}}}}}
timings:
  fest_0ms: {{latency: model, hardware: fest_0ms}}
  fest_800ms: {{latency: model, hardware: fest_800ms}}
experiment:
  results_dir: {tmp_path / "dose"}
  episodes: 4
  workers: 1
  suites:
    - profile: kalibriert
      runs:
        - {{scenario: gut, timing: fest_0ms, label: fest_0ms}}
        - {{scenario: gut, timing: fest_800ms, label: fest_800ms}}
""",
        encoding="utf-8",
    )
    return path


def fleet_config(tmp_path: Path, experiment: Path, workers: int, results: str) -> FleetSimConfig:
    return FleetSimConfig(
        agent=FleetAgentConfig(fps=10.0, actions_per_chunk=15, refill_threshold=6, request_timeout_s=5.0),
        episode_lengths=EpisodeLengthConfig(
            paths=[str(tmp_path / "dose" / "kalibriert" / "fest_0ms" / "episodes.jsonl")]
        ),
        gpus={"gpu": GpuConfig(s1_ms=300.0)},
        batching={"keiner": BatchingConfig(beta=1.0)},
        strategies={"a": StrategyConfig(scheduler="fifo")},
        networks={"gut": NetworkChoice(scenario="gut", margin_ms=0.0)},
        fleet_sizes=[1, 2],
        replications=1,
        horizon_s=60.0,
        window_start_s=5.0,
        results_path=str(tmp_path / "stage1.jsonl"),
        prediction=PredictionConfig(
            experiment=str(experiment),
            profile="kalibriert",
            levels=[LevelRun(0.0, "fest_0ms"), LevelRun(800.0, "fest_800ms")],
            bootstrap=200,
        ),
        validation=ValidationConfig(
            experiment=str(experiment),
            template_run="fest_0ms",
            gpu="gpu",
            batching="keiner",
            network="gut",
            profile="kalibriert",
            results_dir=results,
            episodes=4,
            workers=workers,
            bootstrap=200,
        ),
    )


@pytest.mark.timeout(900)
def test_validation_rollouts_end_to_end(tmp_path: Path, pusht_tiny_checkpoint: Path) -> None:  # noqa: F811
    experiment = write_experiment(tmp_path, pusht_tiny_checkpoint)
    assert run_virtual_main(["--config", str(experiment), "--log-level", "WARNING"]) == 0
    two = fleet_config(tmp_path, experiment, workers=2, results=str(tmp_path / "val2"))
    run_grid(two, grid(two), Path(two.results_path), workers=1)
    case = ValidationCase("a", 2)

    meta = run_case(two, case, tmp_path / "val2" / case.label)
    assert meta["admitted"] == 2
    assert meta["episodes_counted"] == 4
    assert meta["inferences"] > 0
    assert meta["inference_workers"] == 2
    episodes = list(read_jsonl(tmp_path / "val2" / case.label / "episodes.jsonl"))
    counted = [e for e in episodes if e["counted"]]
    assert sorted((e["robot"], e["seed"]) for e in counted) == [(0, 1000), (0, 1002), (1, 1001), (1, 1003)]
    assert all(e["clock"] == "virtual" for e in episodes)
    requests = list(read_jsonl(tmp_path / "val2" / case.label / "requests.jsonl"))
    assert {r["robot"] for r in requests} == {0, 1}

    # the plans do not depend on the number of inference processes: identical rollouts with one worker
    # (PushT physics, pymunk, is not bit-reproducible in the last float digits: compare 9 digits)
    assert run_case(two, case, tmp_path / "val1" / case.label, workers=1)["inference_workers"] == 1
    for name in ("episodes.jsonl", "requests.jsonl"):
        left = sorted(
            json.dumps(rounded(deterministic_view(r)), sort_keys=True)
            for r in read_jsonl(tmp_path / "val2" / case.label / name)
        )
        right = sorted(
            json.dumps(rounded(deterministic_view(r)), sort_keys=True)
            for r in read_jsonl(tmp_path / "val1" / case.label / name)
        )
        assert left == right, name

    assert two.prediction is not None
    dose = load_dose_response(two.prediction)
    assert dose.group_ids == [0]  # PushT: one task
    evaluation = evaluate_case(two, case, dose, tmp_path / "val2" / case.label)
    assert evaluation.episodes == 4
    assert 0.0 <= evaluation.measured <= 1.0
    assert 0.0 <= evaluation.predicted <= 1.0
    assert evaluation.interval[0] <= evaluation.difference <= evaluation.interval[1]
    assert evaluation.predicted_own is not None
    assert 0.0 <= evaluation.predicted_own <= 1.0
    assert evaluation.interval_own is not None
    assert evaluation.latency_ms_stage1 is not None
    assert evaluation.latency_ms_measured is not None
    assert evaluation.latency_ms_measured[0] <= evaluation.latency_ms_measured[1]
    with pytest.raises(ConfigError, match="already holds"):
        run_case(two, case, tmp_path / "val2" / case.label)


def test_validation_config_must_reference_known_entries(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"validation\.gpu"):
        FleetSimConfig(
            gpus={"gpu": GpuConfig(s1_ms=100.0)},
            batching={"keiner": BatchingConfig(beta=1.0)},
            strategies={"a": StrategyConfig(scheduler="fifo")},
            networks={"gut": NetworkChoice(scenario="gut", margin_ms=0.0)},
            validation=ValidationConfig(
                experiment="x.yaml", template_run="r", gpu="other", batching="keiner", network="gut"
            ),
        )
    with pytest.raises(ConfigError):
        ValidationConfig(experiment="x", template_run="r", gpu="g", batching="b", network="n", workers=0)


def capacity_document(per_strategy: dict[str, tuple[int, list[bool]]]) -> dict[str, object]:
    """Stage-2 results of the validation configuration: capacity and criterion (1) at sizes 1, 2, 4, 8."""
    configurations = []
    for strategy, (capacity, meets) in per_strategy.items():
        sizes = [
            {"robots": n, "criterion_1": ok, "miss_share": 0.0 if ok else 0.3}
            for n, ok in zip((1, 2, 4, 8), meets, strict=True)
        ]
        batching = "realistisch"
        configurations.append(
            {
                "network": "gut",
                "gpu": "gpu",
                "batching": batching,
                "strategy": strategy,
                "capacity": capacity,
                "sizes": sizes,
            }
        )
    return {"configurations": configurations}


def selection_fleet() -> FleetSimConfig:
    return FleetSimConfig(
        gpus={"gpu": GpuConfig(s1_ms=420.0)},
        batching={"realistisch": BatchingConfig(beta=0.15)},
        strategies={
            "a": StrategyConfig(scheduler="fifo"),
            "d": StrategyConfig(scheduler="edf_batch", max_batch=16),
            "e": StrategyConfig(scheduler="edf_batch", max_batch=16, admission=True),
        },
        networks={"gut": NetworkChoice(scenario="gut", margin_ms=0.0)},
        validation=ValidationConfig(
            experiment="x.yaml", template_run="fest_0ms", gpu="gpu", batching="realistisch", network="gut"
        ),
    )


def test_select_cases_follows_the_preregistered_rule() -> None:
    from kortyn.fleet_rollout import select_cases

    fleet = selection_fleet()
    document = capacity_document(
        {"a": (2, [True, True, False, False]), "d": (8, [True] * 4), "e": (4, [True, True, True, False])}
    )
    cases = select_cases(document, fleet)
    assert [(c.strategy, c.robots) for c in cases] == [("a", 2), ("a", 4), ("d", 2), ("d", 4)]
    # the worst still meets (1) at 4 robots: n1, n2 = 2, 4
    relaxed = capacity_document({"a": (4, [True] * 4), "d": (8, [True] * 4)})
    assert [(c.strategy, c.robots) for c in select_cases(relaxed, fleet)] == [
        ("a", 2),
        ("a", 4),
        ("d", 2),
        ("d", 4),
    ]
    # the worst meets (1) only with one robot: n1 = 1, n2 = 2
    tight = capacity_document({"a": (1, [True, False, False, False]), "d": (2, [True, True, False, False])})
    assert [(c.strategy, c.robots) for c in select_cases(tight, fleet)] == [
        ("a", 1),
        ("a", 2),
        ("d", 1),
        ("d", 2),
    ]


def test_select_cases_breaks_ties_by_miss_share_and_rejects_degenerate_results() -> None:
    from kortyn.fleet_predict import PredictionError
    from kortyn.fleet_rollout import select_cases

    fleet = selection_fleet()
    document = capacity_document({"a": (2, [True, True, False, False]), "d": (2, [True, True, False, False])})
    entries = document["configurations"]
    assert isinstance(entries, list)
    entries[1]["sizes"][3]["miss_share"] = 0.1  # d misses less at 8 robots: d is the better one
    cases = select_cases(document, fleet)
    assert {c.strategy for c in cases[:2]} == {"a"}
    assert {c.strategy for c in cases[2:]} == {"d"}
    with pytest.raises(PredictionError):
        select_cases({"configurations": []}, fleet)
    with pytest.raises(PredictionError, match="one robot"):
        select_cases(capacity_document({"a": (0, [False] * 4), "d": (1, [True, False, False, False])}), fleet)


def test_episode_latencies_and_weighted_quantiles(tmp_path: Path) -> None:
    from kortyn.fleet_rollout import _weighted_quantiles, episode_latencies

    path = tmp_path / "requests.jsonl"
    rows = [
        {"robot": 0, "robot_episode": 0, "t_obs_s": 0.0, "t_resp_s": 0.4},
        {"robot": 0, "robot_episode": 0, "t_obs_s": 1.25, "t_resp_s": 1.95},
        {"robot": 1, "robot_episode": 0, "t_obs_s": 0.0, "t_resp_s": 10.0},  # timeout: counts at 10 s
        {"robot": 1, "robot_episode": 0, "t_obs_s": 2.0, "t_resp_s": None},  # never answered: skipped
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    latencies = episode_latencies(path)
    assert latencies[(0, 0)] == pytest.approx([400.0, 700.0])
    assert latencies[(1, 0)] == pytest.approx([10_000.0])
    assert _weighted_quantiles([300.0, 100.0, 200.0], [1.0, 1.0, 2.0], (0.5, 0.95)) == [200.0, 300.0]
