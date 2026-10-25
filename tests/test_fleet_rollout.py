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
    one = fleet_config(tmp_path, experiment, workers=1, results=str(tmp_path / "val1"))
    run_grid(two, grid(two), Path(two.results_path), workers=1)
    case = ValidationCase("a", 2)

    meta = run_case(two, case, tmp_path / "val2" / case.label)
    assert meta["admitted"] == 2
    assert meta["episodes_counted"] == 4
    assert meta["inferences"] > 0
    episodes = list(read_jsonl(tmp_path / "val2" / case.label / "episodes.jsonl"))
    counted = [e for e in episodes if e["counted"]]
    assert sorted((e["robot"], e["seed"]) for e in counted) == [(0, 1000), (0, 1002), (1, 1001), (1, 1003)]
    assert all(e["clock"] == "virtual" for e in episodes)
    requests = list(read_jsonl(tmp_path / "val2" / case.label / "requests.jsonl"))
    assert {r["robot"] for r in requests} == {0, 1}

    # the plans do not depend on the number of inference processes: identical rollouts with one worker
    run_case(one, case, tmp_path / "val1" / case.label)
    for name in ("episodes.jsonl", "requests.jsonl"):
        left = sorted(
            json.dumps(deterministic_view(r), sort_keys=True)
            for r in read_jsonl(tmp_path / "val2" / case.label / name)
        )
        right = sorted(
            json.dumps(deterministic_view(r), sort_keys=True)
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
