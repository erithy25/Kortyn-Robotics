"""Virtual-time runner end to end: real PushT env, tiny random diffusion policy, parallel workers, digests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kortyn.config import DEFAULT_BASE_CONFIG, ConfigError, RunSpec, load_config
from kortyn.metrics import read_jsonl
from kortyn.run_virtual import main, run_digest, select_virtual_plans

PUSHT_IMAGE = (3, 96, 96)


@pytest.fixture(scope="module")
def pusht_tiny_checkpoint(tmp_path_factory) -> Path:
    """A randomly initialized diffusion policy with PushT's input shapes, marked as already migrated."""
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    torch.manual_seed(0)
    config = DiffusionConfig(
        input_features={
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=PUSHT_IMAGE),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        down_dims=(16, 32),
        kernel_size=3,
        n_groups=4,
        diffusion_step_embed_dim=16,
        spatial_softmax_num_keypoints=4,
        crop_shape=(84, 84),
        crop_is_random=True,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        use_group_norm=True,
        num_train_timesteps=20,
        noise_scheduler_type="DDIM",
        num_inference_steps=4,
        device="cpu",
    )
    stats = {
        "observation.image": {
            "mean": torch.tensor([0.9, 0.9, 0.9]).reshape(3, 1, 1),
            "std": torch.tensor([0.2, 0.2, 0.2]).reshape(3, 1, 1),
        },
        "observation.state": {"min": torch.tensor([13.0, 32.0]), "max": torch.tensor([496.0, 510.0])},
        "action": {"min": torch.tensor([12.0, 25.0]), "max": torch.tensor([511.0, 511.0])},
    }
    directory = tmp_path_factory.mktemp("tiny_pusht")
    policy = DiffusionPolicy(config)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    policy.save_pretrained(directory)
    preprocessor.save_pretrained(directory)
    postprocessor.save_pretrained(directory)
    (directory / "kortyn_migration.json").write_text(
        json.dumps({"repo_id": "test/tiny-pusht", "revision": "0", "normalized_features": []})
    )
    return directory


def write_config(tmp_path: Path, checkpoint: Path, results: Path) -> Path:
    config = f"""
extends: {DEFAULT_BASE_CONFIG}
brain:
  checkpoint: {{repo_id: test/tiny-pusht, revision: "0", local_dir: {checkpoint}}}
env:
  max_episode_steps: 60
hardware_profiles:
  test_cpu:
    service:
      kalibriert: {{kind: lognormal, mean_ms: 400.0, std_ms: 150.0}}
timings:
  modell:
    latency: model
    hardware: test_cpu
    client_overhead: {{kind: constant, value_ms: 0.3}}
    transport_overhead: {{kind: uniform, low_ms: 2.0, high_ms: 5.0}}
    delivery_lateness: {{kind: constant, value_ms: 1.0}}
    tick_lateness: {{kind: uniform, low_ms: 0.0, high_ms: 2.0}}
experiment:
  results_dir: {results}
  episodes: 3
  workers: 2
  torch_threads: 1
  suites:
    - profile: kalibriert
      runs:
        - {{scenario: referenz_sync, timing: modell}}
        - {{scenario: gut, timing: modell}}
        - {{scenario: sehr_schlecht, timing: modell}}
"""
    path = tmp_path / "virtual.yaml"
    path.write_text(config, encoding="utf-8")
    return path


@pytest.mark.timeout(600)
def test_runs_are_reproducible_independent_of_worker_count_and_resumable(tmp_path, pusht_tiny_checkpoint):
    first, second = tmp_path / "a", tmp_path / "b"
    config = write_config(tmp_path, pusht_tiny_checkpoint, first)
    assert main(["--config", str(config), "--log-level", "WARNING"]) == 0
    # the same experiment again, with one worker (other order, other cache history) and resumed halfway
    assert (
        main(["--config", str(config), "--results-dir", str(second), "--workers", "1", "--episodes", "2"])
        == 0
    )
    assert main(["--config", str(config), "--results-dir", str(second), "--workers", "1"]) == 0
    for run in ("modell__referenz_sync", "modell__gut", "modell__sehr_schlecht"):
        a = first / "kalibriert" / run
        b = second / "kalibriert" / run
        assert run_digest(a) == run_digest(b), run
        episodes = list(read_jsonl(a / "episodes.jsonl"))
        assert sorted(e["seed"] for e in episodes) == [1000, 1001, 1002]
        assert {e["clock"] for e in episodes} == {"virtual"}
    stalls = [e for e in read_jsonl(first / "kalibriert" / "modell__sehr_schlecht" / "episodes.jsonl")]
    assert all(e["proxy"]["network"]["rtt_ms"] == 150.0 for e in stalls)
    latencies = [
        r["e2e_ms"]
        for r in read_jsonl(first / "kalibriert" / "modell__gut" / "requests.jsonl")
        if r["e2e_ms"]
    ]
    assert np.median(latencies) > 200.0  # service time ~ lognormal(400 ms), not the real (tiny) compute time
    meta = json.loads((first / "kalibriert" / "run_meta.json").read_text())
    assert meta["clock"] == "virtual"
    assert meta["runs"]["modell__gut"]["timing"] == "modell"
    assert meta["runs"]["modell__gut"]["service_time"]["kind"] == "lognormal"
    # a finished experiment has nothing left to do
    assert main(["--config", str(config)]) == 0
    assert run_digest(first / "kalibriert" / "modell__gut") == run_digest(
        second / "kalibriert" / "modell__gut"
    )


def test_plans_need_a_timing_and_replay_needs_a_realtime_scenario(tmp_path):
    config = load_config(DEFAULT_BASE_CONFIG)
    with pytest.raises(ConfigError, match="has no timing"):
        select_virtual_plans(config, "kalibriert", ["gut"])
    data = {
        "profiles": {"p": {}},
        "hardware_profiles": {"h": {"service": {"p": {"kind": "constant", "value_ms": 300.0}}}},
        "timings": {
            "m": {"latency": "model", "hardware": "h"},
            "r": {"latency": "replay"},
        },
        "experiment": {
            "suites": [
                {
                    "profile": "p",
                    "runs": [
                        {"scenario": "gut", "timing": "m"},
                        {"scenario": "gut", "timing": "r", "replay_run": "gut_wiederholung"},
                        {"scenario": "referenz_sync", "timing": "r"},
                    ],
                }
            ]
        },
    }
    from kortyn.config import KortynConfig, decode

    decoded = decode(KortynConfig, data, "inline")
    labels = [run.run_label for run in decoded.experiment.suites[0].runs]
    assert labels == ["m__gut", "r__gut_wiederholung", "r__referenz_sync"]
    plans = select_virtual_plans(decoded, "p", ["m__gut", "r__gut_wiederholung"])
    assert [p.replay_run for p in plans] == [None, "gut_wiederholung"]
    with pytest.raises(ConfigError, match="replay needs a real-time scenario"):
        select_virtual_plans(decoded, "p", ["r__referenz_sync"])
    with pytest.raises(ConfigError, match="unknown run label"):
        select_virtual_plans(decoded, "p", ["nope"])


def test_timing_references_are_validated():
    from kortyn.config import KortynConfig, decode

    base = {"profiles": {"p": {}}}
    with pytest.raises(ConfigError, match="unknown hardware profile"):
        decode(KortynConfig, {**base, "timings": {"m": {"latency": "model", "hardware": "x"}}}, "inline")
    with pytest.raises(ConfigError, match="needs a hardware profile"):
        decode(KortynConfig, {**base, "timings": {"m": {"latency": "model"}}}, "inline")
    no_service = {
        **base,
        "hardware_profiles": {"h": {"service": {"other": {"kind": "constant", "value_ms": 1.0}}}},
        "timings": {"m": {"latency": "model", "hardware": "h"}},
        "experiment": {"suites": [{"profile": "p", "runs": [{"scenario": "gut", "timing": "m"}]}]},
    }
    with pytest.raises(ConfigError, match="no service time for Brain profile"):
        decode(KortynConfig, no_service, "inline")
    with pytest.raises(ConfigError, match="unknown timing"):
        decode(
            KortynConfig,
            {
                **base,
                "experiment": {"suites": [{"profile": "p", "runs": [{"scenario": "gut", "timing": "t"}]}]},
            },
            "inline",
        )
    with pytest.raises(ConfigError, match="replay_run needs a replay timing"):
        decode(
            KortynConfig,
            {
                **base,
                "experiment": {
                    "suites": [{"profile": "p", "runs": [{"scenario": "gut", "replay_run": "x"}]}]
                },
            },
            "inline",
        )
    assert RunSpec(scenario="gut", strategy="s", timing="t").run_label == "t__s__gut"


def test_phase3a_experiment_configs_load_and_inherit_phase1():
    from kortyn.config import EXPERIMENTS_DIR

    base = load_config()
    for path in sorted((EXPERIMENTS_DIR / "phase3a").glob("validation_*.yaml")):
        config = load_config(path)
        assert config.env == base.env
        assert config.agent == base.agent
        assert config.profiles == base.profiles
        for suite in config.experiment.suites:
            for plan in select_virtual_plans(config, suite.profile, None):
                assert plan.timing_name in config.timings
