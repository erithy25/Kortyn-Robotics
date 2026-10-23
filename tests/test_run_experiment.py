"""Integration test: real Brain and proxy subprocesses, real PushT env, tiny random policy (no download)."""

from __future__ import annotations

import json
import socket

import pytest

from kortyn import run_experiment
from kortyn.checkpoint import MARKER_FILE
from kortyn.metrics import read_jsonl

REPO_ID = "kortyn-test/tiny-pusht"
REVISION = "test"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def tiny_pusht_checkpoint(tmp_path_factory):
    """Tiny random diffusion policy with PushT-shaped inputs, pre-registered as a migrated checkpoint."""
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    from tests.conftest import tiny_diffusion_config, tiny_stats

    torch.manual_seed(0)
    config = tiny_diffusion_config()
    config.input_features["observation.image"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96))
    config.crop_shape = (84, 84)
    policy = DiffusionPolicy(config)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=tiny_stats())
    directory = tmp_path_factory.mktemp("tiny_pusht")
    policy.save_pretrained(directory)
    preprocessor.save_pretrained(directory)
    postprocessor.save_pretrained(directory)
    (directory / MARKER_FILE).write_text(json.dumps({"repo_id": REPO_ID, "revision": REVISION}))
    return directory


def write_config(tmp_path, checkpoint_dir):
    brain_port, proxy_port, control_port = free_port(), free_port(), free_port()
    text = f"""
brain:
  port: {brain_port}
  device: cpu
  torch_threads: 1
  warmup_inferences: 1
  checkpoint: {{repo_id: {REPO_ID}, revision: {REVISION}, local_dir: "{checkpoint_dir}"}}
proxy: {{port: {proxy_port}, control_port: {control_port}}}
agent:
  max_tick_lateness_ms: 100.0
  max_proxy_lateness_ms: 50.0
  max_cpu_steal_fraction: 1.0
env:
  max_episode_steps: 25
profiles:
  tiny: {{noise_scheduler_type: DDIM, num_inference_steps: 2}}
experiment:
  results_dir: "{tmp_path / "results"}"
  episodes: 2
  seed_start: 1000
  brain_startup_timeout_s: 180
  pin_cpus: false
  suites:
    - profile: tiny
      runs:
        - {{scenario: referenz_sync}}
        - {{scenario: gut}}
        - {{scenario: schlecht}}
"""
    path = tmp_path / "base.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_full_pipeline_writes_results_and_resumes(tmp_path, tiny_pusht_checkpoint):
    config = write_config(tmp_path, tiny_pusht_checkpoint)
    assert run_experiment.main(["--config", str(config), "--log-level", "WARNING"]) == 0

    root = tmp_path / "results" / "tiny"
    meta = json.loads((root / "run_meta.json").read_text())
    assert meta["brain"]["sampler"] == "DDIM-2"
    assert meta["runs"]["schlecht"]["network"]["rtt_ms"] == 80
    for run in ("referenz_sync", "gut", "schlecht"):
        episodes = list(read_jsonl(root / run / "episodes.jsonl"))
        requests = list(read_jsonl(root / run / "requests.jsonl"))
        assert [e["seed"] for e in episodes] == [1000, 1001], run
        assert all(e["valid"] for e in episodes), [e["invalid_reasons"] for e in episodes]
        assert all(e["steps"] == 25 or e["success"] for e in episodes)
        assert requests
        assert all(r["status"] == "ok" for r in requests)

    realtime = [r for r in read_jsonl(root / "schlecht" / "requests.jsonl") if r["phase"] == "realtime"]
    assert realtime
    # schlecht = RTT 80 ± 40 ms: the measured wire time can never be below the 40 ms minimum
    assert min(r["network_ms"] for r in realtime) >= 38.0
    gut = [r for r in read_jsonl(root / "gut" / "requests.jsonl") if r["phase"] == "realtime"]
    assert max(r["network_ms"] for r in gut) < 30.0
    sync = list(read_jsonl(root / "referenz_sync" / "episodes.jsonl"))
    assert all(e["starved_ticks"] == 0 for e in sync)

    # second invocation: everything is complete, nothing is re-run
    sizes = {
        run: (root / run / "episodes.jsonl").stat().st_size for run in ("referenz_sync", "gut", "schlecht")
    }
    assert run_experiment.main(["--config", str(config), "--log-level", "WARNING"]) == 0
    assert sizes == {run: (root / run / "episodes.jsonl").stat().st_size for run in sizes}

    # extending the suite later (ad-hoc run) keeps the metadata of the earlier runs
    extend = ["--config", str(config), "--run", "buero_wlan", "--log-level", "WARNING"]
    assert run_experiment.main(extend) == 0
    meta = json.loads((root / "run_meta.json").read_text())
    assert set(meta["runs"]) == {"referenz_sync", "gut", "schlecht", "buero_wlan"}
    assert meta["runs"]["schlecht"]["network"]["rtt_ms"] == 80
    assert [s["runs"] for s in meta["sessions"]] == [["referenz_sync", "gut", "schlecht"], ["buero_wlan"]]
    assert [e["seed"] for e in read_jsonl(root / "buero_wlan" / "episodes.jsonl")] == [1000, 1001]
    assert sizes == {run: (root / run / "episodes.jsonl").stat().st_size for run in sizes}


def stored_meta(**overrides):
    meta = {
        "profile": "p",
        "profile_settings": {"noise_scheduler_type": "DDIM", "num_inference_steps": 4},
        "brain": {"sampler": "DDIM-4", "device": "cpu", "torch_threads": 3, "warmup_inference_ms": 350.0},
        "runs": {"gut": {"scenario": "gut", "mode": "realtime", "network": {"rtt_ms": 0.0}}},
        "agent": {"refill_threshold": 6},
        "env": {"fps": 10.0},
        "seeds": [1000, 1049],
        "cpu_pinning": {"brain": [0, 1, 2], "agent_and_proxy": [3]},
        "host": {"cpu_count": 4},
        "git_commit": "aaa",
        "started_utc": "2026-01-01T00:00:00+00:00",
    }
    meta.update(overrides)
    return meta


def test_meta_of_an_extended_suite_keeps_earlier_runs_and_sessions():
    previous = stored_meta()  # written before sessions were recorded
    ablation = {"scenario": "x", "mode": "realtime", "network": {"rtt_ms": 80.0}}
    current = stored_meta(
        runs={**previous["runs"], "x": ablation},
        brain={**previous["brain"], "warmup_inference_ms": 360.0},
        seeds=[1000, 1099],
        git_commit="bbb",
        started_utc="2026-01-02T00:00:00+00:00",
    )
    merged = run_experiment.merge_run_meta(previous, current, ["x"], {"gut"})
    assert merged["runs"] == {"gut": previous["runs"]["gut"], "x": ablation}
    assert merged["started_utc"] == "2026-01-01T00:00:00+00:00"
    assert merged["seeds"] == [1000, 1099]
    assert [s["git_commit"] for s in merged["sessions"]] == ["aaa", "bbb"]
    assert [s["runs"] for s in merged["sessions"]] == [["gut"], ["x"]]
    assert [s["brain_warmup_inference_ms"] for s in merged["sessions"]] == [350.0, 360.0]
    assert len(run_experiment.merge_run_meta(merged, current, [], {"gut", "x"})["sessions"]) == 3


def test_meta_merge_refuses_to_mix_different_settings():
    previous = stored_meta()
    with pytest.raises(run_experiment.RunnerError, match="agent"):
        run_experiment.merge_run_meta(previous, stored_meta(agent={"refill_threshold": 4}), ["gut"], {"gut"})
    threads = stored_meta(brain={**previous["brain"], "torch_threads": 1})
    with pytest.raises(run_experiment.RunnerError, match=r"brain\.torch_threads"):
        run_experiment.merge_run_meta(previous, threads, ["gut"], {"gut"})
    changed = stored_meta(runs={"gut": {"scenario": "gut", "mode": "realtime", "network": {"rtt_ms": 5.0}}})
    with pytest.raises(run_experiment.RunnerError, match=r"runs\.gut"):
        run_experiment.merge_run_meta(previous, changed, ["gut"], {"gut"})
    # a measured value (warm-up time) may differ between sessions
    warmer = stored_meta(brain={**previous["brain"], "warmup_inference_ms": 900.0})
    merged = run_experiment.merge_run_meta(previous, warmer, ["gut"], {"gut"})
    assert merged["brain"]["warmup_inference_ms"] == 900.0
    # without stored episodes nothing would be mixed: the new settings replace the old file
    fresh = run_experiment.merge_run_meta(previous, changed, ["gut"], set())
    assert fresh["runs"] == changed["runs"]
    assert len(fresh["sessions"]) == 1


def test_unreadable_meta_is_reported_not_overwritten(tmp_path):
    path = tmp_path / "run_meta.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(run_experiment.RunnerError, match="unreadable"):
        run_experiment.read_run_meta(path)
    assert run_experiment.read_run_meta(tmp_path / "missing.json") is None


@pytest.mark.skipif(not hasattr(__import__("os"), "sched_setaffinity"), reason="Linux CPU affinity only")
def test_cpu_layout_ignores_the_runners_own_pinning():
    import os

    original = os.sched_getaffinity(0)
    if len(original) < 2:
        pytest.skip("needs at least two CPUs")
    brain_cores, agent_cores = run_experiment.cpu_layout(True)
    try:
        os.sched_setaffinity(0, set(agent_cores))  # what SuiteRunner does to itself during a suite
        assert run_experiment.cpu_layout(True) == (brain_cores, agent_cores)
        run_experiment.restore_affinity()
        assert os.sched_getaffinity(0) == original
    finally:
        os.sched_setaffinity(0, original)
    assert sorted(brain_cores + agent_cores) == sorted(original)
    assert run_experiment.cpu_layout(False) == ([], [])
