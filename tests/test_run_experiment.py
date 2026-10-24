"""Integration test: real Brain and proxy subprocesses, real PushT env, tiny random policy (no download)."""

from __future__ import annotations

import json
import socket

import pytest

from kortyn import report, run_experiment
from kortyn.checkpoint import MARKER_FILE
from kortyn.metrics import EpisodeRecord, final_episodes, read_jsonl

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

    # extending the suite later (ad-hoc ablation run) keeps the metadata of the earlier runs
    extend = ["--config", str(config), "--run", "schlecht_ohne_stall", "--log-level", "WARNING"]
    assert run_experiment.main(extend) == 0
    meta = json.loads((root / "run_meta.json").read_text())
    assert set(meta["runs"]) == {"referenz_sync", "gut", "schlecht", "schlecht_ohne_stall"}
    assert meta["runs"]["schlecht"]["network"]["stall"]["duration_ms"] == 300
    assert meta["runs"]["schlecht_ohne_stall"]["network"]["stall"] is None
    assert [s["runs"] for s in meta["sessions"]] == [
        ["referenz_sync", "gut", "schlecht"],
        ["schlecht_ohne_stall"],
    ]
    assert [e["seed"] for e in read_jsonl(root / "schlecht_ohne_stall" / "episodes.jsonl")] == [1000, 1001]
    assert sizes == {run: (root / run / "episodes.jsonl").stat().st_size for run in sizes}

    # the report renders every table and figure from these results, including the ablation section
    reports = tmp_path / "reports"
    render = ["--config", str(config), "--reports-dir", str(reports), "--log-level", "WARNING"]
    assert report.main(render) == 0
    text = (reports / "phase1.md").read_text(encoding="utf-8")
    for heading in ("## Auf einen Blick", "Brain-Profil `tiny`", "### Ablation", "## Definitionen"):
        assert heading in text
    for name in (
        "phase1_erfolgsquote.png",
        "phase1_latenzverteilung.png",
        "phase1_latenz_zerlegung.png",
        "phase1_starvation.png",
        "phase1_queue_tiny.png",
        "phase1_ablation_tiny.png",
        "phase1_aktionsalter.png",
    ):
        assert (reports / name).stat().st_size > 10_000, name


def test_phase_two_strategies_run_end_to_end_with_inpainting(tmp_path, tiny_pusht_checkpoint):
    """Real Brain + proxy + agent with the tiny diffusion policy: guides travel through the proxy, the
    Brain inpaints, load counters are read directly from the Brain, strategies land in the metadata."""
    config = write_config(tmp_path, tiny_pusht_checkpoint)
    text = config.read_text(encoding="utf-8")
    text = text.replace(
        "experiment:",
        "strategies:\n"
        "  baseline: {}\n"
        "  adaptiv_inpaint:\n"
        "    request: {mode: adaptive, target_age_s: 0.6}\n"
        "    smoothing: {mode: inpaint, schedule: exp}\n"
        "    guard: {mode: hold, max_plan_age_s: 1.2}\n"
        "experiment:",
    )
    text = text.replace(
        "        - {scenario: referenz_sync}\n        - {scenario: gut}\n        - {scenario: schlecht}\n",
        "        - {scenario: schlecht, strategy: baseline}\n"
        "        - {scenario: schlecht, strategy: adaptiv_inpaint}\n",
    )
    config.write_text(text, encoding="utf-8")
    assert run_experiment.main(["--config", str(config), "--log-level", "WARNING"]) == 0

    root = tmp_path / "results" / "tiny"
    meta = json.loads((root / "run_meta.json").read_text())
    assert meta["runs"]["adaptiv_inpaint__schlecht"]["strategy"] == "adaptiv_inpaint"
    settings = meta["runs"]["adaptiv_inpaint__schlecht"]["strategy_settings"]
    assert settings["smoothing"]["mode"] == "inpaint"
    assert meta["runs"]["baseline__schlecht"]["strategy_settings"]["request"]["mode"] == "threshold"

    attempts = list(read_jsonl(root / "adaptiv_inpaint__schlecht" / "episodes.jsonl"))
    # the runner repeats attempts that the validity gate rejects (machine noise): check the counted ones
    counted = {(e.seed, e.attempt) for e in final_episodes([EpisodeRecord.from_json(a) for a in attempts])[0]}
    inpaint_episodes = [e for e in attempts if (e["seed"], e["attempt"]) in counted]
    inpaint_requests = [
        r
        for r in read_jsonl(root / "adaptiv_inpaint__schlecht" / "requests.jsonl")
        if r["phase"] == "realtime" and (r["seed"], r["attempt"]) in counted
    ]
    assert len(inpaint_episodes) == 2
    assert all(e["valid"] for e in inpaint_episodes), [e["invalid_reasons"] for e in attempts]
    assert all(e["strategy"] == "adaptiv_inpaint" for e in attempts)
    assert all(r["status"] == "ok" for r in inpaint_requests)
    assert all(r["guide_frozen"] >= 2 for r in inpaint_requests)
    assert {r["trigger"] for r in inpaint_requests} <= {"freshness", "deadline"}
    for episode in inpaint_episodes:
        assert episode["brain_load"] is not None
        assert episode["brain_load"]["requests"] >= 1
        assert len(episode["ticks"]["plan_age_ms"]) == episode["steps"]
    baseline_requests = [
        r for r in read_jsonl(root / "baseline__schlecht" / "requests.jsonl") if r["phase"] == "realtime"
    ]
    assert all(r["guide_frozen"] == 0 and r["trigger"] == "threshold" for r in baseline_requests)


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
