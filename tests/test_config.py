from dataclasses import replace

import pytest

from kortyn.config import (
    BASELINE_STRATEGY,
    AgentConfig,
    ConfigError,
    GuardConfig,
    KortynConfig,
    NetworkConfig,
    RunSpec,
    StallConfig,
    StrategyConfig,
    decode,
    load_config,
    load_scenario,
)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_key_is_rejected_with_its_name(tmp_path):
    path = write(tmp_path, "base.yaml", "brain:\n  prot: 1234\n")
    with pytest.raises(ConfigError, match="prot"):
        load_config(path)


def test_wrong_type_is_rejected(tmp_path):
    path = write(tmp_path, "base.yaml", "brain:\n  port: not-a-number\n")
    with pytest.raises(ConfigError, match="port"):
        load_config(path)


def test_semantic_error_message_survives_draccus_wrapping():
    with pytest.raises(ConfigError, match=r"rtt_ms must be >= 0"):
        decode(NetworkConfig, {"rtt_ms": -1}, "inline")


def test_stall_must_be_shorter_than_interval():
    with pytest.raises(ConfigError, match="shorter"):
        StallConfig(interval_s=1.0, duration_ms=1000.0)


def test_refill_threshold_must_be_below_chunk_length():
    with pytest.raises(ConfigError, match="refill_threshold"):
        AgentConfig(actions_per_chunk=8, refill_threshold=8)


def test_rtt_is_split_evenly_over_both_directions():
    network = NetworkConfig(rtt_ms=150.0, rtt_jitter_ms=80.0)
    assert network.one_way_delay_ms == 75.0
    assert network.one_way_jitter_ms == 40.0


def test_suite_must_reference_a_known_profile():
    data = {"profiles": {"a": {}}, "experiment": {"suites": [{"profile": "b", "runs": []}]}}
    with pytest.raises(ConfigError, match="unknown profile"):
        decode(KortynConfig, data, "inline")


def test_ports_must_be_distinct():
    data = {"brain": {"port": 6000}, "proxy": {"port": 6000, "control_port": 6001}}
    with pytest.raises(ConfigError, match="must all differ"):
        decode(KortynConfig, data, "inline")


def test_duplicate_run_labels_are_rejected():
    data = {
        "profiles": {"p": {}},
        "experiment": {"suites": [{"profile": "p", "runs": [{"scenario": "gut"}, {"scenario": "gut"}]}]},
    }
    with pytest.raises(ConfigError, match="duplicate run labels"):
        decode(KortynConfig, data, "inline")


def test_scenario_name_must_match_file_name(tmp_path):
    path = write(tmp_path, "schlecht.yaml", "name: gut\n")
    with pytest.raises(ConfigError, match="must match the file name"):
        load_scenario(str(path))


def test_missing_file_and_bad_yaml_are_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    path = write(tmp_path, "bad.yaml", "brain: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_extends_merges_mappings_and_replaces_lists(tmp_path):
    write(
        tmp_path,
        "base.yaml",
        "brain:\n  port: 7001\n  warmup_inferences: 2\nprofiles:\n  a: {num_inference_steps: 4}\n"
        "experiment:\n  episodes: 50\n  suites:\n    - profile: a\n      runs: [{scenario: gut}]\n",
    )
    (tmp_path / "phase2").mkdir()
    child = write(
        tmp_path / "phase2",
        "screening.yaml",
        "extends: ../base.yaml\nbrain:\n  port: 7002\nprofiles:\n  b: {num_inference_steps: 8}\n"
        "experiment:\n  episodes: 20\n  suites:\n    - profile: b\n      runs: [{scenario: schlecht}]\n",
    )
    config = load_config(child)
    assert config.brain.port == 7002
    assert config.brain.warmup_inferences == 2  # inherited from the parent mapping
    assert set(config.profiles) == {"a", "b"}
    assert config.experiment.episodes == 20
    assert [s.profile for s in config.experiment.suites] == ["b"]  # lists are replaced, not appended


def test_circular_extends_is_reported(tmp_path):
    write(tmp_path, "a.yaml", "extends: b.yaml\n")
    write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_strategies_decode_partially_and_baseline_is_phase_one():
    data = {
        "profiles": {"p": {}},
        "strategies": {
            "s10": {"request": {"refill_threshold": 10}, "guard": {"mode": "hold", "max_plan_age_s": 1.2}},
            "adaptive_inpaint": {"request": {"mode": "adaptive"}, "smoothing": {"mode": "inpaint"}},
        },
        "experiment": {
            "suites": [
                {
                    "profile": "p",
                    "runs": [
                        {"scenario": "gut", "strategy": "baseline"},
                        {"scenario": "gut", "strategy": "s10"},
                    ],
                }
            ]
        },
    }
    config = decode(KortynConfig, data, "inline")
    assert config.strategy("s10").request.refill_threshold == 10
    assert config.strategy("s10").smoothing.mode == "none"  # defaults fill the rest
    assert config.strategy("adaptive_inpaint").request.target_age_s == 0.7
    baseline = config.strategy(BASELINE_STRATEGY)
    assert baseline.request == StrategyConfig().request
    assert baseline.guard == GuardConfig()
    assert config.strategy(None) == StrategyConfig()
    labels = [run.run_label for run in config.experiment.suites[0].runs]
    assert labels == ["baseline__gut", "s10__gut"]
    assert RunSpec(scenario="gut").run_label == "gut"  # phase-1 runs keep their labels


def test_unknown_strategy_and_invalid_guard_are_rejected():
    data = {
        "profiles": {"p": {}},
        "experiment": {"suites": [{"profile": "p", "runs": [{"scenario": "gut", "strategy": "magic"}]}]},
    }
    with pytest.raises(ConfigError, match="unknown strategy"):
        decode(KortynConfig, data, "inline")
    with pytest.raises(ConfigError, match=r"needs guard\.mode 'hold'"):
        GuardConfig(mode="repeat_last", max_plan_age_s=1.0)
    too_late = {"strategies": {"s": {"request": {"refill_threshold": 15}}}}
    with pytest.raises(ConfigError, match="actions_per_chunk"):
        decode(KortynConfig, too_late, "inline")


def test_phase_two_experiments_inherit_the_phase_one_setup():
    from kortyn.config import EXPERIMENTS_DIR

    base = load_config()
    paths = sorted((EXPERIMENTS_DIR / "phase2").glob("*.yaml"))
    assert EXPERIMENTS_DIR / "phase2" / "screening.yaml" in paths
    for path in paths:
        config = load_config(path)
        ablation = path.stem.startswith("ablation_sync_")  # sync reference with another execution horizon
        assert config.brain == base.brain
        assert config.env == base.env
        if ablation:
            assert config.agent.sync_actions_per_chunk != base.agent.sync_actions_per_chunk
            assert (
                replace(config.agent, sync_actions_per_chunk=base.agent.sync_actions_per_chunk) == base.agent
            )
        else:
            assert config.agent == base.agent
        assert config.profiles["kalibriert"] == base.profiles["kalibriert"]
        assert config.experiment.seed_start == base.experiment.seed_start
        for suite in config.experiment.suites:
            assert suite.profile == "kalibriert"
            for run in suite.runs:
                if ablation:
                    assert (run.scenario, run.strategy) == ("referenz_sync", None)
                    continue
                assert run.strategy is not None
                config.strategy(run.strategy)
                load_scenario(run.scenario)
        assert config.strategy(BASELINE_STRATEGY) == StrategyConfig(
            description=config.strategy(BASELINE_STRATEGY).description
        )
