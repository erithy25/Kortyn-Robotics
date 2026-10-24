import pytest

from kortyn.config import (
    AgentConfig,
    ConfigError,
    KortynConfig,
    NetworkConfig,
    StallConfig,
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
