import numpy as np
import pytest

from kortyn import checkpoint as checkpoint_module
from kortyn.checkpoint import (
    CheckpointError,
    ensure_checkpoint,
    is_processor_format,
    migrate_legacy_checkpoint,
)
from kortyn.config import BrainProfile, CheckpointConfig
from kortyn.policy_runner import PolicyRunner

PROFILE = BrainProfile(noise_scheduler_type="DDIM", num_inference_steps=4)


def test_legacy_checkpoint_migrates_to_an_equivalent_policy(tmp_path, legacy_checkpoint, tiny_checkpoint):
    assert not is_processor_format(legacy_checkpoint)
    report = migrate_legacy_checkpoint(legacy_checkpoint, tmp_path / "migrated")
    assert is_processor_format(tmp_path / "migrated")
    assert report.normalized_features == ("action", "observation.image", "observation.state")
    # image mean/std + state min/max + action min/max for both normalize_targets and unnormalize_outputs
    assert report.removed_keys == 8

    migrated = PolicyRunner(tmp_path / "migrated", PROFILE, device="cpu", torch_threads=1)
    original = PolicyRunner(tiny_checkpoint, PROFILE, device="cpu", torch_threads=1)
    rng = np.random.default_rng(5)
    observation = {
        "observation.image": rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8),
        "observation.state": rng.uniform(0, 512, size=(2, 2)).astype(np.float32),
    }
    np.testing.assert_allclose(
        migrated.infer(observation, seed=3, max_actions=15).actions,
        original.infer(observation, seed=3, max_actions=15).actions,
        rtol=1e-6,
        atol=1e-5,
    )


def test_checkpoint_without_normalization_buffers_is_rejected(tmp_path, tiny_checkpoint):
    with pytest.raises(CheckpointError, match="no normalization buffers"):
        migrate_legacy_checkpoint(tiny_checkpoint, tmp_path / "out")


def test_ensure_checkpoint_downloads_once_and_caches(tmp_path, monkeypatch, legacy_checkpoint):
    calls = []

    def fake_snapshot_download(repo_id, revision, allow_patterns):
        calls.append((repo_id, revision))
        return str(legacy_checkpoint)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    config = CheckpointConfig(repo_id="org/legacy", revision="abc123", local_dir=str(tmp_path / "ckpt"))
    first = ensure_checkpoint(config)
    assert is_processor_format(first)
    assert (first / checkpoint_module.MARKER_FILE).is_file()
    second = ensure_checkpoint(config)
    assert second == first
    assert calls == [("org/legacy", "abc123")]

    # a different pinned revision triggers a rebuild
    ensure_checkpoint(
        CheckpointConfig(repo_id="org/legacy", revision="def456", local_dir=str(tmp_path / "ckpt"))
    )
    assert calls[-1] == ("org/legacy", "def456")


def test_download_failure_is_reported(tmp_path, monkeypatch):
    import huggingface_hub

    def failing_download(**_kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: failing_download(**k))
    with pytest.raises(CheckpointError, match="network unreachable"):
        ensure_checkpoint(CheckpointConfig(local_dir=str(tmp_path / "x")))
