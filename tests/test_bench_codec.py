"""Phase-4 codec benchmark (kortyn.bench_codec): capture, measurement, model-input check, selection rule."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch

from kortyn import bench_codec as bc
from kortyn.config import ConfigError, ImageTransportConfig


def test_random_walk_is_seeded_bounded_and_toggles_the_gripper() -> None:
    a = bc.random_walk(1000, 120, 7)
    assert a.shape == (120, 7)
    assert a.dtype == np.float32
    assert np.array_equal(a, bc.random_walk(1000, 120, 7))
    assert not np.array_equal(a, bc.random_walk(1001, 120, 7))
    assert np.all(np.abs(a) <= 1.0)
    assert set(a[:50, -1].tolist()) == {-1.0}
    assert set(a[50:100, -1].tolist()) == {1.0}


@dataclass
class FakeStep:
    terminated: bool = False
    truncated: bool = False


@dataclass
class FakeEnv:
    """Two cameras whose images encode (seed, tick), so captures can be traced."""

    image_keys: frozenset[str] = frozenset({"cam_b", "cam_a"})
    action_dim: int = 7
    seed: int = 0
    tick: int = 0
    resets: list[int] = field(default_factory=list)

    def reset(self, seed: int) -> None:
        self.seed, self.tick = seed, 0
        self.resets.append(seed)

    def step(self, action: np.ndarray) -> FakeStep:
        assert action.shape == (7,)
        self.tick += 1
        return FakeStep()

    def render_image(self, key: str) -> np.ndarray:
        value = (self.seed + self.tick + (1 if key == "cam_b" else 0)) % 256
        return np.full((8, 8, 3), value, dtype=np.uint8)


def test_capture_takes_both_cameras_at_the_capture_ticks() -> None:
    env = FakeEnv()
    requests = bc.capture_requests(env, [10, 20], (0, 3))
    assert len(requests) == 4
    assert env.resets == [10, 20]
    assert sorted(requests[0]) == ["cam_a", "cam_b"]
    assert requests[0]["cam_a"].shape == (1, 8, 8, 3)
    assert int(requests[1]["cam_a"][0, 0, 0, 0]) == 13  # seed 10, tick 3
    assert int(requests[3]["cam_b"][0, 0, 0, 0]) == 24  # seed 20, tick 3, camera b
    with pytest.raises(ConfigError):
        bc.capture_requests(env, [1], ())


def synthetic_requests() -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    return [
        {
            "a": rng.integers(0, 256, size=(1, 24, 24, 3), dtype=np.uint8),
            "b": np.full((1, 24, 24, 3), i, dtype=np.uint8),
        }
        for i in range(3)
    ]


def test_measure_counts_identical_frames_and_bytes() -> None:
    requests = synthetic_requests()
    lossless, decoded = bc.measure("png_1", ImageTransportConfig(codec="png", level=1), requests)
    assert lossless.identical_frames == 3
    assert lossless.requests == 3
    assert all(np.array_equal(d["a"], r["a"]) for d, r in zip(decoded, requests, strict=True))
    assert lossless.mean_bytes > 0
    assert lossless.encode_ms >= 0
    assert lossless.decode_ms >= 0
    assert lossless.mbit_per_robot == pytest.approx(lossless.mean_bytes * 8 / 1.25 / 1e6)
    lossy, _ = bc.measure("jpeg", ImageTransportConfig(codec="jpeg", quality=50), requests)
    assert lossy.identical_frames < 3


class FakeRunner:
    """Mimics the parts of PolicyRunner the model-input check uses."""

    needs_task = True

    def _to_batch(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            key: torch.from_numpy(np.array(value, dtype=np.float32)) / 255.0
            for key, value in observation.items()
        }

    def preprocessor(self, batch: dict[str, Any]) -> dict[str, Any]:
        return dict(batch)


def test_model_inputs_equal_detects_any_changed_pixel() -> None:
    request = synthetic_requests()[0]
    state = {"state": np.zeros((1, 8), dtype=np.float32)}
    runner = FakeRunner()
    same = {key: value.copy() for key, value in request.items()}
    assert bc.model_inputs_equal(runner, request, same, state, "task")
    changed = {key: value.copy() for key, value in request.items()}
    changed["a"][0, 0, 0, 0] ^= 1
    assert not bc.model_inputs_equal(runner, request, changed, state, "task")


def result(label: str, mean_bytes: float, codec_ms: float, lossless: bool = True) -> bc.CodecResult:
    return bc.CodecResult(
        label=label,
        codec=label.split("_")[0],
        level=None,
        requests=10,
        identical_frames=10 if lossless else 9,
        identical_model_inputs=10 if lossless else 9,
        mean_bytes=mean_bytes,
        p95_bytes=mean_bytes,
        mbit_per_robot=mean_bytes * 8 / 1.25 / 1e6,
        encode_ms=codec_ms / 2,
        decode_ms=codec_ms / 2,
        codec_ms=codec_ms,
    )


def test_selection_rule_takes_the_smallest_fast_lossless_codec() -> None:
    results = [
        result("raw", 777_600, 0.5),
        result("png_6", 300_000, 30.0),  # too slow
        result("webp_4", 250_000, 24.0),
        result("webp_0", 254_000, 9.0),  # within 2 % of webp_4: the faster one wins
        result("lossy_1", 100_000, 2.0, lossless=False),
    ]
    assert bc.select_standard(results) == "webp_0"
    assert bc.select_standard(results[:3]) == "webp_4"
    with pytest.raises(ConfigError):
        bc.select_standard([result("png_6", 300_000, 30.0)])
