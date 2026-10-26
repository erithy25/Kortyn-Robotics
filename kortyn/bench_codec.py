"""Phase 4, step 1: lossless codecs for the camera frames - bit identity, size and speed.

Renders camera frames of real LIBERO states: every episode of the experiment is reset and driven by a fixed
random walk of small end-effector moves (drawn from the episode seed), and the frames of both cameras are
taken at fixed ticks. Every capture is one request as the agent sends it (each camera ``(1, H, W, 3)``
encoded on its own). For every candidate codec the script checks that the decoded frames and the Brain's
model input (``preprocessor(_to_batch(...))`` of the policy) equal those of the raw frames bit for bit, and
measures bytes and encode/decode times on one CPU thread. The pre-registered rule then picks the standard
(reports/phase4_preregistration.md, section 3): the smallest mean bytes per request among the candidates
whose median encode + decode time per request is at most 25 ms; within 2 % the faster one wins.

    UV_PROJECT_ENVIRONMENT=.venv-libero uv run --no-group metaworld --group libero \\
        python -m kortyn.bench_codec --config kortyn/experiments/phase3/libero_main.yaml \\
        --out results/phase4/codec.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kortyn.codec import CodecError, decode_tensor, encode_images
from kortyn.config import ConfigError, ImageTransportConfig, load_config
from kortyn.typing_utils import Array

LOG = logging.getLogger("kortyn.bench_codec")

CANDIDATES: tuple[tuple[str, ImageTransportConfig], ...] = (
    ("raw", ImageTransportConfig(codec="raw")),
    ("png_1", ImageTransportConfig(codec="png", level=1)),
    ("png_6", ImageTransportConfig(codec="png", level=6)),
    ("webp_0", ImageTransportConfig(codec="webp", level=0)),
    ("webp_4", ImageTransportConfig(codec="webp", level=4)),
    ("zstd_3", ImageTransportConfig(codec="zstd", level=3)),
    ("zstd_9", ImageTransportConfig(codec="zstd", level=9)),
)
CAPTURE_TICKS = (0, 50, 100, 150, 200)
MAX_CODEC_MS = 25.0  # selection rule: encode + decode per request (2 % of the 1 250-ms budget)
TIE_SHARE = 0.02  # sizes within 2 % count as equal; the faster codec wins
PERIOD_S = 1.25  # one request every 25 ticks at 20 Hz
REPEATS = 3  # timings per request and codec; the median counts


Request = dict[str, Array]  # camera key -> frames (1, H, W, C), uint8


@dataclass
class CodecResult:
    label: str
    codec: str
    level: int | None
    requests: int
    identical_frames: int  # requests whose decoded frames equal the raw frames bit for bit
    identical_model_inputs: int  # requests whose model input equals the one from raw frames
    mean_bytes: float
    p95_bytes: float
    mbit_per_robot: float  # mean bytes * 8 / one request period
    encode_ms: float  # median over requests (both cameras)
    decode_ms: float
    codec_ms: float  # encode + decode, median over requests of the per-request sum

    @property
    def lossless(self) -> bool:
        return self.identical_frames == self.requests and self.identical_model_inputs == self.requests


def random_walk(seed: int, steps: int, action_dim: int) -> Array:
    """Small end-effector moves (relative control in [-1, 1]); the gripper toggles every 50 ticks."""
    rng = np.random.default_rng(seed)
    moves = np.clip(rng.normal(0.0, 0.3, size=(steps, action_dim)), -1.0, 1.0).astype(np.float32)
    gripper = np.where((np.arange(steps) // 50) % 2 == 0, -1.0, 1.0).astype(np.float32)
    moves[:, -1] = gripper
    return moves


def capture_requests(env: Any, seeds: Sequence[int], ticks: Sequence[int]) -> list[Request]:
    """Frames of both cameras at ``ticks`` of a random walk from every episode's initial state."""
    if not ticks or min(ticks) < 0:
        raise ConfigError("capture ticks must be non-negative")
    keys = sorted(env.image_keys)
    requests: list[Request] = []
    last = max(ticks)
    for seed in seeds:
        env.reset(seed)
        moves = random_walk(seed, last, env.action_dim)
        for tick in range(last + 1):
            if tick in ticks:
                requests.append({key: env.render_image(key)[np.newaxis].copy() for key in keys})
            if tick < last:
                result = env.step(moves[tick])
                if result.terminated or result.truncated:
                    env.reset(seed)  # a finished episode starts over; the capture ticks stay the same
    return requests


def _timed(function: Any, *args: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    value = function(*args)
    return value, (time.perf_counter() - started) * 1000.0


def measure(
    label: str, transport: ImageTransportConfig, requests: Sequence[Request]
) -> tuple[CodecResult, list[Request]]:
    """Encode and decode every request; returns the result and the decoded requests."""
    sizes: list[float] = []
    encode_times: list[float] = []
    decode_times: list[float] = []
    codec_times: list[float] = []
    identical = 0
    decoded_requests: list[Request] = []
    for request in requests:
        encoded = {}
        encode_runs: list[float] = []
        decode_runs: list[float] = []
        decoded: Request = {}
        for _ in range(REPEATS):
            total_encode = 0.0
            for key, frames in request.items():
                tensor, ms = _timed(encode_images, frames, transport)
                encoded[key] = tensor
                total_encode += ms
            total_decode = 0.0
            for key, tensor in encoded.items():
                value, ms = _timed(decode_tensor, tensor)
                decoded[key] = np.asarray(value)
                total_decode += ms
            encode_runs.append(total_encode)
            decode_runs.append(total_decode)
        sizes.append(float(sum(tensor.ByteSize() for tensor in encoded.values())))
        encode_times.append(statistics.median(encode_runs))
        decode_times.append(statistics.median(decode_runs))
        codec_times.append(encode_times[-1] + decode_times[-1])
        if all(
            np.array_equal(decoded[key], frames) and decoded[key].dtype == frames.dtype
            for key, frames in request.items()
        ):
            identical += 1
        decoded_requests.append(decoded)
    mean_bytes = float(np.mean(sizes))
    result = CodecResult(
        label=label,
        codec=transport.codec,
        level=transport.level,
        requests=len(requests),
        identical_frames=identical,
        identical_model_inputs=0,
        mean_bytes=mean_bytes,
        p95_bytes=float(np.percentile(sizes, 95)),
        mbit_per_robot=mean_bytes * 8.0 / PERIOD_S / 1e6,
        encode_ms=float(statistics.median(encode_times)),
        decode_ms=float(statistics.median(decode_times)),
        codec_ms=float(statistics.median(codec_times)),
    )
    return result, decoded_requests


def model_inputs_equal(
    runner: Any, raw: Request, decoded: Request, state: Mapping[str, Array], task: str
) -> bool:
    """The Brain's model input (preprocessed batch) from decoded frames equals the one from raw frames."""
    import torch

    def model_input(images: Request) -> dict[str, Any]:
        batch = runner._to_batch({**images, **state})
        if runner.needs_task:
            batch["task"] = [task]
        return dict(runner.preprocessor(batch))

    a, b = model_input(raw), model_input(decoded)
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        other = b[key]
        if isinstance(value, torch.Tensor):
            if (
                not isinstance(other, torch.Tensor)
                or value.dtype != other.dtype
                or not torch.equal(value, other)
            ):
                return False
        elif value != other:
            return False
    return True


def select_standard(results: Sequence[CodecResult]) -> str:
    """Pre-registered rule: smallest mean bytes with encode + decode <= 25 ms; within 2 % the faster one."""
    eligible = [r for r in results if r.lossless and r.codec_ms <= MAX_CODEC_MS]
    if not eligible:
        raise ConfigError("no lossless candidate meets the time limit")
    smallest = min(r.mean_bytes for r in eligible)
    tied = [r for r in eligible if r.mean_bytes <= smallest * (1.0 + TIE_SHARE)]
    return min(tied, key=lambda r: (r.codec_ms, r.mean_bytes)).label


def _state_of(runner: Any) -> dict[str, Array]:
    """A fixed robot state for the model-input check (the images are what differs between codecs)."""
    state: dict[str, Array] = {}
    for feature in runner.features:
        if feature.kind != "VISUAL":
            state[feature.key] = np.zeros((runner.n_obs_steps, *feature.shape), dtype=np.float32)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-4 lossless codec benchmark on LIBERO frames.")
    parser.add_argument("--config", required=True, help="LIBERO experiment (episodes, cameras, policy)")
    parser.add_argument("--profile", default="smolvla_libero")
    parser.add_argument("--out", default="results/phase4/codec.json")
    parser.add_argument("--episodes", type=int, default=None, help="default: all episodes of the experiment")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    import cv2

    cv2.setNumThreads(1)
    from kortyn.run_virtual import _load_runner
    from kortyn.sim_env import make_env

    try:
        config = load_config(args.config)
        episodes = args.episodes or config.experiment.episodes
        seeds = [config.experiment.seed_start + i for i in range(episodes)]
        env = make_env(config.env)
        started = time.perf_counter()
        requests = capture_requests(env, seeds, CAPTURE_TICKS)
        LOG.info(
            "captured %d requests from %d episodes in %.0f s",
            len(requests),
            len(seeds),
            time.perf_counter() - started,
        )
        task = getattr(env, "task_description", "") or "task"
        env.close()
        runner = _load_runner(config, args.profile, 1, "cpu")
        state = _state_of(runner)
        results: list[CodecResult] = []
        for label, transport in CANDIDATES:
            result, decoded = measure(label, transport, requests)
            result.identical_model_inputs = sum(
                model_inputs_equal(runner, raw, dec, state, task)
                for raw, dec in zip(requests, decoded, strict=True)
            )
            results.append(result)
            LOG.info(
                "%s: %.0f bytes/request (%.2f Mbit/s per robot), encode %.1f ms, decode %.1f ms, "
                "identical %d/%d, model input %d/%d",
                label,
                result.mean_bytes,
                result.mbit_per_robot,
                result.encode_ms,
                result.decode_ms,
                result.identical_frames,
                result.requests,
                result.identical_model_inputs,
                result.requests,
            )
        standard = select_standard(results)
    except (ConfigError, CodecError) as exc:
        LOG.error("%s", exc)
        return 2
    document = {
        "experiment": args.config,
        "episodes": len(seeds),
        "capture_ticks": list(CAPTURE_TICKS),
        "requests": len(requests),
        "frame_shape": {key: list(frames.shape) for key, frames in requests[0].items()},
        "period_s": PERIOD_S,
        "max_codec_ms": MAX_CODEC_MS,
        "tie_share": TIE_SHARE,
        "repeats": REPEATS,
        "standard": standard,
        "candidates": [asdict(r) | {"lossless": r.lossless} for r in results],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    LOG.info("standard codec: %s; written %s", standard, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
