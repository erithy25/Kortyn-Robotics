# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "lerobot[smolvla]==0.6.1",
# ]
# ///
"""Benchmark: SmolVLA service time per batch size on one GPU (phase 3, step 3).

Measures how long the Brain needs for one inference over a batch of ``b`` requests, for every ``b`` from
1 to 32, with the inputs of the phase-3 robots: ``lerobot/smolvla_libero`` (pinned revision), two camera
images of 360 x 360 pixels, an 8-D state, a LIBERO-Long instruction, 10 flow-matching steps, chunk 50.
One timed inference covers what the Brain does per batch: preprocessing (normalization, resize, tokens),
the policy, postprocessing, and the copy of the actions back to the CPU.

The result is a JSON table that the fleet model reads directly (``format: kortyn-batch-service-v1``,
``kortyn.fleet_brain.TableBatchProfile``): per batch size the median and the 90th percentile of the
service time. Batch sizes that run out of GPU memory are marked ``oom``; larger ones are skipped.

With ``--rtc`` (phase 4) every request of a batch carries the rest of a previous plan and the policy runs with
LeRobot's real-time chunking (EXP schedule, guidance weight 10, 25 overlapping actions, the first 5 of them
executed before the new plan arrives), as the phase-4 Brain does; the table then holds the service time
with RTC.

The script is self-contained (it imports nothing from Kortyn), so a single downloaded file runs with

    uv run bench_batch.py                       # all batch sizes 1..32 on the GPU
    uv run bench_batch.py --sizes 1-4 --repeats 5   # a quick check
    uv run bench_batch.py --rtc                 # the same with real-time chunking

``uv`` installs Python 3.12, PyTorch and LeRobot on first use. Step-by-step guide for Windows with WSL2:
``docs/benchmark_rtx3060ti.md``. Nothing is sent anywhere except the anonymous model download from the
Hugging Face Hub (telemetry off).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TABLE_FORMAT = "kortyn-batch-service-v1"
CHECKPOINT = "lerobot/smolvla_libero"
REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
CAMERAS = ("observation.images.camera1", "observation.images.camera2")
STATE_KEY = "observation.state"
STATE_DIM = 8
TASK = "put both the alphabet soup and the tomato sauce in the basket"  # LIBERO-Long task 0
MAX_BATCH = 32
RTC_HORIZON = 25  # current action + 24 queued actions: the overlap of the phase-4 robots
RTC_DELAY = 5  # actions of the old plan executed before the new one arrives (250 ms)


class BenchmarkError(RuntimeError):
    """The benchmark cannot run (no GPU, model not loadable, invalid arguments)."""


# --------------------------------------------------------------------------------------------- pure parts
def parse_sizes(text: str) -> list[int]:
    """Batch sizes from ``"1-32"``, ``"1,2,4,8"`` or mixtures like ``"1-4,8,16"`` (sorted, unique)."""
    sizes: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            if low > high:
                raise BenchmarkError(f"invalid range {part!r}")
            sizes.update(range(low, high + 1))
        elif part.isdigit():
            sizes.add(int(part))
        else:
            raise BenchmarkError(f"invalid batch size {part!r}")
    if not sizes or min(sizes) < 1 or max(sizes) > MAX_BATCH:
        raise BenchmarkError(f"batch sizes must be within 1..{MAX_BATCH}, got {text!r}")
    return sorted(sizes)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (as numpy's default), ``q`` in [0, 100]."""
    if not values:
        raise BenchmarkError("no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(batch_size: int, timings_ms: Sequence[float]) -> dict[str, Any]:
    """One table row: service-time statistics of a batch size."""
    median = statistics.median(timings_ms)
    return {
        "batch_size": batch_size,
        "status": "ok",
        "repeats": len(timings_ms),
        "median_ms": round(median, 3),
        "p90_ms": round(percentile(timings_ms, 90.0), 3),
        "mean_ms": round(statistics.fmean(timings_ms), 3),
        "std_ms": round(statistics.pstdev(timings_ms), 3),
        "min_ms": round(min(timings_ms), 3),
        "max_ms": round(max(timings_ms), 3),
        "per_request_ms": round(median / batch_size, 3),
    }


def is_out_of_memory(error: BaseException) -> bool:
    """CUDA (or CPU allocator) ran out of memory."""
    name = type(error).__name__
    return name == "OutOfMemoryError" or "out of memory" in str(error).lower()


@dataclass
class Settings:
    sizes: list[int]
    repeats: int
    warmup: int


def run_benchmark(
    infer: Callable[[int], None],
    settings: Settings,
    clock: Callable[[], float] = time.perf_counter,
    after_oom: Callable[[], None] = lambda: None,
    report: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Time ``infer(b)`` for every batch size; out of memory marks ``b`` and skips larger sizes.

    ``infer`` must return only when the GPU is done (synchronize).
    """
    if settings.repeats < 1 or settings.warmup < 0:
        raise BenchmarkError("repeats must be >= 1 and warmup >= 0")
    rows: list[dict[str, Any]] = []
    exhausted = False
    for size in settings.sizes:
        if exhausted:
            rows.append({"batch_size": size, "status": "skipped"})
            continue
        try:
            for _ in range(settings.warmup):
                infer(size)
            timings: list[float] = []
            for _ in range(settings.repeats):
                start = clock()
                infer(size)
                timings.append((clock() - start) * 1000.0)
        except Exception as exc:
            if not is_out_of_memory(exc):
                raise
            after_oom()
            rows.append({"batch_size": size, "status": "oom"})
            report(f"  batch {size:>2}: kein GPU-Speicher mehr (out of memory) - groessere Batches entfallen")
            exhausted = True
            continue
        row = summarize(size, timings)
        rows.append(row)
        report(
            f"  batch {size:>2}: median {row['median_ms']:8.1f} ms   p90 {row['p90_ms']:8.1f} ms   "
            f"je Anfrage {row['per_request_ms']:7.1f} ms"
        )
    return rows


def table(rows: Sequence[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """The JSON document the fleet model reads (``TableBatchProfile.from_file``)."""
    if not any(row.get("status") == "ok" and row.get("batch_size") == 1 for row in rows):
        raise BenchmarkError("the table needs a successful measurement at batch size 1")
    return {"format": TABLE_FORMAT, **meta, "results": list(rows)}


def slug(text: str) -> str:
    """File-name friendly form of a device name: ``NVIDIA GeForce RTX 3060 Ti`` -> ``rtx3060ti``."""
    lowered = text.lower().replace("nvidia", "").replace("geforce", "")
    return re.sub(r"[^a-z0-9]+", "", lowered) or "device"


# --------------------------------------------------------------------------------------------- SmolVLA
class SmolVLABatchRunner:
    """``lerobot/smolvla_libero`` with LeRobot's own pre- and postprocessing, batched synthetic inputs."""

    def __init__(
        self, device: str, num_steps: int, image_size: int, revision: str = REVISION, rtc: bool = False
    ) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise BenchmarkError(
                "PyTorch sieht keine NVIDIA-GPU. Unter WSL2: Ist der aktuelle NVIDIA-Treiber unter Windows "
                "installiert und zeigt 'nvidia-smi' in Ubuntu die Karte? (Anleitung, Schritt 2 und 3)"
            )
        self.torch = torch
        self.device = device
        self.image_size = image_size
        path = snapshot_download(repo_id=CHECKPOINT, revision=revision)
        config: Any = PreTrainedConfig.from_pretrained(path)
        config.num_steps = num_steps
        config.device = device
        self.rtc = rtc
        if rtc:
            from lerobot.configs.types import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            config.rtc_config = RTCConfig(
                enabled=True,
                prefix_attention_schedule=RTCAttentionSchedule.EXP,
                max_guidance_weight=10.0,
                execution_horizon=RTC_HORIZON,
            )
        self.chunk = int(config.chunk_size)
        self.max_action_dim = int(config.max_action_dim)
        self.action_dim = int(config.output_features["action"].shape[0])
        self.policy: Any = get_policy_class(config.type).from_pretrained(path, config=config, strict=True)
        self.policy.eval()
        if rtc:
            self.policy.requires_grad_(False)  # RTC's correction step must not trace the network
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.dtype = str(next(self.policy.parameters()).dtype)
        self._inputs: dict[int, dict[str, Any]] = {}

    def _batch(self, size: int) -> dict[str, Any]:
        torch = self.torch
        if size not in self._inputs:
            generator = torch.Generator().manual_seed(size)
            batch: dict[str, Any] = {
                key: torch.randint(
                    0,
                    256,
                    (size, 3, self.image_size, self.image_size),
                    generator=generator,
                    dtype=torch.uint8,
                ).to(torch.float32)
                / 255.0
                for key in CAMERAS
            }
            batch[STATE_KEY] = torch.randn((size, STATE_DIM), generator=generator, dtype=torch.float32)
            batch["task"] = [TASK] * size
            noise = torch.randn((size, self.chunk, self.max_action_dim), generator=generator)
            # the rest of a previous plan, already in the model's normalized action space
            leftover = 0.5 * torch.randn((size, RTC_HORIZON, self.action_dim), generator=generator)
            self._inputs[size] = {"batch": batch, "noise": noise, "leftover": leftover}
        return self._inputs[size]

    def infer(self, size: int) -> None:
        """One Brain service over ``size`` requests, until the actions are back on the CPU."""
        torch = self.torch
        inputs = self._batch(size)
        self.policy.reset()
        batch = self.preprocessor(dict(inputs["batch"]))
        kwargs: dict[str, Any] = {"noise": inputs["noise"].to(self.device)}
        if self.rtc:
            kwargs.update(
                inference_delay=RTC_DELAY,
                prev_chunk_left_over=inputs["leftover"].to(self.device),
                execution_horizon=RTC_HORIZON,
            )
        # RTC differentiates through its correction step: no inference mode on that path
        with torch.no_grad() if self.rtc else torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch, **kwargs)
            if chunk.ndim != 3 or chunk.shape[0] != size:
                raise BenchmarkError(f"expected {size} action chunks, got shape {tuple(chunk.shape)}")
            steps = int(chunk.shape[1])
            # as the Brain does per request: unnormalize the (steps, action_dim) plan, copy it to the CPU
            actions = self.postprocessor(chunk.reshape(size * steps, -1))
            actions.detach().to(device="cpu", dtype=torch.float32).numpy()  # synchronizes the device

    def free_memory(self) -> None:
        self._inputs.clear()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def describe(self) -> dict[str, Any]:
        torch = self.torch
        import lerobot

        info: dict[str, Any] = {
            "device": self.device,
            "torch": str(torch.__version__),
            "lerobot": str(getattr(lerobot, "__version__", "unknown")),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "parameter_dtype": self.dtype,
        }
        if self.device.startswith("cuda"):
            properties = torch.cuda.get_device_properties(torch.device(self.device))
            info.update(
                {
                    "gpu": properties.name,
                    "gpu_memory_gb": round(properties.total_memory / 2**30, 2),
                    "cuda": str(torch.version.cuda),
                    "cudnn": str(torch.backends.cudnn.version()),
                }
            )
        else:
            info["gpu"] = (
                f"CPU ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"
            )
        return info


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SmolVLA service time per batch size (Kortyn phase 3).")
    parser.add_argument("--sizes", default=f"1-{MAX_BATCH}", help="batch sizes, e.g. 1-32 or 1,2,4,8")
    parser.add_argument("--repeats", type=int, default=20, help="timed inferences per batch size")
    parser.add_argument("--warmup", type=int, default=3, help="untimed inferences per batch size first")
    parser.add_argument("--device", default="cuda", help="cuda (default), cuda:1, or cpu for a dry run")
    parser.add_argument("--num-steps", type=int, default=10, help="flow-matching steps (phase 3: 10)")
    parser.add_argument("--image-size", type=int, default=360, help="camera image size (LIBERO: 360)")
    parser.add_argument("--revision", default=REVISION, help="model revision (pinned)")
    parser.add_argument("--rtc", action="store_true", help="with real-time chunking (phase 4)")
    parser.add_argument("--out", default=None, help="output JSON (default: smolvla_batch_<gpu>_<date>.json)")
    args = parser.parse_args(argv)
    try:
        settings = Settings(parse_sizes(args.sizes), args.repeats, args.warmup)
        print(f"Lade {CHECKPOINT} (Revision {args.revision[:8]}) auf {args.device} ...", flush=True)
        runner = SmolVLABatchRunner(args.device, args.num_steps, args.image_size, args.revision, args.rtc)
        device = runner.describe()
        print(f"Geraet: {device['gpu']}. Messe Batchgroessen {args.sizes} ({args.repeats} Wiederholungen).")
        started = time.perf_counter()
        rows = run_benchmark(runner.infer, settings, after_oom=runner.free_memory)
        meta = {
            "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "device": device,
            "model": {
                "checkpoint": CHECKPOINT,
                "revision": args.revision,
                "num_steps": args.num_steps,
                "cameras": list(CAMERAS),
                "image_size": args.image_size,
                "state_dim": STATE_DIM,
                "task": TASK,
                "rtc": (
                    {
                        "schedule": "EXP",
                        "max_guidance_weight": 10.0,
                        "horizon": RTC_HORIZON,
                        "delay": RTC_DELAY,
                    }
                    if args.rtc
                    else None
                ),
            },
            "settings": {"repeats": settings.repeats, "warmup": settings.warmup, "sizes": settings.sizes},
            "duration_s": round(time.perf_counter() - started, 1),
            "measures": "preprocessing + predict_action_chunk + postprocessing, synchronized",
        }
        document = table(rows, meta)
    except BenchmarkError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Abgebrochen.", file=sys.stderr)
        return 130
    suffix = "_rtc" if args.rtc else ""
    out = Path(args.out or f"smolvla_batch{suffix}_{slug(str(device['gpu']))}_{dt.date.today():%Y%m%d}.json")
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(f"Fertig. Ergebnis gespeichert in: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
