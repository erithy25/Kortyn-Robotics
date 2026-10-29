"""SmolVLA service time on an NVIDIA GPU, single and batched: the measurement of the GPU notebook.

Kortyn phase 5 (``notebooks/gpu_benchmark.ipynb``), pre-registered in ``reports/phase5_preregistration.md``
(sections 4.1-4.3). Standalone: Python's standard library, NumPy, Pillow, PyTorch and LeRobot (versions
pinned by the notebook). It does not import kortyn, because the notebook runs it in its own Python 3.12
environment on Google Colab or Kaggle; ``kortyn.notebook_build`` embeds this file into the notebook verbatim.

**What is timed** (as ``kortyn/bench_batch.py``): one Brain service over a batch - LeRobot's preprocessing
(normalization, tokenization), ``predict_action_chunk`` and postprocessing, until the actions are back on the
CPU. Inputs are eight real LIBERO-Long observations (two 360x360 cameras, 8-D state, the task's instruction),
taken in turn to fill a batch, with fixed initial noise.

**Configurations** (in this order): ``standard`` (as LeRobot loads the checkpoint: bf16 VLM and action expert,
fp32 projections), ``fp16`` (every bf16 tensor as fp16), ``fp32`` (everything fp32, TF32 off), then
``compiled``: the fastest numerically valid eager configuration with ``torch.compile`` on ``sample_actions``,
which is what LeRobot's own ``compile_model`` switch does (mode ``default`` instead of LeRobot's
``max-autotune``, for compile time).

**Numerical validity** (before the timing): a batch of the eight observations against the fp32 reference with
the same noise. Valid if every action is finite and the RMS deviation is at most 25 % of the policy's own
sampling spread (RMS distance of two fp32 runs with different noise).

**Timing:** batch sizes 1, 2, 4, 8, 16, 32; 10 warm-up runs, then 100 timed runs. Out of memory, or a
projected duration of the 100 runs above 5 minutes, ends the scan of a configuration (larger sizes are
marked). The JSON (format ``kortyn-gpu-benchmark-v2``) is rewritten after every configuration, so a dropped
session keeps what was measured.
"""

from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import importlib
import io
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FORMAT = "kortyn-gpu-benchmark-v2"
SCRIPT_VERSION = "2026-10-27"
CHECKPOINT = "lerobot/smolvla_libero"
REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
NUM_STEPS = 10  # flow-matching steps (checkpoint default, every Kortyn measurement)
CAMERAS = ("observation.images.camera1", "observation.images.camera2")
STATE_KEY = "observation.state"
SIZES = (1, 2, 4, 8, 16, 32)
WARMUP = 10
REPEATS = 100
TIME_LIMIT_S = 300.0  # projected duration of the timed runs of one batch size
NUMERIC_THRESHOLD = 0.25  # RMS deviation from fp32 / sampling spread
NUMERIC_BATCH = 8
NOISE_SEED = 1
SPREAD_SEED = 2
EAGER = ("standard", "fp16", "fp32")
COMPILED = "compiled"
COMPILE_MODE = "default"
MEASURES = (
    "LeRobot preprocessing + predict_action_chunk + postprocessing, actions copied to the CPU (synchronized)"
)

FloatArray = npt.NDArray[np.float64]


class BenchmarkError(RuntimeError):
    """A failure the notebook reports in plain words (no GPU, unreadable inputs, invalid settings)."""


# --------------------------------------------------------------------------------------------- statistics
def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (NumPy's default), ``q`` in [0, 100]."""
    if not values:
        raise BenchmarkError("no values")
    if not 0.0 <= q <= 100.0:
        raise BenchmarkError(f"percentile must be within [0, 100], got {q}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(batch_size: int, timings_ms: Sequence[float]) -> dict[str, Any]:
    """Service-time statistics of one batch size, with every single measurement."""
    if not timings_ms:
        raise BenchmarkError(f"no timings for batch size {batch_size}")
    median = statistics.median(timings_ms)
    return {
        "batch_size": batch_size,
        "status": "ok",
        "repeats": len(timings_ms),
        "median_ms": round(median, 3),
        "p90_ms": round(percentile(timings_ms, 90.0), 3),
        "p95_ms": round(percentile(timings_ms, 95.0), 3),
        "p99_ms": round(percentile(timings_ms, 99.0), 3),
        "mean_ms": round(statistics.fmean(timings_ms), 3),
        "std_ms": round(statistics.pstdev(timings_ms), 3),
        "min_ms": round(min(timings_ms), 3),
        "max_ms": round(max(timings_ms), 3),
        "per_request_ms": round(median / batch_size, 3),
        "samples_ms": [round(value, 3) for value in timings_ms],
    }


def projected_seconds(warmup_ms: Sequence[float], repeats: int) -> float:
    """Projected duration of ``repeats`` timed runs from the median of the last three warm-up runs."""
    if not warmup_ms:
        return 0.0
    return statistics.median(warmup_ms[-3:]) * repeats / 1000.0


def is_out_of_memory(error: BaseException) -> bool:
    """CUDA (or the CPU allocator) ran out of memory."""
    return type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()


@dataclass(frozen=True)
class Settings:
    sizes: tuple[int, ...] = SIZES
    repeats: int = REPEATS
    warmup: int = WARMUP
    time_limit_s: float = TIME_LIMIT_S

    def __post_init__(self) -> None:
        if not self.sizes or list(self.sizes) != sorted(set(self.sizes)) or self.sizes[0] < 1:
            raise BenchmarkError(f"batch sizes must be strictly increasing and >= 1, got {self.sizes}")
        if self.repeats < 1 or self.warmup < 0 or self.time_limit_s <= 0:
            raise BenchmarkError("need repeats >= 1, warmup >= 0 and a positive time limit")


def time_sizes(
    infer: Callable[[int], None],
    settings: Settings,
    *,
    clock: Callable[[], float] = time.perf_counter,
    reset_memory: Callable[[], None] = lambda: None,
    peak_memory_mb: Callable[[], tuple[float, float]] = lambda: (0.0, 0.0),
    after_oom: Callable[[], None] = lambda: None,
    report: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Time ``infer(b)`` for every batch size; ``infer`` must return only when the device is done.

    Out of memory marks the size ``oom``; a projected duration above the time limit marks it ``skipped``. Both
    end the scan: every larger size gets the same status.
    """
    rows: list[dict[str, Any]] = []
    stop: dict[str, Any] | None = None
    for size in settings.sizes:
        if stop is not None:
            rows.append({"batch_size": size, **stop})
            continue
        warm: list[float] = []
        try:
            reset_memory()
            for _ in range(settings.warmup):
                start = clock()
                infer(size)
                warm.append((clock() - start) * 1000.0)
            projected = projected_seconds(warm, settings.repeats)
            if projected > settings.time_limit_s:
                stop = {"status": "skipped", "reason": f"time limit (projected {projected:.0f} s)"}
                rows.append({"batch_size": size, **stop, "warmup_ms": [round(v, 3) for v in warm]})
                report(
                    f"    batch {size:>2}: übersprungen, 100 Läufe würden ≈ {projected / 60:.0f} min dauern"
                )
                continue
            timings: list[float] = []
            for _ in range(settings.repeats):
                start = clock()
                infer(size)
                timings.append((clock() - start) * 1000.0)
        except Exception as exc:
            if not is_out_of_memory(exc):
                raise
            after_oom()
            stop = {"status": "oom", "reason": "out of memory"}
            rows.append({"batch_size": size, **stop})
            report(f"    batch {size:>2}: kein GPU-Speicher mehr (out of memory), größere Batches entfallen")
            continue
        allocated, reserved = peak_memory_mb()
        row = summarize(size, timings)
        row.update(
            {
                "warmup": settings.warmup,
                "warmup_ms": [round(v, 3) for v in warm],
                "memory_allocated_peak_mb": round(allocated, 1),
                "memory_reserved_peak_mb": round(reserved, 1),
            }
        )
        rows.append(row)
        report(
            f"    batch {size:>2}: Median {row['median_ms']:8.1f} ms   p99 {row['p99_ms']:8.1f} ms   "
            f"je Anfrage {row['per_request_ms']:7.1f} ms   Speicher {allocated:7.0f} MB"
        )
    return rows


# --------------------------------------------------------------------------------------------- numerics
def rms(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape:
        raise BenchmarkError(f"shape mismatch {first.shape} vs {second.shape}")
    return float(np.sqrt(np.mean((first - second) ** 2)))


def numeric_check(
    actions: npt.ArrayLike, reference: npt.ArrayLike, spread: float, threshold: float = NUMERIC_THRESHOLD
) -> dict[str, Any]:
    """Valid if every action is finite and ``rms(actions, reference) <= threshold * spread``."""
    values = np.asarray(actions, dtype=np.float64)
    finite = bool(np.all(np.isfinite(values)))
    if not finite:
        return {
            "finite": False,
            "rms_deviation": None,
            "spread": spread,
            "ratio": None,
            "threshold": threshold,
            "valid": False,
        }
    deviation = rms(values, reference)
    ratio = deviation / spread if spread > 0 else float("inf")
    return {
        "finite": True,
        "rms_deviation": deviation,
        "spread": spread,
        "ratio": ratio,
        "threshold": threshold,
        "valid": bool(spread > 0 and ratio <= threshold),
    }


def batch_one_median(configuration: Mapping[str, Any]) -> float | None:
    for row in configuration.get("results", []):
        if row.get("batch_size") == 1 and row.get("status") == "ok":
            return float(row["median_ms"])
    return None


def is_valid(configuration: Mapping[str, Any]) -> bool:
    """Measured without error, numerically valid, and timed at batch size 1."""
    numerics = configuration.get("numerics") or {}
    return (
        configuration.get("status") == "ok"
        and bool(numerics.get("valid"))
        and batch_one_median(configuration) is not None
    )


def fastest_valid(configurations: Sequence[Mapping[str, Any]], names: Sequence[str]) -> str | None:
    """Name of the valid configuration among ``names`` with the smallest median at batch size 1."""
    candidates = [
        (median, str(c["name"]))
        for c in configurations
        if c.get("name") in names and is_valid(c) and (median := batch_one_median(c)) is not None
    ]
    return min(candidates)[1] if candidates else None


# --------------------------------------------------------------------------------------------- output
def slug(text: str) -> str:
    """File-name friendly device name: ``Tesla T4`` -> ``teslat4``."""
    lowered = text.lower().replace("nvidia", "").replace("geforce", "")
    return re.sub(r"[^a-z0-9]+", "", lowered) or "device"


def output_name(gpu: str, platform_name: str, now: dt.datetime) -> str:
    return f"smolvla_gpu_{slug(gpu)}_{slug(platform_name)}_{now:%Y%m%d-%H%M}.json"


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write atomically (a dropped session never leaves half a file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def summary_lines(configurations: Sequence[Mapping[str, Any]]) -> list[str]:
    """A short table for the notebook: per configuration validity and medians at batch 1, 8 and 32."""
    lines = [f"{'Konfiguration':<12} {'gültig':<7} {'b=1 Median':>11} {'b=1 p99':>9} {'b=8':>9} {'b=32':>9}"]
    for configuration in configurations:
        cells: dict[int, str] = {}
        p99 = "-"
        for row in configuration.get("results", []):
            if row.get("status") == "ok":
                cells[int(row["batch_size"])] = f"{row['median_ms']:.0f} ms"
                if row["batch_size"] == 1:
                    p99 = f"{row['p99_ms']:.0f} ms"
            else:
                cells[int(row["batch_size"])] = str(row.get("status", "-"))
        status = (
            "ja" if is_valid(configuration) else ("Fehler" if configuration.get("status") != "ok" else "nein")
        )
        lines.append(
            f"{configuration.get('name', '?'):<12} {status:<7} {cells.get(1, '-'):>11} {p99:>9} "
            f"{cells.get(8, '-'):>9} {cells.get(32, '-'):>9}"
        )
    return lines


# --------------------------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Observations:
    """Real LIBERO-Long observations: per camera ``(N, H, W, 3)`` uint8, states ``(N, 8)``, instructions."""

    cameras: dict[str, npt.NDArray[np.uint8]]
    states: npt.NDArray[np.float32]
    tasks: tuple[str, ...]
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.tasks)
        if n == 0 or set(self.cameras) != set(CAMERAS):
            raise BenchmarkError(f"need observations with the cameras {CAMERAS}")
        for name, frames in self.cameras.items():
            if frames.ndim != 4 or frames.shape[0] != n or frames.shape[3] != 3:
                raise BenchmarkError(f"camera {name}: expected ({n}, H, W, 3), got {frames.shape}")
        if self.states.shape[0] != n or self.states.ndim != 2:
            raise BenchmarkError(f"states: expected ({n}, D), got {self.states.shape}")

    @property
    def count(self) -> int:
        return len(self.tasks)

    def indices(self, size: int) -> list[int]:
        """The observations of a batch of ``size``: taken in turn."""
        return [i % self.count for i in range(size)]


def load_observations(path: Path) -> Observations:
    """Observations written by the notebook: base64 JPEGs per camera, states and instructions."""
    from PIL import Image

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cameras: dict[str, npt.NDArray[np.uint8]] = {}
        for name in CAMERAS:
            frames = [
                np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"), dtype=np.uint8)
                for encoded in data["cameras"][name]
            ]
            cameras[name] = np.stack(frames)
        states = np.asarray(data["states"], dtype=np.float32)
        tasks = tuple(str(task) for task in data["tasks"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise BenchmarkError(f"cannot read the observations {path}: {exc}") from exc
    return Observations(cameras=cameras, states=states, tasks=tasks, source=dict(data.get("source", {})))


# --------------------------------------------------------------------------------------------- device
def nvidia_smi(query: str) -> list[str]:
    """``nvidia-smi --query-gpu=<query>`` lines, or [] without the tool."""
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return []
    try:
        result = subprocess.run(
            [tool, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return (
        [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode == 0
        else []
    )


def gpu_state() -> dict[str, str]:
    """Temperature, SM clock and power of the first GPU (throttling shows up here)."""
    lines = nvidia_smi("temperature.gpu,clocks.sm,clocks.max.sm,power.draw")
    if not lines:
        return {}
    values = [v.strip() for v in lines[0].split(",")]
    keys = ("temperature_c", "sm_clock_mhz", "sm_clock_max_mhz", "power_w")
    return dict(zip(keys, values, strict=False))


class Runner:
    """SmolVLA in one configuration on one device, with LeRobot's pre- and postprocessing."""

    def __init__(self, base: Any, preprocessor: Any, postprocessor: Any, name: str, device: str) -> None:
        import torch

        self.torch = torch
        self.name = name
        self.device = device
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        policy = copy.deepcopy(base)
        precision = "fp32" if name == "fp32" else ("fp16" if name == "fp16" else "standard")
        if precision == "fp32":
            policy = policy.to(torch.float32)
        elif precision == "fp16":
            for tensor in [*policy.parameters(), *policy.buffers()]:
                if tensor.dtype == torch.bfloat16:
                    tensor.data = tensor.data.to(torch.float16)
        policy.config.device = device
        self.policy: Any = policy.to(device)
        self.policy.eval()
        self.precision = precision
        self.compiled = False
        self.chunk = int(policy.config.chunk_size)
        self.max_action_dim = int(policy.config.max_action_dim)
        self._inputs: dict[tuple[int, int], tuple[dict[str, Any], Any]] = {}
        self._observations: Observations | None = None

    def compile(self) -> None:
        """What LeRobot's ``compile_model`` switch does (in the policy's constructor), mode ``default``."""
        torch = self.torch
        torch.set_float32_matmul_precision("high")
        self.policy.model.sample_actions = torch.compile(self.policy.model.sample_actions, mode=COMPILE_MODE)
        self.compiled = True

    def dtypes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for parameter in self.policy.parameters():
            key = str(parameter.dtype).replace("torch.", "")
            counts[key] = counts.get(key, 0) + int(parameter.numel())
        return counts

    def use(self, observations: Observations) -> None:
        self._observations = observations
        self._inputs.clear()

    def _batch(self, size: int, seed: int) -> tuple[dict[str, Any], Any]:
        torch = self.torch
        if self._observations is None:
            raise BenchmarkError("no observations loaded")
        key = (size, seed)
        if key not in self._inputs:
            obs = self._observations
            picks = obs.indices(size)
            batch: dict[str, Any] = {
                name: torch.from_numpy(np.ascontiguousarray(frames[picks]))
                .permute(0, 3, 1, 2)
                .to(torch.float32)
                / 255.0
                for name, frames in obs.cameras.items()
            }
            batch[STATE_KEY] = torch.from_numpy(np.ascontiguousarray(obs.states[picks]))
            batch["task"] = [obs.tasks[i] for i in picks]
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn((size, self.chunk, self.max_action_dim), generator=generator)
            self._inputs[key] = (batch, noise)
        return self._inputs[key]

    def actions(self, size: int, seed: int = NOISE_SEED) -> npt.NDArray[np.float32]:
        """One Brain service over ``size`` requests: actions ``(size, steps, action_dim)`` on the CPU."""
        torch = self.torch
        batch, noise = self._batch(size, seed)
        self.policy.reset()
        with torch.inference_mode():
            processed = self.preprocessor(dict(batch))
            chunk = self.policy.predict_action_chunk(processed, noise=noise.to(self.device))
            if chunk.ndim != 3 or chunk.shape[0] != size:
                raise BenchmarkError(f"expected {size} action chunks, got shape {tuple(chunk.shape)}")
            steps = int(chunk.shape[1])
            actions = self.postprocessor(chunk.reshape(size * steps, -1))
            result: npt.NDArray[np.float32] = (
                actions.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(size, steps, -1)
            )
        return result

    def infer(self, size: int) -> None:
        self.actions(size, NOISE_SEED)

    def reset_memory(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats()

    def peak_memory_mb(self) -> tuple[float, float]:
        if not self.device.startswith("cuda"):
            return 0.0, 0.0
        cuda = self.torch.cuda
        return cuda.max_memory_allocated() / 2**20, cuda.max_memory_reserved() / 2**20

    def free(self) -> None:
        self._inputs.clear()
        self.policy = None
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


def load_base(device: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    """The checkpoint as LeRobot loads it (on the CPU), with pre- and postprocessors for ``device``."""
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    try:
        hf_logging: Any = importlib.import_module("transformers.utils.logging")
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except ImportError:
        pass
    path = snapshot_download(repo_id=CHECKPOINT, revision=REVISION)
    config: Any = PreTrainedConfig.from_pretrained(path)
    config.num_steps = NUM_STEPS
    config.device = "cpu"
    policy: Any = get_policy_class(config.type).from_pretrained(path, config=config, strict=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    model = {
        "checkpoint": CHECKPOINT,
        "revision": REVISION,
        "num_steps": NUM_STEPS,
        "chunk_size": int(config.chunk_size),
        "cameras": list(CAMERAS),
        "state_key": STATE_KEY,
    }
    return policy, preprocessor, postprocessor, model


def describe_device(device: str) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "device": device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }
    for package in ("lerobot", "transformers", "numpy"):
        try:
            module = __import__(package)
            info[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            info[package] = "missing"
    if device.startswith("cuda"):
        cudnn: Any = torch.backends.cudnn
        index = torch.device(device).index or 0
        properties = torch.cuda.get_device_properties(index)
        info.update(
            {
                "gpu": properties.name,
                "capability": f"{properties.major}.{properties.minor}",
                "gpu_memory_gb": round(properties.total_memory / 2**30, 2),
                "multiprocessors": int(properties.multi_processor_count),
                "cudnn": str(cudnn.version()),
                "bf16_native": bool(torch.cuda.is_bf16_supported(including_emulation=False)),
                "gpus_visible": int(torch.cuda.device_count()),
            }
        )
        driver = nvidia_smi("driver_version")
        info["driver"] = driver[0] if driver else "unknown"
    else:
        info["gpu"] = f"CPU ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"
    return info


# --------------------------------------------------------------------------------------------- run
@dataclass
class Session:
    """Everything measured so far; written after every configuration."""

    out: Path
    meta: dict[str, Any]
    configurations: list[dict[str, Any]] = field(default_factory=list)

    def document(self) -> dict[str, Any]:
        selected = fastest_valid(self.configurations, [*EAGER, COMPILED])
        return {"format": FORMAT, **self.meta, "selected": selected, "configurations": self.configurations}

    def save(self) -> None:
        write_json(self.out, self.document())


def measure_configuration(
    runner: Runner,
    observations: Observations,
    reference: npt.NDArray[np.float32] | None,
    spread: float,
    settings: Settings,
) -> dict[str, Any]:
    """Numerical check, then the timing scan of one configuration."""
    import torch

    runner.use(observations)
    entry: dict[str, Any] = {
        "name": runner.name,
        "precision": runner.precision,
        "compiled": runner.compiled,
        "compile_mode": COMPILE_MODE if runner.compiled else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "parameter_dtypes": runner.dtypes(),
        "gpu_state_before": gpu_state(),
    }
    actions = runner.actions(min(NUMERIC_BATCH, observations.count), NOISE_SEED)
    entry["numerics"] = (
        numeric_check(actions, reference, spread)
        if reference is not None
        else {"valid": False, "reason": "no fp32 reference"}
    )
    verdict = "gültig" if entry["numerics"].get("valid") else "UNGÜLTIG"
    ratio = entry["numerics"].get("ratio")
    print(
        f"  Numerik gegen fp32: {verdict}"
        + (f" (Abweichung {100 * ratio:.1f} % der Streuung)" if ratio else "")
    )
    entry["results"] = time_sizes(
        runner.infer,
        settings,
        reset_memory=runner.reset_memory,
        peak_memory_mb=runner.peak_memory_mb,
        after_oom=lambda: torch.cuda.empty_cache() if torch.cuda.is_available() else None,
        report=lambda line: print(line, flush=True),
    )
    entry["gpu_state_after"] = gpu_state()
    entry["status"] = "ok"
    return entry


def run(args: argparse.Namespace) -> Path:
    import torch

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise BenchmarkError(
            "PyTorch sieht keine NVIDIA-GPU. Bitte die GPU aktivieren (Anleitung im Notebook ganz oben) "
            "und das Notebook neu starten. / PyTorch sees no NVIDIA GPU: enable a GPU runtime and run all "
            "cells again."
        )
    if not device.startswith("cuda") and not args.allow_cpu:
        raise BenchmarkError("this benchmark needs a CUDA device (use --allow-cpu only for tests)")
    settings = Settings(
        sizes=tuple(args.sizes), repeats=args.repeats, warmup=args.warmup, time_limit_s=args.time_limit
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    started = time.perf_counter()
    observations = load_observations(Path(args.observations))
    print(f"Lade {CHECKPOINT} (Revision {REVISION[:8]}) ...", flush=True)
    base, preprocessor, postprocessor, model = load_base(device)
    info = describe_device(device)
    print(f"Gerät: {info['gpu']} (torch {info['torch']}, CUDA {info['cuda']})", flush=True)
    now = dt.datetime.now(dt.UTC)
    out = Path(args.out_dir) / output_name(str(info["gpu"]), args.platform, now)
    model.update(
        {
            "inputs": (
                f"{observations.count} LIBERO-Long observations (JPEG q90), taken in turn; "
                f"instructions: {sorted(set(observations.tasks))}"
            ),
            "observation_source": observations.source,
        }
    )
    session = Session(
        out=out,
        meta={
            "created_utc": now.isoformat(timespec="seconds"),
            "script_version": SCRIPT_VERSION,
            "platform": args.platform,
            "device": info,
            "model": model,
            "settings": {
                "sizes": list(settings.sizes),
                "repeats": settings.repeats,
                "warmup": settings.warmup,
                "time_limit_s": settings.time_limit_s,
                "numeric_threshold": NUMERIC_THRESHOLD,
                "numeric_batch": min(NUMERIC_BATCH, observations.count),
                "noise_seed": NOISE_SEED,
                "spread_seed": SPREAD_SEED,
            },
            "measures": MEASURES,
        },
    )
    print("fp32-Referenz für die numerische Prüfung ...", flush=True)
    reference: npt.NDArray[np.float32] | None = None
    spread = 0.0
    try:
        ref_runner = Runner(base, preprocessor, postprocessor, "fp32", device)
        ref_runner.use(observations)
        batch = min(NUMERIC_BATCH, observations.count)
        reference = ref_runner.actions(batch, NOISE_SEED)
        spread = rms(reference, ref_runner.actions(batch, SPREAD_SEED))
        ref_runner.free()
        del ref_runner
        print(f"  Streuung der Policy (fp32, zwei Rauschwerte): {spread:.5f}", flush=True)
    except Exception as exc:  # reported in the file; every configuration is then numerically invalid
        print(f"  fp32-Referenz fehlgeschlagen: {type(exc).__name__}: {exc}", flush=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    session.meta["numeric_reference"] = {
        "configuration": "fp32",
        "spread": spread,
        "available": reference is not None,
    }
    for name in [n for n in args.configs if n in EAGER]:
        session.configurations.append(
            _measure_named(
                name, base, preprocessor, postprocessor, device, observations, reference, spread, settings
            )
        )
        session.save()
    if COMPILED in args.configs:
        target = fastest_valid(session.configurations, EAGER)
        if target is None:
            entry: dict[str, Any] = {
                "name": COMPILED,
                "status": "error",
                "error": "no numerically valid eager configuration to compile",
                "results": [],
            }
            print("Kompiliert: entfällt (keine gültige eager-Konfiguration)", flush=True)
        else:
            entry = _measure_named(
                target,
                base,
                preprocessor,
                postprocessor,
                device,
                observations,
                reference,
                spread,
                settings,
                True,
            )
        session.configurations.append(entry)
        session.save()
    session.meta["duration_s"] = round(time.perf_counter() - started, 1)
    session.save()
    print("\n" + "\n".join(summary_lines(session.configurations)), flush=True)
    selected = session.document()["selected"]
    print(f"\nSchnellste gültige Konfiguration bei Batch 1: {selected}", flush=True)
    print(f"Fertig. Datei: {out.resolve()}", flush=True)
    return out


def _measure_named(
    name: str,
    base: Any,
    preprocessor: Any,
    postprocessor: Any,
    device: str,
    observations: Observations,
    reference: npt.NDArray[np.float32] | None,
    spread: float,
    settings: Settings,
    compiled: bool = False,
) -> dict[str, Any]:
    import torch

    label = f"{COMPILED} ({name})" if compiled else name
    print(f"\nKonfiguration {label} ...", flush=True)
    runner: Runner | None = None
    try:
        runner = Runner(base, preprocessor, postprocessor, name, device)
        if compiled:
            runner.compile()
        entry = measure_configuration(runner, observations, reference, spread, settings)
    except Exception as exc:  # one configuration failing must not end the notebook
        message = f"{type(exc).__name__}: {exc}"
        print(f"  Fehler, Konfiguration übersprungen: {message[:500]}", flush=True)
        entry = {"name": name, "status": "error", "error": message[:2000], "results": []}
    finally:
        if runner is not None:
            runner.free()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if compiled:
        entry["base"] = name
        entry["name"] = COMPILED
    return entry


def parse_list(text: str, cast: Callable[[str], Any]) -> list[Any]:
    try:
        return [cast(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise BenchmarkError(f"invalid list {text!r}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SmolVLA service time per batch size on a GPU (Kortyn phase 5)."
    )
    parser.add_argument("--observations", required=True, help="observations JSON written by the notebook")
    parser.add_argument("--out-dir", default=".", help="directory of the result JSON")
    parser.add_argument("--platform", default="other", help="colab, kaggle or other (file name and metadata)")
    parser.add_argument(
        "--device", default="cuda", help="cuda (default) or cuda:1; cpu only with --allow-cpu"
    )
    parser.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_S, help="seconds per batch size")
    parser.add_argument("--configs", default=",".join([*EAGER, COMPILED]))
    parser.add_argument("--allow-cpu", action="store_true", help="tests only: run on the CPU")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        args.sizes = parse_list(args.sizes, int)
        args.configs = parse_list(args.configs, str)
        unknown = [name for name in args.configs if name not in (*EAGER, COMPILED)]
        if unknown:
            raise BenchmarkError(f"unknown configurations {unknown}")
        run(args)
    except BenchmarkError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("Abgebrochen.", file=sys.stderr, flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
