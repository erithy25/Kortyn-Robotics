"""GPU service-time measurements (phase 5): read the notebook's file and turn it into a fleet service model.

The GPU notebook (``notebooks/gpu_benchmark.ipynb``, script :mod:`kortyn.notebook_bench`) writes one JSON
file (format ``kortyn-gpu-benchmark-v2``): several configurations of SmolVLA (precision, compiled or not),
each with up to 100 single measurements per batch size and a numerical check against fp32. This module

* reads and validates such a file (:class:`GpuBenchmark`);
* applies the pre-registered validity rules (``reports/phase5_preregistration.md``, sections 4.2 and 5): a
  configuration counts if it ran without error, passed the numerical check and has at least 100 measurements
  at batch size 1; a measurement counts if it ran on a GPU and has such a configuration;
* provides :class:`EmpiricalBatchProfile`, the fleet's service model built from the single measurements: the
  service time of a batch is drawn from the measured distribution of its size; between measured sizes the same
  quantile is interpolated linearly in the batch size. No extrapolation beyond the largest measured size.

:func:`load_service_model` also reads the older table format ``kortyn-batch-service-v1`` of
``kortyn/bench_batch.py`` (median and p90 per batch size, :class:`~kortyn.fleet_brain.TableBatchProfile`).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kortyn.fleet_brain import TABLE_FORMAT, BatchModelError, BatchServiceModel, TableBatchProfile

V2_FORMAT = "kortyn-gpu-benchmark-v2"
MIN_REPEATS = 100  # pre-registered: at least 100 timed runs at batch size 1


class GpuBenchmarkError(BatchModelError):
    """An unreadable or inconsistent GPU measurement file, or a configuration that cannot be used."""


# --------------------------------------------------------------------------------------------- service model
def empirical_quantile(ordered: Sequence[float], u: float) -> float:
    """Quantile ``u`` in [0, 1] of sorted samples, linear between order statistics (NumPy's default)."""
    if not ordered:
        raise GpuBenchmarkError("no samples")
    if not 0.0 <= u <= 1.0:
        raise GpuBenchmarkError(f"quantile must be within [0, 1], got {u}")
    position = (len(ordered) - 1) * u
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class EmpiricalBatchProfile:
    """Service time per batch size from single measurements (milliseconds, sorted per size)."""

    sizes: tuple[int, ...]
    samples_ms: tuple[tuple[float, ...], ...]
    source: str = ""

    def __post_init__(self) -> None:
        if not self.sizes or self.sizes[0] != 1:
            raise GpuBenchmarkError("an empirical service model must start with batch size 1")
        if list(self.sizes) != sorted(set(self.sizes)):
            raise GpuBenchmarkError(f"batch sizes must be strictly increasing, got {self.sizes}")
        if len(self.samples_ms) != len(self.sizes):
            raise GpuBenchmarkError("need one sample list per batch size")
        for size, samples in zip(self.sizes, self.samples_ms, strict=True):
            if not samples:
                raise GpuBenchmarkError(f"batch size {size} has no samples")
            if any(not math.isfinite(v) or v <= 0 for v in samples):
                raise GpuBenchmarkError(f"batch size {size}: samples must be finite and > 0 ms")
            if list(samples) != sorted(samples):
                raise GpuBenchmarkError(f"batch size {size}: samples must be sorted (use from_samples)")

    @classmethod
    def from_samples(cls, samples: Mapping[int, Sequence[float]], source: str = "") -> EmpiricalBatchProfile:
        sizes = tuple(sorted(samples))
        return cls(sizes, tuple(tuple(sorted(float(v) for v in samples[s])) for s in sizes), source)

    @property
    def max_batch(self) -> int:
        return self.sizes[-1]

    def _neighbours(self, batch_size: int) -> tuple[int, int, float]:
        if not 1 <= batch_size <= self.max_batch:
            raise BatchModelError(f"batch size {batch_size} outside [1, {self.max_batch}]")
        for index, size in enumerate(self.sizes):
            if size == batch_size:
                return index, index, 0.0
            if size > batch_size:
                low = self.sizes[index - 1]
                return index - 1, index, (batch_size - low) / (size - low)
        raise BatchModelError(f"batch size {batch_size} beyond the measurement")  # unreachable

    def quantile_ms(self, batch_size: int, u: float) -> float:
        """Quantile ``u`` of the service time at ``batch_size`` (same quantile of both neighbours, linear)."""
        low, high, share = self._neighbours(batch_size)
        q_low = empirical_quantile(self.samples_ms[low], u)
        if share == 0.0:
            return q_low
        q_high = empirical_quantile(self.samples_ms[high], u)
        return q_low + share * (q_high - q_low)

    def mean_s(self, batch_size: int) -> float:
        low, high, share = self._neighbours(batch_size)
        m_low = math.fsum(self.samples_ms[low]) / len(self.samples_ms[low])
        m_high = math.fsum(self.samples_ms[high]) / len(self.samples_ms[high])
        return (m_low + share * (m_high - m_low)) / 1000.0

    def estimate_s(self, batch_size: int) -> float:
        """Planning value of schedulers and admission control: the 90th percentile (as the table model)."""
        return self.quantile_ms(batch_size, 0.9) / 1000.0

    def sample_s(self, batch_size: int, rng: random.Random) -> float:
        return self.quantile_ms(batch_size, rng.random()) / 1000.0


# --------------------------------------------------------------------------------------------- file
@dataclass(frozen=True)
class SizeMeasurement:
    batch_size: int
    status: str  # ok | oom | skipped
    samples_ms: tuple[float, ...] = ()
    median_ms: float | None = None
    p90_ms: float | None = None
    p99_ms: float | None = None
    memory_allocated_peak_mb: float | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.samples_ms)


@dataclass(frozen=True)
class Configuration:
    """One measured configuration of the policy (precision, compiled or not)."""

    name: str
    precision: str
    compiled: bool
    status: str  # ok | error
    error: str
    numerics_valid: bool
    numerics_ratio: float | None
    sizes: tuple[SizeMeasurement, ...]
    base: str | None = None  # compiled: the eager configuration it was compiled from

    @property
    def measured(self) -> tuple[SizeMeasurement, ...]:
        """The successfully measured sizes from batch size 1 on, up to the first failed size."""
        rows: list[SizeMeasurement] = []
        for row in sorted(self.sizes, key=lambda r: r.batch_size):
            if not row.ok:
                break
            rows.append(row)
        return tuple(rows) if rows and rows[0].batch_size == 1 else ()

    def repeats_at_one(self) -> int:
        measured = self.measured
        return len(measured[0].samples_ms) if measured else 0

    def valid(self, min_repeats: int = MIN_REPEATS) -> bool:
        """Pre-registered: no error, numerically valid, at least ``min_repeats`` measurements at batch 1."""
        return self.status == "ok" and self.numerics_valid and self.repeats_at_one() >= min_repeats

    def median_ms(self, batch_size: int) -> float | None:
        for row in self.measured:
            if row.batch_size == batch_size:
                return row.median_ms
        return None

    def profile(self, source: str = "") -> EmpiricalBatchProfile:
        measured = self.measured
        if not measured:
            raise GpuBenchmarkError(
                f"configuration {self.name!r} has no successful measurement at batch size 1"
            )
        return EmpiricalBatchProfile.from_samples(
            {row.batch_size: row.samples_ms for row in measured}, source=f"{source}#{self.name}"
        )


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _size(row: Mapping[str, Any], where: str) -> SizeMeasurement:
    try:
        status = str(row["status"])
        batch_size = int(row["batch_size"])
        samples = tuple(float(v) for v in row.get("samples_ms", ())) if status == "ok" else ()
        if status == "ok" and not samples:
            raise GpuBenchmarkError(f"{where}: batch size {batch_size} is 'ok' but has no samples_ms")
        return SizeMeasurement(
            batch_size=batch_size,
            status=status,
            samples_ms=samples,
            median_ms=_float_or_none(row.get("median_ms")),
            p90_ms=_float_or_none(row.get("p90_ms")),
            p99_ms=_float_or_none(row.get("p99_ms")),
            memory_allocated_peak_mb=_float_or_none(row.get("memory_allocated_peak_mb")),
            reason=str(row.get("reason", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GpuBenchmarkError(f"{where}: malformed row {dict(row)!r}: {exc}") from exc


def _configuration(entry: Mapping[str, Any], where: str) -> Configuration:
    try:
        name = str(entry["name"])
        numerics = entry.get("numerics") or {}
        ratio = numerics.get("ratio")
        return Configuration(
            name=name,
            precision=str(entry.get("precision", name)),
            compiled=bool(entry.get("compiled", False)),
            status=str(entry.get("status", "error")),
            error=str(entry.get("error", "")),
            numerics_valid=bool(numerics.get("valid", False)),
            numerics_ratio=None if ratio is None else float(ratio),
            sizes=tuple(_size(row, f"{where}/{name}") for row in entry.get("results", [])),
            base=None if entry.get("base") is None else str(entry["base"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GpuBenchmarkError(f"{where}: malformed configuration: {exc}") from exc


@dataclass(frozen=True)
class GpuBenchmark:
    """A GPU measurement file of the notebook."""

    source: str
    gpu: str
    platform: str
    created_utc: str
    device: dict[str, Any]
    configurations: tuple[Configuration, ...]
    selected: str | None = None

    @property
    def is_gpu(self) -> bool:
        return str(self.device.get("device", "")).startswith("cuda")

    @classmethod
    def from_file(cls, path: Path | str) -> GpuBenchmark:
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GpuBenchmarkError(f"cannot read GPU measurement {source}: {exc}") from exc
        if not isinstance(data, dict):
            raise GpuBenchmarkError(f"{source}: top level must be a JSON object")
        return cls.from_dict(data, str(source))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source: str = "") -> GpuBenchmark:
        if data.get("format") != V2_FORMAT:
            raise GpuBenchmarkError(f"{source}: format must be {V2_FORMAT!r}, got {data.get('format')!r}")
        device = data.get("device")
        entries = data.get("configurations")
        if not isinstance(device, dict) or not isinstance(entries, list):
            raise GpuBenchmarkError(f"{source}: needs 'device' (object) and 'configurations' (list)")
        configurations = tuple(_configuration(entry, source) for entry in entries)
        names = [c.name for c in configurations]
        if len(set(names)) != len(names):
            raise GpuBenchmarkError(f"{source}: configuration names must be unique, got {names}")
        selected = data.get("selected")
        return cls(
            source=source,
            gpu=str(device.get("gpu", "unknown")),
            platform=str(data.get("platform", "unknown")),
            created_utc=str(data.get("created_utc", "")),
            device=dict(device),
            configurations=configurations,
            selected=None if selected is None else str(selected),
        )

    def configuration(self, name: str) -> Configuration:
        for configuration in self.configurations:
            if configuration.name == name:
                return configuration
        raise GpuBenchmarkError(
            f"{self.source}: no configuration {name!r} (have {[c.name for c in self.configurations]})"
        )

    def valid_configurations(self, min_repeats: int = MIN_REPEATS) -> list[Configuration]:
        return [c for c in self.configurations if c.valid(min_repeats)]

    def measurement_valid(self, min_repeats: int = MIN_REPEATS) -> bool:
        """Pre-registered: measured on a GPU and at least one valid configuration."""
        return self.is_gpu and bool(self.valid_configurations(min_repeats))

    def fastest_valid(self, min_repeats: int = MIN_REPEATS) -> Configuration:
        """The valid configuration with the smallest median at batch size 1."""
        candidates = [
            (median, c.name, c)
            for c in self.valid_configurations(min_repeats)
            if (median := c.median_ms(1)) is not None
        ]
        if not candidates:
            raise GpuBenchmarkError(f"{self.source}: no valid configuration")
        return min(candidates, key=lambda item: (item[0], item[1]))[2]


def load_service_model(path: Path | str, configuration: str | None = None) -> BatchServiceModel:
    """Service model of a measurement file: v2 (a configuration, default the fastest valid one) or v1."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GpuBenchmarkError(f"cannot read GPU measurement {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise GpuBenchmarkError(f"{source}: top level must be a JSON object")
    if data.get("format") == TABLE_FORMAT:
        if configuration is not None:
            raise GpuBenchmarkError(f"{source}: a {TABLE_FORMAT} table has no configurations")
        return TableBatchProfile.from_dict(data, str(source))
    benchmark = GpuBenchmark.from_dict(data, str(source))
    chosen = benchmark.configuration(configuration) if configuration else benchmark.fastest_valid()
    return chosen.profile(str(source))
