"""Reading the GPU notebook's measurement file and the empirical service model (kortyn.gpu_benchmark)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kortyn.config import decode
from kortyn.fleet import FleetSimConfig, GpuConfig, RunKey, run_fleet, service_model
from kortyn.fleet_brain import TABLE_FORMAT, BatchModelError, TableBatchProfile
from kortyn.gpu_benchmark import (
    MIN_REPEATS,
    V2_FORMAT,
    EmpiricalBatchProfile,
    GpuBenchmark,
    GpuBenchmarkError,
    empirical_quantile,
    load_service_model,
)


def _row(batch: int, base_ms: float, n: int = MIN_REPEATS, status: str = "ok") -> dict[str, Any]:
    if status != "ok":
        return {"batch_size": batch, "status": status, "reason": status}
    samples = [base_ms + 0.01 * i for i in range(n)]
    return {
        "batch_size": batch,
        "status": "ok",
        "repeats": n,
        "median_ms": float(np.median(samples)),
        "p90_ms": float(np.percentile(samples, 90)),
        "p99_ms": float(np.percentile(samples, 99)),
        "samples_ms": samples,
        "memory_allocated_peak_mb": 1000.0 + batch,
    }


def _configuration(
    name: str,
    rows: list[dict[str, Any]],
    valid: bool = True,
    status: str = "ok",
    precision: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "precision": precision or name,
        "compiled": name == "compiled",
        "status": status,
        "numerics": {"valid": valid, "ratio": 0.05 if valid else 0.9, "finite": True},
        "results": rows,
    }


def _document(configurations: list[dict[str, Any]], device: str = "cuda") -> dict[str, Any]:
    return {
        "format": V2_FORMAT,
        "created_utc": "2026-10-28T09:00:00+00:00",
        "platform": "colab",
        "device": {"device": device, "gpu": "Tesla T4" if device == "cuda" else "CPU (x86_64, 4 threads)"},
        "selected": None,
        "configurations": configurations,
    }


def _standard_document() -> dict[str, Any]:
    return _document(
        [
            _configuration("standard", [_row(1, 120.0), _row(2, 150.0), _row(4, 210.0), _row(8, 330.0)]),
            _configuration(
                "fp16", [_row(1, 60.0), _row(2, 70.0), _row(4, 95.0), _row(8, 150.0, status="oom")]
            ),
            _configuration("fp32", [_row(1, 200.0), _row(2, 260.0)], valid=False),
            {"name": "compiled", "status": "error", "error": "triton missing", "results": []},
        ]
    )


# --------------------------------------------------------------------------------------------- quantiles
def test_empirical_quantile_interpolates_order_statistics() -> None:
    ordered = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert empirical_quantile(ordered, 0.0) == 10.0
    assert empirical_quantile(ordered, 1.0) == 50.0
    assert empirical_quantile(ordered, 0.5) == 30.0
    assert empirical_quantile(ordered, 0.9) == pytest.approx(float(np.percentile(ordered, 90)))
    with pytest.raises(GpuBenchmarkError):
        empirical_quantile([], 0.5)
    with pytest.raises(GpuBenchmarkError):
        empirical_quantile(ordered, 1.5)


def test_empirical_profile_at_measured_sizes() -> None:
    profile = EmpiricalBatchProfile.from_samples({1: [100.0, 120.0, 110.0], 4: [200.0, 220.0, 210.0]})
    assert profile.max_batch == 4
    assert profile.samples_ms[0] == (100.0, 110.0, 120.0)  # sorted
    assert profile.mean_s(1) == pytest.approx(0.110)
    assert profile.quantile_ms(1, 0.5) == pytest.approx(110.0)
    assert profile.estimate_s(4) == pytest.approx(float(np.percentile([200, 210, 220], 90)) / 1000.0)


def test_empirical_profile_interpolates_the_same_quantile_between_sizes() -> None:
    profile = EmpiricalBatchProfile.from_samples({1: [100.0, 140.0], 3: [200.0, 300.0]})
    # batch 2 lies halfway: same quantile of both neighbours, averaged
    assert profile.quantile_ms(2, 0.0) == pytest.approx(150.0)
    assert profile.quantile_ms(2, 1.0) == pytest.approx(220.0)
    assert profile.mean_s(2) == pytest.approx((120.0 + 250.0) / 2 / 1000.0)


def test_empirical_profile_samples_stay_within_the_measured_range() -> None:
    profile = EmpiricalBatchProfile.from_samples({1: [90.0, 100.0, 130.0], 2: [150.0, 160.0, 190.0]})
    rng = random.Random(1)
    draws = [profile.sample_s(1, rng) * 1000.0 for _ in range(2000)]
    assert min(draws) >= 90.0
    assert max(draws) <= 130.0
    assert np.median(draws) == pytest.approx(100.0, abs=3.0)


def test_empirical_profile_rejects_invalid_input() -> None:
    with pytest.raises(GpuBenchmarkError):
        EmpiricalBatchProfile.from_samples({2: [10.0]})  # must start at batch 1
    with pytest.raises(GpuBenchmarkError):
        EmpiricalBatchProfile.from_samples({1: []})
    with pytest.raises(GpuBenchmarkError):
        EmpiricalBatchProfile.from_samples({1: [10.0, -1.0]})
    with pytest.raises(GpuBenchmarkError):
        EmpiricalBatchProfile(sizes=(1,), samples_ms=((2.0, 1.0),))  # unsorted
    profile = EmpiricalBatchProfile.from_samples({1: [10.0], 2: [12.0]})
    with pytest.raises(BatchModelError):
        profile.mean_s(3)  # no extrapolation
    with pytest.raises(BatchModelError):
        profile.mean_s(0)


# --------------------------------------------------------------------------------------------- file
def test_reads_configurations_and_validity() -> None:
    benchmark = GpuBenchmark.from_dict(_standard_document(), "test.json")
    assert benchmark.gpu == "Tesla T4"
    assert benchmark.platform == "colab"
    assert benchmark.is_gpu
    names = [c.name for c in benchmark.valid_configurations()]
    assert names == ["standard", "fp16"]
    assert benchmark.measurement_valid()
    fp16 = benchmark.configuration("fp16")
    assert [row.batch_size for row in fp16.measured] == [1, 2, 4]  # oom at 8 ends the measured range
    assert fp16.median_ms(1) == pytest.approx(60.0 + 0.01 * 49.5)
    assert benchmark.fastest_valid().name == "fp16"
    assert not benchmark.configuration("compiled").valid()
    assert not benchmark.configuration("fp32").valid()  # numerically invalid


def test_too_few_repeats_make_a_configuration_invalid() -> None:
    document = _document([_configuration("standard", [_row(1, 100.0, n=MIN_REPEATS - 1)])])
    benchmark = GpuBenchmark.from_dict(document)
    assert not benchmark.measurement_valid()
    with pytest.raises(GpuBenchmarkError):
        benchmark.fastest_valid()


def test_cpu_measurement_is_not_a_valid_gpu_measurement() -> None:
    document = _document([_configuration("standard", [_row(1, 100.0)])], device="cpu")
    benchmark = GpuBenchmark.from_dict(document)
    assert benchmark.valid_configurations()
    assert not benchmark.measurement_valid()


def test_a_configuration_without_batch_one_has_no_profile() -> None:
    document = _document([_configuration("standard", [_row(1, 100.0, status="oom"), _row(2, 150.0)])])
    configuration = GpuBenchmark.from_dict(document).configuration("standard")
    assert configuration.measured == ()
    with pytest.raises(GpuBenchmarkError):
        configuration.profile()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(format="kortyn-batch-service-v1"),
        lambda d: d.pop("device"),
        lambda d: d.update(configurations="x"),
        lambda d: d["configurations"].append(dict(d["configurations"][0])),  # duplicate name
        lambda d: d["configurations"][0]["results"][0].pop("samples_ms"),
        lambda d: d["configurations"][0]["results"][0].update(batch_size="one"),
    ],
)
def test_malformed_files_are_rejected(mutate: Any) -> None:
    document = _standard_document()
    mutate(document)
    with pytest.raises(GpuBenchmarkError):
        GpuBenchmark.from_dict(document, "bad.json")


def test_unknown_configuration_is_reported() -> None:
    with pytest.raises(GpuBenchmarkError, match="no configuration"):
        GpuBenchmark.from_dict(_standard_document()).configuration("bf16")


def test_from_file_reports_unreadable_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(GpuBenchmarkError):
        GpuBenchmark.from_file(broken)
    with pytest.raises(GpuBenchmarkError):
        GpuBenchmark.from_file(tmp_path / "missing.json")
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(GpuBenchmarkError):
        GpuBenchmark.from_file(array)


# --------------------------------------------------------------------------------------------- service model
def test_load_service_model_picks_the_fastest_valid_or_a_named_configuration(tmp_path: Path) -> None:
    path = tmp_path / "gpu.json"
    path.write_text(json.dumps(_standard_document()), encoding="utf-8")
    fastest = load_service_model(path)
    assert isinstance(fastest, EmpiricalBatchProfile)
    assert fastest.max_batch == 4
    assert fastest.mean_s(1) == pytest.approx((60.0 + 0.01 * 49.5) / 1000.0)
    standard = load_service_model(path, "standard")
    assert standard.max_batch == 8
    with pytest.raises(GpuBenchmarkError):
        load_service_model(path, "missing")


def test_load_service_model_reads_the_phase3_table(tmp_path: Path) -> None:
    path = tmp_path / "table.json"
    table = {
        "format": TABLE_FORMAT,
        "results": [
            {"batch_size": 1, "status": "ok", "median_ms": 100.0, "p90_ms": 110.0},
            {"batch_size": 2, "status": "ok", "median_ms": 120.0, "p90_ms": 130.0},
        ],
    }
    path.write_text(json.dumps(table), encoding="utf-8")
    model = load_service_model(path)
    assert isinstance(model, TableBatchProfile)
    with pytest.raises(GpuBenchmarkError):
        load_service_model(path, "standard")


def test_fleet_uses_the_measurement_file(tmp_path: Path) -> None:
    path = tmp_path / "gpu.json"
    path.write_text(json.dumps(_standard_document()), encoding="utf-8")
    config = decode(
        FleetSimConfig,
        {
            "gpus": {
                "t4": {"table": str(path)},
                "t4_standard": {"table": str(path), "configuration": "standard"},
            },
            "batching": {"realistisch": {"beta": 0.15}},
            "strategies": {"d": {"scheduler": "edf_batch", "max_batch": 16}},
            "networks": {"buero_wlan": {"scenario": "buero_wlan", "margin_ms": 15.0}},
            "horizon_s": 60.0,
            "window_start_s": 5.0,
        },
        "test",
    )
    assert isinstance(config.gpus["t4"], GpuConfig)
    model = service_model(config, "t4", "realistisch")
    assert isinstance(model, EmpiricalBatchProfile)
    assert model.max_batch == 4
    assert service_model(config, "t4_standard", "realistisch").max_batch == 8
    result = run_fleet(config, RunKey("buero_wlan", "t4", "realistisch", "d", 3, 0), [400])
    latencies = [r.latency_ms_p50 for r in result.robots if r.latency_ms_p50 is not None]
    assert result.requests > 0
    # fp16 serves one request in ~60 ms; with Wi-Fi (10-30 ms) and overheads the median stays below 150 ms
    assert latencies
    assert max(latencies) < 150.0
