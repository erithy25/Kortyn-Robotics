"""Batch benchmark (phase 3, step 3): size parsing, statistics, out-of-memory handling, readable table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kortyn.bench_batch import (
    TABLE_FORMAT,
    BenchmarkError,
    Settings,
    is_out_of_memory,
    parse_sizes,
    percentile,
    run_benchmark,
    slug,
    summarize,
    table,
)
from kortyn.fleet_brain import TABLE_FORMAT as MODEL_TABLE_FORMAT
from kortyn.fleet_brain import TableBatchProfile


def test_parse_sizes() -> None:
    assert parse_sizes("1-32") == list(range(1, 33))
    assert parse_sizes("8, 1,2,4") == [1, 2, 4, 8]
    assert parse_sizes("1-3,8,2") == [1, 2, 3, 8]


@pytest.mark.parametrize("text", ["", "0", "33", "4-2", "a", "1-40", "1,,x"])
def test_parse_sizes_rejects_invalid(text: str) -> None:
    with pytest.raises(BenchmarkError):
        parse_sizes(text)


def test_percentile_matches_numpy() -> None:
    values = [5.0, 1.0, 9.0, 3.0, 7.5, 2.25]
    for q in (0.0, 50.0, 90.0, 100.0):
        assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)))
    with pytest.raises(BenchmarkError):
        percentile([], 50.0)


def test_summarize_row() -> None:
    row = summarize(4, [100.0, 110.0, 120.0, 130.0, 200.0])
    assert row["status"] == "ok"
    assert row["median_ms"] == 120.0
    assert row["p90_ms"] == pytest.approx(172.0)
    assert row["per_request_ms"] == 30.0
    assert row["repeats"] == 5


class FakeClock:
    """Each ``infer(b)`` advances the clock by ``10 + 5 b`` ms."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def infer(self, size: int) -> None:
        self.now += (10.0 + 5.0 * size) / 1000.0


class CudaOutOfMemory(RuntimeError):
    pass


CudaOutOfMemory.__name__ = "OutOfMemoryError"


def test_run_benchmark_marks_out_of_memory_and_skips_larger_sizes() -> None:
    clock = FakeClock()
    freed: list[bool] = []

    def infer(size: int) -> None:
        if size >= 8:
            raise CudaOutOfMemory("CUDA out of memory. Tried to allocate 2.00 GiB")
        clock.infer(size)

    lines: list[str] = []
    rows = run_benchmark(
        infer,
        Settings([1, 2, 4, 8, 16], repeats=3, warmup=1),
        clock,
        lambda: freed.append(True),
        lines.append,
    )
    assert [r["status"] for r in rows] == ["ok", "ok", "ok", "oom", "skipped"]
    assert rows[0]["median_ms"] == pytest.approx(15.0)
    assert rows[2]["median_ms"] == pytest.approx(30.0)
    assert freed == [True]
    assert len(lines) == 4


def test_run_benchmark_propagates_other_errors() -> None:
    def infer(size: int) -> None:
        raise ValueError("shape mismatch")

    with pytest.raises(ValueError, match="shape"):
        run_benchmark(infer, Settings([1], repeats=1, warmup=0), report=lambda _: None)
    with pytest.raises(BenchmarkError):
        run_benchmark(infer, Settings([1], repeats=0, warmup=0))


def test_out_of_memory_detection() -> None:
    assert is_out_of_memory(CudaOutOfMemory("x"))
    assert is_out_of_memory(RuntimeError("DefaultCPUAllocator: can't allocate memory: out of memory"))
    assert not is_out_of_memory(RuntimeError("expected scalar type Float"))


def test_written_table_is_read_by_the_fleet_model(tmp_path: Path) -> None:
    clock = FakeClock()
    rows = run_benchmark(clock.infer, Settings([1, 2, 4], repeats=2, warmup=0), clock, report=lambda _: None)
    document = table(rows, {"device": {"gpu": "test"}})
    assert document["format"] == TABLE_FORMAT == MODEL_TABLE_FORMAT
    path = tmp_path / "table.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    model = TableBatchProfile.from_file(path)
    assert model.max_batch == 4
    assert model.mean_s(1) == pytest.approx(0.015)
    assert model.mean_s(3) == pytest.approx(0.025)


def test_table_needs_batch_one() -> None:
    with pytest.raises(BenchmarkError):
        table([{"batch_size": 1, "status": "oom"}], {})


def test_slug() -> None:
    assert slug("NVIDIA GeForce RTX 3060 Ti") == "rtx3060ti"
    assert slug("cpu") == "cpu"
    assert slug("!!!") == "device"
