"""The measurement script of the GPU notebook (kortyn.notebook_bench), without a GPU and without the model."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from kortyn import notebook_bench as nb

OBSERVATIONS = Path(__file__).resolve().parent.parent / "notebooks" / "libero_observations.json"


# --------------------------------------------------------------------------------------------- statistics
def test_percentile_matches_numpy() -> None:
    values = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0]
    for q in (0.0, 10.0, 50.0, 90.0, 95.0, 99.0, 100.0):
        assert nb.percentile(values, q) == pytest.approx(float(np.percentile(values, q)))
    with pytest.raises(nb.BenchmarkError):
        nb.percentile([], 50.0)
    with pytest.raises(nb.BenchmarkError):
        nb.percentile([1.0], 101.0)


def test_summarize_keeps_every_sample() -> None:
    timings = [10.0 + i for i in range(100)]
    row = nb.summarize(4, timings)
    assert row["status"] == "ok"
    assert row["repeats"] == 100
    assert row["median_ms"] == pytest.approx(59.5)
    assert row["p99_ms"] == pytest.approx(float(np.percentile(timings, 99)))
    assert row["p95_ms"] == pytest.approx(float(np.percentile(timings, 95)))
    assert row["per_request_ms"] == pytest.approx(59.5 / 4)
    assert row["samples_ms"] == timings
    with pytest.raises(nb.BenchmarkError):
        nb.summarize(1, [])


def test_projected_seconds_uses_the_last_warmup_runs() -> None:
    # the first warm-up run compiles (slow); the projection ignores it
    assert nb.projected_seconds([60_000.0, 100.0, 110.0, 120.0], 100) == pytest.approx(11.0)
    assert nb.projected_seconds([], 100) == 0.0


def test_out_of_memory_detection() -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    assert nb.is_out_of_memory(OutOfMemoryError("x"))
    assert nb.is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not nb.is_out_of_memory(RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED"))


def test_settings_validation() -> None:
    with pytest.raises(nb.BenchmarkError):
        nb.Settings(sizes=(2, 1))
    with pytest.raises(nb.BenchmarkError):
        nb.Settings(sizes=(0, 1))
    with pytest.raises(nb.BenchmarkError):
        nb.Settings(repeats=0)
    with pytest.raises(nb.BenchmarkError):
        nb.Settings(time_limit_s=0.0)


class FakeClock:
    """Each infer(b) call advances the clock by ``cost(b)`` milliseconds."""

    def __init__(self, cost_ms: dict[int, float]) -> None:
        self.now = 0.0
        self.cost_ms = cost_ms
        self.calls: list[int] = []

    def __call__(self) -> float:
        return self.now

    def infer(self, size: int) -> None:
        self.calls.append(size)
        self.now += self.cost_ms[size] / 1000.0


def test_time_sizes_measures_every_size() -> None:
    clock = FakeClock({1: 10.0, 2: 15.0, 4: 25.0})
    lines: list[str] = []
    rows = nb.time_sizes(
        clock.infer,
        nb.Settings(sizes=(1, 2, 4), repeats=5, warmup=2),
        clock=clock,
        peak_memory_mb=lambda: (100.0, 128.0),
        report=lines.append,
    )
    assert [row["status"] for row in rows] == ["ok", "ok", "ok"]
    assert rows[1]["median_ms"] == pytest.approx(15.0)
    assert rows[1]["samples_ms"] == [15.0] * 5
    assert rows[2]["warmup_ms"] == [25.0, 25.0]
    assert rows[0]["memory_reserved_peak_mb"] == 128.0
    assert clock.calls.count(4) == 7  # 2 warm-up + 5 timed
    assert len(lines) == 3


def test_time_sizes_stops_at_out_of_memory() -> None:
    freed: list[bool] = []
    clock = FakeClock({1: 10.0, 2: 15.0})

    def infer(size: int) -> None:
        if size >= 4:
            raise RuntimeError("CUDA out of memory")
        clock.infer(size)

    rows = nb.time_sizes(
        infer,
        nb.Settings(sizes=(1, 2, 4, 8), repeats=3, warmup=1),
        clock=clock,
        after_oom=lambda: freed.append(True),
        report=lambda line: None,
    )
    assert [row["status"] for row in rows] == ["ok", "ok", "oom", "oom"]
    assert freed == [True]  # memory freed once, larger sizes are not tried
    assert 8 not in clock.calls


def test_time_sizes_skips_sizes_beyond_the_time_limit() -> None:
    clock = FakeClock({1: 10.0, 2: 4000.0, 4: 8000.0})
    rows = nb.time_sizes(
        clock.infer,
        nb.Settings(sizes=(1, 2, 4), repeats=100, warmup=2, time_limit_s=300.0),
        clock=clock,
        report=lambda line: None,
    )
    # batch 2: 100 x 4 s = 400 s > 300 s
    assert [row["status"] for row in rows] == ["ok", "skipped", "skipped"]
    assert "time limit" in rows[1]["reason"]
    assert clock.calls.count(2) == 2  # only the warm-up ran
    assert 4 not in clock.calls


def test_time_sizes_reraises_other_errors() -> None:
    def infer(size: int) -> None:
        raise ValueError("broken")

    with pytest.raises(ValueError, match="broken"):
        nb.time_sizes(infer, nb.Settings(sizes=(1,), repeats=1, warmup=0), report=lambda line: None)


# --------------------------------------------------------------------------------------------- numerics
def test_numeric_check() -> None:
    reference = np.zeros((2, 3, 7))
    close = reference + 0.001
    check = nb.numeric_check(close, reference, spread=0.01)
    assert check["valid"]
    assert check["ratio"] == pytest.approx(0.1)
    assert not nb.numeric_check(reference + 0.005, reference, spread=0.01)["valid"]  # 50 % > 25 %
    broken = reference.copy()
    broken[0, 0, 0] = np.nan
    assert nb.numeric_check(broken, reference, spread=0.01) == {
        "finite": False,
        "rms_deviation": None,
        "spread": 0.01,
        "ratio": None,
        "threshold": nb.NUMERIC_THRESHOLD,
        "valid": False,
    }
    assert not nb.numeric_check(close, reference, spread=0.0)["valid"]
    with pytest.raises(nb.BenchmarkError):
        nb.rms(np.zeros(3), np.zeros(4))


def _entry(name: str, median: float | None, valid: bool = True, status: str = "ok") -> dict[str, Any]:
    results = (
        []
        if median is None
        else [{"batch_size": 1, "status": "ok", "median_ms": median, "p99_ms": 2 * median}]
    )
    return {"name": name, "status": status, "numerics": {"valid": valid}, "results": results}


def test_validity_and_fastest_configuration() -> None:
    configurations = [
        _entry("standard", 120.0),
        _entry("fp16", 60.0),
        _entry("fp32", 40.0, valid=False),
        _entry("compiled", 30.0, status="error"),
    ]
    assert nb.is_valid(configurations[0])
    assert not nb.is_valid(configurations[2])
    assert not nb.is_valid(_entry("x", None))
    assert nb.fastest_valid(configurations, nb.EAGER) == "fp16"
    assert nb.fastest_valid(configurations, ["fp32", "compiled"]) is None
    assert nb.batch_one_median(configurations[0]) == 120.0


# --------------------------------------------------------------------------------------------- output
def test_file_names() -> None:
    now = dt.datetime(2026, 10, 28, 9, 5, tzinfo=dt.UTC)
    assert nb.slug("Tesla T4") == "teslat4"
    assert nb.slug("NVIDIA GeForce RTX 3060 Ti") == "rtx3060ti"
    assert nb.slug("!!!") == "device"
    assert nb.output_name("Tesla T4", "colab", now) == "smolvla_gpu_teslat4_colab_20261028-0905.json"


def test_write_json_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "result.json"
    nb.write_json(path, {"a": 1, "text": "Größe"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "text": "Größe"}
    assert not list(tmp_path.glob("sub/*.part"))


def test_summary_lines() -> None:
    entry = _entry("fp16", 60.0)
    entry["results"].append({"batch_size": 8, "status": "ok", "median_ms": 150.0, "p99_ms": 160.0})
    entry["results"].append({"batch_size": 32, "status": "oom"})
    lines = nb.summary_lines([entry, _entry("compiled", None, status="error")])
    assert len(lines) == 3
    assert "60 ms" in lines[1]
    assert "120 ms" in lines[1]  # p99 at batch 1
    assert "oom" in lines[1]
    assert "Fehler" in lines[2]


def test_session_document_selects_the_fastest_valid(tmp_path: Path) -> None:
    session = nb.Session(out=tmp_path / "r.json", meta={"platform": "colab"})
    session.configurations += [_entry("standard", 120.0), _entry("fp16", 60.0)]
    session.save()
    document = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert document["format"] == nb.FORMAT
    assert document["selected"] == "fp16"
    assert document["platform"] == "colab"


# --------------------------------------------------------------------------------------------- inputs
def test_embedded_observations_are_real_libero_frames() -> None:
    observations = nb.load_observations(OBSERVATIONS)
    assert observations.count == 8
    for frames in observations.cameras.values():
        assert frames.shape == (8, 360, 360, 3)
        assert frames.dtype == np.uint8
        assert frames.std() > 10  # a rendered scene, not a blank frame
    assert observations.states.shape == (8, 8)
    assert set(observations.tasks) == {
        "put both moka pots on the stove",
        "put both the alphabet soup and the tomato sauce in the basket",
    }
    assert observations.source["tasks"] == [8, 8, 8, 8, 0, 0, 0, 0]
    assert observations.indices(10) == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]


def test_unreadable_observations(tmp_path: Path) -> None:
    path = tmp_path / "obs.json"
    path.write_text(json.dumps({"cameras": {}}), encoding="utf-8")
    with pytest.raises(nb.BenchmarkError):
        nb.load_observations(path)
    with pytest.raises(nb.BenchmarkError):
        nb.Observations(
            cameras={name: np.zeros((2, 4, 4, 3), dtype=np.uint8) for name in nb.CAMERAS},
            states=np.zeros((3, 8), dtype=np.float32),
            tasks=("a", "b"),
        )


# --------------------------------------------------------------------------------------------- runner
class FakeModel(torch.nn.Module):
    def sample_actions(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FakeConfig:
    chunk_size = 50
    max_action_dim = 32
    device = "cpu"


class FakePolicy(torch.nn.Module):
    """Shapes and dtypes of SmolVLA's policy: a bf16 backbone weight and a fp32 projection."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        self.projection = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.model = FakeModel()
        self.config = FakeConfig()
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def predict_action_chunk(self, batch: dict[str, Any], noise: torch.Tensor) -> torch.Tensor:
        state: torch.Tensor = batch[nb.STATE_KEY]
        result: torch.Tensor = noise[:, :, :7] + state[:, :1, None]
        return result


@pytest.fixture
def observations() -> nb.Observations:
    return nb.load_observations(OBSERVATIONS)


def _runner(name: str) -> nb.Runner:
    return nb.Runner(FakePolicy(), lambda batch: batch, lambda actions: actions, name, "cpu")


def test_runner_precisions() -> None:
    assert _runner("standard").dtypes() == {"bfloat16": 4, "float32": 2}
    assert _runner("fp16").dtypes() == {"float16": 4, "float32": 2}
    assert _runner("fp32").dtypes() == {"float32": 6}
    assert _runner("fp16").precision == "fp16"


def test_runner_batches_and_noise(observations: nb.Observations) -> None:
    runner = _runner("standard")
    runner.use(observations)
    actions = runner.actions(10, seed=1)
    assert actions.shape == (10, 50, 7)
    assert actions.dtype == np.float32
    again = runner.actions(10, seed=1)
    other = runner.actions(10, seed=2)
    assert np.array_equal(actions, again)  # fixed noise per seed
    assert not np.array_equal(actions, other)
    # requests 0 and 8 hold the same observation (taken in turn), but not the same noise
    assert not np.array_equal(actions[0], actions[8])
    assert runner.policy.resets == 3
    runner.infer(2)
    assert runner.peak_memory_mb() == (0.0, 0.0)
    runner.free()
    assert runner.policy is None


def test_runner_without_observations_fails() -> None:
    with pytest.raises(nb.BenchmarkError):
        _runner("standard").actions(1)


def test_runner_compile_wraps_sample_actions() -> None:
    runner = _runner("standard")
    original = runner.policy.model.sample_actions
    previous = torch.get_float32_matmul_precision()
    try:
        runner.compile()
        assert runner.compiled
        assert runner.policy.model.sample_actions is not original
        assert torch.get_float32_matmul_precision() == "high"  # as LeRobot's compile_model switch
    finally:
        torch.set_float32_matmul_precision(previous)


# --------------------------------------------------------------------------------------------- CLI
def test_parse_list() -> None:
    assert nb.parse_list("1, 2,4", int) == [1, 2, 4]
    with pytest.raises(nb.BenchmarkError):
        nb.parse_list("1,x", int)


def test_main_rejects_unknown_configurations(capsys: pytest.CaptureFixture[str]) -> None:
    assert nb.main(["--observations", str(OBSERVATIONS), "--configs", "standard,bf16"]) == 2
    assert "unknown configurations" in capsys.readouterr().err


def test_main_refuses_the_cpu_without_the_test_switch(capsys: pytest.CaptureFixture[str]) -> None:
    assert nb.main(["--observations", str(OBSERVATIONS), "--device", "cpu"]) == 2
    assert "needs a CUDA device" in capsys.readouterr().err


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the message without a GPU")
def test_main_without_gpu_explains_what_to_do(capsys: pytest.CaptureFixture[str]) -> None:
    assert nb.main(["--observations", str(OBSERVATIONS)]) == 2
    assert "keine NVIDIA-GPU" in capsys.readouterr().err
