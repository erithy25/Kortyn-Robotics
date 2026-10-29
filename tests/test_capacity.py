"""The capacity tool (kortyn.capacity) and the dose-response export (kortyn.dose_export)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kortyn import capacity as cap
from kortyn.config import ConfigError, decode
from kortyn.dose_export import export_condition
from kortyn.fleet import RobotSummary, RunKey, RunResult
from kortyn.gpu_benchmark import MIN_REPEATS, V2_FORMAT

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CURVE = ROOT / "examples" / "capacity" / "smolvla_libero_long.json"
EXAMPLE_CONFIG = ROOT / "examples" / "capacity" / "example.yaml"
PHASE4_CAPACITY = ROOT / "results" / "phase4" / "fleet" / "capacity.json"
CPU_TEST_FILE = ROOT / "tests" / "data" / "notebook_cpu_test.json"


def _curve_document() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(EXAMPLE_CURVE.read_text(encoding="utf-8"))
    return data


def _write(tmp_path: Path, name: str, data: Any) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------- curve
def test_reads_the_example_curve() -> None:
    dose = cap.read_dose_response(EXAMPLE_CURVE)
    assert len(dose.tasks) == 2
    assert dose.spec.fit_max_ms == 1250.0
    assert dose.tasks[0].latency_ms[:3] == (0.0, 100.0, 200.0)
    assert all(trials == 50 for task in dose.tasks for trials in task.episodes)


def test_example_curve_reproduces_phase4() -> None:
    """Point fit on counts = phase-4 fit on seeds: 45 % at 0 ms, L* = 200 ms (reports/phase4.md 4.1-4.2)."""
    curve = cap.fit_curve(cap.read_dose_response(EXAMPLE_CURVE))
    assert set(curve.families.values()) == {"loglogistic"}
    assert float(curve.value(0.0)) == pytest.approx(0.453, abs=0.002)
    assert curve.tolerable_latency(0.03)[0] == 200.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(format="other"),
        lambda d: d.pop("tasks"),
        lambda d: d.update(tasks=[]),
        lambda d: d["tasks"][0]["levels"][0].update(successes=60),
        lambda d: d["tasks"][0]["levels"][0].update(latency_ms=-5),
        lambda d: d["tasks"][0]["levels"][0].pop("episodes"),
        lambda d: d.update(fit_max_ms=10.0),
    ],
)
def test_malformed_curves_are_rejected(tmp_path: Path, mutate: Any) -> None:
    document = _curve_document()
    mutate(document)
    with pytest.raises(cap.CapacityError):
        cap.read_dose_response(_write(tmp_path, "curve.json", document))


def test_unreadable_curve(tmp_path: Path) -> None:
    with pytest.raises(cap.CapacityError):
        cap.read_dose_response(tmp_path / "missing.json")


def test_curve_needs_four_levels_inside_the_budget(tmp_path: Path) -> None:
    document = _curve_document()
    for task in document["tasks"]:
        task["levels"] = [level for level in task["levels"] if level["latency_ms"] in (0.0, 500.0, 1500.0)]
    with pytest.raises(cap.CapacityError, match="four levels"):
        cap.fit_curve(cap.read_dose_response(_write(tmp_path, "curve.json", document)))


def test_levels_on_the_same_tick_are_pooled(tmp_path: Path) -> None:
    document = _curve_document()
    # 0 ms and 30 ms act as the same tick: adding a 30-ms copy of the 0-ms level doubles its weight only
    for task in document["tasks"]:
        task["levels"].insert(1, dict(task["levels"][0], latency_ms=30.0))
    curve = cap.fit_curve(cap.read_dose_response(_write(tmp_path, "curve.json", document)))
    assert float(curve.value(0.0)) == pytest.approx(0.45, abs=0.03)


def test_bootstrap_curves_are_reproducible() -> None:
    dose = cap.read_dose_response(EXAMPLE_CURVE)
    first = cap.bootstrap_curves(dose, 5, seed=1)
    second = cap.bootstrap_curves(dose, 5, seed=1)
    grid = np.array([0.0, 200.0, 500.0, 1000.0])
    assert len(first) == 5
    for a, b in zip(first, second, strict=True):
        assert np.allclose(a.value(grid), b.value(grid))
        assert np.all((a.value(grid) >= 0) & (a.value(grid) <= 1))
    assert not np.allclose(first[0].value(grid), first[1].value(grid))


def test_bin_values_are_the_curve_at_bin_centres() -> None:
    curve = cap.fit_curve(cap.read_dose_response(EXAMPLE_CURVE))
    values = cap.bin_values(curve, 4)
    assert values.shape == (4,)
    assert values[0] == pytest.approx(float(curve.value(5.0)))
    assert values[3] == pytest.approx(float(curve.value(35.0)))


# --------------------------------------------------------------------------------------------- evaluation
def _result(
    robots: int, histograms: list[dict[int, int]], admitted: list[bool] | None = None, misses: int = 0
) -> RunResult:
    flags = admitted or [True] * robots
    summaries = [
        RobotSummary(
            robot=i,
            admitted=flags[i],
            start_time=0.0,
            episodes=1,
            requests=sum(h.values()),
            misses=0,
            start_requests=1,
            start_misses=0,
            starved_share=0.0,
            plan_age_ms=0.0,
            latency_ms_p50=None,
            latency_ms_p99=None,
            histogram=h,
        )
        for i, h in enumerate(histograms)
    ]
    requests = sum(sum(h.values()) for h in histograms)
    return RunResult(
        key=RunKey("net", "gpu", "b", "d", robots, 0),
        admitted=sum(flags),
        requests=requests,
        misses=misses,
        timeouts=0,
        batches=requests,
        mean_batch=1.0,
        busy_share=0.5,
        robots=summaries,
        wall_s=0.0,
    )


def test_pooled_quantile_is_the_upper_bin_edge() -> None:
    histogram = np.zeros(20)
    histogram[5] = 98  # 50-60 ms
    histogram[14] = 2  # 140-150 ms
    assert cap.pooled_quantile_ms(histogram, 0.5) == 60.0
    assert cap.pooled_quantile_ms(histogram, 0.98) == 60.0
    assert cap.pooled_quantile_ms(histogram, 0.99) == 150.0
    assert cap.pooled_quantile_ms(np.zeros(3), 0.5) is None


def test_latency_target_counts_requests_below_the_bound() -> None:
    point = np.full(30, 0.4)
    draws = point[None, :]
    slo = cap.LatencySlo(quantile=0.99, max_ms=150.0)
    passing = _result(2, [{3: 99, 14: 1}, {3: 100}])  # 1 of 200 in 140-150 ms: below 150
    failing = _result(2, [{3: 97, 15: 3}, {3: 100}])  # 3 of 200 at 150-160 ms: 98.5 % below
    rows = cap.evaluate_sizes({1: [_result(1, [{3: 100}])], 2: [passing]}, point, draws, 0.03, slo)
    assert rows[1].passes_slo is True
    assert rows[1].slo_share == pytest.approx(1.0)
    rows = cap.evaluate_sizes({1: [_result(1, [{3: 100}])], 2: [failing]}, point, draws, 0.03, slo)
    assert rows[1].passes_slo is False
    assert rows[1].slo_share == pytest.approx(0.985)


def test_a_rejected_robot_fails_the_latency_target_and_predicts_zero() -> None:
    point = np.full(30, 0.4)
    slo = cap.LatencySlo(quantile=0.99, max_ms=150.0)
    rejected = _result(2, [{3: 100}, {}], admitted=[True, False])
    rows = cap.evaluate_sizes({1: [_result(1, [{3: 100}])], 2: [rejected]}, point, point[None, :], 0.03, slo)
    assert rows[1].passes_slo is False
    assert rows[1].worst == 0.0
    assert not rows[1].passes


def test_capacity_criteria() -> None:
    # success falls from 40 % (bins below 100 ms) to 30 % beyond: requests of 200 ms cost a robot 10 pp
    point = np.where(np.arange(40) < 10, 0.4, 0.3)
    runs = {
        1: [_result(1, [{2: 100}])],
        2: [_result(2, [{2: 100}, {3: 100}])],
        3: [_result(3, [{2: 100}, {3: 100}, {20: 100}])],
    }
    rows = cap.evaluate_sizes(runs, point, point[None, :], 0.03, None)
    assert [row.passes for row in rows] == [True, True, False]
    assert rows[2].worst == pytest.approx(0.3)
    assert rows[2].loss == pytest.approx(0.1)
    assert rows[0].passes_slo is None
    assert cap.capacity_of(rows, "passes") == 2


def test_late_plans_fail_the_capacity() -> None:
    point = np.full(30, 0.4)
    runs = {1: [_result(1, [{2: 100}])], 2: [_result(2, [{2: 100}, {2: 100}], misses=3)]}
    rows = cap.evaluate_sizes(runs, point, point[None, :], 0.03, None)
    assert rows[1].miss_share == pytest.approx(0.015)
    assert not rows[1].passes


def test_capacity_needs_consecutive_sizes() -> None:
    def row(robots: int, ok: bool) -> cap.SizeRow:
        return cap.SizeRow(
            robots, 1, 1, 0, 0.0, 1.0, 1.0, 0.1, 10.0, 20.0, None, 0.4, 0.4, 0.4, 0.0, 0.0, ok, ok, None
        )

    assert cap.capacity_of([row(1, True), row(2, True), row(3, False), row(4, True)], "passes") == 2
    assert cap.capacity_of([row(2, True)], "passes") == 0  # size 1 missing
    assert cap.capacity_of([row(1, False)], "passes") == 0


def test_evaluation_needs_the_single_robot() -> None:
    with pytest.raises(cap.CapacityError):
        cap.evaluate_sizes({2: [_result(2, [{1: 1}, {1: 1}])]}, np.ones(5), np.ones((1, 5)), 0.03, None)


# --------------------------------------------------------------------------------------------- config
def test_config_validation() -> None:
    with pytest.raises(ConfigError):
        cap.GpuSpec()
    with pytest.raises(ConfigError):
        cap.GpuSpec(measurement="x.json", s1_ms=100.0)
    with pytest.raises(ConfigError):
        cap.GpuSpec(s1_ms=-1.0)
    with pytest.raises(ConfigError):
        cap.LatencySlo(max_ms=155.0)  # not a multiple of the 10-ms histogram bins
    with pytest.raises(ConfigError):
        cap.LatencySlo(quantile=1.0)
    with pytest.raises(ConfigError):
        cap.CriteriaSpec(tolerance_pp=0.0)
    with pytest.raises(ConfigError):
        cap.FleetSpec(window_start_s=700.0)
    with pytest.raises(ConfigError):
        decode(cap.CapacityConfig, {"dose_response": "x", "gpu": {"s1_ms": 100.0}, "bootstrap": -1}, "test")
    config = cap.load_capacity_config(EXAMPLE_CONFIG)
    assert sorted(config.strategies) == ["a", "b", "d", "e"]
    assert config.criteria.latency_slo is not None
    assert config.criteria.latency_slo.robots == 10


def _benchmark_file(tmp_path: Path) -> Path:
    def row(batch: int, base: float) -> dict[str, Any]:
        samples = [base + 0.1 * i for i in range(MIN_REPEATS)]
        return {
            "batch_size": batch,
            "status": "ok",
            "median_ms": float(np.median(samples)),
            "samples_ms": samples,
        }

    document = {
        "format": V2_FORMAT,
        "platform": "colab",
        "device": {"device": "cuda", "gpu": "Tesla T4"},
        "configurations": [
            {
                "name": "standard",
                "status": "ok",
                "numerics": {"valid": True},
                "results": [row(1, 90.0), row(2, 110.0)],
            },
            {
                "name": "fp16",
                "status": "ok",
                "numerics": {"valid": True},
                "results": [row(1, 40.0), row(2, 50.0)],
            },
            {"name": "fp32", "status": "ok", "numerics": {"valid": False}, "results": [row(1, 30.0)]},
        ],
    }
    return _write(tmp_path, "gpu.json", document)


def test_gpu_variants(tmp_path: Path) -> None:
    affine = cap.gpu_variants(cap.GpuSpec(s1_ms=220.0, label="x"))
    assert [v.name for v in affine] == ["x"]
    assert affine[0].batching == "annahme"
    path = _benchmark_file(tmp_path)
    assert [v.name for v in cap.gpu_variants(cap.GpuSpec(measurement=str(path), label="t4"))] == ["t4/fp16"]
    every = cap.gpu_variants(cap.GpuSpec(measurement=str(path), label="t4", configuration=cap.ALL_VALID))
    assert [v.name for v in every] == ["t4/standard", "t4/fp16"]  # fp32 failed the numerical check
    named = cap.gpu_variants(cap.GpuSpec(measurement=str(path), configuration="fp32"))
    assert named[0].gpu.configuration == "fp32"
    with pytest.raises(cap.CapacityError):
        cap.gpu_variants(cap.GpuSpec(measurement=str(path), configuration="missing"))
    with pytest.raises(cap.CapacityError):
        cap.gpu_variants(cap.GpuSpec(measurement=str(tmp_path / "missing.json")))
    table = _write(
        tmp_path,
        "table.json",
        {
            "format": "kortyn-batch-service-v1",
            "results": [{"batch_size": 1, "status": "ok", "median_ms": 100.0, "p90_ms": 110.0}],
        },
    )
    assert cap.gpu_variants(cap.GpuSpec(measurement=str(table)))[0].batching == cap.MEASURED_BATCHING
    with pytest.raises(cap.CapacityError):
        cap.gpu_variants(cap.GpuSpec(measurement=str(table), configuration="fp16"))


def test_no_valid_configuration_is_reported() -> None:
    # the notebook's CPU test: 3 measurements per size, below the pre-registered 100
    with pytest.raises(cap.CapacityError):
        cap.gpu_variants(cap.GpuSpec(measurement=str(CPU_TEST_FILE), configuration=cap.ALL_VALID))
    with pytest.raises(cap.CapacityError):
        cap.gpu_variants(cap.GpuSpec(measurement=str(CPU_TEST_FILE)))


# --------------------------------------------------------------------------------------------- end to end
def test_tool_reproduces_the_phase4_capacity_exactly() -> None:
    """Same inputs as phase 4 (curve without RTC, RTX 3060 Ti assumption, office Wi-Fi): same numbers."""
    config = cap.load_capacity_config(EXAMPLE_CONFIG)
    config.strategies = {name: config.strategies[name] for name in ("a", "d")}
    config.fleet.max_robots = 3
    config.bootstrap = 0
    document = cap.run_tool(config)
    reference = json.loads(PHASE4_CAPACITY.read_text(encoding="utf-8"))
    sizes = {
        c["strategy"]: {s["robots"]: s for s in c["sizes"]}
        for c in reference["configurations"]
        if c["gpu"] == "rtx3060ti_schaetzung"
        and c["batching"] == "realistisch"
        and c["network"] == "buero_wlan"
    }
    for result in document["results"]:
        for row in result["rows"]:
            phase4 = sizes[result["strategy"]][row["robots"]]
            for key in ("worst", "single", "fleet_mean", "miss_share"):
                assert row[key] == pytest.approx(phase4[key], abs=1e-12), (
                    result["strategy"],
                    row["robots"],
                    key,
                )
    capacities = {r["strategy"]: r["capacity"] for r in document["results"]}
    assert capacities == {"a": 2, "d": 2}  # reports/phase4.md 5.1 without RTC, realistic batching
    assert document["latency"]["latency_capacity"] == 0
    assert document["latency"]["meets_target"] is False


def test_tool_with_a_measurement_file(tmp_path: Path) -> None:
    config = cap.load_capacity_config(EXAMPLE_CONFIG)
    config.gpu = cap.GpuSpec(
        measurement=str(_benchmark_file(tmp_path)), configuration=cap.ALL_VALID, label="t4"
    )
    config.strategies = {"d": config.strategies["d"]}
    config.fleet.max_robots = 6
    config.fleet.replications = 2
    config.bootstrap = 10
    document = cap.run_tool(config)
    assert [r["gpu"] for r in document["results"]] == ["t4/standard", "t4/fp16"]
    fp16 = document["results"][1]
    # 40-50 ms of service plus 10-30 ms of Wi-Fi: well inside the 150-ms target at small fleets
    assert fp16["rows"][0]["latency_p99_ms"] < 150.0
    assert fp16["latency_capacity"] >= 1
    assert document["latency"]["gpu"] in ("t4/standard", "t4/fp16")
    lines = cap.report_lines(document)
    assert any("t4/fp16" in line for line in lines)


def test_main_runs_the_example(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = copy.deepcopy(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    config = config.replace("max_robots: 32", "max_robots: 2").replace("bootstrap: 200", "bootstrap: 0")
    path = tmp_path / "config.yaml"
    path.write_text(config, encoding="utf-8")
    out = tmp_path / "out.json"
    assert cap.main(["--config", str(path), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "strategy d: capacity 2 robots" in printed
    assert json.loads(out.read_text(encoding="utf-8"))["best"]["capacity"] == 2


def test_main_with_the_notebook_cpu_test_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A CPU needs 2.3 s per plan: no robot can be served within the 1.25-s budget."""
    out = tmp_path / "cpu.json"
    code = cap.main(
        [
            "--config",
            str(EXAMPLE_CONFIG),
            "--gpu",
            str(CPU_TEST_FILE),
            "--configuration",
            "standard",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert all(result["capacity"] == 0 for result in document["results"])
    printed = capsys.readouterr().out
    assert "NOT met" in printed
    # the measurement is named after its file, not after the config's assumed GPU it replaces
    assert all(result["gpu"].startswith(CPU_TEST_FILE.stem) for result in document["results"])
    assert "rtx3060ti" not in printed
    assert "0 robots within the target" in printed


def test_robot_counts_read_naturally() -> None:
    assert cap._robots(1) == "1 robot"
    assert cap._robots(0) == "0 robots"
    assert cap._robots(12) == "12 robots"


def test_main_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("dose_response: x.json\ngpu: {s1_ms: 100}\nunknown_key: 1\n", encoding="utf-8")
    assert cap.main(["--config", str(path)]) == 2
    assert "error" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------- export
def test_export_matches_the_example_curve() -> None:
    document = export_condition("kortyn/experiments/phase4/analysis.yaml", "ohne_rtc", "x")
    example = _curve_document()
    assert document["tasks"] == example["tasks"]
    assert document["source"]["runs"][0] == "results/phase4/libero/smolvla_libero/fest_0ms"


def test_export_unknown_condition() -> None:
    with pytest.raises(ConfigError):
        export_condition("kortyn/experiments/phase4/analysis.yaml", "missing")
