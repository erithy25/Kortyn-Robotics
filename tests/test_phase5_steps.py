"""Phase-5 evaluation (kortyn.phase5_steps): gate, curves after the gate, GPU decision."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from kortyn import capacity as cap
from kortyn import phase5_steps as p5
from kortyn.gpu_benchmark import MIN_REPEATS, V2_FORMAT
from kortyn.phase3_steps import StepError
from kortyn.phase4_steps import LevelSpec

ROOT = Path(__file__).resolve().parent.parent
CPU_TEST_FILE = ROOT / "tests" / "data" / "notebook_cpu_test.json"


def _run(path: Path, outcomes: dict[int, bool]) -> str:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for seed, success in sorted(outcomes.items()):
            handle.write(json.dumps({"seed": seed, "attempt": 0, "success": success, "steps": 300}) + "\n")
    return str(path)


def _outcomes(seeds: range, rate: float, seed: int) -> dict[int, bool]:
    rng = np.random.default_rng(seed)
    return {s: bool(rng.random() < rate) for s in seeds}


def _gate_config(tmp_path: Path, rtc_rate: float) -> p5.Phase5Config:
    base = _outcomes(range(1000, 1200), 0.5, 1)
    # RTC equals the reference except on a few seeds (equal) or on many (worse)
    rng = np.random.default_rng(2)
    rtc = {s: (v if rng.random() < 0.85 else bool(rng.random() < rtc_rate)) for s, v in base.items()}
    gate = p5.GateSpec(
        reference=_run(tmp_path / "sync25", base),
        control=_run(tmp_path / "async", {s: (v if rng.random() < 0.85 else not v) for s, v in base.items()}),
        test=_run(tmp_path / "rtc", rtc),
        phase4_async=_run(tmp_path / "p4", _outcomes(range(1000, 1100), 0.47, 3)),
        phase3_sync=_run(tmp_path / "p3", _outcomes(range(1000, 1100), 0.53, 4)),
        phase4_rtc=_run(tmp_path / "p4rtc", _outcomes(range(1000, 1100), 0.42, 5)),
        draws=2000,
        output=str(tmp_path / "gate.json"),
    )
    return p5.Phase5Config(gate=gate)


def test_gate_passes_when_rtc_matches_the_reference(tmp_path: Path) -> None:
    document = p5.gate_document(_gate_config(tmp_path, rtc_rate=0.5))
    gate = document["gate"]
    assert gate["pairs"] == 200
    assert gate["clusters"] == 100  # seeds s and s + 100 share the init state
    assert gate["passed"] is True
    assert document["decision"] == p5.DOSE_BLOCK_RUNS
    assert set(document["controls"]) == {"K1_async_vs_sync25", "K2_rtc_vs_async"}
    assert document["controls"]["K1_async_vs_sync25"]["test"] == "async_0ms"
    assert set(document["per_task"]) == {"0", "8"}
    assert document["per_task"]["8"]["gate"]["pairs"] == 100
    assert document["rates"]["sync25"]["episodes"] == 200
    assert document["reproduction"]["async_0ms_vs_phase4_block_a"]["pairs"] == 100


def test_gate_fails_and_stops_the_rtc_measurement(tmp_path: Path) -> None:
    config = _gate_config(tmp_path, rtc_rate=0.5)
    # RTC fails every episode: far below the reference
    failing = dict.fromkeys(range(1000, 1200), False)
    config.gate.test = _run(tmp_path / "rtc_bad", failing)
    document = p5.gate_document(config)
    assert document["gate"]["passed"] is False
    assert document["gate"]["verdict"] == "nicht bestanden"
    assert document["decision"] == p5.STOP


def test_gate_needs_common_seeds(tmp_path: Path) -> None:
    config = _gate_config(tmp_path, rtc_rate=0.5)
    config.gate.test = _run(tmp_path / "other", {5000: True})
    with pytest.raises(StepError):
        p5.gate_document(config)


def _phase4_analysis_copy(tmp_path: Path) -> str:
    data = yaml.safe_load((ROOT / "kortyn/experiments/phase4/analysis.yaml").read_text(encoding="utf-8"))
    data["bootstrap"] = 100
    path = tmp_path / "phase4_analysis.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_curves_refuse_without_a_passed_gate(tmp_path: Path) -> None:
    config = p5.Phase5Config(gate=p5.GateSpec(output=str(tmp_path / "gate.json")))
    with pytest.raises(StepError, match="run the gate"):
        p5.curves_document(config)
    (tmp_path / "gate.json").write_text(json.dumps({"gate": {"passed": False}}), encoding="utf-8")
    with pytest.raises(StepError, match="did not pass"):
        p5.curves_document(config)


def test_curves_path_with_the_phase4_rtc_runs_as_stand_in(tmp_path: Path) -> None:
    """The code path of section 3.7 on real data: phase-4 RTC levels 0/200/500/1000 stand in for rtc_kurz."""
    (tmp_path / "gate.json").write_text(json.dumps({"gate": {"passed": True}}), encoding="utf-8")
    levels = [
        LevelSpec(latency_ms=latency, run=f"results/phase4/libero/smolvla_libero_rtc/rtc_{latency}ms")
        for latency in (0, 200, 500, 1000)
    ]
    config = p5.Phase5Config(
        phase4_analysis=_phase4_analysis_copy(tmp_path),
        gate=p5.GateSpec(output=str(tmp_path / "gate.json")),
        curves=p5.CurvesSpec(levels=levels, output=str(tmp_path / "curves.json")),
    )
    document = p5.curves_document(config)
    assert set(document["conditions"]) == {"ohne_rtc", "mit_rtc", "rtc_kurz"}
    assert document["h1_kurz"]["h1"]["verdict"] in ("bestätigt", "RTC schadet", "nicht nachweisbar")
    levels_out = {row["latency_ms"]: row for row in document["levels"]}
    assert set(levels_out) == {0.0, 200.0, 500.0, 1000.0}
    # phase-4 numbers at 0 ms: 47 % without, 42 % with RTC (reports/phase4.md 3.1)
    assert levels_out[0.0]["rate_a"] == pytest.approx(0.47)
    assert levels_out[0.0]["rate_b"] == pytest.approx(0.42)
    assert document["capacity"]["configurations"]
    assert document["phase4_long"]["h1"]["gain"] == pytest.approx(-0.016, abs=0.01)


def test_condition_seed_window(tmp_path: Path) -> None:
    levels = [
        LevelSpec(latency_ms=0, run=_run(tmp_path / "a", _outcomes(range(1000, 1200), 0.5, 7))),
        LevelSpec(latency_ms=500, run=_run(tmp_path / "b", _outcomes(range(1000, 1100), 0.3, 8))),
    ]
    from kortyn.phase4_steps import libero_layout

    libero = libero_layout("kortyn/experiments/phase5/libero_rtc_kurz.yaml")
    dose = p5.load_condition_on(levels, libero, 1000, 1099)
    assert dose.seeds == tuple(range(1000, 1100))
    with pytest.raises(StepError):
        p5.load_condition_on(levels, libero, 1100, 1199)  # level b has no seed there
    with pytest.raises(StepError):
        p5.load_condition_on(levels[:1], libero, 1000, 1099)


# --------------------------------------------------------------------------------------------- GPU decision
def _gpu_file(folder: Path, base_ms: float, name: str = "smolvla_gpu_teslat4_colab.json") -> Path:
    def row(batch: int, value: float) -> dict[str, Any]:
        samples = [value + 0.05 * i for i in range(MIN_REPEATS)]
        return {
            "batch_size": batch,
            "status": "ok",
            "median_ms": float(np.median(samples)),
            "samples_ms": samples,
        }

    document = {
        "format": V2_FORMAT,
        "platform": "colab",
        "created_utc": "2026-10-28T09:00:00+00:00",
        "device": {"device": "cuda", "gpu": "Tesla T4"},
        "configurations": [
            {
                "name": "fp16",
                "precision": "fp16",
                "status": "ok",
                "numerics": {"valid": True, "ratio": 0.01},
                "results": [row(1, base_ms), row(2, 1.2 * base_ms), row(4, 1.6 * base_ms)],
            },
            {"name": "compiled", "status": "error", "error": "no triton", "results": []},
        ],
    }
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _decision_spec(folder: Path, robots: int) -> p5.GpuDecisionSpec:
    spec = p5.GpuDecisionSpec(measurements=str(folder), output=str(folder / "decision.json"))
    spec.latency_slo = cap.LatencySlo(quantile=0.99, max_ms=150.0, robots=robots)
    spec.fleet = cap.FleetSpec(max_robots=4, replications=1, horizon_s=200.0, window_start_s=20.0)
    spec.wlan_mbit_s = [100.0]
    spec.bootstrap = 0
    return spec


def test_gpu_decision_without_files(tmp_path: Path) -> None:
    config = p5.Phase5Config(gpu_decision=_decision_spec(tmp_path / "empty", robots=2))
    document = p5.gpu_decision_document(config)
    assert document["files"] == []
    assert document["verdict"].startswith("kein Urteil")


def test_gpu_decision_rejects_the_cpu_test_file(tmp_path: Path) -> None:
    folder = tmp_path / "gpu"
    folder.mkdir()
    (folder / "cpu.json").write_text(CPU_TEST_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    document = p5.gpu_decision_document(p5.Phase5Config(gpu_decision=_decision_spec(folder, robots=2)))
    assert document["files"][0]["valid"] is False
    assert document["files"][0]["reason"] == "not measured on a GPU"
    assert document["verdict"].startswith("kein Urteil")


def test_gpu_decision_fast_gpu_says_yes(tmp_path: Path) -> None:
    folder = tmp_path / "gpu"
    _gpu_file(folder, base_ms=30.0)
    document = p5.gpu_decision_document(p5.Phase5Config(gpu_decision=_decision_spec(folder, robots=2)))
    decision = document["files"][0]
    assert decision["valid"] is True
    assert decision["verdict"] == "Ja"
    assert document["verdict"] == "Ja"
    assert document["consequence"] == "the business idea gets a second review"
    assert decision["best"]["latency_capacity"] >= 2
    assert [c["name"] for c in decision["configurations"]] == ["fp16", "compiled"]
    assert decision["secondary"]["batch_1_vs_assumption"]["fp16"] == pytest.approx(
        (30.0 + 0.05 * 49.5) / 220.0
    )
    assert set(decision["secondary"]["wlan"]) == {"100"}


def test_gpu_decision_slow_gpu_says_no(tmp_path: Path) -> None:
    folder = tmp_path / "gpu"
    _gpu_file(folder, base_ms=200.0)  # a single plan already takes 200 ms: p99 > 150 ms with one robot
    document = p5.gpu_decision_document(p5.Phase5Config(gpu_decision=_decision_spec(folder, robots=2)))
    assert document["files"][0]["verdict"] == "Nein"
    assert document["files"][0]["best"]["latency_capacity"] == 0
    assert document["verdict"] == "Nein"
    assert document["consequence"] == "the business idea is finally rejected"


def test_unreadable_measurement_file_gets_no_verdict(tmp_path: Path) -> None:
    folder = tmp_path / "gpu"
    folder.mkdir()
    (folder / "broken.json").write_text("{}", encoding="utf-8")
    decision = p5.file_decision(_decision_spec(folder, robots=2), folder / "broken.json")
    assert decision["verdict"] == "kein Urteil"
    assert decision["valid"] is False


def test_decision_config_uses_the_preregistered_settings(tmp_path: Path) -> None:
    config = p5.load_phase5_config("kortyn/experiments/phase5/analysis.yaml")
    spec = config.gpu_decision
    office = p5.decision_config(spec, tmp_path / "x.json", None)
    assert office.network.scenario == "buero_wlan"
    assert office.network.wlan_mbit_s is None
    assert office.network.request_bytes == 0
    assert office.criteria.latency_slo == cap.LatencySlo(quantile=0.99, max_ms=150.0, robots=10)
    assert office.gpu.configuration == cap.ALL_VALID
    assert office.fleet.replications == 5
    assert office.fleet.seed == 3000
    shared = p5.decision_config(spec, tmp_path / "x.json", 100.0)
    assert shared.network.wlan_mbit_s == 100.0
    assert shared.network.request_bytes == 292133


def test_main_reports_a_missing_gate(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "kortyn/experiments/phase5/analysis.yaml").read_text(encoding="utf-8"))
    config["gate"]["output"] = str(tmp_path / "missing_gate.json")
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert p5.main(["curves", "--config", str(path)]) == 2


def test_main_gpu_decision_without_measurements(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "kortyn/experiments/phase5/analysis.yaml").read_text(encoding="utf-8"))
    config["gpu_decision"]["measurements"] = str(tmp_path / "none")
    config["gpu_decision"]["output"] = str(tmp_path / "decision.json")
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert p5.main(["gpu-decision", "--config", str(path)]) == 0
    assert json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))["verdict"].startswith(
        "kein Urteil"
    )


def test_name_pair() -> None:
    assert p5.name_pair("K1_async_vs_sync25") == ("sync25", "async_0ms")
    assert p5.name_pair("K2_rtc_vs_async") == ("async_0ms", "rtc_kurz_0ms")
    assert replace(p5.GateSpec(), draws=500).draws == 500
