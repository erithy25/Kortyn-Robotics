"""Phase-4 report data and figures (kortyn.report_phase4) on synthetic results in the real file formats."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kortyn import dose_curve as dc
from kortyn import economics_phase4 as ep
from kortyn import fleet_capacity as fc
from kortyn import phase4_steps as p4
from kortyn import report_phase4 as rp
from kortyn.bench_codec import CodecResult
from kortyn.config import LiberoEnvConfig
from kortyn.economics import load_economics
from kortyn.fleet import FleetSimConfig, load_fleet_config

ROOT = Path(__file__).resolve().parents[1]
LIBERO = LiberoEnvConfig(task_ids=[8, 0], episodes_per_task=50)
SEEDS = range(1000, 1100)
SPEC = dc.CurveSpec()
GPU = "rtx3060ti_schaetzung"


def write_run(directory: Path, success: dict[int, bool]) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "episodes.jsonl").open("w", encoding="utf-8") as out:
        for seed, value in success.items():
            out.write(json.dumps({"seed": seed, "attempt": 0, "success": value, "steps": 300}) + "\n")
    return str(directory)


def task_curves(x50: float) -> dict[int, dc.TaskCurve]:
    return {8: dc.TaskCurve(0.55, x50, 3.0), 0: dc.TaskCurve(0.45, 0.9 * x50, 4.0)}


def bootstrap(x50: float) -> dc.BootstrapFit:
    draws = tuple(dc.DoseCurve(spec=SPEC, tasks=task_curves(x50 * f)) for f in (0.85, 0.95, 1.0, 1.05, 1.15))
    return dc.BootstrapFit(point=dc.DoseCurve(spec=SPEC, tasks=task_curves(x50)), draws=draws)


def condition(tmp_path: Path, name: str, x50: float, rng: np.random.Generator) -> list[p4.LevelSpec]:
    levels = []
    for latency in (0.0, 100.0, 200.0, 300.0, 500.0, 750.0, 1000.0):
        x = float(dc.effective_latency_ms(latency))
        outcome = {
            seed: bool(rng.random() < float(task_curves(x50)[8 if seed < 1050 else 0](x))) for seed in SEEDS
        }
        levels.append(p4.LevelSpec(latency, write_run(tmp_path / name / f"{latency:.0f}", outcome)))
    return levels


def curves_document(tmp_path: Path) -> tuple[dict[str, Any], str]:
    """The structure of ``phase4_steps.curves_document`` with known curves instead of fits."""
    rng = np.random.default_rng(11)
    levels = {
        "ohne_rtc": condition(tmp_path, "ohne", 450.0, rng),
        "mit_rtc": condition(tmp_path, "mit", 900.0, rng),
    }
    fits = {"ohne_rtc": bootstrap(450.0), "mit_rtc": bootstrap(900.0)}
    sync = write_run(tmp_path / "sync25", {s: bool(rng.random() < 0.5) for s in SEEDS})
    document = {
        "spec": asdict(SPEC),
        "bootstrap": 5,
        "seed": 1,
        "tolerance": 0.03,
        "seeds": len(SEEDS),
        "tasks": [8, 0],
        "conditions": {
            name: p4.condition_summary(p4.load_condition(levels[name], LIBERO), fits[name], 0.03)
            for name in levels
        },
        "rtc_effect": p4.rtc_effect(fits["ohne_rtc"], fits["mit_rtc"], 0.03),
        "r_rtc": p4.r_rtc_check(Path(levels["mit_rtc"][0].run), Path(sync)),
    }
    return document, sync


def size(robots: int, worst: float, single: float, busy: float, medium: float | None = None) -> fc.SizeResult:
    loss = single - worst
    return fc.SizeResult(
        robots=robots,
        replications=5,
        requests=1000,
        misses=0 if busy < 0.95 else 50,
        miss_share=0.0 if busy < 0.95 else 0.05,
        criterion_1=busy < 0.95,
        admitted_mean=float(robots),
        mean_batch=1.0 + robots / 10,
        busy_share=busy,
        fleet_mean=worst + 0.01,
        worst=worst,
        single=single,
        loss=loss,
        loss_upper=loss + 0.01,
        criterion_2=loss <= 0.03,
        criterion_2_conservative=loss + 0.01 <= 0.03,
        medium_busy_share=medium,
    )


def capacity_result(
    network: str,
    gpu: str,
    batching: str,
    strategy: str,
    condition_name: str,
    robots: int,
    medium: bool = False,
) -> fc.CapacityResult:
    single = 0.45
    sizes = [
        size(n, single - 0.003 * (n - 1), single, min(0.99, 0.1 * n), 0.07 * n if medium else None)
        for n in range(1, robots + 1)
    ]
    return fc.CapacityResult(
        key=(network, gpu, batching, strategy),
        condition=condition_name,
        capacity=fc.capacity_of(sizes),
        capacity_conservative=fc.capacity_of(sizes, conservative=True),
        sizes=sizes,
    )


def fleet_config(tmp_path: Path) -> FleetSimConfig:
    path = tmp_path / "fleet.yaml"
    path.write_text(
        f"gpus: {{{GPU}: {{s1_ms: 220}}, {GPU}_rtc: {{s1_ms: 231}}}}\n"
        "batching: {realistisch: {beta: 0.15}, optimistisch: {beta: 0.05}}\n"
        "strategies:\n"
        "  a: {scheduler: fifo}\n"
        "  d: {scheduler: edf_batch, max_batch: 16}\n"
        "networks:\n"
        "  buero_wlan: {scenario: buero_wlan, margin_ms: 15}\n"
        "  wlan_25: {scenario: buero_wlan, margin_ms: 15, wlan_mbit_s: 25, request_bytes: 300000, "
        "response_bytes: 1424}\n"
        "  wlan_400: {scenario: buero_wlan, margin_ms: 15, wlan_mbit_s: 400, request_bytes: 300000, "
        "response_bytes: 1424}\n"
        "episode_lengths: {values: [100]}\n",
        encoding="utf-8",
    )
    return load_fleet_config(path)


def capacity_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    results = []
    for name, gpu in (("ohne_rtc", GPU), ("mit_rtc", f"{GPU}_rtc")):
        results.append(capacity_result("buero_wlan", gpu, "realistisch", "a", name, 4))
        results.append(capacity_result("buero_wlan", gpu, "realistisch", "d", name, 8))
        results.append(capacity_result("buero_wlan", gpu, "optimistisch", "d", name, 11))
    wlan = [
        capacity_result("wlan_25", GPU, "realistisch", "d", "ohne_rtc", 3, medium=True),
        capacity_result("wlan_400", GPU, "realistisch", "d", "ohne_rtc", 8, medium=True),
        capacity_result("wlan_25", f"{GPU}_rtc", "realistisch", "d", "mit_rtc", 3, medium=True),
    ]
    return fc.result_json(results, 0.03), fc.result_json(wlan, 0.03)


def codec_document() -> dict[str, Any]:
    results = [
        CodecResult("raw", "raw", None, 10, 10, 10, 777_600.0, 777_600.0, 4.98, 0.1, 0.1, 0.2),
        CodecResult("png_1", "png", 1, 10, 10, 10, 350_000.0, 380_000.0, 2.24, 9.0, 3.0, 12.0),
        CodecResult("zstd_3", "zstd", 3, 10, 10, 10, 420_000.0, 450_000.0, 2.69, 2.0, 1.0, 3.0),
    ]
    return {
        "experiment": "x.yaml",
        "episodes": 2,
        "capture_ticks": [0, 50],
        "requests": 10,
        "frame_shape": {"observation.images.image": [2, 360, 360, 3]},
        "period_s": 1.25,
        "max_codec_ms": 25.0,
        "tie_share": 0.02,
        "repeats": 3,
        "standard": "png_1",
        "candidates": [asdict(r) | {"lossless": r.lossless} for r in results],
    }


def rtc_document() -> dict[str, Any]:
    return {
        "pairs": 30,
        "delay": 5,
        "horizon": 25,
        "plain_median_ms": 1000.0,
        "rtc_median_ms": 1050.0,
        "relative_overhead": 0.05,
        "relative_overhead_ci": [0.03, 0.07],
        "applied_overhead": 0.05,
    }


def validation_document() -> dict[str, Any]:
    case = {
        "episodes": 60,
        "measured": 0.42,
        "predicted": 0.45,
        "difference": -0.03,
        "interval": [-0.16, 0.10],
        "passes": True,
        "predicted_own": 0.44,
        "difference_own": -0.02,
        "interval_own": [-0.15, 0.11],
    }
    return {
        "condition": "mit_rtc",
        "confidence": 0.975,
        "cases": {"d_n2": case, "d_n4": case},
        "passes": True,
    }


def retest_document() -> dict[str, Any]:
    case = {
        "episodes": 50,
        "measured": 0.4,
        "predicted_own": 0.42,
        "predicted_stage1": 0.43,
        "predicted_phase3_own": 0.47,
        "predicted_phase3_stage1": 0.48,
    }
    pooled = {
        name: {"bias": bias, "interval": [bias - 0.05, bias + 0.05]}
        for name, bias in (
            ("own", -0.02),
            ("stage1", -0.03),
            ("phase3_own", -0.074),
            ("phase3_stage1", -0.065),
        )
    }
    return {
        "informative_cases": ["a_n2"],
        "episodes": 50,
        "cases": {"a_n2": case},
        "pooled": pooled,
        "verdict": "verschwunden",
        "verdict_stage1": "verkleinert",
    }


def fresh_document() -> dict[str, Any]:
    comparison = {
        "pairs": 50,
        "rate_a": 0.44,
        "rate_b": 0.3,
        "only_a": 12,
        "only_b": 5,
        "difference": -0.14,
        "interval": [-0.28, 0.0],
        "p_value": 0.14,
    }
    return {
        "episodes": 50,
        "seeds": [1000, 1049],
        "success": 0.3,
        "comparisons": {"block_a": comparison, "phase3": {**comparison, "rate_a": 0.26, "difference": 0.04}},
    }


def sensitivity_document() -> dict[str, Any]:
    entry = {
        "families": {"0": "loglogistic", "8": "loglogistic"},
        "lstar_ms": 200.0,
        "censored": False,
        "success_0ms": 0.45,
        "success_at_lstar": 0.44,
        "curve": {"0.0": 0.45, "250.0": 0.42, "300.0": 0.39, "500.0": 0.27, "750.0": 0.17, "1000.0": 0.05},
        "retest_bias": -0.196,
        "retest_interval": [-0.268, -0.117],
        "retest_verdict": "nicht erklärt",
    }
    return {
        "phase3_levels": [{"latency_ms": 0.0, "run": "x"}, {"latency_ms": 500.0, "run": "y"}],
        "levels": [],
        "vorab": entry,
        "gepoolt": {**entry, "lstar_ms": 250.0, "retest_bias": -0.17},
    }


@pytest.fixture
def inputs(tmp_path: Path) -> rp.Inputs:
    curves, sync = curves_document(tmp_path)
    capacity, wlan = capacity_documents()
    fleet = fleet_config(tmp_path)
    economics_config = load_economics(ROOT / "kortyn/experiments/phase3/economics.yaml")
    fits = {name: bootstrap(x50).point for name, x50 in (("ohne_rtc", 450.0), ("mit_rtc", 900.0))}
    economics = ep.economics_document(economics_config, capacity, fits, "mit_rtc", 0.05, ["a"])
    analysis = p4.AnalysisConfig(conditions={"ohne_rtc": [p4.LevelSpec(0.0, "x")]}, reference_sync=sync)
    return rp.Inputs(
        analysis=analysis,
        fleet=fleet,
        economics_config=economics_config,
        codec=codec_document(),
        rtc=rtc_document(),
        curves=curves,
        retest=retest_document(),
        sensitivity=sensitivity_document(),
        fresh=fresh_document(),
        capacity=capacity,
        wlan=wlan,
        validation=validation_document(),
        economics=economics,
    )


def test_build_writes_every_section_and_figure(inputs: rp.Inputs, tmp_path: Path) -> None:
    out = tmp_path / "reports"
    text = "\n".join(rp.build(inputs, out))
    for heading in (
        "## Schritt 1",
        "## Schritt 2",
        "## Schritt 3",
        "### Re-Test",
        "### Sensitivität",
        "### Reihenfolge der Läufe",
        "## Schritt 4",
        "### WLAN",
        "### Validierung",
        "## Schritt 5",
        "**Urteil (Abschnitt 8):",
    ):
        assert heading in text
    assert "**png_1**" in text
    assert "R-RTC" in text
    for figure in ("phase4_kurven.png", "phase4_kapazitaet.png", "phase4_wlan.png", "phase4_kosten.png"):
        assert (out / figure).stat().st_size > 10_000
    assert "Netz ab n = 4" in text  # WLAN 25 Mbit/s: capacity 3 < 8 unlimited
    assert "Quelle" in text


def test_build_skips_missing_inputs(inputs: rp.Inputs, tmp_path: Path) -> None:
    empty = rp.Inputs(
        analysis=inputs.analysis,
        fleet=inputs.fleet,
        economics_config=inputs.economics_config,
        codec=None,
        rtc=None,
        curves=None,
        retest=None,
        sensitivity=None,
        fresh=None,
        capacity=None,
        wlan=inputs.wlan,
        validation=None,
        economics=None,
    )
    lines = rp.build(empty, tmp_path / "reports")
    assert lines[0].startswith("# Phase 4")
    assert not any(line.startswith("## ") for line in lines)


def test_wlan_rows_mark_the_network_bottleneck(inputs: rp.Inputs) -> None:
    assert inputs.wlan is not None
    assert inputs.capacity is not None
    rows = rp.wlan_rows(inputs.wlan, inputs.capacity, inputs.fleet)
    by_key = {(row.condition, row.rate_mbit_s): row for row in rows}
    slow = by_key[("ohne_rtc", 25.0)]
    assert slow.capacity == 3
    assert slow.capacity_unlimited == 8
    assert slow.network_limits
    assert slow.medium_busy == pytest.approx(0.21)
    assert not by_key[("ohne_rtc", 400.0)].network_limits
    broken = json.loads(json.dumps(inputs.wlan))
    broken["configurations"][0]["network"] = "buero_wlan"  # no medium in the config
    with pytest.raises(rp.ReportError):
        rp.wlan_rows(broken, inputs.capacity, inputs.fleet)


def test_entries_share_unbatched_strategies_and_conditions_map_to_one_gpu(inputs: rp.Inputs) -> None:
    assert inputs.capacity is not None
    found = rp.entries_by_key(inputs.capacity, inputs.fleet)
    assert found[("buero_wlan", GPU, "optimistisch", "a")] is found[("buero_wlan", GPU, "realistisch", "a")]
    assert rp.condition_gpus(inputs.capacity) == {"ohne_rtc": GPU, "mit_rtc": f"{GPU}_rtc"}
    mixed = json.loads(json.dumps(inputs.capacity))
    mixed["configurations"][0]["gpu"] = "other"
    with pytest.raises(rp.ReportError):
        rp.condition_gpus(mixed)


def test_cost_curve_stops_at_the_first_size_breaking_criterion_1(inputs: rp.Inputs) -> None:
    assert inputs.capacity is not None
    entry = next(
        e
        for e in inputs.capacity["configurations"]
        if e["strategy"] == "d" and e["batching"] == "optimistisch"
    )
    sizes, costs = rp.cost_curve(inputs.economics_config, entry)
    assert sizes == list(range(1, 10))  # busy share 0.1 n reaches 0.95 at n = 10
    assert all(later < earlier for earlier, later in itertools.pairwise(costs))


def test_formatting_helpers() -> None:
    assert rp.ms(1250.0) == "1 250 ms"
    assert rp.ms(None) == "–"
    assert rp.num(2.345, 2) == "2,35"
    assert rp.kbytes(777_600.0) == "778 kB"
    assert rp.lstar_text({"point_ms": 1250.0, "censored": True}, 1250.0) == "≥ 1 250 ms"
    assert rp.lstar_text({"point_ms": 300.0, "censored": False}, 1250.0) == "300 ms"
    assert rp.bound_text(1250.0, 1250.0).startswith("≥")
    assert rp.table(["a", "b"], [["1", "2"]]) == ["| a | b |", "|---|---|", "| 1 | 2 |"]


def test_load_json_and_cli_errors(tmp_path: Path) -> None:
    assert rp.load_json(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(rp.ReportError):
        rp.load_json(bad)
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(rp.ReportError):
        rp.load_json(bad)
    assert rp.main(["--analysis", str(tmp_path / "missing.yaml"), "--out-dir", str(tmp_path)]) == 2
