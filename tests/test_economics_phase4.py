"""Phase-4 cost per robot at equal success (kortyn.economics_phase4) on synthetic capacity results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kortyn import dose_curve as dc
from kortyn import economics_phase4 as ep
from kortyn.config import ConfigError
from kortyn.economics import ClientOption, EconomicsConfig, GpuOption, HostOption, OnboardOption, PriceSource
from kortyn.fleet_predict import PredictionError

SOURCE = PriceSource(source="test", date="2026-10-26")
SPEC = dc.CurveSpec()


def config() -> EconomicsConfig:
    """No electricity, depreciation over 10 months: Brain = 200 / 10 / n + client 1, devices = price / 10."""
    return EconomicsConfig(
        electricity_eur_per_kwh=0.0,
        electricity_source=SOURCE,
        depreciation_months=10,
        host=HostOption(label="host", price_eur=100.0, power_w=50.0, price_source=SOURCE),
        gpus=[
            GpuOption(
                profile="gpu",
                label="GPU",
                price_eur=100.0,
                board_power_w=100.0,
                idle_power_w=10.0,
                price_source=SOURCE,
            )
        ],
        onboard=[
            OnboardOption("nano", 50.0, 10.0, 600.0, SOURCE, SOURCE),
            OnboardOption("thor", 500.0, 10.0, 50.0, SOURCE, SOURCE),
        ],
        client=ClientOption(label="pi", price_eur=10.0, power_w=5.0, price_source=SOURCE),
    )


def curve(x50: float = 600.0) -> dc.DoseCurve:
    return dc.DoseCurve(spec=SPEC, tasks={0: dc.TaskCurve(0.8, x50, 3.0)})


def size(robots: int, worst: float, criterion_1: bool = True, busy: float = 0.5) -> dict[str, Any]:
    return {"robots": robots, "worst": worst, "criterion_1": criterion_1, "busy_share": busy}


def entry(
    strategy: str,
    capacity: int,
    worsts: list[float],
    condition: str = "mit_rtc",
    batching: str = "realistisch",
) -> dict[str, Any]:
    gpu = "gpu_rtc" if condition == "mit_rtc" else "gpu"
    return {
        "network": "buero_wlan",
        "gpu": gpu,
        "batching": batching,
        "strategy": strategy,
        "condition": condition,
        "capacity": capacity,
        "capacity_conservative": capacity,
        "sizes": [size(n + 1, w) for n, w in enumerate(worsts)],
    }


def test_gpu_option_maps_the_rtc_profile_to_the_same_card() -> None:
    assert ep.gpu_option(config(), "gpu_rtc").label == "GPU"
    assert ep.gpu_option(config(), "gpu").label == "GPU"
    with pytest.raises(ConfigError):
        ep.gpu_option(config(), "other")


def test_brain_row_cost_includes_the_client() -> None:
    row = ep.brain_row(config(), entry("d", 4, [0.7] * 4), size(4, 0.65))
    assert row.robots == 4
    assert row.monthly_eur == pytest.approx(200.0 / 10 / 4 + 1.0)
    assert row.capex_per_robot_eur == pytest.approx(200.0 / 4 + 10.0)
    assert row.worst == pytest.approx(0.65)
    empty = ep.brain_row(config(), entry("d", 0, [0.7]), None)
    assert empty.robots == 0
    assert empty.monthly_eur is None


def test_matched_size_literal_and_monotone() -> None:
    sizes = [size(1, 0.7), size(2, 0.6), size(3, 0.39), size(4, 0.45), size(5, 0.5, criterion_1=False)]
    matched = ep.matched_size(sizes, 0.4)
    assert matched is not None
    assert matched["robots"] == 4  # the largest qualifying size (size 3 misses, size 5 breaks criterion 1)
    assert ep.matched_size_monotone(sizes, 0.4) == 2
    assert ep.matched_size(sizes, 0.9) is None
    assert ep.matched_size_monotone(sizes, 0.9) == 0
    assert ep.matched_size_monotone([size(2, 0.9)], 0.4) == 0  # no single-robot size


def test_device_success_uses_the_curve_and_the_rtc_overhead() -> None:
    curves = {"ohne_rtc": curve(), "mit_rtc": curve()}
    rows = {(row.label, row.condition): row for row in ep.device_rows(config(), curves, "mit_rtc", 0.1)}
    assert rows[("thor", "ohne_rtc")].latency_ms == pytest.approx(50.0)
    assert rows[("thor", "mit_rtc")].latency_ms == pytest.approx(55.0)
    # 55 ms falls into the second tick: effective latency 100 ms
    assert rows[("thor", "mit_rtc")].success == pytest.approx(float(curve().value_x(100.0)))
    assert rows[("thor", "ohne_rtc")].success == pytest.approx(float(curve().value_x(50.0)))
    assert rows[("nano", "ohne_rtc")].monthly_eur == pytest.approx(5.0)
    with pytest.raises(ConfigError):
        ep.device_rows(config(), curves, "mit_rtc", -0.1)


def test_best_strategy_by_capacity_then_tie_order() -> None:
    def row(strategy: str, robots: int, monthly: float | None = 1.0) -> ep.BrainRow:
        return ep.BrainRow("mit_rtc", "gpu_rtc", "realistisch", strategy, robots, 0.5, 0.5, monthly, 1.0)

    assert ep.best_strategy([row("a", 3), row("b", 5), row("d", 4)]) == row("b", 5)
    assert ep.best_strategy([row("a", 5), row("b", 5), row("e", 5), row("d", 5)]) == row("d", 5)
    assert ep.best_strategy([row("a", 5), row("e", 5)]) == row("e", 5)
    assert ep.best_strategy([row("d", 0, None)]) is None


def test_profile_rows_share_strategies_without_batching() -> None:
    a = ep.BrainRow("mit_rtc", "gpu_rtc", "realistisch", "a", 2, 0.5, 0.5, 1.0, 1.0)
    d = ep.BrainRow("mit_rtc", "gpu_rtc", "optimistisch", "d", 6, 0.5, 0.5, 1.0, 1.0)
    rows = ep.profile_rows([a, d], "optimistisch", ["a"])
    assert {(row.strategy, row.batching) for row in rows} == {("a", "optimistisch"), ("d", "optimistisch")}
    assert ep.profile_rows([a, d], "optimistisch", []) == [d]


def capacity_document(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"tolerance": 0.03, "miss_share_limit": 0.01, "configurations": list(entries)}


def test_document_brain_wins_against_the_cheapest_device_that_reaches_its_success() -> None:
    curves = {"ohne_rtc": curve(), "mit_rtc": curve()}
    thor = float(curve().value(50.0))
    nano = float(curve().value(600.0))
    # the worst robot stays above the nano's success up to 8 robots; capacity (3 pp) is 4
    worsts = [thor - 0.01, thor - 0.02, thor - 0.025, thor - 0.03, 0.6, 0.55, 0.5, nano + 0.001, nano - 0.05]
    capacity = capacity_document(
        entry("d", 4, worsts),
        entry("a", 2, worsts[:3]),
        entry("d", 4, worsts, condition="ohne_rtc"),
        entry("d", 6, worsts, batching="optimistisch"),
    )
    document = ep.economics_document(config(), capacity, curves, "mit_rtc", 0.0)
    decision = document["decisions"]["realistisch"]["mit_rtc"]
    assert decision["brain"]["strategy"] == "d"
    assert decision["brain"]["robots"] == 4
    # the nano does not reach the Brain's worst robot at capacity: the reference is the thor
    assert decision["reference"]["label"] == "thor"
    assert decision["reference_reaches_brain"]
    assert decision["cheaper"]  # 200 / 10 / 4 + 1 = 6 EUR < 50 EUR
    assert document["verdict"] == "trägt"
    matches = {(m["device"], m["condition"], m["batching"], m["strategy"]): m for m in document["matches"]}
    nano_match = matches[("nano", "mit_rtc", "realistisch", "d")]
    assert nano_match["brain"]["robots"] == 8
    assert nano_match["brain"]["monthly_eur"] == pytest.approx(200.0 / 10 / 8 + 1.0)
    assert nano_match["winner"] == "Brain"  # 3.50 EUR against 5 EUR
    assert nano_match["robots_monotone"] == 8
    # the thor's success is out of reach at every size: the device wins
    thor_match = matches[("thor", "mit_rtc", "realistisch", "d")]
    assert thor_match["brain"]["robots"] == 0
    assert thor_match["winner"] == "Gerät"
    assert document["devices"][0]["label"] == "nano"


def test_document_verdict_only_under_assumptions_and_not() -> None:
    curves = {"ohne_rtc": curve(), "mit_rtc": curve()}
    thor = float(curve().value(50.0))
    worsts = [thor - 0.01, thor - 0.02, thor - 0.025, thor - 0.03, thor - 0.031]
    # realistic: capacity 3 (< 4 robots); optimistic: capacity 5
    capacity = capacity_document(
        entry("d", 3, worsts),
        entry("d", 5, worsts, batching="optimistisch"),
    )
    document = ep.economics_document(config(), capacity, curves, "mit_rtc", 0.0)
    assert not document["decisions"]["realistisch"]["mit_rtc"]["fleet_ok"]
    assert document["verdict"] == "trägt nur unter Annahmen"
    capacity = capacity_document(entry("d", 3, worsts), entry("d", 3, worsts, batching="optimistisch"))
    assert ep.economics_document(config(), capacity, curves, "mit_rtc", 0.0)["verdict"] == "trägt nicht"
    # a Brain that serves nobody
    capacity = capacity_document(entry("d", 0, [0.1]), entry("d", 0, [0.1], batching="optimistisch"))
    document = ep.economics_document(config(), capacity, curves, "mit_rtc", 0.0)
    assert document["verdict"] == "trägt nicht"
    assert "reason" in document["decisions"]["realistisch"]["mit_rtc"]


def test_document_errors() -> None:
    curves = {"ohne_rtc": curve(), "mit_rtc": curve()}
    with pytest.raises(PredictionError):
        ep.economics_document(config(), capacity_document(), curves, "mit_rtc", 0.0)
    with pytest.raises(PredictionError):
        ep.economics_document(config(), capacity_document(entry("d", 2, [0.7, 0.7])), curves, "other", 0.0)
    other_network = entry("d", 2, [0.7, 0.7])
    other_network["network"] = "wlan_c25"
    with pytest.raises(PredictionError):
        ep.economics_document(config(), capacity_document(other_network), curves, "mit_rtc", 0.0)


def test_cli_reports_missing_inputs(tmp_path: Path) -> None:
    overhead = tmp_path / "rtc.json"
    overhead.write_text(json.dumps({"applied_overhead": 0.1}), encoding="utf-8")
    code = ep.main(["--capacity", str(tmp_path / "missing.json"), "--rtc-overhead", str(overhead)])
    assert code == 2
