"""Phase 3 economics: cost per robot of a shared Brain versus an onboard computer."""

from __future__ import annotations

from pathlib import Path

import pytest

from kortyn.config import ConfigError
from kortyn.economics import (
    ClientOption,
    EconomicsConfig,
    GpuOption,
    HostOption,
    OnboardOption,
    PriceSource,
    brain_cost,
    break_even_capacity,
    client_cost,
    cost_table_rows,
    load_economics,
    onboard_cost,
)

SOURCE = PriceSource(source="test", date="2026-10-25")


def config(**overrides: object) -> EconomicsConfig:
    settings: dict[str, object] = {
        "electricity_eur_per_kwh": 0.25,
        "electricity_source": SOURCE,
        "depreciation_months": 36,
        "operating_hours_per_month": 730.0,
        "host": HostOption(label="host", price_eur=1100.0, power_w=60.0, price_source=SOURCE),
        "gpus": [
            GpuOption(
                profile="rtx3090",
                label="RTX 3090",
                price_eur=1300.0,
                board_power_w=350.0,
                idle_power_w=30.0,
                price_source=SOURCE,
            )
        ],
        "onboard": [
            OnboardOption(
                label="Orin",
                price_eur=600.0,
                power_w=20.0,
                latency_ms=500.0,
                price_source=SOURCE,
                latency_source=SOURCE,
            )
        ],
    }
    settings.update(overrides)
    return EconomicsConfig(**settings)  # type: ignore[arg-type]


def test_brain_cost_per_robot() -> None:
    economics = config()
    cost = brain_cost(economics, economics.gpus[0], capacity=8, busy_share=0.5)
    assert cost is not None
    assert cost.hardware_eur == 2400.0
    assert cost.capex_per_robot_eur == pytest.approx(300.0)
    # electricity: (350 * 0.5 + 30 * 0.5 + 60) W = 250 W -> 182.5 kWh/month -> 45.625 EUR / 8 robots
    assert cost.electricity_per_robot_month_eur == pytest.approx(45.625 / 8)
    assert cost.monthly_per_robot_eur == pytest.approx(2400.0 / 36 / 8 + 45.625 / 8)
    assert brain_cost(economics, economics.gpus[0], capacity=0, busy_share=0.5) is None
    with pytest.raises(ConfigError):
        brain_cost(economics, economics.gpus[0], capacity=2, busy_share=1.5)


def test_onboard_cost_and_break_even() -> None:
    economics = config()
    onboard = onboard_cost(economics, economics.onboard[0])
    assert onboard.capex_per_robot_eur == 600.0
    assert onboard.electricity_per_robot_month_eur == pytest.approx(20 / 1000 * 730 * 0.25)
    assert onboard.monthly_per_robot_eur == pytest.approx(600 / 36 + 3.65)
    capacity = break_even_capacity(economics, economics.gpus[0], economics.onboard[0], busy_share=0.9)
    cheaper = brain_cost(economics, economics.gpus[0], capacity, 0.9)
    dearer = brain_cost(economics, economics.gpus[0], capacity - 1, 0.9)
    assert cheaper is not None
    assert dearer is not None
    assert cheaper.monthly_per_robot_eur < onboard.monthly_per_robot_eur <= dearer.monthly_per_robot_eur


def test_client_supplement_raises_brain_cost_and_break_even() -> None:
    economics = config(
        client=ClientOption(label="Pi", price_eur=108.0, power_w=6.0, price_source=SOURCE),
    )
    client = client_cost(economics)
    assert client is not None
    assert client.electricity_month_eur == pytest.approx(6 / 1000 * 730 * 0.25)
    assert client.monthly_eur == pytest.approx(108 / 36 + 1.095)
    gpu, option = economics.gpus[0], economics.onboard[0]
    plain = break_even_capacity(economics, gpu, option, busy_share=0.9)
    supplemented = break_even_capacity(economics, gpu, option, busy_share=0.9, with_client=True)
    assert supplemented > plain > 0
    brain = brain_cost(economics, gpu, supplemented, 0.9)
    before = brain_cost(economics, gpu, supplemented - 1, 0.9)
    assert brain is not None
    assert before is not None
    target = onboard_cost(economics, option).monthly_per_robot_eur
    assert (
        brain.monthly_per_robot_eur + client.monthly_eur
        < target
        <= before.monthly_per_robot_eur + client.monthly_eur
    )
    # a client as expensive as the onboard computer: the Brain never pays off
    dear = config(client=ClientOption(label="dear", price_eur=600.0, power_w=20.0, price_source=SOURCE))
    assert break_even_capacity(dear, gpu, option, busy_share=0.9, with_client=True) == 0
    assert client_cost(config()) is None
    with pytest.raises(ConfigError):
        break_even_capacity(config(), gpu, option, busy_share=0.9, with_client=True)
    with pytest.raises(ConfigError):
        ClientOption(label="bad", price_eur=-1.0, power_w=6.0, price_source=SOURCE)


def test_cost_table_rows_map_capacities_to_gpu_options() -> None:
    economics = config()
    rows = cost_table_rows(
        economics, {("rtx3090", "a"): (2, 0.6), ("rtx3090", "d"): (0, 0.0), ("other", "a"): (5, 0.5)}
    )
    assert [(r["strategy"], r["capacity"]) for r in rows] == [("a", 2), ("d", 0)]
    assert rows[1]["monthly_per_robot_eur"] is None


def test_validation_and_loading(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        config(host=None)
    with pytest.raises(ConfigError):
        config(operating_hours_per_month=800.0)
    with pytest.raises(ConfigError):
        GpuOption(
            profile="g",
            label="g",
            price_eur=100.0,
            board_power_w=100.0,
            idle_power_w=200.0,
            price_source=SOURCE,
        )
    path = tmp_path / "economics.yaml"
    path.write_text(
        "electricity_eur_per_kwh: 0.2366\n"
        "electricity_source: {source: Destatis, date: '2026-09-30'}\n"
        "host: {label: PC, price_eur: 1130, power_w: 70, price_source: {source: PCGH, date: '2026-07-17'}}\n"
        "gpus:\n"
        "  - {profile: rtx3090, label: RTX 3090, price_eur: 1257, board_power_w: 350, idle_power_w: 25,\n"
        "     price_source: {source: pcprice.watch, date: '2026-10-25'}}\n",
        encoding="utf-8",
    )
    loaded = load_economics(path)
    assert loaded.gpus[0].price_eur == 1257
    assert loaded.depreciation_months == 36
    path.write_text("electricity_eur_per_kwh: 0.2\nunknown: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_economics(path)


def test_phase3_economics_assumptions_load() -> None:
    economics = load_economics(
        Path(__file__).resolve().parents[1] / "kortyn/experiments/phase3/economics.yaml"
    )
    assert economics.depreciation_months == 36
    assert {gpu.profile for gpu in economics.gpus} >= {"rtx3090", "rtx4090", "rtx3060ti_schaetzung"}
    assert all(option.price_source.date and option.latency_source.source for option in economics.onboard)
    assert economics.host is not None
    assert economics.client is not None
    assert economics.client.price_source.date
