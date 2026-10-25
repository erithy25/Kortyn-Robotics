"""Phase 3 economics: hardware cost per robot, shared GPU Brain versus a computer in every robot.

Pre-registered method (``reports/phase3.md``, section 0.7): cost per robot = (GPU price + host share) /
capacity, per strategy and GPU profile; against it an onboard computer that runs the same model within the
latency budget, whose latency enters success through the dose-response curve. Depreciation over 36 months,
electricity at a stated price per kWh. Every price is an assumption with source and access date
(``kortyn/experiments/phase3/economics.yaml``).

Monthly cost of one robot:

* **Brain:** ``(GPU + host) / 36 / capacity + (GPU power x busy share + host power) x hours / capacity``
  (the Brain runs whenever the fleet runs; its GPU draws board power while busy and idle power otherwise).
* **Onboard:** ``computer / 36 + power x hours`` (one computer per robot, at its working power).

Supplement (not pre-registered, reported next to the pre-registered figures): with a shared Brain every robot
still needs a small client computer for its cameras, the network and the motor commands. ``client`` adds its
depreciation and electricity to the Brain cost per robot; the onboard computer does that work itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kortyn.config import ConfigError, decode

HOURS_PER_MONTH = 730.0  # 8 760 h / 12


@dataclass
class PriceSource:
    """Where an assumed price or figure comes from."""

    source: str  # citation (title, URL)
    date: str  # date of the listing or of access (ISO)
    note: str = ""


@dataclass
class GpuOption:
    """A GPU for the Brain: purchase price and power, mapped to a fleet GPU profile."""

    profile: str  # GPU key of the fleet config (capacity source)
    label: str
    price_eur: float
    board_power_w: float
    idle_power_w: float
    price_source: PriceSource
    condition: str = "used"  # new | used

    def __post_init__(self) -> None:
        if self.price_eur <= 0 or self.board_power_w <= 0 or not 0 <= self.idle_power_w <= self.board_power_w:
            raise ConfigError(f"GPU option {self.label!r}: need price > 0 and 0 <= idle <= board power")


@dataclass
class HostOption:
    """The PC that carries one GPU (without the GPU)."""

    label: str
    price_eur: float
    power_w: float  # average draw while the fleet runs
    price_source: PriceSource

    def __post_init__(self) -> None:
        if self.price_eur < 0 or self.power_w < 0:
            raise ConfigError(f"host {self.label!r}: price and power must be >= 0")


@dataclass
class OnboardOption:
    """A computer in every robot that runs the policy itself."""

    label: str
    price_eur: float
    power_w: float
    latency_ms: float  # one inference of the policy (planning latency of the robot)
    price_source: PriceSource
    latency_source: PriceSource
    runtime: str = ""  # how the published latency was achieved (e.g. "vla.cpp, GGUF")

    def __post_init__(self) -> None:
        if self.price_eur <= 0 or self.power_w <= 0 or self.latency_ms <= 0:
            raise ConfigError(f"onboard option {self.label!r}: price, power and latency must be > 0")


@dataclass
class ClientOption:
    """The small computer a robot needs next to a shared Brain (cameras, network, motor commands)."""

    label: str
    price_eur: float
    power_w: float
    price_source: PriceSource

    def __post_init__(self) -> None:
        if self.price_eur < 0 or self.power_w < 0:
            raise ConfigError(f"client {self.label!r}: price and power must be >= 0")


@dataclass
class EconomicsConfig:
    electricity_eur_per_kwh: float
    electricity_source: PriceSource
    depreciation_months: int = 36
    operating_hours_per_month: float = HOURS_PER_MONTH  # the fleet runs around the clock
    host: HostOption | None = None
    gpus: list[GpuOption] = field(default_factory=list)
    onboard: list[OnboardOption] = field(default_factory=list)
    client: ClientOption | None = None  # supplement: per-robot client next to the Brain

    def __post_init__(self) -> None:
        if self.electricity_eur_per_kwh < 0 or self.depreciation_months < 1:
            raise ConfigError("electricity price must be >= 0 and depreciation at least one month")
        if not 0 < self.operating_hours_per_month <= HOURS_PER_MONTH + 1e-9:
            raise ConfigError(f"operating hours per month must be in (0, {HOURS_PER_MONTH}]")
        if self.host is None or not self.gpus:
            raise ConfigError("economics needs a host and at least one GPU option")


def load_economics(path: Path | str) -> EconomicsConfig:
    import yaml

    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read economics assumptions {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    return decode(EconomicsConfig, data, str(source))


# --------------------------------------------------------------------------------------------- costs
@dataclass(frozen=True)
class BrainCost:
    gpu: str
    capacity: int
    hardware_eur: float  # GPU + host
    capex_per_robot_eur: float
    monthly_per_robot_eur: float  # depreciation + electricity
    electricity_per_robot_month_eur: float


@dataclass(frozen=True)
class OnboardCost:
    label: str
    capex_per_robot_eur: float
    monthly_per_robot_eur: float
    electricity_per_robot_month_eur: float
    latency_ms: float


def brain_cost(config: EconomicsConfig, gpu: GpuOption, capacity: int, busy_share: float) -> BrainCost | None:
    """Cost per robot of a Brain serving ``capacity`` robots (``None`` if it serves none)."""
    if capacity < 1:
        return None
    if not 0.0 <= busy_share <= 1.0:
        raise ConfigError(f"busy share must be in [0, 1], got {busy_share}")
    assert config.host is not None
    hardware = gpu.price_eur + config.host.price_eur
    gpu_power = gpu.board_power_w * busy_share + gpu.idle_power_w * (1.0 - busy_share)
    kwh = (gpu_power + config.host.power_w) / 1000.0 * config.operating_hours_per_month
    electricity = kwh * config.electricity_eur_per_kwh / capacity
    depreciation = hardware / config.depreciation_months / capacity
    return BrainCost(
        gpu=gpu.label,
        capacity=capacity,
        hardware_eur=hardware,
        capex_per_robot_eur=hardware / capacity,
        monthly_per_robot_eur=depreciation + electricity,
        electricity_per_robot_month_eur=electricity,
    )


def onboard_cost(config: EconomicsConfig, option: OnboardOption) -> OnboardCost:
    kwh = option.power_w / 1000.0 * config.operating_hours_per_month
    electricity = kwh * config.electricity_eur_per_kwh
    return OnboardCost(
        label=option.label,
        capex_per_robot_eur=option.price_eur,
        monthly_per_robot_eur=option.price_eur / config.depreciation_months + electricity,
        electricity_per_robot_month_eur=electricity,
        latency_ms=option.latency_ms,
    )


@dataclass(frozen=True)
class ClientCost:
    label: str
    capex_eur: float
    monthly_eur: float  # depreciation + electricity of one client
    electricity_month_eur: float


def client_cost(config: EconomicsConfig) -> ClientCost | None:
    """Cost of the per-robot client next to a shared Brain (``None`` without a ``client`` entry)."""
    if config.client is None:
        return None
    kwh = config.client.power_w / 1000.0 * config.operating_hours_per_month
    electricity = kwh * config.electricity_eur_per_kwh
    return ClientCost(
        label=config.client.label,
        capex_eur=config.client.price_eur,
        monthly_eur=config.client.price_eur / config.depreciation_months + electricity,
        electricity_month_eur=electricity,
    )


def break_even_capacity(
    config: EconomicsConfig,
    gpu: GpuOption,
    option: OnboardOption,
    busy_share: float,
    with_client: bool = False,
) -> int:
    """Smallest fleet at which the shared Brain is cheaper per robot and month than the onboard computer.

    ``with_client`` adds the per-robot client (supplement); 0 means never (up to 10 000 robots).
    """
    if with_client and config.client is None:
        raise ConfigError("break-even with client: the economics config has no 'client' entry")
    target = onboard_cost(config, option).monthly_per_robot_eur
    client = client_cost(config) if with_client else None
    extra = 0.0 if client is None else client.monthly_eur
    for capacity in range(1, 10_001):
        cost = brain_cost(config, gpu, capacity, busy_share)
        if cost is not None and cost.monthly_per_robot_eur + extra < target:
            return capacity
    return 0


def cost_table_rows(
    config: EconomicsConfig, capacities: dict[tuple[str, str], tuple[int, float]]
) -> list[dict[str, Any]]:
    """Rows per (GPU option, strategy): capacity and busy share from stage 2 -> cost per robot."""
    rows = []
    for gpu in config.gpus:
        for (profile, strategy), (capacity, busy) in sorted(capacities.items()):
            if profile != gpu.profile:
                continue
            cost = brain_cost(config, gpu, capacity, busy)
            rows.append(
                {
                    "gpu": gpu.label,
                    "strategy": strategy,
                    "capacity": capacity,
                    "capex_per_robot_eur": None if cost is None else cost.capex_per_robot_eur,
                    "monthly_per_robot_eur": None if cost is None else cost.monthly_per_robot_eur,
                }
            )
    return rows
