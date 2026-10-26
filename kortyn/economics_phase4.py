"""Phase 4, step 5: cost per robot at equal success - one Brain against a computer in every robot.

Pre-registered in reports/phase4_preregistration.md, section 8. Prices, power and the onboard latencies come
from the phase-3 economics file (every figure an assumption with source and date); the success of every
option from the phase-4 curves:

* **computer in the robot** (Jetson Orin Nano Super, AGX Orin, Thor): its latency is its inference time (no
  network); with RTC the inference time grows by the measured RTC overhead ``r``; success = pooled point curve
  of the condition (without / with RTC) at that latency;
* **Brain** (RTX 3060 Ti with host PC, plus a Raspberry Pi 5 client per robot), equal success: for every
  device, RTC setting, strategy and batching profile the largest fleet size that keeps criterion 1 (at most
  1 % misses) and whose worst robot reaches the device's success. The Brain wins where its cost per robot at
  that size lies below the device's; if no size reaches the device's success, the device wins. The
  pre-registered text names the largest such size; ``robots_monotone`` adds the largest size up to which
  every size qualifies (it differs only where the worst-robot prediction is not monotone in the fleet size);
* **cost at capacity:** the cost per robot at the capacity of section 7.3 (3-pp criterion);
* **decision** (realistic batching, with RTC, best strategy = largest capacity, ties in the order d, e, b,
  a as in section 7.5): the Brain runs at its capacity ``K`` with worst-robot success ``S``. It "trägt" if
  its cost per robot lies below the cheapest device whose success reaches ``S`` (if none does: the most
  successful device) and ``K >= 4``; "trägt nur unter Annahmen" if that holds in the optimistic profile
  only; else "trägt nicht".

Strategies without batching run under the first batching profile of the fleet config only (their service time
is the same in every profile); the decision counts them in every profile.

    python -m kortyn.economics_phase4 --capacity results/phase4/fleet/capacity.json \\
        --rtc-overhead results/phase4/rtc_overhead.json --out results/phase4/economics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from kortyn import dose_curve as dc
from kortyn.config import ConfigError
from kortyn.economics import (
    EconomicsConfig,
    GpuOption,
    brain_cost,
    client_cost,
    load_economics,
    onboard_cost,
)
from kortyn.fleet_predict import PredictionError

LOG = logging.getLogger("kortyn.economics_phase4")

MIN_FLEET = 4  # decision rule: the Brain must serve at least four robots within the 3-pp tolerance
RTC_SUFFIX = "_rtc"  # GPU profile of the fleet model with the RTC overhead in its service time
TIE_ORDER = ("d", "e", "b", "a")  # best strategy at equal capacity (section 7.5)
DECISION_PROFILES = ("realistisch", "optimistisch")
EPS = 1e-12


@dataclass(frozen=True)
class DeviceRow:
    label: str
    condition: str
    latency_ms: float  # inference time, with RTC times (1 + r)
    success: float
    monthly_eur: float
    capex_eur: float


@dataclass(frozen=True)
class BrainRow:
    condition: str
    gpu: str
    batching: str
    strategy: str
    robots: int  # 0: no size qualifies
    worst: float | None
    busy_share: float | None
    monthly_eur: float | None  # per robot and month, client included
    capex_per_robot_eur: float | None  # GPU + host share, client included


@dataclass(frozen=True)
class MatchRow:
    device: str
    condition: str
    batching: str
    strategy: str
    device_success: float
    device_monthly_eur: float
    brain: BrainRow
    robots_monotone: int
    winner: str  # "Brain" | "Gerät"


def gpu_option(config: EconomicsConfig, profile: str) -> GpuOption:
    """The GPU behind a fleet profile; the RTC profile is the same card."""
    base = profile.removesuffix(RTC_SUFFIX)
    for gpu in config.gpus:
        if gpu.profile == base:
            return gpu
    raise ConfigError(f"no GPU option for fleet profile {profile!r} in the economics config")


def brain_row(config: EconomicsConfig, entry: Mapping[str, Any], size: Mapping[str, Any] | None) -> BrainRow:
    """Cost per robot of the capacity result ``entry`` run at ``size`` (``None``: no size qualifies)."""
    base = BrainRow(
        entry["condition"], entry["gpu"], entry["batching"], entry["strategy"], 0, None, None, None, None
    )
    if size is None:
        return base
    robots = int(size["robots"])
    busy = float(size["busy_share"])
    cost = brain_cost(config, gpu_option(config, entry["gpu"]), robots, busy)
    if cost is None:
        return base
    client = client_cost(config)
    return replace(
        base,
        robots=robots,
        worst=float(size["worst"]),
        busy_share=busy,
        monthly_eur=cost.monthly_per_robot_eur + (0.0 if client is None else client.monthly_eur),
        capex_per_robot_eur=cost.capex_per_robot_eur + (0.0 if client is None else client.capex_eur),
    )


def _qualifies(size: Mapping[str, Any], target: float) -> bool:
    return bool(size["criterion_1"]) and float(size["worst"]) >= target - EPS


def matched_size(sizes: Sequence[Mapping[str, Any]], target: float) -> Mapping[str, Any] | None:
    """Pre-registered: the largest size that keeps criterion 1 and whose worst robot reaches ``target``."""
    qualifying = [size for size in sizes if int(size["robots"]) >= 1 and _qualifies(size, target)]
    return max(qualifying, key=lambda size: int(size["robots"]), default=None)


def matched_size_monotone(sizes: Sequence[Mapping[str, Any]], target: float) -> int:
    """Largest size up to which every size from 1 on qualifies (0 if size 1 does not)."""
    by_robots = {int(size["robots"]): size for size in sizes}
    robots = 0
    while robots + 1 in by_robots and _qualifies(by_robots[robots + 1], target):
        robots += 1
    return robots


def at_capacity(entry: Mapping[str, Any]) -> Mapping[str, Any] | None:
    capacity = int(entry["capacity"])
    return (
        next((size for size in entry["sizes"] if int(size["robots"]) == capacity), None) if capacity else None
    )


def device_rows(
    config: EconomicsConfig, curves: Mapping[str, dc.DoseCurve], rtc_condition: str, overhead: float
) -> list[DeviceRow]:
    """Every onboard computer under every condition; with RTC its inference takes ``1 + overhead`` as long."""
    if overhead < 0:
        raise ConfigError(f"the RTC overhead must be >= 0, got {overhead}")
    rows = []
    for option in config.onboard:
        cost = onboard_cost(config, option)
        for condition, curve in curves.items():
            latency = option.latency_ms * (1.0 + overhead if condition == rtc_condition else 1.0)
            rows.append(
                DeviceRow(
                    label=option.label,
                    condition=condition,
                    latency_ms=latency,
                    success=float(curve.value(latency)),
                    monthly_eur=cost.monthly_per_robot_eur,
                    capex_eur=cost.capex_per_robot_eur,
                )
            )
    return rows


def _tie_rank(strategy: str) -> int:
    return TIE_ORDER.index(strategy) if strategy in TIE_ORDER else len(TIE_ORDER)


def best_strategy(rows: Sequence[BrainRow]) -> BrainRow | None:
    """Largest capacity; at equal capacity the first in the order d, e, b, a (section 7.5)."""
    serving = [row for row in rows if row.monthly_eur is not None]
    if not serving:
        return None
    return min(serving, key=lambda row: (-row.robots, _tie_rank(row.strategy), row.strategy))


def profile_rows(rows: Sequence[BrainRow], batching: str, batch_free: Collection[str]) -> list[BrainRow]:
    """The capacity rows of one batching profile; strategies without batching count in every profile."""
    own = [row for row in rows if row.batching == batching]
    present = {(row.condition, row.gpu, row.strategy) for row in own}
    shared = [
        replace(row, batching=batching)
        for row in rows
        if row.strategy in batch_free
        and row.batching != batching
        and (row.condition, row.gpu, row.strategy) not in present
    ]
    return own + shared


def decision(
    capacity_rows: Sequence[BrainRow], devices: Sequence[DeviceRow], condition: str
) -> dict[str, Any]:
    """Pre-registered rule for one batching profile and condition (see the module docstring)."""
    best = best_strategy([row for row in capacity_rows if row.condition == condition])
    if best is None or best.worst is None or best.monthly_eur is None:
        return {"verdict": False, "reason": "keine Strategie bedient einen Roboter innerhalb der Toleranz"}
    pool = [device for device in devices if device.condition == condition]
    if not pool:
        raise PredictionError(f"no onboard computer evaluated under condition {condition!r}")
    reaching = [device for device in pool if device.success >= best.worst - EPS]
    if reaching:
        reference = min(reaching, key=lambda device: (device.monthly_eur, device.label))
    else:
        reference = max(pool, key=lambda device: (device.success, -device.monthly_eur))
    cheaper = best.monthly_eur < reference.monthly_eur
    fleet_ok = best.robots >= MIN_FLEET
    return {
        "verdict": cheaper and fleet_ok,
        "brain": asdict(best),
        "reference": asdict(reference),
        "reference_reaches_brain": bool(reaching),
        "cheaper": cheaper,
        "fleet_ok": fleet_ok,
    }


def economics_document(
    config: EconomicsConfig,
    capacity: Mapping[str, Any],
    curves: Mapping[str, dc.DoseCurve],
    rtc_condition: str,
    overhead: float,
    batch_free: Collection[str] = ("a",),
    network: str = "buero_wlan",
) -> dict[str, Any]:
    """Equal-success comparison, cost at capacity and decision from a kortyn.fleet_capacity result."""
    if rtc_condition not in curves:
        raise PredictionError(f"no curve for the RTC condition {rtc_condition!r}")
    devices = device_rows(config, curves, rtc_condition, overhead)
    entries = [
        entry
        for entry in capacity["configurations"]
        if entry["condition"] in curves and entry.get("network", network) == network
    ]
    if not entries:
        raise PredictionError(
            f"the capacity results hold no configuration of {sorted(curves)} on {network!r}"
        )
    capacity_rows = [brain_row(config, entry, at_capacity(entry)) for entry in entries]
    matches = []
    for device in devices:
        for entry in entries:
            if entry["condition"] != device.condition:
                continue
            row = brain_row(config, entry, matched_size(entry["sizes"], device.success))
            winner = (
                "Brain" if row.monthly_eur is not None and row.monthly_eur < device.monthly_eur else "Gerät"
            )
            matches.append(
                MatchRow(
                    device=device.label,
                    condition=device.condition,
                    batching=entry["batching"],
                    strategy=entry["strategy"],
                    device_success=device.success,
                    device_monthly_eur=device.monthly_eur,
                    brain=row,
                    robots_monotone=matched_size_monotone(entry["sizes"], device.success),
                    winner=winner,
                )
            )
    decisions = {
        batching: {
            condition: decision(profile_rows(capacity_rows, batching, batch_free), devices, condition)
            for condition in curves
        }
        for batching in DECISION_PROFILES
    }
    if decisions["realistisch"][rtc_condition]["verdict"]:
        verdict = "trägt"
    elif decisions["optimistisch"][rtc_condition]["verdict"]:
        verdict = "trägt nur unter Annahmen"
    else:
        verdict = "trägt nicht"
    client = client_cost(config)
    return {
        "network": network,
        "rtc_condition": rtc_condition,
        "rtc_overhead": overhead,
        "client_monthly_eur": None if client is None else client.monthly_eur,
        "devices": [asdict(device) for device in devices],
        "capacity": [asdict(row) for row in capacity_rows],
        "matches": [asdict(match) for match in matches],
        "decisions": decisions,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    from kortyn.fleet import load_fleet_config
    from kortyn.phase4_steps import libero_layout, load_analysis_config, load_condition

    parser = argparse.ArgumentParser(description="Phase-4 cost per robot at equal success.")
    parser.add_argument("--analysis", default="kortyn/experiments/phase4/analysis.yaml")
    parser.add_argument("--economics", default="kortyn/experiments/phase3/economics.yaml")
    parser.add_argument(
        "--fleet", default="kortyn/experiments/phase4/fleet.yaml", help="strategy definitions"
    )
    parser.add_argument("--capacity", required=True, help="output of kortyn.fleet_capacity")
    parser.add_argument("--rtc-overhead", required=True, help="output of kortyn.bench_rtc")
    parser.add_argument("--network", default="buero_wlan")
    parser.add_argument("--out", default="results/phase4/economics.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        analysis = load_analysis_config(args.analysis)
        economics = load_economics(args.economics)
        fleet = load_fleet_config(args.fleet)
        batch_free = [name for name, strategy in fleet.strategies.items() if not strategy.batches]
        capacity = json.loads(Path(args.capacity).read_text(encoding="utf-8"))
        overhead = float(json.loads(Path(args.rtc_overhead).read_text(encoding="utf-8"))["applied_overhead"])
        libero = libero_layout(analysis.experiment)
        curves = {
            name: dc.fit_dose_curve(load_condition(analysis.conditions[name], libero), analysis.spec)
            for name in (analysis.without, analysis.with_rtc)
        }
        document = economics_document(
            economics, capacity, curves, analysis.with_rtc, overhead, batch_free, args.network
        )
    except (ConfigError, PredictionError, OSError, KeyError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    LOG.info("verdict: %s; written %s", document["verdict"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
