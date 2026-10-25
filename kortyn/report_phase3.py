"""Phase-3 report data: ``reports/phase3_data.md`` and the figures ``reports/phase3_*.png``.

    python -m kortyn.report_phase3

Inputs (sections whose inputs are missing are skipped, so the report can be built while runs continue):

* pilot (step 1.1/1.2): ``results/phase3/pilot/smolvla_libero/{sync25,sync50}`` and the ``lerobot-eval``
  summaries ``results/phase3/pilot/lerobot_eval/*/summary.json``;
* main sample (steps 1.3, 1.4, 2): ``results/phase3/libero/smolvla_libero/<run>``;
* fleet (step 4): ``results/phase3/fleet/capacity.json`` (stage 2) and the stage-1 results;
* validation: ``results/phase3/validation/evaluation.json``;
* economics: ``kortyn/experiments/phase3/economics.yaml``.

Statistics are those of :mod:`kortyn.phase3_steps` (pre-registered rules) and :mod:`kortyn.fleet_predict`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from kortyn.config import ConfigError, LiberoEnvConfig, load_config
from kortyn.economics import (
    EconomicsConfig,
    brain_cost,
    break_even_capacity,
    client_cost,
    load_economics,
    onboard_cost,
)
from kortyn.fleet import FleetSimConfig, load_fleet_config
from kortyn.libero_env import episode_slot, task_language
from kortyn.phase3_steps import (
    Episode,
    Level,
    PairedComparison,
    StepError,
    dose_response,
    jpeg_becomes_standard,
    load_episodes,
    paired_comparison,
    reproduction_verdict,
    select_tasks,
    successes,
    summary_successes,
    task_stats,
)
from kortyn.report import (
    DPI,
    GRID,
    INK,
    INK_2,
    MUTED,
    REFERENCE_GRAY,
    apply_style,
    header,
    header_legend,
    make_figure,
    percent_axis,
)

LOG = logging.getLogger("kortyn.report_phase3")

STRATEGY_ORDER = ["a", "b", "c", "d", "e"]
STRATEGY_COLORS = {"a": "#2a78d6", "b": "#eb6834", "c": "#1baf7a", "d": "#eda100", "e": "#e87ba4"}
STRATEGY_NAMES = {
    "a": "a FIFO",
    "b": "b FIFO + Batching",
    "c": "c Deadline (EDF)",
    "d": "d EDF + Batching",
    "e": "e d + Zugangskontrolle",
}
BATCHING_ORDER = ["optimistisch", "realistisch", "pessimistisch", "kein_gewinn"]
BATCHING_NAMES = {
    "optimistisch": "optimistisch",
    "realistisch": "realistisch",
    "pessimistisch": "pessimistisch",
    "kein_gewinn": "kein Gewinn",
}
NETWORK_NAMES = {"buero_wlan": "Büro-WLAN", "schlecht": "schlechtes Netz"}
TASK_COLORS = ["#2a78d6", "#eb6834"]
POOLED_COLOR = INK


# --------------------------------------------------------------------------------------------- format
def pct(value: float | None, digits: int = 0) -> str:
    return "–" if value is None or math.isnan(value) else f"{100 * value:.{digits}f} %"


def eur(value: float, digits: int = 0) -> str:
    """Euro amount with a thin space as thousands separator."""
    return f"{value:,.{digits}f} €".replace(",", "\u202f")


def pp(value: float) -> str:
    return f"{100 * value:+.0f} pp"


def interval_pp(bounds: tuple[float, float]) -> str:
    return f"[{100 * bounds[0]:+.0f}, {100 * bounds[1]:+.0f}] pp"


def interval_pct(bounds: tuple[float, float]) -> str:
    return f"[{100 * bounds[0]:.0f}, {100 * bounds[1]:.0f}] %"


def p_text(p: float | None) -> str:
    if p is None:
        return "–"
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def comparison_row(name: str, comparison: PairedComparison) -> str:
    return (
        f"| {name} | {comparison.pairs} | {pct(comparison.rate_a)} | {pct(comparison.rate_b)} | "
        f"{pp(comparison.difference)} {interval_pp(comparison.interval)} | "
        f"{comparison.only_a} / {comparison.only_b} | "
        f"{p_text(comparison.p_value)} |"
    )


COMPARISON_HEADER = [
    "| Vergleich | Paare | Erfolg A | Erfolg B | B − A [95-%-KI] | nur A / nur B | McNemar p |",
    "|---|---:|---:|---:|---|---:|---:|",
]


# --------------------------------------------------------------------------------------------- step 1
@dataclass
class PilotData:
    libero: LiberoEnvConfig
    sync25: dict[int, Episode]
    sync50: dict[int, Episode] | None
    lerobot: dict[int, bool] | None


def load_pilot(config_path: Path, profile: str = "smolvla_libero") -> PilotData | None:
    config = load_config(config_path)
    libero = config.env.libero
    root = Path(config.experiment.results_dir)
    if libero is None or not (root / profile / "sync25" / "episodes.jsonl").exists():
        return None
    sync50 = root / profile / "sync50"
    summaries = sorted((root / "lerobot_eval").glob("*/summary.json"))
    return PilotData(
        libero=libero,
        sync25=load_episodes(root / profile / "sync25", config.env.fps),
        sync50=load_episodes(sync50, config.env.fps) if (sync50 / "episodes.jsonl").exists() else None,
        lerobot=summary_successes(summaries) if summaries else None,
    )


def pilot_section(pilot: PilotData, task_names: Mapping[int, str]) -> list[str]:
    libero = pilot.libero
    stats = {t.task_id: t for t in task_stats(pilot.sync25, libero)}
    lines = [
        "### Pilot: zehn LIBERO-Long-Aufgaben × Init-States 0–9",
        "",
        "| Aufgabe | sync25 | sync50 | lerobot-eval | Median-Länge erfolgreicher Episoden (sync25) |",
        "|---|---:|---:|---:|---:|",
    ]

    def per_task(values: Mapping[int, bool] | None, task: int) -> str:
        if values is None:
            return "–"
        seeds = [s for s in values if episode_slot(libero, s)[0] == task]
        return f"{sum(values[s] for s in seeds)}/{len(seeds)}" if seeds else "–"

    sync50 = successes(pilot.sync50) if pilot.sync50 is not None else None
    sync25 = successes(pilot.sync25)
    for task in libero.task_ids:
        name = task_names.get(task, "")
        median = stats[task].median_success_steps
        lines.append(
            f"| {task}{f': {name}' if name else ''} | {per_task(sync25, task)} | {per_task(sync50, task)} | "
            f"{per_task(pilot.lerobot, task)} | {'–' if median is None else f'{median:.0f} Takte'} |"
        )

    def total(values: Mapping[int, bool] | None) -> str:
        return (
            "–"
            if values is None
            else f"{sum(values.values())}/{len(values)} ({pct(sum(values.values()) / len(values))})"
        )

    lines.append(f"| **alle** | **{total(sync25)}** | **{total(sync50)}** | **{total(pilot.lerobot)}** | |")
    lines.append("")
    try:
        selection = select_tasks(list(stats.values()))
        lines.append(
            f"**Aufgabenwahl (Regel 1.2):** Band {selection.band[0]}–{selection.band[1]} Erfolge; gewählt: "
            + ", ".join(
                f"Aufgabe {t} („{task_names[t]}“)" if task_names.get(t) else f"Aufgabe {t}"
                for t in selection.selected
            )
            + "."
        )
    except StepError as exc:
        lines.append(f"**Aufgabenwahl:** nicht anwendbar ({exc}).")
    lines.append("")
    if sync50 is not None and pilot.lerobot is not None:
        common = set(sync50) & set(pilot.lerobot)
        if common:
            comparison = paired_comparison(pilot.lerobot, sync50)
            lines += [
                "**Reproduktion R1** (`sync50` gegen `lerobot-eval` auf denselben Episoden; "
                "A = lerobot-eval, "
                "B = Kortyn `sync50`):",
                "",
                *COMPARISON_HEADER,
                comparison_row("R1", comparison),
                "",
                f"Urteil nach Kriterium R1: **{reproduction_verdict(comparison)}** "
                f"({len(common)} von 100 Episoden ausgewertet).",
                "",
            ]
    return lines


# --------------------------------------------------------------------------------------------- step 2
def figure_dose(levels: Sequence[Level], libero: LiberoEnvConfig, budget_ms: float, path: Path) -> None:
    """Success against fixed latency: pooled (with 95 % Wilson interval) and per task; jitter marked."""
    apply_style()
    fixed = sorted(
        (lv for lv in levels if lv.latency_ms is not None), key=lambda lv: float(lv.latency_ms or 0.0)
    )
    jitter = [lv for lv in levels if lv.latency_ms is None]
    fig, axes, layout = make_figure(1, 1, 6.6, 3.4, header_in=1.25)
    ax = axes[0][0]
    x = np.array([float(lv.latency_ms or 0.0) for lv in fixed])
    ax.axvline(budget_ms, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(budget_ms + 25, 0.03, "Budget 1 250 ms", color=INK_2, fontsize=8, va="bottom")
    handles: list[Any] = []
    for index, task in enumerate(libero.task_ids):
        y = np.array([lv.per_task[task][0] if task in lv.per_task else math.nan for lv in fixed])
        color = TASK_COLORS[index % len(TASK_COLORS)]
        ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=5, zorder=3)
        ax.annotate(
            f"Aufgabe {task}",
            (x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_2,
            fontsize=8,
            va="center",
        )
        handles.append(
            Line2D([0], [0], color=color, linewidth=2.0, marker="o", markersize=5, label=f"Aufgabe {task}")
        )
    pooled = np.array([lv.rate for lv in fixed])
    low = np.array([lv.interval[0] for lv in fixed])
    high = np.array([lv.interval[1] for lv in fixed])
    ax.fill_between(x, low, high, color=GRID, alpha=0.9, zorder=2, linewidth=0)
    ax.plot(x, pooled, color=POOLED_COLOR, linewidth=2.0, marker="o", markersize=5, zorder=4)
    handles.append(
        Line2D(
            [0],
            [0],
            color=POOLED_COLOR,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="beide Aufgaben (95-%-KI grau)",
        )
    )
    for lv in jitter:  # drawn next to the 1 000-ms level (its mean latency), not on top of it
        ax.annotate(
            "Jitter",
            (1070.0, lv.rate),
            xytext=(8, 0),
            textcoords="offset points",
            color=INK_2,
            fontsize=8,
            va="center",
        )
        ax.errorbar(
            [1070.0],
            [lv.rate],
            yerr=[[lv.rate - lv.interval[0]], [lv.interval[1] - lv.rate]],
            fmt="D",
            color=REFERENCE_GRAY,
            markersize=7,
            capsize=3,
            linewidth=1.2,
            zorder=5,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=REFERENCE_GRAY,
                marker="D",
                linestyle="none",
                markersize=7,
                label="Jitter 500–1 500 ms (Mittel 1 000 ms)",
            )
        )
    ax.set_xlim(-60, max(2100.0, float(x.max()) + 250))
    ax.set_ylim(0, 1.0)
    percent_axis(ax)
    ax.set_xlabel("Ende-zu-Ende-Latenz je Plan (ms), Kaltstart")
    ax.set_ylabel("Erfolgsquote")
    ax.xaxis.set_major_locator(FixedLocator([float(v) for v in x]))
    top = header(
        fig,
        layout,
        "Dosis-Wirkung: Erfolg gegen Latenz (LIBERO-Long, SmolVLA)",
        "Ein Roboter, feste Latenz je Anfrage, 100 gepaarte Episoden je Stufe (50 je Aufgabe).",
    )
    header_legend(fig, layout, handles, ncol=2, top_in=top)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def dose_section(levels: Sequence[Level], libero: LiberoEnvConfig) -> list[str]:
    tasks = list(libero.task_ids)
    lines = [
        "| Bedingung | Erfolg (beide) [95-%-KI] | "
        + " | ".join(f"Aufgabe {t}" for t in tasks)
        + " | McNemar p gegen 0 ms (Holm) | Starvation (Anteil Takte) | Median Zeit bis Erfolg |",
        "|---|---:|" + "---:|" * len(tasks) + "---:|---:|---:|",
    ]
    for lv in levels:
        per_task = " | ".join(pct(lv.per_task[t][0]) if t in lv.per_task else "–" for t in tasks)
        p_holm = "Referenz" if lv.p_holm is None else f"{p_text(lv.p_value)} ({p_text(lv.p_holm)})"
        time_text = "–" if lv.median_time_to_success_s is None else f"{lv.median_time_to_success_s:.1f} s"
        lines.append(
            f"| {lv.label} | {pct(lv.rate)} {interval_pct(lv.interval)} | {per_task} | {p_holm} | "
            f"{pct(lv.starved_share, 1)} | {time_text} |"
        )
    return lines


# --------------------------------------------------------------------------------------------- capacity
def load_capacity(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def capacity_lookup(data: Mapping[str, Any], fleet: FleetSimConfig) -> dict[tuple[str, str, str, str], int]:
    """(network, gpu, batching, strategy) -> capacity; strategies without batching apply to every profile."""
    first = next(iter(fleet.batching))
    table: dict[tuple[str, str, str, str], int] = {}
    for entry in data["configurations"]:
        key = (entry["network"], entry["gpu"], entry["batching"], entry["strategy"])
        table[key] = int(entry["capacity"])
        if not fleet.strategies[entry["strategy"]].batches and entry["batching"] == first:
            for profile in fleet.batching:
                table[(entry["network"], entry["gpu"], profile, entry["strategy"])] = int(entry["capacity"])
    return table


def gpu_label(fleet: FleetSimConfig, gpu: str) -> str:
    names = {
        "rtx3060ti_schaetzung": "RTX 3060 Ti (Schätzung)",
        "rtx3090": "RTX 3090",
        "rtx4090": "RTX 4090",
        "rtx4090_kompiliert": "RTX 4090 kompiliert",
        "validierung": "Validierung",
    }
    s1 = fleet.gpus[gpu].s1_ms
    return f"{names.get(gpu, gpu)} ({s1:.0f} ms)" if s1 is not None else f"{names.get(gpu, gpu)} (gemessen)"


def capacity_table(
    table: Mapping[tuple[str, str, str, str], int], fleet: FleetSimConfig, network: str, batching: str
) -> list[str]:
    gpus = [g for g in fleet.gpus if g != "validierung"]
    lines = [
        "| Strategie | " + " | ".join(gpu_label(fleet, g) for g in gpus) + " |",
        "|---|" + "---:|" * len(gpus),
    ]
    for strategy in STRATEGY_ORDER:
        if strategy not in fleet.strategies:
            continue
        cells = [str(table.get((network, g, batching, strategy), "–")) for g in gpus]
        lines.append(f"| {STRATEGY_NAMES[strategy]} | " + " | ".join(cells) + " |")
    return lines


def batching_table(
    table: Mapping[tuple[str, str, str, str], int], fleet: FleetSimConfig, network: str, gpu: str
) -> list[str]:
    profiles = [p for p in BATCHING_ORDER if p in fleet.batching]
    lines = [
        "| Strategie | "
        + " | ".join(f"{BATCHING_NAMES[p]} (β {fleet.batching[p].beta:g})" for p in profiles)
        + " |",
        "|---|" + "---:|" * len(profiles),
    ]
    for strategy in STRATEGY_ORDER:
        if strategy in fleet.strategies:
            cells = [str(table.get((network, gpu, p, strategy), "–")) for p in profiles]
            lines.append(f"| {STRATEGY_NAMES[strategy]} | " + " | ".join(cells) + " |")
    return lines


def figure_capacity(
    table: Mapping[tuple[str, str, str, str], int], fleet: FleetSimConfig, network: str, path: Path, cap: int
) -> None:
    """Capacity against the batching gain (beta), one panel per GPU profile, one line per strategy."""
    apply_style()
    gpus = [g for g in fleet.gpus if g != "validierung"]
    profiles = [p for p in BATCHING_ORDER if p in fleet.batching]
    betas = [fleet.batching[p].beta for p in profiles]
    cols = min(2, len(gpus))
    rows = math.ceil(len(gpus) / cols)
    shown = [v for (n, g, _, _), v in table.items() if n == network and g in gpus]
    top_value = float(max([cap, *shown]))
    fig, axes, layout = make_figure(rows, cols, 3.5, 2.5, header_in=1.45, hspace_in=0.75, sharey=True)
    for index, gpu in enumerate(gpus):
        ax = axes[index // cols][index % cols]
        for strategy in STRATEGY_ORDER:
            if strategy not in fleet.strategies:
                continue
            values = [table.get((network, gpu, p, strategy)) for p in profiles]
            y = np.array([math.nan if v is None else float(v) for v in values])
            ax.plot(
                betas, y, color=STRATEGY_COLORS[strategy], linewidth=2.0, marker="o", markersize=5, zorder=3
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(betas))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.set_yscale("symlog", base=2, linthresh=1)
        ax.set_ylim(0, top_value * 1.3)
        ax.yaxis.set_major_locator(
            FixedLocator([v for v in (0, 1, 2, 4, 8, 16, 32, 64, 128) if v <= top_value * 1.3])
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.set_title(gpu_label(fleet, gpu), loc="left", fontsize=10)
        if index % cols == 0:
            ax.set_ylabel("Kapazität (Roboter)")
        if index // cols == rows - 1:
            ax.set_xlabel("Batching-Zusatzkosten β je weiterer Anfrage")
    for index in range(len(gpus), rows * cols):
        axes[index // cols][index % cols].set_visible(False)
    handles = [
        Line2D(
            [0],
            [0],
            color=STRATEGY_COLORS[s],
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=STRATEGY_NAMES[s],
        )
        for s in STRATEGY_ORDER
        if s in fleet.strategies
    ]
    top = header(
        fig,
        layout,
        "Kapazität: Roboter je GPU nach Strategie und Batching-Profil",
        f"Netz {NETWORK_NAMES.get(network, network)}; Kapazität nach der vorab festgelegten Definition "
        "(≤ 1 % Budget-Verletzungen, schlechtester Roboter nicht signifikant schlechter als ein einzelner); "
        f"Raster 1–{cap}.",
    )
    header_legend(fig, layout, handles, ncol=3, top_in=top)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_miss_share(
    data: Mapping[str, Any],
    fleet: FleetSimConfig,
    network: str,
    gpu: str,
    batching: str,
    path: Path,
    title: str,
) -> None:
    """Share of requests that miss their budget, against fleet size, one line per strategy."""
    apply_style()
    fig, axes, layout = make_figure(1, 1, 6.2, 3.0, header_in=1.35)
    ax = axes[0][0]
    ax.axhline(0.01, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(1.05, 0.012, "Schwelle 1 %", color=INK_2, fontsize=8, va="bottom")
    handles: list[Any] = []
    for strategy in STRATEGY_ORDER:
        if strategy not in fleet.strategies:
            continue
        wanted = batching if fleet.strategies[strategy].batches else next(iter(fleet.batching))
        entry = next(
            (
                e
                for e in data["configurations"]
                if (e["network"], e["gpu"], e["batching"], e["strategy"]) == (network, gpu, wanted, strategy)
            ),
            None,
        )
        if entry is None:
            continue
        sizes = [s["robots"] for s in entry["sizes"]]
        shares = [max(s["miss_share"], 1e-4) for s in entry["sizes"]]
        ax.plot(
            sizes, shares, color=STRATEGY_COLORS[strategy], linewidth=2.0, marker="o", markersize=4, zorder=3
        )
        handles.append(
            Line2D([0], [0], color=STRATEGY_COLORS[strategy], linewidth=2.0, label=STRATEGY_NAMES[strategy])
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g} %".replace(".", ",")))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_xlabel("Flottengröße (Roboter)")
    ax.set_ylabel("Anfragen über Budget")
    top = header(fig, layout, title, "Werte unter 0,01 % sind am unteren Rand dargestellt.")
    header_legend(fig, layout, handles, ncol=3, top_in=top)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- validation
def validation_section(evaluations: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Fall | Roboter (aufgenommen) | Episoden | gemessen | vorhergesagt | Differenz [98,75-%-KI] "
        "| Ergebnis |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for e in evaluations:
        verdict = "✓ KI enthält 0" if e["passes"] else "✗ KI enthält 0 nicht"
        lines.append(
            f"| {e['case']} | {e['robots']} ({e['admitted']}) | {e['episodes']} | {pct(e['measured'])} | "
            f"{pct(e['predicted'])} | {pp(e['difference'])} {interval_pp(tuple(e['interval']))} | {verdict} |"
        )
    return lines


# --------------------------------------------------------------------------------------------- economics
def predicted_success(levels_ms: Sequence[float], curve: Sequence[float], latency_ms: float) -> float:
    """``f(L)`` of the dose-response curve (linear between levels, constant beyond the last)."""
    return float(np.interp(latency_ms, np.asarray(levels_ms, dtype=float), np.asarray(curve, dtype=float)))


def economics_section(
    economics: EconomicsConfig,
    table: Mapping[tuple[str, str, str, str], int],
    busy: Mapping[tuple[str, str, str, str], float],
    network: str,
    batching: str,
    curve: tuple[Sequence[float], Sequence[float]] | None = None,
) -> list[str]:
    client = client_cost(economics)
    lines = [
        f"Annahmen: Abschreibung über {economics.depreciation_months} Monate, Betrieb "
        f"{economics.operating_hours_per_month:.0f} h/Monat, "
        f"Strom {economics.electricity_eur_per_kwh:.4f} €/kWh "
        f"({economics.electricity_source.source}, {economics.electricity_source.date}). Host: "
        f"{economics.host.label if economics.host else '–'} "
        f"({eur(economics.host.price_eur) if economics.host else '–'}, "
        f"{economics.host.power_w if economics.host else 0:.0f} W). Netz {network}, Batching {batching}.",
    ]
    if client is not None and economics.client is not None:
        lines.append(
            f"Ergänzung (nicht präregistriert): Rechner, den jeder Roboter neben dem Brain braucht: "
            f"{client.label} ({eur(client.capex_eur)}, {economics.client.power_w:.0f} W, "
            f"{eur(client.monthly_eur, 2)} je Monat; {economics.client.price_source.source}, "
            f"{economics.client.price_source.date})."
        )
    with_client = client is not None
    lines += [
        "",
        "| GPU (Preis) | Strategie | Kapazität | Hardware je Roboter | Kosten je Roboter und Monat |"
        + (" mit Roboter-Rechner (Ergänzung) |" if with_client else ""),
        "|---|---|---:|---:|---:|" + ("---:|" if with_client else ""),
    ]
    for gpu in economics.gpus:
        for strategy in STRATEGY_ORDER:
            key = (network, gpu.profile, batching, strategy)
            if key not in table:
                continue
            cost = brain_cost(economics, gpu, table[key], busy.get(key, 1.0))
            capex = "–" if cost is None else eur(cost.capex_per_robot_eur)
            monthly = "–" if cost is None else eur(cost.monthly_per_robot_eur, 2)
            row = (
                f"| {gpu.label} ({eur(gpu.price_eur)}, {gpu.condition}) | {STRATEGY_NAMES[strategy]} | "
                f"{table[key]} | {capex} | {monthly} |"
            )
            if client is not None:
                row += (
                    " – |"
                    if cost is None
                    else f" {eur(cost.monthly_per_robot_eur + client.monthly_eur, 2)} |"
                )
            lines.append(row)
    lines += [
        "",
        "| Rechner im Roboter | Preis | Leistung | Latenz je Plan | Erfolg f(L) "
        "| Kosten je Roboter und Monat | Quelle Latenz |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for option in economics.onboard:
        onboard = onboard_cost(economics, option)
        success = "–" if curve is None else pct(predicted_success(curve[0], curve[1], option.latency_ms))
        lines.append(
            f"| {option.label} | {eur(option.price_eur)} | {option.power_w:.0f} W | "
            f"{option.latency_ms:.0f} ms | {success} | "
            f"{eur(onboard.monthly_per_robot_eur, 2)} | {option.latency_source.source} ({option.runtime}) |"
        )
    if economics.onboard and economics.gpus:
        lines += [
            "",
            "Break-even (ab dieser Flottengröße ist das Brain je Roboter und Monat günstiger; GPU zu 90 % "
            "ausgelastet" + ("; in Klammern mit Roboter-Rechner, Ergänzung" if with_client else "") + "):",
            "",
        ]
        for gpu in economics.gpus:
            cells = []
            for option in economics.onboard:
                breakeven = break_even_capacity(economics, gpu, option, busy_share=0.9)
                cell = f"{option.label}: {breakeven if breakeven else 'nie'}"
                if with_client:
                    extra = break_even_capacity(economics, gpu, option, busy_share=0.9, with_client=True)
                    cell += f" ({extra if extra else 'nie'})"
                cells.append(cell)
            lines.append(f"- {gpu.label}: " + "; ".join(cells))
    return lines


# --------------------------------------------------------------------------------------------- main
def build(args: argparse.Namespace) -> list[str]:
    lines = [
        "# Phase 3: Daten (automatisch erzeugt)",
        "",
        f"_Erzeugt am {datetime.now(UTC):%d.%m.%Y %H:%M} UTC mit `python -m kortyn.report_phase3`._",
        "",
    ]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pilot = load_pilot(Path(args.pilot_config))
    if pilot is not None:
        task_names = {t: task_language(pilot.libero.suite, t) for t in pilot.libero.task_ids}
        lines += ["## Schritt 1", "", *pilot_section(pilot, task_names), ""]
    main_config_path = Path(args.main_config)
    if main_config_path.exists():
        config = load_config(main_config_path)
        libero = config.env.libero
        root = Path(config.experiment.results_dir) / "smolvla_libero"

        def run(label: str) -> dict[int, Episode] | None:
            path = root / label / "episodes.jsonl"
            return load_episodes(path, config.env.fps) if path.exists() else None

        sync_raw, sync_q90 = run("sync25"), run("sync25_q90")
        warm, cold = run("warm_1000ms"), run("fest_1000ms")
        rows: list[str] = []
        notes: list[str] = []
        if sync_raw and sync_q90:
            comparison = paired_comparison(successes(sync_raw), successes(sync_q90))
            rows.append(comparison_row("1.4 JPEG q90 (A roh, B q90), sync25", comparison))
            decision = (
                "wird Standard" if jpeg_becomes_standard(comparison) else "bleibt aus, roh bleibt Standard"
            )
            notes.append(
                f"Regel 1.4: JPEG q90 {decision} ({comparison.pairs} von 100 Episodenpaaren ausgewertet)."
            )
        if warm and cold:
            rows.append(
                comparison_row(
                    "1.3 Kalt- gegen Warmstart bei 1 000 ms (A kalt, B warm)",
                    paired_comparison(successes(cold), successes(warm)),
                )
            )
        if rows:
            lines += [
                "### Hauptstichprobe: JPEG und Kaltstart",
                "",
                *COMPARISON_HEADER,
                *rows,
                "",
                *notes,
                "",
            ]
        if libero is not None:
            labels = [
                "fest_0ms",
                "fest_250ms",
                "fest_500ms",
                "fest_1000ms",
                "fest_1250ms",
                "fest_1500ms",
                "fest_2000ms",
                "jitter_500_1500ms",
            ]
            runs: dict[str, tuple[float | None, Mapping[int, Episode]]] = {}
            for label in labels:
                episodes = run(label)
                if episodes is not None:
                    latency = (
                        None if label.startswith("jitter") else float(label.split("_")[1].removesuffix("ms"))
                    )
                    runs[label] = (latency, episodes)
            if "fest_0ms" in runs and len(runs) > 1:
                levels = dose_response(runs, "fest_0ms", libero)
                figure_dose(levels, libero, 1250.0, out / "phase3_dosis_wirkung.png")
                lines += [
                    "## Schritt 2: Dosis-Wirkung",
                    "",
                    *dose_section(levels, libero),
                    "",
                    "![Dosis-Wirkung](phase3_dosis_wirkung.png)",
                    "",
                ]
    fleet = load_fleet_config(args.fleet_config)
    capacity = load_capacity(Path(fleet.prediction.output if fleet.prediction else args.capacity))
    if capacity is not None:
        table = capacity_lookup(capacity, fleet)
        busy: dict[tuple[str, str, str, str], float] = {}
        for entry in capacity["configurations"]:
            cap = int(entry["capacity"])
            at_cap = next((s for s in entry["sizes"] if s["robots"] == cap), None)
            busy[(entry["network"], entry["gpu"], entry["batching"], entry["strategy"])] = (
                float(at_cap["busy_share"]) if at_cap else 0.0
            )
        lines += ["## Schritt 4: Kapazität", ""]
        for network in fleet.networks:
            lines += [
                f"### Netz {network}, Batching realistisch",
                "",
                *capacity_table(table, fleet, network, "realistisch"),
                "",
            ]
            figure_capacity(
                table, fleet, network, out / f"phase3_kapazitaet_{network}.png", max(fleet.fleet_sizes)
            )
            lines += [f"![Kapazität {network}](phase3_kapazitaet_{network}.png)", ""]
            for gpu in fleet.gpus:
                if gpu == "validierung":
                    continue
                lines += [
                    f"#### {gpu_label(fleet, gpu)}: Abhängigkeit vom Batching-Profil",
                    "",
                    *batching_table(table, fleet, network, gpu),
                    "",
                ]
        if "buero_wlan" in fleet.networks and "realistisch" in fleet.batching:
            for gpu in [g for g in fleet.gpus if g != "validierung"]:
                name = f"phase3_verletzungen_{gpu}.png"
                figure_miss_share(
                    capacity,
                    fleet,
                    "buero_wlan",
                    gpu,
                    "realistisch",
                    out / name,
                    f"Budget-Verletzungen: {gpu_label(fleet, gpu)}, Batching realistisch, Büro-WLAN",
                )
                lines += [f"![Budget-Verletzungen {gpu}]({name})", ""]
        scan = load_capacity(Path(args.capacity_scan))
        if scan is not None:
            scan_table = capacity_lookup(scan, fleet)
            lines += [
                "### Ergänzung: jede Flottengröße 1–128 (Büro-WLAN, 3 Wiederholungen)",
                "",
                *[
                    line
                    for profile in [p for p in BATCHING_ORDER if p in fleet.batching]
                    for line in (
                        f"**Batching {BATCHING_NAMES[profile]}:**",
                        "",
                        *capacity_table(scan_table, fleet, "buero_wlan", profile),
                        "",
                    )
                ],
            ]
        p10 = load_capacity(Path(args.capacity_p10))
        if p10 is not None:
            p10_table = capacity_lookup(p10, fleet)
            lines += [
                "### Sensitivität: 10 Aktionen je Plan (nur Kriterium 1, Büro-WLAN)",
                "",
                *capacity_table(p10_table, fleet, "buero_wlan", "realistisch"),
                "",
            ]
        economics_path = Path(args.economics)
        if economics_path.exists():
            economics = load_economics(economics_path)
            lines += [
                "## Wirtschaftlichkeit",
                "",
                *economics_section(
                    economics,
                    table,
                    busy,
                    "buero_wlan",
                    "realistisch",
                    (capacity["levels_ms"], capacity["curve"]) if "curve" in capacity else None,
                ),
                "",
            ]
    evaluation_path = Path(args.validation)
    if evaluation_path.exists():
        evaluations = json.loads(evaluation_path.read_text(encoding="utf-8"))
        lines += ["## Validierung", "", *validation_section(evaluations), ""]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-3 report data and figures.")
    parser.add_argument("--pilot-config", default="kortyn/experiments/phase3/libero_pilot.yaml")
    parser.add_argument("--main-config", default="kortyn/experiments/phase3/libero_main.yaml")
    parser.add_argument("--fleet-config", default="kortyn/experiments/phase3/fleet.yaml")
    parser.add_argument("--capacity", default="results/phase3/fleet/capacity.json")
    parser.add_argument("--capacity-scan", default="results/phase3/fleet/capacity_scan.json")
    parser.add_argument("--capacity-p10", default="results/phase3/fleet/capacity_p10.json")
    parser.add_argument("--validation", default="results/phase3/validation/evaluation.json")
    parser.add_argument("--economics", default="kortyn/experiments/phase3/economics.yaml")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        lines = build(args)
    except (ConfigError, StepError) as exc:
        LOG.error("%s", exc)
        return 2
    out = Path(args.out_dir) / "phase3_data.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    LOG.info("written %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
