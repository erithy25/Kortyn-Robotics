"""Generate ``reports/phase1.md`` (tables) and PNG figures from the JSONL results.

    python -m kortyn.report                      # reads results/, writes reports/
    python -m kortyn.report --results-dir X --reports-dir Y

Figures follow one visual system: scenarios are ordered by network severity and always use the same
one-hue ordinal ramp (light = good network, dark = bad network); the synchronous reference is neutral
gray; Brain profiles are separate panels (small multiples), never extra colors. Starvation, a state, uses
the reserved status color with a marker shape and a legend label.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from kortyn.config import DEFAULT_BASE_CONFIG, ConfigError, KortynConfig, load_config
from kortyn.metrics import (
    EpisodeRecord,
    RequestRecord,
    RunSummary,
    final_episodes,
    mcnemar_exact,
    paired_outcomes,
    read_jsonl,
    summarize_run,
)

LOG = logging.getLogger("kortyn.report")

# --------------------------------------------------------------------------------------------- style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
REFERENCE_GRAY = "#898781"
STALL_SHADE = "#e1e0d9"
STATUS_CRITICAL = "#d03b3b"
# One-hue ordinal ramp (validated: monotone lightness, visible gaps, light end >= 2:1 on the surface).
SEVERITY = {"gut": "#86b6ef", "buero_wlan": "#3987e5", "schlecht": "#1c5cab", "sehr_schlecht": "#0d366b"}
# Categorical slots 1-3 (validated all-pairs; slot 3 < 3:1 contrast -> always direct-labelled + table).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]

SCENARIO_ORDER = ["referenz_sync", "gut", "gut_wiederholung", "buero_wlan", "schlecht", "sehr_schlecht"]
NETWORK_RUNS = ["gut", "buero_wlan", "schlecht", "sehr_schlecht"]
LABELS = {
    "referenz_sync": "Referenz\n(synchron)",
    "gut": "gut",
    "gut_wiederholung": "gut\n(Wiederh.)",
    "buero_wlan": "Büro-WLAN",
    "schlecht": "schlecht",
    "sehr_schlecht": "sehr\nschlecht",
}
DPI = 160


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.size": 9,
            "font.family": "sans-serif",
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def run_color(label: str) -> str:
    if label == "referenz_sync":
        return REFERENCE_GRAY
    if label == "gut_wiederholung":
        return SEVERITY["gut"]
    return SEVERITY.get(label, CATEGORICAL[0])


@dataclass(frozen=True)
class Layout:
    width_in: float
    height_in: float
    left_in: float


def make_figure(
    rows: int,
    cols: int,
    panel_w: float,
    panel_h: float,
    *,
    header_in: float = 1.05,
    footer_in: float = 0.6,
    left_in: float = 0.85,
    right_in: float = 0.25,
    hspace_in: float = 0.5,
    wspace_in: float = 0.35,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[Any, Any, Layout]:
    """Figure with fixed absolute margins, so header, legend and panel titles never collide."""
    width = left_in + right_in + cols * panel_w + (cols - 1) * wspace_in
    height = header_in + footer_in + rows * panel_h + (rows - 1) * hspace_in
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False, sharex=sharex, sharey=sharey)
    fig.subplots_adjust(
        left=left_in / width,
        right=1 - right_in / width,
        top=1 - header_in / height,
        bottom=footer_in / height,
        hspace=hspace_in / panel_h,
        wspace=wspace_in / panel_w,
    )
    return fig, axes, Layout(width, height, left_in)


def header(fig: Any, layout: Layout, title: str, subtitle: str | None = None) -> None:
    x = layout.left_in / layout.width_in
    fig.text(
        x, 1 - 0.14 / layout.height_in, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK
    )
    if subtitle:
        # ~0.065 in per character at 8.5 pt: wrap instead of running off the canvas on narrow figures
        chars = max(40, int((layout.width_in - layout.left_in - 0.2) / 0.065))
        fig.text(
            x,
            1 - 0.45 / layout.height_in,
            textwrap.fill(subtitle, chars),
            ha="left",
            va="top",
            fontsize=8.5,
            color=INK_2,
            linespacing=1.3,
        )


def header_legend(fig: Any, layout: Layout, handles: list[Any], ncol: int) -> None:
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(layout.left_in / layout.width_in - 0.01, 1 - 0.62 / layout.height_in),
        ncol=ncol,
        handlelength=1.6,
        columnspacing=1.4,
    )


def mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def bar_with_round_end(
    ax: Axes, x: float, value: float, width: float, color: str, radius_px: float = 4.0
) -> None:
    """Vertical bar: 4px rounded data end, square at the baseline (axis limits must be final)."""
    if value <= 0:
        return
    ax.figure.canvas.draw()  # make sure the axes geometry is current
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ux = (x1 - x0) / bbox.width
    uy = (y1 - y0) / bbox.height
    rx = min(radius_px * ux, width / 2)
    ry = rx * uy / ux
    left = x - width / 2
    patch = FancyBboxPatch(
        (left, 0.0),
        width,
        value,
        boxstyle=f"round,pad=0,rounding_size={rx}",
        mutation_aspect=uy / ux,
        facecolor=color,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(patch)
    ax.add_patch(
        Rectangle((left, 0.0), width, min(value, ry * 1.5), facecolor=color, edgecolor="none", zorder=3)
    )


# --------------------------------------------------------------------------------------------- data
@dataclass
class RunData:
    label: str
    episodes: list[EpisodeRecord]
    requests: list[RequestRecord]
    summary: RunSummary


@dataclass
class ProfileData:
    name: str
    meta: dict[str, Any]
    runs: dict[str, RunData]


def load_profile(directory: Path) -> ProfileData | None:
    meta_path = directory / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    runs: dict[str, RunData] = {}
    for run_dir in sorted(p for p in directory.iterdir() if p.is_dir() and p.name != "logs"):
        episodes = [EpisodeRecord.from_json(r) for r in read_jsonl(run_dir / "episodes.jsonl")]
        if not episodes:
            continue
        requests = [RequestRecord.from_json(r) for r in read_jsonl(run_dir / "requests.jsonl")]
        runs[run_dir.name] = RunData(run_dir.name, episodes, requests, summarize_run(episodes, requests))
    if not runs:
        return None
    return ProfileData(directory.name, meta, runs)


def ordered(runs: dict[str, RunData]) -> list[str]:
    known = [label for label in SCENARIO_ORDER if label in runs]
    return known + sorted(label for label in runs if label not in SCENARIO_ORDER)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------------------------- formatting
def fmt(value: float | None, digits: int = 0, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:,.{digits}f}{suffix}".replace(",", " ")


def pct(value: float | None, digits: int = 0) -> str:
    return "–" if value is None else f"{100 * value:.{digits}f} %"


def ci(bounds: tuple[float, float] | None) -> str:
    return "–" if bounds is None else f"[{100 * bounds[0]:.0f}–{100 * bounds[1]:.0f}]"


def network_label(meta: dict[str, Any], label: str) -> str:
    run = meta.get("runs", {}).get(label, {})
    if run.get("mode") == "sync":
        return "– (Sim wartet)"
    network = run.get("network", {})
    rtt = network.get("rtt_ms", 0)
    jitter = network.get("rtt_jitter_ms", 0)
    text = f"{rtt:g} ± {jitter:g} ms" if rtt or jitter else "0 ms"
    stall = network.get("stall")
    if stall:
        text += f", Stall {stall['duration_ms']:g} ms/{stall['interval_s']:g} s"
    return text


# --------------------------------------------------------------------------------------------- figures
def figure_success(profiles: list[ProfileData], published: dict[str, Any] | None, path: Path) -> None:
    fig, axes, layout = make_figure(1, len(profiles), 4.9, 3.0, sharey=True, footer_in=0.75, header_in=1.45)
    header(
        fig,
        layout,
        "Erfolgsquote je Szenario",
        "Gleiche Seeds in allen Szenarien · Farbe = Netz-Schweregrad (hell = gut, dunkel = schlecht), "
        "grau = synchrone Referenz",
    )
    legend_handles: list[Any] = [Line2D([], [], color=INK_2, linewidth=1.0, label="Wilson-95 %-KI")]
    if published:
        legend_handles.append(
            Line2D(
                [],
                [],
                color=MUTED,
                linewidth=1.0,
                label=f"publiziert (DDPM-100, GPU, gleiche Seeds): "
                f"{100 * published['successes'] / published['seeds']:.0f} %",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(layout.left_in / layout.width_in - 0.01, 1 - 0.95 / layout.height_in),
    )
    for ax, profile in zip(axes[0], profiles, strict=True):
        labels = ordered(profile.runs)
        xs = np.arange(len(labels), dtype=float)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0 %", "25 %", "50 %", "75 %", "100 %"])
        ax.set_xticks(xs, [LABELS.get(label, label) for label in labels])
        ax.grid(axis="x", visible=False)
        sampler = profile.meta.get("brain", {}).get("sampler", "")
        ax.set_title(f"{profile.name} ({sampler})", loc="left", pad=10)
        if published:
            ax.axhline(published["successes"] / published["seeds"], color=MUTED, linewidth=1.0, zorder=2)
        bbox = ax.get_window_extent()
        width = min(0.55, 24 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / bbox.width)
        for x, label in zip(xs, labels, strict=True):
            summary = profile.runs[label].summary
            bar_with_round_end(ax, x, summary.success_rate, width, run_color(label))
            low, high = summary.success_ci
            cap = width / 4
            ax.plot([x, x], [low, high], color=INK_2, linewidth=1.0, zorder=4)
            ax.plot([x - cap, x + cap], [low, low], color=INK_2, linewidth=1.0, zorder=4)
            ax.plot([x - cap, x + cap], [high, high], color=INK_2, linewidth=1.0, zorder=4)
            ax.annotate(
                f"{100 * summary.success_rate:.0f} %",
                (x + width / 2, summary.success_rate),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                color=INK,
                zorder=5,
            )
    axes[0][0].set_ylabel("Erfolgsquote")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_latency(profiles: list[ProfileData], budget_ms: float, path: Path) -> None:
    fig, axes, layout = make_figure(1, len(profiles), 4.9, 3.0, sharey=True, footer_in=0.75)
    header(
        fig,
        layout,
        "Latenzverteilung je Szenario",
        "Kumulative Verteilung der Ende-zu-Ende-Latenz (Beobachtung → Aktions-Chunk beim Roboter)",
    )
    max_ms = 0.0
    for profile in profiles:
        for label in NETWORK_RUNS:
            if label in profile.runs:
                values = [
                    r.e2e_ms for r in profile.runs[label].requests if r.phase == "realtime" and r.e2e_ms
                ]
                if values:
                    max_ms = max(max_ms, float(np.percentile(values, 99.5)))
    handles: list[Any] = []
    for ax, profile in zip(axes[0], profiles, strict=True):
        handles = []
        for label in NETWORK_RUNS:
            if label not in profile.runs:
                continue
            latencies = np.sort(
                [
                    r.e2e_ms
                    for r in profile.runs[label].requests
                    if r.phase == "realtime" and r.e2e_ms is not None
                ]
            )
            if len(latencies) == 0:
                continue
            y = np.arange(1, len(latencies) + 1) / len(latencies)
            (line,) = ax.step(
                latencies,
                y,
                where="post",
                color=SEVERITY[label],
                linewidth=2.0,
                label=LABELS[label].replace(chr(10), " "),
            )
            handles.append(line)
        ax.axvline(budget_ms, color=STATUS_CRITICAL, linewidth=1.0, zorder=2)
        ax.annotate(
            f"Budget {budget_ms:.0f} ms:\ndarüber verhungert\nder Roboter",
            (budget_ms, 0.04),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_2,
            va="bottom",
        )
        ax.set_xlim(0, max(budget_ms * 1.3, max_ms * 1.05))
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 0.95, 1.0], ["0", "", "p50", "", "p95", "1"])
        ax.set_xlabel("E2E-Latenz pro Anfrage, ms")
        sampler = profile.meta.get("brain", {}).get("sampler", "")
        ax.set_title(f"{profile.name} ({sampler})", loc="left", pad=10)
    axes[0][0].set_ylabel("Anteil der Anfragen ≤ x")
    axes[0][-1].legend(handles=handles, loc="lower right")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_breakdown(profiles: list[ProfileData], path: Path) -> None:
    components = [
        ("Inferenz im Brain", CATEGORICAL[0]),
        ("Netz + Transport", CATEGORICAL[1]),
        ("Warten auf nächsten Takt", CATEGORICAL[2]),
    ]
    fig, axes, layout = make_figure(
        1, len(profiles), 4.9, 2.2, sharex=True, left_in=1.25, wspace_in=1.3, header_in=1.25
    )
    header(fig, layout, "Wohin die Reaktionszeit geht", "Mittelwerte über alle Echtzeit-Anfragen")
    header_legend(fig, layout, [Patch(facecolor=c, label=n) for n, c in components], ncol=3)
    for ax, profile in zip(axes[0], profiles, strict=True):
        labels = [label for label in NETWORK_RUNS if label in profile.runs]
        ys = np.arange(len(labels))[::-1].astype(float)
        for y, label in zip(ys, labels, strict=True):
            requests = [r for r in profile.runs[label].requests if r.phase == "realtime" and r.status == "ok"]
            server = mean_or_nan([r.server_total_ms for r in requests if r.server_total_ms is not None])
            network = mean_or_nan([r.network_ms for r in requests if r.network_ms is not None])
            wait = mean_or_nan(
                [
                    r.reaction_ms - r.e2e_ms
                    for r in requests
                    if r.reaction_ms is not None and r.e2e_ms is not None
                ]
            )
            left = 0.0
            for value, (_, color) in zip((server, network, wait), components, strict=True):
                if math.isnan(value) or value <= 0:
                    continue
                ax.barh(
                    y, value, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2.0, zorder=3
                )
                left += value
            ax.annotate(
                f"{left:.0f} ms",
                (left, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK,
            )
        ax.set_yticks(ys, [LABELS[label].replace(chr(10), " ") for label in labels])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mittlere Reaktionszeit, ms")
        sampler = profile.meta.get("brain", {}).get("sampler", "")
        ax.set_title(f"{profile.name} ({sampler})", loc="left", pad=10)
    xmax = max(ax.get_xlim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_xlim(0, xmax * 1.15)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_queue(profile: ProfileData, seed: int, refill_threshold: int, path: Path) -> None:
    labels = [label for label in NETWORK_RUNS if label in profile.runs]
    fig, axes, layout = make_figure(len(labels), 1, 9.2, 1.25, sharex=True, header_in=1.25, hspace_in=0.55)
    sampler = profile.meta.get("brain", {}).get("sampler", "")
    header(
        fig,
        layout,
        f"Queue-Füllstand über Zeit · {profile.name} ({sampler})",
        f"Eine Episode je Szenario, identischer Startzustand (Seed {seed})",
    )
    header_legend(
        fig,
        layout,
        [
            Line2D([], [], color=SEVERITY["schlecht"], linewidth=2.0, label="Queue-Füllstand"),
            Line2D([], [], color=MUTED, linewidth=0.8, label=f"Refill-Schwelle ({refill_threshold})"),
            Line2D(
                [],
                [],
                color=STATUS_CRITICAL,
                marker="v",
                linestyle="none",
                label="Starvation (Aktion wiederholt)",
            ),
            Patch(facecolor=STALL_SHADE, label="Netz-Stall"),
        ],
        ncol=4,
    )
    for ax, label in zip(axes[:, 0], labels, strict=True):
        chosen, _ = final_episodes(profile.runs[label].episodes)
        episode = next((e for e in chosen if e.seed == seed), None)
        ax.set_ylim(-1.5, 16)
        ax.set_yticks([0, refill_threshold, 15])
        if episode is None:
            ax.set_title(
                f"{LABELS[label].replace(chr(10), ' ')}: Seed {seed} fehlt", loc="left", fontsize=9.5
            )
            continue
        t = np.arange(len(episode.ticks.queue_len)) / episode.fps
        for start, end in (episode.proxy or {}).get("stall_windows_s", []):
            ax.axvspan(start, end, color=STALL_SHADE, alpha=0.9, zorder=1, linewidth=0)
        ax.step(t, episode.ticks.queue_len, where="post", color=SEVERITY[label], linewidth=2.0, zorder=3)
        ax.axhline(refill_threshold, color=MUTED, linewidth=0.8, zorder=2)
        starved = np.flatnonzero(np.array(episode.ticks.starved, dtype=bool))
        if len(starved):
            ax.plot(
                t[starved],
                np.zeros(len(starved)),
                linestyle="none",
                marker="v",
                markersize=4.5,
                color=STATUS_CRITICAL,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                zorder=4,
            )
        outcome = "Erfolg" if episode.success else "kein Erfolg"
        ax.set_title(
            f"{LABELS[label].replace(chr(10), ' ')} · {outcome} nach {episode.duration_s:.1f} s · "
            f"{episode.starved_ticks} Ticks verhungert",
            loc="left",
            fontsize=9.5,
            pad=6,
        )
    axes[-1, 0].set_xlabel("Zeit seit Episodenstart, s")
    axes[0, 0].set_ylabel("Aktionen")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_starvation(profiles: list[ProfileData], path: Path) -> None:
    causes = [("Compute (Brain zu langsam)", CATEGORICAL[0]), ("Netz / Stall", CATEGORICAL[1])]
    fig, axes, layout = make_figure(1, len(profiles), 4.9, 2.6, sharey=True, header_in=1.25, footer_in=0.75)
    header(fig, layout, "Starvation und ihre Ursache", "Verhungerte Zeit pro Minute Laufzeit")
    header_legend(fig, layout, [Patch(facecolor=c, label=n) for n, c in causes], ncol=2)
    for ax, profile in zip(axes[0], profiles, strict=True):
        labels = [label for label in ordered(profile.runs) if label != "referenz_sync"]
        xs = np.arange(len(labels), dtype=float)
        for x, label in zip(xs, labels, strict=True):
            summary = profile.runs[label].summary
            fps = profile.runs[label].episodes[0].fps
            minutes = summary.episode_seconds / 60.0
            compute = summary.starved_ticks_compute / fps / minutes
            network = (summary.starved_ticks_network + summary.starved_ticks_none_pending) / fps / minutes
            ax.bar(x, compute, width=0.5, color=causes[0][1], edgecolor=SURFACE, linewidth=2.0, zorder=3)
            ax.bar(
                x,
                network,
                bottom=compute,
                width=0.5,
                color=causes[1][1],
                edgecolor=SURFACE,
                linewidth=2.0,
                zorder=3,
            )
            ax.annotate(
                f"{compute + network:.1f} s",
                (x, compute + network),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
        ax.set_xticks(xs, [LABELS.get(label, label) for label in labels])
        ax.grid(axis="x", visible=False)
        sampler = profile.meta.get("brain", {}).get("sampler", "")
        ax.set_title(f"{profile.name} ({sampler})", loc="left", pad=10)
    axes[0][0].set_ylabel("s verhungert pro min")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- markdown
def budget_ms(config: KortynConfig) -> float:
    """Latency after which the refill comes too late: the queue holds refill_threshold + 1 ticks of actions."""
    return 1000.0 * (config.agent.refill_threshold + 1) / config.env.fps


def table_main(profile: ProfileData, baseline: str) -> list[str]:
    rows = [
        "| Szenario | Netz (RTT) | n | Erfolg | 95 %-KI | vs. gut (McNemar p) | E2E p50 / p95 / p99 (ms) "
        "| Reaktion p50 (ms) | Netz-RTT p50 (ms) | Jitter (ms) | Starvation/min | verhungert | Payload ↑/↓ (B) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = profile.runs.get(baseline)
    for label in ordered(profile.runs):
        s = profile.runs[label].summary
        if base is not None and label not in (baseline, "referenz_sync"):
            _, only_base, only_run = paired_outcomes(base.summary.success_by_seed, s.success_by_seed)
            delta = s.success_rate - base.summary.success_rate
            vs = f"{100 * delta:+.0f} pp (p={mcnemar_exact(only_base, only_run):.2f})"
        else:
            vs = "–"
        realtime = s.mode == "realtime"
        rows.append(
            "| "
            + " | ".join(
                [
                    f"**{label}**",
                    network_label(profile.meta, label),
                    str(s.episodes),
                    pct(s.success_rate),
                    ci(s.success_ci),
                    vs,
                    f"{fmt(s.e2e_p50_ms)} / {fmt(s.e2e_p95_ms)} / {fmt(s.e2e_p99_ms)}",
                    fmt(s.reaction_p50_ms) if realtime else "–",
                    fmt(s.network_p50_ms, 1),
                    fmt(s.jitter_ipdv_ms, 1),
                    f"{s.starvation_events_per_min:.1f} Ereign. · {s.starved_ticks_per_min:.0f} Ticks"
                    if realtime
                    else "–",
                    pct(s.starved_time_fraction, 1) if realtime else "–",
                    f"{fmt(s.request_bytes_mean)} / {fmt(s.response_bytes_mean)}",
                ]
            )
            + " |"
        )
    return rows


def table_diagnostics(profile: ProfileData, budget: float) -> list[str]:
    rows = [
        "| Szenario | Anfragen/min | über Budget | Aktionsalter Ø (Ticks) | Starvation Compute/Netz (Ticks) "
        "| Inferenz p50 (ms) | Server p95 (ms) | Wire ↑/↓ pro Anfrage (B) | ungültige Versuche | max. Takt-Verspätung |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ordered(profile.runs):
        run = profile.runs[label]
        s = run.summary
        realtime = s.mode == "realtime"
        measured = [r.e2e_ms for r in run.requests if r.phase == "realtime" and r.e2e_ms is not None]
        over = (sum(1 for v in measured if v > budget) / len(measured)) if measured else None
        rows.append(
            "| "
            + " | ".join(
                [
                    f"**{label}**",
                    fmt(s.requests_per_min, 1),
                    pct(over, 1) if realtime else "–",
                    fmt(s.mean_action_age_ticks, 1),
                    f"{s.starved_ticks_compute} / {s.starved_ticks_network + s.starved_ticks_none_pending}"
                    if realtime
                    else "–",
                    fmt(s.server_inference_p50_ms),
                    fmt(s.server_total_p95_ms),
                    f"{fmt(s.wire_bytes_up_per_request)} / {fmt(s.wire_bytes_down_per_request)}"
                    if realtime
                    else "–",
                    str(s.invalid_attempts),
                    f"{s.max_tick_lateness_ms:.1f} ms" if realtime else "–",
                ]
            )
            + " |"
        )
    return rows


def validation_rows(
    published: dict[str, Any] | None, lerobot: dict[str, dict[str, Any]], profiles: list[ProfileData]
) -> list[str]:
    rows = [
        "| Stufe | Was isoliert wird | Erfolg | n |",
        "|---|---|---:|---:|",
    ]
    if published:
        rows.append(
            f"| A · publiziert (`eval_info.json`, DDPM-100, GPU) | Referenz | "
            f"{published['successes'] / published['seeds']:.0%} | {published['seeds']} |"
        )
    for _, summary in sorted(lerobot.items()):
        rows.append(
            f"| B · `lerobot-eval`, migrierter Checkpoint, {summary['sampler']} ({summary['device']}) | "
            f"Migration + Sampler | {summary['success_rate']:.0%} | {summary['episodes']} |"
        )
    for profile in profiles:
        for label, stage, isolates in (
            ("referenz_sync", "C", "Kortyn-Pipeline (synchron)"),
            ("gut", "D", "Echtzeit + Inferenzlatenz"),
            ("gut_wiederholung", "D'", "Lauf-zu-Lauf-Rauschen"),
        ):
            if label in profile.runs:
                s = profile.runs[label].summary
                sampler = profile.meta.get("brain", {}).get("sampler", profile.name)
                rows.append(
                    f"| {stage} · Kortyn `{label}`, {sampler} | {isolates} | {s.success_rate:.0%} | "
                    f"{s.episodes} |"
                )
    return rows


def write_report(
    config: KortynConfig, results_dir: Path, reports_dir: Path, profiles: list[ProfileData]
) -> Path:
    apply_style()
    reports_dir.mkdir(parents=True, exist_ok=True)
    published = read_json(results_dir / "published_reference.json")
    lerobot: dict[str, dict[str, Any]] = {}
    lerobot_dir = results_dir / "lerobot_eval"
    if lerobot_dir.is_dir():
        for path in sorted(lerobot_dir.glob("*/summary.json")):
            data = read_json(path)
            if data:
                lerobot[path.parent.name] = data
    calibration = read_json(results_dir / "calibration" / "summary.json")
    budget = budget_ms(config)

    figures = {
        "success": "phase1_erfolgsquote.png",
        "latency": "phase1_latenzverteilung.png",
        "breakdown": "phase1_latenz_zerlegung.png",
        "starvation": "phase1_starvation.png",
    }
    figure_success(profiles, published, reports_dir / figures["success"])
    figure_latency(profiles, budget, reports_dir / figures["latency"])
    figure_breakdown(profiles, reports_dir / figures["breakdown"])
    figure_starvation(profiles, reports_dir / figures["starvation"])
    seed = config.experiment.seed_start
    queue_figures: list[str] = []
    for profile in profiles:
        name = f"phase1_queue_{profile.name}.png"
        figure_queue(profile, seed, config.agent.refill_threshold, reports_dir / name)
        queue_figures.append(name)

    lines: list[str] = ["# Kortyn Phase 1 – Baseline-Messung", ""]
    lines.append(
        f"_Erzeugt {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} mit `python -m kortyn.report` "
        f"aus `{results_dir}/`._"
    )
    lines.append("")
    first_meta = profiles[0].meta
    host = first_meta.get("host", {})
    brain = first_meta.get("brain", {})
    lines += [
        "## Setup",
        "",
        f"- **Aufgabe:** PushT (gym-pusht via LeRobot-Konfiguration), {config.env.fps:g} Hz Steuertakt, "
        f"max. {config.env.max_episode_steps} Schritte, Erfolg = T-Block ≥ 95 % im Ziel.",
        f"- **Policy:** `{config.brain.checkpoint.repo_id}` @ `{config.brain.checkpoint.revision[:8]}` "
        f"(Diffusion Policy, migriert), LeRobot {brain.get('lerobot_version', '?')}, "
        f"torch {brain.get('torch_version', '?')}.",
        f"- **Brain-Hardware:** {host.get('platform', '?')}, {host.get('cpu_count', '?')} CPUs, "
        f"Device `{brain.get('device', '?')}`, {brain.get('torch_threads', '?')} Threads.",
        f"- **Agent:** Chunk {config.agent.actions_per_chunk} Aktionen, Refill bei ≤ {config.agent.refill_threshold} "
        f"→ Latenzbudget **{budget:.0f} ms**; Start „{config.agent.start_mode}“; Starvation = letzte Aktion "
        "wiederholen.",
        f"- **Seeds:** {config.experiment.seed_start}–{config.experiment.seed_start + config.experiment.episodes - 1}"
        " (identische Startzustände in allen Szenarien, gepaarte Auswertung), Szenarien seed-weise verschränkt.",
        "- **Netz:** Werte sind Round-Trip-Zeiten, je zur Hälfte auf beide Richtungen verteilt, Jitter "
        "gleichverteilt; Stall = Link friert in beide Richtungen ein.",
        "",
    ]
    for profile in profiles:
        sampler = profile.meta.get("brain", {}).get("sampler", profile.name)
        lines += [f"## Ergebnisse · Brain-Profil `{profile.name}` ({sampler})", ""]
        lines += table_main(profile, "gut")
        lines += ["", "<details><summary>Diagnose (Budget, Starvation-Ursache, Messhygiene)</summary>", ""]
        lines += table_diagnostics(profile, budget)
        lines += ["", "</details>", ""]

    lines += [
        "## Abbildungen",
        "",
        f"![Erfolgsquote je Szenario]({figures['success']})",
        "",
        f"![Latenzverteilung]({figures['latency']})",
        "",
        f"![Zerlegung der Reaktionszeit]({figures['breakdown']})",
        "",
        f"![Starvation und Ursache]({figures['starvation']})",
        "",
    ]
    for name in queue_figures:
        lines += [f"![Queue-Füllstand über Zeit]({name})", ""]

    lines += ["## Validierungskette (Abnahmekriterium 2)", ""]
    lines += validation_rows(published, lerobot, profiles)
    lines += [""]
    if calibration:
        lines += [
            "## Kalibrierung des Brain-Profils `kalibriert`",
            "",
            f"Gate: keine Starvation in „gut“ und E2E-p99 ≤ {calibration['max_ticks']:g} Ticks. "
            f"Empfehlung: **DDIM-{calibration['recommended_steps']}**.",
            "",
            "| DDIM-Schritte | Inferenz p50 (ms) | E2E p50 (ms) | E2E p99 (ms) | p99 in Ticks | Starvation/min | Gate |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in calibration["rows"]:
            lines.append(
                f"| {row['num_inference_steps']} | {fmt(row['server_inference_p50_ms'])} | {fmt(row['e2e_p50_ms'])}"
                f" | {fmt(row['e2e_p99_ms'])} | {row['e2e_p99_ticks']:.1f} | {row['starved_ticks_per_min']:.1f} | "
                f"{'✅' if row['passed'] else '❌'} |"
            )
        lines.append("")

    lines += [
        "## Definitionen",
        "",
        "- **E2E-Latenz:** Beobachtung erfasst → Aktions-Chunk beim Roboter.",
        "- **Reaktionszeit:** Beobachtung erfasst → erste Aktion aus diesem Chunk ausgeführt (inkl. Warten auf "
        "den nächsten Takt).",
        "- **Netz-RTT:** Zeit auf der Leitung (Anfrage-RTT minus Serverzeit). **Jitter:** mittlere |ΔRTT| "
        "aufeinanderfolgender Anfragen (RFC 5481).",
        "- **Starvation:** Takt ohne frische Aktion – der Roboter wiederholt die letzte Aktion (PushT: hält das "
        "letzte Positionsziel). Ereignis = zusammenhängende Folge solcher Takte. **Compute** = wäre auch ohne "
        "Netz zu spät gewesen, sonst **Netz/Stall**.",
        "- **über Budget:** Anteil der Anfragen mit E2E-Latenz über dem Budget (dann verhungert der Roboter, "
        "sofern keine Reserve in der Queue ist).",
        "- **Payload:** serialisierte Protobuf-Größe; **Wire** = Bytes durch den Proxy inkl. HTTP/2-Overhead.",
        "- **Gültigkeit:** Episoden mit Takt-Verspätung > "
        f"{config.agent.max_tick_lateness_ms:g} ms, Proxy-Verspätung > {config.agent.max_proxy_lateness_ms:g} ms, "
        "Reconnect, CPU-Steal > "
        f"{config.agent.max_cpu_steal_fraction:.0%} oder Infrastrukturfehler werden wiederholt.",
        "",
    ]
    report = reports_dir / "phase1.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create reports/phase1.md and figures from results/.")
    parser.add_argument("--config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout
    )
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        LOG.error("%s", exc)
        return 2
    results_dir = Path(args.results_dir or config.experiment.results_dir)
    reports_dir = Path(args.reports_dir or config.experiment.reports_dir)
    profiles: list[ProfileData] = []
    for suite in config.experiment.suites:
        directory = results_dir / suite.profile
        if directory.is_dir():
            data = load_profile(directory)
            if data is not None:
                profiles.append(data)
    if not profiles:
        LOG.error("no results found in %s; run python -m kortyn.run_experiment first", results_dir)
        return 1
    report = write_report(config, results_dir, reports_dir, profiles)
    LOG.info("report written to %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
