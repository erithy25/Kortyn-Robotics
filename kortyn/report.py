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
from collections.abc import Callable, Sequence
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
from matplotlib.ticker import FuncFormatter

from kortyn.config import DEFAULT_BASE_CONFIG, ConfigError, KortynConfig, load_config
from kortyn.metrics import (
    EpisodeRecord,
    RequestRecord,
    RunSummary,
    bootstrap_ci,
    final_episodes,
    mcnemar_exact,
    mean,
    paired_mean_difference,
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
ABLATION_SUFFIX = "_ohne_stall"  # <scenario>_ohne_stall = <scenario> with the same latency but no stalls
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


def header(fig: Any, layout: Layout, title: str, subtitle: str | None = None) -> float:
    """Title and (wrapped) subtitle; returns where a header legend can start, in inches from the top."""
    x = layout.left_in / layout.width_in
    fig.text(
        x, 1 - 0.14 / layout.height_in, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK
    )
    if not subtitle:
        return 0.45
    # ~0.065 in per character at 8.5 pt: wrap instead of running off the canvas on narrow figures
    chars = max(40, int((layout.width_in - layout.left_in - 0.2) / 0.065))
    wrapped = textwrap.fill(subtitle, chars)
    fig.text(
        x,
        1 - 0.45 / layout.height_in,
        wrapped,
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK_2,
        linespacing=1.3,
    )
    return 0.45 + 0.155 * (wrapped.count("\n") + 1) + 0.03


def header_legend(fig: Any, layout: Layout, handles: list[Any], ncol: int, top_in: float = 0.62) -> None:
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(layout.left_in / layout.width_in - 0.01, 1 - top_in / layout.height_in),
        ncol=ncol,
        handlelength=1.6,
        columnspacing=1.4,
    )


def published_rate(published: dict[str, Any] | None) -> float | None:
    """Population success rate of the published evaluation (all episodes), else the rate on our seeds."""
    if not published:
        return None
    if published.get("all_success_rate") is not None:
        return float(published["all_success_rate"])
    return float(published["successes"]) / float(published["seeds"]) if published.get("seeds") else None


def percent_axis(ax: Axes, decimals: int = 0) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:.{decimals}f} %"))


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
    others = [label for label in runs if label not in SCENARIO_ORDER]
    return known + sorted(others, key=lambda label: (label.endswith(ABLATION_SUFFIX), label))


def figure_runs(profile: ProfileData) -> list[str]:
    """Runs shown in the overview figures; ablation runs get their own figure."""
    return [label for label in ordered(profile.runs) if not label.endswith(ABLATION_SUFFIX)]


def ablation_pairs(profile: ProfileData) -> list[tuple[str, str]]:
    """(run without stalls, run with stalls) for every ablation run whose counterpart was measured."""
    pairs: list[tuple[str, str]] = []
    for label in ordered(profile.runs):
        if label.endswith(ABLATION_SUFFIX) and label[: -len(ABLATION_SUFFIX)] in profile.runs:
            pairs.append((label, label[: -len(ABLATION_SUFFIX)]))
    return pairs


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
    narrow = len(profiles) == 1  # subtitle wraps and the legend needs one row per entry
    fig, axes, layout = make_figure(
        1, len(profiles), 4.9, 3.0, sharey=True, footer_in=0.75, header_in=1.85 if narrow else 1.45
    )
    top = header(
        fig,
        layout,
        "Erfolgsquote je Szenario",
        "Gleiche Seeds in allen Szenarien · Farbe = Netz-Schweregrad (hell = gut, dunkel = schlecht), "
        "grau = synchrone Referenz",
    )
    legend_handles: list[Any] = [Line2D([], [], color=INK_2, linewidth=1.0, label="Wilson-95 %-KI")]
    reference = published_rate(published)
    if published and reference is not None:
        legend_handles.append(
            Line2D(
                [],
                [],
                color=MUTED,
                linewidth=1.0,
                label=f"publiziert (DDPM-100, GPU): {100 * reference:.0f} % über {published['all_episodes']} "
                f"Episoden ({100 * published['successes'] / published['seeds']:.0f} % auf diesen Seeds)",
            )
        )
    header_legend(fig, layout, legend_handles, ncol=1 if narrow else 2, top_in=top)
    for ax, profile in zip(axes[0], profiles, strict=True):
        labels = figure_runs(profile)
        xs = np.arange(len(labels), dtype=float)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0 %", "25 %", "50 %", "75 %", "100 %"])
        ax.set_xticks(xs, [LABELS.get(label, label) for label in labels])
        ax.grid(axis="x", visible=False)
        sampler = profile.meta.get("brain", {}).get("sampler", "")
        ax.set_title(f"{profile.name} ({sampler})", loc="left", pad=10)
        if reference is not None:
            ax.axhline(reference, color=MUTED, linewidth=1.0, zorder=2)
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
        below = [
            r.e2e_ms <= budget_ms
            for label in NETWORK_RUNS
            if label in profile.runs
            for r in profile.runs[label].requests
            if r.phase == "realtime" and r.e2e_ms is not None
        ]
        text_left = bool(below) and sum(below) < len(below) / 2  # curves right of the budget: write left
        ax.annotate(
            f"Budget {budget_ms:.0f} ms:\ndarüber verhungert\nder Roboter",
            (budget_ms, 0.04),
            xytext=(-4 if text_left else 4, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_2,
            ha="right" if text_left else "left",
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
    top = header(fig, layout, "Wohin die Reaktionszeit geht", "Mittelwerte über alle Echtzeit-Anfragen")
    header_legend(fig, layout, [Patch(facecolor=c, label=n) for n, c in components], ncol=3, top_in=top)
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
    top = header(
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
        top_in=top,
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
    top = header(fig, layout, "Starvation und ihre Ursache", "Verhungerte Zeit pro Minute Laufzeit")
    header_legend(fig, layout, [Patch(facecolor=c, label=n) for n, c in causes], ncol=2, top_in=top)
    for ax, profile in zip(axes[0], profiles, strict=True):
        labels = [label for label in figure_runs(profile) if label != "referenz_sync"]
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


def figure_ablation(profile: ProfileData, baseline: str, path: Path) -> None:
    """Latency alone vs. latency + stalls: the stall-free ablation runs next to the specified scenarios."""
    pairs = ablation_pairs(profile)
    no_stall_of = {stalled: ablation for ablation, stalled in pairs}
    levels = [label for label in (baseline, "buero_wlan") if label in profile.runs]
    levels += [stalled for _, stalled in pairs]
    # (legend, colour, [(x level, run label)], x offset, label side)
    series = [
        (
            "ohne Stall (nur Latenz + Jitter)",
            CATEGORICAL[0],
            [(lv, no_stall_of.get(lv, lv)) for lv in levels],
            -1,
        ),
        (
            "mit Stall (Szenario wie spezifiziert)",
            CATEGORICAL[1],
            [(lv, lv) for lv in levels if lv in no_stall_of],
            1,
        ),
    ]

    def reward_ci(summary: RunSummary) -> tuple[float, float] | None:
        return bootstrap_ci([[value] for value in summary.max_reward_by_seed.values()], mean)

    panels: list[tuple[str, Callable[[RunSummary], float | None], Callable[[RunSummary], Any], str]] = [
        ("Erfolgsquote", lambda s: s.success_rate, lambda s: s.success_ci, "pct0"),
        ("Ø Max-Reward", lambda s: s.max_reward_mean, reward_ci, "f3"),
        ("Zeit verhungert", lambda s: s.starved_time_fraction, lambda s: None, "pct1"),
    ]
    fig, axes, layout = make_figure(1, len(panels), 3.1, 2.7, header_in=1.35, footer_in=0.85, wspace_in=0.7)
    top = header(
        fig,
        layout,
        "Ablation: Latenz oder Stall?",
        f"Brain-Profil {profile.name} · gleiche Seeds · Whisker = 95 %-KI (Erfolg: Wilson, Reward: Bootstrap "
        "über Seeds)",
    )
    header_legend(
        fig,
        layout,
        [Line2D([], [], color=c, marker="o", markersize=6, linewidth=2.0, label=n) for n, c, _, _ in series],
        ncol=2,
        top_in=top,
    )
    x_of = {level: float(index) for index, level in enumerate(levels)}
    for ax, (title, value_of, ci_of, style) in zip(axes[0], panels, strict=True):
        lows: list[float] = []
        highs: list[float] = []
        ends: list[tuple[float, float]] = []
        for _, color, points, side in series:
            xs = [x_of[level] + 0.08 * side for level, _ in points]
            ys = [float(value_of(profile.runs[label].summary) or 0.0) for _, label in points]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                zorder=3,
            )
            for x, y, (_, label) in zip(xs, ys, points, strict=True):
                bounds = ci_of(profile.runs[label].summary)
                low, high = bounds if bounds is not None else (y, y)
                ax.plot([x, x], [low, high], color=color, linewidth=1.0, zorder=2)
                lows.append(low)
                highs.append(high)
            ends.append((xs[-1], ys[-1]))
        if style == "f3":
            ax.set_ylim(min(0.8, math.floor((min(lows) - 0.01) * 20) / 20), 1.0)
        elif style == "pct0":
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylim(0.0, max(0.02, 1.15 * max(highs)))
        if style.startswith("pct"):
            percent_axis(ax, 0 if style == "pct0" else 1)
        # values of the last level, right of both series, nudged apart when they would overlap
        y_low, y_high = ax.get_ylim()
        gap = 0.075 * (y_high - y_low)
        order = sorted(range(len(ends)), key=lambda i: ends[i][1])
        placed = [ends[i][1] for i in order]
        for k in range(1, len(placed)):
            if placed[k] - placed[k - 1] < gap:
                middle = (placed[k] + placed[k - 1]) / 2
                placed[k - 1], placed[k] = middle - gap / 2, middle + gap / 2
        x_text = max(x for x, _ in ends) + 0.14
        for index, y_text in zip(order, placed, strict=True):
            value = ends[index][1]
            text = f"{value:.3f}" if style == "f3" else f"{100 * value:.{0 if style == 'pct0' else 1}f} %"
            ax.text(x_text, y_text, text, ha="left", va="center", fontsize=8, color=INK)
        ax.set_xlim(-0.5, len(levels) - 0.05)
        ax.set_xticks(
            list(x_of.values()),
            [network_label(profile.meta, no_stall_of.get(lv, lv)).replace(" ± ", "±") for lv in levels],
        )
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", pad=8)
    axes[0][len(panels) // 2].set_xlabel("Netz-RTT (Round-Trip ± Jitter)")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


SHORT_LABELS = {
    "referenz_sync": "Sync",
    "gut": "gut",
    "buero_wlan": "Büro-WLAN",
    "schlecht": "schlecht",
    "sehr_schlecht": "sehr schlecht",
}


def annotate_points(
    ax: Axes, points: list[tuple[float, float]], labels: list[tuple[float, float, str]], fontsize: float = 7.5
) -> None:
    """Direct labels that avoid the canvas edge, all other points and each other (first free position)."""
    fig = ax.figure
    box = ax.get_window_extent()
    to_axes = ax.transData + ax.transAxes.inverted()
    px_per_pt = fig.dpi / 72.0
    occupied = [tuple(to_axes.transform(point)) for point in points]
    taken: list[tuple[float, float, float, float]] = []
    marker = 5.0 * px_per_pt / box.width  # half a marker, in axes fractions
    for x, y, text in sorted(labels, key=lambda label: -label[0]):  # the right edge constrains most
        ax_x, ax_y = to_axes.transform((x, y))
        width = 0.6 * fontsize * len(text) * px_per_pt / box.width
        height = 1.25 * fontsize * px_per_pt / box.height
        below = -3 - 1.25 * fontsize
        candidates: list[tuple[float, float, str]] = [
            (6, 3, "left"),
            (6, below, "left"),
            (-6, 3, "right"),
            (-6, below, "right"),
            (0, 7, "center"),
            (0, below - 4, "center"),
            (0, 20, "center"),
        ]
        for dx, dy, ha in candidates:
            shift = {"left": 0.0, "right": width, "center": width / 2}[ha]
            x0 = ax_x + dx * px_per_pt / box.width - shift
            y0 = ax_y + dy * px_per_pt / box.height
            x1, y1 = x0 + width, y0 + height
            inside = x0 >= 0.0 and x1 <= 1.03 and y0 >= 0.0 and y1 <= 1.0  # may reach into the right margin
            clear = all(
                not (x0 - marker < ox < x1 + marker and y0 - marker < oy < y1 + marker)
                for ox, oy in occupied
                if (ox, oy) != (ax_x, ax_y)
            )
            free = all(x1 < a0 or x0 > a1 or y1 < b0 or y0 > b1 for a0, b0, a1, b1 in taken)
            if inside and clear and free:
                taken.append((x0, y0, x1, y1))
                ax.annotate(
                    text,
                    (x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=fontsize,
                    color=INK_2,
                    ha=ha,
                    va="bottom",
                )
                break


def figure_age(profiles: list[ProfileData], path: Path) -> None:
    """Quality against the age of the executed actions, for every run of every profile."""
    runs = [(profile, label) for profile in profiles for label in ordered(profile.runs)]
    counts = sorted({profile.runs[label].summary.episodes for profile, label in runs})
    episodes = f"{counts[0]}" if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
    fig, axes, layout = make_figure(1, 2, 4.3, 2.9, header_in=1.4, footer_in=0.8, wspace_in=0.8)
    top = header(
        fig,
        layout,
        "Je älter der ausgeführte Plan, desto schlechter",
        f"Jeder Punkt = ein Lauf ({episodes} Episoden, gleiche Seeds) · x = mittleres Alter der ausgeführten "
        "Aktion (Beobachtung → Ausführung) · Konfidenzintervalle in den Tabellen",
    )
    colors = {profile.name: CATEGORICAL[index % len(CATEGORICAL)] for index, profile in enumerate(profiles)}
    handles: list[Any] = [
        Line2D(
            [],
            [],
            color=colors[profile.name],
            marker="o",
            markersize=6,
            linestyle="none",
            label=f"{profile.name} ({profile.meta.get('brain', {}).get('sampler', '')})",
        )
        for profile in profiles
    ]
    if any(profile.runs[label].summary.mode == "sync" for profile, label in runs):
        handles.append(
            Line2D(
                [],
                [],
                color=INK_2,
                marker="o",
                markerfacecolor=SURFACE,
                markersize=6,
                linestyle="none",
                label="hohl = synchrone Referenz",
            )
        )
    if any(label.endswith(ABLATION_SUFFIX) for _, label in runs):
        handles.append(
            Line2D(
                [],
                [],
                color=INK_2,
                marker="D",
                markersize=5,
                linestyle="none",
                label="Raute = Ablation ohne Stall",
            )
        )
    header_legend(fig, layout, handles, ncol=len(handles), top_in=top)
    panels: list[tuple[str, Callable[[RunSummary], float | None], bool]] = [
        ("Erfolgsquote", lambda s: s.success_rate, True),
        ("Ø Max-Reward", lambda s: s.max_reward_mean, False),
    ]
    for ax, (title, value_of, is_rate) in zip(axes[0], panels, strict=True):
        points: list[tuple[float, float, str, str, str]] = []
        for profile in profiles:
            for label in ordered(profile.runs):
                run = profile.runs[label]
                value = value_of(run.summary)
                if run.summary.mean_action_age_ticks is None or value is None:
                    continue
                age_s = run.summary.mean_action_age_ticks / run.episodes[0].fps
                points.append((age_s, float(value), label, profile.name, run.summary.mode))
        for age_s, value, label, name, mode in points:
            sync = mode == "sync"
            ax.plot(
                [age_s],
                [value],
                marker="D" if label.endswith(ABLATION_SUFFIX) else "o",
                markersize=6 if label.endswith(ABLATION_SUFFIX) else 7,
                linestyle="none",
                color=colors[name],
                markerfacecolor=SURFACE if sync else colors[name],
                markeredgecolor=colors[name] if sync else SURFACE,
                markeredgewidth=1.5,
                zorder=3,
            )
        ax.set_xlim(0, max(p[0] for p in points) * 1.12)
        if is_rate:
            ax.set_ylim(0, 1.0)
            percent_axis(ax)
        else:
            ax.set_ylim(min(0.8, math.floor((min(p[1] for p in points) - 0.02) * 20) / 20), 1.0)
        named: list[tuple[float, float, str]] = []
        for age_s, value, label, name, _ in points:
            text = SHORT_LABELS.get(label)
            if text is not None and not (label == "referenz_sync" and name != profiles[0].name):
                named.append((age_s, value, text))
        annotate_points(ax, [(p[0], p[1]) for p in points], named)
        ax.set_xlabel("Ø Alter der ausgeführten Aktion, s")
        ax.set_title(title, loc="left", pad=8)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- markdown
def budget_ms(config: KortynConfig) -> float:
    """Latency after which the refill comes too late: the queue holds refill_threshold + 1 ticks of actions."""
    return 1000.0 * (config.agent.refill_threshold + 1) / config.env.fps


def reward_delta(base: RunSummary, run: RunSummary) -> str:
    result = paired_mean_difference(base.max_reward_by_seed, run.max_reward_by_seed)
    if result is None:
        return "–"
    delta, bounds = result
    return f"{delta:+.3f}" + ("" if bounds is None else f" [{bounds[0]:+.3f}; {bounds[1]:+.3f}]")


def table_main(profile: ProfileData, baseline: str) -> list[str]:
    rows = [
        "| Szenario | Netz (RTT) | n | Erfolg [95 %-KI] | Δ Erfolg vs. gut (McNemar p) | Ø Max-Reward "
        "| Δ Reward vs. gut [95 %-KI] | E2E p50 / p95 / p99 (ms) | Reaktion p50 (ms) | Netz-RTT p50 (ms) "
        "| Jitter (ms) | Starvation-Ereignisse/min | verhungert | Payload ↑/↓ (B) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = profile.runs.get(baseline)
    for label in ordered(profile.runs):
        s = profile.runs[label].summary
        if base is not None and label not in (baseline, "referenz_sync"):
            _, only_base, only_run = paired_outcomes(base.summary.success_by_seed, s.success_by_seed)
            delta = s.success_rate - base.summary.success_rate
            vs = f"{100 * delta:+.0f} pp (p={mcnemar_exact(only_base, only_run):.2f})"
            vs_reward = reward_delta(base.summary, s)
        else:
            vs = vs_reward = "–"
        realtime = s.mode == "realtime"
        rows.append(
            "| "
            + " | ".join(
                [
                    f"**{label}**",
                    network_label(profile.meta, label),
                    str(s.episodes),
                    f"{pct(s.success_rate)} {ci(s.success_ci)}",
                    vs,
                    fmt(s.max_reward_mean, 3),
                    vs_reward,
                    f"{fmt(s.e2e_p50_ms)} / {fmt(s.e2e_p95_ms)} / {fmt(s.e2e_p99_ms)}",
                    fmt(s.reaction_p50_ms) if realtime else "–",
                    fmt(s.network_p50_ms, 1),
                    fmt(s.jitter_ipdv_ms, 1),
                    f"{s.starvation_events_per_min:.1f}" if realtime else "–",
                    pct(s.starved_time_fraction, 1) if realtime else "–",
                    f"{fmt(s.request_bytes_mean)} / {fmt(s.response_bytes_mean)}",
                ]
            )
            + " |"
        )
    return rows


def stall_starvation_share(episodes: Sequence[EpisodeRecord], recovery_s: float) -> float | None:
    """Share of starved ticks that fall into a stall window or the ``recovery_s`` after it."""
    starved_total = 0
    near_stall = 0
    for episode in episodes:
        windows = (episode.proxy or {}).get("stall_windows_s", [])
        for index, starved in enumerate(episode.ticks.starved):
            if not starved:
                continue
            starved_total += 1
            t = index / episode.fps
            if any(start <= t <= end + recovery_s for start, end in windows):
                near_stall += 1
    return near_stall / starved_total if starved_total else None


def median_success_duration(episodes: Sequence[EpisodeRecord]) -> float | None:
    durations = [e.duration_s for e in episodes if e.success]
    return float(np.median(durations)) if durations else None


def table_diagnostics(profile: ProfileData, budget: float) -> list[str]:
    rows = [
        "| Szenario | Dauer bis Erfolg (Median) | Anfragen/min | über Budget | Aktionsalter Ø (Ticks) "
        "| Starvation Compute/Netz (Ticks) | davon in Stall-Nähe | Inferenz p50 (ms) | Server p95 (ms) "
        "| Wire ↑/↓ pro Anfrage (B) | ungültige Versuche | max. Takt-Verspätung |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ordered(profile.runs):
        run = profile.runs[label]
        s = run.summary
        realtime = s.mode == "realtime"
        chosen, _ = final_episodes(run.episodes)
        measured = [r.e2e_ms for r in run.requests if r.phase == "realtime" and r.e2e_ms is not None]
        over = (sum(1 for v in measured if v > budget) / len(measured)) if measured else None
        stall_share = stall_starvation_share(chosen, recovery_s=budget / 1000.0) if realtime else None
        rows.append(
            "| "
            + " | ".join(
                [
                    f"**{label}**",
                    fmt(median_success_duration(chosen), 1, " s"),
                    fmt(s.requests_per_min, 1),
                    pct(over, 1) if realtime else "–",
                    fmt(s.mean_action_age_ticks, 1),
                    f"{s.starved_ticks_compute} / {s.starved_ticks_network + s.starved_ticks_none_pending}"
                    if realtime
                    else "–",
                    pct(stall_share) if stall_share is not None else "–",
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


def table_overview(profiles: list[ProfileData]) -> list[str]:
    """One row per scenario, one column per Brain profile: success and starved time at a glance."""
    header_cells = ["Szenario", "Netz (RTT)"]
    for profile in profiles:
        sampler = profile.meta.get("brain", {}).get("sampler", profile.name)
        header_cells.append(f"{profile.name} ({sampler}): Erfolg · verhungert")
    rows = ["| " + " | ".join(header_cells) + " |", "|---|---|" + "---:|" * len(profiles)]
    labels: list[str] = []
    for profile in profiles:
        labels += [label for label in ordered(profile.runs) if label not in labels]
    for label in labels:
        meta = next(p.meta for p in profiles if label in p.runs)
        cells = [f"**{label}**", network_label(meta, label)]
        for profile in profiles:
            run = profile.runs.get(label)
            if run is None:
                cells.append("–")
                continue
            s = run.summary
            starved = pct(s.starved_time_fraction, 1) if s.mode == "realtime" else "–"
            cells.append(f"{pct(s.success_rate)} · {starved}")
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def table_ablation(profile: ProfileData, baseline: str) -> list[str]:
    rows = [
        "| Netz (RTT) | Erfolg ohne → mit Stall | Ø Max-Reward ohne → mit Stall "
        "| Δ Reward durch Latenz (ohne Stall − gut) | Δ Reward durch Stall (mit − ohne) "
        "| verhungert ohne → mit Stall | Aktionsalter Ø ohne → mit (Ticks) | E2E p99 ohne → mit (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = profile.runs[baseline].summary if baseline in profile.runs else None
    for ablation, stalled in ablation_pairs(profile):
        a = profile.runs[ablation].summary
        s = profile.runs[stalled].summary
        rows.append(
            "| "
            + " | ".join(
                [
                    network_label(profile.meta, ablation),
                    f"{pct(a.success_rate)} → {pct(s.success_rate)}",
                    f"{fmt(a.max_reward_mean, 3)} → {fmt(s.max_reward_mean, 3)}",
                    reward_delta(base, a) if base is not None else "–",
                    reward_delta(a, s),
                    f"{pct(a.starved_time_fraction, 1)} → {pct(s.starved_time_fraction, 1)}",
                    f"{fmt(a.mean_action_age_ticks, 1)} → {fmt(s.mean_action_age_ticks, 1)}",
                    f"{fmt(a.e2e_p99_ms)} → {fmt(s.e2e_p99_ms)}",
                ]
            )
            + " |"
        )
    return rows


def validation_rows(
    published: dict[str, Any] | None, lerobot: dict[str, dict[str, Any]], profiles: list[ProfileData]
) -> list[str]:
    rows = [
        "| Stufe | Was isoliert wird | Erfolg | Ø Max-Reward | n |",
        "|---|---|---:|---:|---:|",
    ]
    if published:
        if published.get("all_success_rate") is not None:
            rows.append(
                f"| A₀ · publiziert, alle Episoden (DDPM-100, GPU) | Populationswert des Checkpoints | "
                f"{pct(published['all_success_rate'], 1)} | {fmt(published.get('all_avg_max_reward'), 3)} | "
                f"{published['all_episodes']} |"
            )
        rows.append(
            f"| A · publiziert, Seeds wie hier (`eval_info.json`, DDPM-100, GPU) | Referenz | "
            f"{pct(published['successes'] / published['seeds'])} | {fmt(published.get('avg_max_reward'), 3)} | "
            f"{published['seeds']} |"
        )
    for _, summary in sorted(lerobot.items()):
        rows.append(
            f"| B · `lerobot-eval`, migrierter Checkpoint, {summary['sampler']} ({summary['device']}) | "
            f"Migration + Sampler | {pct(summary['success_rate'])} | {fmt(summary.get('avg_max_reward'), 3)} | "
            f"{summary['episodes']} |"
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
                    f"| {stage} · Kortyn `{label}`, {sampler} | {isolates} | {pct(s.success_rate)} | "
                    f"{fmt(s.max_reward_mean, 3)} | {s.episodes} |"
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
        "age": "phase1_aktionsalter.png",
    }
    figure_success(profiles, published, reports_dir / figures["success"])
    figure_latency(profiles, budget, reports_dir / figures["latency"])
    figure_breakdown(profiles, reports_dir / figures["breakdown"])
    figure_starvation(profiles, reports_dir / figures["starvation"])
    figure_age(profiles, reports_dir / figures["age"])
    ablation_figures: dict[str, str] = {}
    for profile in profiles:
        if ablation_pairs(profile) and "gut" in profile.runs:
            name = f"phase1_ablation_{profile.name}.png"
            figure_ablation(profile, "gut", reports_dir / name)
            ablation_figures[profile.name] = name
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
    lines += ["## Auf einen Blick", ""]
    lines += table_overview(profiles)
    lines += [
        "",
        f"Erfolg = Anteil gelöster Episoden (n = {config.experiment.episodes} je Zeile); verhungert = Anteil der "
        "Takte ohne frische Aktion. Details, Konfidenzintervalle und Tests in den Tabellen unten.",
        "",
    ]
    for profile in profiles:
        sampler = profile.meta.get("brain", {}).get("sampler", profile.name)
        lines += [f"## Ergebnisse · Brain-Profil `{profile.name}` ({sampler})", ""]
        lines += table_main(profile, "gut")
        if "gut_wiederholung" in profile.runs and "gut" in profile.runs:
            noise = reward_delta(profile.runs["gut"].summary, profile.runs["gut_wiederholung"].summary)
            lines += [
                "",
                f"Lesehilfe: `gut_wiederholung` ist exakt dieselbe Bedingung wie `gut`. Ihr Δ ({noise} Reward) "
                "zeigt das Rauschen zwischen zwei Läufen (Echtzeit ist nicht bit-genau wiederholbar). Kleinere "
                "Effekte sind mit n = 50 nicht belastbar.",
            ]
        lines += ["", "<details><summary>Diagnose (Budget, Starvation-Ursache, Messhygiene)</summary>", ""]
        lines += table_diagnostics(profile, budget)
        lines += ["", "</details>", ""]
        if profile.name in ablation_figures:
            lines += [
                f"### Ablation: Latenz oder Stall? (`{profile.name}`)",
                "",
                "Gleiche Latenz und gleicher Jitter wie im Szenario, aber ohne Stalls. „Δ durch Latenz“ ist der "
                "Effekt von Delay + Jitter allein gegenüber „gut“, „Δ durch Stall“ der zusätzliche Effekt der "
                "Aussetzer (gepaarte Differenzen über dieselben Seeds, Bootstrap-95 %-KI).",
                "",
            ]
            lines += table_ablation(profile, "gut")
            lines += ["", f"![Ablation Latenz vs. Stall]({ablation_figures[profile.name]})", ""]

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
        f"![Aktionsalter und Qualität]({figures['age']})",
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
        "Netz zu spät gewesen, sonst **Netz/Stall**. Lesehilfe: In Netz-Szenarien entsteht der Compute-Anteil "
        "meist als Nachlauf eines Stalls – ist die Queue einmal leer, braucht die nächste Anfrage mindestens eine "
        "Inferenzzeit, bevor wieder Aktionen da sind.",
        "- **über Budget:** Anteil der Anfragen mit E2E-Latenz über dem Budget (dann verhungert der Roboter, "
        "sofern keine Reserve in der Queue ist). **in Stall-Nähe:** Anteil der verhungerten Takte, die in ein "
        "Stall-Fenster oder die Budget-Dauer danach fallen.",
        "- **Reaktionszeit** existiert nur für Chunks, die tatsächlich ausgeführt wurden; komplett veraltete Chunks "
        "gehen in die E2E-Latenz und die Starvation ein, nicht in die Reaktionszeit.",
        "- **Ø Max-Reward:** LeRobots `max_reward` pro Episode (PushT: beste Abdeckung / 95 %, gedeckelt auf 1; "
        "1 = Erfolg), gemittelt. Kontinuierlich und daher statistisch trennschärfer als die Erfolgsquote; "
        "Δ-Werte sind gepaarte Differenzen über dieselben Seeds mit Bootstrap-95 %-KI.",
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
