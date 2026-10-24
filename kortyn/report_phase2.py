"""Phase-2 report (freshness engine): ``reports/phase2.md`` and its figures.

    python -m kortyn.report                    # phase 1 and phase 2, whatever results exist
    python -m kortyn.report --phase 2
    python -m kortyn.report_phase2 --select    # stage-1 ranking and the two strategies for stage 2

Inputs (below ``results/`` by default):

* stage 1, screening: ``phase2/screening/<profile>/<strategy>__<scenario>/`` (every strategy, 20 seeds);
* stage 2, confirmation: ``phase2/confirmation/<profile>/<strategy>__<scenario>/`` (two strategies, 50 seeds);
* phase 1: ``<profile>/<scenario>/``. The phase-1 runs are the baseline configuration on the same seeds. They
  are the stage-2 baseline unless stage 2 measured its own ``baseline__<scenario>`` runs, they provide the
  synchronous reference, and they are the gray reference points of the plan-age figures. Stage 1 always
  measures a fresh baseline: comparing it with phase 1 on the same seeds is the reproduction check
  (acceptance criterion 2).

Selection rule for stage 2, fixed before stage 1 ran (:func:`select_strategies`): per strategy the paired
max-reward difference against the baseline on the same seeds, averaged over the screening scenarios. The
best strategy is picked first; among strategies within ``TIE_MARGIN`` of it, the one with fewer requests per
second wins (less Brain load). Then the second one the same way.

Statistics: success rates with Wilson intervals; success against the baseline with the exact McNemar test;
max reward against the baseline as the paired mean difference with a seed-bootstrap interval and a sign-flip
permutation test. Holm-adjusted p-values control the family-wise error over all comparisons of a stage.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from kortyn.config import DEFAULT_BASE_CONFIG, ConfigError, KortynConfig, load_config
from kortyn.metrics import (
    EpisodeRecord,
    RequestRecord,
    RunSummary,
    final_episodes,
    holm_adjust,
    mcnemar_exact,
    paired_mean_difference,
    paired_outcomes,
    paired_permutation_p,
    summarize_run,
)
from kortyn.report import (
    CATEGORICAL,
    DPI,
    INK_2,
    MUTED,
    REFERENCE_GRAY,
    SEVERITY,
    SURFACE,
    ProfileData,
    RunData,
    annotate_points,
    apply_style,
    bar_with_round_end,
    ci,
    fmt,
    header,
    header_legend,
    load_profile,
    make_figure,
    network_label,
    ordered,
    pct,
    percent_axis,
)

LOG = logging.getLogger("kortyn.report.phase2")

PHASE2_DIR = "phase2"
STAGES = (("1", "Screening", "screening"), ("2", "Bestätigung", "confirmation"))
BASELINE = "baseline"
SYNC_RUN = "referenz_sync"
REPEAT_SUFFIX = "_wiederholung"  # phase 1 measured some conditions twice in one session
TIE_MARGIN = 0.01  # reward difference below which the lower request rate decides the selection
SELECT_COUNT = 2
ALPHA = 0.05
FOCUS = ("schlecht", "sehr_schlecht")  # where phase 2 has to win back success
SCENARIO_ORDER = [
    "gut",
    "buero_wlan",
    "schlecht",
    "sehr_schlecht",
    "schlecht_ohne_stall",
    "sehr_schlecht_ohne_stall",
]
SCENARIO_LABELS = {
    "gut": "gut",
    "buero_wlan": "Büro-WLAN",
    "schlecht": "schlecht",
    "sehr_schlecht": "sehr schlecht",
    "schlecht_ohne_stall": "schlecht o. Stall",
    "sehr_schlecht_ohne_stall": "sehr schl. o. Stall",
}
# Request thresholds are ordinal (higher = more requests): one-hue ramp, light to dark. Adaptive is a
# different kind of policy: the categorical accent. Inpainting is a variant of a policy: marker shape or
# texture, never another hue.
THRESHOLD_COLORS = {6: SEVERITY["gut"], 8: SEVERITY["buero_wlan"], 10: SEVERITY["schlecht"]}
ADAPTIVE_COLOR = CATEGORICAL[1]
INPAINT_HATCH = "////"


# --------------------------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Cell:
    """One (strategy, scenario) run."""

    strategy: str
    scenario: str
    run: RunData
    source: str = ""  # "" = measured in this stage; "Phase 1" = phase-1 run on the stage's seeds


@dataclass
class Stage:
    key: str  # "1" | "2"
    title: str
    profile: ProfileData
    cells: dict[tuple[str, str], Cell]
    strategies: list[str]  # configured order; the baseline only if the stage measured it
    scenarios: list[str]
    settings: dict[str, dict[str, Any]]
    baselines: dict[str, Cell] = field(default_factory=dict)
    repetitions: dict[str, Cell] = field(default_factory=dict)  # phase-1 repetition of the baseline condition
    sync: RunSummary | None = None  # phase-1 synchronous reference on the stage's seeds

    @property
    def seeds(self) -> list[int]:
        return sorted({e.seed for cell in self.cells.values() for e in cell.run.episodes})

    @property
    def candidates(self) -> list[str]:
        return [s for s in self.strategies if s != BASELINE]

    def settings_of(self, strategy: str) -> dict[str, Any]:
        return self.settings.get(strategy, {})


def restrict(run: RunData, seeds: Sequence[int]) -> RunData | None:
    """The same run limited to ``seeds`` with a recomputed summary (None if no seed matches)."""
    wanted = set(seeds)
    episodes = [e for e in run.episodes if e.seed in wanted]
    if not episodes:
        return None
    requests = [r for r in run.requests if r.seed in wanted]
    return RunData(run.label, episodes, requests, summarize_run(episodes, requests))


def load_stage(key: str, title: str, directory: Path, phase1: ProfileData | None) -> Stage | None:
    if not directory.is_dir():
        return None
    profile = load_profile(directory)
    if profile is None:
        return None
    entries: dict[str, dict[str, Any]] = profile.meta.get("runs", {})
    labels = [label for label in entries if label in profile.runs]  # the configured order
    labels += sorted(set(profile.runs) - set(labels))
    cells: dict[tuple[str, str], Cell] = {}
    strategies: list[str] = []
    settings: dict[str, dict[str, Any]] = {}
    for label in labels:
        entry = entries.get(label, {})
        summary = profile.runs[label].summary
        strategy = entry.get("strategy") or summary.strategy
        scenario = entry.get("scenario") or summary.scenario
        if not strategy:
            LOG.warning("%s: run %s has no strategy and is skipped", directory, label)
            continue
        cells[(strategy, scenario)] = Cell(strategy, scenario, profile.runs[label])
        if strategy not in strategies:
            strategies.append(strategy)
            settings[strategy] = dict(entry.get("strategy_settings", {}))
    if not cells:
        return None
    present = {scenario for _, scenario in cells}
    scenarios = [s for s in SCENARIO_ORDER if s in present] + sorted(present - set(SCENARIO_ORDER))
    stage = Stage(key, title, profile, cells, strategies, scenarios, settings)
    seeds = stage.seeds
    for scenario in scenarios:
        own = cells.get((BASELINE, scenario))
        if own is not None:
            stage.baselines[scenario] = own
        elif phase1 is not None and scenario in phase1.runs:
            restricted = restrict(phase1.runs[scenario], seeds)
            if restricted is not None:
                stage.baselines[scenario] = Cell(BASELINE, scenario, restricted, source="Phase 1")
        repeat = f"{scenario}{REPEAT_SUFFIX}"
        if phase1 is not None and repeat in phase1.runs:
            restricted = restrict(phase1.runs[repeat], seeds)
            if restricted is not None:
                stage.repetitions[scenario] = Cell(BASELINE, scenario, restricted, source="Phase 1")
    if phase1 is not None and SYNC_RUN in phase1.runs:
        sync = restrict(phase1.runs[SYNC_RUN], seeds)
        stage.sync = None if sync is None else sync.summary
    return stage


def chosen_requests(run: RunData) -> list[RequestRecord]:
    """Answered real-time requests of the episodes that count (first valid attempt per seed)."""
    chosen, _ = final_episodes(run.episodes)
    keys = {(e.seed, e.attempt) for e in chosen}
    return [
        r for r in run.requests if (r.seed, r.attempt) in keys and r.phase == "realtime" and r.status == "ok"
    ]


# --------------------------------------------------------------------------------------------- naming
@dataclass(frozen=True)
class Style:
    family: str
    color: str
    inpaint: bool


def style_of(strategy: str, settings: dict[str, Any]) -> Style:
    if strategy == BASELINE:
        return Style("Baseline", REFERENCE_GRAY, inpaint=False)
    request = settings.get("request", {})
    inpaint = settings.get("smoothing", {}).get("mode") == "inpaint"
    if request.get("mode") == "adaptive":
        return Style(f"Adaptiv {request.get('target_age_s', 0.7):g} s", ADAPTIVE_COLOR, inpaint)
    threshold = request.get("refill_threshold")
    if isinstance(threshold, int):
        color = THRESHOLD_COLORS.get(threshold, SEVERITY["sehr_schlecht"])
        return Style(f"Schwelle {threshold}", color, inpaint)
    return Style("Schwelle (Agent)", SEVERITY["sehr_schlecht"], inpaint)


def display_name(strategy: str, settings: dict[str, Any]) -> str:
    style = style_of(strategy, settings)
    return style.family + (" · Inpainting" if style.inpaint else "")


def scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario)


# --------------------------------------------------------------------------------------------- statistics
@dataclass(frozen=True)
class Comparison:
    """A strategy against the baseline of its scenario on the common seeds."""

    pairs: int
    base_rate: float
    rate: float
    only_base: int
    only_strategy: int
    mcnemar_p: float
    reward_delta: float | None
    reward_ci: tuple[float, float] | None
    reward_p: float | None
    mcnemar_holm: float | None = None
    reward_holm: float | None = None

    @property
    def success_delta(self) -> float:
        return self.rate - self.base_rate


def compare(base: RunSummary, other: RunSummary) -> Comparison | None:
    common = sorted(set(base.success_by_seed) & set(other.success_by_seed))
    if not common:
        return None
    _, only_base, only_other = paired_outcomes(base.success_by_seed, other.success_by_seed)
    reward_seeds = sorted(set(base.max_reward_by_seed) & set(other.max_reward_by_seed))
    diffs = [other.max_reward_by_seed[s] - base.max_reward_by_seed[s] for s in reward_seeds]
    reward = paired_mean_difference(base.max_reward_by_seed, other.max_reward_by_seed)
    return Comparison(
        pairs=len(common),
        base_rate=sum(base.success_by_seed[s] for s in common) / len(common),
        rate=sum(other.success_by_seed[s] for s in common) / len(common),
        only_base=only_base,
        only_strategy=only_other,
        mcnemar_p=mcnemar_exact(only_base, only_other),
        reward_delta=None if reward is None else reward[0],
        reward_ci=None if reward is None else reward[1],
        reward_p=paired_permutation_p(diffs) if diffs else None,
    )


def compare_stage(stage: Stage) -> dict[tuple[str, str], Comparison]:
    """Every strategy against the baseline of its scenario, Holm-adjusted over all comparisons of the stage."""
    raw: dict[tuple[str, str], Comparison] = {}
    for strategy in stage.candidates:
        for scenario in stage.scenarios:
            cell = stage.cells.get((strategy, scenario))
            base = stage.baselines.get(scenario)
            if cell is None or base is None:
                continue
            comparison = compare(base.run.summary, cell.run.summary)
            if comparison is not None:
                raw[(strategy, scenario)] = comparison
    keys = list(raw)
    success_holm = dict(zip(keys, holm_adjust([raw[k].mcnemar_p for k in keys]), strict=True))
    known = [(k, p) for k in keys if (p := raw[k].reward_p) is not None]
    reward_holm = dict(zip([k for k, _ in known], holm_adjust([p for _, p in known]), strict=True))
    return {k: replace(raw[k], mcnemar_holm=success_holm[k], reward_holm=reward_holm.get(k)) for k in keys}


@dataclass(frozen=True)
class Rank:
    strategy: str
    reward_deltas: dict[str, float]
    score: float  # mean paired max-reward difference over the stage's scenarios
    requests_per_s: float  # mean over the stage's scenarios


def rank_strategies(stage: Stage, comparisons: dict[tuple[str, str], Comparison]) -> list[Rank]:
    """Strategies with a comparison in every scenario of the stage, best score first."""
    ranks: list[Rank] = []
    for strategy in stage.candidates:
        deltas: dict[str, float] = {}
        rates: list[float] = []
        for scenario in stage.scenarios:
            comparison = comparisons.get((strategy, scenario))
            cell = stage.cells.get((strategy, scenario))
            if comparison is None or comparison.reward_delta is None or cell is None:
                break
            deltas[scenario] = comparison.reward_delta
            rates.append(cell.run.summary.requests_per_s or 0.0)
        else:
            if deltas:
                score = float(np.mean(list(deltas.values())))
                ranks.append(Rank(strategy, deltas, score, float(np.mean(rates))))
    return sorted(ranks, key=lambda r: (-r.score, r.requests_per_s))


def select_strategies(
    ranks: Sequence[Rank], count: int = SELECT_COUNT, margin: float = TIE_MARGIN
) -> list[str]:
    """The pre-registered rule: the best score; within ``margin`` of it the lowest request rate wins."""
    remaining = list(ranks)
    chosen: list[str] = []
    while remaining and len(chosen) < count:
        best = max(r.score for r in remaining)
        tied = [r for r in remaining if r.score >= best - margin]
        pick = min(tied, key=lambda r: (r.requests_per_s, -r.score))
        chosen.append(pick.strategy)
        remaining.remove(pick)
    return chosen


def gap_recovered(base_rate: float, rate: float, sync_rate: float | None) -> float | None:
    """Share of the gap between baseline and synchronous reference that the strategy closes."""
    if sync_rate is None or sync_rate - base_rate <= 1e-9:
        return None
    return (rate - base_rate) / (sync_rate - base_rate)


def mean_load(stage: Stage, strategy: str) -> tuple[float, float] | None:
    """(requests/s, inference utilization) averaged over the scenarios of the stage."""
    if strategy == BASELINE:
        cells = [stage.baselines.get(s) for s in stage.scenarios]
    else:
        cells = [stage.cells.get((strategy, s)) for s in stage.scenarios]
    rates = [c.run.summary.requests_per_s for c in cells if c is not None]
    loads = [c.run.summary.inference_utilization for c in cells if c is not None]
    rate_values = [r for r in rates if r is not None]
    load_values = [u for u in loads if u is not None]
    if not rate_values or not load_values:
        return None
    return float(np.mean(rate_values)), float(np.mean(load_values))


# --------------------------------------------------------------------------------------------- formatting
def p_text(p: float | None) -> str:
    if p is None or math.isnan(p):
        return "–"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def reward_text(comparison: Comparison) -> str:
    if comparison.reward_delta is None:
        return "–"
    bounds = comparison.reward_ci
    interval = "" if bounds is None else f" [{bounds[0]:+.3f}; {bounds[1]:+.3f}]"
    return f"{comparison.reward_delta:+.3f}{interval}"


def significant(p: float | None) -> bool:
    return p is not None and p < ALPHA


def bold_if(text: str, condition: bool) -> str:
    return f"**{text}**" if condition and text != "–" else text


# --------------------------------------------------------------------------------------------- tables
def strategy_table(stage: Stage) -> list[str]:
    rows = [
        "| Strategie | Schlüssel | Anfrage-Zeitpunkt | Ausrichtung | Übergang | ohne nutzbare Aktion |",
        "|---|---|---|---|---|---|",
    ]
    for strategy in stage.strategies:
        s = stage.settings_of(strategy)
        request = s.get("request", {})
        if request.get("mode") == "adaptive":
            when = (
                f"adaptiv: Ziel-Planalter {request.get('target_age_s'):g} s; Deadline mit Latenz-Median + "
                f"{request.get('latency_quantile_z'):g} · σ (robust) + {request.get('safety_margin_ticks')} Takt"
            )
        else:
            threshold = request.get("refill_threshold")
            when = f"≤ {threshold if threshold is not None else 6} Aktionen in der Queue"
        smoothing = s.get("smoothing", {})
        if smoothing.get("mode") == "inpaint":
            transition = f"Inpainting ({smoothing.get('schedule')}, ≤ {smoothing.get('max_overlap_steps')} Takte Überlappung)"
        else:
            transition = "harter Wechsel"
        guard = s.get("guard", {})
        if guard.get("mode") == "hold":
            limit = guard.get("max_plan_age_s")
            fallback = "anhalten + Notfall" + (f" (auch bei Planalter > {limit:g} s)" if limit else "")
        else:
            fallback = "letzte Aktion wiederholen"
        rows.append(
            f"| {display_name(strategy, s)} | `{strategy}` | {when} | {s.get('alignment', 'timestep')} | "
            f"{transition} | {fallback} |"
        )
    return rows


RESULT_HEADER = [
    "| Strategie | n | Erfolg [95 %-KI] | Δ Erfolg | McNemar p (Holm) | Ø Max-Reward | Δ Reward [95 %-KI] | "
    "Permutation p (Holm) | Planalter Ø / p95 (s) | Zeit verhungert | Notfälle/min | Anfragen/s | "
    "Auslastung Inferenz | Wartezeit Brain (ms) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]


def result_row(name: str, s: RunSummary, c: Comparison | None) -> str:
    if c is None:
        delta = test = reward = reward_test = "–"
    else:
        delta = bold_if(f"{100 * c.success_delta:+.0f} pp", significant(c.mcnemar_holm))
        test = f"{p_text(c.mcnemar_p)} ({p_text(c.mcnemar_holm)})"
        reward = bold_if(reward_text(c), significant(c.reward_holm))
        reward_test = f"{p_text(c.reward_p)} ({p_text(c.reward_holm)})"
    cells = [
        name,
        str(s.episodes),
        f"{pct(s.success_rate)} {ci(s.success_ci)}",
        delta,
        test,
        fmt(s.max_reward_mean, 3),
        reward,
        reward_test,
        f"{fmt(s.plan_age_mean_s, 2)} / {fmt(s.plan_age_p95_s, 2)}",
        pct(s.starved_time_fraction, 1),
        f"{s.emergency_events_per_min:.1f}",
        fmt(s.requests_per_s, 2),
        pct(s.inference_utilization),
        fmt(s.queue_wait_mean_ms, 1),
    ]
    return "| " + " | ".join(cells) + " |"


def stage_tables(
    stage: Stage, comparisons: dict[tuple[str, str], Comparison], highlight: Sequence[str] = ()
) -> list[str]:
    lines: list[str] = []
    for scenario in stage.scenarios:
        label = next(cell.run.label for key, cell in stage.cells.items() if key[1] == scenario)
        network = network_label(stage.profile.meta, label)
        lines += [f"#### {scenario_label(scenario)} ({network})", "", *RESULT_HEADER]
        base = stage.baselines.get(scenario)
        if base is not None:
            name = "Baseline (Phase-1-Lauf)" if base.source else "Baseline"
            lines.append(result_row(name, base.run.summary, None))
            repeat = stage.repetitions.get(scenario)
            if base.source and repeat is not None:  # the same condition again: run-to-run noise, not a test
                noise = compare(base.run.summary, repeat.run.summary)
                lines.append(
                    result_row("_Baseline-Wiederholung (Phase 1, identisch)_", repeat.run.summary, noise)
                )
        for strategy in stage.candidates:
            cell = stage.cells.get((strategy, scenario))
            if cell is None:
                continue
            name = display_name(strategy, stage.settings_of(strategy))
            name = f"**{name}** ⭐" if strategy in highlight else name
            lines.append(result_row(name, cell.run.summary, comparisons.get((strategy, scenario))))
        lines.append("")
    return lines


def ranking_table(stage: Stage, ranks: Sequence[Rank], chosen: Sequence[str]) -> list[str]:
    columns = " | ".join(f"Δ Reward {scenario_label(s)}" for s in stage.scenarios)
    rows = [
        f"| Rang | Strategie | {columns} | Score (Ø Δ Reward) | Anfragen/s (Ø) | Stufe 2 |",
        "|---:|---|" + "---:|" * len(stage.scenarios) + "---:|---:|:---:|",
    ]
    for index, rank in enumerate(ranks, start=1):
        deltas = " | ".join(f"{rank.reward_deltas[s]:+.3f}" for s in stage.scenarios)
        rows.append(
            f"| {index} | {display_name(rank.strategy, stage.settings_of(rank.strategy))} | {deltas} | "
            f"{rank.score:+.3f} | {rank.requests_per_s:.2f} | {'✅' if rank.strategy in chosen else ''} |"
        )
    return rows


@dataclass(frozen=True)
class Reproduction:
    rows: list[str]
    reproduced: bool | None  # no Holm-significant difference (success or reward) in any scenario
    nominal: bool | None  # not even a nominally significant one (p >= 0.05 without correction)
    same_timing: bool  # plan age within 0.05 s and request rate within 5 % everywhere
    strongest: tuple[str, float, float] | None  # (what, nominal p, Holm p) of the smallest p-value
    tests: int


def reproduction_row(
    name: str, c: Comparison, old: RunSummary, new: RunSummary, holm: tuple[str, str]
) -> str:
    return (
        f"| {name} | {c.pairs} | {pct(c.base_rate)} → {pct(c.rate)} | {p_text(c.mcnemar_p)}{holm[0]} | "
        f"{fmt(old.max_reward_mean, 3)} → {fmt(new.max_reward_mean, 3)} | {reward_text(c)} | "
        f"{p_text(c.reward_p)}{holm[1]} | {fmt(old.plan_age_mean_s, 2)} → {fmt(new.plan_age_mean_s, 2)} | "
        f"{pct(old.starved_time_fraction, 1)} → {pct(new.starved_time_fraction, 1)} | "
        f"{fmt(old.requests_per_s, 2)} → {fmt(new.requests_per_s, 2)} | "
        f"{fmt(old.e2e_p50_ms)} → {fmt(new.e2e_p50_ms)} |"
    )


def reproduction(stage: Stage, phase1: ProfileData) -> Reproduction:
    """Stage-1 baseline against the phase-1 runs on the same seeds.

    Every scenario contributes two tests (success: McNemar, max reward: permutation); "reproduced" means no
    difference after Holm adjustment over all of them. Where phase 1 measured the condition twice in one
    session (``<scenario>_wiederholung``), both comparisons with the repetition are shown as the reference
    for run-to-run noise of a real-time measurement; they do not enter the verdict.
    """
    rows = [
        "| Vergleich (alt → neu) | Seeds | Erfolg | McNemar p (Holm) | Ø Max-Reward | Δ Reward [95 %-KI] | "
        "Permutation p (Holm) | Planalter Ø (s) | verhungert | Anfragen/s | E2E p50 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    primary: list[tuple[str, Comparison, RunSummary, RunSummary]] = []
    references: list[tuple[str, Comparison, RunSummary, RunSummary]] = []
    timing: list[bool] = []
    for scenario in stage.scenarios:
        own = stage.cells.get((BASELINE, scenario))
        old_run = restrict(phase1.runs[scenario], stage.seeds) if scenario in phase1.runs else None
        if own is None or old_run is None:
            continue
        old, new = old_run.summary, own.run.summary
        c = compare(old, new)
        if c is None:
            continue
        primary.append((scenario, c, old, new))
        ages = (old.plan_age_mean_s, new.plan_age_mean_s)
        rates = (old.requests_per_s, new.requests_per_s)
        timing.append(
            ages[0] is not None
            and ages[1] is not None
            and abs(ages[1] - ages[0]) <= 0.05
            and rates[0] is not None
            and rates[1] is not None
            and abs(rates[1] - rates[0]) <= 0.05 * rates[0]
        )
        repeat = stage.repetitions.get(scenario)
        if repeat is not None:
            rep = repeat.run.summary
            for label, a, b in (
                (f"{scenario_label(scenario)}: Phase-1-Wiederholung → Stufe 1", rep, new),
                (
                    f"{scenario_label(scenario)}: Phase 1 → Phase-1-Wiederholung (gleiche Sitzung, identisch)",
                    old,
                    rep,
                ),
            ):
                reference = compare(a, b)
                if reference is not None:
                    references.append((label, reference, a, b))
    pvalues: list[tuple[str, float]] = []
    for scenario, c, _, _ in primary:
        pvalues.append((f"Erfolg {scenario_label(scenario)}", c.mcnemar_p))
        if c.reward_p is not None:
            pvalues.append((f"Max-Reward {scenario_label(scenario)}", c.reward_p))
    adjusted = holm_adjust([p for _, p in pvalues])
    holm_of = {name: value for (name, _), value in zip(pvalues, adjusted, strict=True)}
    for scenario, c, old, new in primary:
        label = scenario_label(scenario)
        reward_holm = holm_of.get(f"Max-Reward {label}")
        holm = (
            f" ({p_text(holm_of[f'Erfolg {label}'])})",
            "" if reward_holm is None else f" ({p_text(reward_holm)})",
        )
        rows.append(reproduction_row(f"{label}: Phase 1 → Stufe 1", c, old, new, holm))
    for label, c, old, new in references:
        rows.append(reproduction_row(f"_{label}_", c, old, new, ("", "")))
    strongest = None
    if pvalues:
        index = min(range(len(pvalues)), key=lambda i: pvalues[i][1])
        strongest = (pvalues[index][0], pvalues[index][1], adjusted[index])
    return Reproduction(
        rows=rows,
        reproduced=all(a >= ALPHA for a in adjusted) if adjusted else None,
        nominal=all(p >= ALPHA for _, p in pvalues) if pvalues else None,
        same_timing=bool(timing) and all(timing),
        strongest=strongest,
        tests=len(pvalues),
    )


def reproduction_text(result: Reproduction, noise: Comparison | None) -> list[str]:
    timing = " Planalter (± 0,05 s) und Anfragerate (± 5 %) stimmen überein." if result.same_timing else ""
    reference = ""
    if noise is not None and noise.reward_delta is not None:
        reference = (
            f" Zum Vergleich: Phase 1 gegen ihre eigene, identische Wiederholung in derselben Sitzung ergibt "
            f"Δ Reward {noise.reward_delta:+.3f} (n = {noise.pairs}, Permutation p = {p_text(noise.reward_p)}) – "
            "Unterschiede dieser Größe sind Lauf-zu-Lauf-Rauschen der Echtzeitmessung."
        )
    if result.reproduced is None:
        return []
    if result.reproduced and result.nominal:
        return [
            f"✅ **Reproduziert:** kein Unterschied zu Phase 1, auch ohne Korrektur (alle p ≥ 0,05).{timing}",
            "",
        ]
    if result.reproduced and result.strongest is not None:
        what, nominal, holm = result.strongest
        return [
            f"✅ **Reproduziert innerhalb der Unsicherheit:** {what} nominal p = {p_text(nominal)}, nach "
            f"Holm-Korrektur über alle {result.tests} Reproduktionstests p = {p_text(holm)}.{timing}{reference}",
            "",
        ]
    return [
        "⚠️ **Abweichung:** mindestens ein Szenario unterscheidet sich auch nach Holm-Korrektur von Phase 1 – "
        f"Vergleiche gegen Phase-1-Läufe sind mit Vorsicht zu lesen.{timing}{reference}",
        "",
    ]


def run_noise(phase1: ProfileData | None) -> tuple[str, Comparison] | None:
    """Phase 1 against its own repetition (same condition, same session, all seeds): run-to-run noise."""
    if phase1 is None:
        return None
    for label in ordered(phase1.runs):
        if label.endswith(REPEAT_SUFFIX) and label[: -len(REPEAT_SUFFIX)] in phase1.runs:
            c = compare(phase1.runs[label[: -len(REPEAT_SUFFIX)]].summary, phase1.runs[label].summary)
            if c is not None:
                return label[: -len(REPEAT_SUFFIX)], c
    return None


def overview_table(
    stage: Stage, comparisons: dict[tuple[str, str], Comparison], strategies: Sequence[str]
) -> list[str]:
    names = [display_name(s, stage.settings_of(s)) for s in strategies]
    rows = [
        "| Szenario | Baseline: Erfolg · Planalter · Anfragen/s | "
        + " | ".join(f"{name}: Erfolg (Δ, p Holm) · Planalter · Anfragen/s" for name in names)
        + " |",
        "|---|---:|" + "---:|" * len(strategies),
    ]
    for scenario in stage.scenarios:
        base = stage.baselines.get(scenario)
        if base is None:
            continue
        b = base.run.summary
        cells = [f"{pct(b.success_rate)} · {fmt(b.plan_age_mean_s, 2)} s · {fmt(b.requests_per_s, 2)}"]
        for strategy in strategies:
            cell = stage.cells.get((strategy, scenario))
            c = comparisons.get((strategy, scenario))
            if cell is None or c is None:
                cells.append("–")
                continue
            s = cell.run.summary
            delta = bold_if(f"{100 * c.success_delta:+.0f} pp", significant(c.mcnemar_holm))
            cells.append(
                f"{pct(s.success_rate)} ({delta}, {p_text(c.mcnemar_holm)}) · {fmt(s.plan_age_mean_s, 2)} s · "
                f"{fmt(s.requests_per_s, 2)}"
            )
        rows.append(f"| {scenario_label(scenario)} | " + " | ".join(cells) + " |")
    return rows


def trigger_text(requests: Sequence[RequestRecord]) -> str:
    names = {"threshold": "Schwelle", "deadline": "Deadline", "freshness": "Frische"}
    counts = Counter(r.trigger or "?" for r in requests)
    total = sum(counts.values())
    if not total:
        return "–"
    return " · ".join(f"{names.get(k, k)} {100 * v / total:.0f} %" for k, v in counts.most_common())


def estimator_quality(requests: Sequence[RequestRecord], z: float) -> tuple[float | None, float | None]:
    """(median |E2E - estimated median| in ms, share of requests within the planned latency quantile)."""
    triples = [
        (e2e, r.latency_median_ms, r.latency_spread_ms)
        for r in requests
        if (e2e := r.e2e_ms) is not None
        and r.latency_median_ms is not None
        and r.latency_spread_ms is not None
    ]
    if not triples:
        return None, None
    errors = [abs(e2e - median) for e2e, median, _ in triples]
    covered = sum(1 for e2e, median, spread in triples if e2e <= median + z * spread)
    return float(np.median(errors)), covered / len(triples)


def median_success_seconds(chosen: Sequence[EpisodeRecord]) -> float | None:
    durations = [e.duration_s for e in chosen if e.success]
    return float(np.median(durations)) if durations else None


def diagnostics_table(stage: Stage) -> list[str]:
    rows = [
        "| Strategie | Szenario | ungültige Versuche | Notfall-Takte leer / zu alt | Reaktion p50 (ms) | "
        "Dauer bis Erfolg (Median, s) | Auslastung Records / Zähler | Latenzschätzer: Median-Fehler (ms) | "
        "im Planungs-Quantil | Anfrage-Auslöser | Guide fest / weich (Ø) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for strategy in stage.strategies:
        settings = stage.settings_of(strategy)
        z = float(settings.get("request", {}).get("latency_quantile_z", 1.645))
        for scenario in stage.scenarios:
            cell = stage.cells.get((strategy, scenario))
            if cell is None:
                continue
            s = cell.run.summary
            chosen, _ = final_episodes(cell.run.episodes)
            requests = chosen_requests(cell.run)
            error, coverage = estimator_quality(requests, z)
            age_ticks = sum(e.emergency_ticks_age for e in chosen)
            empty_ticks = sum(e.emergency_ticks for e in chosen) - age_ticks
            attempts = s.episodes + s.invalid_attempts
            guided = [r for r in requests if r.guide_frozen or r.guide_soft]
            guide = (
                f"{np.mean([r.guide_frozen for r in guided]):.1f} / {np.mean([r.guide_soft for r in guided]):.1f}"
                if guided
                else "–"
            )
            invalid = f"{s.invalid_attempts} ({pct(s.invalid_attempts / attempts if attempts else None)})"
            utilization = f"{pct(s.inference_utilization)} / {pct(s.brain_counter_utilization)}"
            rows.append(
                f"| {display_name(strategy, settings)} | {scenario_label(scenario)} | {invalid} | "
                f"{empty_ticks} / {age_ticks} | {fmt(s.reaction_p50_ms)} | {fmt(median_success_seconds(chosen), 1)} | "
                f"{utilization} | {fmt(error)} | {pct(coverage)} (z = {z:g}) | {trigger_text(requests)} | {guide} |"
            )
    return rows


def estimator_coverage(stages: Sequence[Stage]) -> tuple[float, float] | None:
    """Lowest and highest share of requests within the planned latency quantile over all runs."""
    shares: list[float] = []
    for stage in stages:
        for (strategy, _), cell in stage.cells.items():
            z = float(stage.settings_of(strategy).get("request", {}).get("latency_quantile_z", 1.645))
            _, coverage = estimator_quality(chosen_requests(cell.run), z)
            if coverage is not None:
                shares.append(coverage)
    return (min(shares), max(shares)) if shares else None


def invalid_attempts_bias(stages: Sequence[Stage]) -> tuple[float, float, int, int] | None:
    """(failure share of invalid attempts, failure share of valid ones, #invalid, #attempts) over all runs."""
    invalid = [
        e for stage in stages for cell in stage.cells.values() for e in cell.run.episodes if not e.valid
    ]
    valid = [e for stage in stages for cell in stage.cells.values() for e in cell.run.episodes if e.valid]
    if not invalid or not valid:
        return None
    fail_invalid = sum(1 for e in invalid if not e.success) / len(invalid)
    fail_valid = sum(1 for e in valid if not e.success) / len(valid)
    return fail_invalid, fail_valid, len(invalid), len(invalid) + len(valid)


def diagnostic_findings(stages: Sequence[Stage]) -> list[str]:
    lines: list[str] = []
    coverage = estimator_coverage(stages)
    if coverage is not None:
        lines.append(
            f"- **Latenzschätzer:** Das Planungs-Quantil (Median + 1,645 · σ aus der MAD) deckt nur "
            f"{pct(coverage[0])}–{pct(coverage[1])} der Anfragen ab statt der 95 %, die eine Normalverteilung "
            "erwarten ließe – auch ohne Netz: Die E2E-Latenz hat einen schwereren rechten Rand (Scheduling, "
            "Inferenz-Ausreißer), Stalls verstärken das. Für Deadlines ist ein empirisches Quantil des Fensters "
            "(z. B. p95) die bessere Wahl als Median + z · MAD."
        )
    bias = invalid_attempts_bias(stages)
    if bias is not None:
        fail_invalid, fail_valid, n_invalid, n_attempts = bias
        lines.append(
            f"- **Validitäts-Gate:** {n_invalid} von {n_attempts} Versuchen ({pct(n_invalid / n_attempts)}) wurden "
            f"wiederholt. Ungültige Versuche sind häufiger Misserfolge ({pct(fail_invalid)} gegenüber "
            f"{pct(fail_valid)} bei gültigen), weil lange Episoden mehr Takte für eine Verspätung bieten; die "
            "Wiederholung hebt die Erfolgsquote dadurch leicht an (≈ Rate ungültiger Versuche × Differenz, "
            "wenige Prozentpunkte) – für alle Strategien ähnlich, in Vergleichen weitgehend neutral."
        )
    return [*lines, ""] if lines else []


# --------------------------------------------------------------------------------------------- hypothesis
@dataclass(frozen=True)
class Slope:
    """Change of the mean max reward per second of mean plan age, with a seed-bootstrap distribution."""

    value: float
    ci: tuple[float, float]
    samples: npt.NDArray[np.float64]


def pooled_slope(groups: Sequence[Sequence[RunSummary]], n_boot: int = 2000, seed: int = 0) -> Slope | None:
    """Common least-squares slope of max reward over mean plan age within groups (fixed effects: every group
    is centred, e.g. one group per network scenario), with a bootstrap that resamples seeds (the same seeds
    for every run, so the pairing is kept). Plan ages are run means and nearly deterministic; they are held.
    """
    usable = [[s for s in runs if s.plan_age_mean_s is not None and s.max_reward_by_seed] for runs in groups]
    usable = [runs for runs in usable if len(runs) >= 2]
    if not usable:
        return None
    seeds = sorted(set.intersection(*[set(s.max_reward_by_seed) for runs in usable for s in runs]))
    if len(seeds) < 2:
        return None
    ages = [np.array([float(s.plan_age_mean_s or 0.0) for s in runs]) for runs in usable]
    rewards = [np.array([[s.max_reward_by_seed[k] for k in seeds] for s in runs]) for runs in usable]

    def fit(columns: npt.NDArray[np.int64]) -> float | None:
        numerator = denominator = 0.0
        for x, y in zip(ages, rewards, strict=True):
            means = y[:, columns].mean(axis=1)
            centred = x - x.mean()
            numerator += float(np.sum(centred * (means - means.mean())))
            denominator += float(np.sum(centred * centred))
        return numerator / denominator if denominator > 1e-12 else None

    estimate = fit(np.arange(len(seeds), dtype=np.int64))
    if estimate is None:
        return None
    rng = np.random.default_rng(seed)
    draws = [fit(rng.integers(0, len(seeds), size=len(seeds), dtype=np.int64)) for _ in range(n_boot)]
    samples = np.array([d for d in draws if d is not None], dtype=np.float64)
    low, high = np.percentile(samples, [2.5, 97.5])
    return Slope(estimate, (float(low), float(high)), samples)


@dataclass(frozen=True)
class AgeEvidence:
    between: Slope | None  # phase 1: same request timing, plan age varied by network and compute
    within: Slope | None  # stage 1: same network, plan age varied by the request timing (pooled scenarios)
    within_by_scenario: dict[str, float] = field(default_factory=dict)
    actions_per_plan: dict[str, float] = field(default_factory=dict)  # executed actions per requested plan

    @property
    def difference_ci(self) -> tuple[float, float] | None:
        """95 % interval of within - between (independent bootstrap distributions)."""
        if self.between is None or self.within is None:
            return None
        n = min(len(self.between.samples), len(self.within.samples))
        if n < 2:
            return None
        low, high = np.percentile(self.within.samples[:n] - self.between.samples[:n], [2.5, 97.5])
        return float(low), float(high)

    @property
    def verdict(self) -> str:
        """One of "confirmed", "weak", "refuted", "inconclusive", or "" without a phase-1 effect."""
        difference = self.difference_ci
        if self.between is None or self.within is None or difference is None or self.between.ci[1] >= 0:
            return ""
        if difference[0] > 0:  # significantly flatter than the phase-1 relation
            return "refuted" if self.within.ci[1] >= 0 else "weak"
        if self.within.ci[1] < 0:
            return "confirmed"
        return "inconclusive"


def age_evidence(stage: Stage, phase1: ProfileData | None) -> AgeEvidence:
    between = None
    if phase1 is not None:
        between = pooled_slope(
            [[run.summary for run in phase1.runs.values() if run.summary.mode == "realtime"]]
        )
    hard_switch = [s for s in [BASELINE, *stage.candidates] if not style_of(s, stage.settings_of(s)).inpaint]
    groups: list[list[RunSummary]] = []
    by_scenario: dict[str, float] = {}
    per_plan: dict[str, list[float]] = {}
    for scenario in stage.scenarios:
        runs: list[RunSummary] = []
        for strategy in hard_switch:  # only the request timing varies
            cell = (
                stage.baselines.get(scenario)
                if strategy == BASELINE
                else stage.cells.get((strategy, scenario))
            )
            if cell is None:
                continue
            runs.append(cell.run.summary)
            if cell.run.summary.requests_per_s:
                per_plan.setdefault(strategy, []).append(
                    cell.run.episodes[0].fps / cell.run.summary.requests_per_s
                )
        groups.append(runs)
        single = pooled_slope([runs], n_boot=1)
        if single is not None:
            by_scenario[scenario] = single.value
    return AgeEvidence(
        between, pooled_slope(groups), by_scenario, {k: float(np.mean(v)) for k, v in per_plan.items()}
    )


def slope_text(slope: Slope) -> str:
    return f"{0.1 * slope.value:+.3f} [{0.1 * slope.ci[0]:+.3f}; {0.1 * slope.ci[1]:+.3f}]"


def hypothesis_section(evidence: AgeEvidence) -> list[str]:
    verdict = evidence.verdict
    difference = evidence.difference_ci
    if not verdict or evidence.between is None or evidence.within is None or difference is None:
        return []
    scenarios = ", ".join(
        f"{0.1 * value:+.3f} {scenario_label(scenario)}"
        for scenario, value in evidence.within_by_scenario.items()
    )
    lines = [
        "## Ist das Planalter der Hebel?",
        "",
        "Änderung des Ø Max-Reward je +0,1 s Ø-Planalter (Steigung, 95 %-KI per Bootstrap über Seeds):",
        "",
        f"- **Phase 1, zwischen Szenarien** (gleiches Anfrage-Timing; das Planalter variiert durch Netz, Stalls "
        f"und Rechenzeit): {slope_text(evidence.between)}.",
        f"- **Stufe 1, innerhalb eines Szenarios** (gleiches Netz; das Planalter variiert nur durch die "
        f"Anfrage-Häufigkeit, harte Wechsel): {slope_text(evidence.within)}"
        + (f" (einzeln: {scenarios})" if scenarios else "")
        + ".",
        f"- **Differenz** Stufe 1 − Phase 1: {0.1 * (evidence.within.value - evidence.between.value):+.3f} "
        f"[{0.1 * difference[0]:+.3f}; {0.1 * difference[1]:+.3f}].",
        "",
    ]
    if verdict == "confirmed":
        lines.append(
            "Die Steigungen sind vereinbar, und innerhalb eines Szenarios ist der Zusammenhang signifikant: Wer das "
            "Planalter über die Anfrage-Häufigkeit senkt, gewinnt Qualität in der Größenordnung zurück, die Phase 1 "
            "erwarten ließ."
        )
    elif verdict == "weak":
        lines.append(
            "Innerhalb eines Szenarios ist der Zusammenhang signifikant schwächer als zwischen den Szenarien: Nur ein "
            "Teil des Phase-1-Effekts geht auf das mittlere Planalter zurück, der Rest auf die Verzögerung bis zur "
            "ersten Aktion eines Plans (Latenz) und die Stalls."
        )
    elif verdict == "inconclusive":
        lines.append(
            "Mit dieser Seed-Zahl nicht entscheidbar: Der Zusammenhang innerhalb eines Szenarios ist weder "
            "signifikant negativ noch signifikant flacher als in Phase 1."
        )
    else:
        base_actions = evidence.actions_per_plan.get(BASELINE)
        fresher = [
            v
            for k, v in evidence.actions_per_plan.items()
            if k != BASELINE and base_actions and v < base_actions - 0.5
        ]
        per_plan = ""
        if fresher and base_actions:
            low, high = round(min(fresher)), round(max(fresher))
            span = f"{low}" if low == high else f"{low}–{high}"
            per_plan = f" Je Plan werden nur noch {span} statt {base_actions:.0f} Aktionen ausgeführt (LeRobot-Eval: 8)."
        lines.append(
            "**Die Kernhypothese hält in dieser Form nicht:** Zwischen den Szenarien geht ein höheres Planalter mit "
            "schlechterer Qualität einher, aber ein über die Anfrage-Häufigkeit gesenktes Planalter bringt keine "
            "Qualität zurück – der Zusammenhang ist signifikant flacher und nicht negativ. Häufigere Anfragen senken "
            "das *mittlere* Alter, nicht die Verzögerung bis zur ersten Aktion eines Plans (Reaktionszeit in der "
            "Diagnose), und sie erhöhen die Zahl der Planwechsel, bei denen eine multimodale Diffusion-Policy den "
            f"Lösungsweg wechseln kann.{per_plan} Das Planalter ist in Phase 1 ein Symptom von Latenz und Stalls, "
            "kein eigenständiger Hebel."
        )
    return [*lines, ""]


# --------------------------------------------------------------------------------------------- findings
def key_findings(
    final: Stage,
    comparisons: dict[tuple[str, str], Comparison],
    chosen: Sequence[str],
    best: str | None,
    noise: tuple[str, Comparison] | None = None,
) -> list[str]:
    if best is None:
        return ["- Noch keine Strategie mit vollständigen Daten in allen Szenarien der Stufe."]
    name = display_name(best, final.settings_of(best))
    others = [display_name(s, final.settings_of(s)) for s in chosen if s != best]
    if final.key == "1":
        headline = f"- **Beste Strategie nach der vorab festgelegten Regel (Stufe 1):** {name} (`{best}`)"
        second = "zweite Wahl für Stufe 2"
    else:
        headline = (
            f"- **Beste Strategie in Stufe 2** (dieselbe Regel über alle {len(final.scenarios)} Szenarien: "
            f"größte Ø gepaarte Reward-Differenz, bei < {TIE_MARGIN:g} weniger Anfragen/s): {name} (`{best}`)"
        )
        second = "ebenfalls in Stufe 2 gemessen"
    lines = [headline + (f"; {second}: {', '.join(others)}." if others else ".")]
    sync_rate = final.sync.success_rate if final.sync is not None else None
    for scenario in FOCUS:
        c = comparisons.get((best, scenario))
        cell = final.cells.get((best, scenario))
        base = final.baselines.get(scenario)
        if c is None or cell is None or base is None:
            continue
        recovered = gap_recovered(c.base_rate, c.rate, sync_rate)
        verdict = "signifikant" if significant(c.mcnemar_holm) else "nicht signifikant"
        reward_verdict = "signifikant" if significant(c.reward_holm) else "nicht signifikant"
        gap = (
            f", rechnerisch {100 * recovered:.0f} % der Lücke zur Sync-Referenz"
            if recovered is not None and recovered > 0
            else ""
        )
        lines.append(
            f"- **{scenario_label(scenario)}** (n = {c.pairs}): Erfolg {pct(c.base_rate)} → {pct(c.rate)} "
            f"(**{100 * c.success_delta:+.0f} pp**, McNemar p = {p_text(c.mcnemar_p)}, Holm "
            f"{p_text(c.mcnemar_holm)}: {verdict}{gap}); Ø Max-Reward {reward_text(c)} (Permutation p = "
            f"{p_text(c.reward_p)}, Holm {p_text(c.reward_holm)}: {reward_verdict}); Planalter Ø "
            f"{fmt(base.run.summary.plan_age_mean_s, 2)} → {fmt(cell.run.summary.plan_age_mean_s, 2)} s."
        )
    base_load = mean_load(final, BASELINE)
    best_load = mean_load(final, best)
    if (
        base_load is not None
        and best_load is not None
        and base_load[0] > 0
        and best_load[1] > 0 < base_load[1]
    ):
        robots, robots_base = 1 / best_load[1], 1 / base_load[1]
        capacity = (
            f"≈ {robots:.1f} Roboter wie mit der Baseline"
            if abs(robots - robots_base) < 0.05
            else f"≈ {robots:.1f} statt {robots_base:.1f} Roboter"
        )
        lines.append(
            f"- **Brain-Mehrlast** (Ø über {len(final.scenarios)} Szenarien, ein Roboter): {base_load[0]:.2f} → "
            f"{best_load[0]:.2f} Anfragen/s (×{best_load[0] / base_load[0]:.2f}); Auslastung des Inferenz-Threads "
            f"{pct(base_load[1])} → {pct(best_load[1])}. Rechnerisch trägt ein Inferenz-Thread damit {capacity} "
            "(ohne Warteschlangeneffekte)."
        )
    if final.sync is not None:
        lines.append(
            f"- **Sync-Referenz** (Phase 1, Simulation wartet auf das Brain, gleiche Seeds): "
            f"{pct(final.sync.success_rate)} Erfolg, Ø Max-Reward {fmt(final.sync.max_reward_mean, 3)}."
        )
    if noise is not None and noise[1].reward_delta is not None:
        scenario, c = noise
        lines.append(
            f"- **Messrauschen:** Zwei identische Phase-1-Läufe („{scenario_label(scenario)}“) unterscheiden sich "
            f"bereits um Δ Reward {c.reward_delta:+.3f} und {100 * c.success_delta:+.0f} pp Erfolg (n = {c.pairs}). "
            "Kleinere Effekte sind nicht belastbar."
        )
    return lines


# --------------------------------------------------------------------------------------------- figures
def point_style(style: Style) -> dict[str, Any]:
    """Marker of a strategy in scatter plots: filled circle, or a hollow diamond for the inpainting variant
    (hollow, so that it never hides the circle of the same request timing underneath)."""
    if style.inpaint:
        return {
            "marker": "D",
            "markersize": 6.5,
            "color": style.color,
            "markerfacecolor": "none",
            "markeredgecolor": style.color,
            "markeredgewidth": 1.6,
            "linestyle": "none",
        }
    return {
        "marker": "o",
        "markersize": 7.5,
        "color": style.color,
        "markerfacecolor": style.color,
        "markeredgecolor": SURFACE,
        "markeredgewidth": 1.2,
        "linestyle": "none",
    }


def legend_handles(stage: Stage, strategies: Sequence[str], bars: bool) -> list[Any]:
    """One entry per policy family (color) plus the inpainting variant (marker shape or texture)."""
    families: dict[str, str] = {}
    for strategy in strategies:
        style = style_of(strategy, stage.settings_of(strategy))
        families.setdefault(style.family, style.color)
    handles: list[Any] = []
    for family, color in families.items():
        if bars:
            handles.append(Patch(facecolor=color, edgecolor="none", label=family))
        elif family == "Baseline":
            handles.append(
                Line2D(
                    [],
                    [],
                    color=INK_2,
                    marker="o",
                    markersize=6.5,
                    markerfacecolor=REFERENCE_GRAY,
                    markeredgecolor=SURFACE,
                    linestyle="none",
                    label=family,
                )
            )
        else:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    markersize=6.5,
                    markeredgecolor=SURFACE,
                    linestyle="none",
                    label=family,
                )
            )
    if any(style_of(s, stage.settings_of(s)).inpaint for s in strategies):
        if bars:
            handles.append(
                Patch(
                    facecolor=INK_2,
                    edgecolor="none",
                    hatch=INPAINT_HATCH,
                    hatchcolor=SURFACE,
                    label="mit Inpainting",
                )
            )
        else:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=INK_2,
                    marker="D",
                    markersize=5.5,
                    markerfacecolor="none",
                    markeredgewidth=1.5,
                    linestyle="none",
                    label="hohle Raute = mit Inpainting",
                )
            )
    return handles


def figure_age_quality(stage: Stage, phase1_profiles: Sequence[ProfileData], path: Path) -> None:
    """Quality against mean plan age: phase-1 runs in gray, the stage's strategies linked to their baseline."""
    handles = legend_handles(stage, [BASELINE, *stage.candidates], bars=False)
    handles.append(
        Line2D(
            [],
            [],
            color=REFERENCE_GRAY,
            marker="o",
            markersize=5,
            markerfacecolor=SURFACE,
            linestyle="none",
            label="Phase 1 (hohl = synchron)",
        )
    )
    ncol = min(len(handles), 4)
    rows = math.ceil(len(handles) / ncol)
    fig, axes, layout = make_figure(
        1, 2, 4.4, 3.1, header_in=1.35 + 0.24 * rows, footer_in=0.8, wspace_in=0.8
    )
    top = header(
        fig,
        layout,
        f"Planalter und Qualität · Stufe {stage.key} ({stage.title})",
        f"Jeder Punkt = ein Lauf ({len(stage.seeds)} Seeds) · Linie = vom Baseline-Lauf desselben Szenarios zur "
        "Strategie · grau = alle Phase-1-Läufe beider Brain-Profile",
    )
    header_legend(fig, layout, handles, ncol=ncol, top_in=top)
    panels: list[tuple[str, Callable[[RunSummary], float | None], bool]] = [
        ("Erfolgsquote", lambda s: s.success_rate, True),
        ("Ø Max-Reward", lambda s: s.max_reward_mean, False),
    ]
    for ax, (title, value_of, is_rate) in zip(axes[0], panels, strict=True):
        points: list[tuple[float, float]] = []
        for profile in phase1_profiles:
            for label in ordered(profile.runs):
                s = profile.runs[label].summary
                y = value_of(s)
                if s.plan_age_mean_s is None or y is None:
                    continue
                sync = s.mode == "sync"
                ax.plot(
                    [s.plan_age_mean_s],
                    [y],
                    marker="o",
                    markersize=5,
                    linestyle="none",
                    color=REFERENCE_GRAY,
                    markerfacecolor=SURFACE if sync else REFERENCE_GRAY,
                    markeredgecolor=REFERENCE_GRAY if sync else SURFACE,
                    markeredgewidth=1.0,
                    alpha=0.75,
                    zorder=2,
                )
                points.append((s.plan_age_mean_s, float(y)))
        labels: list[tuple[float, float, str]] = []
        for scenario in stage.scenarios:
            base = stage.baselines.get(scenario)
            if base is None:
                continue
            bs = base.run.summary
            by = value_of(bs)
            if bs.plan_age_mean_s is None or by is None:
                continue
            bx = bs.plan_age_mean_s
            for strategy in stage.candidates:
                cell = stage.cells.get((strategy, scenario))
                if cell is None:
                    continue
                s = cell.run.summary
                y = value_of(s)
                if s.plan_age_mean_s is None or y is None:
                    continue
                style = style_of(strategy, stage.settings_of(strategy))
                ax.plot(
                    [bx, s.plan_age_mean_s], [by, y], color=style.color, linewidth=1.0, alpha=0.5, zorder=3
                )
                ax.plot([s.plan_age_mean_s], [y], zorder=5, **point_style(style))
                points.append((s.plan_age_mean_s, float(y)))
            ax.plot(
                [bx],
                [by],
                marker="o",
                markersize=7.5,
                linestyle="none",
                color=INK_2,
                markerfacecolor=REFERENCE_GRAY,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                zorder=6,
            )
            points.append((bx, float(by)))
            labels.append((bx, float(by), scenario_label(scenario)))
        xs = [p[0] for p in points] or [1.0]
        ys = [p[1] for p in points] or [1.0]
        ax.set_xlim(0, max(xs) * 1.12)
        if is_rate:
            ax.set_ylim(0, 1.0)
            percent_axis(ax)
        else:
            ax.set_ylim(min(0.8, math.floor((min(ys) - 0.02) * 20) / 20), 1.0)
        annotate_points(ax, points, labels)
        ax.set_xlabel("Ø Planalter der ausgeführten Aktion, s")
        ax.set_title(title, loc="left", pad=8)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_success(stage: Stage, path: Path) -> None:
    """Success per scenario and strategy (grouped bars, baseline first) with the synchronous reference."""
    shown = [BASELINE, *stage.candidates]
    handles = legend_handles(stage, shown, bars=True)
    handles.append(Line2D([], [], color=INK_2, linewidth=1.0, label="Wilson-95 %-KI"))
    if stage.sync is not None:
        handles.append(
            Line2D(
                [],
                [],
                color=MUTED,
                linewidth=1.2,
                linestyle="--",
                label=f"Sync-Referenz Phase 1 ({pct(stage.sync.success_rate)})",
            )
        )
    groups = stage.scenarios
    step = min(0.8 / len(shown), 0.24)
    panel_w = max(7.4, 0.21 * len(shown) * len(groups) + 0.6 * len(groups))
    ncol = min(len(handles), 4)
    rows = math.ceil(len(handles) / ncol)
    fig, axes, layout = make_figure(1, 1, panel_w, 3.0, header_in=1.3 + 0.24 * rows, footer_in=0.75)
    phase1_baseline = any(cell.source for cell in stage.baselines.values())
    top = header(
        fig,
        layout,
        f"Erfolgsquote je Szenario · Stufe {stage.key} ({stage.title})",
        f"{len(stage.seeds)} Seeds, in allen Balken dieselben · Baseline = Phase-1-Verhalten"
        + (" (die Phase-1-Läufe)" if phase1_baseline else ""),
    )
    header_legend(fig, layout, handles, ncol=ncol, top_in=top)
    ax = axes[0][0]
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 1.0)
    percent_axis(ax)
    ax.set_xticks(range(len(groups)), [scenario_label(s) for s in groups])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Erfolgsquote")
    if stage.sync is not None:
        ax.axhline(stage.sync.success_rate, color=MUTED, linewidth=1.2, linestyle="--", zorder=2)
    for group, scenario in enumerate(groups):
        for index, strategy in enumerate(shown):
            cell = (
                stage.baselines.get(scenario)
                if strategy == BASELINE
                else stage.cells.get((strategy, scenario))
            )
            if cell is None:
                continue
            s = cell.run.summary
            style = style_of(strategy, stage.settings_of(strategy))
            x = group + (index - (len(shown) - 1) / 2) * step
            hatch = INPAINT_HATCH if style.inpaint else None
            bar_with_round_end(ax, x, s.success_rate, step * 0.86, style.color, hatch=hatch)
            low, high = s.success_ci
            ax.plot([x, x], [low, high], color=INK_2, linewidth=0.9, zorder=4)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def adaptive_targets(stage: Stage) -> list[float]:
    targets: set[float] = set()
    for strategy in stage.candidates:
        request = stage.settings_of(strategy).get("request", {})
        if request.get("mode") == "adaptive" and request.get("target_age_s") is not None:
            targets.add(float(request["target_age_s"]))
    return sorted(targets)


def figure_tradeoff(stage: Stage, path: Path) -> None:
    """Mean plan age against the Brain load of one robot, one panel per scenario."""
    shown = [BASELINE, *stage.candidates]
    handles = legend_handles(stage, shown, bars=False)
    targets = adaptive_targets(stage)
    handles += [
        Line2D([], [], color=MUTED, linewidth=1.0, linestyle="--", label=f"Ziel-Planalter {target:g} s")
        for target in targets
    ]
    ncol = min(len(handles), 4)
    rows = math.ceil(len(handles) / ncol)
    fig, axes, layout = make_figure(
        1,
        len(stage.scenarios),
        3.7,
        3.0,
        header_in=1.35 + 0.24 * rows,
        footer_in=0.8,
        wspace_in=0.75,
        sharey=True,
    )
    top = header(
        fig,
        layout,
        "Frischere Pläne kosten Brain-Last",
        f"Stufe {stage.key} ({stage.title}) · x = Anteil der Zeit, in der das Brain für diesen einen Roboter "
        "rechnet · y = mittleres Planalter der ausgeführten Aktionen",
    )
    header_legend(fig, layout, handles, ncol=ncol, top_in=top)
    placed: list[tuple[Any, list[tuple[float, float]], list[tuple[float, float, str]]]] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for ax, scenario in zip(axes[0], stage.scenarios, strict=True):
        points: list[tuple[float, float]] = []
        labels: list[tuple[float, float, str]] = []
        for strategy in shown:
            cell = (
                stage.baselines.get(scenario)
                if strategy == BASELINE
                else stage.cells.get((strategy, scenario))
            )
            if cell is None:
                continue
            s = cell.run.summary
            if s.inference_utilization is None or s.plan_age_mean_s is None:
                continue
            style = style_of(strategy, stage.settings_of(strategy))
            x, y = s.inference_utilization, s.plan_age_mean_s
            baseline = strategy == BASELINE
            marker = point_style(style)
            if baseline:
                marker.update(color=INK_2, markerfacecolor=REFERENCE_GRAY)
            ax.plot([x], [y], zorder=4 if baseline or style.inpaint else 3, **marker)
            points.append((x, y))
            if not style.inpaint:
                labels.append((x, y, style.family))
        for target in targets:
            ax.axhline(target, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
        all_x += [p[0] for p in points]
        all_y += [p[1] for p in points]
        ax.set_title(scenario_label(scenario), loc="left", pad=8)
        ax.set_xlabel("Auslastung Inferenz-Thread")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f} %"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f} s"))
        placed.append((ax, points, labels))
    for ax, points, labels in placed:  # common limits first, then labels that know the final geometry
        ax.set_xlim(0, max(1.0, 1.12 * max(all_x, default=1.0)))
        ax.set_ylim(0, max(1.2, 1.15 * max(all_y, default=1.0)))
        annotate_points(ax, points, labels)
    axes[0][0].set_ylabel("Ø Planalter")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- markdown
STATE_OF_THE_ART = [
    "- **Asynchrone Inferenz mit Queue-Schwelle** (LeRobot Async Inference / SmolVLA, Shukor et al. 2025): neue "
    "Beobachtung, sobald die Aktions-Queue unter einen Anteil fällt. Genau das ist die Phase-1-Baseline; die "
    "Schwelle ist statisch und kennt weder Latenz noch Planalter.",
    "- **Real-Time Chunking (RTC)** (Black, Galliker, Levine 2025; in LeRobot unter `policies/rtc`): der nächste "
    "Chunk wird als Inpainting-Problem erzeugt – die Aktionen, die während der Inferenz noch laufen, sind "
    "eingefroren, der Rest der Überlappung wird mit abfallenden Gewichten (bestes Schema: exponentiell) zum "
    "laufenden Plan gezogen. Durchgesetzt per ΠGDM-Guidance, also mit einem Backward-Pass durch den Denoiser pro "
    "Denoising-Schritt, und in LeRobot nur für Flow-Matching-Policies (π0, π0.5, SmolVLA) umgesetzt.",
    "- **Ersatz-Inpainting** (Diffuser, Janner et al. 2022; RePaint, Lugmayr et al. 2022): bekannte Teile werden in "
    "jedem Denoising-Schritt ersetzt – ohne Gradienten. Kortyn nutzt diese Variante auf der x₀-Vorhersage mit den "
    "RTC-Gewichten (`kortyn/inpainting.py`): ein paar Tensor-Operationen pro Schritt statt eines Backward-Passes, "
    "der das 0,7-s-Budget auf der CPU sprengen würde. Gewicht 0 ist bitgenau LeRobots Sampler.",
    "- **Gewichtetes Überblenden** (Temporal Ensembling, ACT, Zhao et al. 2023) mittelt alte und neue Pläne bei der "
    "Ausführung. Bei multimodalen Diffusion-Policies kann das Mittel zweier gültiger Pläne selbst ungültig sein "
    "(zwischen zwei Wegen um den Block); deshalb Inpainting statt Blending.",
    "- **Bidirectional Decoding** (Liu et al. 2024) wählt unter mehreren gesampelten Chunks den konsistentesten – "
    "mehrfache Inferenz pro Anfrage, also das Gegenteil von „ohne unnötige Brain-Last“.",
]

LIMITATIONS = [
    "- **Eine Maschine, ein Roboter, CPU-Brain.** Alle Zahlen stammen von einer 4-vCPU-VM ohne GPU. Absolute "
    "Latenzen und damit die Planalter gelten nur hier; die Relationen zwischen den Strategien sind übertragbar, "
    "solange das Verhältnis Inferenzzeit zu Takt ähnlich bleibt.",
    "- **Brain-Last mit einem Roboter gemessen.** Mit mehreren Robotern konkurrieren die Anfragen um den "
    "Inferenz-Thread: die Wartezeit steigt, damit Latenz und Planalter. Die adaptive Strategie reagiert darauf "
    "(Rückkopplung über die Latenzschätzung), die Schwellen nicht – beides ist ungetestet (Phase 3).",
    "- **PushT ist quasistatisch.** Anhalten kostet hier wenig, weil der Block liegen bleibt. Bei dynamischen "
    "Aufgaben ist der Zielkonflikt zwischen Anhalten (sicher) und Weiterfahren mit altem Plan anders gewichtet.",
    "- **Ein Inpainting-Schema getestet** (exponentiell, ≤ 10 Takte Überlappung). Ersatz-Inpainting erzwingt den "
    "eingefrorenen Teil exakt, für den weichen Teil ist es eine Heuristik ohne Konsistenzgarantie (RTC löst das "
    "mit Guidance – auf der CPU zu teuer).",
    "- **Statistische Trennschärfe.** n = 20 (Stufe 1) bzw. 50 (Stufe 2) gepaarte Seeds: McNemar erkennt nur "
    "große Unterschiede im Erfolg; der kontinuierliche Max-Reward ist empfindlicher. Alle p-Werte zusätzlich "
    "Holm-korrigiert über die Vergleiche einer Stufe.",
    "- **Auswahl-Verzerrung.** Die Strategien für Stufe 2 wurden auf „gut“ und „sehr schlecht“ mit den Seeds "
    "1000–1019 ausgewählt; diese Seeds sind in Stufe 2 enthalten (neu gemessen, die Policy ist stochastisch). Die "
    "vier übrigen Szenarien waren an der Auswahl nicht beteiligt.",
]


def noise_note(phase1: ProfileData | None) -> list[str]:
    """Limitation bullet: two identical real-time runs differ by more than seed sampling explains."""
    noise = run_noise(phase1)
    if noise is None or noise[1].reward_delta is None:
        return []
    scenario, c = noise
    return [
        f"- **Lauf-zu-Lauf-Rauschen.** Zwei identische Phase-1-Läufe („{scenario_label(scenario)}“ und seine "
        f"Wiederholung, gleiche Sitzung, n = {c.pairs}) unterscheiden sich um Δ Reward {c.reward_delta:+.3f} "
        f"(Permutation p = {p_text(c.reward_p)}) und {100 * c.success_delta:+.0f} pp Erfolg. Echtzeit ist nicht "
        "bitgenau wiederholbar: Latenz-Jitter verschiebt einzelne Planwechsel um einen Takt, danach laufen die "
        "Episoden auseinander. Die gepaarten Tests behandeln nur die Seeds als Zufall, nicht den Lauf; "
        "Unterschiede in dieser Größenordnung sind deshalb nicht belastbar, auch wenn sie nominal signifikant sind."
    ]


def drift_note(final: Stage, screening: Stage | None, phase1: ProfileData | None) -> list[str]:
    """Limitation bullet when stage 2 is compared with the phase-1 runs: they were not measured interleaved."""
    if final.key != "2" or not any(cell.source for cell in final.baselines.values()):
        return []
    text = (
        "- **Stufe-2-Baseline aus Phase 1.** Die Strategien in Stufe 2 laufen gegen die Phase-1-Läufe, nicht "
        "zeitgleich verschränkt; Drift der Maschine zwischen den Messungen ginge in die Differenzen ein."
    )
    if screening is not None and phase1 is not None:
        latencies = []
        for scenario in screening.scenarios:
            own = screening.cells.get((BASELINE, scenario))
            old = restrict(phase1.runs[scenario], screening.seeds) if scenario in phase1.runs else None
            if own is not None and old is not None:
                latencies.append(
                    f"{fmt(old.summary.e2e_p50_ms)} → {fmt(own.run.summary.e2e_p50_ms)} ms ({scenario_label(scenario)})"
                )
        if latencies:
            text += (
                " Die Reproduktion in Stufe 1 begrenzt das: E2E-Latenz der Baseline (Median) Phase 1 → heute "
                + ", ".join(latencies)
                + "."
            )
    return [text]


def built_section(stage: Stage) -> list[str]:
    return [
        "1. **Latenzschätzung pro Roboter** (`kortyn/latency.py`): gleitendes Fenster der letzten E2E-Latenzen "
        "(Beobachtung → Chunk beim Roboter), Median und Streuung als 1,4826 · MAD – robust gegen Ausreißer wie "
        "Stalls; bis 3 Messungen ein konservativer Prior.",
        "2. **Anfrage-Strategien** (`kortyn/freshness.py`): (a) feste Schwellen 6 (Baseline), 8, 10; (b) adaptiv – "
        "Deadline (der neue Plan kommt mit dem Latenz-Quantil vor dem Leerlaufen der Queue an) **und** Frische "
        "(Anfrageperiode P so, dass das erwartete Ø-Planalter D + (P − 1)/2 Takte das Ziel nicht übersteigt), aber "
        "nicht öfter.",
        "3. **Zeitliche Ausrichtung:** jede Aktion trägt ihren Steuertakt; Aktionen, deren Zeitpunkt beim "
        "Eintreffen verstrichen ist, werden übersprungen (`alignment: timestep`; `shift` = naiver Gegenentwurf "
        "für Tests).",
        "4. **Glättung der Übergänge:** gradientenfreies Inpainting im Brain (`kortyn/inpainting.py`); der Agent "
        "schickt mit jeder Anfrage die eingefrorenen Aktionen (vorheriger und aktueller Takt sowie alles, was bis "
        "zum erwarteten Eintreffen läuft) und die weich geführte Überlappung mit.",
        "5. **Notfall nur bei Budgetverletzung:** Queue leer oder Planalter über der harten Grenze → Ereignis "
        "loggen, Roboter hält an (Positionsziel = aktuelle Position), kein lokales Modell.",
        "6. **Brain-Last:** Anfragen/s, Auslastung des Inferenz-Threads (Inferenzzeit / Episodenzeit) und "
        "Wartezeit im Brain aus den Request-Records; die neuen Brain-Zähler (`GetStats`-RPC) prüfen das gegen.",
        "",
        "Alle Strategien sind per YAML schaltbar (`kortyn/experiments/phase2/*.yaml`, Schlüssel `strategies`). Die "
        "Baseline (`StrategyConfig()`) ist exakt das Phase-1-Verhalten.",
        "",
        *strategy_table(stage),
    ]


DEFINITIONS = [
    "- **Planalter:** Zeit von der Freigabe der Beobachtung, aus der ein Plan berechnet wurde, bis zur Ausführung "
    "einer seiner Aktionen; für jede ausgeführte Aktion geloggt (Phase 1: Aktionsalter in Takten / 10 Hz).",
    "- **Zeit verhungert:** Anteil der Takte ohne Aktion in der Queue (Definition aus Phase 1).",
    "- **Notfall:** Takt ohne nutzbare Aktion (Queue leer oder Planalter > Grenze); Ereignis = zusammenhängende "
    "Folge solcher Takte. Strategien mit Notfall-Halt halten an; die Baseline wiederholt die letzte Aktion "
    "(Phase-1-Verhalten).",
    "- **Anfragen/s:** beantwortete Echtzeit-Anfragen pro Sekunde Episodenzeit. **Auslastung Inferenz:** "
    "Inferenzzeit im Brain / Episodenzeit (Anteil, den ein Roboter vom Inferenz-Thread belegt). **Wartezeit "
    "Brain:** Zeit in der Brain-Queue vor der Inferenz.",
    "- **Ø Max-Reward:** LeRobots `max_reward` je Episode (PushT: beste Abdeckung / 95 %, gedeckelt auf 1).",
    "- **Δ:** gepaarte Differenz Strategie − Baseline auf denselben Seeds. **Holm:** p-Werte korrigiert für die "
    "Zahl der Vergleiche einer Stufe; fett = nach Holm signifikant (α = 0,05).",
]


def methodology(config: KortynConfig, profile_name: str, final: Stage) -> list[str]:
    phase1_baseline = final.key == "2" and any(cell.source for cell in final.baselines.values())
    stage2 = (
        "- **Stufe 2 – Bestätigung:** die zwei Besten × 50 Seeds auf allen vier Szenarien plus den zwei "
        "Ablationen ohne Stall."
    )
    if phase1_baseline:
        stage2 += (
            " Baseline = die Phase-1-Läufe (identische Konfiguration, gleiche Seeds; Stufe 1 prüft die "
            "Reproduzierbarkeit) – das spart ≈ 2 h Messzeit."
        )
    return [
        "## Methodik",
        "",
        f"- **Gleich wie Phase 1:** Umgebung (PushT, {config.env.fps:g} Hz), Checkpoint, Brain-Profil "
        f"`{profile_name}`, Netzszenarien, Seeds ab {config.experiment.seed_start}, seed-weise verschränkte "
        "Reihenfolge, Validitäts-Gate mit Wiederholung.",
        "- **Stufe 1 – Screening:** alle Strategien × 20 Seeds auf „gut“ und „sehr schlecht“, Baseline frisch "
        "mitgemessen (Reproduktionsprüfung).",
        "- **Auswahlregel (vor Stufe 1 festgelegt):** größte mittlere gepaarte Δ Ø Max-Reward gegenüber der "
        f"Baseline über beide Szenarien; liegen Strategien weniger als {TIE_MARGIN:g} auseinander, gewinnt die mit "
        "weniger Anfragen/s.",
        stage2,
        "- **Tests:** Erfolg mit exaktem McNemar, Max-Reward mit gepaartem Vorzeichen-Permutationstest und "
        "Bootstrap-KI über Seeds; Holm-Korrektur über alle Vergleiche einer Stufe.",
        "",
    ]


def write_phase2_report(config: KortynConfig, results_dir: Path, reports_dir: Path) -> Path | None:
    """Write ``reports/phase2.md`` with its figures; None if there are no phase-2 results yet."""
    root = results_dir / PHASE2_DIR
    found = sorted(
        {
            p.name
            for _, _, directory in STAGES
            if (root / directory).is_dir()
            for p in (root / directory).iterdir()
            if p.is_dir()
        }
    )
    if not found:
        return None
    suite_profiles = [suite.profile for suite in config.experiment.suites]
    profile_name = next((p for p in suite_profiles if p in found), found[0])
    if len(found) > 1:
        LOG.warning("phase-2 results for several Brain profiles %s; reporting %s", found, profile_name)
    phase1_profiles: list[ProfileData] = []
    for name in dict.fromkeys([profile_name, *suite_profiles]):
        data = load_profile(results_dir / name) if (results_dir / name).is_dir() else None
        if data is not None:
            phase1_profiles.append(data)
    phase1 = next((p for p in phase1_profiles if p.name == profile_name), None)
    stages: list[Stage] = []
    for key, title, directory in STAGES:
        stage = load_stage(key, title, root / directory / profile_name, phase1)
        if stage is not None:
            stages.append(stage)
    if not stages:
        return None
    apply_style()
    reports_dir.mkdir(parents=True, exist_ok=True)
    screening = next((s for s in stages if s.key == "1"), None)
    final = stages[-1]
    comparisons = {stage.key: compare_stage(stage) for stage in stages}
    ranks = rank_strategies(screening, comparisons["1"]) if screening is not None else []
    rule = select_strategies(ranks)
    if final.key == "2":
        confirmed = final.candidates
        winners = select_strategies(rank_strategies(final, comparisons["2"]), count=1)
        best = winners[0] if winners else None
    else:
        confirmed = rule
        best = rule[0] if rule else None

    figures: dict[str, str] = {}
    for stage in stages:
        figures[f"success{stage.key}"] = f"phase2_erfolg_stufe{stage.key}.png"
        figure_success(stage, reports_dir / figures[f"success{stage.key}"])
        figures[f"age{stage.key}"] = f"phase2_planalter_qualitaet_stufe{stage.key}.png"
        figure_age_quality(stage, phase1_profiles, reports_dir / figures[f"age{stage.key}"])
    if screening is not None:
        figures["tradeoff"] = "phase2_planalter_last.png"
        figure_tradeoff(screening, reports_dir / figures["tradeoff"])

    lines: list[str] = [
        "# Kortyn Phase 2 – Freshness Engine",
        "",
        f"_Erzeugt {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} mit `python -m kortyn.report` aus "
        f"`{root}/` (Brain-Profil `{profile_name}`); Phase-1-Referenz: `{results_dir / profile_name}/`._",
        "",
        "**Frage:** Phase 1 zeigte: nicht der Stillstand kostet Qualität, sondern das Alter der ausgeführten Pläne "
        "(ab ≈ 0,75 s Ø-Planalter sinkt der Erfolg). Phase 2 hält das Planalter klein, ohne unnötige Brain-Last, "
        "und misst, wie viel Erfolg in schlechten Netzen zurückkommt.",
        "",
        f"## Kurzfassung (Stufe {final.key}, {len(final.seeds)} Seeds)",
        "",
        *key_findings(final, comparisons[final.key], confirmed, best, run_noise(phase1)),
        "",
    ]
    if final.key == "2":
        lines += [
            "## Auf einen Blick",
            "",
            *overview_table(final, comparisons["2"], final.candidates),
            "",
            "Δ = gepaarte Differenz zur Baseline auf denselben Seeds; fett = signifikant nach Holm-Korrektur "
            f"(α = {ALPHA:g}). Planalter = Ø Alter der ausgeführten Aktionen (Beobachtung → Ausführung).",
            "",
            f"![Erfolg je Szenario, Stufe 2]({figures['success2']})",
            "",
        ]
    if screening is not None:
        lines += hypothesis_section(age_evidence(screening, phase1))
    lines += ["## Was gebaut wurde", "", *built_section(screening or final), ""]
    lines += methodology(config, profile_name, final)
    if screening is not None:
        lines += [f"## Stufe 1 · Screening ({len(screening.seeds)} Seeds)", ""]
        if phase1 is not None:
            result = reproduction(screening, phase1)
            noise = run_noise(phase1)
            lines += ["### Reproduktion der Baseline (Abnahmekriterium 2)", "", *result.rows, ""]
            lines += reproduction_text(result, None if noise is None else noise[1])
        lines += [
            "### Ergebnisse je Szenario",
            "",
            *stage_tables(screening, comparisons["1"], highlight=rule),
        ]
        lines += ["### Auswahl für Stufe 2", "", *ranking_table(screening, ranks, rule), ""]
        if final.key == "2" and sorted(final.candidates) != sorted(rule):
            lines += [
                f"⚠️ In Stufe 2 gemessen: {', '.join(final.candidates)} – laut Regel: {', '.join(rule)}.",
                "",
            ]
        lines += [
            f"![Planalter gegen Brain-Last]({figures['tradeoff']})",
            "",
            f"![Erfolg je Szenario, Stufe 1]({figures['success1']})",
            "",
            f"![Planalter und Qualität, Stufe 1]({figures['age1']})",
            "",
        ]
    if final.key == "2":
        lines += [f"## Stufe 2 · Bestätigung ({len(final.seeds)} Seeds)", ""]
        lines += stage_tables(final, comparisons["2"], highlight=[best] if best else [])
        lines += [f"![Planalter und Qualität, Stufe 2]({figures['age2']})", ""]
    lines += [
        "## Diagnose",
        "",
        "<details><summary>Messhygiene, Notfall-Ursachen, Latenzschätzer, Brain-Zähler</summary>",
        "",
    ]
    for stage in stages:
        lines += [f"**Stufe {stage.key}**", "", *diagnostics_table(stage), ""]
    lines += [
        "Lesehilfe: *Notfall-Takte* = Takte ohne nutzbare Aktion (leer = Queue leer, zu alt = Planalter über der "
        "harten Grenze); bei der Baseline sind das die verhungerten Takte, in denen sie die letzte Aktion "
        "wiederholt. *Reaktion* = Beobachtung → erste ausgeführte Aktion eines Plans. *Auslastung Records / "
        "Zähler* = Inferenzzeit aus den Request-Records gegen die Brain-Zähler (`GetStats`); stimmen sie überein, "
        "ist die Lastmessung konsistent. *im Planungs-Quantil* = Anteil der Anfragen, deren E2E-Latenz unter dem "
        "geschätzten Median + z · σ lag.",
        "",
        *diagnostic_findings(stages),
        "</details>",
        "",
        "## Stand der Technik und Designentscheidungen",
        "",
        *STATE_OF_THE_ART,
        "",
        "## Grenzen",
        "",
        *LIMITATIONS,
        *noise_note(phase1),
        *drift_note(final, screening, phase1),
        "",
        "## Definitionen",
        "",
        *DEFINITIONS,
        "",
    ]
    report = reports_dir / "phase2.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def selection_lines(stage: Stage, comparisons: dict[tuple[str, str], Comparison]) -> list[str]:
    ranks = rank_strategies(stage, comparisons)
    chosen = select_strategies(ranks)
    lines = [f"Stufe {stage.key}: {len(stage.seeds)} Seeds, Szenarien {', '.join(stage.scenarios)}"]
    for index, rank in enumerate(ranks, start=1):
        deltas = ", ".join(f"{s} {d:+.3f}" for s, d in rank.reward_deltas.items())
        mark = "  <- Stufe 2" if rank.strategy in chosen else ""
        lines.append(
            f"{index:2d}. {rank.strategy:20s} Score {rank.score:+.4f} ({deltas}), "
            f"{rank.requests_per_s:.2f} Anfragen/s{mark}"
        )
    lines.append(f"Auswahl: {', '.join(chosen) if chosen else '-'}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase-2 report (reports/phase2.md) or the stage-2 selection."
    )
    parser.add_argument("--config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument(
        "--select", action="store_true", help="print the stage-1 ranking and the strategies for stage 2"
    )
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
    if not args.select:
        report = write_phase2_report(config, results_dir, reports_dir)
        if report is None:
            LOG.error("no phase-2 results in %s", results_dir / PHASE2_DIR)
            return 1
        LOG.info("report written to %s", report)
        return 0
    directory = results_dir / PHASE2_DIR / STAGES[0][2]
    profile_dirs = sorted(p for p in directory.iterdir() if p.is_dir()) if directory.is_dir() else []
    if not profile_dirs:
        LOG.error("no stage-1 results in %s", directory)
        return 1
    for profile_dir in profile_dirs:
        phase1_dir = results_dir / profile_dir.name
        phase1 = load_profile(phase1_dir) if phase1_dir.is_dir() else None
        stage = load_stage(STAGES[0][0], STAGES[0][1], profile_dir, phase1)
        if stage is None:
            LOG.error("no usable runs in %s", profile_dir)
            return 1
        print(f"[{profile_dir.name}]")
        print("\n".join(selection_lines(stage, compare_stage(stage))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
