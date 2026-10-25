"""Phase-3a report data: ``reports/phase3a_data.md`` and the figures ``reports/phase3a_*.png``.

    python -m kortyn.report_phase3a

Inputs (below ``results/``):

* phase 1 (real time): ``<profile>/<run>/``;
* validation in simulation time: ``phase3a/validation_model/<profile>/<timing>__<run>/`` (latency model) and
  ``phase3a/validation_replay/<profile>/replay__<run>/`` (replay of the phase-1 latencies);
* determinism: ``phase3a/determinism/<profile>/<run>/``, the same runs executed a second time;
* SmolVLA on Meta-World: ``phase3a/smolvla/<task>/smolvla/<run>/``;
* delay tolerance of the scripted Meta-World experts: ``phase3a/expert_delay.json``.

Statistics: success rates with Wilson intervals; every other run-level quantity with a percentile bootstrap
over episodes. Differences between two runs on the same seeds are paired: the interval resamples seeds
jointly; p-values from the exact McNemar test (success) or a sign-flip permutation test (per-episode values),
Holm-adjusted per family. Minimal detectable effects (two-sided alpha = 5 %, power 80 %) use McNemar's
sample-size formula (success) and the paired normal approximation (continuous quantities).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from kortyn.metrics import (
    EpisodeRecord,
    RequestRecord,
    final_episodes,
    holm_adjust,
    max_reward,
    mcnemar_exact,
    paired_permutation_p,
    percentile,
    plan_ages_s,
    read_jsonl,
    wilson_interval,
)
from kortyn.report import (
    CATEGORICAL,
    DPI,
    INK_2,
    MUTED,
    REFERENCE_GRAY,
    SEVERITY,
    apply_style,
    header,
    header_legend,
    make_figure,
    percent_axis,
)
from kortyn.virtual_agent import deterministic_view

LOG = logging.getLogger("kortyn.report.phase3a")

ALPHA = 0.05
POWER = 0.8
N_BOOT = 2000
SEED_COUNTS = (50, 100, 200)
PROFILES = ("kalibriert", "ddim10")
VALIDATION_RUNS = (
    "referenz_sync",
    "gut",
    "gut_wiederholung",
    "buero_wlan",
    "schlecht",
    "sehr_schlecht",
    "schlecht_ohne_stall",
    "sehr_schlecht_ohne_stall",
)
SCENARIO_LABELS = {
    "referenz_sync": "Referenz (sync)",
    "gut": "gut",
    "gut_wiederholung": "gut (Wiederholung)",
    "buero_wlan": "Büro-WLAN",
    "schlecht": "schlecht",
    "sehr_schlecht": "sehr schlecht",
    "schlecht_ohne_stall": "schlecht o. Stall",
    "sehr_schlecht_ohne_stall": "sehr schlecht o. Stall",
}
NETWORK_SCENARIOS = ("gut", "buero_wlan", "schlecht", "sehr_schlecht")
METRICS = ("success", "reward", "plan_age", "starvation")
METRIC_LABELS = {
    "success": "Erfolg",
    "reward": "Ø Max-Reward",
    "plan_age": "Planalter",
    "starvation": "Starvation",
}

# SmolVLA on Meta-World
SMOLVLA_TASKS = ("drawer-close-v3", "push-v3", "coffee-push-v3")
TASK_DIRS = {task: task.removesuffix("-v3").replace("-", "_") for task in SMOLVLA_TASKS}
TASK_GROUPS = {"drawer-close-v3": "easy", "push-v3": "hard", "coffee-push-v3": "medium"}
# Published success of lerobot/smolvla_metaworld per task: Early-Exits paper (arXiv 2609.29382v2), Fig. 4,
# 30 episodes per task, "full policy" (read off a heatmap; evaluation protocol not stated in more detail).
PUBLISHED_TASK_SUCCESS = {"drawer-close-v3": 1.00, "push-v3": 0.70, "coffee-push-v3": 0.90}
PUBLISHED_EPISODES = 30
# SmolVLA paper (arXiv 2506.01844, Tab. 2), 0.45B, own checkpoint (100k steps), re-planning after every
# action, 10 trials per task: success per Meta-World difficulty group.
PAPER_GROUP_SUCCESS = {"easy": 0.825, "medium": 0.418, "hard": 0.450, "very hard": 0.600}
GPU_TIMINGS = ("schnell", "mittel", "langsam")
GPU_LABELS = {
    "schnell": "RTX 4090 (99 ms)",
    "mittel": "RTX 3090 (174 ms)",
    "langsam": "Apple M4 (311 ms)",
    "grenzfall": "Apple M1 (722 ms)",
}
SYNC_MATCHED = "referenz_sync25"
SYNC_PUBLISHED = "referenz_sync50"


# --------------------------------------------------------------------------------------------- data
@dataclass
class Run:
    """One run: the final attempt of every seed and the requests of those attempts."""

    label: str
    episodes: dict[int, EpisodeRecord]
    requests: list[RequestRecord]
    invalid_attempts: int

    def seeds(self) -> list[int]:
        return sorted(self.episodes)

    def ordered(self, seeds: Sequence[int] | None = None) -> list[EpisodeRecord]:
        return [self.episodes[s] for s in (seeds if seeds is not None else self.seeds())]

    def requests_of(self, phases: Sequence[str] = ("realtime", "sync")) -> list[RequestRecord]:
        return [r for r in self.requests if r.phase in phases and r.status == "ok"]


def load_run(directory: Path) -> Run | None:
    path = directory / "episodes.jsonl"
    if not path.is_file():
        return None
    episodes, invalid = final_episodes(EpisodeRecord.from_json(record) for record in read_jsonl(path))
    if not episodes:
        return None
    chosen = {(e.seed, e.attempt) for e in episodes}
    requests = [
        RequestRecord.from_json(record)
        for record in read_jsonl(directory / "requests.jsonl")
        if (record["seed"], record["attempt"]) in chosen
    ]
    return Run(directory.name, {e.seed: e for e in episodes}, requests, invalid)


# --------------------------------------------------------------------------------------------- statistics
@dataclass(frozen=True)
class Interval:
    value: float
    low: float
    high: float

    def contains(self, x: float, tolerance: float = 1e-12) -> bool:
        return self.low - tolerance <= x <= self.high + tolerance


def episode_reward(episode: EpisodeRecord) -> float:
    """LeRobot's per-episode max reward: max step reward if recorded, else derived from the PushT coverage."""
    if episode.max_step_reward is not None:
        return float(episode.max_step_reward)
    return max_reward(episode.max_coverage)


def episode_plan_age(episode: EpisodeRecord) -> float:
    ages = plan_ages_s(episode)
    return float(np.mean(ages)) if ages else math.nan


def episode_starvation(episode: EpisodeRecord) -> float:
    return sum(episode.ticks.starved) / episode.steps if episode.steps else 0.0


def pooled(metric: str, episodes: Sequence[EpisodeRecord]) -> float:
    """Run-level value of a metric: rates and means over episodes, plan age and starvation pooled over ticks."""
    if not episodes:
        return math.nan
    if metric == "success":
        return sum(1 for e in episodes if e.success) / len(episodes)
    if metric == "reward":
        return float(np.mean([episode_reward(e) for e in episodes]))
    if metric == "plan_age":
        ages = [age for e in episodes for age in plan_ages_s(e)]
        return float(np.mean(ages)) if ages else math.nan
    if metric == "starvation":
        ticks = sum(e.steps for e in episodes)
        return sum(sum(e.ticks.starved) for e in episodes) / ticks if ticks else 0.0
    raise ValueError(f"unknown metric {metric!r}")


def per_episode(metric: str, episode: EpisodeRecord) -> float:
    if metric == "success":
        return 1.0 if episode.success else 0.0
    if metric == "reward":
        return episode_reward(episode)
    if metric == "plan_age":
        return episode_plan_age(episode)
    if metric == "starvation":
        return episode_starvation(episode)
    raise ValueError(f"unknown metric {metric!r}")


def bootstrap_interval(
    episodes: Sequence[EpisodeRecord],
    statistic: Callable[[Sequence[EpisodeRecord]], float],
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap over episodes."""
    if len(episodes) < 2:
        value = statistic(episodes)
        return value, value
    rng = np.random.default_rng(seed)
    n = len(episodes)
    values = [statistic([episodes[i] for i in rng.integers(0, n, n)]) for _ in range(n_boot)]
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return math.nan, math.nan
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def estimate(metric: str, episodes: Sequence[EpisodeRecord], seed: int = 0) -> Interval:
    value = pooled(metric, episodes)
    if metric == "success":
        low, high = wilson_interval(sum(1 for e in episodes if e.success), len(episodes))
        return Interval(value, low, high)
    low, high = bootstrap_interval(episodes, lambda sample: pooled(metric, sample), seed=seed)
    return Interval(value, low, high)


@dataclass(frozen=True)
class Delta:
    """Paired difference ``b - a`` over the common seeds."""

    metric: str
    n: int
    interval: Interval
    p: float


def paired_delta(a: Run, b: Run, metric: str, n_boot: int = N_BOOT, seed: int = 0) -> Delta | None:
    seeds = sorted(set(a.episodes) & set(b.episodes))
    if not seeds:
        return None
    ea, eb = a.ordered(seeds), b.ordered(seeds)
    point = pooled(metric, eb) - pooled(metric, ea)
    rng = np.random.default_rng(seed)
    n = len(seeds)
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diff = pooled(metric, [eb[i] for i in idx]) - pooled(metric, [ea[i] for i in idx])
        if math.isfinite(diff):
            samples.append(diff)
    low, high = (
        (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
        if samples
        else (point, point)
    )
    if metric == "success":
        only_a = sum(1 for x, y in zip(ea, eb, strict=True) if x.success and not y.success)
        only_b = sum(1 for x, y in zip(ea, eb, strict=True) if y.success and not x.success)
        p = mcnemar_exact(only_a, only_b)
    else:
        diffs = [per_episode(metric, y) - per_episode(metric, x) for x, y in zip(ea, eb, strict=True)]
        finite = [d for d in diffs if math.isfinite(d)]
        p = paired_permutation_p(finite) if finite and any(d != 0 for d in finite) else 1.0
    return Delta(metric, n, Interval(point, low, high), p)


def z_quantile(probability: float) -> float:
    return NormalDist().inv_cdf(probability)


def mde_paired_binary(n: int, discordance: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest difference in success rate a paired McNemar test detects (Connor 1987 sample-size formula).

    Solves ``n = (z_a sqrt(p) + z_b sqrt(p - d^2))^2 / d^2`` for ``d``, where ``p`` is the share of discordant
    pairs. ``nan`` if not even ``d = p`` (all discordant pairs in one direction) is detectable.
    """
    if n < 1 or not 0 < discordance <= 1:
        return math.nan
    za, zb = z_quantile(1 - alpha / 2), z_quantile(power)

    def excess(d: float) -> float:
        return n * d * d - (za * math.sqrt(discordance) + zb * math.sqrt(max(0.0, discordance - d * d))) ** 2

    if excess(discordance) < 0:
        return math.nan
    low, high = 0.0, discordance
    for _ in range(100):
        mid = (low + high) / 2
        if excess(mid) < 0:
            low = mid
        else:
            high = mid
    return high


def mde_paired_mean(n: int, sd: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest mean paired difference detectable with a paired test (normal approximation)."""
    if n < 2 or sd < 0:
        return math.nan
    return (z_quantile(1 - alpha / 2) + z_quantile(power)) * sd / math.sqrt(n)


def newcombe_difference(s1: int, n1: int, s2: int, n2: int) -> Interval:
    """Difference of two independent proportions ``p1 - p2`` with Newcombe's hybrid score interval."""
    p1, p2 = s1 / n1, s2 / n2
    l1, u1 = wilson_interval(s1, n1)
    l2, u2 = wilson_interval(s2, n2)
    d = p1 - p2
    return Interval(
        d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    )


@dataclass(frozen=True)
class PairNoise:
    """How much two runs on the same seeds differ: discordant successes, spread of reward differences."""

    name: str
    n: int
    success_a: float
    success_b: float
    discordance: float
    reward_sd: float
    identical: int


def pair_noise(name: str, a: Run, b: Run) -> PairNoise | None:
    seeds = sorted(set(a.episodes) & set(b.episodes))
    if len(seeds) < 2:
        return None
    ea, eb = a.ordered(seeds), b.ordered(seeds)
    discordant = sum(1 for x, y in zip(ea, eb, strict=True) if x.success != y.success)
    diffs = [episode_reward(y) - episode_reward(x) for x, y in zip(ea, eb, strict=True)]
    identical = sum(
        1
        for x, y in zip(ea, eb, strict=True)
        if x.steps == y.steps and x.max_coverage == y.max_coverage and x.ticks.starved == y.ticks.starved
    )
    return PairNoise(
        name,
        len(seeds),
        pooled("success", ea),
        pooled("success", eb),
        discordant / len(seeds),
        float(np.std(diffs, ddof=1)),
        identical,
    )


# --------------------------------------------------------------------------------------------- replay
@dataclass(frozen=True)
class ReplayMatch:
    episodes: int
    identical: int  # outcome, length, tick series and max coverage bit for bit
    same_outcome: int  # identical apart from the last float bits of the coverage
    diverged: int
    fallbacks: int  # requests the replay could not take from the log


def replay_match(real: Run, replay: Run) -> ReplayMatch:
    identical = same = diverged = 0
    seeds = sorted(set(real.episodes) & set(replay.episodes))
    for seed in seeds:
        x, y = real.episodes[seed], replay.episodes[seed]
        timing_equal = (
            x.success == y.success
            and x.steps == y.steps
            and x.ticks.starved == y.ticks.starved
            and x.ticks.action_age == y.ticks.action_age
            and x.ticks.queue_len == y.ticks.queue_len
        )
        if timing_equal and x.max_coverage == y.max_coverage and x.sum_reward == y.sum_reward:
            identical += 1
        elif timing_equal and abs(x.max_coverage - y.max_coverage) < 1e-9:
            same += 1
        else:
            diverged += 1
    fallbacks = sum(int((e.virtual or {}).get("replay_fallback") or 0) for e in replay.episodes.values())
    return ReplayMatch(len(seeds), identical, same, diverged, fallbacks)


def determinism_check(first: Run, second: Run) -> tuple[int, int]:
    """(episodes with identical deterministic content, common episodes) of two executions of one run."""
    seeds = sorted(set(first.episodes) & set(second.episodes))
    equal = sum(
        1
        for seed in seeds
        if json.dumps(deterministic_view(first.episodes[seed].to_json()), sort_keys=True)
        == json.dumps(deterministic_view(second.episodes[seed].to_json()), sort_keys=True)
    )
    return equal, len(seeds)


# --------------------------------------------------------------------------------------------- formatting
def fmt_pct(value: float, digits: int = 0) -> str:
    return "–" if not math.isfinite(value) else f"{100 * value:.{digits}f} %"


def fmt_interval(interval: Interval, kind: str) -> str:
    """Value with its 95 % interval; ``kind``: pct (rates), pp (rate differences), s, ms, num."""
    if not math.isfinite(interval.value):
        return "–"
    if kind == "pct":
        return f"{100 * interval.value:.0f} % [{100 * interval.low:.0f}–{100 * interval.high:.0f}]"
    if kind == "pctf":
        return f"{100 * interval.value:.1f} % [{100 * interval.low:.1f}–{100 * interval.high:.1f}]"
    if kind == "pp":
        return f"{100 * interval.value:+.0f} pp [{100 * interval.low:+.0f}, {100 * interval.high:+.0f}]"
    if kind == "ppf":
        return f"{100 * interval.value:+.1f} pp [{100 * interval.low:+.1f}, {100 * interval.high:+.1f}]"
    if kind == "s":
        return f"{interval.value:.3f} s [{interval.low:.3f}–{interval.high:.3f}]"
    if kind == "ds":
        return f"{1000 * interval.value:+.0f} ms [{1000 * interval.low:+.0f}, {1000 * interval.high:+.0f}]"
    if kind == "dnum":
        return f"{interval.value:+.3f} [{interval.low:+.3f}, {interval.high:+.3f}]"
    return f"{interval.value:.3f} [{interval.low:.3f}–{interval.high:.3f}]"


DELTA_KIND = {"success": "pp", "reward": "dnum", "plan_age": "ds", "starvation": "ppf"}
LEVEL_KIND = {"success": "pct", "reward": "num", "plan_age": "s", "starvation": "pctf"}


def mark(delta: Delta | None) -> str:
    if delta is None:
        return "–"
    return "✓" if delta.interval.contains(0.0) else "✗"


def p_text(p: float) -> str:
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def ms(value: float | None) -> str:
    return "–" if value is None or not math.isfinite(value) else f"{value:.0f} ms"


# --------------------------------------------------------------------------------------------- validation
@dataclass
class ValidationCase:
    profile: str
    scenario: str
    real: Run
    model: Run | None
    replay: Run | None
    model_deltas: dict[str, Delta]
    replay_deltas: dict[str, Delta]
    episode: Run | None = None  # latency model, service time drawn from the same phase-1 episode
    episode_deltas: dict[str, Delta] | None = None

    def deltas(self, source: str) -> dict[str, Delta]:
        if source == "model":
            return self.model_deltas
        if source == "episode":
            return self.episode_deltas or {}
        return self.replay_deltas


SOURCES = (
    ("model", "Simulationszeit: Modell (gepoolt)"),
    ("episode", "Simulationszeit: Modell (episodengenau)"),
    ("replay", "Simulationszeit: Replay"),
)
SOURCE_SHORT = {"model": "Modell", "episode": "Modell episodengenau", "replay": "Replay"}


def load_validation(results: Path) -> list[ValidationCase]:
    cases: list[ValidationCase] = []
    for profile in PROFILES:
        for scenario in VALIDATION_RUNS:
            real = load_run(results / profile / scenario)
            if real is None:
                continue
            model_label = "gut" if scenario == "gut_wiederholung" else scenario
            model_root = results / "phase3a" / "validation_model" / profile
            model = load_run(model_root / f"modell__{model_label}")
            episode = load_run(model_root / f"episode__{scenario}")
            replay = load_run(results / "phase3a" / "validation_replay" / profile / f"replay__{scenario}")
            if model is None and replay is None and episode is None:
                continue
            cases.append(
                ValidationCase(
                    profile,
                    scenario,
                    real,
                    model,
                    replay,
                    all_deltas(real, model),
                    all_deltas(real, replay),
                    episode,
                    all_deltas(real, episode),
                )
            )
    return cases


def all_deltas(real: Run, other: Run | None) -> dict[str, Delta]:
    if other is None:
        return {}
    return {m: d for m in METRICS if (d := paired_delta(real, other, m)) is not None}


@dataclass(frozen=True)
class Verdict:
    comparisons: int
    inside: int  # paired 95 % interval of the difference contains 0 (main criterion)
    holm_significant: int
    outside: list[str]
    in_real_interval: int  # the simulation-time value lies in the 95 % interval of the real-time measurement


def source_run(case: ValidationCase, source: str) -> Run | None:
    return {"model": case.model, "episode": case.episode, "replay": case.replay}[source]


def verdict(cases: Sequence[ValidationCase], source: str) -> Verdict:
    labels: list[str] = []
    deltas: list[Delta] = []
    in_real = 0
    for case in cases:
        table = case.deltas(source)
        run = source_run(case, source)
        for metric, delta in table.items():
            labels.append(f"{case.profile}/{SCENARIO_LABELS[case.scenario]}: {METRIC_LABELS[metric]}")
            deltas.append(delta)
            if run is not None:
                seeds = sorted(set(case.real.episodes) & set(run.episodes))
                real_interval = estimate(metric, case.real.ordered(seeds))
                in_real += int(real_interval.contains(pooled(metric, run.ordered(seeds)), tolerance=1e-9))
    adjusted = holm_adjust([d.p for d in deltas]) if deltas else []
    outside = [label for label, d in zip(labels, deltas, strict=True) if not d.interval.contains(0.0)]
    return Verdict(
        len(deltas), len(deltas) - len(outside), sum(1 for p in adjusted if p < ALPHA), outside, in_real
    )


def validation_table(cases: Sequence[ValidationCase], profile: str) -> list[str]:
    lines = [
        "| Szenario | Messung | Erfolg [95-%-KI] | Ø Max-Reward [KI] | Planalter [KI] | Starvation [KI] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in (c for c in cases if c.profile == profile):
        sources: list[tuple[str, Run | None]] = [
            ("Echtzeit (Phase 1)", case.real),
            ("Simulationszeit: Modell (gepoolt)", case.model),
            ("Simulationszeit: Modell (episodengenau)", case.episode),
            ("Simulationszeit: Replay", case.replay),
        ]
        first = True
        for name, run in sources:
            if run is None:
                continue
            episodes = run.ordered()
            cells = [fmt_interval(estimate(m, episodes), LEVEL_KIND[m]) for m in METRICS]
            scenario = f"**{SCENARIO_LABELS[case.scenario]}**" if first else ""
            first = False
            lines.append(f"| {scenario} | {name} | " + " | ".join(cells) + " |")
    return lines


def delta_table(cases: Sequence[ValidationCase], profile: str) -> list[str]:
    lines = [
        "| Szenario | Quelle | Δ Erfolg [KI] | Δ Ø Max-Reward [KI] | Δ Planalter [KI] | Δ Starvation [KI] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in (c for c in cases if c.profile == profile):
        for source, _ in SOURCES:
            table = case.deltas(source)
            name = SOURCE_SHORT[source]
            if not table:
                continue
            cells = [
                f"{mark(table.get(m))} {fmt_interval(table[m].interval, DELTA_KIND[m])}"
                if m in table
                else "–"
                for m in METRICS
            ]
            lines.append(f"| {SCENARIO_LABELS[case.scenario]} | {name} | " + " | ".join(cells) + " |")
    return lines


def latency_table(cases: Sequence[ValidationCase]) -> list[str]:
    lines = [
        "| Profil | Szenario | E2E p50 Echtzeit / Modell | E2E p95 Echtzeit / Modell | Netzanteil Ø Echtzeit / Modell |",
        "|---|---|---:|---:|---:|",
    ]
    for case in cases:
        if case.model is None or case.scenario in ("referenz_sync", "gut_wiederholung"):
            continue
        real_r, model_r = case.real.requests_of(("realtime",)), case.model.requests_of(("realtime",))

        def values(records: Sequence[RequestRecord], attribute: str) -> list[float]:
            return [float(v) for r in records if (v := getattr(r, attribute)) is not None]

        e2e_real, e2e_model = values(real_r, "e2e_ms"), values(model_r, "e2e_ms")
        net_real, net_model = values(real_r, "network_ms"), values(model_r, "network_ms")
        lines.append(
            f"| {case.profile} | {SCENARIO_LABELS[case.scenario]} | "
            f"{ms(percentile(e2e_real, 50))} / {ms(percentile(e2e_model, 50))} | "
            f"{ms(percentile(e2e_real, 95))} / {ms(percentile(e2e_model, 95))} | "
            f"{np.mean(net_real):.1f} / {np.mean(net_model):.1f} ms |"
        )
    return lines


def replay_table(cases: Sequence[ValidationCase]) -> tuple[list[str], ReplayMatch]:
    lines = [
        "| Profil | Szenario | Episoden | bitgleich | gleich bis auf letzte Float-Bits | abweichend | Fallbacks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    total = ReplayMatch(0, 0, 0, 0, 0)
    for case in cases:
        if case.replay is None:
            continue
        match = replay_match(case.real, case.replay)
        total = ReplayMatch(
            total.episodes + match.episodes,
            total.identical + match.identical,
            total.same_outcome + match.same_outcome,
            total.diverged + match.diverged,
            total.fallbacks + match.fallbacks,
        )
        lines.append(
            f"| {case.profile} | {SCENARIO_LABELS[case.scenario]} | {match.episodes} | {match.identical} | "
            f"{match.same_outcome} | {match.diverged} | {match.fallbacks} |"
        )
    lines.append(
        f"| **Summe** | | **{total.episodes}** | **{total.identical}** | **{total.same_outcome}** | "
        f"**{total.diverged}** | **{total.fallbacks}** |"
    )
    return lines, total


# Executions of the same model runs: (description, first results root, second results root) below results/.
DETERMINISM_PAIRS = (
    (
        "1. gegen 2. Ausführung (4 statt 1 Worker; Code-Stand der Validierung gegen aktuellen)",
        "phase3a/validation_model",
        "phase3a/determinism",
    ),
    (
        "2. gegen 3. Ausführung (1 statt 2 Worker, gleicher Code)",
        "phase3a/determinism",
        "phase3a/determinism_b",
    ),
)


@dataclass(frozen=True)
class DeterminismRow:
    comparison: str
    profile: str
    run: str
    match: ReplayMatch  # identical / identical but the last float bits of the coverage / diverged
    same_requests: int
    requests: int


def determinism_rows(results: Path) -> list[DeterminismRow]:
    """Repeated executions of model runs (other worker count, other order of the seeds, fresh processes)."""
    rows: list[DeterminismRow] = []
    for comparison, first_root, second_root in DETERMINISM_PAIRS:
        root = results / second_root
        if not root.is_dir():
            continue
        for profile_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in profile_dir.iterdir() if p.is_dir()):
                first = load_run(results / first_root / profile_dir.name / run_dir.name)
                second = load_run(run_dir)
                if first is None or second is None:
                    continue
                seeds = set(first.episodes) & set(second.episodes)

                def views(run: Run, seeds: set[int] = seeds) -> dict[tuple[int, int, int], str]:
                    return {
                        (r.seed, r.attempt, r.request_id): json.dumps(r.to_json(), sort_keys=True)
                        for r in run.requests
                        if r.seed in seeds
                    }

                a, b = views(first), views(second)
                rows.append(
                    DeterminismRow(
                        comparison,
                        profile_dir.name,
                        run_dir.name,
                        replay_match(first, second),
                        sum(1 for key, value in a.items() if b.get(key) == value),
                        len(a.keys() | b.keys()),
                    )
                )
    return rows


def determinism_table(rows: Sequence[DeterminismRow]) -> list[str]:
    lines = [
        "| Vergleich | Profil | Lauf | Episoden | bitgleich | gleich bis auf letzte Float-Bits | abweichend "
        "| Anfragen identisch |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row.match
        lines.append(
            f"| {row.comparison} | {row.profile} | {row.run} | {m.episodes} | {m.identical} | {m.same_outcome} "
            f"| {m.diverged} | {row.same_requests} / {row.requests} |"
        )
    return lines


def service_drift(results: Path, profile: str, block: int = 10) -> list[str]:
    """Server time of the real-time machine per phase-1 run and per block of seeds: the hardware drifted
    between sessions and within a session, which a pooled service-time profile cannot follow."""
    runs: list[tuple[str, list[RequestRecord]]] = []
    for scenario in VALIDATION_RUNS:
        run = load_run(results / profile / scenario)
        if run is None:
            continue
        requests = [r for r in run.requests_of(("realtime",)) if r.server_total_ms is not None]
        if requests:
            runs.append((scenario, requests))
    if not runs:
        return []
    seeds = sorted({r.seed for _, requests in runs for r in requests})
    blocks = [seeds[i : i + block] for i in range(0, len(seeds), block)]
    lines = [
        "| Lauf | Median | " + " | ".join(f"Seeds {b[0]}–{b[-1]}" for b in blocks) + " |",
        "|---|---:|" + "---:|" * len(blocks),
    ]
    for scenario, requests in runs:
        cells = []
        for members in blocks:
            chosen = set(members)
            values = [
                float(r.server_total_ms)
                for r in requests
                if r.seed in chosen and r.server_total_ms is not None
            ]
            cells.append(f"{np.mean(values):.0f}" if values else "–")
        median = float(
            np.median([float(r.server_total_ms) for r in requests if r.server_total_ms is not None])
        )
        lines.append(f"| {SCENARIO_LABELS[scenario]} | {median:.0f} | " + " | ".join(cells) + " |")
    return lines


# --------------------------------------------------------------------------------------------- noise, power
@dataclass(frozen=True)
class NoiseModel:
    name: str
    description: str
    discordance: float
    reward_sd: float


def noise_rows(results: Path) -> tuple[list[PairNoise], list[NoiseModel]]:
    model = results / "phase3a" / "validation_model"
    pairs: list[tuple[str, Path, Path]] = [
        ("Echtzeit A/A: gut vs. gut (Wiederholung), kalibriert", results / "kalibriert" / "gut", results / "kalibriert" / "gut_wiederholung"),
        ("Simulationszeit A/A, neue Latenzen: gut, kalibriert", model / "kalibriert" / "modell__gut", model / "kalibriert" / "modell_r2__gut"),
        ("Simulationszeit A/A, neue Latenzen: sehr schlecht, kalibriert", model / "kalibriert" / "modell__sehr_schlecht", model / "kalibriert" / "modell_r2__sehr_schlecht"),
        ("Simulationszeit A/A, neue Latenzen: gut, ddim10", model / "ddim10" / "modell__gut", model / "ddim10" / "modell_r2__gut"),
        ("Echtzeit: gut vs. Büro-WLAN, kalibriert", results / "kalibriert" / "gut", results / "kalibriert" / "buero_wlan"),
        ("Simulationszeit: gut vs. Büro-WLAN, kalibriert", model / "kalibriert" / "modell__gut", model / "kalibriert" / "modell__buero_wlan"),
        ("Echtzeit: Referenz (sync) vs. gut, kalibriert", results / "kalibriert" / "referenz_sync", results / "kalibriert" / "gut"),
        ("Simulationszeit: Referenz (sync) vs. gut, kalibriert", model / "kalibriert" / "modell__referenz_sync", model / "kalibriert" / "modell__gut"),
        ("Echtzeit: gut vs. sehr schlecht, kalibriert", results / "kalibriert" / "gut", results / "kalibriert" / "sehr_schlecht"),
        ("Simulationszeit: gut vs. sehr schlecht, kalibriert", model / "kalibriert" / "modell__gut", model / "kalibriert" / "modell__sehr_schlecht"),
    ]  # fmt: skip
    noise: list[PairNoise] = []
    for name, first, second in pairs:
        a, b = load_run(first), load_run(second)
        if a is not None and b is not None and (row := pair_noise(name, a, b)) is not None:
            noise.append(row)
    by_name = {row.name: row for row in noise}
    models: list[NoiseModel] = []
    realtime_aa = by_name.get(pairs[0][0])
    if realtime_aa is not None:
        models.append(
            NoiseModel(
                "Echtzeit (Phase 1)",
                "identische Bedingung zweimal gemessen",
                realtime_aa.discordance,
                realtime_aa.reward_sd,
            )
        )
    redrawn = [by_name[p[0]] for p in pairs[1:4] if p[0] in by_name]
    if redrawn:
        models.append(
            NoiseModel(
                "Simulationszeit, Latenzen neu gezogen",
                "Mittel der A/A-Replikationen",
                float(np.mean([r.discordance for r in redrawn])),
                float(np.sqrt(np.mean([r.reward_sd**2 for r in redrawn]))),
            )
        )
    sync_pair = by_name.get(pairs[7][0])
    if sync_pair is not None:
        models.append(
            NoiseModel(
                "Simulationszeit, Bedingung vs. Sync-Referenz",
                "Planungswert für Schritt 3 (Referenz vs. gut)",
                sync_pair.discordance,
                sync_pair.reward_sd,
            )
        )
    return noise, models


def noise_table(rows: Sequence[PairNoise]) -> list[str]:
    lines = [
        "| Vergleich (gleiche Seeds) | n | Erfolg A / B | diskordante Paare | SD der Reward-Differenz | identische Episoden |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.n} | {fmt_pct(row.success_a)} / {fmt_pct(row.success_b)} | "
            f"{fmt_pct(row.discordance)} | {row.reward_sd:.3f} | {row.identical} |"
        )
    return lines


def power_table(models: Sequence[NoiseModel]) -> list[str]:
    head = " | ".join(f"n = {n}" for n in SEED_COUNTS)
    lines = [
        f"| Rauschmodell | diskordant | SD Reward-Diff. | MDE Erfolg ({head}) | MDE Ø Max-Reward ({head}) |",
        "|---|---:|---:|---|---|",
    ]
    for model in models:
        success = " / ".join(fmt_pct(mde_paired_binary(n, model.discordance), 0) for n in SEED_COUNTS)
        reward = " / ".join(f"{mde_paired_mean(n, model.reward_sd):.3f}" for n in SEED_COUNTS)
        lines.append(
            f"| {model.name} ({model.description}) | {fmt_pct(model.discordance)} | {model.reward_sd:.3f} | "
            f"{success} | {reward} |"
        )
    return lines


# --------------------------------------------------------------------------------------------- SmolVLA
@dataclass
class TaskData:
    task: str
    runs: dict[str, Run]
    meta: dict[str, Any]


def load_smolvla(results: Path) -> list[TaskData]:
    tasks: list[TaskData] = []
    for task in SMOLVLA_TASKS:
        directory = results / "phase3a" / "smolvla" / TASK_DIRS[task] / "smolvla"
        if not directory.is_dir():
            continue
        runs = {
            p.name: run
            for p in sorted(directory.iterdir())
            if p.is_dir() and (run := load_run(p)) is not None
        }
        meta_path = directory / "run_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        if runs:
            tasks.append(TaskData(task, runs, meta))
    return tasks


def run_bandwidth(run: Run) -> tuple[float, float, float]:
    """(KB per request on the wire, requests per second, Mbit/s per robot in both directions)."""
    requests = run.requests_of()
    duration = sum(e.steps / e.fps for e in run.episodes.values())
    if not requests or duration <= 0:
        return math.nan, math.nan, math.nan
    up = float(np.mean([r.request_bytes for r in requests]))
    total = sum(r.request_bytes + (r.response_bytes or 0) for r in requests)
    return up / 1000.0, len(requests) / duration, 8.0 * total / duration / 1e6


def label_parts(label: str) -> tuple[str, str]:
    """(timing, scenario) of a SmolVLA run label; references map to ("sync", label)."""
    if label.startswith("referenz_"):
        return "sync", label
    timing, _, scenario = label.partition("__")
    return timing, scenario


def smolvla_table(data: TaskData) -> list[str]:
    reference = data.runs.get(SYNC_MATCHED)
    lines = [
        "| Bedingung | Erfolg [95-%-KI] | Δ vs. Sync-25 [KI] (p) | Ø Max-Reward | E2E p50 / p95 | Planalter | Starvation | Bandbreite/Roboter |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = (
        [SYNC_PUBLISHED, SYNC_MATCHED]
        + [f"{t}__{s}" for t in GPU_TIMINGS for s in NETWORK_SCENARIOS]
        + ["grenzfall__gut"]
    )
    for label in order:
        run = data.runs.get(label)
        if run is None:
            continue
        episodes = run.ordered()
        success = estimate("success", episodes)
        delta = (
            paired_delta(reference, run, "success")
            if reference is not None and label != SYNC_MATCHED
            else None
        )
        delta_text = "–" if delta is None else f"{fmt_interval(delta.interval, 'pp')} ({p_text(delta.p)})"
        e2e = [float(r.e2e_ms) for r in run.requests_of() if r.e2e_ms is not None]
        timing, scenario = label_parts(label)
        name = {
            SYNC_PUBLISHED: "Sync, 50 Aktionen/Plan (LeRobot-Eval)",
            SYNC_MATCHED: "Sync, 25 Aktionen/Plan (Referenz)",
        }.get(label, f"{GPU_LABELS.get(timing, timing)}, {SCENARIO_LABELS.get(scenario, scenario)}")
        _, _, mbps = run_bandwidth(run)
        realtime = (
            timing != "sync"
        )  # the synchronous references wait for every plan: no latency, no starvation
        starvation = pooled("starvation", episodes) if realtime else math.nan
        latency = f"{ms(percentile(e2e, 50))} / {ms(percentile(e2e, 95))}" if realtime else "–"
        lines.append(
            f"| {name} | {fmt_interval(success, 'pct')} | {delta_text} | {pooled('reward', episodes):.2f} | "
            f"{latency} | {pooled('plan_age', episodes):.2f} s | {fmt_pct(starvation, 1)} | {mbps:.1f} Mbit/s |"
        )
    return lines


def realized_power(tasks: Sequence[TaskData]) -> list[str]:
    """Noise of the SmolVLA comparisons as measured (each real-time condition against Sync-25 on the same
    seeds) and the minimal detectable effect it implies at the measured number of seeds."""
    lines = [
        "| Aufgabe | Bedingungen | Seeds | diskordante Paare Ø [min–max] | SD Reward-Diff. Ø | MDE Erfolg | "
        "MDE Ø Max-Reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for data in tasks:
        reference = data.runs.get(SYNC_MATCHED)
        if reference is None:
            continue
        rows = [
            noise
            for label, run in data.runs.items()
            if label_parts(label)[0] in (*GPU_TIMINGS, "grenzfall")
            and (noise := pair_noise(label, reference, run)) is not None
        ]
        if not rows:
            continue
        n = min(row.n for row in rows)
        discordance = [row.discordance for row in rows]
        sd = float(np.mean([row.reward_sd for row in rows]))
        mean_discordance = float(np.mean(discordance))
        lines.append(
            f"| {data.task} | {len(rows)} | {n} | {fmt_pct(mean_discordance)} "
            f"[{fmt_pct(min(discordance))}–{fmt_pct(max(discordance))}] | {sd:.2f} | "
            f"{fmt_pct(mde_paired_binary(n, mean_discordance))} | {mde_paired_mean(n, sd):.2f} |"
        )
    return lines if len(lines) > 2 else []


def published_comparison(tasks: Sequence[TaskData], first_seed: int = 1000) -> list[str]:
    """Sync references against the published success of the checkpoint: over all seeds, and over the seeds
    a LeRobot evaluation with its default seed and the published episode count would use."""
    last_seed = first_seed + PUBLISHED_EPISODES - 1
    lines = [
        "| Aufgabe | Gruppe | Kortyn Sync-50 (LeRobot-Protokoll) | Kortyn Sync-25 | publiziert (dieser Checkpoint) "
        f"| Δ Sync-50 − publiziert [KI] | Sync-50, Seeds {first_seed}–{last_seed} | Δ auf diesen Seeds [KI] |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for data in tasks:
        sync50, sync25 = data.runs.get(SYNC_PUBLISHED), data.runs.get(SYNC_MATCHED)
        published = PUBLISHED_TASK_SUCCESS.get(data.task)
        cells: list[str] = []
        for run in (sync50, sync25):
            cells.append("–" if run is None else fmt_interval(estimate("success", run.ordered()), "pct"))
        delta = matched = matched_delta = "–"
        if sync50 is not None and published is not None:
            reference = round(published * PUBLISHED_EPISODES)
            successes = sum(1 for e in sync50.episodes.values() if e.success)
            diff = newcombe_difference(successes, len(sync50.episodes), reference, PUBLISHED_EPISODES)
            delta = f"{mark_interval(diff)} {fmt_interval(diff, 'pp')}"
            seeds = [s for s in range(first_seed, last_seed + 1) if s in sync50.episodes]
            if len(seeds) == PUBLISHED_EPISODES:
                hits = sum(1 for s in seeds if sync50.episodes[s].success)
                matched = f"{hits}/{len(seeds)} ({fmt_pct(hits / len(seeds))})"
                same = newcombe_difference(hits, len(seeds), reference, PUBLISHED_EPISODES)
                matched_delta = f"{mark_interval(same)} {fmt_interval(same, 'pp')}"
        lines.append(
            f"| {data.task} | {TASK_GROUPS[data.task]} | {cells[0]} | {cells[1]} | "
            f"{fmt_pct(published or math.nan)} (n = {PUBLISHED_EPISODES}) | {delta} | {matched} | {matched_delta} |"
        )
    return lines


def mark_interval(interval: Interval) -> str:
    return "✓" if interval.contains(0.0) else "✗"


# --------------------------------------------------------------------------------------------- latency effect
@dataclass(frozen=True)
class EffectPoint:
    family: str  # "PushT kalibriert", "push-v3", ...
    condition: str
    latency_ms: float  # median end-to-end latency
    relative_success: float  # success / success of the matched sync reference
    success: float
    period_ms: float  # time between two plans when the latency fits (refill period of the agent)

    @property
    def latency_periods(self) -> float:
        """Latency in units of the planning period: how stale a plan is compared with how long it lasts."""
        return self.latency_ms / self.period_ms


def plan_period_ms(meta: dict[str, Any], fps: float) -> float:
    """Ticks between two requests at zero latency (chunk - threshold - 1), in ms."""
    agent = meta.get("agent", {})
    ticks = int(agent.get("actions_per_chunk", 15)) - int(agent.get("refill_threshold", 6)) - 1
    return 1000.0 * ticks / fps


def read_meta(directory: Path) -> dict[str, Any]:
    path = directory / "run_meta.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def latency_effects(results: Path, tasks: Sequence[TaskData]) -> list[EffectPoint]:
    points: list[EffectPoint] = []
    model = results / "phase3a" / "validation_model"
    for profile in PROFILES:
        sync = load_run(model / profile / "modell__referenz_sync")
        if sync is None:
            continue
        base = pooled("success", sync.ordered())
        period = plan_period_ms(read_meta(model / profile), sync.ordered()[0].fps)
        for scenario in NETWORK_SCENARIOS:
            run = load_run(model / profile / f"modell__{scenario}")
            if run is None or base <= 0:
                continue
            e2e = [float(r.e2e_ms) for r in run.requests_of(("realtime",)) if r.e2e_ms is not None]
            success = pooled("success", run.ordered())
            points.append(
                EffectPoint(
                    f"PushT {profile}", scenario, float(np.median(e2e)), success / base, success, period
                )
            )
    for data in tasks:
        sync = data.runs.get(SYNC_MATCHED)
        if sync is None:
            continue
        base = pooled("success", sync.ordered())
        period = plan_period_ms(data.meta, sync.ordered()[0].fps)
        for label, run in data.runs.items():
            timing, _scenario = label_parts(label)
            if timing not in (*GPU_TIMINGS, "grenzfall") or base <= 0:
                continue
            e2e = [float(r.e2e_ms) for r in run.requests_of(("realtime",)) if r.e2e_ms is not None]
            if not e2e:
                continue
            success = pooled("success", run.ordered())
            points.append(
                EffectPoint(data.task, label, float(np.median(e2e)), success / base, success, period)
            )
    return points


def delta_difference(
    ref_a: Run, cond_a: Run, ref_b: Run, cond_b: Run, n_boot: int = N_BOOT, seed: int = 0
) -> Interval | None:
    """(success drop of b) - (success drop of a): paired within each task, independent between tasks."""
    seeds_a = sorted(set(ref_a.episodes) & set(cond_a.episodes))
    seeds_b = sorted(set(ref_b.episodes) & set(cond_b.episodes))
    if not seeds_a or not seeds_b:
        return None
    da = [float(cond_a.episodes[s].success) - float(ref_a.episodes[s].success) for s in seeds_a]
    db = [float(cond_b.episodes[s].success) - float(ref_b.episodes[s].success) for s in seeds_b]
    point = float(np.mean(db) - np.mean(da))
    rng = np.random.default_rng(seed)
    samples = [
        float(np.mean(rng.choice(db, len(db))) - np.mean(rng.choice(da, len(da)))) for _ in range(n_boot)
    ]
    return Interval(point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def task_comparison(tasks: Sequence[TaskData]) -> list[str]:
    """Success drop against the matched sync reference per task and condition, and dynamic minus static."""
    by_task = {data.task: data for data in tasks}
    if not by_task:
        return []
    names = [t for t in SMOLVLA_TASKS if t in by_task]
    head = " | ".join(f"Δ {t}" for t in names)
    lines = [
        f"| Bedingung | {head} | coffee-push − push [KI] |",
        "|---|" + "---:|" * len(names) + "---:|",
    ]
    labels = [f"{t}__{s}" for t in GPU_TIMINGS for s in NETWORK_SCENARIOS] + ["grenzfall__gut"]
    for label in labels:
        cells: list[str] = []
        for task in names:
            data = by_task[task]
            ref, run = data.runs.get(SYNC_MATCHED), data.runs.get(label)
            delta = paired_delta(ref, run, "success") if ref is not None and run is not None else None
            cells.append("–" if delta is None else fmt_interval(delta.interval, "pp"))
        diff = "–"
        if "push-v3" in by_task and "coffee-push-v3" in by_task:
            static, dynamic = by_task["push-v3"], by_task["coffee-push-v3"]
            parts = (
                static.runs.get(SYNC_MATCHED),
                static.runs.get(label),
                dynamic.runs.get(SYNC_MATCHED),
                dynamic.runs.get(label),
            )
            if all(part is not None for part in parts):
                ref_a, cond_a, ref_b, cond_b = (p for p in parts if p is not None)
                interval = delta_difference(ref_a, cond_a, ref_b, cond_b)
                if interval is not None:
                    diff = f"{mark_interval(interval)} {fmt_interval(interval, 'pp')}"
        timing, scenario = label_parts(label)
        lines.append(
            f"| {GPU_LABELS[timing]}, {SCENARIO_LABELS[scenario]} | " + " | ".join(cells) + f" | {diff} |"
        )
    return lines


def transport_table(tasks: Sequence[TaskData]) -> list[str]:
    """Image transport ablation: payload, bandwidth and success against raw frames (same seeds)."""
    lines: list[str] = []
    for data in tasks:
        variants = sorted(label for label in data.runs if label.startswith("transport_"))
        reference = data.runs.get(SYNC_MATCHED)
        if not variants or reference is None:
            continue
        lines += [
            f"**{data.task}** (synchrone Referenz, 25 Aktionen pro Plan):",
            "",
            "| Bildübertragung | KB pro Anfrage | Bandbreite/Roboter bei 3,2 Anfragen/s | Erfolg [KI] | Δ vs. roh [KI] (p) |",
            "|---|---:|---:|---:|---:|",
        ]
        for label in [SYNC_MATCHED, *variants]:
            run = data.runs[label]
            kb, _, _ = run_bandwidth(run)
            seeds = sorted(set(run.episodes) & set(reference.episodes))
            success = estimate("success", run.ordered(seeds))
            delta = paired_delta(reference, run, "success") if label != SYNC_MATCHED else None
            name = {SYNC_MATCHED: "roh, 480×480 (LeRobot-Eval)"}.get(label, label.removeprefix("transport_"))
            per_robot = 8.0 * kb * 1000.0 * (80.0 / 25.0) / 1e6
            delta_text = "–" if delta is None else f"{fmt_interval(delta.interval, 'pp')} ({p_text(delta.p)})"
            lines.append(
                f"| {name} | {kb:.1f} | {per_robot:.1f} Mbit/s | {fmt_interval(success, 'pct')} | {delta_text} |"
            )
        lines.append("")
    return lines


def latency_slope(points: Sequence[EffectPoint], family: str, max_latency_ms: float) -> float | None:
    """Least-squares slope of the relative success per 100 ms of median latency (latencies <= max)."""
    chosen = [p for p in points if p.family == family and p.latency_ms <= max_latency_ms]
    if len(chosen) < 3:
        return None
    x = np.array([p.latency_ms for p in chosen]) / 100.0
    y = np.array([p.relative_success for p in chosen])
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


# --------------------------------------------------------------------------------------------- figures
def figure_validation(cases: Sequence[ValidationCase], path: Path) -> None:
    rows = [c for c in cases if any(c.deltas(source) for source, _ in SOURCES)]
    if not rows:
        return
    fig, axes, layout = make_figure(1, 2, 3.6, 0.32 * len(rows) + 0.6, left_in=2.6, header_in=1.25)
    top = header(
        fig,
        layout,
        "Simulationszeit gegen Echtzeit (gepaarte Differenz, 95-%-KI)",
        "Je Profil und Szenario: Simulationszeit minus Echtzeit auf denselben 50 Seeds. Liegt das Intervall "
        "über der Null, stimmen beide Messungen innerhalb der Unsicherheit überein.",
    )
    sources = (
        ("Modell (gepoolt)", "model", CATEGORICAL[0], "o"),
        ("Modell (episodengenau)", "episode", CATEGORICAL[2], "D"),
        ("Replay", "replay", CATEGORICAL[1], "s"),
    )
    labels = [f"{c.profile}: {SCENARIO_LABELS[c.scenario]}" for c in rows]
    for ax, metric, scale, unit in zip(axes[0], ("success", "reward"), (100.0, 1.0), ("pp", ""), strict=True):
        for offset, (_name, source, color, marker) in zip((-0.22, 0.0, 0.22), sources, strict=True):
            for y, case in enumerate(rows):
                delta = case.deltas(source).get(metric)
                if delta is None:
                    continue
                iv = delta.interval
                ax.plot([scale * iv.low, scale * iv.high], [y + offset, y + offset], color=color, linewidth=2)
                ax.plot(scale * iv.value, y + offset, marker=marker, color=color, markersize=6, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.8)  # fmt: skip
        ax.axvline(0.0, color=MUTED, linewidth=1.0, zorder=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels if metric == "success" else [""] * len(rows))
        ax.invert_yaxis()
        ax.set_title(f"Δ {METRIC_LABELS[metric]}" + (f" ({unit})" if unit else ""), loc="left")
        ax.grid(axis="y", visible=False)
    handles = [
        Line2D([], [], color=color, marker=marker, linewidth=2, label=f"Simulationszeit: {name}")
        for name, _, color, marker in sources
    ]
    header_legend(fig, layout, handles, ncol=3, top_in=top + 0.05)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_latency_cdf(cases: Sequence[ValidationCase], path: Path) -> None:
    profiles = [p for p in PROFILES if any(c.profile == p and c.model is not None for c in cases)]
    if not profiles:
        return
    fig, axes, layout = make_figure(1, len(profiles), 3.8, 2.8, header_in=1.3)
    top = header(
        fig,
        layout,
        "Ende-zu-Ende-Latenz: Echtzeit (durchgezogen) gegen Latenzmodell (gestrichelt)",
        "Empirische Verteilungsfunktion der Latenz je Anfrage (Beobachtung bis Plan beim Roboter).",
    )
    for ax, profile in zip(axes[0], profiles, strict=True):
        for case in cases:
            if case.profile != profile or case.model is None or case.scenario not in NETWORK_SCENARIOS:
                continue
            color = SEVERITY[case.scenario]
            for run, style in ((case.real, "-"), (case.model, "--")):
                values = np.sort(
                    [float(r.e2e_ms) for r in run.requests_of(("realtime",)) if r.e2e_ms is not None]
                )
                if values.size:
                    ax.plot(
                        values, np.arange(1, values.size + 1) / values.size, style, color=color, linewidth=1.8
                    )
        ax.set_title(profile, loc="left")
        ax.set_xlabel("Latenz (ms)")
        ax.set_xscale("log")
        percent_axis(ax)
    handles = [
        Line2D([], [], color=SEVERITY[s], linewidth=2, label=SCENARIO_LABELS[s]) for s in NETWORK_SCENARIOS
    ]
    header_legend(fig, layout, handles, ncol=4, top_in=top + 0.05)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_power(models: Sequence[NoiseModel], path: Path) -> None:
    if not models:
        return
    fig, axes, layout = make_figure(1, 2, 3.6, 2.6, header_in=1.3)
    top = header(
        fig,
        layout,
        "Kleinster nachweisbarer Effekt (gepaart, α = 5 %, Power 80 %)",
        "Je mehr Seeds, desto kleiner der Effekt, der sicher erkannt wird; die Rauschmodelle stammen aus den "
        "A/A-Vergleichen und dem Vergleich Referenz gegen Netz.",
    )
    ns = np.arange(20, 301, 5)
    colors = [REFERENCE_GRAY, CATEGORICAL[0], CATEGORICAL[1]]
    for model, color in zip(models, colors, strict=False):
        axes[0][0].plot(
            ns, [mde_paired_binary(int(n), model.discordance) for n in ns], color=color, linewidth=2
        )
        axes[0][1].plot(ns, [mde_paired_mean(int(n), model.reward_sd) for n in ns], color=color, linewidth=2)
    for ax, title in zip(axes[0], ("Erfolgsquote", "Ø Max-Reward"), strict=True):
        for n in SEED_COUNTS:
            ax.axvline(n, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Seeds je Bedingung")
    percent_axis(axes[0][0])
    handles = [
        Line2D([], [], color=c, linewidth=2, label=m.name) for m, c in zip(models, colors, strict=False)
    ]
    header_legend(fig, layout, handles, ncol=3, top_in=top + 0.05)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_smolvla(tasks: Sequence[TaskData], path: Path) -> None:
    if not tasks:
        return
    fig, axes, layout = make_figure(1, len(tasks), 3.3, 2.7, header_in=1.35, sharey=True)
    top = header(
        fig,
        layout,
        "SmolVLA auf Meta-World: Erfolgsquote je Netz-Szenario und GPU-Profil",
        "Linie: synchrone Referenz (25 Aktionen pro Plan). Fehlerbalken: 95-%-Wilson-Intervall.",
    )
    colors = {
        "schnell": SEVERITY["buero_wlan"],
        "mittel": SEVERITY["schlecht"],
        "langsam": SEVERITY["sehr_schlecht"],
    }
    for ax, data in zip(axes[0], tasks, strict=False):
        sync = data.runs.get(SYNC_MATCHED)
        if sync is not None:
            ax.axhline(pooled("success", sync.ordered()), color=REFERENCE_GRAY, linewidth=1.5, linestyle="--")
        for offset, timing in zip((-0.2, 0.0, 0.2), GPU_TIMINGS, strict=True):
            for x, scenario in enumerate(NETWORK_SCENARIOS):
                run = data.runs.get(f"{timing}__{scenario}")
                if run is None:
                    continue
                iv = estimate("success", run.ordered())
                ax.plot([x + offset] * 2, [iv.low, iv.high], color=colors[timing], linewidth=1.5)
                ax.plot(
                    x + offset, iv.value, "o", color=colors[timing], markersize=5, markeredgecolor="white"
                )
        ax.set_xticks(range(len(NETWORK_SCENARIOS)))
        ax.set_xticklabels([SCENARIO_LABELS[s].replace(" ", "\n") for s in NETWORK_SCENARIOS])
        ax.set_title(data.task, loc="left")
        ax.set_ylim(-0.03, 1.03)
        percent_axis(ax)
    handles = [
        Line2D([], [], color=colors[t], marker="o", linewidth=1.5, label=GPU_LABELS[t]) for t in GPU_TIMINGS
    ]
    handles.append(Line2D([], [], color=REFERENCE_GRAY, linestyle="--", linewidth=1.5, label="Sync-Referenz"))
    header_legend(fig, layout, handles, ncol=4, top_in=top + 0.05)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_latency_effect(points: Sequence[EffectPoint], path: Path) -> None:
    if not points:
        return
    families = list(dict.fromkeys(p.family for p in points))
    fig, axes, layout = make_figure(1, 2, 3.6, 3.0, header_in=1.45, sharey=True)
    top = header(
        fig,
        layout,
        "Schadet Latenz SmolVLA mehr als PushT?",
        "Erfolgsquote relativ zur passenden synchronen Referenz gegen die mediane Ende-zu-Ende-Latenz, links in "
        "ms, rechts in Planperioden (PushT: 8 Ticks bei 10 Hz = 800 ms, SmolVLA: 25 Ticks bei 80 Hz = 313 ms). "
        "Jeder Punkt ist eine Bedingung (Netz-Szenario × Brain-Rechenzeit).",
    )
    palette = {
        "PushT kalibriert": REFERENCE_GRAY,
        "PushT ddim10": MUTED,
        "drawer-close-v3": CATEGORICAL[2],
        "push-v3": CATEGORICAL[0],
        "coffee-push-v3": CATEGORICAL[1],
    }
    markers = {"PushT kalibriert": "s", "PushT ddim10": "D"}
    for ax, attribute, xlabel in zip(
        axes[0],
        ("latency_ms", "latency_periods"),
        ("Mediane Ende-zu-Ende-Latenz (ms)", "Latenz in Planperioden"),
        strict=True,
    ):
        for family in families:
            chosen = sorted((p for p in points if p.family == family), key=lambda p: p.latency_ms)
            ax.plot(
                [getattr(p, attribute) for p in chosen],
                [p.relative_success for p in chosen],
                linestyle="none",
                marker=markers.get(family, "o"),
                color=palette.get(family, INK_2),
                markersize=6,
                markeredgecolor="white",
                label=family,
            )
        ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        percent_axis(ax)
    axes[0][0].set_ylabel("Erfolg relativ zur Sync-Referenz")
    handles = [
        Line2D([], [], linestyle="none", marker=markers.get(f, "o"), color=palette.get(f, INK_2), label=f)
        for f in families
    ]
    header_legend(fig, layout, handles, ncol=min(5, len(handles)), top_in=top + 0.05)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- report
def expert_delay_table(results: Path) -> list[str]:
    path = results / "phase3a" / "expert_delay.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    delays: list[int] = data["delays_ticks"]
    fps = float(data["fps"])
    head = " | ".join(f"{1000 * d / fps:.0f} ms" for d in delays)
    lines = [f"| Aufgabe | {head} |", "|---|" + "---:|" * len(delays)]
    rows = sorted(data["success"].items(), key=lambda item: -sum(item[1].values()))
    for task, values in rows:
        cells = " | ".join(fmt_pct(values[str(d)]) for d in delays)
        name = f"**{task}**" if task in SMOLVLA_TASKS else task
        lines.append(f"| {name} | {cells} |")
    return lines


def write_report(results: Path, reports: Path) -> Path:
    apply_style()
    figures = reports  # next to the markdown, like the phase-1 and phase-2 figures
    figures.mkdir(parents=True, exist_ok=True)
    cases = load_validation(results)
    verdicts = {source: verdict(cases, source) for source, _ in SOURCES}
    noise, models = noise_rows(results)
    tasks = load_smolvla(results)
    points = latency_effects(results, tasks)
    figure_validation(cases, figures / "phase3a_validation.png")
    figure_latency_cdf(cases, figures / "phase3a_latency.png")
    figure_power(models, figures / "phase3a_power.png")
    figure_smolvla(tasks, figures / "phase3a_smolvla.png")
    figure_latency_effect(points, figures / "phase3a_latency_effect.png")
    replay_lines, replay_total = replay_table(cases)

    lines: list[str] = [
        "# Phase 3a: Simulationszeit-Modus und SmolVLA-Baseline",
        "",
        f"_Erzeugt am {datetime.now(UTC).strftime('%d.%m.%Y %H:%M')} UTC mit `python -m kortyn.report_phase3a`._",
        "",
        "## Validierung: Simulationszeit gegen Echtzeit",
        "",
        "Gepaarte Vergleiche je Profil, Szenario und Metrik (Erfolg, Ø Max-Reward, Planalter, Starvation): Zahl "
        "der 95-%-Intervalle der Differenz, die die Null enthalten, und der nach Holm signifikanten Differenzen.",
        "",
        "| Quelle | Vergleiche | KI der Differenz enthält 0 | Wert im KI der Echtzeitmessung | Holm-signifikant |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {name} | {verdicts[source].comparisons} | {verdicts[source].inside} | "
            f"{verdicts[source].in_real_interval} | {verdicts[source].holm_significant} |"
            for source, name in SOURCES
        ],
        "",
    ]
    for source, name in SOURCES:
        if verdicts[source].outside:
            lines += [f"Intervalle ohne die Null ({name}): " + "; ".join(verdicts[source].outside) + ".", ""]
    lines += [
        f"Replay-Treue: von {replay_total.episodes} Phase-1-Episoden sind {replay_total.identical} bitgleich "
        f"reproduziert, {replay_total.same_outcome} gleich bis auf die letzten Float-Bits der Abdeckung, "
        f"{replay_total.diverged} abweichend; {replay_total.fallbacks} Anfragen ohne Log-Gegenstück.",
        "",
        *replay_lines,
        "",
        "![Validierung](phase3a_validation.png)",
        "",
    ]
    for profile in PROFILES:
        if not any(c.profile == profile for c in cases):
            continue
        lines += [
            f"### Profil {profile}",
            "",
            *validation_table(cases, profile),
            "",
            *delta_table(cases, profile),
            "",
        ]
        drift = service_drift(results, profile)
        if drift:
            lines += [
                "Servicezeit der Echtzeit-Maschine in Phase 1 (Serverzeit je Anfrage in ms: Median des Laufs, "
                "Mittel je Block aus zehn Seeds):",
                "",
                *drift,
                "",
            ]
    lines += ["### Latenzen", "", *latency_table(cases), "", "![Latenz](phase3a_latency.png)", ""]
    determinism = determinism_rows(results)
    if determinism:
        lines += [
            "### Determinismus",
            "",
            "Wiederholte Ausführungen derselben Läufe (andere Worker-Zahl, andere Reihenfolge der Seeds, frische "
            "Prozesse). Episoden: bitgleich = Erfolg, Länge, alle Takt-Serien, Abdeckung und Reward identisch; "
            "„letzte Float-Bits“ = alles identisch bis auf |Δ Abdeckung| < 1e-9. Anfragen: alle Felder.",
            "",
            *determinism_table(determinism),
            "",
        ]
    lines += [
        "## Restrauschen und Power",
        "",
        *noise_table(noise),
        "",
        *power_table(models),
        "",
        "![Power](phase3a_power.png)",
        "",
    ]
    delay = expert_delay_table(results)
    if delay:
        lines += ["## Verzögerungstoleranz der Meta-World-Experten", "", *delay, ""]
    if tasks:
        lines += ["## SmolVLA", "", *published_comparison(tasks), ""]
        for data in tasks:
            lines += [f"### {data.task}", "", *smolvla_table(data), ""]
        lines += ["![SmolVLA](phase3a_smolvla.png)", ""]
        lines += ["### Erfolgsverlust gegen die Sync-Referenz je Aufgabe", "", *task_comparison(tasks), ""]
        power = realized_power(tasks)
        if power:
            lines += [
                "### Realisierte Power (Rauschen der SmolVLA-Vergleiche)",
                "",
                "Je Aufgabe über alle Echtzeit-Bedingungen gemittelt: Anteil der Seeds mit verschiedenem Ausgang "
                "gegenüber Sync-25 und Streuung der Reward-Differenzen; daraus der kleinste Effekt, den die "
                "gemessene Seed-Zahl mit 80 % Power (α = 5 %, zweiseitig) erkennt.",
                "",
                *power,
                "",
            ]
        transport = transport_table(tasks)
        if transport:
            lines += ["### Bildübertragung", "", *transport]
    if points:
        lines += [
            "## Latenzwirkung: SmolVLA gegen PushT",
            "",
            "![Latenzwirkung](phase3a_latency_effect.png)",
            "",
        ]
        for family in dict.fromkeys(p.family for p in points):
            slope = latency_slope(points, family, max_latency_ms=1000.0)
            if slope is not None:
                lines.append(f"- {family}: {100 * slope:+.1f} pp relativer Erfolg pro 100 ms (Latenz ≤ 1 s)")
        lines.append("")
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "phase3a_data.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-3a report data (tables and figures).")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    path = write_report(Path(args.results_dir), Path(args.reports_dir))
    LOG.info("written %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
