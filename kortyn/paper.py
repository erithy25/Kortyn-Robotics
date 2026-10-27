"""Numbers, tables and figures of the technical report ``paper/report.md``, computed from repository files.

    python -m kortyn.paper            # paper/figures/*.png, paper/provenance.md, the report's tables
    python -m kortyn.paper --check    # exit 1 if provenance.md or a generated table in report.md is outdated

**Provenance.** Every quantity the report states is a :class:`Fact`: the text exactly as printed in the
report, the file it comes from and how it is derived. ``paper/provenance.md`` lists them all.
``tests/test_paper.py`` checks that the report quotes every fact verbatim, that every number with a unit in
the report's prose is a fact, and that the generated tables in the report equal what this module computes.
Tables sit in the report between ``<!-- table:NAME -->`` and ``<!-- /table:NAME -->``; this module rewrites
them.

Sources are the machine-readable results (``results/**``: episode logs, evaluation JSON files) where they
exist, and the generated data reports of earlier phases (``reports/*.md``) for the few statistics that only
exist there; each fact names its file.

**Figures** share one style (``STYLE``): the validated categorical palette of the data-visualization guide in
fixed slot order, thin marks, hairline grid, a legend for two or more series and no second axis.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from kortyn.metrics import wilson_interval
from kortyn.phase3_steps import load_episodes

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
REPORT = PAPER / "report.md"
PROVENANCE = PAPER / "provenance.md"

MINUS = "−"
TABLE_START = "<!-- table:{name} -->"
TABLE_END = "<!-- /table:{name} -->"


class PaperError(RuntimeError):
    """A source file is missing or does not hold what the report needs."""


# --------------------------------------------------------------------------------------------- formatting
def pct(value: float, digits: int = 0) -> str:
    """``0.473`` -> ``47%`` (English style, no space)."""
    return f"{100.0 * value:.{digits}f}%"


def signed(value: float, digits: int = 1) -> str:
    """``-0.016`` (as a fraction) is not used here: ``value`` is already in the printed unit."""
    text = f"{abs(value):.{digits}f}"
    if float(text) == 0.0:
        return f"±{text}"
    return f"+{text}" if value > 0 else f"{MINUS}{text}"


def pp(value: float, digits: int = 1) -> str:
    """A difference of proportions in percentage points: ``-0.016`` -> ``−1.6 pp``."""
    return f"{signed(100.0 * value, digits)} pp"


def pp_interval(low: float, high: float, digits: int = 1) -> str:
    return f"[{signed(100.0 * low, digits)}, {signed(100.0 * high, digits)}] pp"


def ms(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f} ms"


def p_value(value: float) -> str:
    return f"p = {value:.2f}" if value >= 0.01 else "p < 0.01"


# --------------------------------------------------------------------------------------------- facts
@dataclass(frozen=True)
class Fact:
    key: str
    text: str  # exactly as the report prints it
    source: str  # file in the repository
    detail: str  # field or computation


@dataclass
class Facts:
    items: dict[str, Fact] = field(default_factory=dict)

    def add(self, key: str, text: str, source: str | Path, detail: str) -> str:
        if key in self.items:
            raise PaperError(f"fact {key!r} defined twice")
        path = Path(source)
        relative = path.relative_to(ROOT) if path.is_absolute() else path
        self.items[key] = Fact(key, text, str(relative), detail)
        return text

    def texts(self) -> set[str]:
        return {fact.text for fact in self.items.values()}


def _json(path: Path) -> Any:
    if not path.exists():
        raise PaperError(f"{path.relative_to(ROOT)} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> Any:
    if not path.exists():
        raise PaperError(f"{path.relative_to(ROOT)} does not exist")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _rate(run: Path) -> tuple[int, int]:
    """(successes, episodes) of a run directory (final attempt per seed)."""
    episodes = load_episodes(run)
    return sum(e.success for e in episodes.values()), len(episodes)


def _grep(path: Path, pattern: str) -> re.Match[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text)
    if match is None:
        raise PaperError(f"{path.relative_to(ROOT)}: pattern {pattern!r} not found")
    return match


# --------------------------------------------------------------------------------------------- phase 1
PHASE1_SCENARIOS = (
    ("referenz_sync", "synchronous reference (simulation waits)"),
    ("gut", "LAN, 0 ms"),
    ("buero_wlan", "office Wi-Fi, RTT 20 ± 10 ms"),
    ("schlecht", "poor, RTT 80 ± 40 ms + 300-ms stalls"),
    ("sehr_schlecht", "very poor, RTT 150 ± 80 ms + 1-s stalls"),
    ("schlecht_ohne_stall", "poor without stalls"),
    ("sehr_schlecht_ohne_stall", "very poor without stalls"),
)


def phase1(facts: Facts) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scenario, label in PHASE1_SCENARIOS:
        row: dict[str, Any] = {"scenario": scenario, "label": label}
        for profile in ("kalibriert", "ddim10"):
            run = ROOT / "results" / profile / scenario
            if not run.exists():
                continue
            wins, n = _rate(run)
            low, high = wilson_interval(wins, n)
            row[profile] = (wins / n, low, high, n)
            source = run / "episodes.jsonl"
            facts.add(
                f"p1.{profile}.{scenario}", pct(wins / n), source, "success rate, final attempt per seed"
            )
        rows.append(row)
    return {"rows": rows}


def table_phase1(data: Mapping[str, Any]) -> str:
    lines = [
        "| Network scenario | DDIM-4 profile (inside the budget) | DDIM-10 profile (compute-bound) |",
        "|---|---:|---:|",
    ]
    for row in data["rows"]:
        cells = []
        for profile in ("kalibriert", "ddim10"):
            if profile in row:
                rate, low, high, _ = row[profile]
                cells.append(f"{pct(rate)} [{pct(low)}, {pct(high)}]")
            else:
                cells.append("–")
        lines.append(f"| {row['label']} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- phase 2
def phase2(facts: Facts) -> None:
    report = ROOT / "reports" / "phase2.md"
    pooled = _grep(
        report,
        r"Über alle 6 Szenarien gepoolt:\*\* Schwelle 6 ([+−-]?\d+) pp Erfolg \(p = ([\d.]+)\).*?"
        r"Schwelle 6 · Inpainting ([+−-]?\d+) pp Erfolg \(p = ([\d.]+)\)",
    )
    facts.add(
        "p2.threshold.pooled",
        f"{signed(float(pooled.group(1).replace('−', '-')), 0)} pp",
        report,
        "pooled Δ success, threshold 6",
    )
    facts.add("p2.threshold.p", f"p = {float(pooled.group(2)):.2f}", report, "pooled McNemar p, threshold 6")
    facts.add(
        "p2.inpaint.pooled",
        f"{signed(float(pooled.group(3).replace('−', '-')), 0)} pp",
        report,
        "pooled Δ success, inpainting",
    )
    facts.add("p2.inpaint.p", f"p = {float(pooled.group(4)):.2f}", report, "pooled McNemar p, inpainting")
    noise = _grep(report, r"unterscheiden sich bereits um Δ Reward ([−-][\d.]+) und ([−-]\d+) pp Erfolg")
    facts.add(
        "p2.noise.success",
        f"{signed(float(noise.group(2).replace('−', '-')), 0)} pp",
        report,
        "A/A noise, identical phase-1 runs",
    )


# --------------------------------------------------------------------------------------------- phase 3a
def phase3a(facts: Facts) -> dict[str, Any]:
    data_report = ROOT / "reports" / "phase3a_data.md"
    replay = _grep(
        data_report, r"von (\d+) Phase-1-Episoden sind (\d+) bitgleich reproduziert, (\d+) gleich bis auf"
    )
    facts.add("p3a.replay.episodes", f"{int(replay.group(1))} episodes", data_report, "replay fidelity line")
    facts.add(
        "p3a.replay.identical", f"{int(replay.group(2))} bit-identical", data_report, "replay fidelity line"
    )
    report = ROOT / "reports" / "phase3a.md"
    episode_model = _grep(report, r"Keine der (\d+) gepaarten Differenzen ist nach Holm")
    pooled_model = _grep(report, r"(\d+) von (\d+) Differenzen sind")
    facts.add(
        "p3a.episode_model", f"0 of {int(episode_model.group(1))}", report, "episode-level service model"
    )
    facts.add(
        "p3a.pooled_model",
        f"{int(pooled_model.group(1))} of {int(pooled_model.group(2))}",
        report,
        "pooled profile",
    )
    noise = _grep(report, r"SD der Reward-Differenz „gut“ gegen „Büro-WLAN“: ([\d,]+) statt ([\d,]+)")
    facts.add(
        "p3a.sd.sim",
        noise.group(1).replace(",", "."),
        report,
        "SD of the paired reward difference, simulation time",
    )
    facts.add("p3a.sd.real", noise.group(2).replace(",", "."), report, "same, real time")
    # simulation time against real time, per condition (figure 2)
    points: list[dict[str, Any]] = []
    for profile, scenarios in (
        (
            "kalibriert",
            (
                "referenz_sync",
                "gut",
                "buero_wlan",
                "schlecht",
                "sehr_schlecht",
                "schlecht_ohne_stall",
                "sehr_schlecht_ohne_stall",
            ),
        ),
        ("ddim10", ("referenz_sync", "gut", "buero_wlan", "schlecht", "sehr_schlecht")),
    ):
        for scenario in scenarios:
            real = ROOT / "results" / profile / scenario
            simulated = ROOT / "results" / "phase3a" / "validation_model" / profile / f"modell__{scenario}"
            if not (real.exists() and simulated.exists()):
                raise PaperError(f"missing validation pair {profile}/{scenario}")
            rw, rn = _rate(real)
            sw, sn = _rate(simulated)
            points.append({"profile": profile, "scenario": scenario, "real": (rw, rn), "sim": (sw, sn)})
    return {"points": points}


# --------------------------------------------------------------------------------------------- phase 3
def phase3(facts: Facts) -> dict[str, Any]:
    base = ROOT / "results" / "phase3" / "libero" / "smolvla_libero"
    levels: dict[int, tuple[int, int]] = {}
    for latency in (0, 500, 1000, 1250, 1500, 2000):
        run = base / f"fest_{latency}ms"
        levels[latency] = _rate(run)
        facts.add(
            f"p3.level.{latency}",
            pct(levels[latency][0] / levels[latency][1]),
            run / "episodes.jsonl",
            "success rate",
        )
    sync = base / "sync25"
    sync_rate = _rate(sync)
    facts.add(
        "p3.sync25",
        pct(sync_rate[0] / sync_rate[1]),
        sync / "episodes.jsonl",
        "success rate, synchronous 25 actions/plan",
    )
    r1 = _json(ROOT / "results" / "phase3" / "step1" / "r1.json")
    r1_path = ROOT / "results" / "phase3" / "step1" / "r1.json"
    facts.add("p3.r1.kortyn", pct(r1["rate_a"]), r1_path, "rate_a (Kortyn sync50)")
    facts.add("p3.r1.lerobot", pct(r1["rate_b"]), r1_path, "rate_b (lerobot-eval)")
    facts.add("p3.r1.p", p_value(r1["p_value"]), r1_path, "p_value (McNemar exact)")
    capacity_path = ROOT / "results" / "phase3" / "fleet" / "capacity.json"
    capacity = _json(capacity_path)
    capacities = {int(c["capacity"]) for c in capacity["configurations"]}
    if capacities != {1}:
        raise PaperError(f"phase-3 capacity is not 1 everywhere: {sorted(capacities)}")
    facts.add(
        "p3.capacity.configurations",
        f"{len(capacity['configurations'])} configurations",
        capacity_path,
        "every configuration has capacity 1",
    )
    validation_path = ROOT / "results" / "phase3" / "validation" / "evaluation.json"
    validation = _json(validation_path)
    shortfalls = [-case["difference"] for case in validation if case["predicted"] > 0]
    facts.add(
        "p3.validation.shortfall",
        f"{100 * min(shortfalls):.0f}–{100 * max(shortfalls):.0f} pp",
        validation_path,
        "−difference of the cases with a positive prediction",
    )
    return {"levels": levels, "sync25": sync_rate, "validation": validation}


# --------------------------------------------------------------------------------------------- phase 4
def phase4(facts: Facts) -> dict[str, Any]:
    folder = ROOT / "results" / "phase4"
    curves_path = folder / "curves.json"
    curves = _json(curves_path)
    without = curves["conditions"]["ohne_rtc"]
    with_rtc = curves["conditions"]["mit_rtc"]
    for name, condition in (("without", without), ("rtc", with_rtc)):
        for level in condition["levels"]:
            latency = int(level["latency_ms"])
            facts.add(
                f"p4.{name}.{latency}",
                pct(level["pooled"]),
                curves_path,
                f"conditions.{'ohne_rtc' if name == 'without' else 'mit_rtc'}.levels[{latency} ms].pooled",
            )
        lstar = condition["lstar"]
        facts.add(f"p4.{name}.lstar", ms(lstar["point_ms"]), curves_path, "lstar.point_ms")
    by_level = {int(level["latency_ms"]): level["pooled"] for level in without["levels"]}
    losses = [
        by_level[int(level["latency_ms"])] - level["pooled"]
        for level in with_rtc["levels"]
        if level["latency_ms"] <= 200
    ]
    facts.add(
        "p4.rtc.small_latency_loss",
        f"{100 * min(losses):.0f}–{100 * max(losses):.0f} pp",
        curves_path,
        "conditions.ohne_rtc.levels − conditions.mit_rtc.levels at 0, 100, 200 ms",
    )
    r_half = (curves["r_rtc"]["interval"][1] - curves["r_rtc"]["interval"][0]) / 2
    facts.add(
        "p4.pair.halfwidth",
        f"±{100 * r_half:.0f} pp",
        curves_path,
        "half-width of r_rtc.interval (100 pairs)",
    )
    report4 = ROOT / "reports" / "phase4.md"
    stretch = _grep(report4, r"k = 1,5 entspräche \+(\d+) bis \+(\d+) pp")
    facts.add(
        "p4.h1.stretch",
        f"{stretch.group(1)}–{stretch.group(2)} pp",
        report4,
        "gain a 1.5-fold stretch of the latency axis would give",
    )
    wifi = _grep(report4, r"Ab (\d+) Mbit/s Nutzrate \(gutes Wi-Fi 5/6\) begrenzt mit Codec die GPU")
    facts.add(
        "p4.wifi.limit",
        f"{wifi.group(1)} Mbit/s",
        report4,
        "shared Wi-Fi limits the fleet below this effective rate (with codec)",
    )
    h1 = curves["rtc_effect"]["h1"]
    facts.add("p4.h1.gain", pp(h1["gain"]), curves_path, "rtc_effect.h1.gain")
    facts.add("p4.h1.interval", pp_interval(*h1["interval"]), curves_path, "rtc_effect.h1.interval")
    r_rtc = curves["r_rtc"]
    facts.add("p4.rrtc.sync", pct(r_rtc["rate_a"]), curves_path, "r_rtc.rate_a (sync25, phase 3)")
    facts.add("p4.rrtc.rtc", pct(r_rtc["rate_b"]), curves_path, "r_rtc.rate_b (RTC at 0 ms)")
    facts.add("p4.rrtc.diff", pp(r_rtc["difference"], 0), curves_path, "r_rtc.difference")
    facts.add(
        "p4.rrtc.interval",
        pp_interval(r_rtc["interval"][0], r_rtc["interval"][1], digits=0),
        curves_path,
        "r_rtc.interval",
    )
    facts.add("p4.rrtc.p", p_value(r_rtc["p_value"]), curves_path, "r_rtc.p_value")
    # capacity (RTX 3060 Ti assumption, realistic batching, office Wi-Fi)
    capacity_path = folder / "fleet" / "capacity.json"
    capacity = _json(capacity_path)
    table: dict[tuple[str, str], tuple[int, int]] = {}
    sizes: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for configuration in capacity["configurations"]:
        if configuration["batching"] != "realistisch" or configuration["network"] != "buero_wlan":
            continue
        rtc = "rtc" if configuration["gpu"].endswith("_rtc") else "without"
        key = (rtc, configuration["strategy"])
        table[key] = (int(configuration["capacity"]), int(configuration["capacity_conservative"]))
        sizes[key] = configuration["sizes"]
    best_without = max(c for (r, _), (c, _) in table.items() if r == "without")
    best_rtc = max(c for (r, _), (c, _) in table.items() if r == "rtc")
    facts.add(
        "p4.capacity.without.best",
        f"{best_without} robots",
        capacity_path,
        "max capacity without RTC, realistic batching",
    )
    facts.add(
        "p4.capacity.rtc.best",
        f"{best_rtc} robots",
        capacity_path,
        "max capacity with RTC, realistic batching",
    )
    rtc_overhead_path = folder / "rtc_overhead.json"
    overhead = _json(rtc_overhead_path)
    facts.add(
        "p4.rtc.overhead", pct(overhead["relative_overhead"], 1), rtc_overhead_path, "relative_overhead"
    )
    # codec
    codec_path = folder / "codec.json"
    codec = _json(codec_path)
    candidates = {c["label"]: c for c in codec["candidates"]}
    raw, standard = candidates["raw"], candidates[codec["standard"]]
    facts.add(
        "p4.codec.raw.kb", f"{raw['mean_bytes'] / 1000:.0f} kB", codec_path, "candidates[raw].mean_bytes"
    )
    facts.add(
        "p4.codec.zstd.kb",
        f"{standard['mean_bytes'] / 1000:.0f} kB",
        codec_path,
        f"candidates[{codec['standard']}].mean_bytes",
    )
    facts.add(
        "p4.codec.raw.mbit",
        f"{raw['mbit_per_robot']:.1f} Mbit/s",
        codec_path,
        "candidates[raw].mbit_per_robot",
    )
    facts.add(
        "p4.codec.zstd.mbit",
        f"{standard['mbit_per_robot']:.1f} Mbit/s",
        codec_path,
        f"candidates[{codec['standard']}].mbit_per_robot",
    )
    facts.add(
        "p4.codec.identical",
        f"{standard['identical_model_inputs']}/{standard['requests']} requests",
        codec_path,
        "identical model inputs",
    )
    # validation (first run with the render fault, corrected run)
    faulty_path = folder / "validation_renderfehler" / "evaluation.json"
    valid_path = folder / "validation" / "evaluation.json"
    faulty = _json(faulty_path)["cases"]["d_n5"]
    facts.add("p4.val.faulty.measured", pct(faulty["measured"]), faulty_path, "cases.d_n5.measured")
    facts.add("p4.val.faulty.predicted", pct(faulty["predicted"]), faulty_path, "cases.d_n5.predicted")
    corrected = _json(valid_path)["cases"]
    for case in ("d_n5", "d_n10"):
        entry = corrected[case]
        facts.add(f"p4.val.{case}.measured", pct(entry["measured"], 1), valid_path, f"cases.{case}.measured")
        facts.add(
            f"p4.val.{case}.predicted", pct(entry["predicted"], 1), valid_path, f"cases.{case}.predicted"
        )
        facts.add(f"p4.val.{case}.diff", pp(entry["difference"]), valid_path, f"cases.{case}.difference")
        facts.add(
            f"p4.val.{case}.interval", pp_interval(*entry["interval"]), valid_path, f"cases.{case}.interval"
        )
    facts.add(
        "p4.val.d_n10.episodes",
        f"{corrected['d_n10']['episodes']} episodes",
        valid_path,
        "cases.d_n10.episodes",
    )
    # fresh-process check
    fresh_path = folder / "fresh.json"
    fresh = _json(fresh_path)
    facts.add("p4.fresh.rate", pct(fresh["success"]), fresh_path, "success (task 8, 500 ms, fresh processes)")
    facts.add(
        "p4.fresh.phase3",
        pct(fresh["comparisons"]["phase3"]["rate_a"]),
        fresh_path,
        "comparisons.phase3.rate_a",
    )
    facts.add(
        "p4.fresh.block_a",
        pct(fresh["comparisons"]["block_a"]["rate_a"]),
        fresh_path,
        "comparisons.block_a.rate_a",
    )
    facts.add(
        "p4.fresh.p",
        p_value(fresh["comparisons"]["phase3"]["p_value"]),
        fresh_path,
        "comparisons.phase3.p_value",
    )
    facts.add("p4.fresh.episodes", f"{fresh['episodes']} episodes", fresh_path, "episodes")
    retest_path = folder / "retest.json"
    retest = _json(retest_path)
    facts.add(
        "p4.retest.bias",
        pp(retest["pooled"]["own"]["bias"]),
        retest_path,
        "pooled.own.bias (phase-3 validation vs dense curve)",
    )
    facts.add(
        "p4.val.d_n5.episodes", f"{corrected['d_n5']['episodes']} episodes", valid_path, "cases.d_n5.episodes"
    )
    pooled = _grep(report4, r"Gepoolt \(nach Sichtung\) ([−-][\d,]+) pp \[([−-][\d,]+); ([−+-]?[\d,]+)\]")

    def number(text: str) -> float:
        return float(text.replace("−", "-").replace(",", "."))

    facts.add(
        "p4.val.pooled", pp(number(pooled.group(1)) / 100), report4, "pooled validation difference (post hoc)"
    )
    facts.add(
        "p4.val.pooled.interval",
        pp_interval(number(pooled.group(2)) / 100, number(pooled.group(3)) / 100),
        report4,
        "pooled validation interval (post hoc)",
    )
    # economics
    economics_path = folder / "economics.json"
    economics = _json(economics_path)
    devices = {(d["label"], d["condition"]): d for d in economics["devices"]}
    for label, short in (
        ("Jetson Orin Nano Super 8 GB (Dev Kit)", "nano"),
        ("Jetson AGX Orin 64 GB (Dev Kit)", "agx"),
        ("Jetson AGX Thor (Dev Kit)", "thor"),
    ):
        device = devices[(label, "ohne_rtc")]
        facts.add(
            f"p4.econ.{short}.eur",
            f"€{device['monthly_eur']:.2f}",
            economics_path,
            f"devices[{label}, ohne_rtc].monthly_eur",
        )
        facts.add(
            f"p4.econ.{short}.success",
            pct(device["success"]),
            economics_path,
            f"devices[{label}, ohne_rtc].success",
        )
    decision = economics["decisions"]["realistisch"]["mit_rtc"]["brain"]
    facts.add(
        "p4.econ.brain.eur",
        f"€{decision['monthly_eur']:.2f}",
        economics_path,
        "decisions.realistisch.mit_rtc.brain.monthly_eur",
    )
    facts.add(
        "p4.econ.brain.robots",
        f"{decision['robots']} robots",
        economics_path,
        "decisions.realistisch.mit_rtc.brain.robots",
    )
    facts.add(
        "p4.econ.brain.success",
        pct(decision["worst"], 1),
        economics_path,
        "decisions.realistisch.mit_rtc.brain.worst",
    )
    return {
        "curves": curves,
        "capacity_table": table,
        "capacity_sizes": sizes,
        "faulty": faulty,
        "corrected": corrected,
        "economics": economics,
    }


def table_capacity(data: Mapping[str, Any]) -> str:
    table: Mapping[tuple[str, str], tuple[int, int]] = data["capacity_table"]
    names = {
        "a": "a: FIFO",
        "b": "b: FIFO + batching",
        "d": "d: EDF + batching",
        "e": "e: EDF + batching + admission",
    }
    lines = ["| Scheduling strategy | Without RTC | With RTC (H ≤ 25) |", "|---|---:|---:|"]
    for strategy in ("a", "b", "d", "e"):
        cells = []
        for rtc in ("without", "rtc"):
            point, conservative = table[(rtc, strategy)]
            cells.append(f"{point} ({conservative})")
        lines.append(f"| {names[strategy]} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def table_dose(p3: Mapping[str, Any], p4: Mapping[str, Any], p5: Mapping[str, Any] | None) -> str:
    curves = p4["curves"]["conditions"]
    columns: list[tuple[str, dict[int, float]]] = [
        ("Phase 3, no RTC", {k: v[0] / v[1] for k, v in p3["levels"].items()}),
        (
            "Phase 4, no RTC",
            {int(level["latency_ms"]): level["pooled"] for level in curves["ohne_rtc"]["levels"]},
        ),
        (
            "Phase 4, RTC H ≤ 25",
            {int(level["latency_ms"]): level["pooled"] for level in curves["mit_rtc"]["levels"]},
        ),
    ]
    if p5 is not None and p5.get("curves") is not None:
        columns.append(
            (
                "Phase 5, RTC H = D + 4",
                {
                    int(level["latency_ms"]): level["pooled"]
                    for level in p5["curves"]["conditions"]["rtc_kurz"]["levels"]
                },
            )
        )
    latencies = sorted({latency for _, values in columns for latency in values})
    header = "| Latency | " + " | ".join(name for name, _ in columns) + " |"
    lines = [header, "|---:|" + "---:|" * len(columns)]
    for latency in latencies:
        cells = [pct(values[latency]) if latency in values else "–" for _, values in columns]
        lines.append(f"| {ms(latency)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _validation_row(label: str, robots: int, entry: Mapping[str, Any], status: str) -> str:
    difference = f"{pp(entry['difference'])} {pp_interval(entry['interval'][0], entry['interval'][1])}"
    return (
        f"| {label} | {robots} | {entry['episodes']} | {pct(entry['measured'], 1)} | "
        f"{pct(entry['predicted'], 1)} | {difference} | {status} |"
    )


def table_validation(p3: Mapping[str, Any], p4: Mapping[str, Any]) -> str:
    lines = [
        "| Run | Robots | Episodes | Measured | Predicted | Measured − predicted [CI] | Status |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    invalid = "invalid (render fault)"
    for case in p3["validation"]:
        lines.append(_validation_row(f"Phase 3, strategy {case['case'][0]}", case["robots"], case, invalid))
    lines.append(_validation_row("Phase 4, first run", 5, p4["faulty"], invalid))
    for case, robots in (("d_n5", 5), ("d_n10", 10)):
        entry = p4["corrected"][case]
        lines.append(
            _validation_row("Phase 4, corrected", robots, entry, "passes" if entry["passes"] else "fails")
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- phase 5
def phase5(facts: Facts) -> dict[str, Any]:
    folder = ROOT / "results" / "phase5"
    power_path = folder / "power.json"
    power = _json(power_path)
    cells = {
        (row["pairs"], round(row["p_test"], 2), row["rho_total"]): row["pass_rate"]
        for row in power["scenarios"]
    }
    for pairs in (100, 200):
        equal = [cells[(pairs, 0.53, rho)] for rho in (0.3, 0.5, 0.7)]
        facts.add(
            f"p5.power.{pairs}.equal",
            f"{pct(min(equal))}–{pct(max(equal))}",
            power_path,
            f"pass_rate, {pairs} pairs, p_test = p_reference",
        )
    false_pass = [cells[(200, 0.38, rho)] for rho in (0.3, 0.5, 0.7)]
    facts.add(
        "p5.power.200.false",
        pct(max(false_pass), 1),
        power_path,
        "max pass_rate, 200 pairs, p_test = p_reference − 15 pp",
    )
    gap = [cells[(200, 0.47, rho)] for rho in (0.3, 0.5, 0.7)]
    facts.add(
        "p5.power.200.gap",
        f"{pct(min(gap))}–{pct(max(gap))}",
        power_path,
        "pass_rate, 200 pairs, p_test = p_reference − 6 pp",
    )
    facts.add("p5.power.design.100", "100 pairs", power_path, "scenarios with replicates = 1")
    facts.add("p5.power.design.200", "200 pairs", power_path, "scenarios with replicates = 2")
    facts.add("p5.power.gap.size", f"{round(100 * power['async_gap'])} pp", power_path, "async_gap")
    phase4_config = ROOT / "kortyn" / "experiments" / "phase4" / "libero_rtc.yaml"
    long_guidance = 1 + _yaml(phase4_config)["strategies"]["rtc"]["smoothing"]["max_overlap_steps"]
    facts.add(
        "c.p4.guidance",
        f"{long_guidance} actions",
        phase4_config,
        "1 frozen action at d = 1 + strategies.rtc.smoothing.max_overlap_steps",
    )
    gate_config = ROOT / "kortyn" / "experiments" / "phase5" / "libero_rtc_kurz.yaml"
    settings = _yaml(gate_config)
    facts.add(
        "c.p5.gate.episodes",
        f"{settings['experiment']['episodes']} episodes",
        gate_config,
        "experiment.episodes (per condition)",
    )
    facts.add(
        "c.p5.overlap",
        f"{settings['strategies']['rtc_kurz']['smoothing']['max_overlap_steps']} actions",
        gate_config,
        "strategies.rtc_kurz.smoothing.max_overlap_steps",
    )
    dose_config = ROOT / "kortyn" / "experiments" / "phase5" / "libero_rtc_kurz_dose.yaml"
    facts.add(
        "c.p5.dose.episodes",
        f"{_yaml(dose_config)['experiment']['episodes']} episodes",
        dose_config,
        "experiment.episodes (per level)",
    )
    data: dict[str, Any] = {"power": power, "gate": None, "curves": None}
    gate_path = folder / "gate.json"
    if gate_path.exists():
        gate = _json(gate_path)
        data["gate"] = gate
        g = gate["gate"]
        facts.add("p5.gate.sync", pct(g["reference_rate"], 1), gate_path, "gate.reference_rate (sync25)")
        facts.add("p5.gate.rtc", pct(g["test_rate"], 1), gate_path, "gate.test_rate (rtc_kurz_0ms)")
        facts.add("p5.gate.diff", pp(g["difference"]), gate_path, "gate.difference")
        facts.add(
            "p5.gate.interval", pp_interval(*g["interval"]), gate_path, "gate.interval (cluster bootstrap)"
        )
        facts.add("p5.gate.p", p_value(g["p_value"]), gate_path, "gate.p_value (McNemar exact)")
        facts.add(
            "p5.gate.episodes",
            f"{gate['rates']['sync25']['episodes']} episodes",
            gate_path,
            "rates.sync25.episodes",
        )
        async_rate = gate["rates"]["async_0ms"]["rate"]
        facts.add("p5.async.rate", pct(async_rate, 1), gate_path, "rates.async_0ms.rate")
        for name in ("K1_async_vs_sync25", "K2_rtc_vs_async"):
            control = gate["controls"][name]
            short = name.split("_")[0]
            facts.add(f"p5.{short}.diff", pp(control["difference"]), gate_path, f"controls.{name}.difference")
            facts.add(
                f"p5.{short}.interval",
                pp_interval(*control["interval"]),
                gate_path,
                f"controls.{name}.interval",
            )
            facts.add(f"p5.{short}.p", p_value(control["p_value"]), gate_path, f"controls.{name}.p_value")
    curves_path = folder / "curves.json"
    if curves_path.exists():
        curves = _json(curves_path)
        data["curves"] = curves
        h1 = curves["h1_kurz"]["h1"]
        facts.add("p5.h1.gain", pp(h1["gain"]), curves_path, "h1_kurz.h1.gain")
        facts.add("p5.h1.interval", pp_interval(*h1["interval"]), curves_path, "h1_kurz.h1.interval")
        for level in curves["conditions"]["rtc_kurz"]["levels"]:
            latency = int(level["latency_ms"])
            facts.add(
                f"p5.short.{latency}",
                pct(level["pooled"]),
                curves_path,
                f"conditions.rtc_kurz.levels[{latency} ms].pooled",
            )
        facts.add(
            "p5.short.lstar",
            ms(curves["conditions"]["rtc_kurz"]["lstar"]["point_ms"]),
            curves_path,
            "conditions.rtc_kurz.lstar.point_ms",
        )
    cpu_path = ROOT / "tests" / "data" / "notebook_cpu_test.json"
    cpu = _json(cpu_path)
    standard = next(c for c in cpu["configurations"] if c["name"] == "standard")
    first = next(row for row in standard["results"] if row["batch_size"] == 1)
    facts.add(
        "p5.cpu.batch1",
        f"{first['median_ms'] / 1000:.1f} s",
        cpu_path,
        "configurations[standard].results[batch 1].median_ms",
    )
    ratios = {
        c["name"]: c["numerics"]["ratio"]
        for c in cpu["configurations"]
        if c["numerics"].get("ratio") is not None
    }
    facts.add(
        "p5.cpu.numerics.standard",
        pct(ratios["standard"], 1),
        cpu_path,
        "configurations[standard].numerics.ratio",
    )
    facts.add("p5.cpu.numerics.fp16", pct(ratios["fp16"], 1), cpu_path, "configurations[fp16].numerics.ratio")
    decision_path = folder / "gpu_decision.json"
    if decision_path.exists():
        data["decision"] = _json(decision_path)
    return data


def table_gate(p5: Mapping[str, Any]) -> str:
    gate = p5["gate"]
    rows = [
        ("Gate: RTC short − sync25", gate["gate"]),
        ("K1: asynchronous − sync25", gate["controls"]["K1_async_vs_sync25"]),
        ("K2: RTC short − asynchronous", gate["controls"]["K2_rtc_vs_async"]),
    ]
    for task, entry in sorted(gate["per_task"].items(), key=lambda item: -int(item[0])):
        rows.append((f"Gate, task {task} only", entry["gate"]))
    lines = [
        "| Comparison (0 ms) | Pairs | Reference | Test | Difference [95% CI] | McNemar | Gate criterion |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for label, entry in rows:
        rates = f"{pct(entry['reference_rate'], 1)} | {pct(entry['test_rate'], 1)}"
        difference = f"{pp(entry['difference'])} {pp_interval(entry['interval'][0], entry['interval'][1])}"
        verdict = "passed" if entry["passed"] else "failed"
        lines.append(
            f"| {label} | {entry['pairs']} | {rates} | {difference} | "
            f"{p_value(entry['p_value'])} | {verdict} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- constants
def _setting(path: Path, *keys: str | int) -> Any:
    """A value of a YAML config file; the key must be in the file itself (no ``extends`` resolution)."""
    value: Any = _yaml(path)
    for key in keys:
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError) as error:
            dotted = ".".join(str(k) for k in keys)
            raise PaperError(f"{path.relative_to(ROOT)}: no setting {dotted}") from error
    return value


def _plain(value: float) -> str:
    """A setting as printed: 20.0 -> "20", 1.25 -> "1.25"."""
    return f"{value:g}"


def constants(facts: Facts) -> None:
    """Settings the report quotes, read from the config file or module that fixes them."""
    from kortyn import notebook_bench
    from kortyn.fleet import MISS_SHARE_LIMIT
    from kortyn.rtc_gate import GATE_MARGIN

    experiments = ROOT / "kortyn" / "experiments"
    pusht = experiments / "base.yaml"
    pusht_fps = _setting(pusht, "env", "fps")
    facts.add("c.pushT.fps", f"{_plain(pusht_fps)} Hz", pusht, "env.fps")
    pusht_budget = (_setting(pusht, "agent", "refill_threshold") + 1) / pusht_fps
    facts.add("c.pushT.budget", ms(1000 * pusht_budget), pusht, "(agent.refill_threshold + 1) / env.fps")
    first = _setting(pusht, "experiment", "seed_start")
    facts.add(
        "c.seeds.pusht",
        f"seeds {first}–{first + _setting(pusht, 'experiment', 'episodes') - 1}",
        pusht,
        "experiment.seed_start … seed_start + episodes − 1",
    )
    _, n = _rate(ROOT / "results" / "kalibriert" / "gut")
    facts.add(
        "c.p1.episodes",
        f"{n} episodes",
        ROOT / "results" / "kalibriert" / "gut" / "episodes.jsonl",
        "episodes per condition",
    )

    libero = experiments / "phase3" / "libero_base.yaml"
    fps = _setting(libero, "env", "fps")
    ticks = _setting(libero, "agent", "refill_threshold") + 1
    facts.add("c.fps", f"{_plain(fps)} Hz", libero, "env.fps")
    facts.add(
        "c.chunk",
        f"{_setting(libero, 'agent', 'actions_per_chunk')} actions",
        libero,
        "agent.actions_per_chunk",
    )
    facts.add("c.ticks", f"{ticks} ticks", libero, "agent.refill_threshold + 1 (a request every … ticks)")
    facts.add("c.budget", ms(1000 * ticks / fps), libero, "(agent.refill_threshold + 1) / env.fps")
    facts.add("c.period", f"{_plain(ticks / fps)} s", libero, "(agent.refill_threshold + 1) / env.fps")
    facts.add(
        "c.steps",
        f"{_setting(libero, 'profiles', 'smolvla_libero', 'num_inference_steps')} flow-matching steps",
        libero,
        "profiles.smolvla_libero.num_inference_steps",
    )
    main = experiments / "phase3" / "libero_main.yaml"
    episodes = _setting(main, "experiment", "episodes")
    first = _setting(libero, "experiment", "seed_start")
    facts.add("c.episodes", f"{episodes} episodes", main, "experiment.episodes")
    facts.add(
        "c.seeds.libero",
        f"seeds {first}–{first + episodes - 1}",
        main,
        "seed_start (libero_base.yaml) … seed_start + experiment.episodes − 1",
    )
    facts.add(
        "c.init",
        f"init states 0–{_setting(main, 'env', 'libero', 'episodes_per_task') - 1}",
        main,
        "env.libero.episodes_per_task",
    )
    sync = _setting(main, "experiment", "suites", 0, "runs", 0)
    if sync.get("label") != "sync25":
        raise PaperError(f"{main.relative_to(ROOT)}: the first run is not sync25")
    facts.add(
        "c.sync25", f"{sync['sync_actions_per_chunk']} actions", main, "runs[sync25].sync_actions_per_chunk"
    )

    fleet = experiments / "phase4" / "fleet.yaml"
    facts.add(
        "c.s1",
        ms(_setting(fleet, "gpus", "rtx3060ti_schaetzung", "s1_ms")),
        fleet,
        "gpus.rtx3060ti_schaetzung.s1_ms (assumption)",
    )
    facts.add(
        "c.beta", pct(_setting(fleet, "batching", "realistisch", "beta")), fleet, "batching.realistisch.beta"
    )
    facts.add(
        "c.wait", ms(_setting(fleet, "strategies", "b", "max_wait_ms")), fleet, "strategies.b.max_wait_ms"
    )
    facts.add(
        "c.validation.confidence",
        pct(_setting(fleet, "validation", "confidence"), 1),
        fleet,
        "validation.confidence",
    )
    wifi = experiments / "buero_wlan.yaml"
    facts.add(
        "c.wifi",
        f"{_setting(wifi, 'network', 'rtt_ms')} ± {_setting(wifi, 'network', 'rtt_jitter_ms')} ms",
        wifi,
        "network.rtt_ms ± rtt_jitter_ms",
    )
    for name in ("schlecht", "sehr_schlecht"):
        scenario = experiments / f"{name}.yaml"
        rtt, jitter = _setting(scenario, "network", "rtt_ms"), _setting(scenario, "network", "rtt_jitter_ms")
        facts.add(
            f"c.net.{name}",
            f"{rtt} ± {jitter} ms",
            scenario,
            "network.rtt_ms ± rtt_jitter_ms",
        )
    facts.add("c.margin.gate", pp(-GATE_MARGIN, 0), ROOT / "kortyn" / "rtc_gate.py", "GATE_MARGIN")
    analysis4 = experiments / "phase4" / "analysis.yaml"
    facts.add("c.tolerance", f"{_plain(100 * _setting(analysis4, 'tolerance'))} pp", analysis4, "tolerance")
    facts.add("c.tick", ms(_setting(analysis4, "tick_ms")), analysis4, "tick_ms (one control tick at 20 Hz)")
    facts.add("c.miss", pct(MISS_SHARE_LIMIT), ROOT / "kortyn" / "fleet.py", "MISS_SHARE_LIMIT")
    analysis5 = experiments / "phase5" / "analysis.yaml"
    slo = _setting(analysis5, "gpu_decision", "latency_slo")
    facts.add("c.slo", ms(slo["max_ms"]), analysis5, "gpu_decision.latency_slo.max_ms")
    facts.add("c.slo.robots", f"{slo['robots']} robots", analysis5, "gpu_decision.latency_slo.robots")
    bench = ROOT / "kortyn" / "notebook_bench.py"
    facts.add("c.numerics", pct(notebook_bench.NUMERIC_THRESHOLD), bench, "NUMERIC_THRESHOLD")
    facts.add(
        "c.nb.sizes",
        f"{min(notebook_bench.SIZES)} to {max(notebook_bench.SIZES)} requests",
        bench,
        "SIZES (batch sizes)",
    )
    facts.add("c.nb.repeats", f"{notebook_bench.REPEATS} runs", bench, "REPEATS (timed runs per batch size)")
    rtc = experiments / "phase4" / "libero_rtc.yaml"
    weight = _setting(rtc, "profiles", "smolvla_libero_rtc", "rtc", "max_guidance_weight")
    facts.add(
        "c.rtc.weight",
        f"maximum guidance weight of {_plain(weight)}",
        rtc,
        "profiles.smolvla_libero_rtc.rtc.max_guidance_weight",
    )
    gate = experiments / "phase5" / "libero_rtc_kurz.yaml"
    transport = _setting(gate, "experiment", "suites", 0, "runs", 0, "image_transport")
    facts.add("c.zstd", f"{transport['codec']}, level {transport['level']}", gate, "runs[*].image_transport")
    facts.add(
        "c.confidence",
        "95%",
        ROOT / "kortyn" / "metrics.py",
        "wilson_interval z = 1.96; bootstrap percentiles 2.5/97.5",
    )
    dose = experiments / "phase4" / "libero_dose.yaml"
    dose_text = dose.read_text(encoding="utf-8")
    for latency in (0, 100, 200, 300, 500, 750, 1000, 1500, 2000):
        if f"fest_{latency}ms" not in dose_text and latency not in (0, 1500, 2000):
            raise PaperError(f"{dose.relative_to(ROOT)}: no level fest_{latency}ms")
        facts.add(f"c.level.{latency}", ms(latency), dose, "dose-response latency level (fest_<L>ms)")
    prereg4 = ROOT / "reports" / "phase4_preregistration.md"
    if "1,5 oder 2" not in prereg4.read_text(encoding="utf-8"):
        raise PaperError(f"{prereg4.relative_to(ROOT)}: no stretch factor 1,5")
    facts.add("c.stretch", "1.5-fold", prereg4, "H1 power: stretch k of the latency axis (1, 1,5 or 2)")


# --------------------------------------------------------------------------------------------- figures
STYLE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),  # categorical slots 1-4, fixed order
}


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.edgecolor": STYLE["axis"],
            "axes.labelcolor": STYLE["ink2"],
            "axes.facecolor": STYLE["surface"],
            "figure.facecolor": STYLE["surface"],
            "axes.grid": True,
            "grid.color": STYLE["grid"],
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": STYLE["ink2"],
            "ytick.color": STYLE["ink2"],
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
    return plt


def _wilson(wins: int, n: int) -> tuple[float, float, float]:
    low, high = wilson_interval(wins, n)
    return wins / n, low, high


def figure_phase1(data: Mapping[str, Any], out: Path) -> None:
    plt = _pyplot()
    rows = data["rows"]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for offset, (profile, label, color) in enumerate(
        (
            ("kalibriert", "DDIM-4 (inside the 700-ms budget)", STYLE["series"][0]),
            ("ddim10", "DDIM-10 (compute-bound)", STYLE["series"][1]),
        )
    ):
        ys, xs, lows, highs = [], [], [], []
        for index, row in enumerate(rows):
            if profile in row:
                rate, low, high, _ = row[profile]
                ys.append(len(rows) - 1 - index + (0.5 - offset) * 0.3)
                xs.append(100 * rate)
                lows.append(100 * (rate - low))
                highs.append(100 * (high - rate))
        ax.errorbar(
            xs,
            ys,
            xerr=[lows, highs],
            fmt="o",
            ms=5,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=0,
            label=label,
        )
    ax.set_yticks(range(len(rows)), [row["label"] for row in reversed(rows)], fontsize=7.5)
    ax.set_xlabel("success rate (%), 50 episodes per point, 95% Wilson interval")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("Phase 1: Diffusion Policy on PushT in real time", loc="left")
    fig.savefig(out)
    plt.close(fig)


def figure_simtime(data: Mapping[str, Any], out: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for profile, label, color in (
        ("kalibriert", "DDIM-4 profile", STYLE["series"][0]),
        ("ddim10", "DDIM-10 profile", STYLE["series"][1]),
    ):
        points = [p for p in data["points"] if p["profile"] == profile]
        real = [_wilson(*p["real"]) for p in points]
        sim = [_wilson(*p["sim"]) for p in points]
        ax.errorbar(
            [100 * r[0] for r in real],
            [100 * s[0] for s in sim],
            xerr=[[100 * (r[0] - r[1]) for r in real], [100 * (r[2] - r[0]) for r in real]],
            yerr=[[100 * (s[0] - s[1]) for s in sim], [100 * (s[2] - s[0]) for s in sim]],
            fmt="o",
            ms=5,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            alpha=0.9,
            label=label,
        )
    ax.plot([0, 100], [0, 100], color=STYLE["muted"], linewidth=1, linestyle="--", label="equal")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("success in real time, phase 1 (%)")
    ax.set_ylabel("success in simulation time (%)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("Simulation time against real time", loc="left")
    fig.savefig(out)
    plt.close(fig)


def figure_dose(p4: Mapping[str, Any], p5: Mapping[str, Any] | None, out: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    conditions = [
        ("no RTC (phase 4)", p4["curves"]["conditions"]["ohne_rtc"], STYLE["series"][0]),
        ("RTC, guidance H ≤ 25 (phase 4)", p4["curves"]["conditions"]["mit_rtc"], STYLE["series"][1]),
    ]
    if p5 is not None and p5.get("curves") is not None:
        conditions.append(
            ("RTC, guidance H = D + 4 (phase 5)", p5["curves"]["conditions"]["rtc_kurz"], STYLE["series"][2])
        )
    for label, condition, color in conditions:
        curve = condition["curve"]
        grid = np.asarray(curve["latency_ms"], dtype=float)
        keep = grid <= 1500.0
        ax.fill_between(
            grid[keep],
            100 * np.asarray(curve["low"])[keep],
            100 * np.asarray(curve["high"])[keep],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(grid[keep], 100 * np.asarray(curve["point"])[keep], color=color, label=label)
        xs = [level["latency_ms"] for level in condition["levels"] if level["latency_ms"] <= 1500]
        ys = [100 * level["pooled"] for level in condition["levels"] if level["latency_ms"] <= 1500]
        ax.plot(xs, ys, "o", ms=4.5, color=color, markeredgecolor=STYLE["surface"], markeredgewidth=0.8)
    ax.axvline(1250, color=STYLE["muted"], linewidth=0.8, linestyle=":")
    ax.text(1240, 20, "latency budget\n1,250 ms", ha="right", va="bottom", fontsize=7, color=STYLE["ink2"])
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 55)
    ax.set_xlabel("fixed end-to-end latency (ms)")
    ax.set_ylabel("success rate (%)")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title(
        "SmolVLA on LIBERO-Long (tasks 8 and 0), 100 episodes per level; curve with 95% bootstrap band",
        loc="left",
    )
    fig.savefig(out)
    plt.close(fig)


def figure_capacity(p4: Mapping[str, Any], out: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    names = {"a": "a FIFO", "b": "b FIFO + batching", "d": "d EDF + batching", "e": "e d + admission"}
    for ax, rtc, title in zip(axes, ("without", "rtc"), ("without RTC", "with RTC (H ≤ 25)"), strict=True):
        for index, strategy in enumerate(("a", "b", "d", "e")):
            sizes = [s for s in p4["capacity_sizes"][(rtc, strategy)] if s["robots"] <= 12]
            robots = [s["robots"] for s in sizes]
            loss = [100 * (s["single"] - s["worst"]) for s in sizes]
            dashed = strategy == "e"  # d and e coincide until admission control turns robots away
            ax.plot(
                robots,
                loss,
                marker="o" if not dashed else None,
                ms=3.5,
                linestyle="--" if dashed else "-",
                color=STYLE["series"][index],
                label=names[strategy],
            )
        ax.axhline(3.0, color=STYLE["muted"], linewidth=0.8, linestyle="--")
        ax.text(12, 3.3, "3-pp tolerance", ha="right", va="bottom", fontsize=7, color=STYLE["ink2"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("robots sharing one GPU")
        ax.set_xlim(0.5, 12.5)
    axes[0].set_ylabel("worst robot below single robot (pp)")
    axes[0].set_ylim(-0.5, 12)
    axes[1].legend(loc="upper left", fontsize=7)
    fig.suptitle(
        "Predicted loss of the worst robot, RTX 3060 Ti assumption (220 ms), office Wi-Fi",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.82, wspace=0.08)
    fig.savefig(out)
    plt.close(fig)


def figure_fleet_validation(p3: Mapping[str, Any], p4: Mapping[str, Any], out: Path) -> None:
    plt = _pyplot()
    rows: list[tuple[str, Mapping[str, Any], bool]] = [
        (f"phase 3, {c['case'][0]}, {c['robots']} robots", c, False) for c in p3["validation"]
    ]
    rows.append(("phase 4, first run, 5 robots", p4["faulty"], False))
    rows.append(("phase 4, corrected, 5 robots", p4["corrected"]["d_n5"], True))
    rows.append(("phase 4, corrected, 10 robots", p4["corrected"]["d_n10"], True))
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    for index, (_, entry, valid) in enumerate(rows):
        y = len(rows) - 1 - index
        color = STYLE["series"][0] if valid else STYLE["muted"]
        low, high = entry["interval"]
        ax.plot([100 * low, 100 * high], [y, y], color=color, linewidth=1.4)
        ax.plot([100 * entry["difference"]], [y], "o", ms=5, color=color)
    ax.axvline(0, color=STYLE["ink2"], linewidth=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=7.5)
    ax.set_xlabel("measured − predicted success (pp), with the pre-registered interval")
    ax.set_title(
        "Fleet validation: gray = invalidated by the render fault, blue = corrected runs", loc="left"
    )
    fig.savefig(out)
    plt.close(fig)


def figure_economics(p4: Mapping[str, Any], out: Path) -> None:
    plt = _pyplot()
    economics = p4["economics"]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    devices = [d for d in economics["devices"] if d["condition"] == "ohne_rtc"]
    short = {
        "Jetson Orin Nano Super 8 GB (Dev Kit)": "Orin Nano Super",
        "Jetson AGX Orin 64 GB (Dev Kit)": "AGX Orin",
        "Jetson AGX Thor (Dev Kit)": "AGX Thor",
    }
    ax.plot(
        [d["monthly_eur"] for d in devices],
        [100 * d["success"] for d in devices],
        "o",
        ms=6,
        color=STYLE["series"][0],
        label="computer in every robot",
    )
    for d in devices:
        ax.annotate(
            short[d["label"]],
            (d["monthly_eur"], 100 * d["success"]),
            textcoords="offset points",
            xytext=(6, -3),
            fontsize=7,
            color=STYLE["ink2"],
        )
    brains = []
    for rtc, color, label in (
        ("ohne_rtc", STYLE["series"][1], "shared Brain, no RTC"),
        ("mit_rtc", STYLE["series"][2], "shared Brain, RTC H ≤ 25"),
    ):
        points = [
            c for c in economics["capacity"] if c["condition"] == rtc and c["batching"] == "realistisch"
        ]
        best = max(points, key=lambda c: (c["robots"], c["worst"]))
        brains.append(best)
        ax.plot([best["monthly_eur"]], [100 * best["worst"]], "s", ms=6, color=color, label=label)
        offset = (-6, 8) if rtc == "mit_rtc" else (6, -12)
        ax.annotate(
            f"{best['robots']} robots per GPU",
            (best["monthly_eur"], 100 * best["worst"]),
            textcoords="offset points",
            xytext=offset,
            ha="right" if rtc == "mit_rtc" else "left",
            fontsize=7,
            color=STYLE["ink2"],
        )
    ax.set_xscale("log")
    ticks = [10, 20, 50, 100, 200]
    ax.set_xticks(ticks, [f"€{tick}" for tick in ticks])
    ax.minorticks_off()
    ax.set_xlim(10, 250)
    ax.set_xlabel("cost per robot and month (EUR, log scale)")
    ax.set_ylabel("predicted success (%)")
    ax.set_ylim(0, 55)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("Equal-success economics, RTX 3060 Ti assumption, realistic batching (phase 4)", loc="left")
    fig.savefig(out)
    plt.close(fig)


def figure_gate(p5: Mapping[str, Any], out: Path) -> None:
    """Gate and controls as paired differences; the colour is the tested condition, as in Figure 3."""
    plt = _pyplot()
    gate = p5["gate"]
    short, asynchronous = STYLE["series"][2], STYLE["series"][0]
    rows = [
        (f"gate: RTC short − sync25 ({gate['gate']['pairs']} pairs)", gate["gate"], short),
        ("K1: asynchronous − sync25", gate["controls"]["K1_async_vs_sync25"], asynchronous),
        ("K2: RTC short − asynchronous", gate["controls"]["K2_rtc_vs_async"], short),
    ]
    for task, entry in sorted(gate["per_task"].items(), key=lambda item: -int(item[0])):
        rows.append((f"gate, task {task} only ({entry['gate']['pairs']} pairs)", entry["gate"], short))
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for index, (_, entry, color) in enumerate(rows):
        y = len(rows) - 1 - index
        low, high = entry["interval"]
        ax.plot([100 * low, 100 * high], [y, y], color=color, linewidth=1.4)
        ax.plot([100 * entry["difference"]], [y], "o", ms=5, color=color)
    ax.axvline(0, color=STYLE["ink2"], linewidth=0.8)
    ax.axvline(-15, color=STYLE["muted"], linewidth=0.8, linestyle="--")
    ax.text(-14.6, -1.05, "−15-pp margin", ha="left", va="center", fontsize=7, color=STYLE["ink2"])
    ax.set_ylim(-1.5, len(rows) - 0.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=7.5)
    ax.set_xlabel("paired difference in success at 0 ms (pp), 95% cluster-bootstrap interval")
    ax.set_title("Phase 5: zero-latency gate of RTC with short guidance", loc="left")
    handles = [
        plt.Line2D([], [], color=short, marker="o", ms=5, linewidth=1.4, label="tested: RTC, H = D + 4"),
        plt.Line2D([], [], color=asynchronous, marker="o", ms=5, linewidth=1.4, label="tested: no RTC"),
    ]
    ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=7)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- assembly
@dataclass
class Build:
    facts: Facts
    tables: dict[str, str]
    figures: dict[str, Callable[[Path], None]]


def build() -> Build:
    facts = Facts()
    constants(facts)
    p1 = phase1(facts)
    phase2(facts)
    p3a = phase3a(facts)
    p3 = phase3(facts)
    p4 = phase4(facts)
    p5 = phase5(facts)
    tables = {
        "phase1": table_phase1(p1),
        "dose": table_dose(p3, p4, p5),
        "capacity": table_capacity(p4),
        "validation": table_validation(p3, p4),
    }
    figures: dict[str, Callable[[Path], None]] = {
        "fig1_phase1_pusht.png": lambda out: figure_phase1(p1, out),
        "fig2_simulation_time.png": lambda out: figure_simtime(p3a, out),
        "fig3_dose_response.png": lambda out: figure_dose(p4, p5, out),
        "fig4_capacity.png": lambda out: figure_capacity(p4, out),
        "fig5_economics.png": lambda out: figure_economics(p4, out),
        "fig6_fleet_validation.png": lambda out: figure_fleet_validation(p3, p4, out),
    }
    if p5.get("gate") is not None:
        tables["gate"] = table_gate(p5)
        figures["fig7_gate.png"] = lambda out: figure_gate(p5, out)
    return Build(facts=facts, tables=tables, figures=figures)


def provenance_text(facts: Facts) -> str:
    intro = (
        "Generated by `python -m kortyn.paper` from the files named below. Every quantity the report "
        "states is listed here with the file it comes from; `tests/test_paper.py` checks that the report "
        "quotes each of them verbatim and that no number with a unit appears in the report's prose without "
        "an entry here. The tables of the report are generated from the same files "
        "(`<!-- table:... -->` blocks)."
    )
    lines = [
        "# Provenance of the numbers in `paper/report.md`",
        "",
        intro,
        "",
        "| Key | As printed | Source file | Field or computation |",
        "|---|---|---|---|",
    ]
    for fact in sorted(facts.items.values(), key=lambda f: f.key):
        lines.append(f"| `{fact.key}` | {fact.text} | `{fact.source}` | {fact.detail} |")
    return "\n".join(lines) + "\n"


def _constant(text: str) -> Callable[[re.Match[str]], str]:
    """A ``re.sub`` replacement that inserts ``text`` literally (no backslash escapes)."""

    def replace(_: re.Match[str]) -> str:
        return text

    return replace


def replace_tables(report: str, tables: Mapping[str, str]) -> str:
    for name, body in tables.items():
        start, end = TABLE_START.format(name=name), TABLE_END.format(name=name)
        pattern = re.compile(re.escape(start) + r"\n.*?\n" + re.escape(end), re.DOTALL)
        if start in report:
            report = pattern.sub(_constant(f"{start}\n{body}\n{end}"), report)
    return report


def table_names(report: str) -> list[str]:
    return re.findall(r"<!-- table:([a-z0-9_]+) -->", report)


UNIT_NUMBER = re.compile(
    r"(?<![\w.-])[−+±]?\d[\d,]*(?:\.\d+)?"
    r"(?:\s?(?:%|pp|ms|s|Hz|kB|MB|GB|Mbit/s|robots|episodes|pairs|actions|ticks|steps|runs|requests))(?![\w])"
)


def prose(report: str) -> str:
    """The report without code blocks, generated tables, the reference list and link targets."""
    text = re.sub(r"```.*?```", "", report, flags=re.DOTALL)
    text = re.sub(r"<!-- table:([a-z0-9_]+) -->.*?<!-- /table:\1 -->", "", text, flags=re.DOTALL)
    text = text.split("\n## References", 1)[0]
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return text


def normalize(text: str) -> str:
    """Line breaks and runs of spaces as single spaces (the report wraps lines anywhere)."""
    return re.sub(r"\s+", " ", text)


def numbers_with_units(text: str) -> list[str]:
    return [match.group(0).strip() for match in UNIT_NUMBER.finditer(normalize(text))]


def unbacked(report: str, facts: Facts) -> list[str]:
    """Numbers with a unit in the report's prose that do not lie inside a quoted fact at that position."""
    text = normalize(prose(report))
    covered: list[tuple[int, int]] = []
    for fact_text in facts.texts():
        start = text.find(fact_text)
        while start != -1:
            covered.append((start, start + len(fact_text)))
            start = text.find(fact_text, start + 1)
    missing = []
    for match in UNIT_NUMBER.finditer(text):
        if not any(low <= match.start() and match.end() <= high for low, high in covered):
            missing.append(match.group(0))
    return missing


def unquoted(report: str, facts: Facts) -> list[str]:
    """Facts the report does not quote verbatim."""
    text = normalize(report)
    return sorted(f"{fact.key}: {fact.text}" for fact in facts.items.values() if fact.text not in text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Numbers, tables and figures of paper/report.md.")
    parser.add_argument(
        "--check", action="store_true", help="only check provenance.md and the report's tables"
    )
    parser.add_argument("--no-figures", action="store_true", help="skip the figures")
    args = parser.parse_args(argv)
    try:
        result = build()
    except PaperError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    provenance = provenance_text(result.facts)
    report = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
    updated = replace_tables(report, result.tables)
    if args.check:
        stale = []
        if not PROVENANCE.exists() or PROVENANCE.read_text(encoding="utf-8") != provenance:
            stale.append(str(PROVENANCE.relative_to(ROOT)))
        if updated != report:
            stale.append(f"{REPORT.relative_to(ROOT)} (generated tables)")
        if stale:
            print("out of date: " + ", ".join(stale) + " - run python -m kortyn.paper", file=sys.stderr)
            return 1
        print("paper is up to date")
        return 0
    PAPER.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(provenance, encoding="utf-8")
    if report:
        REPORT.write_text(updated, encoding="utf-8")
    if not args.no_figures:
        FIGURES.mkdir(parents=True, exist_ok=True)
        for name, draw in result.figures.items():
            draw(FIGURES / name)
    print(f"{len(result.facts.items)} facts, {len(result.tables)} tables, {len(result.figures)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
