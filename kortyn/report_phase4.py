"""Phase-4 report data: ``reports/phase4_data.md`` and the figures ``reports/phase4_*.png``.

    python -m kortyn.report_phase4

Inputs (sections whose inputs are missing are skipped, so the report can be built while runs continue):

* codec (step 1): ``results/phase4/codec.json`` (:mod:`kortyn.bench_codec`);
* RTC (step 2): ``results/phase4/rtc_overhead.json`` (:mod:`kortyn.bench_rtc`) and R-RTC from the curves;
* curves (step 3): ``results/phase4/curves.json`` and ``results/phase4/retest.json``
  (:mod:`kortyn.phase4_steps`);
* capacity and WLAN (step 4): ``results/phase4/fleet/capacity.json`` and
  ``results/phase4/fleet/capacity_wlan.json`` (:mod:`kortyn.fleet_capacity`), validation
  ``results/phase4/validation/evaluation.json``;
* economics (step 5): ``results/phase4/economics.json`` (:mod:`kortyn.economics_phase4`) and the price
  assumptions ``kortyn/experiments/phase3/economics.yaml``.

All numbers come from those files; this module only formats them (tables) and draws the figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from kortyn import economics_phase4 as ep
from kortyn.config import ConfigError
from kortyn.economics import EconomicsConfig, PriceSource, load_economics
from kortyn.fleet import FleetSimConfig, load_fleet_config
from kortyn.phase3_steps import StepError, load_episodes, successes
from kortyn.phase4_steps import AnalysisConfig, load_analysis_config
from kortyn.report import (
    CATEGORICAL,
    DPI,
    GRID,
    INK_2,
    MUTED,
    REFERENCE_GRAY,
    apply_style,
    header,
    header_legend,
    make_figure,
    percent_axis,
)
from kortyn.report_phase3 import BATCHING_NAMES, BATCHING_ORDER, STRATEGY_COLORS, STRATEGY_NAMES, p_text

LOG = logging.getLogger("kortyn.report_phase4")

CONDITION_COLORS = {"ohne_rtc": CATEGORICAL[0], "mit_rtc": CATEGORICAL[1]}
CONDITION_NAMES = {"ohne_rtc": "ohne RTC", "mit_rtc": "mit RTC"}
STRATEGY_ORDER = ["a", "b", "d", "e"]
BASE_NETWORK = "buero_wlan"
WLAN_STRATEGY = "d"
WLAN_BATCHING = "realistisch"
EPS = 1e-9


class ReportError(ValueError):
    """An input file exists but cannot be used."""


# --------------------------------------------------------------------------------------------- format
def de(value: float, digits: int = 0, sign: bool = False) -> str:
    """German number format: decimal comma, thin space between thousands."""
    text = f"{value:{'+' if sign else ''},.{digits}f}"
    return text.replace(",", "\u202f").replace(".", ",")


def ms(value: float | None) -> str:
    return "–" if value is None or math.isnan(value) else f"{de(value)} ms"


def num(value: float | None, digits: int = 1) -> str:
    return "–" if value is None or math.isnan(value) else de(value, digits)


def pct(value: float | None, digits: int = 0) -> str:
    return "–" if value is None or math.isnan(value) else f"{de(100 * value, digits)} %"


def pp(value: float, digits: int = 1) -> str:
    """A difference of shares in percentage points, with sign."""
    return f"{de(100 * value, digits, sign=True)} pp"


def interval(bounds: Sequence[float], digits: int = 1, scale: float = 100.0, unit: str = "pp") -> str:
    return f"[{de(scale * bounds[0], digits, True)}, {de(scale * bounds[1], digits, True)}] {unit}"


def eur(value: float, digits: int = 0) -> str:
    return f"{de(value, digits)} €"


def kbytes(value: float) -> str:
    return f"{de(value / 1000)} kB"


def listing(values: Sequence[Any]) -> str:
    return ", ".join(str(value) for value in values)


def condition_name(name: str) -> str:
    return CONDITION_NAMES.get(name, name)


def lstar_text(lstar: Mapping[str, Any], budget_ms: float) -> str:
    """``L*`` point estimate; censored at the budget as "≥ budget"."""
    point = float(lstar["point_ms"])
    return f"≥ {ms(budget_ms)}" if bool(lstar["censored"]) else ms(point)


def bound_text(value: float, budget_ms: float) -> str:
    return f"≥ {ms(budget_ms)}" if value >= budget_ms - EPS else ms(value)


def table(header_row: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header_row) + " |", "|" + "|".join("---" for _ in header_row) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


# --------------------------------------------------------------------------------------------- inputs
def load_json(path: Path) -> dict[str, Any] | None:
    """A results file, ``None`` if it does not exist yet."""
    if not path.exists():
        LOG.info("missing %s: section skipped", path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReportError(f"{path}: top level must be an object")
    return data


@dataclass(frozen=True)
class Inputs:
    analysis: AnalysisConfig
    fleet: FleetSimConfig
    economics_config: EconomicsConfig
    codec: dict[str, Any] | None
    rtc: dict[str, Any] | None
    curves: dict[str, Any] | None
    retest: dict[str, Any] | None
    sensitivity: dict[str, Any] | None
    fresh: dict[str, Any] | None
    capacity: dict[str, Any] | None
    wlan: dict[str, Any] | None
    validation: dict[str, Any] | None
    economics: dict[str, Any] | None


def load_inputs(args: argparse.Namespace) -> Inputs:
    analysis = load_analysis_config(args.analysis)
    return Inputs(
        analysis=analysis,
        fleet=load_fleet_config(args.fleet_config),
        economics_config=load_economics(args.economics_config),
        codec=load_json(Path(args.codec)),
        rtc=load_json(Path(args.rtc_overhead)),
        curves=load_json(Path(analysis.output)),
        retest=load_json(Path(analysis.retest_output)),
        sensitivity=load_json(Path(analysis.sensitivity_output)),
        fresh=load_json(Path(analysis.fresh.output)),
        capacity=load_json(Path(args.capacity)),
        wlan=load_json(Path(args.wlan_capacity)),
        validation=load_json(Path(analysis.validation.output)),
        economics=load_json(Path(args.economics)),
    )


# --------------------------------------------------------------------------------------------- step 1
def codec_section(codec: Mapping[str, Any]) -> list[str]:
    candidates = list(codec["candidates"])
    raw = next((c for c in candidates if c["codec"] == "raw"), None)
    raw_bytes = float(raw["mean_bytes"]) if raw is not None else math.nan
    rows = []
    for c in candidates:
        identical = f"{c['identical_frames']}/{c['requests']} · {c['identical_model_inputs']}/{c['requests']}"
        share = float(c["mean_bytes"]) / raw_bytes if raw_bytes > 0 else math.nan
        label = f"**{c['label']}**" if c["label"] == codec["standard"] else c["label"]
        rows.append(
            [
                label,
                identical,
                kbytes(float(c["mean_bytes"])),
                kbytes(float(c["p95_bytes"])),
                pct(share),
                num(float(c["mbit_per_robot"])),
                num(float(c["encode_ms"])),
                num(float(c["decode_ms"])),
                num(float(c["codec_ms"])),
            ]
        )
    shapes = ", ".join(
        f"{key} {'×'.join(str(v) for v in shape)}" for key, shape in codec["frame_shape"].items()
    )
    return [
        f"{codec['requests']} Anfragen aus {codec['episodes']} Episoden "
        f"(Takte {listing(codec['capture_ticks'])}), "
        f"Bilder {shapes}. Zeiten: Median je Anfrage über beide Kameras, {codec['repeats']} Wiederholungen. "
        f"Bandbreite je Roboter = mittlere Bytes · 8 / {num(float(codec['period_s']), 2)} s Anfrageabstand.",
        "",
        *table(
            [
                "Kandidat",
                "bitgleich (Bild · Modelleingabe)",
                "Bytes je Anfrage",
                "p95",
                "Anteil an roh",
                "Mbit/s je Roboter",
                "Kodieren ms",
                "Dekodieren ms",
                "Summe ms",
            ],
            rows,
        ),
        "",
        f"**Standard (Regel aus Abschnitt 3):** `{codec['standard']}` – kleinste mittlere Bytes unter den "
        f"verlustfreien Kandidaten mit Kodieren + Dekodieren ≤ {num(float(codec['max_codec_ms']), 0)} ms; "
        f"innerhalb von {pct(float(codec['tie_share']))} der schnellere.",
    ]


# --------------------------------------------------------------------------------------------- step 2
def rtc_section(rtc: Mapping[str, Any] | None, curves: Mapping[str, Any] | None, s1_ms: float) -> list[str]:
    lines: list[str] = []
    if rtc is not None:
        low, high = rtc["relative_overhead_ci"]
        applied = float(rtc["applied_overhead"])
        lines += [
            *table(
                [
                    "Messung",
                    "ohne RTC (Median)",
                    "mit RTC (Median)",
                    "r [95-%-KI]",
                    "angewandt",
                    "s₁ mit RTC",
                ],
                [
                    [
                        f"{rtc['pairs']} abwechselnde Paare, d = {rtc['delay']}, H = {rtc['horizon']}",
                        ms(float(rtc["plain_median_ms"])),
                        ms(float(rtc["rtc_median_ms"])),
                        f"{de(100 * float(rtc['relative_overhead']), 1, True)} % "
                        f"{interval([float(low), float(high)], 1, unit='%')}",
                        f"{de(100 * applied, 1)} %",
                        ms(s1_ms * (1.0 + applied)),
                    ]
                ],
            ),
            "",
        ]
    if curves is not None and "r_rtc" in curves:
        check = curves["r_rtc"]
        lines += [
            "**R-RTC** (RTC bei 0 ms gegen die synchrone Referenz `sync25`, gleiche Seeds):",
            "",
            *table(
                [
                    "Paare",
                    "sync25",
                    "RTC 0 ms",
                    "Differenz [95-%-KI]",
                    "nur sync / nur RTC",
                    "McNemar p",
                    "Urteil",
                ],
                [
                    [
                        str(check["pairs"]),
                        pct(float(check["rate_a"])),
                        pct(float(check["rate_b"])),
                        f"{pp(float(check['difference']), 0)} {interval(check['interval'], 0)}",
                        f"{check['only_a']} / {check['only_b']}",
                        p_text(float(check["p_value"])),
                        str(check["verdict"]),
                    ]
                ],
            ),
        ]
    return lines


# --------------------------------------------------------------------------------------------- step 3
def condition_tables(name: str, condition: Mapping[str, Any], budget_ms: float) -> list[str]:
    levels = condition["levels"]
    tasks = sorted(levels[0]["tasks"], key=int) if levels else []
    curve = condition["curve"]
    by_latency = {float(x): i for i, x in enumerate(curve["latency_ms"])}
    rows = []
    for level in levels:
        index = by_latency.get(float(level["latency_ms"]))
        fitted = "–"
        if index is not None:
            fitted = (
                f"{pct(float(curve['point'][index]))} "
                f"[{100 * float(curve['low'][index]):.0f}, {100 * float(curve['high'][index]):.0f}] %"
            )
        rows.append(
            [
                ms(float(level["latency_ms"])),
                ms(float(level["effective_ms"])),
                pct(float(level["pooled"])),
                *(pct(float(level["tasks"][t])) for t in tasks),
                fitted,
            ]
        )
    fit_rows = []
    for task, entry in sorted(condition["fit"].items(), key=lambda item: int(item[0])):
        family = "log-logistisch" if entry["family"] == "loglogistic" else "isoton (PAVA + PCHIP)"
        fit_rows.append(
            [
                f"Aufgabe {task}",
                family,
                pct(float(entry["p"])) if "p" in entry else "–",
                ms(float(entry["x50_ms"])) if "x50_ms" in entry else "–",
                num(float(entry["h"]), 2) if "h" in entry else "–",
                f"{num(float(entry['deviance']), 2)} / {entry['df']:.0f}",
                p_text(float(entry["p_value"])),
            ]
        )
    return [
        f"**{condition_name(name)}** – gemessene Stufen und glatte Kurve (95-%-Bootstrap-Band):",
        "",
        *table(
            [
                "Stufe",
                "wirksam (Takt-Regel)",
                "Erfolg (Aufgaben gleich gewichtet)",
                *(f"Aufgabe {t}" for t in tasks),
                "Kurve [95-%-Band]",
            ],
            rows,
        ),
        "",
        *table(["Aufgabe", "Familie", "p", "x₅₀", "h", "Devianz / FG", "GOF p"], fit_rows),
        "",
    ]


def lstar_rows(curves: Mapping[str, Any], names: Sequence[str]) -> list[list[str]]:
    budget = float(curves["spec"]["fit_max_ms"])
    rows = []
    for name in names:
        condition = curves["conditions"].get(name)
        if condition is None:
            continue
        lstar = condition["lstar"]
        rows.append(
            [
                condition_name(name),
                lstar_text(lstar, budget),
                bound_text(float(lstar["conservative_ms"]), budget),
                bound_text(float(lstar["upper_ms"]), budget),
                ms(float(lstar["x_point_ms"])),
                pct(float(condition["curve"]["point"][0])),
                pct(float(condition["success_at_lstar"])),
            ]
        )
    return rows


def curves_section(curves: Mapping[str, Any], names: Sequence[str]) -> list[str]:
    budget = float(curves["spec"]["fit_max_ms"])
    lines = [
        f"{curves['seeds']} Seeds je Stufe (Aufgaben {listing(curves['tasks'])}), "
        "gepaarter Seed-Bootstrap mit "
        f"{curves['bootstrap']} Ziehungen (Seed {curves['seed']}), "
        f"Toleranz {pct(float(curves['tolerance']))}.",
        "",
    ]
    for name in names:
        if name in curves["conditions"]:
            lines += condition_tables(name, curves["conditions"][name], budget)
    lines += [
        "**L\\*** (größte Latenz mit höchstens 3 pp vorhergesagtem Verlust gegenüber 0 ms derselben "
        "Bedingung):",
        "",
        *table(
            [
                "Bedingung",
                "L* (Punkt)",
                "konservativ (2,5 %)",
                "obere Grenze (97,5 %)",
                "x* (stetig)",
                "Erfolg bei 0 ms",
                "Erfolg bei L*",
            ],
            lstar_rows(curves, names),
        ),
        "",
    ]
    effect = curves.get("rtc_effect")
    if effect is not None:
        h1, h2 = effect["h1"], effect["h2"]
        lines += table(
            ["Hypothese", "Schätzung", "95-%-KI (gepaart)", "Urteil"],
            [
                [
                    "H1: mittlerer Gewinn über die 25 Takte im Budget",
                    pp(float(h1["gain"])),
                    interval(h1["interval"]),
                    str(h1["verdict"]),
                ],
                [
                    "H2: ΔL* = L*(mit) − L*(ohne)",
                    f"{de(float(h2['difference_ms']), 0, True)} ms",
                    interval(h2["interval_ms"], 0, scale=1.0, unit="ms"),
                    str(h2["verdict"]),
                ],
            ],
        )
    return lines


def sensitivity_section(sensitivity: Mapping[str, Any]) -> list[str]:
    """Supplement: the curve without RTC with and without the pooled phase-3 levels."""
    names = {"vorab": "präregistriert (nur Phase 4)", "gepoolt": "mit Phase-3-Stufen gepoolt"}
    rows = []
    for key, label in names.items():
        entry = sensitivity[key]
        curve = entry["curve"]
        rows.append(
            [
                label,
                ", ".join(f"{task}: {family}" for task, family in sorted(entry["families"].items())),
                f"{'≥ ' if entry['censored'] else ''}{ms(float(entry['lstar_ms']))}",
                pct(float(entry["success_0ms"])),
                " / ".join(pct(float(curve[x])) for x in ("250.0", "500.0", "750.0")),
                f"{pp(float(entry['retest_bias']))} {interval(entry['retest_interval'])}",
                str(entry["retest_verdict"]),
            ]
        )
    levels = ", ".join(ms(float(level["latency_ms"])) for level in sensitivity["phase3_levels"])
    return [
        f"Ergänzung (nicht präregistriert): Die Phase-3-Episoden der wieder gemessenen Stufen ({levels}) "
        "gepoolt in die Kurve ohne RTC; Punktschätzung mit derselben Anpassungsregel.",
        "",
        *table(
            [
                "Kurve ohne RTC",
                "Familie je Aufgabe",
                "L*",
                "Erfolg bei 0 ms",
                "250 / 500 / 750 ms",
                "Re-Test-Bias [95-%-KI]",
                "Urteil",
            ],
            rows,
        ),
    ]


def fresh_section(fresh: Mapping[str, Any]) -> list[str]:
    """Supplement: one episode per fresh process against the dose-response runs of the same level."""
    names = {"block_a": "Block A (alle Stufen eines Seeds nacheinander)", "phase3": "Phase 3"}
    rows = [
        [
            names.get(label, label),
            str(comparison["pairs"]),
            pct(float(comparison["rate_a"])),
            pct(float(comparison["rate_b"])),
            f"{pp(float(comparison['difference']), 0)} {interval(comparison['interval'], 0)}",
            f"{comparison['only_a']} / {comparison['only_b']}",
            p_text(float(comparison["p_value"])),
        ]
        for label, comparison in fresh["comparisons"].items()
    ]
    return [
        f"Ergänzung (nicht präregistriert): Seeds {fresh['seeds'][0]}–{fresh['seeds'][1]} bei 500 ms, jede "
        f"Episode in einem frischen Prozess ({fresh['episodes']} Episoden, "
        f"Erfolg {pct(float(fresh['success']))}).",
        "",
        *table(
            [
                "Vergleich",
                "Paare",
                "Referenz",
                "frischer Prozess",
                "Differenz [95-%-KI]",
                "nur Referenz / nur frisch",
                "McNemar p",
            ],
            rows,
        ),
    ]


def retest_section(retest: Mapping[str, Any]) -> list[str]:
    rows = [
        [
            case,
            str(values["episodes"]),
            pct(float(values["measured"])),
            pct(float(values["predicted_own"])),
            pct(float(values["predicted_stage1"])),
            pct(float(values["predicted_phase3_own"])),
            pct(float(values["predicted_phase3_stage1"])),
        ]
        for case, values in sorted(retest["cases"].items())
    ]
    names = {
        "own": "dichte Kurve, eigene Latenzen",
        "stage1": "dichte Kurve, Latenzen aus Stufe 1",
        "phase3_own": "Phase-3-Regel, eigene Latenzen",
        "phase3_stage1": "Phase-3-Regel, Latenzen aus Stufe 1",
    }
    pooled_rows = [
        [
            names[key],
            pp(float(value["bias"])),
            interval(value["interval"]),
        ]
        for key, value in retest["pooled"].items()
    ]
    return [
        *table(
            [
                "Fall",
                "Episoden",
                "gemessen",
                "dicht · eigene",
                "dicht · Stufe 1",
                "Phase 3 · eigene",
                "Phase 3 · Stufe 1",
            ],
            rows,
        ),
        "",
        "Bias gemessen − vorhergesagt über die aussagekräftigen Fälle "
        f"{listing(retest['informative_cases'])} "
        f"({retest['episodes']} Episoden, Cluster-Bootstrap über Seeds):",
        "",
        *table(["Vorhersage", "Bias", "95-%-KI"], pooled_rows),
        "",
        f"**Urteil (eigene Latenzen):** {retest['verdict']} · **Urteil (Stufe-1-Latenzen):** "
        f"{retest['verdict_stage1']}",
    ]


# --------------------------------------------------------------------------------------------- step 4
ConfigKey = tuple[str, str, str, str]  # network, gpu, batching, strategy


def entries_by_key(capacity: Mapping[str, Any], fleet: FleetSimConfig) -> dict[ConfigKey, Mapping[str, Any]]:
    """Capacity entries; strategies without batching apply to every batching profile."""
    first = next(iter(fleet.batching))
    found: dict[ConfigKey, Mapping[str, Any]] = {}
    for entry in capacity["configurations"]:
        strategy = entry["strategy"]
        key = (entry["network"], entry["gpu"], entry["batching"], strategy)
        found[key] = entry
        definition = fleet.strategies.get(strategy)
        if definition is not None and not definition.batches and entry["batching"] == first:
            for profile in fleet.batching:
                found.setdefault((entry["network"], entry["gpu"], profile, strategy), entry)
    return found


def size_at(entry: Mapping[str, Any], robots: int) -> Mapping[str, Any] | None:
    return next((s for s in entry["sizes"] if int(s["robots"]) == robots), None)


def condition_gpus(capacity: Mapping[str, Any]) -> dict[str, str]:
    """Condition -> GPU profile of the capacity results (one GPU profile per condition)."""
    pairs: dict[str, str] = {}
    for entry in capacity["configurations"]:
        known = pairs.setdefault(entry["condition"], entry["gpu"])
        if known != entry["gpu"]:
            raise ReportError(
                f"condition {entry['condition']!r} appears with GPU {known!r} and {entry['gpu']!r}"
            )
    return pairs


def capacity_section(capacity: Mapping[str, Any], fleet: FleetSimConfig, names: Sequence[str]) -> list[str]:
    found = entries_by_key(capacity, fleet)
    gpus = condition_gpus(capacity)
    rows = []
    for profile in [p for p in BATCHING_ORDER if p in fleet.batching]:
        for strategy in [s for s in STRATEGY_ORDER if s in fleet.strategies]:
            row = [BATCHING_NAMES.get(profile, profile), STRATEGY_NAMES.get(strategy, strategy)]
            for name in names:
                entry = found.get((BASE_NETWORK, gpus.get(name, ""), profile, strategy))
                if entry is None:
                    row += ["–", "–", "–"]
                    continue
                capacity_point = int(entry["capacity"])
                at = size_at(entry, capacity_point) if capacity_point else None
                row += [
                    f"**{capacity_point}** ({entry['capacity_conservative']})",
                    "–" if at is None else f"{pct(float(at['worst']))} ({de(100 * float(at['loss']), 1)} pp)",
                    "–"
                    if at is None
                    else f"{pct(float(at['busy_share']))} / {pct(float(at['miss_share']), 2)}",
                ]
            rows.append(row)
    header_row = ["Batching", "Strategie"]
    for name in names:
        label = condition_name(name)
        header_row += [
            f"{label}: Kapazität (konservativ)",
            f"{label}: schlechtester (Verlust) bei K",
            f"{label}: GPU belegt / verpasst bei K",
        ]
    return [
        f"Netz Büro-WLAN (Bandbreite unbegrenzt), Toleranz {pct(float(capacity['tolerance']))}, höchstens "
        f"{pct(float(capacity['miss_share_limit']))} verpasste Anfragen; Kapazität = größte Größe, bei der "
        "sie und jede kleinere beide Kriterien erfüllen. Strategie a batcht nicht und gilt für jedes Profil.",
        "",
        *table(header_row, rows),
    ]


@dataclass(frozen=True)
class WlanRow:
    rate_mbit_s: float
    request_bytes: int
    condition: str
    capacity: int
    capacity_unlimited: int
    gpu_busy: float | None
    medium_busy: float | None

    @property
    def network_limits(self) -> bool:
        return self.capacity < self.capacity_unlimited


def wlan_rows(wlan: Mapping[str, Any], capacity: Mapping[str, Any], fleet: FleetSimConfig) -> list[WlanRow]:
    """Section 7.4: capacity with the shared medium against the unlimited network (strategy d, realistic)."""
    unlimited = entries_by_key(capacity, fleet)
    rows = []
    for entry in wlan["configurations"]:
        if entry["strategy"] != WLAN_STRATEGY or entry["batching"] != WLAN_BATCHING:
            continue
        network = fleet.networks.get(entry["network"])
        if network is None or network.wlan_mbit_s is None:
            raise ReportError(f"network {entry['network']!r} has no shared WLAN medium in the fleet config")
        base = unlimited.get((BASE_NETWORK, entry["gpu"], WLAN_BATCHING, WLAN_STRATEGY))
        if base is None:
            raise ReportError(f"no unlimited-network capacity for GPU {entry['gpu']!r}")
        capacity_c = int(entry["capacity"])
        at = size_at(entry, capacity_c) if capacity_c else None
        medium = None if at is None or at.get("medium_busy_share") is None else float(at["medium_busy_share"])
        rows.append(
            WlanRow(
                rate_mbit_s=float(network.wlan_mbit_s),
                request_bytes=int(network.request_bytes),
                condition=str(entry["condition"]),
                capacity=capacity_c,
                capacity_unlimited=int(base["capacity"]),
                gpu_busy=None if at is None else float(at["busy_share"]),
                medium_busy=medium,
            )
        )
    return sorted(rows, key=lambda r: (r.request_bytes, r.condition, r.rate_mbit_s))


def wlan_section(rows: Sequence[WlanRow]) -> list[str]:
    body = []
    for row in rows:
        limit = f"Netz ab n = {row.capacity + 1}" if row.network_limits else "GPU (Netz begrenzt nicht)"
        body.append(
            [
                f"{row.rate_mbit_s:g} Mbit/s",
                kbytes(row.request_bytes),
                condition_name(row.condition),
                f"**{row.capacity}**",
                str(row.capacity_unlimited),
                limit,
                pct(row.gpu_busy),
                pct(row.medium_busy),
            ]
        )
    return [
        "Strategie d, Batching realistisch; ein gemeinsames Funkmedium (FIFO, beide Richtungen) zusätzlich "
        "zum Latenzmodell des Büro-WLANs. Auslastungen an der Kapazitätsgrenze mit WLAN.",
        "",
        *table(
            [
                "Nutzrate C",
                "Nutzlast je Anfrage",
                "Bedingung",
                "Kapazität mit WLAN",
                "Kapazität unbegrenzt",
                "Engpass",
                "GPU belegt",
                "Funk belegt",
            ],
            body,
        ),
    ]


def validation_section(validation: Mapping[str, Any]) -> list[str]:
    confidence = float(validation["confidence"])
    rows = []
    for case, values in sorted(validation["cases"].items()):
        low, high = values["interval"]
        own_low, own_high = values["interval_own"]
        rows.append(
            [
                case,
                str(values["episodes"]),
                pct(float(values["measured"])),
                pct(float(values["predicted"])),
                f"{pp(float(values['difference']))} {interval([low, high])}",
                "erfüllt" if values["passes"] else "**verfehlt**",
                f"{pct(float(values['predicted_own']))}: {pp(float(values['difference_own']))} "
                f"{interval([own_low, own_high])}",
            ]
        )
    return [
        *table(
            [
                "Fall",
                "Episoden",
                "gemessen",
                "vorhergesagt (Stufe 1)",
                f"Differenz [{de(100 * confidence, 1)}-%-KI]",
                "Kriterium V",
                "Diagnose: Vorhersage aus eigenen Latenzen",
            ],
            rows,
        ),
        "",
        f"**Validierung insgesamt:** {'erfüllt' if validation['passes'] else 'nicht erfüllt'} "
        f"(Bedingung {condition_name(str(validation['condition']))}).",
    ]


# --------------------------------------------------------------------------------------------- step 5
def source_text(source: PriceSource) -> str:
    note = f" ({source.note})" if source.note else ""
    return f"{source.source}, Stand {source.date}{note}"


def prices_section(config: EconomicsConfig) -> list[str]:
    rows = []
    gpu = next((g for g in config.gpus if g.profile == "rtx3060ti_schaetzung"), None)
    if gpu is not None:
        rows.append(
            [gpu.label, eur(gpu.price_eur), f"{gpu.board_power_w:.0f} W", source_text(gpu.price_source)]
        )
    if config.host is not None:
        rows.append(
            [
                config.host.label,
                eur(config.host.price_eur),
                f"{config.host.power_w:.0f} W",
                source_text(config.host.price_source),
            ]
        )
    if config.client is not None:
        rows.append(
            [
                config.client.label,
                eur(config.client.price_eur, 2),
                f"{config.client.power_w:.0f} W",
                source_text(config.client.price_source),
            ]
        )
    for option in config.onboard:
        rows.append(
            [
                f"{option.label} · SmolVLA {ms(option.latency_ms)} ({option.runtime or 'Laufzeit offen'})",
                eur(option.price_eur, 2),
                f"{option.power_w:.0f} W",
                f"{source_text(option.price_source)}; Latenz: {source_text(option.latency_source)}",
            ]
        )
    return [
        "**Annahmen** (alle Preise netto, Abschreibung "
        f"{config.depreciation_months} Monate, Strom {num(config.electricity_eur_per_kwh, 4)} €/kWh "
        f"[{source_text(config.electricity_source)}], "
        f"{config.operating_hours_per_month:.0f} h Betrieb je Monat):",
        "",
        *table(["Komponente", "Preis (Annahme)", "Leistung", "Quelle"], rows),
    ]


def best_matches(
    economics: Mapping[str, Any], batch_free: Collection[str]
) -> list[tuple[str, Mapping[str, Any]]]:
    """Equal-success rows of the decision's best strategy per batching profile and condition.

    A strategy without batching is stored under one profile only; its rows count for every profile.
    """
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for batching, by_condition in economics["decisions"].items():
        for condition, decided in by_condition.items():
            if "brain" not in decided:
                continue
            strategy = decided["brain"]["strategy"]
            for match in economics["matches"]:
                if match["condition"] != condition or match["strategy"] != strategy:
                    continue
                if match["batching"] == batching or strategy in batch_free:
                    rows.append((batching, match))
    return rows


def economics_section(
    economics: Mapping[str, Any], config: EconomicsConfig, batch_free: Collection[str] = ("a",)
) -> list[str]:
    devices = [
        [
            str(d["label"]),
            condition_name(str(d["condition"])),
            ms(float(d["latency_ms"])),
            pct(float(d["success"])),
            eur(float(d["monthly_eur"]), 2),
        ]
        for d in economics["devices"]
    ]
    capacity_rows = [
        [
            condition_name(str(r["condition"])),
            BATCHING_NAMES.get(str(r["batching"]), str(r["batching"])),
            STRATEGY_NAMES.get(str(r["strategy"]), str(r["strategy"])),
            str(r["robots"]),
            pct(r["worst"]),
            "–" if r["monthly_eur"] is None else eur(float(r["monthly_eur"]), 2),
        ]
        for r in economics["capacity"]
    ]
    matches = [
        [
            str(m["device"]),
            condition_name(str(m["condition"])),
            BATCHING_NAMES.get(batching, batching),
            str(m["strategy"]),
            pct(float(m["device_success"])),
            eur(float(m["device_monthly_eur"]), 2),
            f"{m['brain']['robots']} ({m['robots_monotone']})",
            "–" if m["brain"]["monthly_eur"] is None else eur(float(m["brain"]["monthly_eur"]), 2),
            f"**{m['winner']}**",
        ]
        for batching, m in best_matches(economics, batch_free)
    ]
    decision_rows = []
    for batching, by_condition in economics["decisions"].items():
        for condition, decided in by_condition.items():
            if "brain" not in decided:
                decision_rows.append(
                    [
                        BATCHING_NAMES.get(batching, batching),
                        condition_name(condition),
                        "–",
                        "–",
                        "–",
                        str(decided.get("reason", "")),
                    ]
                )
                continue
            brain, reference = decided["brain"], decided["reference"]
            decision_rows.append(
                [
                    BATCHING_NAMES.get(batching, batching),
                    condition_name(condition),
                    f"{brain['strategy']}, K = {brain['robots']}, "
                    f"schlechtester {pct(float(brain['worst']))}, {eur(float(brain['monthly_eur']), 2)}",
                    f"{reference['label']} ({pct(float(reference['success']))}, "
                    f"{eur(float(reference['monthly_eur']), 2)})",
                    "ja" if decided["cheaper"] else "nein",
                    "ja" if decided["fleet_ok"] else "nein",
                ]
            )
    client = economics.get("client_monthly_eur")
    return [
        *prices_section(config),
        "",
        "**Rechner im Roboter** (Erfolg aus der gepoolten Kurve der Bedingung bei seiner Inferenzzeit, mit "
        f"RTC · (1 + r), r = {de(100 * float(economics['rtc_overhead']), 1)} %; "
        "Kosten je Roboter und Monat):",
        "",
        *table(["Gerät", "Bedingung", "Latenz", "Erfolg", "€ je Roboter und Monat"], devices),
        "",
        "**Brain an der Kapazität (7.3)** (je Roboter und Monat inklusive Client"
        + (f" {eur(float(client), 2)}" if client is not None else "")
        + "):",
        "",
        *table(
            ["Bedingung", "Batching", "Strategie", "K", "schlechtester", "€ je Roboter und Monat"],
            capacity_rows,
        ),
        "",
        "**Gleicher Erfolg** (beste Strategie je Profil; größte Größe mit Kriterium 1, deren schlechtester "
        "Roboter den Erfolg des Geräts erreicht; in Klammern: bis zu der jede kleinere Größe ihn erreicht; "
        "alle Strategien in `results/phase4/economics.json`):",
        "",
        *table(
            [
                "Gerät",
                "Bedingung",
                "Batching",
                "Strategie",
                "Erfolg Gerät",
                "€ Gerät",
                "Brain-Größe",
                "€ Brain je Roboter",
                "günstiger",
            ],
            matches,
        ),
        "",
        "**Entscheidungsregel** (beste Strategie = größte Kapazität, Gleichstand d, e, b, a):",
        "",
        *table(
            ["Batching", "Bedingung", "Brain", "Referenzgerät", "Brain günstiger", "K ≥ 4"],
            decision_rows,
        ),
        "",
        f"**Urteil (Abschnitt 8): {economics['verdict']}**",
    ]


# --------------------------------------------------------------------------------------------- figures
def reference_rate(analysis: AnalysisConfig) -> float | None:
    path = Path(analysis.reference_sync) / "episodes.jsonl"
    if not path.exists():
        return None
    outcomes = successes(load_episodes(path))
    return float(np.mean(list(outcomes.values()))) if outcomes else None


def figure_curves(
    curves: Mapping[str, Any], names: Sequence[str], sync_rate: float | None, path: Path
) -> None:
    """Left: both pooled curves with bands and the measured levels; right: loss against 0 ms with L*."""
    apply_style()
    budget = float(curves["spec"]["fit_max_ms"])
    tolerance = float(curves["tolerance"])
    fig, axes, layout = make_figure(1, 2, 3.9, 3.0, header_in=1.4, wspace_in=0.75)
    absolute, relative = axes[0][0], axes[0][1]
    handles: list[Any] = []
    for name in names:
        condition = curves["conditions"].get(name)
        if condition is None:
            continue
        color = CONDITION_COLORS.get(name, CATEGORICAL[2])
        smooth = condition["smooth"]
        x = np.array(smooth["x_ms"], dtype=float)
        absolute.fill_between(
            x, smooth["low"], smooth["high"], color=color, alpha=0.18, linewidth=0, zorder=2
        )
        absolute.plot(x, smooth["point"], color=color, linewidth=2.0, zorder=3)
        levels = condition["levels"]
        absolute.plot(
            [float(level["effective_ms"]) for level in levels],
            [float(level["pooled"]) for level in levels],
            linestyle="none",
            marker="o",
            markersize=5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=4,
        )
        start = float(np.asarray(smooth["point"], dtype=float)[0])
        loss = start - np.asarray(smooth["point"], dtype=float)
        relative.plot(x, 100 * loss, color=color, linewidth=2.0, zorder=3)
        lstar = condition["lstar"]
        x_star = float(lstar["x_point_ms"])
        if not bool(lstar["censored"]):
            relative.plot([x_star], [100 * tolerance], marker="o", markersize=6, color=color, zorder=5)
            low, high = float(lstar["conservative_ms"]), float(lstar["upper_ms"])
            relative.plot(
                [low, high], [100 * tolerance] * 2, color=color, linewidth=5.0, alpha=0.35, zorder=4
            )
        handles.append(
            Line2D([0], [0], color=color, linewidth=2.0, marker="o", markersize=5, label=condition_name(name))
        )
    for ax in (absolute, relative):
        ax.axvline(budget, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlim(0, budget * 1.04)
        ax.set_xlabel("wirksame Latenz (ms, Takt-Regel)")
    absolute.text(budget - 20, 0.98, "Budget", color=INK_2, fontsize=8, ha="right", va="top")
    if sync_rate is not None:
        absolute.axhline(sync_rate, color=REFERENCE_GRAY, linewidth=1.0, linestyle=(0, (1, 2)), zorder=1)
        absolute.text(
            budget - 20,
            sync_rate + 0.015,
            "synchron (sync25)",
            color=INK_2,
            fontsize=8,
            ha="right",
            va="bottom",
        )
    absolute.set_ylim(0, 1)
    percent_axis(absolute)
    absolute.set_ylabel("Erfolg (Aufgaben gleich gewichtet)")
    absolute.set_title("Erfolg absolut", loc="left", fontsize=10)
    relative.axhline(100 * tolerance, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    relative.text(
        budget - 20,
        100 * tolerance - 0.6,
        f"Toleranz {100 * tolerance:.0f} pp",
        color=INK_2,
        fontsize=8,
        ha="right",
        va="top",
    )
    relative.set_ylim(-2, max(10.0, relative.get_ylim()[1]))
    relative.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} pp"))
    relative.set_ylabel("Verlust gegenüber 0 ms")
    relative.set_title("Verlust und L* (Punkt, 95-%-Intervall)", loc="left", fontsize=10)
    top = header(
        fig,
        layout,
        "Dosis-Wirkung: SmolVLA auf LIBERO-Long ohne und mit RTC",
        "Glatte monotone Kurve je Aufgabe, gepoolt; Band = 95-%-Bootstrap über Seeds; Punkte = gemessene "
        "Stufen an ihrer wirksamen Latenz.",
    )
    header_legend(fig, layout, handles, ncol=len(handles), top_in=top)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_capacity(
    capacity: Mapping[str, Any], fleet: FleetSimConfig, names: Sequence[str], path: Path
) -> None:
    """Capacity against the batching gain, one panel per condition, one line per strategy."""
    apply_style()
    found = entries_by_key(capacity, fleet)
    gpus = condition_gpus(capacity)
    profiles = [p for p in BATCHING_ORDER if p in fleet.batching]
    betas = [fleet.batching[p].beta for p in profiles]
    shown = [n for n in names if n in gpus]
    values = [int(e["capacity"]) for e in capacity["configurations"]]
    top_value = float(max([1, *values]))
    fig, axes, layout = make_figure(1, max(1, len(shown)), 3.5, 2.6, header_in=1.45, sharey=True)
    for index, name in enumerate(shown):
        ax = axes[0][index]
        for strategy in [s for s in STRATEGY_ORDER if s in fleet.strategies]:
            y = [found.get((BASE_NETWORK, gpus[name], p, strategy)) for p in profiles]
            ax.plot(
                betas,
                [math.nan if e is None else float(e["capacity"]) for e in y],
                color=STRATEGY_COLORS[strategy],
                linewidth=2.0,
                marker="o",
                markersize=5,
                zorder=3,
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
        ax.set_title(f"RTX 3060 Ti, {condition_name(name)}", loc="left", fontsize=10)
        ax.set_xlabel("Batching-Zusatzkosten β je weiterer Anfrage")
        if index == 0:
            ax.set_ylabel("Kapazität (Roboter)")
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
        "Kapazität: Roboter je RTX 3060 Ti nach Strategie und Batching-Profil",
        "Büro-WLAN; Kapazität = größte Flotte mit ≤ 1 % verpassten Anfragen und schlechtestem Roboter "
        "höchstens 3 pp unter dem Einzelroboter (Punktschätzung).",
    )
    header_legend(fig, layout, handles, ncol=2, top_in=top)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _condition_rank(condition: str) -> int:
    order = list(CONDITION_COLORS)
    return order.index(condition) if condition in order else len(order)


def figure_wlan(rows: Sequence[WlanRow], path: Path) -> None:
    """Capacity against the WLAN rate; dashed: the same configuration with unlimited bandwidth."""
    apply_style()
    fig, axes, layout = make_figure(1, 1, 5.6, 3.0, header_in=1.45)
    ax = axes[0][0]
    payloads = sorted({row.request_bytes for row in rows})
    styles: list[Literal["-"] | tuple[float, tuple[float, ...]]] = ["-", (0, (1, 1.5))]
    handles: list[Any] = []
    rates = sorted({row.rate_mbit_s for row in rows})
    conditions = sorted({row.condition for row in rows}, key=_condition_rank)
    for position, condition in enumerate(conditions):
        color = CONDITION_COLORS.get(condition, CATEGORICAL[2])
        dodge = 1.0 + 0.04 * (position - (len(conditions) - 1) / 2)  # equal capacities stay visible
        unlimited = [row.capacity_unlimited for row in rows if row.condition == condition]
        if unlimited:
            ax.axhline(unlimited[0], color=color, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        for index, payload in enumerate(payloads):
            selected = sorted(
                (row for row in rows if row.condition == condition and row.request_bytes == payload),
                key=lambda row: row.rate_mbit_s,
            )
            if not selected:
                continue
            ax.plot(
                [row.rate_mbit_s * dodge for row in selected],
                [row.capacity for row in selected],
                color=color,
                linewidth=2.0,
                linestyle=styles[index % len(styles)],
                marker="o",
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=3,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2.0,
                    linestyle=styles[index % len(styles)],
                    marker="o",
                    markersize=5,
                    label=f"{condition_name(condition)}, {kbytes(payload)} je Anfrage",
                )
            )
    handles.append(
        Line2D([0], [0], color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), label="Netz unbegrenzt")
    )
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(rates))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Nutzrate des geteilten WLANs C (Mbit/s, Annahme)")
    ax.set_ylabel("Kapazität (Roboter)")
    ax.set_ylim(
        0, max([1, *(row.capacity_unlimited for row in rows), *(row.capacity for row in rows)]) * 1.15
    )
    ax.grid(axis="x", color=GRID)
    top = header(
        fig,
        layout,
        "Ab wann das WLAN statt der GPU begrenzt",
        "Strategie d, Batching realistisch, RTX 3060 Ti; ein gemeinsames Funkmedium für alle Roboter. Liegt "
        "eine Linie unter ihrer gestrichelten Linie, begrenzt das Netz.",
    )
    header_legend(fig, layout, handles, ncol=2, top_in=top)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def short_device(label: str) -> str:
    """'Jetson AGX Orin 64 GB (Dev Kit)' -> 'AGX Orin'."""
    text = re.sub(r"\s*\(.*\)$", "", label)
    text = re.sub(r"^Jetson\s+", "", text)
    return re.sub(r"\s+\d+\s*GB$", "", text).strip()


def cost_curve(config: EconomicsConfig, entry: Mapping[str, Any]) -> tuple[list[int], list[float]]:
    """Cost per robot and month (client included) at every scanned size that keeps criterion 1."""
    sizes, costs = [], []
    for size in sorted(entry["sizes"], key=lambda s: int(s["robots"])):
        if not bool(size["criterion_1"]):
            break
        row = ep.brain_row(config, entry, size)
        if row.monthly_eur is not None:
            sizes.append(row.robots)
            costs.append(row.monthly_eur)
    return sizes, costs


def figure_costs(
    economics: Mapping[str, Any],
    capacity: Mapping[str, Any],
    fleet: FleetSimConfig,
    config: EconomicsConfig,
    names: Sequence[str],
    path: Path,
) -> None:
    """Brain cost per robot against the fleet size (best strategy, realistic) and the onboard computers."""
    apply_style()
    found = entries_by_key(capacity, fleet)
    gpus = condition_gpus(capacity)
    fig, axes, layout = make_figure(1, 1, 6.2, 3.3, header_in=1.45, right_in=2.3)
    ax = axes[0][0]
    handles: list[Any] = []
    largest = 1
    for name in names:
        decided = economics["decisions"]["realistisch"].get(name, {})
        if "brain" not in decided or name not in gpus:
            continue
        brain = decided["brain"]
        entry = found.get((BASE_NETWORK, gpus[name], "realistisch", brain["strategy"]))
        if entry is None:
            continue
        sizes, costs = cost_curve(config, entry)
        if not sizes:
            continue
        largest = max(largest, sizes[-1])
        color = CONDITION_COLORS.get(name, CATEGORICAL[2])
        ax.plot(sizes, costs, color=color, linewidth=2.0, zorder=3)
        k = int(brain["robots"])
        if k in sizes:
            ax.plot([k], [costs[sizes.index(k)]], marker="o", markersize=7, color=color, zorder=4)
            ax.annotate(
                f"K = {k}",
                (k, costs[sizes.index(k)]),
                xytext=(4, 6),
                textcoords="offset points",
                color=INK_2,
                fontsize=8,
            )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=6,
                label=f"Brain {condition_name(name)} (Strategie {brain['strategy']}; Punkt = Kapazität)",
            )
        )
    with_rtc = economics["rtc_condition"]
    devices = [d for d in economics["devices"] if d["condition"] == with_rtc]
    for device in devices:
        value = float(device["monthly_eur"])
        ax.axhline(value, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            f"{short_device(str(device['label']))}: {eur(value)} · {pct(float(device['success']))}",
            (1.0, value),
            xycoords=("axes fraction", "data"),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_2,
            fontsize=8,
            va="center",
            annotation_clip=False,
        )
    ax.set_yscale("log")
    low, high = ax.get_ylim()
    ticks = [v for v in (5, 10, 20, 50, 100, 200, 500, 1000) if low <= v <= high]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f} €".replace(",", "\u202f")))
    ax.set_xlim(0.5, largest + 0.5)
    ax.set_xlabel("Roboter am Brain (Größen mit ≤ 1 % verpassten Anfragen)")
    ax.set_ylabel("€ je Roboter und Monat")
    top = header(
        fig,
        layout,
        "Kosten je Roboter: ein Brain gegen einen Rechner in jedem Roboter",
        "RTX 3060 Ti + Host-PC + Raspberry Pi 5 je Roboter, Batching realistisch; gestrichelt: Jetson im "
        "Roboter (Kosten · Erfolg mit RTC). Alle Preise sind Annahmen (Quellen im Datenteil).",
    )
    header_legend(fig, layout, handles, ncol=1, top_in=top)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- build
def build(inputs: Inputs, out_dir: Path) -> list[str]:
    analysis = inputs.analysis
    names = [analysis.without, analysis.with_rtc]
    s1 = (
        inputs.fleet.gpus["rtx3060ti_schaetzung"].s1_ms
        if "rtx3060ti_schaetzung" in inputs.fleet.gpus
        else None
    )
    lines = [
        "# Phase 4: Daten (automatisch erzeugt)",
        "",
        f"_Erzeugt am {datetime.now(UTC):%d.%m.%Y %H:%M} UTC mit `python -m kortyn.report_phase4`._",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    if inputs.codec is not None:
        lines += ["## Schritt 1: verlustfreier Bildtransport", "", *codec_section(inputs.codec), ""]
    if inputs.rtc is not None or (inputs.curves is not None and "r_rtc" in inputs.curves):
        lines += ["## Schritt 2: RTC", "", *rtc_section(inputs.rtc, inputs.curves, s1 or math.nan), ""]
    if inputs.curves is not None:
        lines += ["## Schritt 3: Dosis-Wirkung", "", *curves_section(inputs.curves, names), ""]
        figure_curves(inputs.curves, names, reference_rate(analysis), out_dir / "phase4_kurven.png")
        lines += ["![Dosis-Wirkung](phase4_kurven.png)", ""]
    if inputs.retest is not None:
        lines += ["### Re-Test des Phase-3-Validierungsbias", "", *retest_section(inputs.retest), ""]
    if inputs.fresh is not None:
        lines += ["### Reihenfolge der Läufe: frische Prozesse", "", *fresh_section(inputs.fresh), ""]
    if inputs.sensitivity is not None:
        lines += [
            "### Sensitivität: Phase-3-Stufen gepoolt",
            "",
            *sensitivity_section(inputs.sensitivity),
            "",
        ]
    if inputs.capacity is not None:
        lines += ["## Schritt 4: Kapazität", "", *capacity_section(inputs.capacity, inputs.fleet, names), ""]
        figure_capacity(inputs.capacity, inputs.fleet, names, out_dir / "phase4_kapazitaet.png")
        lines += ["![Kapazität](phase4_kapazitaet.png)", ""]
        if inputs.wlan is not None:
            rows = wlan_rows(inputs.wlan, inputs.capacity, inputs.fleet)
            if rows:
                lines += ["### WLAN mit geteilter Bandbreite", "", *wlan_section(rows), ""]
                figure_wlan(rows, out_dir / "phase4_wlan.png")
                lines += ["![WLAN](phase4_wlan.png)", ""]
    if inputs.validation is not None:
        lines += ["### Validierung mit echten Rollouts", "", *validation_section(inputs.validation), ""]
    if inputs.economics is not None:
        lines += [
            "## Schritt 5: Wirtschaftlichkeit",
            "",
            *economics_section(
                inputs.economics,
                inputs.economics_config,
                [name for name, strategy in inputs.fleet.strategies.items() if not strategy.batches],
            ),
            "",
        ]
        if inputs.capacity is not None:
            figure_costs(
                inputs.economics,
                inputs.capacity,
                inputs.fleet,
                inputs.economics_config,
                names,
                out_dir / "phase4_kosten.png",
            )
            lines += ["![Kosten](phase4_kosten.png)", ""]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-4 report data and figures.")
    parser.add_argument("--analysis", default="kortyn/experiments/phase4/analysis.yaml")
    parser.add_argument("--fleet-config", default="kortyn/experiments/phase4/fleet.yaml")
    parser.add_argument("--economics-config", default="kortyn/experiments/phase3/economics.yaml")
    parser.add_argument("--codec", default="results/phase4/codec.json")
    parser.add_argument("--rtc-overhead", default="results/phase4/rtc_overhead.json")
    parser.add_argument("--capacity", default="results/phase4/fleet/capacity.json")
    parser.add_argument("--wlan-capacity", default="results/phase4/fleet/capacity_wlan.json")
    parser.add_argument("--economics", default="results/phase4/economics.json")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        inputs = load_inputs(args)
        lines = build(inputs, Path(args.out_dir))
    except (ConfigError, StepError, ReportError, KeyError) as exc:
        LOG.error("%s", exc)
        return 2
    out = Path(args.out_dir) / "phase4_data.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    LOG.info("written %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
