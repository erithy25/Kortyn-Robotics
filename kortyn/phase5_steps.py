"""Phase 5 evaluation, pre-registered in ``reports/phase5_preregistration.md``.

    python -m kortyn.phase5_steps gate           # 3.3-3.4: the 0-ms gate of RTC with short guidance, controls
    python -m kortyn.phase5_steps curves         # 3.7: only after a passed gate - H1, levels, L*, capacity
    python -m kortyn.phase5_steps gpu-decision   # 5: the business rule on every file in results/phase5/gpu

**Gate:** ``rtc_kurz_0ms`` against ``sync25`` of the same block (200 pairs: every init state twice), exactly
:func:`kortyn.rtc_gate.evaluate_gate` - the code of the power analysis. Controls with the same procedure:
asynchronous without RTC against ``sync25`` (K1), RTC against asynchronous without RTC (K2), both per task,
and the reproduction of the phase-4 block A and the phase-3 ``sync25`` on seeds 1000-1099 (K3).

**Curves:** the condition ``rtc_kurz`` (0 ms from the gate block, seeds 1000-1099; 200, 500, 1000 ms from the
dose block) is fitted with the phase-4 conditions without RTC and with long guidance on the same seeds and
bootstrap multiplicities (:mod:`kortyn.dose_curve`); H1-kurz is the mean gain over the 25 budget ticks.

**GPU decision:** the capacity tool (:mod:`kortyn.capacity`) with the pre-registered settings on every
measurement file: latency capacity with p99 <= 150 ms for every numerically valid configuration and strategy;
"Ja" if one of them reaches 10 robots.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kortyn import capacity as cap
from kortyn.config import ConfigError, LiberoEnvConfig, decode
from kortyn.fleet import EpisodeLengthConfig, FleetAgentConfig
from kortyn.fleet_capacity import bin_curves, evaluate, read_runs, result_json
from kortyn.fleet_predict import DoseResponse, PredictionError
from kortyn.gpu_benchmark import GpuBenchmark, GpuBenchmarkError
from kortyn.libero_env import episode_slot
from kortyn.phase3_steps import StepError, load_episodes, paired_comparison, successes
from kortyn.phase4_steps import (
    LevelSpec,
    condition_summary,
    fit_conditions,
    libero_layout,
    load_analysis_config,
    rtc_effect,
)
from kortyn.rtc_gate import (
    GATE_ALPHA,
    GATE_CONFIDENCE,
    GATE_DRAWS,
    GATE_MARGIN,
    GateError,
    GateResult,
    evaluate_gate,
    paired_sample,
    subset,
)

LOG = logging.getLogger("kortyn.phase5_steps")

DOSE_BLOCK_RUNS = "D runs (dose block 200 / 500 / 1000 ms)"
STOP = "RTC measurement stops: the question stays open"


# --------------------------------------------------------------------------------------------- config
@dataclass
class GateSpec:
    reference: str = "results/phase5/libero/smolvla_libero_rtc/sync25"
    control: str = "results/phase5/libero/smolvla_libero_rtc/async_0ms"
    test: str = "results/phase5/libero/smolvla_libero_rtc/rtc_kurz_0ms"
    phase4_async: str = "results/phase4/libero/smolvla_libero/fest_0ms"
    phase3_sync: str = "results/phase3/libero/smolvla_libero/sync25"
    phase4_rtc: str = "results/phase4/libero/smolvla_libero_rtc/rtc_0ms"
    margin: float = GATE_MARGIN
    draws: int = GATE_DRAWS
    seed: int = 5000
    output: str = "results/phase5/gate.json"

    def __post_init__(self) -> None:
        if not 0 < self.margin < 1 or self.draws < 100:
            raise ConfigError("gate needs a margin in (0, 1) and at least 100 draws")


@dataclass
class CurvesSpec:
    seeds: list[int] = field(default_factory=lambda: [1000, 1099])  # inclusive range of the curve's seeds
    levels: list[LevelSpec] = field(default_factory=list)
    stage1: str = "results/phase4/fleet/scan_rtc.jsonl"
    gpu: str = "rtx3060ti_schaetzung_rtc"
    output: str = "results/phase5/curves.json"

    def __post_init__(self) -> None:
        if len(self.seeds) != 2 or self.seeds[0] > self.seeds[1]:
            raise ConfigError("curves.seeds must be an inclusive range [first, last]")


@dataclass
class GpuDecisionSpec:
    """The pre-registered settings of section 5 (everything but the measurement file)."""

    measurements: str = "results/phase5/gpu"
    dose_response: str = "examples/capacity/smolvla_libero_long.json"
    network: cap.NetworkSpec = field(default_factory=cap.NetworkSpec)
    wlan_mbit_s: list[float] = field(default_factory=lambda: [100.0, 200.0])
    request_bytes: int = 292133
    response_bytes: int = 1424
    agent: FleetAgentConfig = field(default_factory=FleetAgentConfig)
    episode_lengths: EpisodeLengthConfig = field(
        default_factory=lambda: EpisodeLengthConfig(
            paths=["results/phase3/libero/smolvla_libero/fest_0ms/episodes.jsonl"]
        )
    )
    latency_slo: cap.LatencySlo = field(
        default_factory=lambda: cap.LatencySlo(quantile=0.99, max_ms=150.0, robots=10)
    )
    tolerance_pp: float = 3.0
    fleet: cap.FleetSpec = field(default_factory=cap.FleetSpec)
    bootstrap: int = 200
    assumption_s1_ms: float = 220.0  # phases 3-4: RTX 3060 Ti, for the comparison only
    output: str = "results/phase5/gpu_decision.json"


@dataclass
class Phase5Config:
    experiment: str = "kortyn/experiments/phase5/libero_rtc_kurz.yaml"  # episode layout (task per seed)
    phase4_analysis: str = (
        "kortyn/experiments/phase4/analysis.yaml"  # conditions without RTC and with long RTC
    )
    gate: GateSpec = field(default_factory=GateSpec)
    curves: CurvesSpec = field(default_factory=CurvesSpec)
    gpu_decision: GpuDecisionSpec = field(default_factory=GpuDecisionSpec)


def load_phase5_config(path: str | Path) -> Phase5Config:
    import yaml

    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read the phase-5 analysis config {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    return decode(Phase5Config, data, str(source))


# --------------------------------------------------------------------------------------------- gate
def _outcomes(run: str) -> dict[int, bool]:
    return successes(load_episodes(Path(run)))


def _comparison(reference: Mapping[int, bool], test: Mapping[int, bool], seed: int) -> dict[str, Any]:
    """The phase-4 paired comparison (seed bootstrap, McNemar) of ``test - reference``."""
    return {key: value for key, value in asdict(paired_comparison(reference, test, seed=seed)).items()}


def _gate_json(result: GateResult, reference: str, test: str) -> dict[str, Any]:
    return {"reference": reference, "test": test, **result.to_json()}


def gate_document(config: Phase5Config) -> dict[str, Any]:
    spec = config.gate
    libero = libero_layout(config.experiment)

    def slot(seed: int) -> tuple[int, int]:
        task, init_state, _ = episode_slot(libero, seed)
        return task, init_state

    runs = {
        "sync25": _outcomes(spec.reference),
        "async_0ms": _outcomes(spec.control),
        "rtc_kurz_0ms": _outcomes(spec.test),
    }
    common = set(runs["sync25"]) & set(runs["async_0ms"]) & set(runs["rtc_kurz_0ms"])
    if not common:
        raise StepError("the gate runs have no seed in common")
    trimmed = {name: {s: v for s, v in outcomes.items() if s in common} for name, outcomes in runs.items()}

    def gate(reference: str, test: str, keep: Any = None, seed_offset: int = 0) -> GateResult:
        sample = paired_sample(trimmed[reference], trimmed[test], slot)
        if keep is not None:
            sample = subset(sample, keep(sample))
        return evaluate_gate(
            sample,
            margin=spec.margin,
            alpha=GATE_ALPHA,
            confidence=GATE_CONFIDENCE,
            draws=spec.draws,
            seed=spec.seed + seed_offset,
        )

    main = gate("sync25", "rtc_kurz_0ms")
    controls = {
        "K1_async_vs_sync25": gate("sync25", "async_0ms", seed_offset=1),
        "K2_rtc_vs_async": gate("async_0ms", "rtc_kurz_0ms", seed_offset=2),
    }
    per_task: dict[str, dict[str, Any]] = {}
    tasks = sorted({slot(s)[0] for s in common})
    for index, task in enumerate(tasks):

        def only(sample: Any, task: int = task) -> Any:
            return sample.strata == task

        per_task[str(task)] = {
            "gate": _gate_json(
                gate("sync25", "rtc_kurz_0ms", only, 10 + 3 * index), "sync25", "rtc_kurz_0ms"
            ),
            "K1": _gate_json(gate("sync25", "async_0ms", only, 11 + 3 * index), "sync25", "async_0ms"),
            "K2": _gate_json(
                gate("async_0ms", "rtc_kurz_0ms", only, 12 + 3 * index), "async_0ms", "rtc_kurz_0ms"
            ),
        }
    first = sorted(s for s in common if (s - libero.seed_base) < libero.episodes)
    reproduction = {
        "async_0ms_vs_phase4_block_a": _comparison(
            _outcomes(spec.phase4_async), {s: trimmed["async_0ms"][s] for s in first}, spec.seed + 20
        ),
        "sync25_vs_phase3": _comparison(
            _outcomes(spec.phase3_sync), {s: trimmed["sync25"][s] for s in first}, spec.seed + 21
        ),
        "rtc_kurz_vs_phase4_rtc_long_at_0ms": _comparison(
            _outcomes(spec.phase4_rtc), {s: trimmed["rtc_kurz_0ms"][s] for s in first}, spec.seed + 22
        ),
    }
    rates = {
        name: {
            "rate": float(np.mean([float(v) for v in outcomes.values()])),
            "episodes": len(outcomes),
            "tasks": {
                str(task): float(np.mean([float(v) for s, v in outcomes.items() if slot(s)[0] == task]))
                for task in tasks
            },
        }
        for name, outcomes in trimmed.items()
    }
    return {
        "pre_registration": "reports/phase5_preregistration.md, sections 3.3-3.4",
        "runs": {"sync25": spec.reference, "async_0ms": spec.control, "rtc_kurz_0ms": spec.test},
        "rates": rates,
        "gate": _gate_json(main, "sync25", "rtc_kurz_0ms"),
        "controls": {name: _gate_json(result, *name_pair(name)) for name, result in controls.items()},
        "per_task": per_task,
        "reproduction": reproduction,
        "decision": DOSE_BLOCK_RUNS if main.passed else STOP,
    }


def name_pair(control: str) -> tuple[str, str]:
    return ("sync25", "async_0ms") if control.startswith("K1") else ("async_0ms", "rtc_kurz_0ms")


# --------------------------------------------------------------------------------------------- curves
def load_condition_on(
    levels: Sequence[LevelSpec], libero: LiberoEnvConfig, first: int, last: int
) -> DoseResponse:
    """A condition on the seeds ``first..last`` (the gate's 0-ms run holds 200 seeds, the curve 100)."""
    if len(levels) < 2:
        raise StepError("a condition needs at least two levels")
    ordered = sorted(levels, key=lambda level: level.latency_ms)
    per_level = []
    for level in ordered:
        outcomes = {s: v for s, v in _outcomes(level.run).items() if first <= s <= last}
        per_level.append(outcomes)
    seeds = sorted(set.intersection(*(set(level) for level in per_level)))
    missing = sorted(set.union(*(set(level) for level in per_level)) - set(seeds))
    if missing or not seeds:
        raise StepError(f"seeds {missing[:5]} are missing in some levels (or no seed in {first}..{last})")
    return DoseResponse(
        levels_ms=np.array([level.latency_ms for level in ordered], dtype=float),
        seeds=tuple(seeds),
        groups=tuple(episode_slot(libero, seed)[0] for seed in seeds),
        outcomes=np.array([[float(level[s]) for s in seeds] for level in per_level]),
    )


def level_comparisons(without: DoseResponse, with_rtc: DoseResponse, seed: int) -> list[dict[str, Any]]:
    """Paired RTC - without at every latency both conditions measured, pooled and per task."""
    rows = []
    for index, latency in enumerate(with_rtc.levels_ms):
        matches = np.flatnonzero(np.isclose(without.levels_ms, latency))
        if matches.size == 0:
            continue
        a = {s: bool(without.outcomes[matches[0], i]) for i, s in enumerate(without.seeds)}
        b = {s: bool(with_rtc.outcomes[index, i]) for i, s in enumerate(with_rtc.seeds)}
        row: dict[str, Any] = {"latency_ms": float(latency), **_comparison(a, b, seed + index)}
        groups = dict(zip(with_rtc.seeds, with_rtc.groups, strict=True))
        row["tasks"] = {
            str(task): _comparison(
                {s: v for s, v in a.items() if groups.get(s) == task},
                {s: v for s, v in b.items() if groups.get(s) == task},
                seed + 100 + index,
            )
            for task in sorted(set(with_rtc.groups))
        }
        rows.append(row)
    return rows


def curves_document(config: Phase5Config) -> dict[str, Any]:
    gate_path = Path(config.gate.output)
    if not gate_path.exists():
        raise StepError(f"{gate_path} does not exist: run the gate first")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate["gate"]["passed"]:
        raise StepError("the gate did not pass: the dose block is not evaluated (pre-registration 3.3)")
    phase4 = load_analysis_config(config.phase4_analysis)
    libero = libero_layout(config.experiment)
    first, last = config.curves.seeds
    doses = {
        "ohne_rtc": load_condition_on(phase4.conditions[phase4.without], libero, first, last),
        "mit_rtc": load_condition_on(phase4.conditions[phase4.with_rtc], libero, first, last),
        "rtc_kurz": load_condition_on(config.curves.levels, libero, first, last),
    }
    fits = fit_conditions(doses, phase4.spec, phase4.bootstrap, phase4.seed)
    stage1 = read_runs(Path(config.curves.stage1))
    bins = max((run.fleet[0].histogram.size for run in stage1 if run.fleet), default=1)
    capacity = evaluate(
        stage1,
        {"rtc_kurz": bin_curves(fits["rtc_kurz"], bins)},
        {config.curves.gpu: "rtc_kurz"},
        phase4.tolerance,
    )
    return {
        "pre_registration": "reports/phase5_preregistration.md, section 3.7",
        "spec": asdict(phase4.spec),
        "bootstrap": phase4.bootstrap,
        "seed": phase4.seed,
        "tolerance": phase4.tolerance,
        "seeds": [first, last],
        "conditions": {name: condition_summary(doses[name], fits[name], phase4.tolerance) for name in doses},
        "h1_kurz": rtc_effect(fits["ohne_rtc"], fits["rtc_kurz"], phase4.tolerance),
        "phase4_long": rtc_effect(fits["ohne_rtc"], fits["mit_rtc"], phase4.tolerance),
        "short_vs_long": rtc_effect(fits["mit_rtc"], fits["rtc_kurz"], phase4.tolerance),
        "levels": level_comparisons(doses["ohne_rtc"], doses["rtc_kurz"], 6000),
        "capacity": result_json(capacity, phase4.tolerance),
    }


# --------------------------------------------------------------------------------------------- GPU decision
def decision_config(
    spec: GpuDecisionSpec, measurement: Path, wlan_mbit_s: float | None
) -> cap.CapacityConfig:
    network = cap.NetworkSpec(
        scenario=spec.network.scenario,
        margin_ms=spec.network.margin_ms,
        wlan_mbit_s=wlan_mbit_s,
        request_bytes=spec.request_bytes if wlan_mbit_s is not None else 0,
        response_bytes=spec.response_bytes if wlan_mbit_s is not None else 0,
    )
    return cap.CapacityConfig(
        dose_response=spec.dose_response,
        gpu=cap.GpuSpec(measurement=str(measurement), configuration=cap.ALL_VALID, label=measurement.stem),
        network=network,
        agent=spec.agent,
        episode_lengths=spec.episode_lengths,
        criteria=cap.CriteriaSpec(tolerance_pp=spec.tolerance_pp, latency_slo=spec.latency_slo),
        fleet=spec.fleet,
        bootstrap=spec.bootstrap,
    )


def _row_at(result: Mapping[str, Any], robots: int) -> Mapping[str, Any] | None:
    return next((row for row in result["rows"] if row["robots"] == robots), None)


def file_decision(spec: GpuDecisionSpec, path: Path) -> dict[str, Any]:
    """Verdict of one measurement file (section 5)."""
    try:
        benchmark = GpuBenchmark.from_file(path)
    except GpuBenchmarkError as exc:
        return {"file": str(path), "valid": False, "reason": str(exc), "verdict": "kein Urteil"}
    entry: dict[str, Any] = {
        "file": str(path),
        "gpu": benchmark.gpu,
        "platform": benchmark.platform,
        "created_utc": benchmark.created_utc,
        "configurations": [
            {
                "name": c.name,
                "precision": c.precision,
                "compiled": c.compiled,
                "status": c.status,
                "numerics_valid": c.numerics_valid,
                "numerics_ratio": c.numerics_ratio,
                "repeats_at_batch_1": c.repeats_at_one(),
                "median_ms_batch_1": c.median_ms(1),
                "largest_batch": c.measured[-1].batch_size if c.measured else None,
                "valid": c.valid(),
            }
            for c in benchmark.configurations
        ],
    }
    if not benchmark.measurement_valid():
        reason = "not measured on a GPU" if not benchmark.is_gpu else "no valid configuration"
        return {**entry, "valid": False, "reason": reason, "verdict": "kein Urteil"}
    office = cap.run_tool(decision_config(spec, path, None))
    target = spec.latency_slo.robots or 10
    k150 = {f"{r['gpu']}|{r['strategy']}": r["latency_capacity"] for r in office["results"]}
    best = max(office["results"], key=lambda r: (r["latency_capacity"] or 0, r["capacity"]))
    secondary: dict[str, Any] = {
        "capacity_phase4_criterion": {
            f"{r['gpu']}|{r['strategy']}": r["capacity"] for r in office["results"]
        },
        "best_capacity_phase4_criterion": office["best"],
        "at_target": None,
        "at_latency_capacity": None,
        "wlan": {},
        "batch_1_vs_assumption": {
            c["name"]: None
            if c["median_ms_batch_1"] is None
            else c["median_ms_batch_1"] / spec.assumption_s1_ms
            for c in entry["configurations"]
        },
    }
    row = _row_at(best, target)
    if row is not None:
        secondary["at_target"] = {
            "robots": target,
            "mean_robot": row["fleet_mean"],
            "worst_robot": row["worst"],
            "latency_p99_ms": row["latency_p99_ms"],
        }
    if best["latency_capacity"]:
        row = _row_at(best, int(best["latency_capacity"]))
        if row is not None:
            secondary["at_latency_capacity"] = {
                "robots": row["robots"],
                "mean_robot": row["fleet_mean"],
                "worst_robot": row["worst"],
                "latency_p99_ms": row["latency_p99_ms"],
            }
    for rate in spec.wlan_mbit_s:
        shared = cap.run_tool(decision_config(spec, path, rate))
        secondary["wlan"][f"{rate:g}"] = {
            f"{r['gpu']}|{r['strategy']}": r["latency_capacity"] for r in shared["results"]
        }
    verdict = "Ja" if (best["latency_capacity"] or 0) >= target else "Nein"
    return {
        **entry,
        "valid": True,
        "latency_capacity": k150,
        "best": {
            "gpu": best["gpu"],
            "strategy": best["strategy"],
            "latency_capacity": best["latency_capacity"],
        },
        "verdict": verdict,
        "secondary": secondary,
        "office_wifi_details": office,
    }


def gpu_decision_document(config: Phase5Config) -> dict[str, Any]:
    spec = config.gpu_decision
    folder = Path(spec.measurements)
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    decisions = [file_decision(spec, path) for path in files]
    valid = [d for d in decisions if d["valid"]]
    if not valid:
        overall = "kein Urteil (keine gültige Messdatei)"
    else:
        overall = "Ja" if any(d["verdict"] == "Ja" for d in valid) else "Nein"
    return {
        "pre_registration": "reports/phase5_preregistration.md, section 5",
        "rule": (
            "Ja if a numerically valid configuration and a strategy keep the pooled "
            f"p{100 * spec.latency_slo.quantile:g} end-to-end latency <= {spec.latency_slo.max_ms:g} ms "
            f"for {spec.latency_slo.robots} robots and every smaller fleet ({spec.network.scenario})"
        ),
        "files": decisions,
        "verdict": overall,
        "consequence": {
            "Ja": "the business idea gets a second review",
            "Nein": "the business idea is finally rejected",
        }.get(overall, "no decision until a valid measurement file is in results/phase5/gpu"),
    }


# --------------------------------------------------------------------------------------------- CLI
def _write(path: str, document: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    LOG.info("written %s", out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-5 evaluation (pre-registered).")
    parser.add_argument("step", choices=["gate", "curves", "gpu-decision"])
    parser.add_argument("--config", default="kortyn/experiments/phase5/analysis.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        config = load_phase5_config(args.config)
        if args.step == "gate":
            document = gate_document(config)
            _write(config.gate.output, document)
            g = document["gate"]
            LOG.info(
                "gate: rtc_kurz %.1f %% vs sync25 %.1f %%, %+.1f pp [%+.1f; %+.1f], McNemar p %.3f: %s -> %s",
                100 * g["test_rate"],
                100 * g["reference_rate"],
                100 * g["difference"],
                100 * g["interval"][0],
                100 * g["interval"][1],
                g["p_value"],
                g["verdict"],
                document["decision"],
            )
        elif args.step == "curves":
            document = curves_document(config)
            _write(config.curves.output, document)
            h1 = document["h1_kurz"]["h1"]
            LOG.info(
                "H1-kurz: %+.1f pp [%+.1f; %+.1f]: %s",
                100 * h1["gain"],
                100 * h1["interval"][0],
                100 * h1["interval"][1],
                h1["verdict"],
            )
        else:
            document = gpu_decision_document(config)
            _write(config.gpu_decision.output, document)
            for decision in document["files"]:
                LOG.info("%s (%s): %s", decision["file"], decision.get("gpu", "?"), decision["verdict"])
            LOG.info("GPU decision: %s - %s", document["verdict"], document["consequence"])
    except (ConfigError, StepError, PredictionError, GateError, cap.CapacityError) as exc:
        LOG.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
