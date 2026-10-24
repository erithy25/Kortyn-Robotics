"""Phase-2 report: statistics, selection rule, hypothesis check and the full report on synthetic results."""

from __future__ import annotations

import json

import numpy as np
import pytest

from kortyn import report, report_phase2
from kortyn.config import (
    EXPERIMENTS_DIR,
    GuardConfig,
    RequestConfig,
    SmoothingConfig,
    StrategyConfig,
    load_config,
)
from kortyn.metrics import RunSummary, append_jsonl, summarize_run
from kortyn.report import RunData
from kortyn.report_phase2 import (
    AgeEvidence,
    Rank,
    Slope,
    compare,
    display_name,
    estimator_quality,
    gap_recovered,
    pooled_slope,
    restrict,
    select_strategies,
    style_of,
    trigger_text,
)
from tests.test_metrics import episode, request

GUARD = GuardConfig(mode="hold", max_plan_age_s=1.2)
STRATEGIES = {
    "baseline": StrategyConfig(description="Phase 1"),
    "schwelle10": StrategyConfig(request=RequestConfig(refill_threshold=10), guard=GUARD),
    "adaptiv_inpaint": StrategyConfig(
        request=RequestConfig(mode="adaptive", target_age_s=0.7),
        smoothing=SmoothingConfig(mode="inpaint", schedule="exp"),
        guard=GUARD,
    ),
}
NETWORK = {
    "gut": {"rtt_ms": 0.0, "rtt_jitter_ms": 0.0},
    "schlecht": {"rtt_ms": 80.0, "rtt_jitter_ms": 40.0},
    "sehr_schlecht": {"rtt_ms": 150.0, "rtt_jitter_ms": 80.0},
}


def summary_of(success: dict[int, bool], reward: dict[int, float] | None = None) -> RunSummary:
    episodes = [
        episode(seed, success=ok, max_coverage=0.95 * (reward or {}).get(seed, 1.0 if ok else 0.5))
        for seed, ok in success.items()
    ]
    return summarize_run(episodes, [])


# --------------------------------------------------------------------------------------------- units
def test_selection_rule_prefers_fewer_requests_within_the_margin():
    ranks = [
        Rank("a", {}, score=0.050, requests_per_s=2.0),
        Rank("b", {}, score=0.045, requests_per_s=1.0),
        Rank("c", {}, score=0.020, requests_per_s=0.5),
    ]
    assert select_strategies(ranks) == ["b", "a"]  # b ties with a (< 0.01 apart) and asks less often
    assert select_strategies(ranks, count=3) == ["b", "a", "c"]
    assert select_strategies(ranks, margin=0.0) == ["a", "b"]
    assert select_strategies([]) == []


def test_compare_pairs_on_common_seeds():
    base = summary_of({s: s % 2 == 0 for s in range(20)})  # 10 of 20
    better = summary_of({s: s < 18 for s in range(20)} | {99: True})  # seed 99 is not paired
    c = compare(base, better)
    assert c is not None
    assert c.pairs == 20
    assert c.base_rate == pytest.approx(0.5)
    assert c.rate == pytest.approx(0.9)
    assert (c.only_base, c.only_strategy) == (1, 9)  # seed 18 only in base; odd seeds < 18 only in better
    assert c.mcnemar_p == pytest.approx(0.021484375)
    assert c.reward_delta == pytest.approx((9 * 0.5 - 0.5) / 20)  # rewards: success 1.0, failure 0.5
    assert c.reward_p is not None
    assert c.reward_p < 0.05
    assert compare(base, summary_of({100: True})) is None


def test_gap_to_the_synchronous_reference():
    assert gap_recovered(0.4, 0.5, 0.6) == pytest.approx(0.5)
    assert gap_recovered(0.4, 0.3, 0.6) == pytest.approx(-0.5)
    assert gap_recovered(0.6, 0.7, 0.6) is None  # no gap to close
    assert gap_recovered(0.4, 0.5, None) is None


def test_restrict_recomputes_the_summary_on_a_seed_subset():
    episodes = [episode(seed, success=seed < 1003) for seed in range(1000, 1010)]
    run = RunData("gut", episodes, [], summarize_run(episodes, []))
    subset = restrict(run, [1000, 1001, 1002, 1003])
    assert subset is not None
    assert subset.summary.episodes == 4
    assert subset.summary.success_rate == pytest.approx(0.75)
    assert restrict(run, [5]) is None


def test_strategy_names_and_styles():
    settings = {name: strategy.to_dict() for name, strategy in STRATEGIES.items()}
    assert display_name("schwelle10", settings["schwelle10"]) == "Schwelle 10"
    assert display_name("adaptiv_inpaint", settings["adaptiv_inpaint"]) == "Adaptiv 0.7 s · Inpainting"
    assert style_of("adaptiv_inpaint", settings["adaptiv_inpaint"]).inpaint
    assert style_of("baseline", settings["baseline"]).family == "Baseline"
    assert style_of("schwelle10", settings["schwelle10"]).color != style_of("baseline", {}).color


def test_estimator_quality_and_triggers():
    records = [request(1000, i, 1.0 * i, 400.0, 300.0) for i in range(4)]
    for record, median in zip(records, [380.0, 390.0, 400.0, 300.0], strict=True):
        record.latency_median_ms = median
        record.latency_spread_ms = 20.0
        record.trigger = "freshness"
    records[3].trigger = "deadline"
    error, coverage = estimator_quality(records, z=1.645)
    assert error == pytest.approx(15.0)  # |400 - [380, 390, 400, 300]| -> median of 20, 10, 0, 100
    assert coverage == pytest.approx(0.75)  # 400 > 300 + 1.645 * 20 for the last one
    assert estimator_quality([], 1.645) == (None, None)
    assert trigger_text(records) == "Frische 75 % · Deadline 25 %"
    assert trigger_text([]) == "–"


def run_summary(age_s: float, rewards: dict[int, float]) -> RunSummary:
    episodes = []
    for seed, value in rewards.items():
        record = episode(seed, success=value >= 1.0, max_coverage=0.95 * value, steps=50)
        record.ticks.action_age = [round(age_s * 10)] * 50
        episodes.append(record)
    return summarize_run(episodes, [])


def test_pooled_slope_recovers_a_known_relation_and_its_uncertainty():
    seeds = range(1000, 1030)
    exact = [run_summary(age, {s: 1.0 - 0.2 * age for s in seeds}) for age in (0.4, 0.7, 1.0)]
    slope = pooled_slope([exact])
    assert slope is not None
    assert slope.value == pytest.approx(-0.2, abs=1e-9)
    assert slope.ci == pytest.approx((-0.2, -0.2), abs=1e-9)
    # two groups with different levels but the same slope: fixed effects remove the level
    shifted = [run_summary(age, {s: 0.8 - 0.2 * age for s in seeds}) for age in (0.5, 0.9)]
    assert pooled_slope([exact, shifted]).value == pytest.approx(-0.2, abs=1e-9)
    rng = np.random.default_rng(0)
    noisy = [
        run_summary(age, {s: float(np.clip(0.95 - 0.2 * age + rng.normal(0, 0.05), 0, 1)) for s in seeds})
        for age in (0.4, 0.7, 1.0)
    ]
    estimate = pooled_slope([noisy])
    assert estimate is not None
    assert estimate.ci[0] < estimate.value < estimate.ci[1]
    assert pooled_slope([[exact[0]]]) is None  # one run: no slope
    flat = [run_summary(0.7, {s: 0.9 for s in seeds}) for _ in range(3)]
    assert pooled_slope([flat]) is None  # no spread in plan age


def slope(value: float, low: float, high: float, seed: int) -> Slope:
    rng = np.random.default_rng(seed)  # independent bootstrap distributions for the two data sets
    return Slope(value, (low, high), rng.normal(value, (high - low) / 3.92, size=2000))


@pytest.mark.parametrize(
    ("between", "within", "verdict"),
    [
        (slope(-0.4, -0.6, -0.2, 1), slope(0.3, -0.1, 0.7, 2), "refuted"),
        (slope(-0.4, -0.5, -0.3, 1), slope(-0.1, -0.15, -0.05, 2), "weak"),
        (slope(-0.4, -0.6, -0.2, 1), slope(-0.35, -0.6, -0.1, 2), "confirmed"),
        (slope(-0.4, -0.6, -0.2, 1), slope(-0.2, -0.6, 0.2, 2), "inconclusive"),
        (slope(-0.1, -0.3, 0.1, 1), slope(0.3, 0.1, 0.5, 2), ""),  # phase 1 itself shows no effect
    ],
)
def test_age_hypothesis_verdicts(between, within, verdict):
    evidence = AgeEvidence(between, within, {"gut": within.value}, {"baseline": 8.0, "schwelle10": 4.0})
    assert evidence.verdict == verdict
    section = report_phase2.hypothesis_section(evidence)
    assert bool(section) == bool(verdict)
    if verdict == "refuted":
        assert any("Je Plan werden nur noch 4 statt 8 Aktionen" in line for line in section)


# --------------------------------------------------------------------------------------------- full report
def write_run(
    directory, label, scenario, strategy, seeds, success, age_ticks, requests_per_episode, mode="realtime"
):
    episodes = []
    requests = []
    for seed in seeds:
        ok = success(seed)
        record = episode(
            seed,
            success=ok,
            steps=100,
            run=label,
            scenario=scenario,
            mode=mode,
            strategy=strategy,
            max_coverage=0.95 if ok else 0.6,
            n_requests=requests_per_episode,
        )
        record.ticks.action_age = [age_ticks] * 100
        if strategy is not None:
            record.ticks.plan_age_ms = [100.0 * age_ticks + 3.0] * 100
            record.ticks.emergency = [0] * 100
            record.brain_load = {
                "busy_ms": 300.0 * requests_per_episode,
                "window_ms": 10_000.0,
                "requests": 1.0,
            }
        episodes.append(record)
        for index in range(requests_per_episode):
            item = request(
                seed, index + 1, 0.5 * index, 400.0, 300.0, phase="sync" if mode == "sync" else "realtime"
            )
            item.server_queue_ms = 0.4
            item.latency_median_ms = 390.0
            item.latency_spread_ms = 20.0
            item.trigger = "threshold" if strategy != "adaptiv_inpaint" else "freshness"
            if strategy == "adaptiv_inpaint":
                item.guide_frozen, item.guide_soft = 5, 4
            requests.append(item)
    append_jsonl(directory / label / "episodes.jsonl", [e.to_json() for e in episodes])
    append_jsonl(directory / label / "requests.jsonl", [r.to_json() for r in requests])


def write_meta(directory, runs):
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"profile": "kalibriert", "brain": {"sampler": "DDIM-4"}, "runs": runs}
    (directory / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def results(tmp_path):
    """Phase 1, stage 1 and stage 2 results in the layout of the real experiments."""
    root = tmp_path / "results"
    seeds20 = range(1000, 1020)
    seeds50 = range(1000, 1050)
    # phase 1: baseline behaviour, plan age grows with the network, quality drops
    phase1 = root / "kalibriert"
    phase1_runs = {
        "referenz_sync": ("referenz_sync", 4, lambda s: s % 10 < 7, "sync"),
        "gut": ("gut", 7, lambda s: s % 10 < 6, "realtime"),
        "gut_wiederholung": ("gut", 7, lambda s: s % 10 < 5, "realtime"),
        "schlecht": ("schlecht", 9, lambda s: s % 10 < 5, "realtime"),
        "sehr_schlecht": ("sehr_schlecht", 10, lambda s: s % 10 < 4, "realtime"),
    }
    meta = {}
    for label, (scenario, age, success, mode) in phase1_runs.items():
        write_run(phase1, label, scenario, None, seeds50, success, age, 6, mode=mode)
        meta[label] = {"scenario": scenario, "mode": mode, "network": NETWORK.get(scenario, {})}
    write_meta(phase1, meta)
    # stage 1: fresh baseline (same outcomes as phase 1), a fresher but worse strategy, a better one
    screening = root / "phase2" / "screening" / "kalibriert"
    outcomes = {
        "baseline": (lambda age: age, lambda base: base, 12),
        "schwelle10": (lambda age: age - 2, lambda base: lambda s: base(s) and s % 10 < 3, 25),
        "adaptiv_inpaint": (lambda age: age - 1, lambda base: lambda s: base(s) or s % 10 == 8, 14),
    }
    meta = {}
    for scenario in ("gut", "sehr_schlecht"):
        _, age, success, _ = phase1_runs[scenario]
        for strategy, (age_of, success_of, n_requests) in outcomes.items():
            label = f"{strategy}__{scenario}"
            write_run(
                screening, label, scenario, strategy, seeds20, success_of(success), age_of(age), n_requests
            )
            meta[label] = {
                "scenario": scenario,
                "mode": "realtime",
                "network": NETWORK[scenario],
                "strategy": strategy,
                "strategy_settings": STRATEGIES[strategy].to_dict(),
            }
    write_meta(screening, meta)
    # stage 2: the two selected strategies on 50 seeds, baseline = the phase-1 runs
    confirmation = root / "phase2" / "confirmation" / "kalibriert"
    meta = {}
    for scenario in ("gut", "schlecht", "sehr_schlecht"):
        _, age, success, _ = phase1_runs[scenario]
        for strategy in ("adaptiv_inpaint", "schwelle10"):
            age_of, success_of, n_requests = outcomes[strategy]
            label = f"{strategy}__{scenario}"
            write_run(
                confirmation, label, scenario, strategy, seeds50, success_of(success), age_of(age), n_requests
            )
            meta[label] = {
                "scenario": scenario,
                "mode": "realtime",
                "network": NETWORK[scenario],
                "strategy": strategy,
                "strategy_settings": STRATEGIES[strategy].to_dict(),
            }
    write_meta(confirmation, meta)
    return root


def test_phase_two_report_from_all_stages(tmp_path, results):
    reports = tmp_path / "reports"
    args = ["--results-dir", str(results), "--reports-dir", str(reports), "--log-level", "WARNING"]
    assert report.main([*args, "--phase", "2"]) == 0
    text = (reports / "phase2.md").read_text(encoding="utf-8")
    for section in (
        "## Kurzfassung (Stufe 2, 50 Seeds)",
        "## Auf einen Blick",
        "## Ist das Planalter der Hebel?",
        "## Was gebaut wurde",
        "### Reproduktion der Baseline (Abnahmekriterium 2)",
        "✅ **Reproduziert",
        "### Auswahl für Stufe 2",
        "## Stufe 2 · Bestätigung (50 Seeds)",
        "## Grenzen",
        "## Definitionen",
    ):
        assert section in text, section
    # the better strategy wins the pre-registered rule and is reported against the phase-1 baseline
    assert "- **Beste Strategie in Stufe 2** (dieselbe Regel über alle 3 Szenarien" in text
    assert (
        "Adaptiv 0.7 s · Inpainting (`adaptiv_inpaint`); ebenfalls in Stufe 2 gemessen: Schwelle 10." in text
    )
    assert "Baseline (Phase-1-Lauf)" in text
    assert "| _Baseline-Wiederholung (Phase 1, identisch)_ | 50 |" in text  # run-to-run noise next to it
    assert "_gut: Phase 1 → Phase-1-Wiederholung (gleiche Sitzung, identisch)_" in text
    assert "- **Messrauschen:** Zwei identische Phase-1-Läufe („gut“)" in text
    assert "- **Lauf-zu-Lauf-Rauschen.**" in text
    assert "- **Stufe-2-Baseline aus Phase 1.**" in text
    assert "**Die Kernhypothese hält in dieser Form nicht:**" in text  # fresher but worse in this data
    assert "| Schwelle 10 | `schwelle10` |" in text
    for name in (
        "phase2_erfolg_stufe1.png",
        "phase2_erfolg_stufe2.png",
        "phase2_planalter_qualitaet_stufe1.png",
        "phase2_planalter_qualitaet_stufe2.png",
        "phase2_planalter_last.png",
    ):
        assert (reports / name).stat().st_size > 10_000, name
        assert f"]({name})" in text


def test_reproduction_verdicts():
    noise = compare(summary_of({s: s < 6 for s in range(50)}), summary_of({s: s < 5 for s in range(50)}))
    clean = report_phase2.Reproduction([], True, True, True, ("Erfolg gut", 0.8, 1.0), 4)
    assert report_phase2.reproduction_text(clean, noise)[0].startswith(
        "✅ **Reproduziert:** kein Unterschied"
    )
    within = report_phase2.Reproduction([], True, False, False, ("Max-Reward gut", 0.03, 0.12), 4)
    text = report_phase2.reproduction_text(within, noise)[0]
    assert text.startswith("✅ **Reproduziert innerhalb der Unsicherheit:** Max-Reward gut nominal p = 0.03")
    assert "Holm-Korrektur über alle 4 Reproduktionstests p = 0.12" in text
    assert "identische Wiederholung" in text
    failed = report_phase2.Reproduction([], False, False, True, ("Erfolg gut", 0.001, 0.004), 4)
    assert report_phase2.reproduction_text(failed, None)[0].startswith("⚠️ **Abweichung:**")
    assert (
        report_phase2.reproduction_text(report_phase2.Reproduction([], None, None, False, None, 0), None)
        == []
    )


def test_selection_cli_prints_the_ranking(results, capsys):
    assert report_phase2.main(["--select", "--results-dir", str(results), "--log-level", "WARNING"]) == 0
    out = capsys.readouterr().out
    assert "[kalibriert]" in out
    assert "Auswahl: adaptiv_inpaint, schwelle10" in out
    assert out.index("adaptiv_inpaint") < out.index("schwelle10")


def test_report_without_phase_two_results(tmp_path):
    empty = tmp_path / "results"
    empty.mkdir()
    args = ["--results-dir", str(empty), "--reports-dir", str(tmp_path / "r"), "--log-level", "WARNING"]
    assert report.main([*args, "--phase", "2"]) == 1
    assert report_phase2.main(args) == 1
    assert report_phase2.main([*args, "--select"]) == 1
    assert report.main(args) == 1  # neither phase


def test_phase_two_experiment_files_store_results_where_the_report_reads_them():
    screening = load_config(EXPERIMENTS_DIR / "phase2" / "screening.yaml")
    assert screening.experiment.results_dir == "results/phase2/screening"
    confirmation = load_config(EXPERIMENTS_DIR / "phase2" / "confirmation.yaml")
    assert confirmation.experiment.results_dir == "results/phase2/confirmation"
    assert confirmation.experiment.episodes == 50
    assert confirmation.experiment.seed_start == screening.experiment.seed_start == 1000
    runs = [(run.scenario, run.strategy) for suite in confirmation.experiment.suites for run in suite.runs]
    strategies = {strategy for _, strategy in runs}
    assert len(strategies) == 2
    assert strategies <= set(screening.strategies)
    assert {scenario for scenario, _ in runs} == set(report_phase2.SCENARIO_ORDER)
    assert len(runs) == 12
    assert {run.strategy for suite in screening.experiment.suites for run in suite.runs} == set(
        screening.strategies
    )
