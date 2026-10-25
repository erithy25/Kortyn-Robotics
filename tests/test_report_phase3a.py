"""Statistics and comparisons of the phase-3a report."""

from __future__ import annotations

import math

import pytest

from kortyn.metrics import EpisodeRecord, TickSeries
from kortyn.report_phase3a import (
    Interval,
    Run,
    determinism_check,
    estimate,
    mde_paired_binary,
    mde_paired_mean,
    newcombe_difference,
    pair_noise,
    paired_delta,
    pooled,
    replay_match,
    z_quantile,
)


def episode(
    seed, success=False, coverage=0.5, starved=(0, 0, 0, 0), ages=(100.0, 200.0, 300.0, 400.0), **extra
):
    ticks = TickSeries(
        queue_len=[5] * len(starved),
        starved=list(starved),
        lateness_ms=[0.0] * len(starved),
        action_age=[1] * len(starved),
        pending_request=[-1] * len(starved),
        plan_age_ms=list(ages),
        emergency=list(starved),
    )
    return EpisodeRecord(
        run="r", profile="p", scenario="s", mode="realtime", seed=seed, attempt=0, success=success,
        steps=len(starved), fps=10.0, wall_duration_s=0.4, max_coverage=coverage, sum_reward=1.0,
        n_requests=1, n_failed_requests=0, starved_ticks=sum(starved), starvation_events=0,
        starved_ticks_compute=0, starved_ticks_network=0, starved_ticks_none_pending=0, tick_overruns=0,
        max_tick_lateness_ms=0.0, valid=True, ticks=ticks, **extra,
    )  # fmt: skip


def run(episodes) -> Run:
    return Run("r", {e.seed: e for e in episodes}, [], 0)


def test_mde_for_paired_success_inverts_mcnemars_sample_size_formula():
    za, zb = z_quantile(0.975), z_quantile(0.8)
    for n, discordance in ((50, 0.4), (100, 0.3), (200, 0.26)):
        d = mde_paired_binary(n, discordance)
        assert 0 < d <= discordance
        required = (za * math.sqrt(discordance) + zb * math.sqrt(discordance - d * d)) ** 2 / d**2
        assert required == pytest.approx(n, rel=1e-6)
    assert mde_paired_binary(100, 0.3) < mde_paired_binary(50, 0.3)
    assert math.isnan(mde_paired_binary(3, 0.05))  # too few seeds to detect anything
    assert math.isnan(mde_paired_binary(100, 0.0))


def test_mde_for_paired_means():
    assert mde_paired_mean(100, 0.15) == pytest.approx((1.959964 + 0.841621) * 0.015, rel=1e-5)
    assert mde_paired_mean(200, 0.15) == pytest.approx(mde_paired_mean(100, 0.15) / math.sqrt(2))
    assert math.isnan(mde_paired_mean(1, 0.1))


def test_newcombe_interval_matches_the_published_example():
    # Newcombe (1998), Stat Med 17:873, Table II, method 10: 56/70 - 48/80
    interval = newcombe_difference(56, 70, 48, 80)
    assert interval.value == pytest.approx(0.2)
    assert interval.low == pytest.approx(0.0524, abs=5e-4)
    assert interval.high == pytest.approx(0.3339, abs=5e-4)
    assert Interval(0.0, -0.1, 0.1).contains(0.0)
    assert not Interval(0.2, 0.05, 0.3).contains(0.0)


def test_pooled_metrics_and_estimates():
    episodes = [
        episode(1, success=True, coverage=0.95, starved=(0, 1, 1, 0)),
        episode(2, success=False, coverage=0.475, ages=(100.0, 100.0, -1.0, -1.0)),
    ]
    assert pooled("success", episodes) == 0.5
    assert pooled("reward", episodes) == pytest.approx((1.0 + 0.5) / 2)
    assert pooled("starvation", episodes) == pytest.approx(2 / 8)
    # plan age pooled over executed plan actions (held ticks, -1, excluded)
    assert pooled("plan_age", episodes) == pytest.approx((100 + 200 + 300 + 400 + 100 + 100) / 6 / 1000)
    success = estimate("success", episodes)
    assert success.low < 0.5 < success.high
    meta = episode(3, coverage=0.0, max_step_reward=7.5)  # Meta-World: reward from the step rewards
    assert pooled("reward", [meta]) == 7.5


def test_paired_delta_is_exact_for_identical_runs_and_detects_consistent_shifts():
    a = run([episode(s, success=s % 2 == 0, coverage=0.3 + 0.005 * s) for s in range(40)])
    same = paired_delta(a, a, "success")
    assert same is not None
    assert (same.interval.value, same.interval.low, same.interval.high, same.p) == (0.0, 0.0, 0.0, 1.0)
    better = run([episode(s, success=True, coverage=0.395 + 0.005 * s) for s in range(40)])  # +0.1 reward
    delta = paired_delta(a, better, "success")
    assert delta.interval.value == pytest.approx(0.5)
    assert delta.interval.low > 0
    assert delta.p < 0.001
    reward = paired_delta(a, better, "reward")
    assert reward.interval.value == pytest.approx(0.1)
    assert reward.p < 0.001
    disjoint = run([episode(100)])
    assert paired_delta(a, disjoint, "success") is None


def test_pair_noise_counts_discordant_seeds_and_identical_episodes():
    a = run([episode(s, success=s < 5) for s in range(10)])
    b = run([episode(s, success=s < 3 or s == 9) for s in range(10)])
    noise = pair_noise("x", a, b)
    assert noise is not None
    assert noise.discordance == pytest.approx(0.3)
    assert noise.identical == 10  # same length, coverage and starvation, just other successes


def test_replay_match_categories():
    real = run([episode(1, coverage=0.9), episode(2, coverage=0.9), episode(3, coverage=0.9)])
    replay = run(
        [
            episode(1, coverage=0.9),
            episode(2, coverage=0.9 + 3e-16),
            episode(3, coverage=0.9, starved=(1, 0, 0, 0)),
        ]
    )
    match = replay_match(real, replay)
    assert (match.episodes, match.identical, match.same_outcome, match.diverged) == (3, 1, 1, 1)


def test_determinism_check_ignores_machine_dependent_diagnostics():
    first = run([episode(1, virtual={"events": 10, "compute_s": 1.0, "cache_hits": 0, "inference_ms": 3.0})])
    second = run([episode(1, virtual={"events": 10, "compute_s": 9.0, "cache_hits": 4, "inference_ms": 1.0})])
    assert determinism_check(first, second) == (1, 1)
    third = run([episode(1, virtual={"events": 11, "compute_s": 1.0, "cache_hits": 0, "inference_ms": 3.0})])
    assert determinism_check(first, third) == (0, 1)


def test_difference_of_success_drops_between_tasks():
    from kortyn.report_phase3a import delta_difference, plan_period_ms

    ref_a = run([episode(s, success=True) for s in range(30)])
    cond_a = run([episode(s, success=s % 10 != 0) for s in range(30)])  # static task: -10 pp
    ref_b = run([episode(s, success=True) for s in range(30)])
    cond_b = run([episode(s, success=s % 2 == 0) for s in range(30)])  # dynamic task: -50 pp
    interval = delta_difference(ref_a, cond_a, ref_b, cond_b)
    assert interval is not None
    assert interval.value == pytest.approx(-0.4)
    assert interval.high < 0
    assert delta_difference(ref_a, cond_a, ref_b, run([episode(99)])) is None
    assert plan_period_ms({"agent": {"actions_per_chunk": 15, "refill_threshold": 6}}, 10.0) == 800.0
    assert plan_period_ms({"agent": {"actions_per_chunk": 50, "refill_threshold": 24}}, 80.0) == 312.5


def test_validation_cases_collect_every_source_and_the_verdict_flags_shifted_runs(tmp_path):
    import json

    from kortyn.report_phase3a import load_validation, verdict

    def write(directory, episodes):
        directory.mkdir(parents=True)
        (directory / "episodes.jsonl").write_text("".join(json.dumps(e.to_json()) + "\n" for e in episodes))
        (directory / "requests.jsonl").write_text("")

    seeds = range(1000, 1030)
    real = [episode(s, success=s % 3 == 0, coverage=0.5 + 0.01 * (s % 7)) for s in seeds]
    root = tmp_path / "results"
    model_root = root / "phase3a" / "validation_model" / "kalibriert"
    write(root / "kalibriert" / "gut", real)
    write(root / "kalibriert" / "gut_wiederholung", real)
    write(model_root / "modell__gut", real)  # the pooled model serves both real-time runs of "gut"
    write(model_root / "episode__gut", real)
    # episode-calibrated counterpart of the repetition: every episode starves half of its ticks
    write(model_root / "episode__gut_wiederholung", [episode(s, starved=(1, 1, 0, 0)) for s in seeds])
    write(root / "phase3a" / "validation_replay" / "kalibriert" / "replay__gut", real)

    cases = {case.scenario: case for case in load_validation(root)}
    assert set(cases) == {"gut", "gut_wiederholung"}
    gut, repetition = cases["gut"], cases["gut_wiederholung"]
    assert gut.model is not None
    assert gut.episode is not None
    assert gut.replay is not None
    assert repetition.model is not None
    assert repetition.replay is None
    assert repetition.model.episodes == gut.model.episodes
    assert set(gut.deltas("episode")) == {"success", "reward", "plan_age", "starvation"}

    model = verdict(list(cases.values()), "model")
    assert (model.comparisons, model.inside, model.holm_significant, model.in_real_interval) == (8, 8, 0, 8)
    episode_verdict = verdict(list(cases.values()), "episode")
    assert episode_verdict.comparisons == 8
    assert "kalibriert/gut (Wiederholung): Starvation" in episode_verdict.outside
    assert episode_verdict.holm_significant >= 1
    replay = verdict(list(cases.values()), "replay")
    assert (replay.comparisons, replay.inside) == (4, 4)


def test_determinism_rows_compare_episodes_and_requests_of_two_executions(tmp_path):
    import dataclasses
    import json

    from kortyn.metrics import RequestRecord
    from kortyn.report_phase3a import determinism_rows, determinism_table

    def request(seed, request_id, server_ms=300.0):
        return RequestRecord(
            run="modell__gut", profile="kalibriert", seed=seed, attempt=0, request_id=request_id,
            phase="realtime", status="ok", obs_timestep=8 * request_id, t_obs_s=0.8 * request_id,
            t_send_s=0.8 * request_id, request_bytes=100, queue_at_send=6, t_resp_s=0.8 * request_id + 0.3,
            server_total_ms=server_ms,
        )  # fmt: skip

    def write(directory, episodes, requests):
        directory.mkdir(parents=True)
        (directory / "episodes.jsonl").write_text("".join(json.dumps(e.to_json()) + "\n" for e in episodes))
        (directory / "requests.jsonl").write_text("".join(json.dumps(r.to_json()) + "\n" for r in requests))

    virtual = {"events": 10, "compute_s": 1.0, "cache_hits": 0, "inference_ms": 3.0}
    first = [episode(s, virtual=virtual) for s in (1000, 1001, 1002)]
    requests = [request(s, i) for s in (1000, 1001, 1002) for i in (1, 2)]
    write(tmp_path / "phase3a" / "validation_model" / "kalibriert" / "modell__gut", first, requests)
    # second execution: two seeds only, other machine diagnostics, one request differs
    second = [episode(s, virtual={**virtual, "compute_s": 5.0, "cache_hits": 2}) for s in (1000, 1001)]
    changed = [
        request(1000, 1),
        request(1000, 2),
        request(1001, 1),
        dataclasses.replace(request(1001, 2), server_total_ms=301.0),
    ]
    write(tmp_path / "phase3a" / "determinism" / "kalibriert" / "modell__gut", second, changed)
    # third execution: seed 1001 differs in the last float bits of its coverage
    third = [episode(1000, virtual=virtual), episode(1001, coverage=0.5 + 1e-15, virtual=virtual)]
    write(tmp_path / "phase3a" / "determinism_b" / "kalibriert" / "modell__gut", third, changed)
    first_pair, second_pair = determinism_rows(tmp_path)
    assert (first_pair.profile, first_pair.run) == ("kalibriert", "modell__gut")
    assert (first_pair.match.episodes, first_pair.match.identical, first_pair.match.diverged) == (2, 2, 0)
    assert (first_pair.same_requests, first_pair.requests) == (3, 4)
    assert (second_pair.match.identical, second_pair.match.same_outcome) == (1, 1)
    assert (second_pair.same_requests, second_pair.requests) == (4, 4)
    assert len(determinism_table([first_pair, second_pair])) == 4
    assert determinism_rows(tmp_path / "missing") == []


def test_realized_power_pools_the_noise_of_all_real_time_conditions():
    from kortyn.report_phase3a import TaskData, realized_power

    seeds = range(100)
    reference = run([episode(s, success=s % 2 == 0) for s in seeds])
    # 20 % and 40 % of the seeds change their outcome under latency
    mild = run([episode(s, success=(s % 2 == 0) != (s % 10 < 2)) for s in seeds])
    strong = run([episode(s, success=(s % 2 == 0) != (s % 10 < 4)) for s in seeds])
    data = TaskData(
        "push-v3",
        {"referenz_sync25": reference, "schnell__gut": mild, "langsam__sehr_schlecht": strong},
        {},
    )
    lines = realized_power([data])
    assert len(lines) == 3
    cells = [c.strip() for c in lines[2].strip("|").split("|")]
    assert cells[:4] == ["push-v3", "2", "100", "30 % [20 %–40 %]"]
    assert cells[5] == f"{100 * mde_paired_binary(100, 0.3):.0f} %"
    assert realized_power([TaskData("push-v3", {"referenz_sync25": reference}, {})]) == []


def test_published_comparison_over_all_seeds_and_the_default_evaluation_seeds():
    from kortyn.report_phase3a import TaskData, published_comparison

    # 60 % over 100 seeds, 22/30 on the seeds a default LeRobot evaluation of 30 episodes uses
    successes = set(range(1000, 1022)) | set(range(1030, 1068))
    sync50 = run([episode(s, success=s in successes) for s in range(1000, 1100)])
    lines = published_comparison([TaskData("coffee-push-v3", {"referenz_sync50": sync50}, {})])
    assert "Seeds 1000–1029" in lines[0]
    cells = [c.strip() for c in lines[2].strip("|").split("|")]
    assert cells[0] == "coffee-push-v3"
    assert cells[2].startswith("60 %")
    assert cells[3] == "–"  # no Sync-25 run
    assert cells[5].startswith("✗")  # 60/100 vs 27/30: the interval excludes 0
    assert cells[6] == "22/30 (73 %)"
    assert cells[7].startswith("✓")  # 22/30 vs 27/30: consistent


def test_lerobot_eval_rows_compare_the_official_evaluation_on_the_same_seeds(tmp_path):
    import json

    from kortyn.report_phase3a import TaskData, lerobot_eval_rows

    sync50 = run([episode(s, success=s % 3 != 0) for s in range(1000, 1010)])
    # LeRobot also solves seed 1002, which Kortyn's Sync-50 fails
    summary = {"success_by_seed": {str(s): s % 3 != 0 or s == 1002 for s in range(1000, 1010)}}
    target = tmp_path / "phase3a" / "smolvla" / "coffee_push" / "lerobot_eval" / "smolvla"
    target.mkdir(parents=True)
    (target / "summary.json").write_text(json.dumps(summary))
    lines = lerobot_eval_rows(tmp_path, [TaskData("coffee-push-v3", {"referenz_sync50": sync50}, {})])
    assert len(lines) == 3
    cells = [c.strip() for c in lines[2].strip("|").split("|")]
    assert cells[:6] == ["coffee-push-v3", "1000–1009", "80 % (8/10)", "70 % (7/10)", "9/10", "1 / 0"]
    assert (
        lerobot_eval_rows(tmp_path / "none", [TaskData("coffee-push-v3", {"referenz_sync50": sync50}, {})])
        == []
    )


def test_transport_table_compares_on_the_seeds_all_variants_share():
    from kortyn.report_phase3a import TaskData, transport_name, transport_table

    assert transport_name("transport_jpeg90") == "JPEG q90, 480×480"
    assert transport_name("transport_jpeg75_240px").startswith("JPEG q75, 240×240")
    assert transport_name("transport_png") == "PNG, 480×480"
    assert transport_name("transport_custom") == "custom"
    # the raw reference has 100 seeds (80 % successes overall, 100 % on the first 50), the variant 50
    reference = run([episode(s, success=s < 1050 or s % 5 == 0) for s in range(1000, 1100)])
    variant = run([episode(s, success=s % 2 == 0) for s in range(1000, 1050)])
    lines = transport_table(
        [TaskData("push-v3", {"referenz_sync25": reference, "transport_jpeg90": variant}, {})]
    )
    assert "n = 50" in lines[0]
    rows = [line for line in lines if line.startswith("| roh") or line.startswith("| JPEG")]
    assert rows[0].split("|")[4].strip().startswith("100 %")  # the reference on the common seeds
    assert rows[1].split("|")[4].strip().startswith("50 %")
    assert rows[1].split("|")[5].strip().startswith("-50 pp")


def test_pooled_effects_split_by_latency_budget_and_compare_tasks():
    from kortyn.report_phase3a import (
        EffectFamily,
        effect_summary_table,
        latency_budget_ms,
        latency_effects,
        pooled_difference,
        pooled_effects,
    )

    assert latency_budget_ms({"agent": {"refill_threshold": 6}}, 10.0) == 700.0
    assert latency_budget_ms({"agent": {"refill_threshold": 24}}, 80.0) == 312.5
    seeds = range(1000, 1040)
    reference = run([episode(s, success=True) for s in seeds])
    fast = run([episode(s, success=True) for s in seeds])  # within the budget: no loss
    slow = run([episode(s, success=s % 2 == 0) for s in seeds])  # above the budget: -50 pp
    boundary = run([episode(s, success=False) for s in seeds])  # excluded from the pooled effects

    def family(name, slow_run):
        return EffectFamily(
            name,
            reference,
            {"schnell__gut": fast, "langsam__gut": slow_run, "grenzfall__gut": boundary},
            312.5,
            {"schnell__gut": 105.0, "langsam__gut": 320.0, "grenzfall__gut": 730.0},
        )

    static = family("drawer-close-v3", fast)
    dynamic = family("coffee-push-v3", slow)
    effects = {(e.family, e.regime): e for e in pooled_effects([static, dynamic])}
    assert effects[("coffee-push-v3", "alle Bedingungen")].delta.value == pytest.approx(-0.25)
    assert effects[("coffee-push-v3", "alle Bedingungen")].conditions == 2  # boundary case excluded
    assert effects[("coffee-push-v3", "Latenz im Budget")].delta.value == pytest.approx(0.0)
    assert effects[("coffee-push-v3", "Latenz über Budget")].delta.value == pytest.approx(-0.5)
    assert effects[("drawer-close-v3", "Latenz über Budget")].delta.value == pytest.approx(0.0)
    difference = pooled_difference(
        effects[("drawer-close-v3", "Latenz über Budget")], effects[("coffee-push-v3", "Latenz über Budget")]
    )
    assert difference.value == pytest.approx(-0.5)
    assert difference.high < 0
    table = effect_summary_table(list(effects.values()))
    assert any(
        line.startswith("| coffee-push-v3 − drawer-close-v3 | Latenz über Budget | ✗") for line in table
    )
    points = {p.condition: p for p in latency_effects([dynamic])}
    assert points["langsam__gut"].latency_budgets == pytest.approx(320.0 / 312.5)
    assert points["langsam__gut"].delta == pytest.approx(-0.5)
    assert points["grenzfall__gut"].relative_success == 0.0
