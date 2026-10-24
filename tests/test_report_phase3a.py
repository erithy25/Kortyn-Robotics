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
