"""Freshness engine decisions: request timing, plan alignment, inpainting guides and the age guard."""

from __future__ import annotations

import numpy as np
import pytest

from kortyn.config import GuardConfig, LatencyEstimatorConfig, RequestConfig, SmoothingConfig
from kortyn.freshness import (
    AdaptivePolicy,
    RequestContext,
    ThresholdPolicy,
    build_guide,
    make_request_policy,
    overlap_weights,
    placement_step,
    plan_is_usable,
    ticks_until_usable,
)
from kortyn.latency import LatencyEstimator

DT = 0.1
CHUNK = 15


def steady_latency(delay_ticks: int, spread_ms: float = 5.0) -> LatencyEstimator:
    """Estimator whose median latency makes a chunk usable exactly ``delay_ticks`` ticks after sending."""
    est = LatencyEstimator(LatencyEstimatorConfig(window=8, min_spread_ms=spread_ms))
    for _ in range(8):
        est.observe((delay_ticks - 0.5) * DT)
    return est


def simulate(policy, delay_ticks: int, ticks: int = 200) -> tuple[list[int], list[int], int]:
    """Agent loop with a constant delay: request steps, executed plan ages and starved ticks.

    The warm-up plan (observation 0) covers steps 0..14 at the start, exactly like the real agent.
    """
    latency = steady_latency(delay_ticks)
    plans: dict[int, int] = dict.fromkeys(range(CHUNK), 0)  # step -> observation step of its plan
    in_flight: int | None = None  # observation step of the request in flight
    arrival = -1
    last_obs = 0
    requests: list[int] = []
    ages: list[int] = []
    starved = 0
    for k in range(ticks):
        if in_flight is not None and k >= arrival:
            for offset in range(CHUNK):
                if in_flight + offset >= k:
                    plans[in_flight + offset] = in_flight
            in_flight = None
        if k in plans:
            ages.append(k - plans.pop(k))
        else:
            starved += 1
        for old in [s for s in plans if s <= k]:
            del plans[old]
        available = sum(1 for s in plans if s > k)
        ctx = RequestContext(k, available, 0 if in_flight is None else 1, last_obs, 0.001)
        decision = policy.decide(ctx, latency, DT)
        if decision.send:
            assert decision.expected_delay_ticks == delay_ticks
            requests.append(k)
            in_flight, arrival, last_obs = k, k + delay_ticks, k
    return requests, ages, starved


def steady_state_mean_age(ages: list[int], period: int) -> float:
    """Mean over whole request periods (any window of complete periods of a periodic sequence)."""
    return float(np.mean(ages[60 : 60 + 10 * period]))


def test_ticks_until_usable_rounds_up_to_the_next_tick_start():
    assert ticks_until_usable(0.344, 0.002, DT) == 4
    assert ticks_until_usable(0.4, 0.0, DT) == 4  # arriving exactly at a tick start is usable there
    assert ticks_until_usable(0.401, 0.0, DT) == 5
    assert ticks_until_usable(0.0, 0.0, DT) == 1  # never in the tick it was requested in
    with pytest.raises(ValueError, match="dt"):
        ticks_until_usable(0.3, 0.0, 0.0)


def test_threshold_six_reproduces_phase_one_every_eight_ticks():
    requests, ages, starved = simulate(ThresholdPolicy(6), delay_ticks=4)
    assert requests[:4] == [8, 16, 24, 32]
    assert starved == 0
    assert steady_state_mean_age(ages, 8) == pytest.approx(4 + (8 - 1) / 2)  # D + (P - 1) / 2


@pytest.mark.parametrize(("threshold", "period"), [(6, 8), (8, 6), (10, 4)])
def test_higher_thresholds_request_more_often_and_execute_fresher_plans(threshold, period):
    requests, ages, starved = simulate(ThresholdPolicy(threshold), delay_ticks=4)
    assert set(np.diff(requests)) == {period}
    assert starved == 0
    assert steady_state_mean_age(ages, period) == pytest.approx(4 + (period - 1) / 2)


def test_threshold_policy_waits_while_a_request_is_in_flight():
    policy = ThresholdPolicy(6)
    latency = steady_latency(4)
    assert not policy.decide(RequestContext(10, 2, 1, 8, 0.0), latency, DT).send
    decision = policy.decide(RequestContext(10, 6, 0, 8, 0.0), latency, DT)
    assert decision.send
    assert decision.reason == "threshold"


@pytest.mark.parametrize(("delay", "period", "mean_age"), [(4, 7, 7.0), (5, 5, 7.0), (6, 6, 8.5)])
def test_adaptive_policy_meets_the_age_target_with_the_fewest_requests(delay, period, mean_age):
    """Target 0.7 s = 7 ticks: P = 2 (7 - D) + 1, but never below D with one request in flight."""
    policy = AdaptivePolicy(target_age_s=0.7, latency_quantile_z=1.645, safety_margin_ticks=1, dt=DT)
    requests, ages, starved = simulate(policy, delay_ticks=delay)
    assert starved == 0
    assert set(np.diff(requests[2:])) == {period}
    assert steady_state_mean_age(ages, period) == pytest.approx(mean_age)
    assert policy.period_ticks(delay) == 2 * (7 - delay) + 1


def test_adaptive_policy_reports_an_unreachable_target():
    policy = AdaptivePolicy(0.7, 1.645, 1, DT)
    decision = policy.decide(RequestContext(20, 8, 0, 17, 0.0), steady_latency(6), DT)
    assert decision.period_ticks is not None
    assert decision.period_ticks < decision.expected_delay_ticks  # back-to-back requests cannot reach it


def test_adaptive_deadline_fires_before_freshness_when_latency_is_uncertain():
    """A wide latency spread makes the deadline rule request although the freshness period is not over."""
    policy = AdaptivePolicy(target_age_s=1.4, latency_quantile_z=2.0, safety_margin_ticks=1, dt=DT)
    noisy = LatencyEstimator(LatencyEstimatorConfig(window=8, min_spread_ms=0.0))
    for value in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        noisy.observe(value)
    ctx = RequestContext(step=3, available_after=5, in_flight=0, last_obs_step=0, send_offset_s=0.0)
    decision = policy.decide(ctx, noisy, DT)
    assert decision.send
    assert decision.reason == "deadline"
    assert decision.safe_delay_ticks > decision.expected_delay_ticks
    relaxed = RequestContext(step=3, available_after=11, in_flight=0, last_obs_step=0, send_offset_s=0.0)
    assert not policy.decide(relaxed, noisy, DT).send


def test_adaptive_policy_requests_immediately_without_a_previous_plan():
    policy = AdaptivePolicy(0.7, 1.645, 1, DT)
    decision = policy.decide(RequestContext(0, 0, 0, None, 0.0), steady_latency(4), DT)
    assert decision.send


def test_make_request_policy_uses_the_agent_threshold_by_default():
    baseline = make_request_policy(RequestConfig(), default_threshold=6, dt=DT)
    assert isinstance(baseline, ThresholdPolicy)
    assert baseline.threshold == 6
    explicit = make_request_policy(RequestConfig(refill_threshold=10), default_threshold=6, dt=DT)
    assert isinstance(explicit, ThresholdPolicy)
    assert explicit.threshold == 10
    adaptive = make_request_policy(RequestConfig(mode="adaptive", target_age_s=0.6), 6, DT)
    assert isinstance(adaptive, AdaptivePolicy)
    assert adaptive.target_age_ticks == pytest.approx(6.0)


def test_alignment_skips_past_steps_or_replays_naively():
    assert placement_step(obs_step=8, merge_step=12, alignment="timestep") == 8
    assert placement_step(obs_step=8, merge_step=12, alignment="shift") == 12
    assert placement_step(obs_step=8, merge_step=8, alignment="shift") == 8
    with pytest.raises(ValueError, match="alignment"):
        placement_step(0, 0, "teleport")


@pytest.mark.parametrize("schedule", ["linear", "exp", "zeros"])
@pytest.mark.parametrize(("frozen", "soft"), [(3, 4), (5, 0), (1, 9)])
def test_overlap_weights_match_lerobot_real_time_chunking(schedule, frozen, soft):
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    processor = RTCProcessor(RTCConfig(prefix_attention_schedule=RTCAttentionSchedule[schedule.upper()]))
    expected = processor.get_prefix_weights(frozen, frozen + soft, frozen + soft).numpy()
    np.testing.assert_allclose(overlap_weights(frozen, soft, schedule), expected, atol=1e-6)


def test_build_guide_freezes_what_runs_before_the_new_plan_and_softly_guides_the_rest():
    queued = [np.array([float(s), 0.0], dtype=np.float32) for s in range(11, 18)]  # steps 11..17
    guide = build_guide(
        step=10,
        previous_action=np.array([9.0, 0.0], dtype=np.float32),
        current_action=np.array([10.0, 0.0], dtype=np.float32),
        queued=queued,
        expected_delay_ticks=4,  # steps 11, 12, 13 still run from the old plan
        config=SmoothingConfig(mode="inpaint", schedule="linear", max_overlap_steps=3),
    )
    assert guide.start_step == 9
    np.testing.assert_array_equal(guide.actions[:, 0], [9, 10, 11, 12, 13, 14, 15, 16])
    assert guide.frozen == 5
    assert guide.soft == 3
    np.testing.assert_allclose(guide.weights, [1, 1, 1, 1, 1, 0.75, 0.5, 0.25])


def test_build_guide_with_an_empty_queue_freezes_only_executed_actions():
    guide = build_guide(
        step=0,
        previous_action=None,
        current_action=np.array([5.0, 5.0], dtype=np.float32),
        queued=[],
        expected_delay_ticks=6,
        config=SmoothingConfig(mode="inpaint"),
    )
    assert guide.start_step == 0
    assert guide.frozen == 1
    assert guide.soft == 0
    assert guide.actions.shape == (1, 2)
    with pytest.raises(ValueError, match="expected_delay_ticks"):
        build_guide(
            step=0,
            previous_action=None,
            current_action=np.zeros(2, dtype=np.float32),
            queued=[],
            expected_delay_ticks=0,
            config=SmoothingConfig(mode="inpaint"),
        )


def test_guard_rejects_only_plans_older_than_the_hard_limit():
    unlimited = GuardConfig()
    assert plan_is_usable(30, DT, unlimited)
    guard = GuardConfig(mode="hold", max_plan_age_s=1.2)
    assert plan_is_usable(12, DT, guard)
    assert not plan_is_usable(13, DT, guard)
