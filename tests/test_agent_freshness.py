"""Phase-2 agent behaviour (freshness strategies) against a fake Brain with controllable latency.

The fake Brain's actions encode their control step (``action[0]``), so the tests can check which plan
action ran at which tick, what the agent sent as an inpainting guide, and how old every plan was.
"""

from __future__ import annotations

import threading
import time
from concurrent import futures

import grpc
import numpy as np
import pytest

from kortyn.agent import Agent, BrainClient, BrainSpec, EpisodeContext, IncompatibleBrainError
from kortyn.codec import decode_tensor
from kortyn.config import (
    AgentConfig,
    GuardConfig,
    RequestConfig,
    SmoothingConfig,
    StrategyConfig,
)
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.proto import kortyn_brain_pb2_grpc as pb_grpc
from tests.test_agent import DT, FakeBrain, FakeEnv, assert_on_time, executed_steps, free_port


class GuidedFakeBrain(FakeBrain):
    """Fake Brain that advertises guidance, records guides and serves load counters."""

    def __init__(self, latency_s: float, supports_guidance: bool = True) -> None:
        super().__init__(latency_s)
        self.supports_guidance = supports_guidance
        self.guides: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        self._started = time.perf_counter()
        self._stats_lock = threading.Lock()

    def GetInfo(self, request, context):
        info = super().GetInfo(request, context)
        info.supports_guidance = self.supports_guidance
        return info

    def Infer(self, request, context):
        if request.HasField("guide_actions"):
            with self._stats_lock:
                self.guides.append(
                    (
                        request.obs_timestep,
                        request.guide_start_timestep,
                        decode_tensor(request.guide_actions).copy(),
                        np.array(request.guide_weights, dtype=np.float32),
                    )
                )
        return super().Infer(request, context)

    def GetStats(self, request, context):
        with self._lock:
            count = len(self.requests)
        return pb.BrainStats(
            requests=count,
            inferences=count,
            busy_ms=count * self.latency_s * 1000.0,
            uptime_ms=(time.perf_counter() - self._started) * 1000.0,
        )


@pytest.fixture
def make_agent():
    created = []

    def factory(latency_s: float, env: FakeEnv | None = None, supports_guidance: bool = True, **kwargs):
        brain_impl = GuidedFakeBrain(latency_s, supports_guidance)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb_grpc.add_KortynBrainServicer_to_server(brain_impl, server)
        port = free_port()
        server.add_insecure_port(f"127.0.0.1:{port}")
        server.start()
        client = BrainClient(f"127.0.0.1:{port}")
        stats = BrainClient(f"127.0.0.1:{port}")
        spec = BrainSpec.from_proto(client.get_info(timeout_s=5))
        env = env or FakeEnv()
        config = AgentConfig(**{"actions_per_chunk": 15, "refill_threshold": 6, **kwargs})
        agent = Agent(config, env, client, spec, stats_client=stats)
        created.append((server, client, stats))
        return agent, env, brain_impl

    yield factory
    for server, client, stats in created:
        client.close()
        stats.close()
        server.stop(0)


def ctx(strategy: str = "test", seed: int = 1000) -> EpisodeContext:
    return EpisodeContext(
        run="test", profile="fake", scenario="test", mode="realtime", seed=seed, strategy=strategy
    )


def realtime_steps(brain: FakeBrain) -> list[int]:
    return [step for step, _ in brain.requests[1:]]  # without the warm-up request


def test_threshold_ten_requests_every_four_ticks(make_agent):
    agent, env, brain = make_agent(latency_s=0.02)  # usable in the next tick
    strategy = StrategyConfig(request=RequestConfig(refill_threshold=10))
    episode = agent.run_episode(ctx("schwelle10"), strategy).episode
    assert episode.valid, episode.invalid_reasons
    assert set(np.diff(realtime_steps(brain))) == {4}
    assert episode.starved_ticks == 0
    np.testing.assert_array_equal(executed_steps(env), np.arange(60))  # timestep alignment
    assert episode.strategy == "schwelle10"
    assert_on_time(episode)


def test_adaptive_policy_hits_the_age_target_with_fewer_requests_than_threshold_ten(make_agent):
    # 120 ms latency = 3 ticks at 20 Hz; target 0.25 s = 5 ticks -> period 2 * (5 - 3) + 1 = 5 ticks
    agent, _, brain = make_agent(latency_s=0.12)
    adaptive = StrategyConfig(request=RequestConfig(mode="adaptive", target_age_s=0.25))
    outcome = agent.run_episode(ctx("adaptiv"), adaptive)
    episode = outcome.episode
    assert episode.starved_ticks == 0
    steps = realtime_steps(brain)
    assert np.median(np.diff(steps)) == 5
    ages = np.array(episode.ticks.action_age[20:])
    assert ages.mean() <= 5.5  # D + (P - 1) / 2 = 5, plus timing noise
    realtime = [r for r in outcome.requests if r.phase == "realtime"]
    assert {r.trigger for r in realtime} <= {"freshness", "deadline"}
    assert all(r.expected_delay_ticks == 3 for r in realtime[3:])
    assert all(r.latency_median_ms is not None and 100.0 < r.latency_median_ms < 160.0 for r in realtime)

    agent10, _, brain10 = make_agent(latency_s=0.12)
    agent10.run_episode(ctx("schwelle10"), StrategyConfig(request=RequestConfig(refill_threshold=10)))
    assert len(realtime_steps(brain)) < len(realtime_steps(brain10))


def test_hold_guard_stops_the_robot_and_logs_emergencies(make_agent):
    agent, env, _ = make_agent(latency_s=0.6)  # 12 ticks > budget of 7 ticks
    strategy = StrategyConfig(guard=GuardConfig(mode="hold"))
    episode = agent.run_episode(ctx("halt"), strategy).episode
    starved = np.array(episode.ticks.starved, dtype=bool)
    emergency = np.array(episode.ticks.emergency)
    assert starved.any()
    np.testing.assert_array_equal(emergency[starved], 1)
    assert episode.emergency_ticks == int(starved.sum())
    assert episode.emergency_events >= 1
    assert episode.emergency_ticks_age == 0
    # holding = target is the current position (FakeEnv state = [step index, 0]), no plan action, no age
    for k in np.flatnonzero(starved):
        np.testing.assert_array_equal(env.executed[k][1], [k, 0.0])
        assert episode.ticks.plan_age_ms[k] == -1.0
        assert episode.ticks.action_age[k] == -1


def test_age_limit_rejects_old_plans_before_the_queue_runs_dry(make_agent):
    # 0.3 s latency = 6-7 ticks; threshold 6 executes plan ages up to ~13 ticks; limit 0.4 s = 8 ticks
    agent, _, _ = make_agent(latency_s=0.3)
    strategy = StrategyConfig(guard=GuardConfig(mode="hold", max_plan_age_s=0.4))
    episode = agent.run_episode(ctx("limit"), strategy).episode
    emergency = np.array(episode.ticks.emergency)
    assert episode.emergency_ticks_age > 0
    assert (emergency == 2).sum() == episode.emergency_ticks_age
    executed_ages = np.array(episode.ticks.action_age)
    assert executed_ages.max() <= 8  # nothing older than the limit was executed
    ages_ms = np.array(episode.ticks.plan_age_ms)
    assert ages_ms[emergency == 2].max() == -1.0


def test_inpainting_guides_freeze_what_runs_before_the_new_plan(make_agent):
    agent, _, brain = make_agent(latency_s=0.12)  # usable 3 ticks after sending
    strategy = StrategyConfig(smoothing=SmoothingConfig(mode="inpaint", max_overlap_steps=10))
    outcome = agent.run_episode(ctx("inpaint"), strategy)
    assert outcome.episode.starved_ticks == 0
    assert len(brain.guides) >= 5
    for obs_step, start, actions, weights in brain.guides:
        assert start == obs_step - 1  # the previous tick's action comes first
        # executed and queued plan actions encode their control steps: the guide is contiguous in time
        np.testing.assert_array_equal(actions[:, 0], np.arange(start, start + len(actions)))
        frozen = int((weights == 1.0).sum())
        assert frozen == 4  # previous, current, and the 2 ticks before the plan is usable (D = 3)
        assert np.all(np.diff(weights) <= 0)
        assert weights[-1] < 1.0
    realtime = [r for r in outcome.requests if r.phase == "realtime"]
    assert all(r.guide_frozen == 4 for r in realtime)
    # threshold 6: 6 queued actions at sending time, 2 of them run before the plan is usable -> 4 soft
    assert all(r.guide_soft == 4 for r in realtime)


def test_inpainting_needs_a_brain_that_supports_guides(make_agent):
    agent, _, _ = make_agent(latency_s=0.02, supports_guidance=False)
    with pytest.raises(IncompatibleBrainError, match="does not support guides"):
        agent.run_episode(ctx("inpaint"), StrategyConfig(smoothing=SmoothingConfig(mode="inpaint")))


def test_naive_shift_alignment_executes_stale_actions(make_agent):
    agent, env, _ = make_agent(latency_s=0.12)
    episode = agent.run_episode(ctx("shift"), StrategyConfig(alignment="shift")).episode
    steps = executed_steps(env)
    ticks = np.arange(len(steps))
    assert (steps < ticks).any()  # actions planned for earlier steps ran late
    assert episode.starved_ticks == 0
    aligned_agent, aligned_env, _ = make_agent(latency_s=0.12)
    aligned_agent.run_episode(ctx("timestep"), StrategyConfig())
    np.testing.assert_array_equal(executed_steps(aligned_env), np.arange(60))


def test_plan_age_is_logged_for_every_executed_action_and_brain_load_is_measured(make_agent):
    agent, _, brain = make_agent(latency_s=0.02)
    outcome = agent.run_episode(ctx("baseline"), StrategyConfig())
    episode = outcome.episode
    ages_ms = np.array(episode.ticks.plan_age_ms)
    ages = np.array(episode.ticks.action_age)
    assert len(ages_ms) == episode.steps
    assert (ages_ms >= 0).all()
    np.testing.assert_allclose(ages_ms, ages * DT * 1000.0, atol=8.0)  # clock jitter of a few ms
    assert episode.brain_load is not None
    realtime = [r for r in outcome.requests if r.phase == "realtime"]
    assert episode.brain_load["requests"] == len(realtime)
    assert episode.brain_load["busy_ms"] == pytest.approx(len(realtime) * 20.0)
    assert episode.brain_load["window_ms"] >= (episode.steps - 1) * DT * 1000.0  # last tick starts it
    assert len(brain.requests) == len(realtime) + 1  # plus the warm-up
