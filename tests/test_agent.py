"""Real-time behaviour of the Kortyn Agent against a fake Brain with controllable latency.

The fake Brain returns actions that encode their own control step (``action[0] = step``) and the request
id (``action[1]``), so the tests can check exactly which action ran at which tick.
"""

from __future__ import annotations

import socket
import threading
import time
from concurrent import futures
from dataclasses import dataclass

import grpc
import numpy as np
import pytest

from kortyn.agent import (
    ActionQueue,
    Agent,
    BrainClient,
    BrainSpec,
    EpisodeContext,
    IncompatibleBrainError,
    request_seed,
)
from kortyn.codec import decode_observation, encode_array
from kortyn.config import AgentConfig
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.proto import kortyn_brain_pb2_grpc as pb_grpc

FPS = 20.0
DT = 1.0 / FPS


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class FakeStep:
    observation: dict
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    coverage: float | None


class FakeEnv:
    """Observation state = [step index, 0]; records every executed action with its time."""

    action_dim = 2

    def __init__(self, max_steps: int = 60, success_at: int | None = None) -> None:
        self._max_steps = max_steps
        self._success_at = success_at
        self.step_index = 0
        self.executed: list[tuple[float, np.ndarray]] = []

    @property
    def fps(self) -> float:
        return FPS

    @property
    def max_episode_steps(self) -> int:
        return self._max_steps

    def _observation(self) -> dict:
        return {
            "observation.image": np.zeros((8, 8, 3), dtype=np.uint8),
            "observation.state": np.array([self.step_index, 0.0], dtype=np.float32),
        }

    def reset(self, seed: int) -> dict:
        self.step_index = 0
        self.executed = []
        return self._observation()

    def step(self, action: np.ndarray) -> FakeStep:
        self.executed.append((time.monotonic(), np.array(action, dtype=np.float32)))
        self.step_index += 1
        success = self._success_at is not None and self.step_index >= self._success_at
        return FakeStep(self._observation(), 0.1, success, False, success, 0.5)

    def hold_action(self, observation: dict) -> np.ndarray:
        return observation["observation.state"].copy()


class FakeBrain(pb_grpc.KortynBrainServicer):
    def __init__(self, latency_s: float, fail_from_step: int | None = None) -> None:
        self.latency_s = latency_s
        self.fail_from_step = fail_from_step
        self.requests: list[tuple[int, np.ndarray]] = []
        self._lock = threading.Lock()

    def GetInfo(self, request, context):
        return pb.BrainInfo(
            policy_type="fake",
            device="cpu",
            n_obs_steps=2,
            max_actions=15,
            action_dim=2,
            sampler="fake",
            input_features=[
                pb.FeatureSpec(key="observation.image", kind="VISUAL", shape=[3, 8, 8]),
                pb.FeatureSpec(key="observation.state", kind="STATE", shape=[2]),
            ],
        )

    def Infer(self, request, context):
        states = decode_observation(request.observation)["observation.state"]
        with self._lock:
            self.requests.append((request.obs_timestep, states.copy()))
        if self.fail_from_step is not None and request.obs_timestep >= self.fail_from_step:
            context.abort(grpc.StatusCode.UNAVAILABLE, "fake brain is down")
        time.sleep(self.latency_s)
        n = request.max_actions or 15
        actions = np.array(
            [[request.obs_timestep + j, request.request_id] for j in range(n)], dtype=np.float32
        )
        return pb.InferResponse(
            request_id=request.request_id,
            start_timestep=request.obs_timestep,
            actions=encode_array(actions),
            server_queue_ms=0.0,
            server_inference_ms=self.latency_s * 1000.0,
            server_total_ms=self.latency_s * 1000.0,
        )


@pytest.fixture
def make_agent():
    created = []

    def factory(
        latency_s: float, fail_from_step: int | None = None, env: FakeEnv | None = None, **agent_kwargs
    ):
        brain_impl = FakeBrain(latency_s, fail_from_step)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb_grpc.add_KortynBrainServicer_to_server(brain_impl, server)
        port = free_port()
        server.add_insecure_port(f"127.0.0.1:{port}")
        server.start()
        client = BrainClient(f"127.0.0.1:{port}")
        spec = BrainSpec.from_proto(client.get_info(timeout_s=5))
        env = env or FakeEnv()
        config = AgentConfig(**{"actions_per_chunk": 15, "refill_threshold": 6, **agent_kwargs})
        agent = Agent(config, env, client, spec)
        created.append((server, client))
        return agent, env, brain_impl

    yield factory
    for server, client in created:
        client.close()
        server.stop(0)


def assert_on_time(episode) -> None:
    lateness = np.array(episode.ticks.lateness_ms)
    assert np.median(lateness) < 5.0
    assert lateness.max() < 1000.0 * DT  # never a full tick late
    assert episode.tick_overruns == 0
    # no drift: the episode took exactly steps * dt of wall-clock time (plus at most one late last tick)
    assert episode.wall_duration_s == pytest.approx(episode.steps * DT, abs=0.06)


def ctx(mode="realtime", seed=1000):
    return EpisodeContext(run="test", profile="fake", scenario="test", mode=mode, seed=seed)


def executed_steps(env: FakeEnv) -> np.ndarray:
    return np.array([action[0] for _, action in env.executed])


def test_fast_brain_never_starves_and_executes_each_step_exactly_on_time(make_agent):
    agent, env, brain = make_agent(latency_s=0.02)
    outcome = agent.run_episode(ctx())
    episode = outcome.episode
    assert episode.valid, episode.invalid_reasons
    assert episode.steps == 60
    assert episode.starved_ticks == 0
    # action executed at tick k is the action planned for step k
    np.testing.assert_array_equal(executed_steps(env), np.arange(60))
    # refill every 8 ticks: identical observation pairs to LeRobot eval (n_action_steps=8)
    assert [step for step, _ in brain.requests] == [0, 8, 16, 24, 32, 40, 48, 56]
    np.testing.assert_array_equal(brain.requests[0][1], [[0, 0], [0, 0]])  # history padded at the start
    np.testing.assert_array_equal(brain.requests[1][1], [[7, 0], [8, 0]])  # (o_{k-1}, o_k), oldest first
    # absolute deadlines: no drift; lateness stays small (robust statistics: CI machines are noisy)
    assert_on_time(episode)
    realtime = [r for r in outcome.requests if r.phase == "realtime"]
    assert all(r.status == "ok" for r in outcome.requests)
    assert all(r.e2e_ms is not None and r.e2e_ms >= 20.0 for r in realtime)
    assert all(r.reaction_ms is not None and r.reaction_ms >= r.e2e_ms for r in realtime[:-1])


def test_slow_brain_starves_but_the_simulation_never_waits(make_agent):
    agent, env, _ = make_agent(latency_s=0.6)  # 12 ticks > budget of 7 ticks
    outcome = agent.run_episode(ctx())
    episode = outcome.episode
    assert episode.steps == 60
    assert episode.starved_ticks > 0
    assert episode.starvation_events >= 1
    # starvation here is caused by server time; only the boundary tick of a starvation run can flip to
    # "network" because of the ~1 ms transport overhead on top of the server time
    assert episode.starved_ticks_compute >= episode.starved_ticks - episode.starvation_events
    assert episode.starved_ticks_none_pending == 0
    # the control clock kept running at 20 Hz regardless of the Brain
    assert_on_time(episode)
    times = np.array([t for t, _ in env.executed])
    assert np.median(np.diff(times)) == pytest.approx(DT, abs=0.005)
    steps = executed_steps(env)
    starved = np.array(episode.ticks.starved, dtype=bool)
    # starved ticks repeat the previous action, fresh ticks execute the action planned for that step
    for k in np.flatnonzero(starved):
        assert steps[k] == steps[k - 1]
    np.testing.assert_array_equal(steps[~starved], np.flatnonzero(~starved))
    # stale actions were dropped on arrival, and action age grows while starving
    assert any(r.n_stale > 0 for r in outcome.requests if r.phase == "realtime")
    ages = np.array(episode.ticks.action_age)
    assert ages[starved].max() > ages[~starved].max()


def test_latency_within_budget_does_not_starve(make_agent):
    agent, _, _ = make_agent(latency_s=0.2)  # 4-5 ticks incl. quantization <= refill_threshold + 1
    episode = agent.run_episode(ctx()).episode
    assert episode.starved_ticks == 0


def test_brain_failure_mid_episode_is_recorded_and_the_loop_survives(make_agent):
    agent, _, _ = make_agent(latency_s=0.01, fail_from_step=16)
    outcome = agent.run_episode(ctx())
    episode = outcome.episode
    assert episode.steps == 60  # the robot kept its control rate the whole time
    assert episode.starved_ticks > 0
    assert not episode.valid
    assert any("unavailable" in reason for reason in episode.invalid_reasons)
    assert {r.status for r in outcome.requests} == {"ok", "unavailable"}


def test_sync_mode_waits_and_executes_eight_actions_per_chunk(make_agent):
    agent, env, brain = make_agent(latency_s=0.05, env=FakeEnv(max_steps=30))
    episode = agent.run_episode(ctx(mode="sync")).episode
    assert episode.steps == 30
    assert episode.starved_ticks == 0
    np.testing.assert_array_equal(executed_steps(env), np.arange(30))
    assert [step for step, _ in brain.requests] == [0, 8, 16, 24]


def test_success_ends_the_episode(make_agent):
    agent, _, _ = make_agent(latency_s=0.01, env=FakeEnv(success_at=12))
    episode = agent.run_episode(ctx()).episode
    assert episode.success
    assert episode.steps == 12


def test_cold_start_holds_position_until_the_first_chunk(make_agent):
    agent, env, _ = make_agent(latency_s=0.2, start_mode="cold")
    episode = agent.run_episode(ctx()).episode
    assert episode.warmup_ms is None
    assert episode.ticks.starved[0] == 1
    assert episode.ticks.action_age[0] == -1
    first_hold = env.executed[0][1]
    np.testing.assert_array_equal(first_hold, [0.0, 0.0])  # hold action = current position


def test_unreachable_brain_during_warmup_yields_an_invalid_episode():
    client = BrainClient(f"127.0.0.1:{free_port()}")
    spec = BrainSpec(
        n_obs_steps=2,
        max_actions=15,
        action_dim=2,
        features={"observation.image": ("VISUAL", (3, 8, 8)), "observation.state": ("STATE", (2,))},
        sampler="fake",
        device="cpu",
        policy_type="fake",
        warmup_inference_ms=0.0,
    )
    agent = Agent(AgentConfig(warmup_timeout_s=2.0), FakeEnv(), client, spec)
    episode = agent.run_episode(ctx()).episode
    client.close()
    assert not episode.valid
    assert episode.steps == 0
    assert "warm-up request failed" in episode.invalid_reasons[0]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"action_dim": 3}, "3-D actions"),
        ({"max_actions": 10}, "exceeds the Brain maximum"),
        ({"features": {"observation.image": ("VISUAL", (3, 16, 16))}}, "policy expects"),
        ({"features": {"observation.depth": ("VISUAL", (1, 8, 8))}}, "policy needs"),
    ],
)
def test_incompatible_brain_is_rejected(override, message):
    base = dict(
        n_obs_steps=2,
        max_actions=15,
        action_dim=2,
        features={"observation.image": ("VISUAL", (3, 8, 8)), "observation.state": ("STATE", (2,))},
        sampler="fake",
        device="cpu",
        policy_type="fake",
        warmup_inference_ms=0.0,
    )
    base.update(override)
    with pytest.raises(IncompatibleBrainError, match=message):
        Agent(AgentConfig(), FakeEnv(), BrainClient("127.0.0.1:1"), BrainSpec(**base))


def test_action_queue_merge_semantics():
    queue = ActionQueue()
    chunk = np.array([[s, 1] for s in range(0, 15)], dtype=np.float32)
    assert queue.merge(0, chunk, request_id=1, current_step=0) == (0, 15)
    newer = np.array([[s, 2] for s in range(8, 23)], dtype=np.float32)
    # arrives at step 12: steps 8..11 are stale, 12..22 replace/extend the queue
    assert queue.merge(8, newer, request_id=2, current_step=12) == (4, 11)
    assert queue.available(12) == 11
    action, request_id, chunk_step = queue.pop(12)
    assert (action[0], request_id, chunk_step) == (12, 2, 8)
    # an older chunk can never overwrite a newer one
    assert queue.merge(0, chunk, request_id=1, current_step=13) == (15, 0)
    assert queue.pop(13)[1] == 2
    assert queue.pop(100) is None
    assert queue.available(0) == 0


def test_request_seed_depends_only_on_episode_seed_and_step():
    assert request_seed(1000, 8) == request_seed(1000, 8)
    assert request_seed(1000, 8) != request_seed(1000, 16)
    assert request_seed(1000, 8) != request_seed(1001, 8)
