"""Simulation-time agent: conformance with the real-time agent, determinism, sync mode, failures, replay."""

from __future__ import annotations

import json
from concurrent import futures

import grpc
import numpy as np
import pytest

from kortyn.agent import Agent, BrainClient, BrainSpec, EpisodeContext
from kortyn.config import (
    AgentConfig,
    DistributionConfig,
    NetworkConfig,
    StallConfig,
    TimingConfig,
)
from kortyn.metrics import EpisodeRecord
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.proto import kortyn_brain_pb2_grpc as pb_grpc
from kortyn.virtual_agent import VirtualTimeAgent, deterministic_view
from kortyn.virtual_time import (
    ComputeResult,
    DurationSampler,
    ModelTiming,
    ReplayLog,
    TimingSamplers,
)
from tests.test_agent import FakeBrain, FakeEnv, free_port

SPEC = BrainSpec(
    n_obs_steps=2,
    max_actions=15,
    action_dim=2,
    features={"observation.image": ("VISUAL", (3, 8, 8)), "observation.state": ("STATE", (2,))},
    sampler="fake",
    device="cpu",
    policy_type="fake",
    warmup_inference_ms=0.0,
)


class FakeCompute:
    """Same plans as the fake gRPC Brain of test_agent: action = [control step, request id]."""

    def __init__(self, fail_from_step: int | None = None) -> None:
        self.fail_from_step = fail_from_step
        self.cache_hits = 0
        self.steps: list[int] = []

    def __call__(self, request: pb.InferRequest) -> ComputeResult:
        self.steps.append(int(request.obs_timestep))
        if self.fail_from_step is not None and request.obs_timestep >= self.fail_from_step:
            return ComputeResult(None, "INTERNAL: fake brain is down")
        n = int(request.max_actions or 15)
        actions = np.array(
            [[request.obs_timestep + j, request.request_id] for j in range(n)], dtype=np.float32
        )
        return ComputeResult(actions, compute_ms=0.1)


def constant(ms: float) -> DurationSampler:
    return DurationSampler(DistributionConfig(kind="constant", value_ms=ms))


def model_samplers(service: DistributionConfig, tick: DistributionConfig | None = None) -> TimingSamplers:
    return TimingSamplers(
        model=ModelTiming(DurationSampler(service), constant(0.0), constant(0.0), constant(0.0)),
        tick=DurationSampler(tick or DistributionConfig()),
    )


def virtual_agent(
    service_ms: float | DistributionConfig = 170.0,
    env: FakeEnv | None = None,
    compute: FakeCompute | None = None,
    tick: DistributionConfig | None = None,
    timing_seed: int = 0,
    **agent_kwargs,
):
    service = (
        service_ms
        if isinstance(service_ms, DistributionConfig)
        else DistributionConfig(kind="constant", value_ms=service_ms)
    )
    env = env or FakeEnv()
    config = AgentConfig(**{"actions_per_chunk": 15, "refill_threshold": 6, **agent_kwargs})
    timing = TimingConfig(latency="model", hardware="test", timing_seed=timing_seed)
    agent = VirtualTimeAgent(
        config, env, SPEC, compute or FakeCompute(), timing, model_samplers(service, tick), timing_name="test"
    )
    return agent, env


def ctx(mode="realtime", seed=1000):
    return EpisodeContext(run="test", profile="fake", scenario="test", mode=mode, seed=seed)


def executed(env: FakeEnv) -> list[tuple[float, float]]:
    return [(float(a[0]), float(a[1])) for _, a in env.executed]


@pytest.fixture
def realtime_agent():
    created = []

    def factory(latency_s: float):
        brain = FakeBrain(latency_s)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb_grpc.add_KortynBrainServicer_to_server(brain, server)
        port = free_port()
        server.add_insecure_port(f"127.0.0.1:{port}")
        server.start()
        client = BrainClient(f"127.0.0.1:{port}")
        env = FakeEnv()
        agent = Agent(AgentConfig(actions_per_chunk=15, refill_threshold=6), env, client, SPEC)
        created.append((server, client))
        return agent, env

    yield factory
    for server, client in created:
        client.close()
        server.stop(0)


# Latencies chosen away from tick boundaries (3.4 and 11.4 ticks at 20 Hz), so the ~1 ms of real transport
# overhead of the real-time run cannot move a merge to another tick.
@pytest.mark.parametrize("latency_s", [0.17, 0.57])
def test_virtual_time_executes_exactly_what_real_time_executes(realtime_agent, latency_s):
    agent, env = realtime_agent(latency_s)
    real = agent.run_episode(ctx()).episode
    virtual_agent_, virtual_env = virtual_agent(service_ms=1000.0 * latency_s)
    virtual = virtual_agent_.run_episode(ctx()).episode
    assert executed(virtual_env) == executed(env)
    for series in ("starved", "action_age", "queue_len", "pending_request", "emergency"):
        assert getattr(virtual.ticks, series) == getattr(real.ticks, series), series
    assert virtual.ticks.plan_age_ms == pytest.approx(real.ticks.plan_age_ms, abs=8.0)
    assert (virtual.steps, virtual.n_requests, virtual.starved_ticks) == (
        real.steps,
        real.n_requests,
        real.starved_ticks,
    )
    assert (virtual.starved_ticks_compute, virtual.starved_ticks_network) == (
        real.starved_ticks_compute,
        real.starved_ticks_network,
    )
    assert virtual.clock == "virtual"
    assert virtual.ticks.lateness_ms == [0.0] * virtual.steps
    if latency_s > 0.5:
        assert virtual.starved_ticks > 0  # 12 ticks of latency > budget of 7 ticks


def test_two_identical_runs_give_identical_results():
    network = NetworkConfig(
        rtt_ms=150.0, rtt_jitter_ms=80.0, stall=StallConfig(interval_s=1.0, duration_ms=400.0)
    )
    service = DistributionConfig(kind="lognormal", mean_ms=300.0, std_ms=60.0)
    jitter = DistributionConfig(kind="uniform", low_ms=0.0, high_ms=5.0)

    def run(timing_seed: int = 0) -> tuple[list[dict], list[list[dict]]]:
        agent, _ = virtual_agent(service, tick=jitter, timing_seed=timing_seed)
        episodes, requests = [], []
        for seed in (1000, 1001):
            outcome = agent.run_episode(ctx(seed=seed), network=network)
            episodes.append(deterministic_view(outcome.episode.to_json()))
            requests.append([r.to_json() for r in outcome.requests])
        return episodes, requests

    first = run()
    assert json.dumps(first) == json.dumps(run())
    other = run(timing_seed=1)
    assert json.dumps(first) != json.dumps(other)  # another replication draws other latencies
    episode = first[0][0]
    assert episode["proxy"]["stall_windows_s"]  # the stall schedule ran on the simulation clock
    assert episode["virtual"]["latency"] == "model"
    assert episode["starved_ticks"] > 0


def test_sync_mode_waits_and_executes_eight_actions_per_chunk():
    compute = FakeCompute()
    agent, env = virtual_agent(service_ms=300.0, env=FakeEnv(max_steps=30), compute=compute)
    outcome = agent.run_episode(ctx(mode="sync"))
    assert outcome.episode.steps == 30
    assert outcome.episode.starved_ticks == 0
    assert [a for a, _ in executed(env)] == list(range(30))
    assert compute.steps == [0, 8, 16, 24]
    assert [r.e2e_ms for r in outcome.requests] == pytest.approx([300.0] * 4)
    assert outcome.episode.proxy is None


def test_brain_failure_and_timeouts_are_recorded_like_in_real_time():
    agent, _ = virtual_agent(service_ms=10.0, compute=FakeCompute(fail_from_step=16))
    outcome = agent.run_episode(ctx())
    assert outcome.episode.steps == 60
    assert outcome.episode.starved_ticks > 0
    assert not outcome.episode.valid
    assert {r.status for r in outcome.requests} == {"ok", "error"}

    slow, _ = virtual_agent(service_ms=900.0, request_timeout_s=0.5)
    late = slow.run_episode(ctx())
    statuses = [r.status for r in late.requests]
    assert statuses[0] == "ok"  # warm-up has its own, long timeout
    assert set(statuses[1:]) == {"timeout"}
    assert late.episode.valid  # a timeout is a robot-side outcome, not a measurement failure


def test_warmup_failure_yields_an_invalid_episode():
    agent, _ = virtual_agent(compute=FakeCompute(fail_from_step=0))
    episode = agent.run_episode(ctx()).episode
    assert not episode.valid
    assert episode.steps == 0
    assert "warm-up request failed" in episode.invalid_reasons[0]


def test_replay_of_a_logged_run_reproduces_it_exactly(tmp_path):
    network = NetworkConfig(
        rtt_ms=80.0, rtt_jitter_ms=40.0, stall=StallConfig(interval_s=1.5, duration_ms=300.0)
    )
    service = DistributionConfig(kind="uniform", low_ms=150.0, high_ms=450.0)
    jitter = DistributionConfig(kind="uniform", low_ms=0.0, high_ms=3.0)
    agent, env = virtual_agent(service, tick=jitter)
    run_dir = tmp_path / "logged"
    run_dir.mkdir()
    originals = {}
    with (
        (run_dir / "episodes.jsonl").open("w") as episodes,
        (run_dir / "requests.jsonl").open("w") as requests,
    ):
        for seed in (1000, 1001, 1002):
            outcome = agent.run_episode(ctx(seed=seed), network=network)
            originals[seed] = (outcome, executed(env))
            episodes.write(json.dumps(outcome.episode.to_json()) + "\n")
            requests.writelines(json.dumps(r.to_json()) + "\n" for r in outcome.requests)

    log = ReplayLog(run_dir)
    for seed, (original, original_actions) in originals.items():
        env = FakeEnv()
        replay = VirtualTimeAgent(
            AgentConfig(actions_per_chunk=15, refill_threshold=6),
            env,
            SPEC,
            FakeCompute(),
            TimingConfig(latency="replay"),
            replay=log,
            timing_name="replay",
        )
        outcome = replay.run_episode(ctx(seed=seed))
        assert executed(env) == original_actions
        a, b = outcome.episode, original.episode
        for series in ("starved", "action_age", "queue_len", "pending_request"):
            assert getattr(a.ticks, series) == getattr(b.ticks, series)
        assert a.ticks.plan_age_ms == pytest.approx(b.ticks.plan_age_ms, abs=1e-6)
        assert a.ticks.lateness_ms == pytest.approx(b.ticks.lateness_ms, abs=1e-9)
        assert [r.t_resp_s for r in outcome.requests] == pytest.approx(
            [r.t_resp_s for r in original.requests]
        )
        assert a.virtual["replay_exact"] == len(original.requests)
        assert a.virtual["replay_fallback"] == 0
        assert a.starved_ticks_compute == b.starved_ticks_compute


def test_episode_records_round_trip_with_the_virtual_fields():
    agent, _ = virtual_agent()
    episode = agent.run_episode(ctx()).episode
    restored = EpisodeRecord.from_json(json.loads(json.dumps(episode.to_json())))
    assert restored.clock == "virtual"
    assert restored.virtual == episode.virtual
    assert set(deterministic_view(episode.to_json())["virtual"]) == {
        "timing",
        "latency",
        "timing_seed",
        "hardware",
        "replay_run",
        "replay_exact",
        "replay_fallback",
        "events",
    }


# --------------------------------------------------------------------------------------------- local brain
def test_local_brain_runs_the_brain_path_and_memoizes_plans(tiny_checkpoint):
    from kortyn.codec import encode_array, encode_observation
    from kortyn.config import BrainProfile
    from kortyn.policy_runner import PolicyRunner
    from kortyn.virtual_agent import LocalBrain

    runner = PolicyRunner(tiny_checkpoint, BrainProfile(num_inference_steps=4), device="cpu", torch_threads=1)
    brain = LocalBrain(runner, cache_entries=2)
    assert brain.spec.max_actions == runner.max_actions
    assert brain.spec.features["observation.image"] == ("VISUAL", (3, 32, 32))
    rng = np.random.default_rng(0)
    observation = {
        "observation.image": rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8),
        "observation.state": rng.uniform(0, 512, size=(2, 2)).astype(np.float32),
    }
    request = pb.InferRequest(
        request_id=1, obs_timestep=8, seed=42, max_actions=10, observation=encode_observation(observation)
    )
    first = brain(request)
    assert first.error is None
    assert not first.cache_hit
    np.testing.assert_array_equal(first.actions, runner.infer(observation, seed=42, max_actions=10).actions)
    again = brain(request)
    assert again.cache_hit
    np.testing.assert_array_equal(again.actions, first.actions)
    assert (brain.inferences, brain.cache_hits) == (1, 1)
    # the request id and the step do not change the plan; the seed and a guide do
    moved = pb.InferRequest()
    moved.CopyFrom(request)
    moved.request_id, moved.obs_timestep = 7, 16
    assert brain(moved).cache_hit
    guided = pb.InferRequest()
    guided.CopyFrom(request)
    guided.guide_start_timestep = 7
    guided.guide_actions.CopyFrom(encode_array(np.full((3, 2), 100.0, dtype=np.float32)))
    guided.guide_weights.extend([1.0, 1.0, 0.5])
    result = brain(guided)
    assert result.error is None
    assert not result.cache_hit
    bad = pb.InferRequest()
    bad.CopyFrom(request)
    del bad.observation["observation.state"]
    assert brain(bad).error.startswith("INVALID_ARGUMENT")
    too_long = pb.InferRequest()
    too_long.CopyFrom(request)
    too_long.max_actions = runner.max_actions + 1
    assert "exceeds" in brain(too_long).error
