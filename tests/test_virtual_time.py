"""Discrete-event core of the simulation-time mode: clock, random streams, latency models, Brain, replay."""

from __future__ import annotations

import dataclasses
import functools
import json
import random
import statistics

import numpy as np
import pytest

from kortyn.config import (
    ConfigError,
    DistributionConfig,
    EmpiricalSourceConfig,
    NetworkConfig,
    StallConfig,
)
from kortyn.metrics import EpisodeRecord, RequestRecord, TickSeries
from kortyn.netem_proxy import DOWN, UP, NetworkState
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.virtual_time import (
    BrainJob,
    ComputeResult,
    Delivery,
    DurationSampler,
    FifoScheduler,
    ModelChannel,
    ModelTiming,
    Priority,
    ReplayChannel,
    ReplayLog,
    ReplayTicks,
    SampledTicks,
    Signal,
    Simulation,
    SimulationError,
    VirtualBrain,
    VirtualLink,
    WakeAt,
    load_delta,
    load_empirical_ms,
    stream_rng,
    stream_seed,
)


# --------------------------------------------------------------------------------------------- clock
def test_events_run_in_time_then_priority_then_insertion_order():
    sim = Simulation()
    log: list[str] = []
    sim.at(2.0, Priority.TICK, lambda: log.append("tick@2"))
    sim.at(1.0, Priority.TICK, lambda: log.append("tick@1"))
    sim.at(2.0, Priority.DELIVERY, lambda: log.append("delivery@2"))
    sim.at(2.0, Priority.BRAIN, lambda: log.append("brain@2a"))
    sim.at(2.0, Priority.BRAIN, lambda: log.append("brain@2b"))
    sim.run()
    assert log == ["tick@1", "delivery@2", "brain@2a", "brain@2b", "tick@2"]
    assert sim.now == 2.0
    assert sim.events_processed == 5


def test_run_until_stops_at_the_horizon_and_advances_the_clock():
    sim = Simulation()
    log: list[float] = []
    for t in (0.5, 1.0, 1.5):
        sim.at(t, Priority.TICK, lambda t=t: log.append(t))
    sim.run(until=1.0)
    assert log == [0.5, 1.0]
    assert sim.now == 1.0
    sim.run(until=1.2)
    assert sim.now == 1.2
    sim.run()
    assert log == [0.5, 1.0, 1.5]


def test_events_in_the_past_are_rejected_but_rounding_is_tolerated():
    sim = Simulation(start=1.0)
    with pytest.raises(SimulationError, match="before now"):
        sim.at(0.5, Priority.TICK, lambda: None)
    sim.at(1.0 - 1e-12, Priority.TICK, lambda: None)  # float rounding: treated as "now"
    with pytest.raises(SimulationError, match="finite"):
        sim.at(float("nan"), Priority.TICK, lambda: None)


def test_processes_wait_for_time_and_signals_and_return_results():
    sim = Simulation()
    signal = Signal(sim)
    sim.at(3.0, Priority.DELIVERY, lambda: signal.trigger("plan"))

    def robot():
        yield WakeAt(1.0)
        first = sim.now
        value = yield signal
        return first, sim.now, value

    assert sim.run_process(robot()) == (1.0, 3.0, "plan")


def test_waiting_on_an_already_triggered_signal_resumes_immediately():
    sim = Simulation()
    signal = Signal(sim)
    signal.trigger(7)

    def waiter():
        value = yield signal
        return value

    assert sim.run_process(waiter()) == 7
    with pytest.raises(SimulationError, match="twice"):
        signal.trigger(8)


def test_deadlock_and_bad_commands_are_reported():
    sim = Simulation()

    def stuck():
        yield Signal(sim)

    with pytest.raises(SimulationError, match="deadlock"):
        sim.run_process(stuck())

    def bad():
        yield 1.0

    with pytest.raises(SimulationError, match="expected WakeAt or Signal"):
        Simulation().run_process(bad())


# --------------------------------------------------------------------------------------------- random
def test_stream_seeds_are_stable_and_independent():
    # Fixed value: the derivation must not change between processes or Python versions (reproducibility).
    assert stream_seed(1000, 0, "kortyn-agent-0", "service", 8) == stream_seed(
        1000, 0, "kortyn-agent-0", "service", 8
    )
    assert stream_seed(1, "a") == 6125863210126610986
    assert stream_seed(1000, "service", 8) != stream_seed(1000, "service", 16)
    assert stream_seed(1000, "service", 8) != stream_seed(1001, "service", 8)
    assert stream_rng(5, "x").random() == stream_rng(5, "x").random()


# --------------------------------------------------------------------------------------------- durations
def test_constant_uniform_and_clipping():
    rng = random.Random(0)
    assert DurationSampler(DistributionConfig(kind="constant", value_ms=40.0)).sample_s(rng) == 0.04
    uniform = DurationSampler(DistributionConfig(kind="uniform", low_ms=10.0, high_ms=20.0))
    values = [uniform.sample_ms(rng) for _ in range(500)]
    assert min(values) >= 10.0
    assert max(values) <= 20.0
    clipped = DurationSampler(
        DistributionConfig(kind="normal", mean_ms=5.0, std_ms=50.0, min_ms=1.0, max_ms=9.0)
    )
    values = [clipped.sample_ms(rng) for _ in range(500)]
    assert min(values) == 1.0
    assert max(values) == 9.0
    scaled = DurationSampler(DistributionConfig(kind="constant", value_ms=100.0, scale=1.5))
    assert scaled.sample_ms(rng) == 150.0
    assert scaled.mean_ms() == 150.0
    assert DurationSampler(DistributionConfig()).is_zero


def test_lognormal_has_the_configured_mean_and_spread():
    sampler = DurationSampler(DistributionConfig(kind="lognormal", mean_ms=100.0, std_ms=10.0))
    rng = random.Random(1)
    values = [sampler.sample_ms(rng) for _ in range(20000)]
    assert statistics.fmean(values) == pytest.approx(100.0, rel=0.01)
    assert statistics.stdev(values) == pytest.approx(10.0, rel=0.05)
    assert min(values) > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "uniform", "low_ms": 5.0, "high_ms": 1.0}, "low_ms <= high_ms"),
        ({"kind": "lognormal", "mean_ms": 0.0}, "mean_ms > 0"),
        ({"kind": "empirical"}, "needs an 'empirical' source"),
        ({"kind": "constant", "empirical": EmpiricalSourceConfig(paths=["x"])}, "given for kind"),
        ({"scale": 0.0}, "scale"),
        ({"min_ms": 5.0, "max_ms": 1.0}, "max_ms"),
    ],
)
def test_invalid_distributions_are_rejected(kwargs, message):
    with pytest.raises(ConfigError, match=message):
        DistributionConfig(**kwargs)


def _episode(seed, attempt=0, valid=True, mode="realtime", lateness=(0.1, 0.2), run="r"):
    ticks = TickSeries(lateness_ms=list(lateness), starved=[0] * len(lateness))
    return EpisodeRecord(
        run=run, profile="p", scenario="s", mode=mode, seed=seed, attempt=attempt, success=False,
        steps=len(lateness), fps=10.0, wall_duration_s=0.2, max_coverage=0.0, sum_reward=0.0, n_requests=1,
        n_failed_requests=0, starved_ticks=0, starvation_events=0, starved_ticks_compute=0,
        starved_ticks_network=0, starved_ticks_none_pending=0, tick_overruns=0, max_tick_lateness_ms=0.2,
        valid=valid, ticks=ticks,
    )  # fmt: skip


def _request(seed, obs, t_obs, t_send, t_resp, attempt=0, phase="realtime", total=300.0, status="ok"):
    return RequestRecord(
        run="r", profile="p", seed=seed, attempt=attempt, request_id=obs + 1, phase=phase, status=status,
        obs_timestep=obs, t_obs_s=t_obs, t_send_s=t_send, request_bytes=100, queue_at_send=0, t_resp_s=t_resp,
        server_queue_ms=0.5, server_inference_ms=total - 0.5, server_total_ms=total,
    )  # fmt: skip


def write_run(run_dir, episodes, requests):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "episodes.jsonl").write_text("".join(json.dumps(e.to_json()) + "\n" for e in episodes))
    (run_dir / "requests.jsonl").write_text("".join(json.dumps(r.to_json()) + "\n" for r in requests))


def test_empirical_samples_use_final_attempts_of_realtime_runs_only(tmp_path):
    write_run(
        tmp_path / "results" / "p" / "gut",
        [_episode(1, attempt=0, valid=False), _episode(1, attempt=1, lateness=(0.3,)), _episode(2)],
        [
            _request(1, 8, 0.8, 0.8004, 1.2, attempt=0, total=999.0),  # invalid attempt: ignored
            _request(1, 8, 0.8, 0.8003, 1.2, attempt=1, total=310.0),
            _request(2, 8, 0.8, 0.8002, 1.2, total=320.0),
            _request(2, 0, -0.3, -0.3, 0.0, phase="warmup", total=330.0),  # warm-up: ignored
            _request(2, 16, 1.6, 1.6, 2.0, total=777.0, status="timeout"),  # failed: ignored
        ],
    )
    write_run(
        tmp_path / "results" / "p" / "referenz_sync", [_episode(1, mode="sync")], [_request(1, 0, 0, 0, 1)]
    )
    service = EmpiricalSourceConfig(paths=["results/p/*"], quantity="server_total_ms")
    assert load_empirical_ms(service, root=tmp_path) == (310.0, 320.0)
    offsets = load_empirical_ms(
        EmpiricalSourceConfig(paths=["results/p/gut"], quantity="send_offset_ms"), tmp_path
    )
    assert offsets == pytest.approx((0.2, 0.3))
    ticks = load_empirical_ms(
        EmpiricalSourceConfig(paths=["results/p/gut"], quantity="tick_lateness_ms"), tmp_path
    )
    assert ticks == (0.1, 0.2, 0.3)
    sampler = DurationSampler(DistributionConfig(kind="empirical", empirical=service), root=tmp_path)
    assert {sampler.sample_ms(random.Random(i)) for i in range(50)} == {310.0, 320.0}
    with pytest.raises(ConfigError, match="no logged"):
        load_empirical_ms(EmpiricalSourceConfig(paths=["results/none/*"]), root=tmp_path)


# --------------------------------------------------------------------------------------------- network
def test_virtual_link_is_the_proxy_model_on_the_simulation_clock():
    network = NetworkConfig(
        rtt_ms=80.0, rtt_jitter_ms=40.0, stall=StallConfig(interval_s=2.0, duration_ms=300.0)
    )
    sim = Simulation()
    link = VirtualLink(sim, network, seed=1003)
    reference = NetworkState(network, 1003, clock=lambda: 0.0)
    reference.arm()
    link.arm()
    for t in np.arange(0.0, 6.0, 0.37):
        for direction in (UP, DOWN):
            delivery, held = link.transmit(direction, float(t), 55_000)
            assert delivery == reference.links[direction].schedule(float(t), 55_000)
            assert held == (delivery > t)
    stats = link.stats()
    assert stats["bytes_up"] == 55_000 * len(np.arange(0.0, 6.0, 0.37))
    assert stats["connections_opened_since_configure"] == 0
    assert stats["stall_windows_s"] == reference.stats_dict()["stall_windows_s"] or stats["armed"]


def test_stall_holds_a_message_until_the_window_ends():
    network = NetworkConfig(stall=StallConfig(interval_s=10.0, duration_ms=1000.0))
    sim = Simulation()
    link = VirtualLink(sim, network, seed=7)
    link.arm()
    start = link.state.stalls.phase_s
    delivery, held = link.transmit(UP, start + 0.2, 100)
    assert held
    assert delivery == pytest.approx(start + 1.0)


# --------------------------------------------------------------------------------------------- brain
class EchoCompute:
    """Plan = [[obs_step + j, request_id]]: shows which request produced an action."""

    def __init__(self, error: str | None = None) -> None:
        self.calls: list[int] = []
        self.cache_hits = 0
        self.error = error

    def __call__(self, request: pb.InferRequest) -> ComputeResult:
        self.calls.append(int(request.obs_timestep))
        if self.error is not None:
            return ComputeResult(None, self.error)
        n = int(request.max_actions or 15)
        actions = np.array(
            [[request.obs_timestep + j, request.request_id] for j in range(n)], dtype=np.float32
        )
        return ComputeResult(actions, compute_ms=1.0)


def make_request(obs_step: int, request_id: int, max_actions: int = 15) -> pb.InferRequest:
    return pb.InferRequest(
        client_id="c", request_id=request_id, obs_timestep=obs_step, max_actions=max_actions
    )


def job(sim, request_id, arrival, service, done, deadline=None):
    return BrainJob(
        client_id="c", request=make_request(request_id, request_id), t_sent=arrival, t_arrival=arrival,
        service_s=service, deadline=deadline, on_done=done.append,
    )  # fmt: skip


def test_fifo_brain_serves_one_request_after_the_other():
    sim = Simulation()
    brain = VirtualBrain(sim, EchoCompute(), FifoScheduler())
    done: list[BrainJob] = []
    for request_id, arrival in ((1, 0.0), (2, 0.1), (3, 0.15)):
        sim.at(
            arrival, Priority.BRAIN, functools.partial(brain.submit, job(sim, request_id, arrival, 0.3, done))
        )
    sim.run()
    assert [j.request.request_id for j in done] == [1, 2, 3]
    assert [j.t_start for j in done] == pytest.approx([0.0, 0.3, 0.6])
    assert [j.t_done for j in done] == pytest.approx([0.3, 0.6, 0.9])
    assert brain.counters.queue_wait_ms == pytest.approx(1000.0 * (0.2 + 0.45))
    assert brain.counters.busy_ms == pytest.approx(900.0)
    snapshot = brain.counters.snapshot(0.9)
    assert load_delta({**snapshot, "requests": 1.0, "uptime_ms": 0.0}, snapshot)["requests"] == 2.0


def test_two_servers_work_in_parallel_and_abandoned_requests_are_skipped():
    sim = Simulation()
    compute = EchoCompute()
    brain = VirtualBrain(sim, compute, FifoScheduler(), servers=2)
    done: list[BrainJob] = []
    sim.at(0.0, Priority.BRAIN, lambda: brain.submit(job(sim, 1, 0.0, 0.5, done)))
    sim.at(0.0, Priority.BRAIN, lambda: brain.submit(job(sim, 2, 0.0, 0.5, done)))
    # waits until 0.5 for a server, but its client gave up at 0.4: never computed
    sim.at(0.1, Priority.BRAIN, lambda: brain.submit(job(sim, 3, 0.1, 0.5, done, deadline=0.4)))
    sim.run()
    assert [j.t_done for j in done] == pytest.approx([0.5, 0.5])
    assert brain.counters.abandoned == 1
    assert compute.calls == [1, 2]
    with pytest.raises(SimulationError):
        VirtualBrain(sim, compute, FifoScheduler(), servers=0)


# --------------------------------------------------------------------------------------------- channels
def model_timing(service_ms=300.0, client_ms=0.0, transport_ms=0.0, lateness_ms=0.0):
    def constant(value):
        return DurationSampler(DistributionConfig(kind="constant", value_ms=value))

    return ModelTiming(
        constant(service_ms), constant(client_ms), constant(transport_ms), constant(lateness_ms)
    )


def test_model_channel_composes_link_queue_and_service_time():
    sim = Simulation()
    compute = EchoCompute()
    brain = VirtualBrain(sim, compute, FifoScheduler())
    link = VirtualLink(sim, NetworkConfig(rtt_ms=40.0), seed=1)
    channel = ModelChannel(sim, brain, link, model_timing(300.0, 0.5, 3.0, 1.0), 1000, 0, "c")
    link.arm()
    received: list[Delivery] = []
    request = make_request(8, 1)
    channel.send(request, t_obs=0.0, t_send=0.0005, timeout_s=5.0, deliver=received.append)
    sim.run()
    (delivery,) = received
    # client 0.5 ms (applied by the agent) + transport 3 ms + 20 ms + 1 ms lateness + 300 ms + 20 ms + 1 ms
    assert delivery.status == "ok"
    assert delivery.t_resp == pytest.approx(0.0005 + 0.003 + 0.021 + 0.300 + 0.021)
    assert delivery.response is not None
    assert delivery.response.server_total_ms == pytest.approx(300.0)
    assert delivery.response.start_timestep == 8
    assert channel.client_overhead_s(8) == pytest.approx(0.0005)
    assert link.stats()["max_delivery_lateness_ms"] == pytest.approx(1.0)


def test_model_channel_times_out_and_the_brain_skips_the_abandoned_request():
    sim = Simulation()
    compute = EchoCompute()
    brain = VirtualBrain(sim, compute, FifoScheduler())
    channel = ModelChannel(sim, brain, VirtualLink(sim, NetworkConfig(), 1), model_timing(600.0), 1, 0, "c")
    received: list[Delivery] = []
    for step, request_id in ((0, 1), (8, 2), (16, 3)):  # all sent at t=0 with a 1 s deadline
        channel.send(make_request(step, request_id), 0.0, 0.0, timeout_s=1.0, deliver=received.append)
    sim.run()
    assert [(d.request_id, d.status) for d in received] == [(1, "ok"), (2, "timeout"), (3, "timeout")]
    assert [d.t_resp for d in received] == pytest.approx([0.6, 1.0, 1.0])
    # #2 starts at 0.6 s (client still waiting) and is computed although its answer comes too late;
    # #3 would start at 1.2 s, after its client gave up: skipped, like the real Brain does
    assert compute.calls == [0, 8]
    assert brain.counters.abandoned == 1


def test_model_channel_reports_compute_errors():
    sim = Simulation()
    brain = VirtualBrain(sim, EchoCompute(error="INVALID_ARGUMENT: bad"), FifoScheduler())
    channel = ModelChannel(sim, brain, VirtualLink(sim, NetworkConfig(), 1), model_timing(), 1, 0, "c")
    received: list[Delivery] = []
    channel.send(make_request(0, 1), 0.0, 0.0, timeout_s=1.0, deliver=received.append)
    sim.run()
    assert [(d.status, d.error) for d in received] == [("error", "INVALID_ARGUMENT: bad")]


def test_service_times_are_keyed_by_seed_and_step_not_by_call_order():
    def delivery_time(order):
        sim = Simulation()
        brain = VirtualBrain(sim, EchoCompute(), FifoScheduler(), servers=4)
        sampler = DurationSampler(DistributionConfig(kind="uniform", low_ms=100.0, high_ms=500.0))
        timing = dataclasses.replace(model_timing(), service=sampler)
        channel = ModelChannel(sim, brain, VirtualLink(sim, NetworkConfig(), 1), timing, 1000, 0, "c")
        received: dict[int, float] = {}
        for step in order:
            channel.send(
                make_request(step, step + 1),
                0.0,
                0.0,
                5.0,
                lambda d: received.update({d.request_id: d.t_resp}),
            )
        sim.run()
        return received

    assert delivery_time([0, 8, 16]) == delivery_time([16, 0, 8])


# --------------------------------------------------------------------------------------------- replay
def test_replay_plays_back_logged_requests_exactly_and_falls_back_to_the_nearest(tmp_path):
    run_dir = tmp_path / "gut"
    write_run(
        run_dir,
        [_episode(1000, lateness=(0.25, 0.1, 0.4))],
        [
            _request(1000, 0, -0.34, -0.34, -0.0002, phase="warmup", total=339.0),
            _request(1000, 8, 0.80025, 0.8006, 1.1381, total=334.0),
            _request(1000, 16, 1.6002, 1.6006, 2.0381, total=430.0),
        ],
    )
    log = ReplayLog(run_dir)
    episode = log.episode(1000)
    assert sorted(episode.requests) == [8, 16]
    sim = Simulation(start=episode.warmup.t_obs_s)
    channel = ReplayChannel(sim, episode, EchoCompute())
    received: list[Delivery] = []
    channel.send(make_request(0, 1), sim.now, sim.now, 60.0, received.append)
    sim.run()
    assert received[-1].t_resp == -0.0002
    assert channel.episode_start(sim.now) == 0.0
    assert channel.client_overhead_s(8) == pytest.approx(0.00035)
    channel.send(make_request(8, 2), 0.80025, 0.8006, 5.0, received.append)
    sim.run()
    assert received[-1].t_resp == 1.1381
    assert received[-1].response.server_total_ms == 334.0
    sim.run(until=2.4)
    channel.send(make_request(24, 3), 2.4, 2.4, 5.0, received.append)  # not logged: nearest is step 16
    sim.run()
    assert received[-1].t_resp == pytest.approx(2.4 + (2.0381 - 1.6002))
    assert (channel.counts.exact, channel.counts.fallback) == (2, 1)
    assert channel.brain_load()["requests"] == 3.0
    ticks = ReplayTicks(episode.tick_lateness_ms)
    assert [ticks.lateness_s(k) for k in (0, 2, 3)] == pytest.approx([0.00025, 0.0004, 0.0])
    with pytest.raises(ConfigError, match="not logged"):
        log.episode(1)
    with pytest.raises(ConfigError, match="no logged run"):
        ReplayLog(tmp_path / "missing")


def test_sampled_ticks_are_ideal_by_default_and_reproducible_when_jittered():
    ideal = SampledTicks(DurationSampler(DistributionConfig()), 1000, 0, "c")
    assert {ideal.lateness_s(k) for k in range(20)} == {0.0}
    jitter = DistributionConfig(kind="uniform", low_ms=0.0, high_ms=10.0)
    a = SampledTicks(DurationSampler(jitter), 1000, 0, "c")
    b = SampledTicks(DurationSampler(jitter), 1000, 0, "c")
    other = SampledTicks(DurationSampler(jitter), 1000, 1, "c")
    assert [a.lateness_s(k) for k in range(20)] == [b.lateness_s(k) for k in range(20)]
    assert [a.lateness_s(k) for k in range(20)] != [other.lateness_s(k) for k in range(20)]
    assert max(a.lateness_s(k) for k in range(200)) <= 0.010
