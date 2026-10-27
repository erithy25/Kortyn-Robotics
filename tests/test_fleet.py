"""Fleet simulation (phase 3, stage 1): timing robots equal the agent, budgets, grid, reproducibility."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from kortyn.agent import EpisodeContext
from kortyn.config import (
    AgentConfig,
    ConfigError,
    DistributionConfig,
    NetworkConfig,
    StrategyConfig,
    TimingConfig,
)
from kortyn.fleet import (
    BatchingConfig,
    EpisodeLengthConfig,
    FleetAgentConfig,
    FleetSimConfig,
    GpuConfig,
    NetworkChoice,
    RequestLog,
    RobotStats,
    RunKey,
    TimingRobot,
    _ZeroCompute,
    done_keys,
    grid,
    load_episode_lengths,
    load_fleet_config,
    run_fleet,
    run_grid,
    start_times,
)
from kortyn.fleet import (
    StrategyConfig as FleetStrategy,
)
from kortyn.fleet_brain import AffineBatchProfile, BatchingBrain, FifoScheduler
from kortyn.virtual_agent import VirtualTimeAgent
from kortyn.virtual_time import (
    Command,
    DurationSampler,
    ModelChannel,
    ModelTiming,
    SampledTicks,
    Simulation,
    VirtualLink,
)
from tests.test_agent import FPS, FakeEnv
from tests.test_virtual_agent import SPEC, FakeCompute, model_samplers


def zero() -> DurationSampler:
    return DurationSampler(DistributionConfig(kind="constant", value_ms=0.0))


def agent_requests(latency_ms: float, length: int) -> tuple[list[tuple[int, float, float]], int]:
    """Request timeline of the real virtual-time agent behind a FIFO batching Brain (cold start)."""
    sim = Simulation()
    compute = FakeCompute()
    brain = BatchingBrain(
        sim, compute, FifoScheduler(), AffineBatchProfile(s1_ms=latency_ms, beta=1.0, cv=0.0), seed=1
    )
    timing = ModelTiming(zero(), zero(), zero(), zero())
    channel = ModelChannel(sim, brain, VirtualLink(sim, NetworkConfig(), 5), timing, 1000, 0, "robot-0")
    config = AgentConfig(actions_per_chunk=15, refill_threshold=6, start_mode="cold")
    agent = VirtualTimeAgent(
        config,
        FakeEnv(max_steps=length),
        SPEC,
        compute,
        TimingConfig(latency="model", hardware="test"),
        model_samplers(DistributionConfig(kind="constant", value_ms=0.0)),
    )
    ctx = EpisodeContext(run="test", profile="fake", scenario="test", mode="realtime", seed=1000)
    ticks = SampledTicks(zero(), 1000, 0, "robot-0")
    outcome = sim.run_process(agent.episode_process(ctx, StrategyConfig(), sim, channel, ticks, brain))
    requests = [
        (r.obs_timestep, r.t_obs_s, float(r.t_resp_s or -1.0))
        for r in sorted(outcome.requests, key=lambda r: r.request_id)
    ]
    return requests, outcome.episode.starved_ticks


def robot_requests(latency_ms: float, length: int) -> tuple[list[tuple[int, float, float]], RobotStats]:
    sim = Simulation()
    brain = BatchingBrain(
        sim,
        _ZeroCompute(15, 2),
        FifoScheduler(),
        AffineBatchProfile(s1_ms=latency_ms, beta=1.0, cv=0.0),
        seed=1,
    )
    agent = FleetAgentConfig(
        fps=FPS, actions_per_chunk=15, refill_threshold=6, client_overhead_ms=0.0, transport_overhead_ms=0.0
    )
    stats = RobotStats(robot=0, start_time=0.0)
    robot = TimingRobot(0, sim, brain, agent, "gut", [length], 7, stats, 1e9, _ZeroCompute(15, 2))
    sim.run_process(robot.episode(0, length))
    return [(r.obs_step, r.t_obs, float(r.t_resp or -1.0)) for r in stats.requests], stats


@pytest.mark.parametrize("latency_ms", [1.0, 120.0, 330.0, 360.0, 410.0, 900.0])
def test_timing_robot_reproduces_the_agents_request_timeline(latency_ms: float) -> None:
    """Same latency model -> identical request steps, send and delivery times and starvation as the agent."""
    expected, starved = agent_requests(latency_ms, 80)
    got, stats = robot_requests(latency_ms, 80)
    assert [r[0] for r in got] == [r[0] for r in expected]
    assert [r[1:] for r in got] == pytest.approx([r[1:] for r in expected])
    assert stats.starved_ticks == starved
    assert stats.ticks == 80


def test_request_log_budget_rules() -> None:
    regular = RequestLog(robot=0, episode=0, obs_step=25, t_obs=1.25, t_send=1.251, due=2.5, start=False)
    regular.t_resp, regular.status = 2.5, "ok"
    assert not regular.missed(1.25)  # arrives exactly at the tick that would starve: merged in time
    regular.t_resp = 2.5001
    assert regular.missed(1.25)
    start = RequestLog(robot=0, episode=0, obs_step=0, t_obs=0.0, t_send=0.0, due=0.05, start=True)
    start.t_resp, start.status = 1.2, "ok"
    assert not start.missed(1.25)  # start requests: nominal budget from the observation
    start.t_resp = 1.3
    assert start.missed(1.25)
    timeout = RequestLog(robot=0, episode=0, obs_step=0, t_obs=0.0, t_send=0.0, due=0.05, start=True)
    timeout.t_resp, timeout.status = 10.0, "timeout"
    assert timeout.missed(1.25)


def small_config(tmp_path: Path, **overrides: object) -> FleetSimConfig:
    settings: dict[str, object] = {
        "agent": FleetAgentConfig(fps=20.0, actions_per_chunk=50, refill_threshold=24),
        "episode_lengths": EpisodeLengthConfig(values=[200, 300, 520]),
        "gpus": {"gpu": GpuConfig(s1_ms=150.0)},
        "batching": {"keiner": BatchingConfig(beta=1.0), "gut": BatchingConfig(beta=0.2)},
        "strategies": {
            "a": FleetStrategy(scheduler="fifo"),
            "d": FleetStrategy(scheduler="edf_batch", max_batch=8),
            "e": FleetStrategy(scheduler="edf_batch", max_batch=8, admission=True),
        },
        "networks": {"buero_wlan": NetworkChoice(scenario="buero_wlan", margin_ms=15.0)},
        "fleet_sizes": [1, 4],
        "replications": 2,
        "horizon_s": 120.0,
        "window_start_s": 20.0,
        "results_path": str(tmp_path / "stage1.jsonl"),
    }
    settings.update(overrides)
    return FleetSimConfig(**settings)  # type: ignore[arg-type]


def test_budget_and_period_of_the_agent() -> None:
    agent = FleetAgentConfig(fps=20.0, actions_per_chunk=50, refill_threshold=24)
    assert agent.budget_s == pytest.approx(1.25)
    assert agent.period_s == pytest.approx(1.25)
    assert FleetAgentConfig(fps=20.0, actions_per_chunk=50, refill_threshold=40).period_s == pytest.approx(
        0.45
    )
    assert FleetAgentConfig(fps=20.0, actions_per_chunk=50, refill_threshold=49).period_s == pytest.approx(
        0.05
    )


def test_single_robot_fast_gpu_misses_nothing(tmp_path: Path) -> None:
    config = small_config(tmp_path)
    result = run_fleet(config, RunKey("buero_wlan", "gpu", "keiner", "a", 1, 0), (200, 300, 520))
    assert result.requests > 50
    assert result.misses == 0
    robot = result.robots[0]
    assert robot.admitted
    assert robot.latency_ms_p50 == pytest.approx(150 + 1 + 5 + 10, abs=8)  # service + overheads + ~RTT
    assert sum(robot.histogram.values()) == robot.requests


def test_overloaded_fifo_misses_and_batching_helps(tmp_path: Path) -> None:
    config = small_config(tmp_path, gpus={"slow": GpuConfig(s1_ms=400.0)}, fleet_sizes=[8])
    lengths = (200, 300, 520)
    fifo = run_fleet(config, RunKey("buero_wlan", "slow", "keiner", "a", 8, 0), lengths)
    batched = run_fleet(config, RunKey("buero_wlan", "slow", "gut", "d", 8, 0), lengths)
    assert fifo.misses / fifo.requests > 0.2  # 8 robots x 0.4 s per 1.25 s period: overloaded
    assert batched.misses / batched.requests < fifo.misses / fifo.requests
    assert batched.mean_batch > 1.0


def test_admission_control_rejects_robots_it_cannot_serve(tmp_path: Path) -> None:
    config = small_config(tmp_path, gpus={"slow": GpuConfig(s1_ms=400.0)}, fleet_sizes=[8])
    result = run_fleet(config, RunKey("buero_wlan", "slow", "keiner", "e", 8, 0), (200, 300, 520))
    assert 0 < result.admitted < 8
    assert sum(not r.admitted for r in result.robots) == 8 - result.admitted
    assert all(r.requests == 0 for r in result.robots if not r.admitted)
    assert result.misses == 0


def test_runs_are_reproducible_and_start_times_shared(tmp_path: Path) -> None:
    config = small_config(tmp_path)
    key = RunKey("buero_wlan", "gpu", "gut", "d", 4, 1)
    first = run_fleet(config, key, (200, 300, 520)).to_json()
    second = run_fleet(config, key, (200, 300, 520)).to_json()
    first.pop("wall_s")
    second.pop("wall_s")
    assert first == second
    assert start_times(config, 4, 1) == start_times(config, 4, 1)
    assert start_times(config, 4, 1) != start_times(config, 4, 0)


def test_grid_skips_batching_profiles_for_strategies_without_batching(tmp_path: Path) -> None:
    config = small_config(tmp_path)
    keys = grid(config)
    a_runs = [k for k in keys if k.strategy == "a"]
    d_runs = [k for k in keys if k.strategy == "d"]
    assert {k.batching for k in a_runs} == {"keiner"}
    assert {k.batching for k in d_runs} == {"keiner", "gut"}
    assert len(keys) == (1 + 2 + 2) * 2 * 2


def test_run_grid_writes_and_resumes(tmp_path: Path) -> None:
    config = small_config(tmp_path, fleet_sizes=[1], replications=1, horizon_s=60.0, window_start_s=10.0)
    out = Path(config.results_path)
    keys = grid(config)
    assert run_grid(config, keys, out, workers=1) == len(keys)
    assert done_keys(out) == set(keys)
    assert run_grid(config, keys, out, workers=1) == 0  # everything stored
    record = json.loads(out.read_text().splitlines()[0])
    assert {"key", "requests", "misses", "robots", "miss_rate"} <= set(record)


def test_episode_lengths_from_logged_runs(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    rows = [
        {"seed": 1000, "attempt": 0, "steps": 100},
        {"seed": 1000, "attempt": 1, "steps": 150},  # final attempt counts
        {"seed": 1001, "attempt": 0, "steps": 520},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert load_episode_lengths(EpisodeLengthConfig(paths=[str(path)])) == (150, 520)
    with pytest.raises(ConfigError):
        load_episode_lengths(EpisodeLengthConfig(paths=[str(tmp_path / "missing.jsonl")]))


def test_fleet_config_validation_and_loading(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        FleetStrategy(scheduler="fifo", max_batch=4)
    with pytest.raises(ConfigError):
        FleetStrategy(scheduler="edf", admission=True)
    with pytest.raises(ConfigError):
        EpisodeLengthConfig()
    with pytest.raises(ConfigError):
        FleetAgentConfig(refill_threshold=50)
    path = tmp_path / "fleet.yaml"
    path.write_text(
        "gpus: {g: {s1_ms: 100}}\nbatching: {b: {beta: 0.5}}\nstrategies: {a: {scheduler: fifo}}\n"
        "networks: {n: {scenario: gut, margin_ms: 0}}\nepisode_lengths: {values: [100]}\n"
    )
    config = load_fleet_config(path)
    assert config.gpus["g"].s1_ms == 100
    path.write_text("gpus: {g: {s1_ms: 100, unknown: 1}}\n")
    with pytest.raises(ConfigError):
        load_fleet_config(path)


def test_grid_subsets(tmp_path: Path) -> None:
    config = small_config(tmp_path)
    keys = grid(config, networks=["buero_wlan"], gpus=["gpu"], batching=["gut"])
    assert {k.batching for k in keys if k.strategy == "d"} == {"gut"}
    assert {k.batching for k in keys if k.strategy == "a"} == {"keiner"}  # no batching: first profile
    with pytest.raises(ConfigError):
        grid(config, gpus=["unknown"])
    with pytest.raises(ConfigError):
        grid(config, networks=[])
    only_d = grid(config, strategies=["d"])
    assert {k.strategy for k in only_d} == {"d"}
    assert len(only_d) == len([k for k in grid(config) if k.strategy == "d"])
    with pytest.raises(ConfigError):
        grid(config, strategies=["z"])


def test_phase3_fleet_config_loads() -> None:
    config = load_fleet_config(Path(__file__).resolve().parents[1] / "kortyn/experiments/phase3/fleet.yaml")
    assert config.agent.budget_s == pytest.approx(1.25)
    assert config.agent.period_s == pytest.approx(1.25)
    assert set(config.strategies) == {"a", "b", "c", "d", "e"}
    assert config.validation is not None
    assert config.validation.gpu == "validierung"
    assert config.prediction is not None
    assert config.prediction.levels[0].latency_ms == 0.0


def test_scan_stops_after_consecutive_overload(tmp_path: Path) -> None:
    from kortyn.fleet import ScanSettings, run_scan

    config = small_config(
        tmp_path, gpus={"slow": GpuConfig(s1_ms=400.0)}, horizon_s=60.0, window_start_s=10.0
    )
    out = tmp_path / "scan.jsonl"
    base = RunKey("buero_wlan", "slow", "keiner", "a", 1, 0)
    written = run_scan(
        config, [base], ScanSettings(max_size=12, replications=1, stop_after=2), out, workers=1
    )
    sizes = sorted(json.loads(line)["key"]["robots"] for line in out.read_text().splitlines())
    assert written == len(sizes)
    # FIFO with 0.4 s per plan every 1.25 s: overloaded from 4 robots on; two failing sizes in a row stop it
    assert sizes[0] == 1
    assert max(sizes) < 12
    assert (
        run_scan(config, [base], ScanSettings(max_size=12, replications=1, stop_after=2), out, workers=1) == 0
    )
    with pytest.raises(ConfigError):
        ScanSettings(max_size=0)


def test_scan_can_stop_at_the_first_rejected_robot(tmp_path: Path) -> None:
    from kortyn.fleet import ScanSettings, run_scan

    config = small_config(
        tmp_path, gpus={"slow": GpuConfig(s1_ms=400.0)}, horizon_s=60.0, window_start_s=10.0
    )
    base = RunKey("buero_wlan", "slow", "keiner", "e", 1, 0)
    phase3 = tmp_path / "phase3.jsonl"
    run_scan(config, [base], ScanSettings(max_size=10, replications=1, stop_after=2), phase3, workers=1)
    phase4 = tmp_path / "phase4.jsonl"
    settings = ScanSettings(max_size=10, replications=1, stop_after=2, stop_on_rejection=True)
    run_scan(config, [base], settings, phase4, workers=1)
    records = [json.loads(line) for line in phase4.read_text().splitlines()]
    sizes = sorted(record["key"]["robots"] for record in records)
    # admission control never misses the budget, so phase 3 scans to the end; phase 4 stops at the first
    # size whose fleet it cannot fully admit
    assert max(json.loads(line)["key"]["robots"] for line in phase3.read_text().splitlines()) == 10
    largest = max(records, key=lambda record: record["key"]["robots"])
    assert largest["admitted"] < largest["key"]["robots"]
    assert all(record["admitted"] == record["key"]["robots"] for record in records if record is not largest)
    assert sizes == list(range(1, max(sizes) + 1))


def test_timing_robots_reproduce_agents_sharing_an_edf_batching_brain() -> None:
    """Fleet semantics: request steps, due times, batching and deliveries equal the agents' (3 robots)."""
    from kortyn.fleet_brain import EdfBatchScheduler
    from kortyn.virtual_time import Priority, WakeAt

    length, starts = 90, (0.0, 0.37, 0.61)
    model = AffineBatchProfile(s1_ms=650.0, beta=0.3, cv=0.0)  # overloaded: requests wait and get batched
    timing = ModelTiming(zero(), zero(), zero(), zero())

    def agents_run() -> list[list[tuple[int, float, float]]]:
        sim = Simulation()
        compute = FakeCompute()
        brain = BatchingBrain(sim, compute, EdfBatchScheduler(max_batch=4), model, seed=1)
        timelines: list[list[tuple[int, float, float]]] = [[], [], []]

        def robot(index: int) -> Generator[Command, Any, None]:
            yield WakeAt(starts[index], Priority.TICK)
            config = AgentConfig(
                actions_per_chunk=15, refill_threshold=6, start_mode="cold", client_id=f"robot-{index}"
            )
            agent = VirtualTimeAgent(
                config,
                FakeEnv(max_steps=length),
                SPEC,
                compute,
                TimingConfig(latency="model", hardware="test"),
                model_samplers(DistributionConfig(kind="constant", value_ms=0.0)),
            )
            link = VirtualLink(sim, NetworkConfig(), 5)
            channel = ModelChannel(sim, brain, link, timing, 1000, 0, f"robot-{index}")
            ctx = EpisodeContext(
                run="test", profile="fake", scenario="test", mode="realtime", seed=1000 + index
            )
            ticks = SampledTicks(zero(), 1000 + index, 0, f"robot-{index}")
            outcome = yield from agent.episode_process(ctx, StrategyConfig(), sim, channel, ticks, brain)
            timelines[index] = [
                (r.obs_timestep, r.t_obs_s, float(r.t_resp_s or -1.0))
                for r in sorted(outcome.requests, key=lambda r: r.request_id)
            ]

        for index in range(3):
            sim.process(robot(index))
        sim.run()
        return timelines

    def robots_run() -> tuple[list[list[tuple[int, float, float]]], dict[int, int]]:
        sim = Simulation()
        compute = _ZeroCompute(15, 2)
        brain = BatchingBrain(sim, compute, EdfBatchScheduler(max_batch=4), model, seed=1)
        agent = FleetAgentConfig(
            fps=FPS,
            actions_per_chunk=15,
            refill_threshold=6,
            client_overhead_ms=0.0,
            transport_overhead_ms=0.0,
        )
        stats = [RobotStats(robot=i, start_time=starts[i]) for i in range(3)]

        def robot(index: int) -> Generator[Command, Any, None]:
            yield WakeAt(starts[index], Priority.TICK)
            timing_robot = TimingRobot(
                index, sim, brain, agent, "gut", [length], 7, stats[index], 1e9, compute
            )
            yield from timing_robot.episode(0, length)

        for index in range(3):
            sim.process(robot(index))
        sim.run()
        # the agent records times on its episode clock (0 at the episode start)
        timelines = [
            [(r.obs_step, r.t_obs - s.start_time, float(r.t_resp or -1.0) - s.start_time) for r in s.requests]
            for s in stats
        ]
        return timelines, dict(brain.counters.batch_sizes)

    expected = agents_run()
    got, batch_sizes = robots_run()
    for index in range(3):
        assert [r[0] for r in got[index]] == [r[0] for r in expected[index]], index
        assert [r[1:] for r in got[index]] == pytest.approx([r[1:] for r in expected[index]]), index
    assert max(batch_sizes) >= 2  # requests of several robots were batched
