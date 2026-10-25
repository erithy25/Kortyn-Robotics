"""Fleet simulation of the Brain's queue (phase 3, stage 1): many robots, one GPU, no policy.

Each robot is a :class:`TimingRobot`: it follows exactly the request, merge and starvation rules of the
virtual-time agent (``VirtualTimeAgent._realtime`` with cold start, a refill threshold, timestep alignment
and at most one request in flight), but without simulator and policy, so a whole fleet of 32 robots runs
for ten minutes of simulated time in about a second. Robots talk to one shared
:class:`~kortyn.fleet_brain.BatchingBrain` through the same :class:`~kortyn.virtual_time.ModelChannel`
(client overhead, the scenario's link with its own seeds per robot and episode, transport overhead) that the
real rollouts use, so stage 1 and the validation rollouts share every scheduling and network line of code.

Episodes end after a number of control ticks drawn from measured episode lengths (success ends an episode
early); then the robot resets (instantly) and starts the next one cold. For every request the run records
the end-to-end latency and whether it missed its budget: the plan arrived after the tick at which the
robot's queue ran dry (episode-start requests: later than the nominal budget after the observation).

Run a grid of configurations (strategy x fleet size x GPU x batching profile x network x replication):

    python -m kortyn.fleet --config kortyn/experiments/phase3/fleet.yaml
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import logging
import math
import os
import sys
import time
from collections.abc import Generator, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from kortyn.config import ConfigError, DistributionConfig, decode, load_scenario
from kortyn.fleet_brain import (
    AdmissionController,
    AffineBatchProfile,
    BatchingBrain,
    BatchServiceModel,
    SchedulerName,
    TableBatchProfile,
    make_batch_scheduler,
)
from kortyn.metrics import read_jsonl
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.virtual_time import (
    TIME_EPS_S,
    Command,
    ComputeResult,
    Delivery,
    DurationSampler,
    ModelChannel,
    ModelTiming,
    Priority,
    Signal,
    Simulation,
    VirtualLink,
    WakeAt,
    stream_rng,
    stream_seed,
)

LOG = logging.getLogger("kortyn.fleet")

HISTOGRAM_BIN_MS = 10.0  # latency histograms per robot: 10 ms bins (prediction interpolates within)


# --------------------------------------------------------------------------------------------- config
@dataclass
class FleetAgentConfig:
    """The robots' control loop (must match the agent of the rollouts the prediction is built on)."""

    fps: float = 20.0
    actions_per_chunk: int = 50
    refill_threshold: int = 24
    request_timeout_s: float = 10.0
    client_overhead_ms: float = 1.0
    transport_overhead_ms: float = 5.0

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.actions_per_chunk < 1:
            raise ConfigError("fleet agent needs fps > 0 and actions_per_chunk >= 1")
        if not 0 <= self.refill_threshold < self.actions_per_chunk:
            raise ConfigError("fleet agent refill_threshold must be in [0, actions_per_chunk)")
        if self.request_timeout_s <= 0 or self.client_overhead_ms < 0 or self.transport_overhead_ms < 0:
            raise ConfigError("fleet agent timeout must be > 0 and overheads >= 0")

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def budget_s(self) -> float:
        """Latency the action queue bridges after a request: (threshold + 1) ticks."""
        return (self.refill_threshold + 1) * self.dt

    @property
    def period_s(self) -> float:
        """Request period while plans arrive on time: chunk - threshold - 1 ticks (at least one).

        A plan for step ``s`` covers ``s .. s + chunk - 1``; the next request goes out at the first step
        with at most ``threshold`` actions left after it, ``s + chunk - 1 - threshold``.
        """
        return max(1, self.actions_per_chunk - self.refill_threshold - 1) * self.dt


@dataclass
class EpisodeLengthConfig:
    """Episode lengths in ticks: resampled from logged episodes, or fixed values (tests)."""

    paths: list[str] = field(default_factory=list)  # episodes.jsonl files
    values: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.paths and not self.values:
            raise ConfigError("episode_lengths needs 'paths' (episodes.jsonl) or 'values'")
        if any(v < 1 for v in self.values):
            raise ConfigError("episode lengths must be >= 1 tick")


@dataclass
class GpuConfig:
    """Single-request service time of one GPU for the fleet's workload."""

    description: str = ""
    s1_ms: float = 100.0
    # Measured table (benchmark script); replaces the affine batching profiles for this GPU when set.
    table: str | None = None

    def __post_init__(self) -> None:
        if self.s1_ms <= 0:
            raise ConfigError(f"gpu s1_ms must be > 0, got {self.s1_ms}")


@dataclass
class BatchingConfig:
    """Named batching profile: each extra request in a batch costs ``beta`` single inferences."""

    description: str = ""
    beta: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ConfigError(f"batching beta must be in [0, 1], got {self.beta}")


@dataclass
class StrategyConfig:
    """Scheduling strategy of the Brain (phase 3 a-e)."""

    description: str = ""
    scheduler: Literal["fifo", "fifo_batch", "edf", "edf_batch"] = "fifo"
    max_batch: int = 1
    max_wait_ms: float = 0.0
    admission: bool = False
    max_utilization: float = 0.9

    def __post_init__(self) -> None:
        if self.max_batch < 1 or self.max_wait_ms < 0:
            raise ConfigError("strategy needs max_batch >= 1 and max_wait_ms >= 0")
        if self.scheduler in ("fifo", "edf") and self.max_batch != 1:
            raise ConfigError(f"scheduler {self.scheduler} does not batch: max_batch must be 1")
        if self.admission and self.scheduler != "edf_batch":
            raise ConfigError("admission control is defined on top of edf_batch (strategy e)")
        if not 0 < self.max_utilization <= 1:
            raise ConfigError("max_utilization must be in (0, 1]")

    @property
    def batches(self) -> bool:
        return self.max_batch > 1


@dataclass
class NetworkChoice:
    """A network scenario (``kortyn/experiments/<name>.yaml``) and the Brain's return-trip margin for it."""

    scenario: str = "buero_wlan"
    margin_ms: float = 15.0

    def __post_init__(self) -> None:
        if self.margin_ms < 0:
            raise ConfigError("network margin_ms must be >= 0")


@dataclass
class LevelRun:
    """One fixed-latency condition of the dose-response experiment."""

    latency_ms: float
    run: str  # run label in the experiment's results directory

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ConfigError(f"dose-response level must be >= 0 ms, got {self.latency_ms}")


@dataclass
class PredictionConfig:
    """Where the dose-response runs are and how the prediction is computed."""

    experiment: str  # experiment YAML of steps 1-2 (results directory, LIBERO tasks and seeds)
    profile: str = "smolvla_libero"
    levels: list[LevelRun] = field(default_factory=list)
    bootstrap: int = 10_000
    seed: int = 4242
    output: str = "results/phase3/fleet/capacity.json"

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ConfigError("prediction needs at least two dose-response levels")
        latencies = [level.latency_ms for level in self.levels]
        if len(set(latencies)) != len(latencies):
            raise ConfigError("dose-response levels must have distinct latencies")
        if self.bootstrap < 100:
            raise ConfigError("prediction.bootstrap must be >= 100 draws")


@dataclass
class ValidationConfig:
    """Stage 3: real rollouts of whole fleets (``kortyn.fleet_rollout``) on one GPU, batching and network."""

    experiment: str  # LIBERO experiment of the dose-response runs (env, policy, seeds)
    template_run: str  # run label whose agent, strategy and image settings the robots use
    gpu: str
    batching: str
    network: str
    profile: str = "smolvla_libero"
    results_dir: str = "results/phase3/validation"
    episodes: int = 100  # counted episodes per case (each robot plays ceil(episodes / n))
    workers: int = 2  # inference processes (forked, shared weights)
    confidence: float = 0.9875  # Bonferroni over four cases: family-wise 95 %
    bootstrap: int = 10_000
    seed: int = 5150

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.workers < 1 or self.bootstrap < 100:
            raise ConfigError("validation needs episodes >= 1, workers >= 1 and bootstrap >= 100")
        if not 0.5 < self.confidence < 1.0:
            raise ConfigError(f"validation confidence must be in (0.5, 1), got {self.confidence}")


@dataclass
class FleetSimConfig:
    """A grid of fleet simulations (stage 1 of phase 3)."""

    description: str = ""
    agent: FleetAgentConfig = field(default_factory=FleetAgentConfig)
    episode_lengths: EpisodeLengthConfig = field(default_factory=lambda: EpisodeLengthConfig(values=[520]))
    gpus: dict[str, GpuConfig] = field(default_factory=dict)
    batching: dict[str, BatchingConfig] = field(default_factory=dict)
    strategies: dict[str, StrategyConfig] = field(default_factory=dict)
    networks: dict[str, NetworkChoice] = field(default_factory=dict)
    fleet_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    replications: int = 10
    start_spread_s: float = 10.0
    window_start_s: float = 30.0
    horizon_s: float = 630.0
    service_cv: float = 0.05
    batch_limit: int = 32
    seed: int = 3000
    results_path: str = "results/phase3/fleet/stage1.jsonl"
    workers: int = 0
    prediction: PredictionConfig | None = None  # stage 2 (kortyn.fleet_predict)
    validation: ValidationConfig | None = None  # stage 3 (kortyn.fleet_rollout)

    def __post_init__(self) -> None:
        for name, mapping in (
            ("gpus", self.gpus),
            ("batching", self.batching),
            ("strategies", self.strategies),
            ("networks", self.networks),
        ):
            if not mapping:
                raise ConfigError(f"fleet config needs at least one entry in {name!r}")
        if not self.fleet_sizes or min(self.fleet_sizes) < 1:
            raise ConfigError("fleet_sizes must be a non-empty list of positive sizes")
        if self.replications < 1 or self.start_spread_s < 0:
            raise ConfigError("replications must be >= 1 and start_spread_s >= 0")
        if not 0 <= self.window_start_s < self.horizon_s:
            raise ConfigError("need 0 <= window_start_s < horizon_s")
        if self.service_cv < 0 or self.batch_limit < 1 or self.workers < 0:
            raise ConfigError("service_cv must be >= 0, batch_limit >= 1, workers >= 0")
        if self.validation is not None:
            for name, mapping, value in (
                ("gpu", self.gpus, self.validation.gpu),
                ("batching", self.batching, self.validation.batching),
                ("network", self.networks, self.validation.network),
            ):
                if value not in mapping:
                    raise ConfigError(f"validation.{name} {value!r} is not defined in the fleet config")


def load_fleet_config(path: Path | str) -> FleetSimConfig:
    import yaml

    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read fleet config {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    return decode(FleetSimConfig, data, str(source))


# --------------------------------------------------------------------------------------------- models
def service_model(config: FleetSimConfig, gpu: str, batching: str) -> BatchServiceModel:
    """Service-time model of one GPU with one batching profile (or the GPU's measured table)."""
    gpu_config = config.gpus[gpu]
    if gpu_config.table is not None:
        return TableBatchProfile.from_file(gpu_config.table)
    profile = config.batching[batching]
    return AffineBatchProfile(
        s1_ms=gpu_config.s1_ms, beta=profile.beta, cv=config.service_cv, batch_limit=config.batch_limit
    )


def load_episode_lengths(config: EpisodeLengthConfig) -> tuple[int, ...]:
    """Episode lengths in ticks: the logged ``steps`` of the final attempt of every seed, or fixed values."""
    if config.values:
        return tuple(int(v) for v in config.values)
    lengths: list[int] = []
    for path in config.paths:
        source = Path(path)
        if not source.exists():
            raise ConfigError(f"episode-length source {source} does not exist")
        final: dict[int, dict[str, Any]] = {}
        for record in read_jsonl(source):
            seed = int(record["seed"])
            if seed not in final or int(record["attempt"]) >= int(final[seed]["attempt"]):
                final[seed] = record
        lengths.extend(int(record["steps"]) for record in final.values() if int(record["steps"]) > 0)
    if not lengths:
        raise ConfigError(f"no episodes in {config.paths}")
    return tuple(sorted(lengths))


# --------------------------------------------------------------------------------------------- robot
@dataclass
class RequestLog:
    """One request of a timing robot (times in simulation seconds)."""

    robot: int
    episode: int
    obs_step: int
    t_obs: float
    t_send: float
    due: float
    start: bool  # first request of an episode (empty queue, cold start)
    t_resp: float | None = None
    status: str = "pending"  # ok | timeout | error

    def latency_s(self) -> float | None:
        return None if self.t_resp is None else self.t_resp - self.t_obs

    def missed(self, budget_s: float) -> bool:
        """Plan later than the tick at which the queue ran dry (start requests: nominal budget)."""
        if self.t_resp is None or self.status != "ok":
            return True
        if self.start:
            return self.t_resp - self.t_obs > budget_s + TIME_EPS_S
        return self.t_resp > self.due + TIME_EPS_S


@dataclass
class RobotStats:
    """Per-robot counters of a fleet run (episode and tick level)."""

    robot: int
    start_time: float
    admitted: bool = True
    episodes: int = 0
    ticks: int = 0
    starved_ticks: int = 0
    plan_age_ticks: float = 0.0  # summed over executed actions
    executed: int = 0
    requests: list[RequestLog] = field(default_factory=list)


class _ZeroCompute:
    """The policy stand-in of stage 1: every plan is zeros (only its timing matters)."""

    def __init__(self, actions: int, action_dim: int = 7) -> None:
        self._actions = np.zeros((actions, action_dim), dtype=np.float32)

    def __call__(self, request: pb.InferRequest) -> ComputeResult:
        return ComputeResult(self._actions)


def _timing(agent: FleetAgentConfig) -> ModelTiming:
    return ModelTiming(
        service=DurationSampler(DistributionConfig(kind="constant", value_ms=0.0)),
        client_overhead=DurationSampler(
            DistributionConfig(kind="constant", value_ms=agent.client_overhead_ms)
        ),
        transport_overhead=DurationSampler(
            DistributionConfig(kind="constant", value_ms=agent.transport_overhead_ms)
        ),
        delivery_lateness=DurationSampler(DistributionConfig(kind="constant", value_ms=0.0)),
    )


class TimingRobot:
    """A robot without simulator and policy: the agent's timing rules on the simulation clock.

    Per episode (cold start): request at tick 0; a plan for observation step ``s`` covers steps
    ``s .. s + chunk - 1`` and is merged at the first tick that starts at or after its delivery (steps that
    passed are dropped); the next request goes out at the first tick ``k`` at or after the merge with at
    most ``threshold`` actions left after it (``coverage_end - k <= threshold``); without an action the
    robot starves. The episode ends after its drawn number of ticks; a request still in flight is awaited
    before the next episode starts (as the agent does).
    """

    def __init__(
        self,
        index: int,
        sim: Simulation,
        brain: BatchingBrain,
        agent: FleetAgentConfig,
        network_name: str,
        lengths: Sequence[int],
        seed: int,
        stats: RobotStats,
        horizon_s: float,
        compute: _ZeroCompute,
    ) -> None:
        self.index = index
        self._sim = sim
        self._brain = brain
        self._agent = agent
        self._network = load_scenario(network_name).network
        self._lengths = lengths
        self._seed = seed
        self.stats = stats
        self._horizon_s = horizon_s
        self._timing = _timing(agent)
        self._compute = compute

    def run(self) -> Generator[Command, Any, None]:
        yield WakeAt(self.stats.start_time, Priority.TICK)
        episode = 0
        while self._sim.now < self._horizon_s:
            rng = stream_rng("fleet-episode", self._seed, self.index, episode)
            length = self._lengths[rng.randrange(len(self._lengths))]
            yield from self.episode(episode, length)
            episode += 1

    def episode(self, episode: int, length: int) -> Generator[Command, Any, None]:
        """One episode of ``length`` control ticks, starting now (cold)."""
        agent = self._agent
        dt = agent.dt
        chunk = agent.actions_per_chunk
        threshold = agent.refill_threshold
        sim = self._sim
        episode_seed = stream_seed("fleet-link", self._seed, self.index, episode)
        link = VirtualLink(sim, self._network, episode_seed % (2**31))
        channel = ModelChannel(sim, self._brain, link, self._timing, episode_seed, 0, f"robot-{self.index}")
        t0 = sim.now
        channel.arm()
        stats = self.stats
        stats.episodes += 1
        coverage_end = -1  # last control step with an action
        plan_step = -1  # observation step of the plan that covers the queue
        next_request = 0
        executed_until = 0  # ticks [0, executed_until) are accounted for
        request_id = 0

        def account(until: int) -> None:
            # ticks [executed_until, until): an action exists iff the step is covered
            nonlocal executed_until
            end = min(until, length)
            for tick in range(executed_until, end):
                if tick <= coverage_end and plan_step >= 0:
                    stats.executed += 1
                    stats.plan_age_ticks += tick - plan_step
                else:
                    stats.starved_ticks += 1
            stats.ticks += max(0, end - executed_until)
            executed_until = max(executed_until, end)

        while next_request < length:
            tick_start = t0 + next_request * dt
            yield WakeAt(tick_start, Priority.TICK)
            available_after = max(0, coverage_end - next_request)
            due = t0 + (next_request + available_after + 1) * dt
            request_id += 1
            request = pb.InferRequest(
                client_id=f"robot-{self.index}",
                episode_id=episode,
                request_id=request_id,
                obs_timestep=next_request,
                max_actions=chunk,
            )
            log = RequestLog(
                robot=self.index,
                episode=episode,
                obs_step=next_request,
                t_obs=tick_start,
                t_send=tick_start + channel.client_overhead_s(next_request),
                due=due,
                start=coverage_end < 0,
            )
            stats.requests.append(log)
            arrived = Signal(sim)
            channel.send(request, log.t_obs, log.t_send, agent.request_timeout_s, arrived.trigger, due=due)
            delivery: Delivery = yield arrived
            log.t_resp = delivery.t_resp
            log.status = delivery.status
            # merged at the first tick starting at or after delivery, never in the tick that sent it
            merge = max(log.obs_step + 1, math.ceil((delivery.t_resp - t0) / dt - 1e-9))
            if merge >= length:
                break
            account(merge)
            if delivery.status == "ok":
                plan_step = log.obs_step
                coverage_end = log.obs_step + chunk - 1
            next_request = max(merge, coverage_end - threshold)
        account(length)
        end = t0 + (length - 1) * dt
        if sim.now < end:
            yield WakeAt(end, Priority.TICK)


# --------------------------------------------------------------------------------------------- one run
@dataclass(frozen=True)
class RunKey:
    network: str
    gpu: str
    batching: str
    strategy: str
    robots: int
    replication: int


@dataclass
class RobotSummary:
    robot: int
    admitted: bool
    start_time: float
    episodes: int
    requests: int
    misses: int
    start_requests: int
    start_misses: int
    starved_share: float
    plan_age_ms: float
    latency_ms_p50: float | None
    latency_ms_p99: float | None
    histogram: dict[int, int]  # latency bin (HISTOGRAM_BIN_MS) -> requests in the window


@dataclass
class RunResult:
    key: RunKey
    admitted: int
    requests: int
    misses: int
    timeouts: int
    batches: int
    mean_batch: float
    busy_share: float
    robots: list[RobotSummary]
    wall_s: float

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["miss_rate"] = self.misses / self.requests if self.requests else None
        return data


def start_times(config: FleetSimConfig, robots: int, replication: int) -> list[float]:
    """Join times of the robots: uniform over the start spread, keyed by replication (same for all runs)."""
    rng = stream_rng("fleet-start", config.seed, robots, replication)
    return sorted(rng.uniform(0.0, config.start_spread_s) for _ in range(robots))


def run_fleet(config: FleetSimConfig, key: RunKey, lengths: Sequence[int]) -> RunResult:
    """Simulate one fleet configuration for ``horizon_s`` and summarize requests in the window."""
    started = time.perf_counter()
    agent = config.agent
    strategy = config.strategies[key.strategy]
    network = config.networks[key.network]
    model = service_model(config, key.gpu, key.batching)
    sim = Simulation()
    margin = network.margin_ms / 1000.0
    scheduler: SchedulerName = strategy.scheduler
    brain = BatchingBrain(
        sim,
        _ZeroCompute(agent.actions_per_chunk),
        make_batch_scheduler(scheduler, max_batch=strategy.max_batch, max_wait_s=strategy.max_wait_ms / 1000),
        model,
        seed=stream_seed("fleet-brain", config.seed, key.robots, key.replication) % (2**31),
        downlink_margin_s=margin,
    )
    controller = (
        AdmissionController(
            service=model,
            budget_s=agent.budget_s,
            period_s=agent.period_s,
            max_batch=strategy.max_batch,
            uplink_margin_s=margin,
            downlink_margin_s=margin,
            max_utilization=strategy.max_utilization,
        )
        if strategy.admission
        else None
    )
    compute = _ZeroCompute(agent.actions_per_chunk)
    stats: list[RobotStats] = []
    admitted = 0
    run_seed = stream_seed("fleet-run", config.seed, key.network, key.robots, key.replication) % (2**31)
    for index, start in enumerate(start_times(config, key.robots, key.replication)):
        robot_stats = RobotStats(robot=index, start_time=start)
        stats.append(robot_stats)
        if controller is not None and not controller.admits(admitted + 1):
            robot_stats.admitted = False
            continue
        admitted += 1
        robot = TimingRobot(
            index,
            sim,
            brain,
            agent,
            network.scenario,
            lengths,
            run_seed,
            robot_stats,
            config.horizon_s,
            compute,
        )
        sim.process(robot.run())
    sim.run(until=config.horizon_s)
    summaries: list[RobotSummary] = []
    total_requests = 0
    total_misses = 0
    timeouts = 0
    budget = agent.budget_s
    for robot_stats in stats:
        window = [
            r for r in robot_stats.requests if config.window_start_s <= r.t_obs and r.t_resp is not None
        ]
        latencies = np.array([r.latency_s() for r in window], dtype=float) * 1000.0
        misses = sum(r.missed(budget) for r in window)
        starts = [r for r in window if r.start]
        histogram: dict[int, int] = {}
        for value in latencies:
            bin_index = int(value // HISTOGRAM_BIN_MS)
            histogram[bin_index] = histogram.get(bin_index, 0) + 1
        timeouts += sum(r.status == "timeout" for r in window)
        total_requests += len(window)
        total_misses += misses
        executed = robot_stats.executed
        summaries.append(
            RobotSummary(
                robot=robot_stats.robot,
                admitted=robot_stats.admitted,
                start_time=robot_stats.start_time,
                episodes=robot_stats.episodes,
                requests=len(window),
                misses=int(misses),
                start_requests=len(starts),
                start_misses=int(sum(r.missed(budget) for r in starts)),
                starved_share=robot_stats.starved_ticks / robot_stats.ticks if robot_stats.ticks else 0.0,
                plan_age_ms=1000.0 * agent.dt * robot_stats.plan_age_ticks / executed if executed else 0.0,
                latency_ms_p50=float(np.percentile(latencies, 50)) if latencies.size else None,
                latency_ms_p99=float(np.percentile(latencies, 99)) if latencies.size else None,
                histogram=histogram,
            )
        )
    counters = brain.counters
    sizes = counters.batch_sizes
    mean_batch = sum(size * count for size, count in sizes.items()) / max(1, sum(sizes.values()))
    return RunResult(
        key=key,
        admitted=admitted,
        requests=total_requests,
        misses=total_misses,
        timeouts=timeouts,
        batches=counters.batches,
        mean_batch=mean_batch,
        busy_share=counters.busy_ms / (1000.0 * config.horizon_s),
        robots=summaries,
        wall_s=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------------------------- grid
def grid(config: FleetSimConfig, sizes: Iterable[int] | None = None) -> list[RunKey]:
    """All runs of a config. Strategies without batching run once per GPU (batching profile irrelevant)."""
    keys: list[RunKey] = []
    first_batching = next(iter(config.batching))
    for network, gpu, strategy_name in itertools.product(config.networks, config.gpus, config.strategies):
        strategy = config.strategies[strategy_name]
        if config.gpus[gpu].table is not None:
            profiles = ["gemessen"]
        elif strategy.batches:
            profiles = list(config.batching)
        else:
            profiles = [first_batching]
        for batching, robots, replication in itertools.product(
            profiles, sizes or config.fleet_sizes, range(config.replications)
        ):
            keys.append(RunKey(network, gpu, batching, strategy_name, robots, replication))
    return keys


_WORKER_CONFIG: FleetSimConfig | None = None
_WORKER_LENGTHS: tuple[int, ...] = ()


def _init_worker(config: FleetSimConfig, lengths: tuple[int, ...]) -> None:
    global _WORKER_CONFIG, _WORKER_LENGTHS
    _WORKER_CONFIG = config
    _WORKER_LENGTHS = lengths


def _run_key(key: RunKey) -> dict[str, Any]:
    if _WORKER_CONFIG is None:
        raise RuntimeError("fleet worker not initialized")
    return run_fleet(_WORKER_CONFIG, key, _WORKER_LENGTHS).to_json()


def done_keys(path: Path) -> set[RunKey]:
    """Runs already stored in a results file (the grid is resumable)."""
    if not path.exists():
        return set()
    keys: set[RunKey] = set()
    for record in read_jsonl(path):
        keys.add(RunKey(**record["key"]))
    return keys


def run_grid(config: FleetSimConfig, keys: Sequence[RunKey], out_path: Path, workers: int) -> int:
    """Run ``keys`` (skipping stored ones) in parallel and append each result to ``out_path``."""
    stored = done_keys(out_path)
    pending = [key for key in keys if key not in stored]
    LOG.info("%d fleet runs pending (%d stored), %d workers", len(pending), len(stored), workers)
    if not pending:
        return 0
    lengths = load_episode_lengths(config.episode_lengths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (
        out_path.open("a", encoding="utf-8") as out,
        concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(config, lengths)
        ) as pool,
    ):
        futures = {pool.submit(_run_key, key): key for key in pending}
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            out.write(json.dumps(result, separators=(",", ":")) + "\n")
            out.flush()
            if count % 50 == 0 or count == len(pending):
                elapsed = time.perf_counter() - started
                LOG.info("%d/%d runs (%.0f s)", count, len(pending), elapsed)
    return len(pending)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-3 fleet simulation of the Brain's queue (stage 1).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sizes", default=None, help="comma-separated fleet sizes (default: from config)")
    parser.add_argument("--out", default=None, help="results file (default: from config)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        config = load_fleet_config(args.config)
        sizes = None if args.sizes is None else [int(s) for s in args.sizes.split(",") if s.strip()]
        keys = grid(config, sizes)
        workers = args.workers if args.workers is not None else (config.workers or os.cpu_count() or 1)
        run_grid(config, keys, Path(args.out or config.results_path), max(1, workers))
    except ConfigError as exc:
        LOG.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
