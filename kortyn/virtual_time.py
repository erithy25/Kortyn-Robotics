"""Discrete-event simulation (virtual time) of the Kortyn Agent's link and the Kortyn Brain.

Phase 1 measured in real time, so every number carried the noise of the machine: inference time drifted
with CPU frequency and steal, the control clock jittered, and identical runs differed by up to 16 points.
The virtual-time mode replaces the wall clock by a **simulation clock**:

* a request sent at simulation time ``t`` delivers its plan at ``t + L``; until then the agent keeps
  executing its queue tick by tick, exactly like the real-time agent (``kortyn/virtual_agent.py``);
* ``L`` is built from explicit, seeded models (:class:`ModelChannel`): client overhead, the scenario's link
  (the netem proxy's own :class:`~kortyn.netem_proxy.NetworkState` with the proxy's seeds, now driven by the
  simulation clock), transport overhead, a Brain with ``servers`` inference servers behind a pluggable
  queue discipline (:class:`FifoScheduler`) and a service time per hardware profile; or ``L`` is played back
  request by request from the logs of a real-time run (:class:`ReplayChannel`);
* the policy still runs for every request (real inference, deterministic for fixed seeds and thread
  count); only *when* the plan arrives is simulated.

Every random quantity is drawn from a stream keyed by (episode seed, timing seed, client, component,
observation step), so two runs with the same configuration produce identical results, and conditions that
share a seed share their random numbers wherever they draw the same quantity (common random numbers).

Event order at equal times: deliveries to a robot, then Brain events, then resumed processes, then control
ticks. A plan that arrives exactly at a tick's start is merged by that tick, as in real time (the agent
merges every completion whose timestamp is not later than the tick start).

This module does not import torch; the policy enters through the :class:`Compute` callable.
"""

from __future__ import annotations

import glob
import hashlib
import heapq
import itertools
import math
import random
from collections import deque
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from kortyn.codec import encode_array
from kortyn.config import (
    ConfigError,
    DistributionConfig,
    EmpiricalSourceConfig,
    NetworkConfig,
)
from kortyn.metrics import EpisodeRecord, RequestRecord, final_episodes, read_jsonl
from kortyn.netem_proxy import DOWN, UP, NetworkState
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.typing_utils import FloatArray

# Scheduling an event this little in the past (float rounding) is treated as "now".
TIME_EPS_S = 1e-9


class SimulationError(RuntimeError):
    """Inconsistent use of the simulation (event in the past, deadlock, process misuse)."""


# --------------------------------------------------------------------------------------------- core
class Priority(IntEnum):
    """Order of events that fall on the same instant (lower first)."""

    DELIVERY = 0  # a message reaches a robot: visible to a tick at the same instant
    BRAIN = 1  # a request reaches the Brain, a service completes
    RESUME = 2  # a process waiting for a signal continues
    TICK = 3  # a control tick: sees everything up to and including its start time


@dataclass(frozen=True)
class WakeAt:
    """Process command: continue at simulation time ``time``."""

    time: float
    priority: Priority = Priority.TICK


class Signal:
    """One-shot event inside the simulation; a process can wait for it with ``value = yield signal``."""

    def __init__(self, sim: Simulation) -> None:
        self._sim = sim
        self.triggered = False
        self.value: Any = None
        self._waiters: list[Callable[[Any], None]] = []

    def trigger(self, value: Any = None) -> None:
        if self.triggered:
            raise SimulationError("signal triggered twice")
        self.triggered = True
        self.value = value
        waiters, self._waiters = self._waiters, []
        for waiter in waiters:
            self._sim.at(self._sim.now, Priority.RESUME, partial(waiter, value))

    def wait(self, callback: Callable[[Any], None]) -> None:
        if self.triggered:
            self._sim.at(self._sim.now, Priority.RESUME, partial(callback, self.value))
        else:
            self._waiters.append(callback)


Command = WakeAt | Signal


class Process[T]:
    """A generator driven by the simulation: yields :class:`WakeAt` or :class:`Signal`, returns ``T``."""

    def __init__(self, sim: Simulation, generator: Generator[Command, Any, T]) -> None:
        self._sim = sim
        self._generator = generator
        self.done = False
        self._result: T | None = None
        sim.at(sim.now, Priority.RESUME, partial(self._resume, None))

    @property
    def result(self) -> T:
        if not self.done:
            raise SimulationError("process has not finished")
        return cast(T, self._result)

    def _resume(self, value: Any) -> None:
        try:
            command = self._generator.send(value)
        except StopIteration as stop:
            self.done = True
            self._result = cast(T, stop.value)
            return
        if isinstance(command, WakeAt):
            self._sim.at(command.time, command.priority, partial(self._resume, None))
        elif isinstance(command, Signal):
            command.wait(self._resume)
        else:
            raise SimulationError(f"process yielded {command!r}; expected WakeAt or Signal")


class Simulation:
    """Event loop with a virtual clock: events run in (time, priority, insertion) order."""

    def __init__(self, start: float = 0.0) -> None:
        if not math.isfinite(start):
            raise SimulationError(f"start time must be finite, got {start}")
        self._now = start
        self._queue: list[tuple[float, int, int, Callable[[], None]]] = []
        self._sequence = itertools.count()
        self.events_processed = 0

    @property
    def now(self) -> float:
        return self._now

    def at(self, time: float, priority: Priority, callback: Callable[[], None]) -> None:
        """Schedule ``callback`` at simulation time ``time``."""
        if not math.isfinite(time):
            raise SimulationError(f"event time must be finite, got {time}")
        if time < self._now:
            if time < self._now - TIME_EPS_S:
                raise SimulationError(f"event at t={time:.9f} s lies before now={self._now:.9f} s")
            time = self._now
        heapq.heappush(self._queue, (time, int(priority), next(self._sequence), callback))

    def process[T](self, generator: Generator[Command, Any, T]) -> Process[T]:
        return Process(self, generator)

    def step(self) -> bool:
        """Run the next event; ``False`` if none is left."""
        if not self._queue:
            return False
        time, _, _, callback = heapq.heappop(self._queue)
        self._now = time
        callback()
        self.events_processed += 1
        return True

    def run(self, until: float | None = None) -> None:
        """Run all events up to and including ``until`` (all events if ``None``)."""
        while self._queue and (until is None or self._queue[0][0] <= until):
            self.step()
        if until is not None and until > self._now:
            self._now = until

    def run_process[T](self, generator: Generator[Command, Any, T]) -> T:
        """Start ``generator`` as a process and run events until it has finished."""
        process = self.process(generator)
        while not process.done:
            if not self.step():
                raise SimulationError("deadlock: the process waits, but no event is scheduled")
        return process.result


# --------------------------------------------------------------------------------------------- randomness
def stream_seed(*key: int | str) -> int:
    """64-bit seed of an independent random stream, stable across processes and Python versions."""
    digest = hashlib.sha256(repr(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def stream_rng(*key: int | str) -> random.Random:
    return random.Random(stream_seed(*key))


# --------------------------------------------------------------------------------------------- durations
_EMPIRICAL_CACHE: dict[tuple[tuple[str, ...], str, str], tuple[float, ...]] = {}


def _run_directories(patterns: Sequence[str], root: Path) -> list[Path]:
    directories: set[Path] = set()
    for pattern in patterns:
        path = Path(pattern)
        full = path if path.is_absolute() else root / path
        matches = [Path(m) for m in glob.glob(str(full))] if any(c in pattern for c in "*?[") else [full]
        directories.update(m for m in matches if (m / "episodes.jsonl").is_file())
    return sorted(directories)


def load_empirical_ms(source: EmpiricalSourceConfig, root: Path = Path()) -> tuple[float, ...]:
    """All logged values of ``source.quantity`` (ms), sorted, from the final attempts of real-time runs."""
    key = (tuple(source.paths), source.quantity, str(root.resolve()))
    cached = _EMPIRICAL_CACHE.get(key)
    if cached is not None:
        return cached
    values: list[float] = []
    for run_dir in _run_directories(source.paths, root):
        episodes, _ = final_episodes(
            EpisodeRecord.from_json(r) for r in read_jsonl(run_dir / "episodes.jsonl")
        )
        episodes = [e for e in episodes if e.mode == "realtime"]
        if not episodes:
            continue
        if source.quantity == "tick_lateness_ms":
            values.extend(value for e in episodes for value in e.ticks.lateness_ms)
            continue
        chosen = {(e.seed, e.attempt) for e in episodes}
        for data in read_jsonl(run_dir / "requests.jsonl"):
            record = RequestRecord.from_json(data)
            if (record.seed, record.attempt) not in chosen or record.phase != "realtime":
                continue
            if record.status != "ok":
                continue
            value = _request_quantity(record, source.quantity)
            if value is not None and math.isfinite(value):
                values.append(value)
    if not values:
        raise ConfigError(f"no logged {source.quantity} values found in {source.paths}")
    result = tuple(sorted(values))
    _EMPIRICAL_CACHE[key] = result
    return result


def _request_quantity(record: RequestRecord, quantity: str) -> float | None:
    if quantity == "server_total_ms":
        return record.server_total_ms
    if quantity == "server_inference_ms":
        return record.server_inference_ms
    if quantity == "send_offset_ms":
        return 1000.0 * (record.t_send_s - record.t_obs_s)
    if quantity == "network_ms":
        return record.network_ms
    if quantity == "e2e_ms":
        return record.e2e_ms
    raise ConfigError(f"unknown request quantity {quantity!r}")


class DurationSampler:
    """Draws durations from a :class:`~kortyn.config.DistributionConfig` (configured in ms)."""

    def __init__(self, config: DistributionConfig, root: Path = Path()) -> None:
        self.config = config
        self._samples: tuple[float, ...] = ()
        self._mu = 0.0
        self._sigma = 0.0
        if config.kind == "empirical":
            assert config.empirical is not None  # checked by DistributionConfig
            self._samples = load_empirical_ms(config.empirical, root)
        elif config.kind == "lognormal":
            variance = math.log1p((config.std_ms / config.mean_ms) ** 2)
            self._sigma = math.sqrt(variance)
            self._mu = math.log(config.mean_ms) - variance / 2.0

    @property
    def is_zero(self) -> bool:
        return self.config.kind == "constant" and self.config.value_ms == 0.0

    def sample_ms(self, rng: random.Random) -> float:
        config = self.config
        kind = config.kind
        if kind == "constant":
            value = config.value_ms
        elif kind == "uniform":
            value = rng.uniform(config.low_ms, config.high_ms)
        elif kind == "normal":
            value = rng.gauss(config.mean_ms, config.std_ms)
        elif kind == "lognormal":
            value = rng.lognormvariate(self._mu, self._sigma)
        else:
            value = self._samples[rng.randrange(len(self._samples))]
        value *= config.scale
        value = max(config.min_ms, value)
        if config.max_ms is not None:
            value = min(config.max_ms, value)
        return value

    def sample_s(self, rng: random.Random) -> float:
        return self.sample_ms(rng) / 1000.0

    def mean_ms(self) -> float:
        """Mean of the configured distribution before clipping (diagnostics)."""
        config = self.config
        if config.kind == "constant":
            base = config.value_ms
        elif config.kind == "uniform":
            base = (config.low_ms + config.high_ms) / 2.0
        elif config.kind in ("normal", "lognormal"):
            base = config.mean_ms
        else:
            base = sum(self._samples) / len(self._samples)
        return base * config.scale


# --------------------------------------------------------------------------------------------- network
class VirtualLink:
    """One robot's link in both directions on the simulation clock.

    It is the netem proxy's :class:`NetworkState` (stall schedule, bandwidth, delay and jitter with the same
    per-direction seeds, in-order delivery), fed one message at a time instead of TCP reads.
    """

    def __init__(self, sim: Simulation, network: NetworkConfig, seed: int) -> None:
        self._sim = sim
        self.state = NetworkState(network, seed, lambda: sim.now)

    def arm(self) -> None:
        """Anchor the stall schedule at the episode start (the proxy's ``arm``)."""
        self.state.arm()

    def transmit(self, direction: str, t: float, nbytes: int) -> tuple[float, bool]:
        """Delivery time of a message of ``nbytes`` entering the link at ``t``; ``True`` if it was held."""
        delivery = self.state.links[direction].schedule(t, nbytes)
        stats = self.state.stats.up if direction == UP else self.state.stats.down
        stats.bytes += nbytes
        stats.reads += 1
        return delivery, delivery > t

    def note_lateness(self, direction: str, lateness_s: float) -> None:
        stats = self.state.stats.up if direction == UP else self.state.stats.down
        stats.max_lateness_s = max(stats.max_lateness_s, lateness_s)

    def stats(self) -> dict[str, Any]:
        """The proxy's ``stats`` reply for this episode (bytes, stall windows, network settings)."""
        return {"ok": True, **self.state.stats_dict()}


# --------------------------------------------------------------------------------------------- brain
@dataclass(frozen=True)
class ComputeResult:
    """Outcome of running the policy for one request (content only; its timing is simulated)."""

    actions: FloatArray | None
    error: str | None = None
    compute_ms: float = 0.0  # wall-clock time of the real computation (diagnostics, not simulated time)
    cache_hit: bool = False


class Compute(Protocol):
    """Runs the policy for one request (validation, guide, inference), like the Brain's ``Infer``."""

    def __call__(self, request: pb.InferRequest) -> ComputeResult: ...


@dataclass
class BrainJob:
    """One request inside the virtual Brain."""

    client_id: str
    request: pb.InferRequest
    t_sent: float
    t_arrival: float
    service_s: float
    deadline: float | None  # absolute time after which the client has given up (gRPC deadline)
    on_done: Callable[[BrainJob], None]
    t_start: float | None = None
    t_done: float | None = None
    result: ComputeResult | None = None


class Scheduler(Protocol):
    """Queue discipline of the Brain: which waiting request is served next."""

    def push(self, job: BrainJob, now: float) -> None: ...

    def pop(self, now: float) -> BrainJob | None: ...

    def __len__(self) -> int: ...


class FifoScheduler:
    """First come, first served (the phase-1 Brain: one inference thread, requests in arrival order)."""

    def __init__(self) -> None:
        self._queue: deque[BrainJob] = deque()

    def push(self, job: BrainJob, now: float) -> None:
        self._queue.append(job)

    def pop(self, now: float) -> BrainJob | None:
        return self._queue.popleft() if self._queue else None

    def __len__(self) -> int:
        return len(self._queue)


def make_scheduler(name: str) -> Scheduler:
    if name == "fifo":
        return FifoScheduler()
    raise ConfigError(f"unknown Brain scheduler {name!r}")


@dataclass
class BrainCounters:
    """Cumulative load counters, as ``GetStats`` reports them for the real Brain."""

    requests: int = 0
    inferences: int = 0
    abandoned: int = 0
    busy_ms: float = 0.0
    queue_wait_ms: float = 0.0

    def snapshot(self, now_s: float) -> dict[str, float]:
        return {
            "requests": float(self.requests),
            "inferences": float(self.inferences),
            "abandoned": float(self.abandoned),
            "busy_ms": self.busy_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "uptime_ms": 1000.0 * now_s,
        }


def load_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Counter differences over an interval, in the format of ``EpisodeRecord.brain_load``."""
    return {
        "requests": after["requests"] - before["requests"],
        "inferences": after["inferences"] - before["inferences"],
        "abandoned": after["abandoned"] - before["abandoned"],
        "busy_ms": after["busy_ms"] - before["busy_ms"],
        "queue_wait_ms": after["queue_wait_ms"] - before["queue_wait_ms"],
        "window_ms": after["uptime_ms"] - before["uptime_ms"],
    }


class VirtualBrain:
    """The Brain in virtual time: ``servers`` identical inference servers fed by a scheduler.

    A request occupies a server for its service time; the plan's content comes from ``compute`` (the real
    policy) when the service starts. Requests whose client deadline has passed before their service starts
    are skipped, as the real Brain does. Several robots can share one instance (phase 3).
    """

    def __init__(self, sim: Simulation, compute: Compute, scheduler: Scheduler, servers: int = 1) -> None:
        if servers < 1:
            raise SimulationError(f"servers must be >= 1, got {servers}")
        self._sim = sim
        self._compute = compute
        self._scheduler = scheduler
        self._servers = servers
        self._busy = 0
        self.counters = BrainCounters()
        self.compute_ms = 0.0
        self.cache_hits = 0

    @property
    def queue_length(self) -> int:
        return len(self._scheduler)

    def submit(self, job: BrainJob) -> None:
        """A request reaches the Brain (call at its arrival time)."""
        self.counters.requests += 1
        self._scheduler.push(job, self._sim.now)
        self._dispatch()

    def _dispatch(self) -> None:
        while self._busy < self._servers:
            job = self._scheduler.pop(self._sim.now)
            if job is None:
                return
            now = self._sim.now
            if job.deadline is not None and now >= job.deadline:
                self.counters.abandoned += 1
                continue
            job.t_start = now
            job.result = self._compute(job.request)
            self.compute_ms += job.result.compute_ms
            self.cache_hits += int(job.result.cache_hit)
            self.counters.queue_wait_ms += 1000.0 * (now - job.t_arrival)
            service = job.service_s if job.result.error is None else 0.0
            self._busy += 1
            self._sim.at(now + service, Priority.BRAIN, partial(self._complete, job, service))

    def _complete(self, job: BrainJob, service_s: float) -> None:
        job.t_done = self._sim.now
        self._busy -= 1
        self.counters.inferences += 1
        self.counters.busy_ms += 1000.0 * service_s
        job.on_done(job)
        self._dispatch()


# --------------------------------------------------------------------------------------------- channels
@dataclass(frozen=True)
class Delivery:
    """A request's outcome reaching the robot (same fields as the real-time agent's completions)."""

    request_id: int
    t_resp: float
    response: pb.InferResponse | None
    status: str  # "ok" | "timeout" | "error"
    error: str | None


Deliver = Callable[[Delivery], None]


class Channel(Protocol):
    """How a robot's requests reach the Brain and how their plans come back, in virtual time."""

    def client_overhead_s(self, obs_step: int) -> float:
        """Time between releasing observation ``obs_step`` (tick start) and sending its request."""
        ...

    def send(
        self, request: pb.InferRequest, t_obs: float, t_send: float, timeout_s: float, deliver: Deliver
    ) -> None: ...

    def episode_start(self, warmup_delivery: float) -> float:
        """Episode clock start t0, given the time the warm-up plan arrived."""
        ...

    def arm(self) -> None:
        """The episode clock starts now (anchors the stall schedule)."""
        ...

    def link_stats(self) -> dict[str, Any] | None: ...


def _response(
    request: pb.InferRequest, actions: FloatArray, queue_ms: float, service_ms: float
) -> pb.InferResponse:
    return pb.InferResponse(
        request_id=request.request_id,
        start_timestep=request.obs_timestep,
        actions=encode_array(actions),
        server_queue_ms=queue_ms,
        server_inference_ms=service_ms,
        server_total_ms=queue_ms + service_ms,
    )


@dataclass
class _Outstanding:
    """Client side of one request: delivered at most once (response, error or timeout)."""

    deliver: Deliver
    done: bool = False

    def complete(self, delivery: Delivery) -> None:
        if not self.done:
            self.done = True
            self.deliver(delivery)


@dataclass(frozen=True)
class ModelTiming:
    """Samplers of the latency components of the model channel."""

    service: DurationSampler
    client_overhead: DurationSampler
    transport_overhead: DurationSampler
    delivery_lateness: DurationSampler


class ModelChannel:
    """Latency from models: client overhead -> uplink -> Brain queue + service -> downlink.

    Random components are keyed by (episode seed, timing seed, client, component, observation step); the
    link's delay samples follow the proxy's per-direction streams (one sample per message).
    """

    def __init__(
        self,
        sim: Simulation,
        brain: VirtualBrain,
        link: VirtualLink,
        timing: ModelTiming,
        episode_seed: int,
        timing_seed: int,
        client_id: str,
    ) -> None:
        self._sim = sim
        self._brain = brain
        self._link = link
        self._timing = timing
        self._key: tuple[int | str, ...] = (episode_seed, timing_seed, client_id)
        self._client_id = client_id

    def _rng(self, component: str, obs_step: int) -> random.Random:
        return stream_rng(*self._key, component, obs_step)

    def client_overhead_s(self, obs_step: int) -> float:
        return self._timing.client_overhead.sample_s(self._rng("client", obs_step))

    def episode_start(self, warmup_delivery: float) -> float:
        return warmup_delivery

    def arm(self) -> None:
        self._link.arm()

    def link_stats(self) -> dict[str, Any] | None:
        return self._link.stats()

    def _held(self, direction: str, t: float, nbytes: int, obs_step: int) -> float:
        delivery, held = self._link.transmit(direction, t, nbytes)
        if held and not self._timing.delivery_lateness.is_zero:
            late = self._timing.delivery_lateness.sample_s(self._rng(f"late-{direction}", obs_step))
            self._link.note_lateness(direction, late)
            delivery += late
        return delivery

    def send(
        self, request: pb.InferRequest, t_obs: float, t_send: float, timeout_s: float, deliver: Deliver
    ) -> None:
        obs_step = int(request.obs_timestep)
        outstanding = _Outstanding(deliver)
        transport = self._timing.transport_overhead.sample_s(self._rng("transport", obs_step))
        arrival = self._held(UP, t_send + transport, request.ByteSize(), obs_step)
        deadline = t_send + timeout_s
        job = BrainJob(
            client_id=self._client_id,
            request=request,
            t_sent=t_send,
            t_arrival=arrival,
            service_s=self._timing.service.sample_s(self._rng("service", obs_step)),
            deadline=deadline,
            on_done=partial(self._on_done, outstanding),
        )
        self._sim.at(arrival, Priority.BRAIN, partial(self._brain.submit, job))
        self._sim.at(
            deadline,
            Priority.DELIVERY,
            partial(
                outstanding.complete,
                Delivery(int(request.request_id), deadline, None, "timeout", "deadline exceeded"),
            ),
        )

    def _on_done(self, outstanding: _Outstanding, job: BrainJob) -> None:
        assert job.result is not None
        assert job.t_start is not None
        assert job.t_done is not None
        request = job.request
        request_id = int(request.request_id)
        if job.result.error is not None or job.result.actions is None:
            message = job.result.error or "no actions"
            downlink = self._held(DOWN, job.t_done, 64, int(request.obs_timestep))
            self._sim.at(
                downlink,
                Priority.DELIVERY,
                partial(outstanding.complete, Delivery(request_id, downlink, None, "error", message)),
            )
            return
        response = _response(
            request,
            job.result.actions,
            queue_ms=1000.0 * (job.t_start - job.t_arrival),
            service_ms=1000.0 * (job.t_done - job.t_start),
        )
        delivery = self._held(DOWN, job.t_done, response.ByteSize(), int(request.obs_timestep))
        self._sim.at(
            delivery,
            Priority.DELIVERY,
            partial(outstanding.complete, Delivery(request_id, delivery, response, "ok", None)),
        )


# --------------------------------------------------------------------------------------------- replay
@dataclass
class ReplayEpisode:
    """Timing of one logged real-time episode (times relative to its episode clock start t0)."""

    seed: int
    requests: dict[int, RequestRecord]  # by observation step: real-time requests that completed
    warmup: RequestRecord | None
    tick_lateness_ms: list[float]
    proxy: dict[str, Any] | None

    def nearest(self, obs_step: int) -> RequestRecord:
        """Logged request of the closest observation step (ties: the earlier step)."""
        if not self.requests:
            if self.warmup is None:
                raise SimulationError(f"seed {self.seed}: the logged episode has no completed request")
            return self.warmup
        step = min(self.requests, key=lambda s: (abs(s - obs_step), s))
        return self.requests[step]


class ReplayLog:
    """Logged real-time episodes of one run (``<results_dir>/<profile>/<run>/``), final attempt per seed."""

    def __init__(self, run_dir: Path) -> None:
        if not (run_dir / "episodes.jsonl").is_file():
            raise ConfigError(f"replay: no logged run at {run_dir}")
        self.run_dir = run_dir
        episodes, _ = final_episodes(
            EpisodeRecord.from_json(r) for r in read_jsonl(run_dir / "episodes.jsonl")
        )
        realtime = [e for e in episodes if e.mode == "realtime"]
        if not realtime:
            raise ConfigError(f"replay: {run_dir} holds no real-time episodes")
        chosen = {(e.seed, e.attempt): e for e in realtime}
        requests: dict[int, dict[int, RequestRecord]] = {e.seed: {} for e in realtime}
        warmups: dict[int, RequestRecord] = {}
        for data in read_jsonl(run_dir / "requests.jsonl"):
            record = RequestRecord.from_json(data)
            if (record.seed, record.attempt) not in chosen or record.status != "ok":
                continue
            if record.t_resp_s is None:
                continue
            if record.phase == "warmup":
                warmups[record.seed] = record
            elif record.phase == "realtime":
                requests[record.seed][record.obs_timestep] = record
        self.episodes = {
            e.seed: ReplayEpisode(
                seed=e.seed,
                requests=requests[e.seed],
                warmup=warmups.get(e.seed),
                tick_lateness_ms=list(e.ticks.lateness_ms),
                proxy=e.proxy,
            )
            for e in realtime
        }

    def episode(self, seed: int) -> ReplayEpisode:
        try:
            return self.episodes[seed]
        except KeyError as exc:
            raise ConfigError(f"replay: seed {seed} is not logged in {self.run_dir}") from exc


@dataclass
class ReplayCounts:
    exact: int = 0
    fallback: int = 0


class ReplayChannel:
    """Latency played back from a logged real-time episode.

    The simulation runs on that episode's own clock (t0 = 0). A request for observation step ``k`` that the
    logged episode also sent (same step, so the same tick) is sent and answered at exactly the logged
    times; otherwise (the replayed episode diverged, e.g. it lasted longer) the logged request of the
    nearest step lends its end-to-end latency and server timings, and the request is counted as a fallback.
    The plan's content is always computed by the policy.
    """

    def __init__(self, sim: Simulation, episode: ReplayEpisode, compute: Compute) -> None:
        self._sim = sim
        self.episode = episode
        self._compute = compute
        self.counts = ReplayCounts()
        self.compute_ms = 0.0
        self.cache_hits = 0
        self._busy_ms = 0.0
        self._queue_ms = 0.0
        self._served = 0

    @property
    def start_time(self) -> float:
        """Simulation start: the logged warm-up observation time (so the episode clock starts at 0)."""
        warmup = self.episode.warmup
        return warmup.t_obs_s if warmup is not None else 0.0

    def _logged(self, request: pb.InferRequest) -> tuple[RequestRecord, bool]:
        obs_step = int(request.obs_timestep)
        warmup = self.episode.warmup
        if warmup is not None and obs_step == warmup.obs_timestep and self._served == 0:
            return warmup, True
        exact = self.episode.requests.get(obs_step)
        if exact is not None:
            return exact, True
        return self.episode.nearest(obs_step), False

    def client_overhead_s(self, obs_step: int) -> float:
        exact = self.episode.requests.get(obs_step)
        if exact is None:
            exact = self.episode.nearest(obs_step)
        return max(0.0, exact.t_send_s - exact.t_obs_s)

    def episode_start(self, warmup_delivery: float) -> float:
        return 0.0 if self.episode.warmup is not None else warmup_delivery

    def arm(self) -> None:
        """Stall windows are part of the logged latencies."""

    def link_stats(self) -> dict[str, Any] | None:
        return self.episode.proxy

    def brain_load(self) -> dict[str, float]:
        return {
            "requests": float(self._served),
            "inferences": float(self._served),
            "abandoned": 0.0,
            "busy_ms": self._busy_ms,
            "queue_wait_ms": self._queue_ms,
        }

    def send(
        self, request: pb.InferRequest, t_obs: float, t_send: float, timeout_s: float, deliver: Deliver
    ) -> None:
        logged, exact = self._logged(request)
        self._served += 1
        if exact:
            self.counts.exact += 1
            t_resp = logged.t_resp_s
        else:
            self.counts.fallback += 1
            t_resp = t_obs + (cast(float, logged.t_resp_s) - logged.t_obs_s)
        assert t_resp is not None
        result = self._compute(request)
        self.compute_ms += result.compute_ms
        self.cache_hits += int(result.cache_hit)
        request_id = int(request.request_id)
        queue_ms = float(logged.server_queue_ms or 0.0)
        service_ms = float(logged.server_inference_ms or 0.0)
        total_ms = float(logged.server_total_ms or queue_ms + service_ms)
        self._busy_ms += service_ms
        self._queue_ms += queue_ms
        if t_resp > t_send + timeout_s:
            delivery = Delivery(request_id, t_send + timeout_s, None, "timeout", "deadline exceeded")
        elif result.error is not None or result.actions is None:
            delivery = Delivery(request_id, t_resp, None, "error", result.error or "no actions")
        else:
            response = pb.InferResponse(
                request_id=request.request_id,
                start_timestep=request.obs_timestep,
                actions=encode_array(result.actions),
                server_queue_ms=queue_ms,
                server_inference_ms=service_ms,
                server_total_ms=total_ms,
            )
            delivery = Delivery(request_id, t_resp, response, "ok", None)
        self._sim.at(delivery.t_resp, Priority.DELIVERY, partial(deliver, delivery))


# --------------------------------------------------------------------------------------------- ticks
class TickTiming(Protocol):
    """Lateness of control tick ``step`` relative to its deadline ``t0 + step * dt``."""

    def lateness_s(self, step: int) -> float: ...


class SampledTicks:
    """Tick lateness drawn per tick (constant 0 = ideal clock), keyed by seed and step."""

    def __init__(self, sampler: DurationSampler, episode_seed: int, timing_seed: int, client_id: str) -> None:
        self._sampler = sampler
        self._key: tuple[int | str, ...] = (episode_seed, timing_seed, client_id, "tick")

    def lateness_s(self, step: int) -> float:
        if self._sampler.is_zero:
            return 0.0
        return self._sampler.sample_s(stream_rng(*self._key, step))


class ReplayTicks:
    """Tick lateness exactly as logged; ticks the logged episode did not reach start on time."""

    def __init__(self, lateness_ms: Sequence[float]) -> None:
        self._lateness_ms = list(lateness_ms)

    def lateness_s(self, step: int) -> float:
        if 0 <= step < len(self._lateness_ms):
            return self._lateness_ms[step] / 1000.0
        return 0.0


@dataclass(frozen=True)
class TimingSamplers:
    """The samplers a timing configuration needs for one Brain profile (model channel and ticks)."""

    model: ModelTiming | None
    tick: DurationSampler
