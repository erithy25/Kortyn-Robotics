"""The Brain of a robot fleet in virtual time (phase 3): batched inference, schedulers, admission control.

One GPU serves the requests of many robots. Unlike phase 1-3a's :class:`~kortyn.virtual_time.VirtualBrain`
(one request per inference), the fleet Brain runs **batches**: the service time of a batch of ``b``
requests follows a :class:`BatchServiceModel` (a named profile or a measured table), and every request of
the batch completes when the batch does. Which waiting requests form the next batch is the scheduler's
decision:

* :class:`FifoScheduler` (a): first come, first served, one request per inference;
* :class:`FifoBatchScheduler` (b): the oldest requests, up to ``max_batch``; while fewer are waiting, the
  Brain waits at most ``max_wait_s`` after the oldest arrival for more;
* :class:`EdfScheduler` (c): earliest deadline first, one request per inference. A request's deadline is
  the moment its robot's action queue runs dry (reported by the robot, ``BrainJob.due``) minus the return
  trip; a robot that is already starving comes first;
* :class:`EdfBatchScheduler` (d): earliest deadlines first, in the largest batch for which a greedy
  look-ahead over all waiting requests misses no deadline that can still be met;
* strategy (e) adds :class:`AdmissionController` to (d): a new robot is only admitted when a worst-case
  prediction (all admitted robots request at once) keeps every budget.

Scheduling decisions use a conservative estimate of the service time (``estimate_s``, the 90th percentile);
the simulated service time is drawn from the model with a stream keyed by the Brain's seed and the batch
number, so a fleet run is reproducible bit for bit. Plans are computed per request (``compute``); batching
changes when they arrive, not what they contain.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol

from kortyn.virtual_time import (
    TIME_EPS_S,
    BrainCounters,
    BrainJob,
    Compute,
    Priority,
    Simulation,
    SimulationError,
    stream_rng,
)

Z90 = 1.2815515655446004  # standard normal 90th percentile
TABLE_FORMAT = "kortyn-batch-service-v1"  # format tag of measured service-time tables


class BatchModelError(ValueError):
    """Invalid batch service model (profile parameters or measurement table)."""


# --------------------------------------------------------------------------------------------- service
class BatchServiceModel(Protocol):
    """Service time of one inference over a batch of ``batch_size`` requests on one GPU."""

    @property
    def max_batch(self) -> int:
        """Largest batch size the model knows (larger batches are not allowed)."""
        ...

    def mean_s(self, batch_size: int) -> float: ...

    def estimate_s(self, batch_size: int) -> float:
        """Conservative planning value (90th percentile) for schedulers and admission control."""
        ...

    def sample_s(self, batch_size: int, rng: random.Random) -> float: ...


def _check_batch(batch_size: int, max_batch: int) -> None:
    if not 1 <= batch_size <= max_batch:
        raise BatchModelError(f"batch size {batch_size} outside [1, {max_batch}]")


def _lognormal_sigma(cv: float) -> float:
    return math.sqrt(math.log1p(cv * cv))


@dataclass(frozen=True)
class AffineBatchProfile:
    """``s(b) = s1 * (1 + beta * (b - 1))``, lognormal noise with coefficient of variation ``cv`` per batch.

    ``beta`` is the marginal cost of one more request relative to a single inference: 1 = no batching gain
    (``b`` requests cost ``b`` inferences), 0.1 = each extra request adds 10 % of a single inference.
    """

    s1_ms: float
    beta: float
    cv: float = 0.05
    batch_limit: int = 32

    def __post_init__(self) -> None:
        if self.s1_ms <= 0:
            raise BatchModelError(f"s1_ms must be > 0, got {self.s1_ms}")
        if not 0.0 <= self.beta <= 1.0:
            raise BatchModelError(f"beta must be in [0, 1], got {self.beta}")
        if self.cv < 0:
            raise BatchModelError(f"cv must be >= 0, got {self.cv}")
        if self.batch_limit < 1:
            raise BatchModelError(f"batch_limit must be >= 1, got {self.batch_limit}")

    @property
    def max_batch(self) -> int:
        return self.batch_limit

    def mean_s(self, batch_size: int) -> float:
        _check_batch(batch_size, self.batch_limit)
        return self.s1_ms * (1.0 + self.beta * (batch_size - 1)) / 1000.0

    def estimate_s(self, batch_size: int) -> float:
        sigma = _lognormal_sigma(self.cv)
        # mean-1 lognormal: median exp(-sigma^2/2), quantile q = median * exp(z_q * sigma)
        return self.mean_s(batch_size) * math.exp(Z90 * sigma - sigma * sigma / 2.0)

    def sample_s(self, batch_size: int, rng: random.Random) -> float:
        mean = self.mean_s(batch_size)
        if self.cv == 0:
            return mean
        sigma = _lognormal_sigma(self.cv)
        return mean * math.exp(rng.gauss(-sigma * sigma / 2.0, sigma))


@dataclass(frozen=True)
class TablePoint:
    batch_size: int
    median_ms: float
    p90_ms: float


@dataclass(frozen=True)
class TableBatchProfile:
    """Measured service times (``python -m kortyn.bench_batch``): linear interpolation between batch sizes.

    The 90th percentile is the planning estimate; samples are lognormal around the interpolated median with
    the spread the measurement showed (``p90 / median``). No extrapolation beyond the largest measured batch.
    """

    points: tuple[TablePoint, ...]
    source: str = ""

    def __post_init__(self) -> None:
        if not self.points or self.points[0].batch_size != 1:
            raise BatchModelError("a service-time table must start with batch size 1")
        sizes = [p.batch_size for p in self.points]
        if sizes != sorted(set(sizes)):
            raise BatchModelError(f"batch sizes must be strictly increasing, got {sizes}")
        for point in self.points:
            if not 0 < point.median_ms <= point.p90_ms:
                raise BatchModelError(f"need 0 < median <= p90 at batch {point.batch_size}, got {point}")

    @classmethod
    def from_file(cls, path: Path | str) -> TableBatchProfile:
        """Read a table written by the benchmark script; batch sizes that ran out of memory are skipped."""
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BatchModelError(f"cannot read service-time table {source}: {exc}") from exc
        return cls.from_dict(data, str(source))

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "") -> TableBatchProfile:
        if data.get("format") != TABLE_FORMAT:
            raise BatchModelError(f"{source}: format must be {TABLE_FORMAT!r}, got {data.get('format')!r}")
        points: list[TablePoint] = []
        for row in data.get("results", []):
            if row.get("status") != "ok":
                continue
            try:
                points.append(
                    TablePoint(int(row["batch_size"]), float(row["median_ms"]), float(row["p90_ms"]))
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise BatchModelError(f"{source}: malformed row {row}: {exc}") from exc
        points.sort(key=lambda p: p.batch_size)
        return cls(tuple(points), source)

    @property
    def max_batch(self) -> int:
        return self.points[-1].batch_size

    def _interpolate(self, batch_size: int, attribute: str) -> float:
        _check_batch(batch_size, self.max_batch)
        previous = self.points[0]
        for point in self.points:
            if point.batch_size == batch_size:
                return float(getattr(point, attribute)) / 1000.0
            if point.batch_size > batch_size:
                low = float(getattr(previous, attribute))
                high = float(getattr(point, attribute))
                share = (batch_size - previous.batch_size) / (point.batch_size - previous.batch_size)
                return (low + share * (high - low)) / 1000.0
            previous = point
        raise BatchModelError(f"batch size {batch_size} beyond the table")  # unreachable after the check

    def mean_s(self, batch_size: int) -> float:
        return self._interpolate(batch_size, "median_ms")

    def estimate_s(self, batch_size: int) -> float:
        return self._interpolate(batch_size, "p90_ms")

    def sample_s(self, batch_size: int, rng: random.Random) -> float:
        median = self.mean_s(batch_size)
        spread = self.estimate_s(batch_size) / median
        sigma = math.log(spread) / Z90 if spread > 1.0 else 0.0
        return median if sigma == 0.0 else median * math.exp(rng.gauss(0.0, sigma))


# --------------------------------------------------------------------------------------------- schedulers
@dataclass(frozen=True)
class SchedulerContext:
    """What a scheduler may use to decide: the service model and the margins of the deadline estimate."""

    service: BatchServiceModel
    downlink_margin_s: float = 0.0


Decision = list[BrainJob] | float | None  # a batch to start now | wake up at this time | nothing to do


class BatchScheduler(Protocol):
    """Queue discipline of the fleet Brain: which waiting requests are served next, and together."""

    name: str

    def push(self, job: BrainJob, now: float) -> None: ...

    def next_batch(self, now: float, ctx: SchedulerContext) -> Decision:
        """Remove and return the next batch; or a time to look again (waiting for a fuller batch)."""
        ...

    def remove_abandoned(self, now: float) -> list[BrainJob]:
        """Drop and return requests whose client gave up (gRPC deadline passed)."""
        ...

    def __len__(self) -> int: ...


def brain_deadline(job: BrainJob, downlink_margin_s: float) -> float:
    """When the Brain must have finished ``job``: its robot starves from ``due`` on, minus the trip back.

    Requests without a reported due time (older agents) are treated as due on arrival.
    """
    due = job.due if job.due is not None else job.t_arrival
    return due - downlink_margin_s


class _Queue:
    """Waiting jobs with arrival order; shared by all schedulers."""

    def __init__(self) -> None:
        self.jobs: list[BrainJob] = []
        self._order: dict[int, int] = {}
        self._counter = 0

    def push(self, job: BrainJob) -> None:
        self._order[id(job)] = self._counter
        self._counter += 1
        self.jobs.append(job)

    def order(self, job: BrainJob) -> int:
        return self._order[id(job)]

    def take(self, chosen: Sequence[BrainJob]) -> list[BrainJob]:
        ids = {id(job) for job in chosen}
        self.jobs = [job for job in self.jobs if id(job) not in ids]
        for job in chosen:
            self._order.pop(id(job), None)
        return list(chosen)

    def remove_abandoned(self, now: float) -> list[BrainJob]:
        gone = [job for job in self.jobs if job.deadline is not None and now >= job.deadline]
        return self.take(gone) if gone else []


class FifoScheduler:
    """(a) First come, first served, one request per inference."""

    name = "fifo"

    def __init__(self) -> None:
        self._queue = _Queue()

    def push(self, job: BrainJob, now: float) -> None:
        self._queue.push(job)

    def next_batch(self, now: float, ctx: SchedulerContext) -> Decision:
        jobs = self._queue.jobs
        return self._queue.take(jobs[:1]) if jobs else None

    def remove_abandoned(self, now: float) -> list[BrainJob]:
        return self._queue.remove_abandoned(now)

    def __len__(self) -> int:
        return len(self._queue.jobs)


class FifoBatchScheduler:
    """(b) The oldest requests, up to ``max_batch``; wait at most ``max_wait_s`` for a fuller batch."""

    name = "fifo_batch"

    def __init__(self, max_batch: int, max_wait_s: float) -> None:
        if max_batch < 1 or max_wait_s < 0:
            raise BatchModelError("FIFO batching needs max_batch >= 1 and max_wait_s >= 0")
        self.max_batch = max_batch
        self.max_wait_s = max_wait_s
        self._queue = _Queue()

    def push(self, job: BrainJob, now: float) -> None:
        self._queue.push(job)

    def next_batch(self, now: float, ctx: SchedulerContext) -> Decision:
        jobs = self._queue.jobs
        if not jobs:
            return None
        limit = min(self.max_batch, ctx.service.max_batch)
        ready_at = jobs[0].t_arrival + self.max_wait_s
        if len(jobs) >= limit or now >= ready_at - TIME_EPS_S:
            return self._queue.take(jobs[:limit])
        return ready_at

    def remove_abandoned(self, now: float) -> list[BrainJob]:
        return self._queue.remove_abandoned(now)

    def __len__(self) -> int:
        return len(self._queue.jobs)


class EdfScheduler:
    """(c) Earliest deadline first (robot that runs dry soonest), one request per inference."""

    name = "edf"

    def __init__(self) -> None:
        self._queue = _Queue()

    def push(self, job: BrainJob, now: float) -> None:
        self._queue.push(job)

    def _sorted(self, margin_s: float) -> list[BrainJob]:
        return sorted(self._queue.jobs, key=lambda j: (brain_deadline(j, margin_s), self._queue.order(j)))

    def next_batch(self, now: float, ctx: SchedulerContext) -> Decision:
        if not self._queue.jobs:
            return None
        return self._queue.take(self._sorted(ctx.downlink_margin_s)[:1])

    def remove_abandoned(self, now: float) -> list[BrainJob]:
        return self._queue.remove_abandoned(now)

    def __len__(self) -> int:
        return len(self._queue.jobs)


class EdfBatchScheduler(EdfScheduler):
    """(d) Earliest deadlines first, in batches as large as the deadlines allow.

    Candidates are prefixes of the waiting requests in deadline order. A request is *reachable* if it can
    still meet its deadline when served alone right now; requests that are late anyway never limit a batch.
    The scheduler takes the largest prefix size ``b`` (at most ``max_batch``) for which a greedy look-ahead
    misses no reachable deadline: this batch first, then the remaining requests in deadline order, each
    batch as large as its own reachable members allow (conservative service estimates, no new arrivals).
    If no size passes (overload), it takes the largest size whose own reachable members still finish in
    time, so the Brain keeps serving as many robots on time as it can.
    """

    name = "edf_batch"

    def __init__(self, max_batch: int) -> None:
        super().__init__()
        if max_batch < 1:
            raise BatchModelError(f"max_batch must be >= 1, got {max_batch}")
        self.max_batch = max_batch

    @staticmethod
    def _largest_fitting(
        start: float, deadlines: Sequence[float], first: int, limit: int, ctx: SchedulerContext
    ) -> int:
        """Largest batch from index ``first`` whose members that are reachable at ``start`` finish in time."""
        single = ctx.service.estimate_s(1)
        available = min(limit, len(deadlines) - first)
        for size in range(available, 0, -1):
            finish = start + ctx.service.estimate_s(size)
            members = deadlines[first : first + size]
            if all(finish <= d + TIME_EPS_S for d in members if start + single <= d + TIME_EPS_S):
                return size
        return 1

    def _look_ahead_ok(
        self,
        now: float,
        size: int,
        deadlines: Sequence[float],
        reachable: Sequence[bool],
        limit: int,
        ctx: SchedulerContext,
    ) -> bool:
        """``True`` if serving ``size`` now and the rest greedily misses no deadline reachable now."""
        single = ctx.service.estimate_s(1)
        time = now + ctx.service.estimate_s(size)
        if any(reachable[i] and time > deadlines[i] + TIME_EPS_S for i in range(size)):
            return False
        index = size
        while index < len(deadlines):
            if reachable[index] and time + single > deadlines[index] + TIME_EPS_S:
                return False  # a request that could have been served on time now cannot be any more
            batch = self._largest_fitting(time, deadlines, index, limit, ctx)
            time += ctx.service.estimate_s(batch)
            if any(reachable[j] and time > deadlines[j] + TIME_EPS_S for j in range(index, index + batch)):
                return False
            index += batch
        return True

    def next_batch(self, now: float, ctx: SchedulerContext) -> Decision:
        if not self._queue.jobs:
            return None
        ordered = self._sorted(ctx.downlink_margin_s)
        deadlines = [brain_deadline(job, ctx.downlink_margin_s) for job in ordered]
        single = ctx.service.estimate_s(1)
        reachable = [now + single <= deadline + TIME_EPS_S for deadline in deadlines]
        limit = min(self.max_batch, ctx.service.max_batch)
        largest = min(limit, len(ordered))
        for size in range(largest, 0, -1):
            if self._look_ahead_ok(now, size, deadlines, reachable, limit, ctx):
                return self._queue.take(ordered[:size])
        fallback = self._largest_fitting(now, deadlines, 0, limit, ctx)
        return self._queue.take(ordered[:fallback])


SchedulerName = Literal["fifo", "fifo_batch", "edf", "edf_batch"]


def make_batch_scheduler(
    name: SchedulerName, max_batch: int = 16, max_wait_s: float = 0.025
) -> BatchScheduler:
    if name == "fifo":
        return FifoScheduler()
    if name == "fifo_batch":
        return FifoBatchScheduler(max_batch, max_wait_s)
    if name == "edf":
        return EdfScheduler()
    if name == "edf_batch":
        return EdfBatchScheduler(max_batch)
    raise BatchModelError(f"unknown scheduler {name!r}")


# --------------------------------------------------------------------------------------------- brain
@dataclass
class BatchCounters(BrainCounters):
    """Load counters of the fleet Brain (the phase-1 counters plus batches and their sizes)."""

    batches: int = 0
    batch_sizes: dict[int, int] = field(default_factory=dict)


class BatchingBrain:
    """One GPU in virtual time: batches chosen by a scheduler, service time from a batch model.

    Requests whose client deadline (gRPC timeout) has passed before their batch starts are dropped, as
    the real Brain does. ``on_batch`` (optional) sees every batch when it starts (diagnostics).
    """

    def __init__(
        self,
        sim: Simulation,
        compute: Compute,
        scheduler: BatchScheduler,
        service: BatchServiceModel,
        seed: int,
        downlink_margin_s: float = 0.0,
        on_batch: Callable[[list[BrainJob], float], None] | None = None,
    ) -> None:
        if downlink_margin_s < 0:
            raise SimulationError(f"downlink margin must be >= 0, got {downlink_margin_s}")
        self._sim = sim
        self._compute = compute
        self._scheduler = scheduler
        self._ctx = SchedulerContext(service, downlink_margin_s)
        self._service = service
        self._seed = seed
        self._busy = False
        self._wake_at: float | None = None
        self._on_batch = on_batch
        self.counters = BatchCounters()
        self.compute_ms = 0.0
        self.cache_hits = 0

    @property
    def queue_length(self) -> int:
        return len(self._scheduler)

    @property
    def scheduler(self) -> BatchScheduler:
        return self._scheduler

    def submit(self, job: BrainJob) -> None:
        """A request reaches the Brain (call at its arrival time)."""
        self.counters.requests += 1
        self._scheduler.push(job, self._sim.now)
        self._dispatch()

    def _wake(self, time: float) -> None:
        if self._wake_at is not None and self._wake_at == time:
            self._wake_at = None
            self._dispatch()

    def _dispatch(self) -> None:
        if self._busy:
            return
        now = self._sim.now
        self.counters.abandoned += len(self._scheduler.remove_abandoned(now))
        decision = self._scheduler.next_batch(now, self._ctx)
        if decision is None:
            return
        if isinstance(decision, float):
            if decision <= now:
                raise SimulationError(f"scheduler asked to wait until {decision}, now is {now}")
            if self._wake_at is None or decision < self._wake_at:
                self._wake_at = decision
                self._sim.at(decision, Priority.BRAIN, partial(self._wake, decision))
            return
        batch = decision
        if not batch:
            raise SimulationError("scheduler returned an empty batch")
        self._wake_at = None
        size = len(batch)
        self.counters.batches += 1
        self.counters.batch_sizes[size] = self.counters.batch_sizes.get(size, 0) + 1
        service = self._service.sample_s(
            size, stream_rng("brain", self._seed, "batch", self.counters.batches)
        )
        for job in batch:
            job.t_start = now
            job.batch_size = size
            job.result = self._compute(job.request)
            self.compute_ms += job.result.compute_ms
            self.cache_hits += int(job.result.cache_hit)
            self.counters.queue_wait_ms += 1000.0 * (now - job.t_arrival)
        if self._on_batch is not None:
            self._on_batch(batch, service)
        self._busy = True
        self._sim.at(now + service, Priority.BRAIN, partial(self._complete, batch, service))

    def _complete(self, batch: list[BrainJob], service_s: float) -> None:
        self._busy = False
        self.counters.inferences += len(batch)
        self.counters.busy_ms += 1000.0 * service_s
        for job in batch:
            job.t_done = self._sim.now
            job.on_done(job)
        self._dispatch()


# --------------------------------------------------------------------------------------------- admission
@dataclass(frozen=True)
class AdmissionController:
    """(e) Admit a robot only if a worst-case prediction keeps every robot's latency budget.

    Worst case: all admitted robots request at the same instant (identical periods put them back into the
    same phase every period). The Brain serves them in deadline order, in batches of at most ``max_batch``;
    the last batch must be finished, conservatively estimated, within the budget minus both network
    margins, and the utilization with full batches must stay at or below ``max_utilization``.
    """

    service: BatchServiceModel
    budget_s: float
    period_s: float
    max_batch: int
    uplink_margin_s: float
    downlink_margin_s: float
    max_utilization: float = 0.9

    def __post_init__(self) -> None:
        if self.budget_s <= 0 or self.period_s <= 0:
            raise BatchModelError("budget_s and period_s must be > 0")
        if self.max_batch < 1 or not 0 < self.max_utilization <= 1:
            raise BatchModelError("max_batch must be >= 1 and max_utilization in (0, 1]")

    def batch_plan(self, robots: int) -> list[int]:
        """Batch sizes the Brain needs for ``robots`` simultaneous requests."""
        limit = min(self.max_batch, self.service.max_batch)
        full, rest = divmod(robots, limit)
        return [limit] * full + ([rest] if rest else [])

    def worst_case_latency_s(self, robots: int) -> float:
        """Request to response for the last robot served in the worst case (both network margins)."""
        work = sum(self.service.estimate_s(size) for size in self.batch_plan(robots))
        return self.uplink_margin_s + work + self.downlink_margin_s

    def utilization(self, robots: int) -> float:
        if robots == 0:
            return 0.0
        limit = min(self.max_batch, self.service.max_batch, robots)
        return robots / self.period_s * self.service.mean_s(limit) / limit

    def admits(self, robots: int) -> bool:
        """``True`` if ``robots`` robots (including the new one) keep their budgets in the worst case."""
        if robots < 1:
            raise BatchModelError(f"robots must be >= 1, got {robots}")
        return (
            self.worst_case_latency_s(robots) <= self.budget_s + TIME_EPS_S
            and self.utilization(robots) <= self.max_utilization + TIME_EPS_S
        )

    def capacity(self, limit: int = 1024) -> int:
        """Largest fleet the controller admits (robots join one by one)."""
        robots = 0
        while robots < limit and self.admits(robots + 1):
            robots += 1
        return robots
