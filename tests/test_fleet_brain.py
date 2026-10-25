"""Fleet Brain (phase 3): batch service models, schedulers a-d, batching Brain, admission control (e)."""

from __future__ import annotations

import functools
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kortyn.fleet_brain import (
    TABLE_FORMAT,
    AdmissionController,
    AffineBatchProfile,
    BatchingBrain,
    BatchModelError,
    EdfBatchScheduler,
    EdfScheduler,
    FifoBatchScheduler,
    FifoScheduler,
    SchedulerContext,
    SchedulerName,
    TableBatchProfile,
    brain_deadline,
    make_batch_scheduler,
)
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.virtual_time import BrainJob, ComputeResult, Priority, Simulation


def job(arrival: float, due: float | None = None, deadline: float | None = None, name: int = 0) -> BrainJob:
    return BrainJob(
        client_id=f"robot-{name}",
        request=pb.InferRequest(request_id=name),
        t_sent=arrival,
        t_arrival=arrival,
        service_s=0.0,
        deadline=deadline,
        on_done=lambda _job: None,
        due=due,
    )


def constant_model(s1_ms: float = 100.0, beta: float = 0.5, limit: int = 32) -> AffineBatchProfile:
    return AffineBatchProfile(s1_ms=s1_ms, beta=beta, cv=0.0, batch_limit=limit)


# --------------------------------------------------------------------------------------------- models
def test_affine_profile_mean_and_estimate() -> None:
    model = AffineBatchProfile(s1_ms=100.0, beta=0.25, cv=0.05, batch_limit=16)
    assert model.mean_s(1) == pytest.approx(0.100)
    assert model.mean_s(5) == pytest.approx(0.200)
    assert model.estimate_s(1) > model.mean_s(1)  # 90th percentile above the mean for 5 % spread
    assert model.estimate_s(1) == pytest.approx(0.1 * np.exp(1.28155 * 0.04997 - 0.04997**2 / 2), rel=1e-3)
    with pytest.raises(BatchModelError):
        model.mean_s(17)
    with pytest.raises(BatchModelError):
        model.mean_s(0)


def test_affine_profile_samples_have_requested_mean_and_spread() -> None:
    model = AffineBatchProfile(s1_ms=200.0, beta=1.0, cv=0.05)
    rng = random.Random(1)
    samples = np.array([model.sample_s(2, rng) for _ in range(20_000)])
    assert samples.mean() == pytest.approx(0.4, rel=5e-3)
    assert samples.std() / samples.mean() == pytest.approx(0.05, rel=0.05)
    assert np.quantile(samples, 0.9) == pytest.approx(model.estimate_s(2), rel=0.01)


@pytest.mark.parametrize(
    "kwargs", [{"s1_ms": 0, "beta": 0.5}, {"s1_ms": 1, "beta": 1.5}, {"s1_ms": 1, "beta": 0.5, "cv": -1}]
)
def test_affine_profile_rejects_invalid(kwargs: dict[str, Any]) -> None:
    with pytest.raises(BatchModelError):
        AffineBatchProfile(**kwargs)


def table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"format": TABLE_FORMAT, "gpu": "test", "results": rows}


def test_table_profile_interpolates_and_skips_oom(tmp_path: Path) -> None:
    rows = [
        {"batch_size": 1, "median_ms": 100.0, "p90_ms": 110.0, "status": "ok"},
        {"batch_size": 4, "median_ms": 160.0, "p90_ms": 180.0, "status": "ok"},
        {"batch_size": 8, "median_ms": 240.0, "p90_ms": 260.0, "status": "ok"},
        {"batch_size": 16, "status": "oom"},
    ]
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table(rows)))
    model = TableBatchProfile.from_file(path)
    assert model.max_batch == 8
    assert model.mean_s(1) == pytest.approx(0.100)
    assert model.mean_s(2) == pytest.approx(0.120)  # linear between 1 and 4
    assert model.estimate_s(6) == pytest.approx(0.220)
    with pytest.raises(BatchModelError):
        model.mean_s(9)  # no extrapolation
    rng = random.Random(3)
    samples = np.array([model.sample_s(4, rng) for _ in range(20_000)])
    assert np.median(samples) == pytest.approx(0.160, rel=0.01)
    assert np.quantile(samples, 0.9) == pytest.approx(0.180, rel=0.01)


@pytest.mark.parametrize(
    "data",
    [
        {"format": "other", "results": []},
        table([{"batch_size": 2, "median_ms": 1, "p90_ms": 1, "status": "ok"}]),
        table([{"batch_size": 1, "median_ms": 2, "p90_ms": 1, "status": "ok"}]),
        table([{"batch_size": 1, "status": "ok"}]),
    ],
)
def test_table_profile_rejects_malformed(data: dict[str, Any]) -> None:
    with pytest.raises(BatchModelError):
        TableBatchProfile.from_dict(data, "test")


def test_table_profile_unreadable_file(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(BatchModelError, match="cannot read"):
        TableBatchProfile.from_file(tmp_path / "bad.json")


# --------------------------------------------------------------------------------------------- schedulers
def ctx(model: AffineBatchProfile | None = None, margin: float = 0.0) -> SchedulerContext:
    return SchedulerContext(model or constant_model(), margin)


def test_fifo_serves_in_arrival_order_one_at_a_time() -> None:
    scheduler = FifoScheduler()
    first, second = job(0.0, name=1), job(0.1, name=2)
    scheduler.push(first, 0.0)
    scheduler.push(second, 0.1)
    assert scheduler.next_batch(0.2, ctx()) == [first]
    assert scheduler.next_batch(0.2, ctx()) == [second]
    assert scheduler.next_batch(0.2, ctx()) is None


def test_fifo_batch_waits_for_fuller_batch_then_serves() -> None:
    scheduler = FifoBatchScheduler(max_batch=3, max_wait_s=0.025)
    a = job(1.0, name=1)
    scheduler.push(a, 1.0)
    assert scheduler.next_batch(1.0, ctx()) == pytest.approx(1.025)  # wait for more
    b = job(1.01, name=2)
    scheduler.push(b, 1.01)
    assert scheduler.next_batch(1.01, ctx()) == pytest.approx(1.025)
    assert scheduler.next_batch(1.025, ctx()) == [a, b]  # waited long enough
    jobs = [job(2.0 + i * 0.001, name=10 + i) for i in range(4)]
    for item in jobs:
        scheduler.push(item, item.t_arrival)
    assert scheduler.next_batch(2.003, ctx()) == jobs[:3]  # full batch: no waiting
    assert scheduler.next_batch(2.003, ctx()) == pytest.approx(2.003 + 0.025)


def test_fifo_batch_respects_model_limit() -> None:
    scheduler = FifoBatchScheduler(max_batch=16, max_wait_s=0.0)
    jobs = [job(0.0, name=i) for i in range(10)]
    for item in jobs:
        scheduler.push(item, 0.0)
    assert scheduler.next_batch(0.0, ctx(constant_model(limit=4))) == jobs[:4]


def test_edf_serves_earliest_due_first_and_starving_robot_first() -> None:
    scheduler = EdfScheduler()
    late = job(0.0, due=5.0, name=1)
    soon = job(0.1, due=1.0, name=2)
    starving = job(0.2, due=None, name=3)  # no due time reported: due on arrival
    for item in (late, soon, starving):
        scheduler.push(item, item.t_arrival)
    order = [scheduler.next_batch(0.3, ctx()) for _ in range(3)]
    assert order == [[starving], [soon], [late]]


def test_brain_deadline_subtracts_return_trip() -> None:
    assert brain_deadline(job(0.0, due=2.0), 0.015) == pytest.approx(1.985)
    assert brain_deadline(job(0.7, due=None), 0.015) == pytest.approx(0.685)


def test_edf_batch_takes_largest_batch_without_misses() -> None:
    model = constant_model(s1_ms=100.0, beta=0.5)  # s(b) = 0.1 (1 + 0.5 (b - 1))
    scheduler = EdfBatchScheduler(max_batch=16)
    early = [job(0.0, due=0.26, name=i) for i in range(3)]
    later = [job(0.0, due=1.0, name=10 + i) for i in range(4)]
    for item in (*early, *later):
        scheduler.push(item, 0.0)
    # 7, 6 or 5 at once finish at 0.4 / 0.35 / 0.3, after the early deadline 0.26. Four (the early three
    # and one later request) finish at 0.25 <= 0.26, and the remaining three right after at 0.45 <= 1.0:
    # the largest batch that misses nothing.
    assert scheduler.next_batch(0.0, ctx(model)) == [*early, later[0]]


def test_edf_batch_overload_serves_as_many_as_possible_on_time() -> None:
    model = constant_model(s1_ms=100.0, beta=0.5)
    scheduler = EdfBatchScheduler(max_batch=16)
    jobs = [job(0.0, due=0.28 + 0.001 * i, name=i) for i in range(5)]
    for item in jobs:
        scheduler.push(item, 0.0)
    # five due at ~0.28 cannot all be on time: s(5) = 0.3. No look-ahead plan avoids every miss, so the
    # scheduler takes the largest batch whose members finish in time: four at 0.25 (the fifth is late).
    assert scheduler.next_batch(0.0, ctx(model)) == jobs[:4]


def test_edf_batch_ignores_requests_that_are_late_anyway() -> None:
    model = constant_model(s1_ms=100.0, beta=0.25)
    scheduler = EdfBatchScheduler(max_batch=8)
    late = job(0.0, due=0.05, name=0)  # cannot be met even alone (0.1 s service)
    others = [job(0.0, due=2.0, name=i) for i in range(1, 6)]
    for item in (late, *others):
        scheduler.push(item, 0.0)
    batch = scheduler.next_batch(0.0, ctx(model))
    assert isinstance(batch, list)
    assert len(batch) == 6  # the late one does not cap the batch
    assert batch[0] is late


def test_edf_batch_with_margin_and_single_job() -> None:
    model = constant_model(s1_ms=100.0, beta=0.5)
    scheduler = EdfBatchScheduler(max_batch=4)
    only = job(0.0, due=0.105, name=1)
    scheduler.push(only, 0.0)
    assert scheduler.next_batch(0.0, ctx(model, margin=0.01)) == [only]  # late with margin: still served


def test_abandoned_jobs_are_removed() -> None:
    scheduler = make_batch_scheduler("fifo_batch", max_batch=4, max_wait_s=0.01)
    gone = job(0.0, deadline=1.0, name=1)
    kept = job(0.5, deadline=5.0, name=2)
    scheduler.push(gone, 0.0)
    scheduler.push(kept, 0.5)
    assert scheduler.remove_abandoned(1.0) == [gone]
    assert len(scheduler) == 1


def test_make_batch_scheduler_rejects_unknown() -> None:
    with pytest.raises(BatchModelError):
        make_batch_scheduler("lifo")  # type: ignore[arg-type]
    with pytest.raises(BatchModelError):
        FifoBatchScheduler(max_batch=0, max_wait_s=0.0)
    with pytest.raises(BatchModelError):
        EdfBatchScheduler(max_batch=0)


# --------------------------------------------------------------------------------------------- brain
class RecordingCompute:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: pb.InferRequest) -> ComputeResult:
        self.calls += 1
        return ComputeResult(np.zeros((2, 2), dtype=np.float32))


def run_brain(
    scheduler_name: SchedulerName, arrivals: list[tuple[float, float | None]], beta: float = 0.5
) -> list[BrainJob]:
    sim = Simulation()
    model = constant_model(s1_ms=100.0, beta=beta)
    brain = BatchingBrain(
        sim,
        RecordingCompute(),
        make_batch_scheduler(scheduler_name, max_batch=8, max_wait_s=0.02),
        model,
        seed=7,
    )
    done: list[BrainJob] = []
    jobs = []
    for index, (arrival, due) in enumerate(arrivals):
        item = job(arrival, due=due, name=index)
        item.on_done = done.append
        jobs.append(item)
        sim.at(arrival, Priority.BRAIN, functools.partial(brain.submit, item))
    sim.run()
    assert len(done) == len(arrivals)
    return jobs


def test_fifo_brain_serves_sequentially() -> None:
    jobs = run_brain("fifo", [(0.0, None), (0.0, None), (0.0, None)])
    assert [j.t_done for j in jobs] == pytest.approx([0.1, 0.2, 0.3])
    assert all(j.batch_size == 1 for j in jobs)


def test_fifo_batch_brain_batches_waiting_requests() -> None:
    jobs = run_brain("fifo_batch", [(0.0, None), (0.005, None), (0.01, None), (0.5, None)])
    # first three wait until 0.02 (oldest + 20 ms), served together: 0.1 * (1 + 0.5 * 2) = 0.2 s
    assert [j.t_start for j in jobs[:3]] == pytest.approx([0.02, 0.02, 0.02])
    assert [j.t_done for j in jobs[:3]] == pytest.approx([0.22, 0.22, 0.22])
    assert jobs[0].batch_size == 3
    assert jobs[3].t_start == pytest.approx(0.52)  # alone after its own wait


def test_brain_counts_batches_and_abandons_expired() -> None:
    sim = Simulation()
    brain = BatchingBrain(sim, RecordingCompute(), FifoScheduler(), constant_model(), seed=1)
    done: list[BrainJob] = []
    first = job(0.0, name=1)
    expired = job(0.0, deadline=0.05, name=2)  # client gives up while the first is served
    for item in (first, expired):
        item.on_done = done.append
        sim.at(0.0, Priority.BRAIN, functools.partial(brain.submit, item))
    sim.run()
    assert done == [first]
    assert brain.counters.abandoned == 1
    assert brain.counters.batches == 1
    assert brain.counters.batch_sizes == {1: 1}


def test_brain_service_draws_are_reproducible() -> None:
    def run(seed: int) -> list[float]:
        sim = Simulation()
        model = AffineBatchProfile(s1_ms=100.0, beta=0.5, cv=0.1)
        brain = BatchingBrain(sim, RecordingCompute(), FifoScheduler(), model, seed=seed)
        jobs = [job(0.0, name=i) for i in range(5)]
        for item in jobs:
            sim.at(0.0, Priority.BRAIN, functools.partial(brain.submit, item))
        sim.run()
        return [float(j.t_done or 0.0) for j in jobs]

    assert run(3) == run(3)
    assert run(3) != run(4)


# --------------------------------------------------------------------------------------------- admission
def test_admission_controller_worst_case_and_capacity() -> None:
    model = constant_model(s1_ms=100.0, beta=0.25, limit=32)  # s(b) = 0.1 + 0.025 (b - 1)
    controller = AdmissionController(
        service=model,
        budget_s=1.25,
        period_s=1.25,
        max_batch=16,
        uplink_margin_s=0.015,
        downlink_margin_s=0.015,
    )
    assert controller.batch_plan(20) == [16, 4]
    # 20 robots: s(16) + s(4) = 0.475 + 0.175 = 0.65 (+ margins 0.03)
    assert controller.worst_case_latency_s(20) == pytest.approx(0.68)
    assert controller.admits(20)
    capacity = controller.capacity()
    assert controller.admits(capacity)
    assert not controller.admits(capacity + 1)
    # utilization bound: 16 per batch every 0.475 s -> at most 0.9 * 1.25 / 0.475 * 16 ~ 37.9 robots
    assert controller.utilization(capacity) <= 0.9


def test_admission_controller_without_batching_gain() -> None:
    model = constant_model(s1_ms=300.0, beta=1.0)
    controller = AdmissionController(
        service=model, budget_s=1.25, period_s=1.25, max_batch=16, uplink_margin_s=0.0, downlink_margin_s=0.0
    )
    # no gain: n robots need n * 0.3 s; utilization n * 0.3 / 1.25 <= 0.9 -> n <= 3
    assert controller.capacity() == 3


def test_admission_controller_rejects_invalid() -> None:
    with pytest.raises(BatchModelError):
        AdmissionController(constant_model(), 0.0, 1.0, 4, 0.0, 0.0)
    controller = AdmissionController(constant_model(), 1.0, 1.0, 4, 0.0, 0.0)
    with pytest.raises(BatchModelError):
        controller.admits(0)
