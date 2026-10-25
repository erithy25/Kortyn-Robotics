"""Validation rollouts (phase 3, stage 3): real LIBERO robots sharing one batching Brain in virtual time.

Stage 1 predicts each robot's latencies with timing robots; stage 2 turns them into success via the
dose-response curve. This module checks that prediction with the real thing: ``n`` robots, each with its own
LIBERO simulator and the virtual-time agent of the dose-response runs, share one
:class:`~kortyn.fleet_brain.BatchingBrain` (same scheduler, service model, network and start times as
replication 0 of stage 1). Every robot plays its share of the dose-response seeds (round robin). Robots that
finish their share early keep playing (uncounted filler episodes) until all are done, so the Brain's load
stays that of ``n`` robots.

The policy runs in worker processes forked after loading it once (shared weights). The Brain starts a
request's inference when the request arrives and collects the result when its batch completes in simulated
time; plans depend only on the request, so the rollouts are the same as with a single process.

    python -m kortyn.fleet_rollout --config kortyn/experiments/phase3/fleet.yaml --strategy d --robots 4
    python -m kortyn.fleet_rollout --config kortyn/experiments/phase3/fleet.yaml --evaluate
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import logging
import math
import multiprocessing
import sys
import time
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from kortyn.agent import EpisodeContext
from kortyn.config import ConfigError, DistributionConfig, StrategyConfig, load_config
from kortyn.fleet import (
    FleetSimConfig,
    ValidationConfig,
    _timing,
    load_fleet_config,
    service_model,
    start_times,
)
from kortyn.fleet_brain import AdmissionController, BatchingBrain, make_batch_scheduler
from kortyn.fleet_predict import (
    DoseResponse,
    PredictionError,
    episode_prediction,
    load_dose_response,
    paired_difference_interval,
    read_stage1,
)
from kortyn.metrics import append_jsonl, read_jsonl
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.virtual_time import (
    Command,
    ComputeResult,
    DurationSampler,
    ModelChannel,
    Priority,
    SampledTicks,
    Simulation,
    VirtualLink,
    WakeAt,
    stream_seed,
)

LOG = logging.getLogger("kortyn.fleet_rollout")

EPISODES_FILE = "episodes.jsonl"
REQUESTS_FILE = "requests.jsonl"
META_FILE = "meta.json"


# --------------------------------------------------------------------------------------------- compute
_POOL_BRAIN: Any = None  # the worker's LocalBrain (set by the pool initializer)


def _pool_init(profile: str) -> None:
    """Worker initializer: wrap the policy the parent loaded before forking (shared weights)."""
    global _POOL_BRAIN
    from kortyn.run_virtual import _PRELOADED
    from kortyn.virtual_agent import LocalBrain

    runner = _PRELOADED.get(profile)
    if runner is None:
        raise RuntimeError(f"policy {profile!r} was not preloaded before forking the inference workers")
    _POOL_BRAIN = LocalBrain(runner)


def _pool_infer(payload: bytes) -> ComputeResult:
    if _POOL_BRAIN is None:
        raise RuntimeError("inference worker not initialized")
    result: ComputeResult = _POOL_BRAIN(pb.InferRequest.FromString(payload))
    return result


class PoolCompute:
    """Prefetching compute: requests run in forked worker processes that share the preloaded policy.

    Futures are keyed by the request's content, so a stale entry (a request the Brain dropped) can only
    ever stand for an identical request, whose plan is identical.
    """

    def __init__(self, profile: str, workers: int) -> None:
        if workers < 1:
            raise ConfigError(f"validation needs at least one inference worker, got {workers}")
        self._pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_pool_init,
            initargs=(profile,),
        )
        self._pending: dict[bytes, concurrent.futures.Future[ComputeResult]] = {}
        self.cache_hits = 0
        self.inferences = 0
        self.wait_s = 0.0  # wall time the simulation waited for results

    @staticmethod
    def _key(payload: bytes) -> bytes:
        return hashlib.sha256(payload).digest()

    def _submit(self, payload: bytes) -> concurrent.futures.Future[ComputeResult]:
        key = self._key(payload)
        future = self._pending.get(key)
        if future is None:
            future = self._pool.submit(_pool_infer, payload)
            self._pending[key] = future
        return future

    def prefetch(self, request: pb.InferRequest) -> None:
        self._submit(request.SerializeToString(deterministic=True))

    def __call__(self, request: pb.InferRequest) -> ComputeResult:
        payload = request.SerializeToString(deterministic=True)
        future = self._submit(payload)
        started = time.perf_counter()
        result = future.result()
        self.wait_s += time.perf_counter() - started
        self._pending.pop(self._key(payload), None)
        self.inferences += 1
        self.cache_hits += int(result.cache_hit)
        return result

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


# --------------------------------------------------------------------------------------------- case
@dataclass(frozen=True)
class ValidationCase:
    strategy: str
    robots: int

    @property
    def label(self) -> str:
        return f"{self.strategy}_n{self.robots}"


def assign_seeds(seeds: list[int], robots: int, episodes: int) -> list[list[int]]:
    """Round robin: robot ``i`` plays ``seeds[i], seeds[i + n], ...``; ``ceil(episodes / n)`` each."""
    if robots < 1 or episodes < 1 or not seeds:
        raise ConfigError("need robots >= 1, episodes >= 1 and at least one seed")
    per_robot = math.ceil(episodes / robots)
    return [[seeds[(i + j * robots) % len(seeds)] for j in range(per_robot)] for i in range(robots)]


@dataclass
class _Progress:
    unfinished: int  # robots that have not finished their counted episodes
    counted_done: int = 0
    total_counted: int = 0
    started: float = field(default_factory=time.perf_counter)


def experiment_seeds(experiment_path: str) -> list[int]:
    experiment = load_config(experiment_path).experiment
    return list(range(experiment.seed_start, experiment.seed_start + experiment.episodes))


def _check_consistency(fleet: FleetSimConfig, experiment: Any) -> None:
    """The robots of stage 1 must follow the rollout agent's control loop."""
    agent = experiment.agent
    problems = []
    if agent.actions_per_chunk != fleet.agent.actions_per_chunk:
        problems.append(f"chunk {agent.actions_per_chunk} vs {fleet.agent.actions_per_chunk}")
    if agent.refill_threshold != fleet.agent.refill_threshold:
        problems.append(f"threshold {agent.refill_threshold} vs {fleet.agent.refill_threshold}")
    if abs(experiment.env.fps - fleet.agent.fps) > 1e-9:
        problems.append(f"fps {experiment.env.fps} vs {fleet.agent.fps}")
    if abs(agent.request_timeout_s - fleet.agent.request_timeout_s) > 1e-9:
        problems.append(f"timeout {agent.request_timeout_s} vs {fleet.agent.request_timeout_s}")
    if problems:
        raise ConfigError("experiment and fleet agent differ: " + ", ".join(problems))


def run_case(fleet: FleetSimConfig, case: ValidationCase, out_dir: Path) -> dict[str, Any]:
    """Run one validation case and write its episodes and requests to ``out_dir``."""
    from kortyn.run_virtual import (
        _PRELOADED,
        preload_policy,
        release_memory,
        select_virtual_plans,
        timing_samplers,
    )
    from kortyn.sim_env import make_env
    from kortyn.virtual_agent import LocalBrain, VirtualTimeAgent

    validation = _validation(fleet)
    experiment = load_config(validation.experiment)
    _check_consistency(fleet, experiment)
    if case.strategy not in fleet.strategies:
        raise ConfigError(f"unknown strategy {case.strategy!r}")
    strategy = fleet.strategies[case.strategy]
    plan = select_virtual_plans(experiment, validation.profile, [validation.template_run])[0]
    device = experiment.brain.device if experiment.brain.device != "auto" else "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / EPISODES_FILE).exists():
        raise ConfigError(f"{out_dir} already holds a validation run; delete it to run the case again")

    preload_policy(experiment, validation.profile, 1, device)  # before forking the inference workers
    spec = LocalBrain(_PRELOADED[validation.profile]).spec
    compute = PoolCompute(validation.profile, validation.workers)
    try:
        agent_config = plan.agent_config(experiment.agent)
        samplers = timing_samplers(experiment, plan.timing, validation.profile)
        agent_strategy = plan.strategy or StrategyConfig()
        network = fleet.networks[validation.network]
        margin = network.margin_ms / 1000.0
        sim = Simulation()
        model = service_model(fleet, validation.gpu, _batching_key(fleet, validation, case.strategy))
        brain = BatchingBrain(
            sim,
            compute,
            make_batch_scheduler(
                strategy.scheduler, max_batch=strategy.max_batch, max_wait_s=strategy.max_wait_ms / 1000.0
            ),
            model,
            seed=stream_seed("fleet-brain", fleet.seed, case.robots, 0) % (2**31),
            downlink_margin_s=margin,
        )
        controller = (
            AdmissionController(
                service=model,
                budget_s=fleet.agent.budget_s,
                period_s=fleet.agent.period_s,
                max_batch=strategy.max_batch,
                uplink_margin_s=margin,
                downlink_margin_s=margin,
                max_utilization=strategy.max_utilization,
            )
            if strategy.admission
            else None
        )
        timing = _timing(fleet.agent)
        no_lateness = DurationSampler(DistributionConfig(kind="constant", value_ms=0.0))
        run_seed = stream_seed("fleet-run", fleet.seed, validation.network, case.robots, 0) % (2**31)
        scenario_network = _scenario_network(network.scenario)
        seeds = experiment_seeds(validation.experiment)
        assignments = assign_seeds(seeds, case.robots, validation.episodes)
        starts = start_times(fleet, case.robots, 0)
        admitted = [controller is None or controller.admits(index + 1) for index in range(case.robots)]
        robots = [index for index in range(case.robots) if admitted[index]]
        progress = _Progress(unfinished=len(robots), total_counted=sum(len(assignments[i]) for i in robots))
        LOG.info(
            "[%s] %d of %d robots admitted, %d counted episodes, %d inference workers",
            case.label,
            len(robots),
            case.robots,
            progress.total_counted,
            validation.workers,
        )

        def robot(index: int, agent: Any) -> Generator[Command, Any, None]:
            yield WakeAt(starts[index], Priority.TICK)
            assigned = assignments[index]
            episode = 0
            while True:
                counted = episode < len(assigned)
                if not counted and progress.unfinished == 0:
                    return
                seed = assigned[episode % len(assigned)]
                episode_seed = stream_seed("fleet-link", run_seed, index, episode)
                link = VirtualLink(sim, scenario_network, episode_seed % (2**31))
                channel = ModelChannel(sim, brain, link, timing, episode_seed, 0, f"robot-{index}")
                ticks = SampledTicks(no_lateness, seed, 0, f"robot-{index}")
                ctx = EpisodeContext(
                    run=case.label,
                    profile=validation.profile,
                    scenario=validation.network,
                    mode="realtime",
                    seed=seed,
                    strategy=plan.strategy_name,
                )
                t_start = sim.now
                outcome = yield from agent.episode_process(ctx, agent_strategy, sim, channel, ticks, brain)
                outcome.episode.clock = "virtual"
                record = outcome.episode.to_json()
                record.update(
                    {
                        "case": case.label,
                        "robot": index,
                        "robot_episode": episode,
                        "counted": counted,
                        "t_start_s": t_start,
                        "t_end_s": sim.now,
                    }
                )
                append_jsonl(out_dir / EPISODES_FILE, [record])
                requests = [
                    r.to_json() | {"robot": index, "robot_episode": episode} for r in outcome.requests
                ]
                append_jsonl(out_dir / REQUESTS_FILE, requests)
                episode += 1
                if counted:
                    progress.counted_done += 1
                    elapsed = time.perf_counter() - progress.started
                    LOG.info(
                        "[%s] %d/%d episodes (robot %d, seed %d, %s, %.0f s simulated, %.0f s wall)",
                        case.label,
                        progress.counted_done,
                        progress.total_counted,
                        index,
                        seed,
                        "S" if record["success"] else "f",
                        sim.now,
                        elapsed,
                    )
                if episode == len(assigned):
                    progress.unfinished -= 1

        for index in robots:
            env = make_env(experiment.env)
            agent = VirtualTimeAgent(
                replace(agent_config, client_id=f"robot-{index}"),
                env,
                spec,
                compute,
                plan.timing,
                samplers=samplers,
                timing_name=plan.timing_name,
            )
            sim.process(robot(index, agent))
        gc.collect()
        release_memory()
        started = time.perf_counter()
        sim.run()
        counters = brain.counters
        meta = {
            "case": case.label,
            "strategy": case.strategy,
            "robots": case.robots,
            "admitted": len(robots),
            "gpu": validation.gpu,
            "batching": _batching_key(fleet, validation, case.strategy),
            "network": validation.network,
            "template_run": validation.template_run,
            "episodes_counted": progress.counted_done,
            "simulated_s": sim.now,
            "wall_s": time.perf_counter() - started,
            "inferences": compute.inferences,
            "cache_hits": compute.cache_hits,
            "wait_s": compute.wait_s,
            "batches": counters.batches,
            "batch_sizes": counters.batch_sizes,
            "busy_share": counters.busy_ms / (1000.0 * sim.now) if sim.now > 0 else 0.0,
            "starts": starts,
        }
        (out_dir / META_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return meta
    finally:
        compute.close()


def _scenario_network(name: str) -> Any:
    from kortyn.config import load_scenario

    return load_scenario(name).network


def _validation(fleet: FleetSimConfig) -> ValidationConfig:
    if fleet.validation is None:
        raise ConfigError("the fleet config has no 'validation' section")
    return fleet.validation


def _batching_key(fleet: FleetSimConfig, validation: ValidationConfig, strategy: str) -> str:
    """Stage 1 runs strategies without batching under the first batching profile only (irrelevant there)."""
    if fleet.gpus[validation.gpu].table is not None:
        return "gemessen"
    return validation.batching if fleet.strategies[strategy].batches else next(iter(fleet.batching))


# --------------------------------------------------------------------------------------------- evaluation
@dataclass
class CaseEvaluation:
    case: str
    episodes: int
    measured: float
    predicted: float
    difference: float
    interval: tuple[float, float]
    passes: bool
    admitted: int
    robots: int


def evaluate_case(
    fleet: FleetSimConfig, case: ValidationCase, dose: DoseResponse, case_dir: Path
) -> CaseEvaluation:
    """Measured minus predicted success over the counted episodes, with the pre-registered interval."""
    validation = _validation(fleet)
    path = Path(fleet.results_path)
    batching = _batching_key(fleet, validation, case.strategy)
    key = (validation.network, validation.gpu, batching, case.strategy)
    records = [
        r
        for r in read_stage1(path, dose.levels_ms)
        if r.key == key and r.robots == case.robots and r.replication == 0
    ]
    if len(records) != 1:
        raise PredictionError(
            f"stage 1 has {len(records)} runs for {key}, {case.robots} robots, replication 0"
        )
    robots = records[0].robot_records
    measured: list[float] = []
    predicted: list[float] = []
    for episode in read_jsonl(case_dir / EPISODES_FILE):
        if not episode["counted"]:
            continue
        weights = robots[int(episode["robot"])].weights
        if weights is None:
            raise PredictionError(f"robot {episode['robot']} has no stage-1 latencies in {key}")
        measured.append(float(bool(episode["success"])))
        predicted.append(episode_prediction(dose, int(episode["seed"]), weights))
    difference, interval = paired_difference_interval(
        measured, predicted, validation.confidence, validation.bootstrap, validation.seed
    )
    meta = json.loads((case_dir / META_FILE).read_text(encoding="utf-8"))
    return CaseEvaluation(
        case=case.label,
        episodes=len(measured),
        measured=sum(measured) / len(measured),
        predicted=sum(predicted) / len(predicted),
        difference=difference,
        interval=interval,
        passes=interval[0] <= 0.0 <= interval[1],
        admitted=int(meta["admitted"]),
        robots=case.robots,
    )


def evaluate_all(fleet: FleetSimConfig) -> list[CaseEvaluation]:
    validation = _validation(fleet)
    if fleet.prediction is None:
        raise ConfigError("the fleet config has no 'prediction' section")
    dose = load_dose_response(fleet.prediction)
    root = Path(validation.results_dir)
    evaluations = []
    for case_dir in sorted(p for p in root.iterdir() if (p / META_FILE).exists()):
        meta = json.loads((case_dir / META_FILE).read_text(encoding="utf-8"))
        case = ValidationCase(str(meta["strategy"]), int(meta["robots"]))
        evaluations.append(evaluate_case(fleet, case, dose, case_dir))
    return evaluations


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase-3 validation: real robots sharing one batching Brain."
    )
    parser.add_argument("--config", required=True, help="fleet config with 'validation' and 'prediction'")
    parser.add_argument("--strategy", default=None, help="strategy of the case (key in the fleet config)")
    parser.add_argument("--robots", type=int, default=None, help="fleet size of the case")
    parser.add_argument("--evaluate", action="store_true", help="compare all finished cases with stage 2")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        fleet = load_fleet_config(args.config)
        validation = _validation(fleet)
        if args.evaluate:
            evaluations = evaluate_all(fleet)
            out = Path(validation.results_dir) / "evaluation.json"
            out.write_text(json.dumps([e.__dict__ for e in evaluations], indent=1), encoding="utf-8")
            for e in evaluations:
                LOG.info(
                    "%s: measured %.3f, predicted %.3f, difference %+.3f [%+.3f, %+.3f] -> %s",
                    e.case,
                    e.measured,
                    e.predicted,
                    e.difference,
                    *e.interval,
                    "pass" if e.passes else "FAIL",
                )
            return 0 if all(e.passes for e in evaluations) else 1
        if args.strategy is None or args.robots is None:
            parser.error("--strategy and --robots are required (or --evaluate)")
        case = ValidationCase(args.strategy, args.robots)
        run_case(fleet, case, Path(validation.results_dir) / case.label)
    except (ConfigError, PredictionError) as exc:
        LOG.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
