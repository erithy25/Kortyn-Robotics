"""Run experiments in simulation time (phase 3a): parallel, resumable and reproducible bit for bit.

Usage::

    python -m kortyn.run_virtual --config kortyn/experiments/phase3a/validation_model.yaml
    python -m kortyn.run_virtual --config ... --profile kalibriert --run modell__gut --episodes 2
    python -m kortyn.run_virtual --digest results/phase3a/validation_model/kalibriert/modell__gut

Every run of a suite names a timing (``timings`` in the config): the latency model or the replay of a logged
real-time run. A suite (= Brain profile) is processed seed by seed: a worker process takes one seed and runs
all pending runs of the suite for it, so conditions that share a seed also share the inference cache.
Results are written in the phase-1 format to ``<results_dir>/<profile>/<run>/{episodes,requests}.jsonl``
plus ``run_meta.json``; finished episodes are skipped on restart. Episodes depend neither on the number of
workers nor on their order nor on the cache (they do depend on ``experiment.torch_threads``); ``--digest``
prints a hash over the deterministic content of run directories so that two executions can be compared.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import ctypes.util
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from kortyn.config import (
    AgentConfig,
    BrainProfile,
    ConfigError,
    ImageTransportConfig,
    KortynConfig,
    NetworkConfig,
    RunSpec,
    ScenarioConfig,
    StrategyConfig,
    TimingConfig,
    load_config,
    load_scenario,
)
from kortyn.metrics import append_jsonl, read_jsonl
from kortyn.run_experiment import (
    EPISODES_FILE,
    META_FILE,
    REQUESTS_FILE,
    RunnerError,
    git_commit,
    host_info,
    merge_run_meta,
    read_run_meta,
)
from kortyn.virtual_agent import deterministic_view

LOG = logging.getLogger("kortyn.run_virtual")


# --------------------------------------------------------------------------------------------- plans
@dataclass(frozen=True)
class VirtualRunPlan:
    """One condition of a suite in simulation time (picklable: it is sent to the worker processes)."""

    label: str
    scenario: ScenarioConfig
    timing_name: str
    timing: TimingConfig
    strategy_name: str | None
    strategy: StrategyConfig | None
    replay_run: str | None
    sync_actions_per_chunk: int | None = None
    image_transport: ImageTransportConfig | None = None
    start_mode: Literal["warm", "cold"] | None = None

    @property
    def network(self) -> NetworkConfig:
        return self.scenario.network if self.scenario.mode == "realtime" else NetworkConfig()

    def agent_config(self, base: AgentConfig) -> AgentConfig:
        """The suite's agent settings with this run's overrides."""
        config = base
        if self.sync_actions_per_chunk is not None:
            config = replace(config, sync_actions_per_chunk=self.sync_actions_per_chunk)
        if self.image_transport is not None:
            config = replace(config, image_transport=self.image_transport)
        if self.start_mode is not None:
            config = replace(config, start_mode=self.start_mode)
        return config


def select_virtual_plans(config: KortynConfig, profile: str, runs: list[str] | None) -> list[VirtualRunPlan]:
    """Runs of the profile's suite (filtered by label); every run must name a timing."""
    suite = next((s for s in config.experiment.suites if s.profile == profile), None)
    specs: list[RunSpec] = list(suite.runs) if suite is not None else []
    if runs:
        by_label = {spec.run_label: spec for spec in specs}
        unknown = [name for name in runs if name not in by_label]
        if unknown:
            raise ConfigError(
                f"profile {profile!r}: unknown run label(s) {unknown}; known: {sorted(by_label)}"
            )
        specs = [by_label[name] for name in dict.fromkeys(runs)]
    plans: list[VirtualRunPlan] = []
    for spec in specs:
        if spec.timing is None:
            raise ConfigError(f"run {spec.run_label!r} has no timing; virtual-time runs need one")
        timing = config.timing(spec.timing)
        scenario = load_scenario(spec.scenario)
        if timing.latency == "replay" and scenario.mode != "realtime":
            raise ConfigError(
                f"run {spec.run_label!r}: replay needs a real-time scenario, got {scenario.name!r}"
            )
        plans.append(
            VirtualRunPlan(
                label=spec.run_label,
                scenario=scenario,
                timing_name=spec.timing,
                timing=timing,
                strategy_name=spec.strategy,
                strategy=config.strategy(spec.strategy) if spec.strategy is not None else None,
                replay_run=(spec.replay_run or spec.scenario) if timing.latency == "replay" else None,
                sync_actions_per_chunk=spec.sync_actions_per_chunk,
                image_transport=spec.image_transport,
                start_mode=spec.start_mode,
            )
        )
    return plans


def virtual_run_entry(config: KortynConfig, profile: str, plan: VirtualRunPlan) -> dict[str, Any]:
    """Metadata of one virtual-time run: scenario, strategy and everything that shapes its latency."""
    import draccus

    entry: dict[str, Any] = {
        "scenario": plan.scenario.name,
        "mode": plan.scenario.mode,
        "network": plan.scenario.network.to_dict(),
        "timing": plan.timing_name,
        "timing_settings": draccus.encode(plan.timing),
    }
    if plan.timing.hardware is not None:
        hardware = config.hardware_profiles[plan.timing.hardware]
        entry["service_time"] = draccus.encode(hardware.service[profile])
    if plan.replay_run is not None:
        entry["replay_run"] = plan.replay_run
    if plan.sync_actions_per_chunk is not None:
        entry["sync_actions_per_chunk"] = plan.sync_actions_per_chunk
    if plan.image_transport is not None:
        entry["image_transport"] = draccus.encode(plan.image_transport)
    if plan.start_mode is not None:
        entry["start_mode"] = plan.start_mode
    if plan.strategy_name is not None:
        entry["strategy"] = plan.strategy_name
        entry["strategy_settings"] = (plan.strategy or StrategyConfig()).to_dict()
    return entry


# --------------------------------------------------------------------------------------------- worker
@dataclass(frozen=True)
class WorkerSetup:
    config_path: str
    profile: str
    plans: tuple[VirtualRunPlan, ...]
    torch_threads: int
    device: str


@dataclass
class _WorkerState:
    agents: dict[str, Any]
    plans: dict[str, VirtualRunPlan]
    brain_raw: dict[str, Any]
    env: Any


_STATE: _WorkerState | None = None
# Policies loaded in the parent before forking the workers (``share_model``): profile -> PolicyRunner.
# The forked workers inherit the weights copy-on-write instead of loading a private copy each.
_PRELOADED: dict[str, Any] = {}


def _limit_threads(torch_threads: int) -> None:
    threads = str(torch_threads)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = threads
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # forked workers: no tokenizer thread pools


def _load_runner(config: KortynConfig, profile: str, torch_threads: int, device: str) -> Any:
    from kortyn.checkpoint import ensure_checkpoint
    from kortyn.policy_runner import PolicyRunner

    return PolicyRunner(
        ensure_checkpoint(config.brain.checkpoint),
        config.profile(profile),
        device=device,
        torch_threads=torch_threads,
    )


def preload_policy(config: KortynConfig, profile: str, torch_threads: int, device: str) -> None:
    """Load the policy of ``profile`` in this process, for workers forked afterwards (``share_model``)."""
    if profile in _PRELOADED:
        return
    _limit_threads(torch_threads)
    _PRELOADED[profile] = _load_runner(config, profile, torch_threads, device)
    release_memory()


def release_memory() -> bool:
    """Return freed heap pages to the operating system (glibc ``malloc_trim``).

    Loading a VLA leaves about 1 GB of freed but resident heap behind; workers are memory-bound, so they
    trim after loading and after every seed. Returns whether a trim ran (False without glibc).
    """
    name = ctypes.util.find_library("c")
    if name is None:
        return False
    try:
        libc = ctypes.CDLL(name)
        trim = libc.malloc_trim
    except (OSError, AttributeError):
        return False
    trim(0)
    return True


def _init_worker(setup: WorkerSetup) -> None:
    """Load the policy, the environment and the timing models once per worker process."""
    global _STATE
    _limit_threads(setup.torch_threads)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from kortyn.sim_env import make_env
    from kortyn.virtual_agent import LocalBrain, VirtualTimeAgent
    from kortyn.virtual_time import ReplayLog

    config = load_config(setup.config_path)
    runner = _PRELOADED.get(setup.profile)
    if runner is None:
        runner = _load_runner(config, setup.profile, setup.torch_threads, setup.device)
    compute = LocalBrain(runner)
    env = make_env(config.env)
    agents: dict[str, Any] = {}
    for plan in setup.plans:
        replay = None
        if plan.replay_run is not None:
            replay = ReplayLog(Path(plan.timing.replay.results_dir) / setup.profile / plan.replay_run)
        agents[plan.label] = VirtualTimeAgent(
            plan.agent_config(config.agent),
            env,
            compute.spec,
            compute,
            plan.timing,
            samplers=timing_samplers(config, plan.timing, setup.profile),
            replay=replay,
            timing_name=plan.timing_name,
        )
    raw = dict(compute.spec.raw)
    _STATE = _WorkerState(agents=agents, plans={p.label: p for p in setup.plans}, brain_raw=raw, env=env)
    release_memory()


def timing_samplers(config: KortynConfig, timing: TimingConfig, profile: str) -> Any:
    """Samplers of a timing for one Brain profile (``None`` model part for replay timings)."""
    from kortyn.virtual_time import DurationSampler, ModelTiming, TimingSamplers

    model = None
    if timing.latency == "model":
        assert timing.hardware is not None
        service = config.hardware_profiles[timing.hardware].service[profile]
        model = ModelTiming(
            service=DurationSampler(service),
            client_overhead=DurationSampler(timing.client_overhead),
            transport_overhead=DurationSampler(timing.transport_overhead),
            delivery_lateness=DurationSampler(timing.delivery_lateness),
        )
    return TimingSamplers(model=model, tick=DurationSampler(timing.tick_lateness))


@dataclass(frozen=True)
class SeedResult:
    seed: int
    brain_raw: dict[str, Any]
    outcomes: list[tuple[str, dict[str, Any], list[dict[str, Any]]]]  # (label, episode, requests)
    compute_s: float


def _run_seed(seed: int, labels: list[str], profile: str) -> SeedResult:
    from kortyn.agent import EpisodeContext

    if _STATE is None:
        raise RunnerError("worker not initialized")
    started = time.perf_counter()
    outcomes: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for label in labels:
        plan = _STATE.plans[label]
        ctx = EpisodeContext(
            run=label,
            profile=profile,
            scenario=plan.scenario.name,
            mode=plan.scenario.mode,
            seed=seed,
            strategy=plan.strategy_name,
        )
        outcome = _STATE.agents[label].run_episode(ctx, plan.strategy, network=plan.network)
        outcomes.append((label, outcome.episode.to_json(), [r.to_json() for r in outcome.requests]))
    release_memory()
    return SeedResult(seed, _STATE.brain_raw, outcomes, time.perf_counter() - started)


# --------------------------------------------------------------------------------------------- runner
def done_seeds(run_dir: Path) -> set[int]:
    """Seeds with a stored episode (virtual-time episodes are deterministic: never repeated)."""
    return {int(record["seed"]) for record in read_jsonl(run_dir / EPISODES_FILE)}


def resolve_workers(requested: int, torch_threads: int) -> int:
    if requested > 0:
        return requested
    return max(1, (os.cpu_count() or 1) // torch_threads)


class VirtualSuiteRunner:
    """Runs the pending episodes of one suite (Brain profile) in worker processes.

    ``stop_file``: once this file exists, no further seed is started; the seeds in progress finish and are
    written completely, so the run can be resumed later without gaps (a graceful alternative to a signal,
    which could interrupt writing a seed's records).
    """

    def __init__(
        self,
        config: KortynConfig,
        config_path: Path,
        profile: str,
        plans: list[VirtualRunPlan],
        seeds: list[int],
        results_dir: Path,
        workers: int,
        torch_threads: int,
        stop_file: Path | None = None,
        share_model: bool = False,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.profile_name = profile
        self.profile: BrainProfile = config.profile(profile)
        self.plans = plans
        self.seeds = seeds
        self.root = results_dir / profile
        self.workers = workers
        self.torch_threads = torch_threads
        self.stop_file = stop_file
        self.share_model = share_model
        self.stopped = False

    def pending(self) -> dict[int, list[str]]:
        done = {plan.label: done_seeds(self.root / plan.label) for plan in self.plans}
        tasks: dict[int, list[str]] = {}
        for seed in self.seeds:
            labels = [plan.label for plan in self.plans if seed not in done[plan.label]]
            if labels:
                tasks[seed] = labels
        return tasks

    def run(self) -> None:
        tasks = self.pending()
        total = sum(len(labels) for labels in tasks.values())
        if total == 0:
            LOG.info("[%s] all runs complete; nothing to do", self.profile_name)
            return
        workers = min(self.workers, len(tasks))
        LOG.info(
            "[%s] %d episodes pending (%d seeds x up to %d runs), %d workers x %d torch threads",
            self.profile_name,
            total,
            len(tasks),
            len(self.plans),
            workers,
            self.torch_threads,
        )
        setup = WorkerSetup(
            config_path=str(self.config_path),
            profile=self.profile_name,
            plans=tuple(self.plans),
            torch_threads=self.torch_threads,
            device=self.config.brain.device if self.config.brain.device != "auto" else "cpu",
        )
        context: multiprocessing.context.BaseContext
        if self.share_model:
            # load once here, fork the workers: they share the weights copy-on-write (memory-bound hosts)
            preload_policy(self.config, self.profile_name, self.torch_threads, setup.device)
            context = multiprocessing.get_context("fork")
        else:
            context = multiprocessing.get_context("spawn")
        started = time.monotonic()
        finished = 0
        meta_written = False
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_init_worker, initargs=(setup,)
        ) as pool:
            # Seeds are handed out on demand (never more in flight than workers), so a stop request takes
            # effect after the seeds in progress.
            queue = iter(tasks.items())
            running: dict[concurrent.futures.Future[SeedResult], int] = {}

            def submit_next() -> None:
                item = next(queue, None)
                if item is not None:
                    seed, labels = item
                    running[pool.submit(_run_seed, seed, labels, self.profile_name)] = seed

            for _ in range(workers):
                submit_next()
            try:
                while running:
                    done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in sorted(done, key=running.__getitem__):
                        del running[future]
                        result = future.result()
                        if not meta_written:
                            self._write_meta(
                                result.brain_raw, [label for labels in tasks.values() for label in labels]
                            )
                            meta_written = True
                        for label, episode, requests in result.outcomes:
                            run_dir = self.root / label
                            append_jsonl(run_dir / EPISODES_FILE, [episode])
                            append_jsonl(run_dir / REQUESTS_FILE, requests)
                        finished += len(result.outcomes)
                        self._log(result, finished, total, started)
                        if not self.stopped and self.stop_file is not None and self.stop_file.exists():
                            self.stopped = True
                            LOG.info(
                                "[%s] stop requested (%s): %d seeds in progress are finished, %d not started",
                                self.profile_name,
                                self.stop_file,
                                len(running),
                                sum(1 for _ in queue),
                            )
                        if not self.stopped:
                            submit_next()
            except BaseException:
                for future in running:
                    future.cancel()
                raise

    def _log(self, result: SeedResult, finished: int, total: int, started: float) -> None:
        elapsed = time.monotonic() - started
        eta_min = elapsed / finished * (total - finished) / 60.0
        summary = " ".join(
            f"{label}:{'S' if episode['success'] else 'f'}{episode['starved_ticks']}"
            for label, episode, _ in result.outcomes
        )
        LOG.info(
            "[%s %d/%d, ETA %.0f min] seed %d (%.0f s) %s",
            self.profile_name,
            finished,
            total,
            eta_min,
            result.seed,
            result.compute_s,
            summary,
        )

    def _write_meta(self, brain_raw: dict[str, Any], session_runs: list[str]) -> None:
        current: dict[str, Any] = {
            "profile": self.profile_name,
            "profile_settings": {
                "noise_scheduler_type": self.profile.noise_scheduler_type,
                "num_inference_steps": self.profile.num_inference_steps,
                "n_action_steps": self.profile.n_action_steps,
            },
            "clock": "virtual",
            "brain": brain_raw,
            "runs": {
                plan.label: virtual_run_entry(self.config, self.profile_name, plan) for plan in self.plans
            },
            "agent": {
                "actions_per_chunk": self.config.agent.actions_per_chunk,
                "refill_threshold": self.config.agent.refill_threshold,
                "start_mode": self.config.agent.start_mode,
                "request_timeout_s": self.config.agent.request_timeout_s,
                "sync_actions_per_chunk": self.config.agent.sync_actions_per_chunk,
            },
            "env": env_meta(self.config),
            "seeds": [self.seeds[0], self.seeds[-1]],
            "workers": self.workers,
            "torch_threads": self.torch_threads,
            "share_model": self.share_model,
            "host": host_info(),
            "git_commit": git_commit(),
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path = self.root / META_FILE
        previous = read_run_meta(path)
        labels = set(current["runs"]) | set(previous.get("runs", {}) if previous is not None else {})
        with_data = {label for label in labels if (self.root / label / EPISODES_FILE).is_file()}
        meta = merge_run_meta(previous, current, sorted(set(session_runs)), with_data)
        self.root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def env_meta(config: KortynConfig) -> dict[str, Any]:
    """Environment identity of a results directory (phase-1 keys; Meta-World adds task and camera)."""
    env = config.env
    meta: dict[str, Any] = {"gym_id": env.gym_id, "fps": env.fps, "max_episode_steps": env.max_episode_steps}
    if env.kind == "metaworld" and env.metaworld is not None:
        meta = {
            "kind": "metaworld",
            "task": env.metaworld.task,
            "camera_name": env.metaworld.camera_name,
            "instruction": env.metaworld.instruction,
            "fps": env.fps,
            "max_episode_steps": env.max_episode_steps,
        }
    return meta


# --------------------------------------------------------------------------------------------- digest
def run_digest(run_dir: Path) -> str:
    """SHA-256 over the deterministic content of a run directory (independent of record order)."""
    episodes = [deterministic_view(record) for record in read_jsonl(run_dir / EPISODES_FILE)]
    requests = list(read_jsonl(run_dir / REQUESTS_FILE))
    if not episodes:
        raise RunnerError(f"{run_dir}: no episodes")
    episodes.sort(key=lambda e: (e["seed"], e["attempt"]))
    requests.sort(key=lambda r: (r["seed"], r["attempt"], r["request_id"]))
    digest = hashlib.sha256()
    for record in (*episodes, *requests):
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Kortyn experiments in simulation time (phase 3a).")
    parser.add_argument("--config", default=None, help="experiment YAML with timings and suites")
    parser.add_argument("--profile", action="append", help="Brain profile(s) to run (default: all suites)")
    parser.add_argument("--run", action="append", help="run label(s) (default: all runs of the suite)")
    parser.add_argument("--episodes", type=int, default=None, help="episodes per run (default from config)")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--workers", type=int, default=None, help="worker processes (0 = per CPU core)")
    parser.add_argument("--threads", type=int, default=None, help="torch threads per worker")
    parser.add_argument("--fresh", action="store_true", help="delete existing results of the selected runs")
    parser.add_argument("--digest", nargs="+", default=None, help="print the digest of run directories")
    parser.add_argument(
        "--stop-file",
        default=None,
        help="stop gracefully once this file exists (seeds in progress are finished and written)",
    )
    parser.add_argument(
        "--share-model",
        action="store_true",
        help="load the policy once and fork the workers (shared weights; Linux; for memory-bound hosts)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    if args.digest:
        try:
            for directory in args.digest:
                print(f"{run_digest(Path(directory))}  {directory}")
        except RunnerError as exc:
            LOG.error("%s", exc)
            return 1
        return 0
    if args.config is None:
        parser.error("--config is required (unless --digest is given)")
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
        experiment = config.experiment
        if args.episodes is not None:
            experiment.episodes = args.episodes
        if args.seed_start is not None:
            experiment.seed_start = args.seed_start
        if args.results_dir is not None:
            experiment.results_dir = args.results_dir
        threads = args.threads if args.threads is not None else experiment.torch_threads
        workers = resolve_workers(args.workers if args.workers is not None else experiment.workers, threads)
        profiles = args.profile or [suite.profile for suite in experiment.suites]
        plans = {profile: select_virtual_plans(config, profile, args.run) for profile in profiles}
    except ConfigError as exc:
        LOG.error("%s", exc)
        return 2
    seeds = list(range(experiment.seed_start, experiment.seed_start + experiment.episodes))
    results_dir = Path(experiment.results_dir)
    stop_file = Path(args.stop_file) if args.stop_file else None
    try:
        for profile in profiles:
            if not plans[profile]:
                LOG.warning("profile %s: no runs selected", profile)
                continue
            if args.fresh:
                for plan in plans[profile]:
                    shutil.rmtree(results_dir / profile / plan.label, ignore_errors=True)
            runner = VirtualSuiteRunner(
                config,
                config_path,
                profile,
                plans[profile],
                seeds,
                results_dir,
                workers,
                threads,
                stop_file,
                share_model=args.share_model,
            )
            runner.run()
            if runner.stopped:
                LOG.info("stopped on request; finished episodes are kept and the run can be resumed")
                return 0
    except (RunnerError, ConfigError) as exc:
        LOG.error("experiment aborted: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOG.warning("interrupted; finished episodes are kept and the run can be resumed")
        return 130
    LOG.info("done. Results in %s/", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
