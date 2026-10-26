"""Run the phase-1 experiment suite: Brain + netem proxy + real-time agent, N episodes per scenario.

For every suite (= Brain profile) the runner

1. starts the Brain (downloads and migrates the checkpoint on first use) and waits until it serves,
2. starts the netem proxy once, in front of the Brain (never restarted: no reconnects inside episodes),
3. runs the scenarios **interleaved seed by seed** (seed 1000: all scenarios, then seed 1001, ...), so slow
   drift of the machine cannot favour one scenario, re-configuring the proxy before every episode,
4. appends every attempt to ``results/<profile>/<run>/{episodes,requests}.jsonl`` immediately
   (crash-safe and resumable: finished seeds are skipped on the next start),
5. repeats attempts that fail the validity gate (up to ``max_attempts_per_episode``).

Examples::

    python -m kortyn.run_experiment                                  # full suite from base.yaml
    python -m kortyn.run_experiment --profile kalibriert --run gut --episodes 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from kortyn.agent import (
    Agent,
    BrainClient,
    BrainSpec,
    BrainUnavailableError,
    EpisodeContext,
    IncompatibleBrainError,
    ProxyControl,
    ProxyControlError,
)
from kortyn.config import (
    DEFAULT_BASE_CONFIG,
    BrainProfile,
    ConfigError,
    KortynConfig,
    NetworkConfig,
    ScenarioConfig,
    StrategyConfig,
    load_config,
    load_scenario,
)
from kortyn.metrics import EpisodeRecord, append_jsonl, read_jsonl
from kortyn.process_utils import set_cpu_affinity
from kortyn.proto import kortyn_brain_pb2 as pb

LOG = logging.getLogger("kortyn.run")

EPISODES_FILE = "episodes.jsonl"
REQUESTS_FILE = "requests.jsonl"
META_FILE = "run_meta.json"
PUBLISHED_REFERENCE_FILE = "published_reference.json"

# Settings that must be identical for every episode stored under one results/<profile>/ directory:
# the runs of a profile are compared with each other, so a resumed or extended suite must not mix them.
META_MATCHING_KEYS = ("profile", "profile_settings", "agent", "env")
# Settings introduced after results were stored, with the value that reproduces the earlier behaviour: a
# stored run entry without such a key matches a current entry that has the key with exactly this value.
BACKWARD_COMPATIBLE_DEFAULTS: dict[str, Any] = {
    "per_seed": False,  # phase 3a: empirical latency sources drawn per episode seed
    "image_size": None,  # phase 3: camera resolution accepted by policies that resize images themselves
    "libero": None,  # phase 3: LIBERO section of the env config
}
BRAIN_IDENTITY_KEYS = (
    "policy_type",
    "checkpoint",
    "device",
    "sampler",
    "n_obs_steps",
    "max_actions",
    "lerobot_version",
    "torch_version",
    "torch_threads",
)


class RunnerError(RuntimeError):
    """Infrastructure failure (Brain/proxy crashed, ports busy, ...)."""


# --------------------------------------------------------------------------------------------- processes
@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[bytes]
    log_path: Path
    log_handle: IO[bytes]

    def alive(self) -> bool:
        return self.process.poll() is None

    def tail(self, lines: int = 40) -> str:
        self.log_handle.flush()
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def stop(self, timeout_s: float = 15.0) -> None:
        if self.alive():
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                LOG.warning("%s did not stop within %.0f s; killing", self.name, timeout_s)
                self.process.kill()
                self.process.wait(timeout=5)
        self.log_handle.close()


def spawn(name: str, args: list[str], log_path: Path) -> ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    LOG.info("starting %s: %s (log: %s)", name, " ".join(args), log_path)
    process = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT)
    return ManagedProcess(name, process, log_path, handle)


# Cores this process may use at start-up. The runner pins *itself* to the agent core during a suite, so the
# layout must never be derived from the current (narrowed) affinity - later suites would otherwise start
# their Brain unpinned, inherit the single agent core and run several times slower.
_INITIAL_AFFINITY: frozenset[int] | None = (
    frozenset(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
)


def cpu_layout(
    pin: bool, available: frozenset[int] | None = _INITIAL_AFFINITY
) -> tuple[list[int], list[int]]:
    """(Brain cores, agent+proxy cores) on Linux; empty lists where pinning is off or unsupported."""
    if not pin or available is None:
        return [], []
    cores = sorted(available)
    if len(cores) < 2:
        return [], []
    return cores[:-1], cores[-1:]


def restore_affinity() -> None:
    if _INITIAL_AFFINITY is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, _INITIAL_AFFINITY)


def start_brain(
    config_path: Path,
    profile: str,
    config: KortynConfig,
    log_path: Path,
    cores: list[int],
    overrides: list[str],
) -> ManagedProcess:
    args = [
        sys.executable,
        "-m",
        "kortyn.brain",
        "--config",
        str(config_path),
        "--profile",
        profile,
        "--host",
        config.brain.host,
        "--port",
        str(config.brain.port),
        "--device",
        config.brain.device,
        "--threads",
        str(len(cores) if cores and config.brain.torch_threads == 0 else config.brain.torch_threads),
        *overrides,
    ]
    if cores:
        args = ["taskset", "-c", ",".join(str(c) for c in cores), *args] if shutil.which("taskset") else args
    return spawn("brain", args, log_path)


def wait_for_brain(process: ManagedProcess, address: str, timeout_s: float) -> pb.BrainInfo:
    deadline = time.monotonic() + timeout_s
    client = BrainClient(address)
    try:
        while True:
            if not process.alive():
                raise RunnerError(f"Brain exited with code {process.process.returncode}:\n{process.tail()}")
            try:
                return client.get_info(timeout_s=5.0)
            except BrainUnavailableError:
                if time.monotonic() > deadline:
                    raise RunnerError(f"Brain not ready after {timeout_s:.0f} s:\n{process.tail()}") from None
    finally:
        client.close()


def start_proxy(
    config_path: Path, config: KortynConfig, log_path: Path, cores: list[int]
) -> tuple[ManagedProcess, ProxyControl]:
    args = [
        sys.executable,
        "-m",
        "kortyn.netem_proxy",
        "--config",
        str(config_path),
        "--scenario",
        "gut",
        "--upstream",
        config.brain.address,
        "--host",
        config.proxy.host,
        "--port",
        str(config.proxy.port),
        "--control-port",
        str(config.proxy.control_port),
    ]
    if cores and shutil.which("taskset"):
        args = ["taskset", "-c", ",".join(str(c) for c in cores), *args]
    process = spawn("proxy", args, log_path)
    control = ProxyControl(config.proxy.host, config.proxy.control_port)
    deadline = time.monotonic() + config.experiment.proxy_startup_timeout_s
    while True:
        if not process.alive():
            raise RunnerError(f"proxy exited with code {process.process.returncode}:\n{process.tail()}")
        try:
            control.ping()
            return process, control
        except ProxyControlError:
            if time.monotonic() > deadline:
                raise RunnerError(f"proxy not ready:\n{process.tail()}") from None
            time.sleep(0.2)


# --------------------------------------------------------------------------------------------- results
def load_attempts(run_dir: Path) -> dict[int, list[EpisodeRecord]]:
    attempts: dict[int, list[EpisodeRecord]] = {}
    for record in read_jsonl(run_dir / EPISODES_FILE):
        episode = EpisodeRecord.from_json(record)
        attempts.setdefault(episode.seed, []).append(episode)
    return attempts


def git_commit() -> str | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return commit + ("-dirty" if dirty else "")


def host_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def read_run_meta(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{path}: unreadable run metadata ({exc}); repair or delete it") from exc
    if not isinstance(data, dict):
        raise RunnerError(f"{path}: run metadata must be a JSON object")
    return data


def _session_of(meta: dict[str, Any], runs: Sequence[str]) -> dict[str, Any]:
    return {
        "started_utc": meta.get("started_utc"),
        "git_commit": meta.get("git_commit"),
        "runs": list(runs),
        "cpu_pinning": meta.get("cpu_pinning"),
        "host": meta.get("host"),
        "brain_warmup_inference_ms": meta.get("brain", {}).get("warmup_inference_ms"),
    }


def same_settings(stored: Any, current: Any) -> bool:
    """Equal settings, where a key added later is equal to its absence if it has its backward-compatible value
    (:data:`BACKWARD_COMPATIBLE_DEFAULTS`)."""
    if isinstance(stored, dict) and isinstance(current, dict):
        for key in stored.keys() | current.keys():
            if key not in stored and key in BACKWARD_COMPATIBLE_DEFAULTS:
                if current[key] != BACKWARD_COMPATIBLE_DEFAULTS[key]:
                    return False
                continue
            if key not in stored or key not in current or not same_settings(stored[key], current[key]):
                return False
        return True
    if isinstance(stored, list) and isinstance(current, list):
        return len(stored) == len(current) and all(
            same_settings(a, b) for a, b in zip(stored, current, strict=True)
        )
    return bool(stored == current)


def merge_run_meta(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    session_runs: Sequence[str],
    labels_with_data: Collection[str],
) -> dict[str, Any]:
    """Combine this session's metadata with the one of earlier sessions in the same results directory.

    ``current`` describes this session (all runs of the suite), ``session_runs`` the runs that get new
    episodes now, ``labels_with_data`` the runs that already have stored episodes. Run entries and sessions
    of earlier invocations are kept (resumed or extended suites), the first start time stays. Raises
    :class:`RunnerError` if a setting that must be identical for all stored episodes has changed.
    """
    with_data = set(labels_with_data)
    previous_runs: dict[str, Any] = previous.get("runs", {}) if previous is not None else {}
    if previous is None or not with_data & set(previous_runs):
        return {**current, "sessions": [_session_of(current, session_runs)]}

    problems = [
        f"{key}: {previous.get(key)!r} -> {current.get(key)!r}"
        for key in META_MATCHING_KEYS
        if previous.get(key) != current.get(key)
    ]
    old_brain, new_brain = previous.get("brain", {}), current.get("brain", {})
    problems += [
        f"brain.{key}: {old_brain.get(key)!r} -> {new_brain.get(key)!r}"
        for key in BRAIN_IDENTITY_KEYS
        if old_brain.get(key) != new_brain.get(key)
    ]
    problems += [
        f"runs.{label}: {previous_runs[label]!r} -> {entry!r}"
        for label, entry in current["runs"].items()
        if label in with_data and label in previous_runs and not same_settings(previous_runs[label], entry)
    ]
    if problems:
        raise RunnerError(
            "the stored results were produced with different settings ("
            + "; ".join(problems)
            + "); restore the settings, or use --fresh / another --results-dir"
        )

    sessions = list(previous.get("sessions") or [_session_of(previous, list(previous_runs))])
    sessions.append(_session_of(current, session_runs))
    runs = {label: entry for label, entry in previous_runs.items() if label in with_data}
    runs.update(current["runs"])
    old_seeds = previous.get("seeds") or current["seeds"]
    return {
        **current,
        "runs": runs,
        "seeds": [min(old_seeds[0], current["seeds"][0]), max(old_seeds[1], current["seeds"][1])],
        "started_utc": previous.get("started_utc", current["started_utc"]),
        "sessions": sessions,
    }


def write_published_reference(config: KortynConfig, seeds: Sequence[int], results_dir: Path) -> None:
    """Store the checkpoint's published per-seed evaluation results for the seeds used here."""
    target = results_dir / PUBLISHED_REFERENCE_FILE
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            config.brain.checkpoint.repo_id, "eval_info.json", revision=config.brain.checkpoint.revision
        )
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("published reference not available (%s); continuing without it", exc)
        return
    episodes = [e for e in data.get("per_episode", []) if "seed" in e]
    per_seed = {int(e["seed"]): bool(e["success"]) for e in episodes}
    rewards = {int(e["seed"]): float(e["max_reward"]) for e in episodes if "max_reward" in e}
    selected = {seed: per_seed[seed] for seed in seeds if seed in per_seed}
    selected_rewards = [rewards[seed] for seed in seeds if seed in rewards]
    payload = {
        "source": f"{config.brain.checkpoint.repo_id}@{config.brain.checkpoint.revision}/eval_info.json",
        "sampler": "DDPM-100 (checkpoint default, GPU, LeRobot eval)",
        "seeds": len(selected),
        "successes": sum(selected.values()),
        "avg_max_reward": sum(selected_rewards) / len(selected_rewards) if selected_rewards else None,
        "all_episodes": len(per_seed),
        "all_success_rate": (sum(per_seed.values()) / len(per_seed)) if per_seed else None,
        "all_avg_max_reward": data.get("aggregated", {}).get("avg_max_reward"),
        "success_by_seed": {str(k): v for k, v in sorted(selected.items())},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------------------------- runner
@dataclass(frozen=True)
class RunPlan:
    label: str
    scenario: ScenarioConfig
    # Phase 2: freshness strategy of this run. ``strategy_name`` None = phase-1 run (baseline behaviour,
    # phase-1 metadata format).
    strategy_name: str | None = None
    strategy: StrategyConfig | None = None


class SuiteRunner:
    """Runs all selected runs of one Brain profile."""

    def __init__(
        self,
        config: KortynConfig,
        config_path: Path,
        profile_name: str,
        plans: list[RunPlan],
        seeds: list[int],
        results_dir: Path,
        brain_overrides: list[str],
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.profile_name = profile_name
        self.profile: BrainProfile = config.profile(profile_name)
        self.plans = plans
        self.seeds = seeds
        self.root = results_dir / profile_name
        self.brain_overrides = brain_overrides
        self.max_attempts = config.experiment.max_attempts_per_episode

    def _pending(self) -> dict[str, dict[int, int]]:
        """run label -> {seed: next attempt index} for seeds that still need an episode."""
        pending: dict[str, dict[int, int]] = {}
        for plan in self.plans:
            attempts = load_attempts(self.root / plan.label)
            todo: dict[int, int] = {}
            for seed in self.seeds:
                previous = attempts.get(seed, [])
                if any(e.valid for e in previous) or len(previous) >= self.max_attempts:
                    continue
                todo[seed] = len(previous)
            if todo:
                pending[plan.label] = todo
        return pending

    def run(self) -> None:
        pending = self._pending()
        total = sum(len(v) for v in pending.values())
        if total == 0:
            LOG.info("[%s] all runs complete; nothing to do", self.profile_name)
            return
        LOG.info("[%s] %d episodes pending across %d runs", self.profile_name, total, len(pending))
        brain_cores, agent_cores = cpu_layout(self.config.experiment.pin_cpus)
        logs = self.root / "logs"
        brain = start_brain(
            self.config_path,
            self.profile_name,
            self.config,
            logs / "brain.log",
            brain_cores,
            self.brain_overrides,
        )
        proxy: ManagedProcess | None = None
        control: ProxyControl | None = None
        client: BrainClient | None = None
        stats_client: BrainClient | None = None
        env = None
        try:
            info = wait_for_brain(
                brain, self.config.brain.address, self.config.experiment.brain_startup_timeout_s
            )
            LOG.info(
                "[%s] Brain ready: %s %s on %s, warm-up median %.1f ms, %d threads",
                self.profile_name,
                info.policy_type,
                info.sampler,
                info.device,
                info.warmup_inference_ms,
                info.torch_threads,
            )
            proxy, control = start_proxy(self.config_path, self.config, logs / "proxy.log", agent_cores)
            set_cpu_affinity(agent_cores, LOG)
            from kortyn.sim_env import SimEnv

            env = SimEnv(self.config.env)
            client = BrainClient(self.config.proxy.address)
            # Load counters are control-plane traffic: read directly from the Brain, not through the proxy.
            stats_client = BrainClient(self.config.brain.address)
            spec = BrainSpec.from_proto(client.get_info(timeout_s=30.0))
            agent = Agent(self.config.agent, env, client, spec, control, stats_client=stats_client)
            self._write_meta(spec, brain_cores, agent_cores, list(pending))
            self._loop(agent, control, pending, brain, proxy)
        finally:
            if env is not None:
                env.close()
            if client is not None:
                client.close()
            if stats_client is not None:
                stats_client.close()
            if control is not None:
                control.close()
            if proxy is not None:
                proxy.stop()
            brain.stop()
            restore_affinity()

    def _loop(
        self,
        agent: Agent,
        control: ProxyControl,
        pending: dict[str, dict[int, int]],
        brain: ManagedProcess,
        proxy: ManagedProcess,
    ) -> None:
        plans = {plan.label: plan for plan in self.plans}
        done = 0
        total = sum(len(v) for v in pending.values())
        started = time.monotonic()
        for seed in self.seeds:
            for label, plan in plans.items():
                if seed not in pending.get(label, {}):
                    continue
                attempt = pending[label][seed]
                while attempt < self.max_attempts:
                    network = plan.scenario.network if plan.scenario.mode == "realtime" else NetworkConfig()
                    control.configure(network, seed)
                    ctx = EpisodeContext(
                        run=label,
                        profile=self.profile_name,
                        scenario=plan.scenario.name,
                        mode=plan.scenario.mode,
                        seed=seed,
                        attempt=attempt,
                        strategy=plan.strategy_name,
                    )
                    outcome = agent.run_episode(ctx, plan.strategy)
                    run_dir = self.root / label
                    append_jsonl(run_dir / EPISODES_FILE, [outcome.episode.to_json()])
                    append_jsonl(run_dir / REQUESTS_FILE, [r.to_json() for r in outcome.requests])
                    self._log_episode(outcome.episode, done + 1, total, started)
                    if outcome.episode.valid:
                        break
                    attempt += 1
                    for process in (brain, proxy):
                        if not process.alive():
                            raise RunnerError(f"{process.name} died during the run:\n{process.tail()}")
                done += 1

    def _log_episode(self, episode: EpisodeRecord, index: int, total: int, started: float) -> None:
        elapsed = time.monotonic() - started
        eta = elapsed / index * (total - index)
        LOG.info(
            "[%s %d/%d, ETA %.0f min] seed %d %-17s %s steps=%3d starved=%3d emergency=%3d requests=%3d%s",
            self.profile_name,
            index,
            total,
            eta / 60.0,
            episode.seed,
            episode.run,
            "SUCCESS" if episode.success else "fail   ",
            episode.steps,
            episode.starved_ticks,
            episode.emergency_ticks,
            episode.n_requests,
            "" if episode.valid else f" INVALID ({'; '.join(episode.invalid_reasons)})",
        )

    def _write_meta(
        self, spec: BrainSpec, brain_cores: list[int], agent_cores: list[int], session_runs: list[str]
    ) -> None:
        """Write ``run_meta.json``, merged with earlier sessions (see :func:`merge_run_meta`)."""
        current: dict[str, Any] = {
            "profile": self.profile_name,
            "profile_settings": {
                "noise_scheduler_type": self.profile.noise_scheduler_type,
                "num_inference_steps": self.profile.num_inference_steps,
                "n_action_steps": self.profile.n_action_steps,
            },
            "brain": spec.raw,
            "runs": {plan.label: run_entry(plan) for plan in self.plans},
            "agent": {
                "actions_per_chunk": self.config.agent.actions_per_chunk,
                "refill_threshold": self.config.agent.refill_threshold,
                "start_mode": self.config.agent.start_mode,
                "request_timeout_s": self.config.agent.request_timeout_s,
                "sync_actions_per_chunk": self.config.agent.sync_actions_per_chunk,
            },
            "env": {
                "gym_id": self.config.env.gym_id,
                "fps": self.config.env.fps,
                "max_episode_steps": self.config.env.max_episode_steps,
            },
            "seeds": [self.seeds[0], self.seeds[-1]],
            "cpu_pinning": {"brain": brain_cores, "agent_and_proxy": agent_cores},
            "host": host_info(),
            "git_commit": git_commit(),
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path = self.root / META_FILE
        previous = read_run_meta(path)
        labels = set(current["runs"]) | set(previous.get("runs", {}) if previous is not None else {})
        with_data = {label for label in labels if (self.root / label / EPISODES_FILE).is_file()}
        meta = merge_run_meta(previous, current, session_runs, with_data)
        self.root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_entry(plan: RunPlan) -> dict[str, Any]:
    """Metadata of one run. Phase-1 runs (no strategy) keep their original format."""
    entry: dict[str, Any] = {
        "scenario": plan.scenario.name,
        "mode": plan.scenario.mode,
        "network": plan.scenario.network.to_dict(),
    }
    if plan.strategy_name is not None:
        entry["strategy"] = plan.strategy_name
        entry["strategy_settings"] = (plan.strategy or StrategyConfig()).to_dict()
    return entry


def select_plans(config: KortynConfig, profile: str, runs: list[str] | None) -> list[RunPlan]:
    """Runs of the profile's suite; ``runs`` filters by run label (a scenario name not used as a label in
    the suite runs that scenario ad hoc, labelled with its name)."""
    from kortyn.config import RunSpec

    suite = next((s for s in config.experiment.suites if s.profile == profile), None)
    specs = list(suite.runs) if suite is not None else []
    if runs:
        by_label = {spec.run_label: spec for spec in specs}
        specs = [by_label.get(name, RunSpec(scenario=name)) for name in dict.fromkeys(runs)]
    return [
        RunPlan(
            label=s.run_label,
            scenario=load_scenario(s.scenario),
            strategy_name=s.strategy,
            strategy=config.strategy(s.strategy),
        )
        for s in specs
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Kortyn phase-1 experiment suite.")
    parser.add_argument("--config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--profile", action="append", help="Brain profile(s) to run (default: all suites)")
    parser.add_argument("--run", action="append", help="run label(s) or scenario name(s) (default: all)")
    parser.add_argument("--episodes", type=int, default=None, help="episodes per run (default from config)")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--fresh", action="store_true", help="delete existing results of the selected runs")
    parser.add_argument("--device", default=None, help="Brain device override (auto/cpu/mps/cuda)")
    parser.add_argument("--no-pin-cpus", action="store_true", help="disable Linux CPU pinning")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
        if args.episodes is not None:
            config.experiment.episodes = args.episodes
        if args.seed_start is not None:
            config.experiment.seed_start = args.seed_start
        if args.results_dir is not None:
            config.experiment.results_dir = args.results_dir
        if args.device is not None:
            config.brain.device = args.device
        if args.no_pin_cpus:
            config.experiment.pin_cpus = False
        profiles = args.profile or [suite.profile for suite in config.experiment.suites]
        for name in profiles:
            config.profile(name)
    except ConfigError as exc:
        LOG.error("%s", exc)
        return 2

    seeds = list(
        range(config.experiment.seed_start, config.experiment.seed_start + config.experiment.episodes)
    )
    results_dir = Path(config.experiment.results_dir)
    write_published_reference(config, seeds, results_dir)
    brain_overrides: list[str] = []
    try:
        for profile in profiles:
            plans = select_plans(config, profile, args.run)
            if not plans:
                LOG.warning("profile %s: no runs selected", profile)
                continue
            if args.fresh:
                for plan in plans:
                    shutil.rmtree(results_dir / profile / plan.label, ignore_errors=True)
            SuiteRunner(config, config_path, profile, plans, seeds, results_dir, brain_overrides).run()
    except (
        RunnerError,
        ConfigError,
        IncompatibleBrainError,
        BrainUnavailableError,
        ProxyControlError,
    ) as exc:
        LOG.error("experiment aborted: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOG.warning("interrupted; finished episodes are kept and the run can be resumed")
        return 130
    LOG.info("done. Results in %s/ - create the report with: python -m kortyn.report", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
