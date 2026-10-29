"""Reproducibility of LIBERO rollouts (phase 5): episode fingerprints, bit-exact probes, fixture positions.

    python -m kortyn.reproducibility fingerprints     # identical episodes between runs, per task
    python -m kortyn.reproducibility probe --name fresh --seeds 1000 [--threads 4] [--deterministic]
                                           [--without-scene-fix]
    python -m kortyn.reproducibility compare fresh warm        # first bit-level difference of two probes
    python -m kortyn.reproducibility stove --seeds 1000,1000,1000,1050 [--without-scene-fix]
    python -m kortyn.reproducibility gate-fixed   # the phase-5 gate with task 8 re-run under the fix

``probe`` and ``stove`` need the LIBERO environment (``UV_PROJECT_ENVIRONMENT=.venv-libero``). Results go to
``results/phase5/reproducibility/``.

**Fingerprints.** A run's episode of a seed ends with a success flag and a step count; a successful episode
ends at the tick of success. Two runs of the same seed and condition whose episodes agree in both are, with
high probability, the same trajectory. ``fingerprints`` compares runs of different phases per task.

**Probe.** Runs episodes of one run label in one process with Kortyn's own code path (simulation time,
``LocalBrain`` with the plan cache off) and records SHA-256 digests of every array of the MuJoCo model and of
the simulator state after each reset, and per request of every observation tensor and of the returned plan.
``--without-scene-fix`` restores LIBERO's accumulation of property samplers that
:func:`kortyn.libero_env._reset_property_samplers` undoes, to show the behaviour before the fix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "phase5" / "reproducibility"
CONFIG = "kortyn/experiments/phase5/libero_rtc_kurz.yaml"
PROFILE = "smolvla_libero_rtc"
RUN = "sync25"
LOG = logging.getLogger("kortyn.reproducibility")

# MuJoCo state fields compared after a reset (all of them are set or derived by the reset).
STATE_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "qacc",
    "mocap_pos",
    "mocap_quat",
    "xpos",
    "qfrc_applied",
    "xfrc_applied",
)


class ReproducibilityError(RuntimeError):
    """Missing or inconsistent inputs."""


# --------------------------------------------------------------------------------------------- fingerprints
@dataclass(frozen=True)
class Episode:
    success: bool
    steps: int


@dataclass(frozen=True)
class RunPair:
    """Two runs of the same condition on the same seeds (``a`` the earlier one)."""

    name: str
    a: str
    b: str
    tasks: tuple[tuple[str, int, int], ...]  # (task, first seed, last seed)


TASKS_8_0 = (("task 8", 1000, 1049), ("task 0", 1050, 1099))
TASK_8 = (("task 8", 1000, 1049),)
LIBERO = "results/phase{}/libero/{}/{}"
PAIRS = (
    RunPair(
        "synchronous reference, phase 3 vs phase 5",
        LIBERO.format(3, "smolvla_libero", "sync25"),
        LIBERO.format(5, "smolvla_libero_rtc", "sync25"),
        TASKS_8_0,
    ),
    RunPair(
        "asynchronous at 0 ms, phase 3 vs phase 4",
        LIBERO.format(3, "smolvla_libero", "fest_0ms"),
        LIBERO.format(4, "smolvla_libero", "fest_0ms"),
        TASKS_8_0,
    ),
    RunPair(
        "asynchronous at 0 ms, phase 4 vs phase 5",
        LIBERO.format(4, "smolvla_libero", "fest_0ms"),
        LIBERO.format(5, "smolvla_libero_rtc", "async_0ms"),
        TASKS_8_0,
    ),
    RunPair(
        "500 ms, phase-4 block A vs a fresh process per episode (same session)",
        LIBERO.format(4, "smolvla_libero", "fest_500ms"),
        "results/phase4/fresh/stream*/smolvla_libero/fest_500ms",
        TASK_8,
    ),
)


def load_episodes(pattern: str, root: Path = ROOT) -> dict[int, Episode]:
    """Final attempt per seed of one run directory (or several, as a glob pattern relative to ``root``)."""
    directories = sorted(root.glob(pattern)) if any(c in pattern for c in "*?[") else [root / pattern]
    episodes: dict[int, tuple[int, Episode]] = {}
    for directory in directories:
        path = directory / "episodes.jsonl"
        if not path.exists():
            raise ReproducibilityError(f"{path} does not exist")
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            seed, attempt = int(record["seed"]), int(record.get("attempt", 0))
            if seed not in episodes or attempt >= episodes[seed][0]:
                episodes[seed] = (attempt, Episode(bool(record["success"]), int(record["steps"])))
    if not episodes:
        raise ReproducibilityError(f"no episodes in {pattern}")
    return {seed: episode for seed, (_, episode) in episodes.items()}


def compare_episodes(
    a: Mapping[int, Episode], b: Mapping[int, Episode], first: int, last: int
) -> dict[str, int]:
    """Per seed range: successes in each run, identical episodes, and successes ending at the same tick."""
    seeds = [seed for seed in range(first, last + 1) if seed in a and seed in b]
    if len(seeds) != last - first + 1:
        raise ReproducibilityError(f"seeds {first}-{last} are not in both runs")
    both = [seed for seed in seeds if a[seed].success and b[seed].success]
    return {
        "episodes": len(seeds),
        "successes_a": sum(a[seed].success for seed in seeds),
        "successes_b": sum(b[seed].success for seed in seeds),
        "identical": sum(a[seed] == b[seed] for seed in seeds),
        "success_in_both": len(both),
        "same_success_tick": sum(a[seed].steps == b[seed].steps for seed in both),
    }


def fingerprints_document(pairs: Sequence[RunPair] = PAIRS, root: Path = ROOT) -> dict[str, Any]:
    rows = []
    for pair in pairs:
        a, b = load_episodes(pair.a, root), load_episodes(pair.b, root)
        rows.append(
            {
                "name": pair.name,
                "a": pair.a,
                "b": pair.b,
                "tasks": {task: compare_episodes(a, b, first, last) for task, first, last in pair.tasks},
            }
        )
    return {
        "description": "Same seed, same condition, two runs: identical episodes (success flag and step "
        "count) per task. Seeds 1000-1049 are LIBERO-10 task 8, 1050-1099 task 0.",
        "pairs": rows,
    }


# --------------------------------------------------------------------------------------------- digests
def digest(value: Any) -> str:
    """SHA-256 (16 hex digits) of an array's dtype, shape and bytes."""
    array = np.ascontiguousarray(np.asarray(value))
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()[:16]


def model_digests(model: Any) -> dict[str, str]:
    """Digest of every non-empty array attribute of a MuJoCo model."""
    digests = {}
    for name in sorted(dir(model)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:  # some MjModel attributes raise for this model: not an array
            continue
        if isinstance(value, np.ndarray) and value.size:
            digests[name] = digest(value)
    return digests


def state_digests(data: Any) -> dict[str, str]:
    return {name: digest(getattr(data, name)) for name in STATE_FIELDS if hasattr(data, name)}


def compare_records(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """First bit-level difference between two probe records of the same seed."""
    if a["seed"] != b["seed"]:
        raise ReproducibilityError(f"different seeds {a['seed']} and {b['seed']}")
    model = sorted(k for k in set(a["model"]) | set(b["model"]) if a["model"].get(k) != b["model"].get(k))
    state = sorted(k for k in set(a["state"]) | set(b["state"]) if a["state"].get(k) != b["state"].get(k))
    first: dict[str, Any] | None = None
    for index, (ra, rb) in enumerate(zip(a["requests"], b["requests"], strict=False)):
        observations = sorted(
            k for k in ra["observation"] if ra["observation"][k] != rb["observation"].get(k)
        )
        if observations or ra["plan"] != rb["plan"]:
            first = {
                "request": index,
                "obs_timestep": ra["obs_timestep"],
                "observations_differing": observations,
                "plan_identical": ra["plan"] == rb["plan"],
                "first_action_a": ra["first_action"],
                "first_action_b": rb["first_action"],
            }
            break
    return {
        "seed": a["seed"],
        "outcome_a": [a["success"], a["steps"]],
        "outcome_b": [b["success"], b["steps"]],
        "model_fields_differing": model,
        "model_fields": len(a["model"]),
        "state_fields_differing": state,
        "requests_compared": min(len(a["requests"]), len(b["requests"])),
        "first_difference": first,
        "identical": not model and not state and first is None and len(a["requests"]) == len(b["requests"]),
    }


# --------------------------------------------------------------------------------------------- probe (LIBERO)
class _TracedCompute:
    """The agent's compute (``LocalBrain``) with a callback per answered request; other attributes pass."""

    def __init__(self, inner: Any, on_result: Callable[[Any, Any], None]) -> None:
        self._inner = inner
        self._on_result = on_result

    def __call__(self, request: Any) -> Any:
        result = self._inner(request)
        self._on_result(request, result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _limit_threads(threads: int) -> None:
    from kortyn.run_virtual import _limit_threads as limit

    limit(threads)


def probe(seeds: Sequence[int], threads: int, deterministic: bool, scene_fix: bool) -> list[dict[str, Any]]:
    """Run the episodes of ``seeds`` (in this order, one process) and record their digests."""
    _limit_threads(threads)
    import torch

    from kortyn import libero_env
    from kortyn.agent import EpisodeContext
    from kortyn.config import load_config
    from kortyn.run_virtual import _load_runner, select_virtual_plans, timing_samplers
    from kortyn.sim_env import make_env
    from kortyn.virtual_agent import LocalBrain, VirtualTimeAgent

    if deterministic:
        torch.use_deterministic_algorithms(True)
    if not scene_fix:
        libero_env._reset_property_samplers = lambda task_env: None  # LIBERO's behaviour without the fix
    config = load_config(ROOT / CONFIG)
    plan = select_virtual_plans(config, PROFILE, [RUN])[0]
    brain = LocalBrain(_load_runner(config, PROFILE, threads, "cpu"), cache_entries=0)
    env = make_env(config.env)
    if not isinstance(env, libero_env.LiberoEnv):
        raise ReproducibilityError(f"{CONFIG} does not describe a LIBERO environment")
    agent = VirtualTimeAgent(
        plan.agent_config(config.agent),
        env,
        brain.spec,
        brain,
        plan.timing,
        samplers=timing_samplers(config, plan.timing, PROFILE),
        replay=None,
        timing_name=plan.timing_name,
    )
    current: dict[str, Any] = {}

    def record_request(request: Any, result: Any) -> None:
        actions = None if result.actions is None else np.asarray(result.actions, dtype=np.float32)
        current["requests"].append(
            {
                "obs_timestep": int(request.obs_timestep),
                "observation": {
                    name: hashlib.sha256(request.observation[name].data).hexdigest()[:16]
                    for name in sorted(request.observation)
                },
                "plan": None if actions is None else digest(actions),
                "first_action": None if actions is None else [float(x) for x in actions[0]],
            }
        )

    reset = env.reset

    def traced_reset(seed: int) -> Any:
        observation = reset(seed)
        sim = env._sim_env.sim
        current["model"] = model_digests(getattr(sim.model, "_model", sim.model))
        current["state"] = state_digests(getattr(sim.data, "_data", sim.data))
        current["task_id"], current["init_state"] = env.task_id, env.init_state_id
        return observation

    agent.compute = _TracedCompute(brain, record_request)
    env.reset = traced_reset  # type: ignore[method-assign]
    records = []
    for seed in seeds:
        current.clear()
        current.update({"seed": seed, "requests": []})
        context = EpisodeContext(
            run=plan.label,
            profile=PROFILE,
            scenario=plan.scenario.name,
            mode=plan.scenario.mode,
            seed=seed,
            strategy=plan.strategy_name,
        )
        episode = agent.run_episode(context, plan.strategy, network=plan.network).episode.to_json()
        current.update({"success": bool(episode["success"]), "steps": int(episode["steps"])})
        records.append(dict(current))
        LOG.info("seed %d: success %s after %d steps", seed, episode["success"], episode["steps"])
    return records


def stove_positions(seeds: Sequence[int], scene_fix: bool) -> list[dict[str, Any]]:
    """Body positions after resets of ``seeds`` (one process); lists bodies that moved between resets."""
    from kortyn import libero_env
    from kortyn.config import load_config
    from kortyn.sim_env import make_env

    if not scene_fix:
        libero_env._reset_property_samplers = lambda task_env: None
    env = make_env(load_config(ROOT / CONFIG).env)
    if not isinstance(env, libero_env.LiberoEnv):
        raise ReproducibilityError(f"{CONFIG} does not describe a LIBERO environment")
    first: dict[int, np.ndarray[Any, np.dtype[np.float64]]] = {}
    rows = []
    for index, seed in enumerate(seeds):
        env.reset(seed)
        sim = env._sim_env.sim
        model = getattr(sim.model, "_model", sim.model)
        positions = np.array(model.body_pos)
        moved = []
        if seed in first:
            for body in np.where(np.any(positions != first[seed], axis=1))[0]:
                moved.append(
                    {
                        "body": sim.model.body_id2name(int(body)),
                        "first": [float(x) for x in first[seed][body]],
                        "now": [float(x) for x in positions[body]],
                    }
                )
        else:
            first[seed] = positions
        rows.append(
            {
                "reset": index,
                "seed": seed,
                "task_id": env.task_id,
                "body_pos": digest(positions),
                "moved": moved,
            }
        )
    return rows


# ---------------------------------------------------------------------------------------- gate, fixed scenes
GATE_RUNS = ("sync25", "async_0ms", "rtc_kurz_0ms")
ORIGINAL = "results/phase5/libero/smolvla_libero_rtc"
FIXED = "results/phase5/libero_fixed/smolvla_libero_rtc"
ANALYSIS = "kortyn/experiments/phase5/analysis.yaml"
AFFECTED_TASK = 8


def merge_fixed(
    original: Mapping[str, Mapping[int, bool]],
    fixed: Mapping[str, Mapping[int, bool]],
    affected: Callable[[int], bool],
) -> dict[str, dict[int, bool]]:
    """Outcomes per run: the re-run (with the scene fix) on affected seeds, the original elsewhere."""
    merged: dict[str, dict[int, bool]] = {}
    for run, outcomes in original.items():
        missing = sorted(seed for seed in outcomes if affected(seed) and seed not in fixed.get(run, {}))
        if missing:
            raise ReproducibilityError(
                f"{run}: the re-run lacks {len(missing)} affected seeds, e.g. {missing[:3]}"
            )
        merged[run] = {
            seed: (fixed[run][seed] if affected(seed) else value) for seed, value in outcomes.items()
        }
    return merged


def discordance(reference: Mapping[int, bool], test: Mapping[int, bool], seeds: Sequence[int]) -> float:
    """Share of seeds on which two conditions disagree (the discordant pairs of McNemar's test)."""
    if not seeds:
        raise ReproducibilityError("no seeds")
    return sum(reference[seed] != test[seed] for seed in seeds) / len(seeds)


def gate_with_fixed_scenes(root: Path = ROOT) -> dict[str, Any]:
    """The pre-registered gate code on block G, with task 8 replaced by its re-run under the scene fix."""
    from kortyn.libero_env import episode_slot
    from kortyn.phase3_steps import load_episodes as load_runs
    from kortyn.phase3_steps import successes
    from kortyn.phase4_steps import libero_layout
    from kortyn.phase5_steps import gate_from_outcomes, load_phase5_config

    config = load_phase5_config(str(root / ANALYSIS))
    libero = libero_layout(config.experiment)

    def affected(seed: int) -> bool:
        return episode_slot(libero, seed)[0] == AFFECTED_TASK

    original = {run: successes(load_runs(root / ORIGINAL / run)) for run in GATE_RUNS}
    fixed = {run: successes(load_runs(root / FIXED / run)) for run in GATE_RUNS}
    merged = merge_fixed(original, fixed, affected)
    sources = {run: f"{FIXED}/{run} (task {AFFECTED_TASK}) + {ORIGINAL}/{run} (task 0)" for run in GATE_RUNS}
    document = gate_from_outcomes(config, merged, sources)
    seeds = sorted(seed for seed in original["sync25"] if affected(seed))
    document["pairing_task8"] = {
        label: {
            "seeds": len(seeds),
            "discordant_rtc_vs_sync25": discordance(runs["sync25"], runs["rtc_kurz_0ms"], seeds),
            "discordant_async_vs_sync25": discordance(runs["sync25"], runs["async_0ms"], seeds),
        }
        for label, runs in (("original", original), ("fixed", fixed))
    }
    document["note"] = (
        "Not pre-registered. Task 8 of block G re-run with the scene fix (kortyn.libero_env."
        "_reset_property_samplers); task 0 reused, it is unaffected (identical episodes across runs)."
    )
    return document


# --------------------------------------------------------------------------------------------- CLI
def _write(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    LOG.info("written %s", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def _seeds(text: str) -> list[int]:
    try:
        seeds = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ReproducibilityError(f"seeds must be integers separated by commas: {text!r}") from exc
    if not seeds:
        raise ReproducibilityError("no seeds given")
    return seeds


def _probe_path(name: str) -> Path:
    return OUTPUT / "probes" / f"{name}.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducibility of LIBERO rollouts (Kortyn phase 5).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fingerprints", help="identical episodes between runs, per task")
    run = sub.add_parser("probe", help="run episodes and record bit-exact digests (LIBERO environment)")
    run.add_argument("--name", required=True, help="probe name (file results/phase5/reproducibility/probes/)")
    run.add_argument("--seeds", required=True, help="comma-separated, run in this order in one process")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms(True)")
    run.add_argument("--without-scene-fix", action="store_true", help="LIBERO's sampler accumulation")
    compare = sub.add_parser("compare", help="first bit-level difference of two probes, per common seed")
    compare.add_argument("a")
    compare.add_argument("b")
    stove = sub.add_parser("stove", help="body positions after resets (LIBERO environment)")
    stove.add_argument("--seeds", required=True)
    stove.add_argument("--without-scene-fix", action="store_true")
    stove.add_argument("--name", default="stove")
    sub.add_parser("gate-fixed", help="the phase-5 gate with task 8 re-run under the scene fix")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "fingerprints":
            _write(OUTPUT / "fingerprints.json", fingerprints_document())
        elif args.command == "probe":
            records = probe(_seeds(args.seeds), args.threads, args.deterministic, not args.without_scene_fix)
            settings = {
                "seeds": _seeds(args.seeds),
                "threads": args.threads,
                "deterministic": args.deterministic,
                "scene_fix": not args.without_scene_fix,
                "config": CONFIG,
                "profile": PROFILE,
                "run": RUN,
            }
            _write(_probe_path(args.name), {"settings": settings, "episodes": records})
        elif args.command == "compare":
            documents = [
                json.loads(_probe_path(name).read_text(encoding="utf-8")) for name in (args.a, args.b)
            ]
            by_seed = [{record["seed"]: record for record in d["episodes"]} for d in documents]
            common = [seed for seed in by_seed[0] if seed in by_seed[1]]
            if not common:
                raise ReproducibilityError(f"probes {args.a} and {args.b} have no seed in common")
            result = {
                "a": {"name": args.a, **documents[0]["settings"]},
                "b": {"name": args.b, **documents[1]["settings"]},
                "seeds": [compare_records(by_seed[0][seed], by_seed[1][seed]) for seed in common],
            }
            _write(OUTPUT / "comparisons" / f"{args.a}__{args.b}.json", result)
        elif args.command == "gate-fixed":
            _write(OUTPUT / "gate_fixed_scenes.json", gate_with_fixed_scenes())
        else:
            rows = stove_positions(_seeds(args.seeds), not args.without_scene_fix)
            document = {"seeds": _seeds(args.seeds), "scene_fix": not args.without_scene_fix, "resets": rows}
            _write(OUTPUT / f"{args.name}.json", document)
    except (ReproducibilityError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.stdout.flush()
    code = main()
    sys.stdout.flush()
    # LIBERO's EGL contexts fail noisily in interpreter shutdown; nothing is left to clean up here.
    os._exit(code)
