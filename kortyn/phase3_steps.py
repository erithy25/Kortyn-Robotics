"""Phase 3, steps 1 and 2: the pre-registered decision rules on the rollout results.

Every rule here is fixed in ``reports/phase3.md`` (section 0.2/0.3) before the runs it judges:

* **Task selection (1.2):** among the pilot's tasks with 3-9 successes out of 10 (``sync25``), the two with
  the longest median length of their successful episodes; ties go to the higher success rate. With fewer
  than two eligible tasks the band widens to 2-10, then 1-10.
* **Reproduction R1 (1.1):** ``sync50`` against ``lerobot-eval`` on the same episodes. Reproduced if the
  exact McNemar test gives p >= 0.05 and the 95 % paired-bootstrap interval of the difference lies within
  +-15 percentage points; p >= 0.05 alone means "not distinguishable, precision insufficient"; p < 0.05 stops
  the main experiments.
* **JPEG q90 (1.4):** becomes the standard iff McNemar p >= 0.05 and the lower bound of the 95 % interval of
  (q90 - raw) is at least -10 percentage points.
* **Cold vs. warm start (1.3):** difference with interval and McNemar (descriptive).
* **Dose-response (2):** success per latency level with Wilson intervals, pooled and per task; McNemar against
  0 ms with Holm's correction over the other conditions.

    python -m kortyn.phase3_steps select-tasks --pilot-config kortyn/experiments/phase3/libero_pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kortyn.config import ConfigError, LiberoEnvConfig, load_config
from kortyn.libero_env import episode_slot
from kortyn.metrics import holm_adjust, mcnemar_exact, read_jsonl, wilson_interval

BOOTSTRAP_DRAWS = 10_000
R1_MARGIN = 0.15  # reproduction: interval of the difference within +-15 pp
JPEG_MARGIN = 0.10  # JPEG q90: lower bound of (q90 - raw) at least -10 pp
SELECTION_BANDS = ((3, 9), (2, 10), (1, 10))


class StepError(ValueError):
    """Missing or inconsistent run results."""


# --------------------------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Episode:
    seed: int
    success: bool
    steps: int
    starved_ticks: int
    time_to_success_s: float | None


def load_episodes(path: Path, fps: float = 20.0) -> dict[int, Episode]:
    """Final attempt per seed of a run (``episodes.jsonl`` or its directory)."""
    source = path / "episodes.jsonl" if path.is_dir() else path
    if not source.exists():
        raise StepError(f"run results {source} do not exist")
    final: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(source):
        seed = int(record["seed"])
        if seed not in final or int(record.get("attempt", 0)) >= int(final[seed].get("attempt", 0)):
            final[seed] = record
    return {
        seed: Episode(
            seed=seed,
            success=bool(r["success"]),
            steps=int(r["steps"]),
            starved_ticks=int(r.get("starved_ticks", 0)),
            time_to_success_s=int(r["steps"]) / fps if bool(r["success"]) else None,
        )
        for seed, r in final.items()
    }


def successes(episodes: Mapping[int, Episode]) -> dict[int, bool]:
    return {seed: e.success for seed, e in episodes.items()}


def summary_successes(paths: Sequence[str | Path]) -> dict[int, bool]:
    """Per-seed success of lerobot-eval runs (summary.json of kortyn.lerobot_reference), merged."""
    merged: dict[int, bool] = {}
    for path in paths:
        source = Path(path)
        if not source.exists():
            raise StepError(f"lerobot-eval summary {source} does not exist")
        for seed, value in json.loads(source.read_text(encoding="utf-8"))["success_by_seed"].items():
            if int(seed) in merged:
                raise StepError(f"seed {seed} appears in more than one summary")
            merged[int(seed)] = bool(value)
    return merged


# --------------------------------------------------------------------------------------------- paired
@dataclass(frozen=True)
class PairedComparison:
    """``b - a`` over the common seeds: success rates, difference with interval, exact McNemar."""

    pairs: int
    rate_a: float
    rate_b: float
    only_a: int
    only_b: int
    difference: float
    interval: tuple[float, float]
    p_value: float


def paired_comparison(
    a: Mapping[int, bool], b: Mapping[int, bool], confidence: float = 0.95, seed: int = 0
) -> PairedComparison:
    common = sorted(set(a) & set(b))
    if not common:
        raise StepError("no common seeds")
    x = np.array([float(a[s]) for s in common])
    y = np.array([float(b[s]) for s in common])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(common), size=(BOOTSTRAP_DRAWS, len(common)))
    diffs = (y[picks] - x[picks]).mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(diffs, [tail, 100.0 - tail])
    only_a = int(np.sum((x == 1.0) & (y == 0.0)))
    only_b = int(np.sum((x == 0.0) & (y == 1.0)))
    return PairedComparison(
        pairs=len(common),
        rate_a=float(x.mean()),
        rate_b=float(y.mean()),
        only_a=only_a,
        only_b=only_b,
        difference=float(y.mean() - x.mean()),
        interval=(float(low), float(high)),
        p_value=mcnemar_exact(only_a, only_b),
    )


def reproduction_verdict(comparison: PairedComparison, margin: float = R1_MARGIN) -> str:
    """R1: ``reproduziert``, ``nicht unterscheidbar (Präzision unzureichend)`` or ``abweichend``."""
    if comparison.p_value < 0.05:
        return "abweichend"
    low, high = comparison.interval
    if -margin <= low and high <= margin:
        return "reproduziert"
    return "nicht unterscheidbar (Präzision unzureichend)"


def jpeg_becomes_standard(raw_vs_q90: PairedComparison, margin: float = JPEG_MARGIN) -> bool:
    """1.4: q90 is the standard iff McNemar p >= 0.05 and the lower bound of (q90 - raw) >= -10 pp."""
    return raw_vs_q90.p_value >= 0.05 and raw_vs_q90.interval[0] >= -margin - 1e-12


# --------------------------------------------------------------------------------------------- tasks
@dataclass(frozen=True)
class TaskStats:
    task_id: int
    episodes: int
    successes: int
    median_success_steps: float | None


@dataclass(frozen=True)
class TaskSelection:
    band: tuple[int, int]
    selected: tuple[int, ...]
    tasks: tuple[TaskStats, ...]


def task_stats(episodes: Mapping[int, Episode], libero: LiberoEnvConfig) -> list[TaskStats]:
    by_task: dict[int, list[Episode]] = {task: [] for task in libero.task_ids}
    for seed, episode in episodes.items():
        by_task[episode_slot(libero, seed)[0]].append(episode)
    result = []
    for task, items in sorted(by_task.items()):
        lengths = [e.steps for e in items if e.success]
        result.append(
            TaskStats(
                task_id=task,
                episodes=len(items),
                successes=len(lengths),
                median_success_steps=float(statistics.median(lengths)) if lengths else None,
            )
        )
    return result


def select_tasks(stats: Sequence[TaskStats], count: int = 2) -> TaskSelection:
    """The pre-registered rule: eligible band, then longest median successful episode, then success rate."""
    for band in SELECTION_BANDS:
        eligible = [
            t for t in stats if band[0] <= t.successes <= band[1] and t.median_success_steps is not None
        ]
        if len(eligible) >= count:
            ranked = sorted(
                eligible, key=lambda t: (-(t.median_success_steps or 0.0), -t.successes, t.task_id)
            )
            return TaskSelection(
                band=band, selected=tuple(t.task_id for t in ranked[:count]), tasks=tuple(stats)
            )
    raise StepError(
        f"fewer than {count} tasks with at least one success; the selection rule cannot be applied"
    )


# --------------------------------------------------------------------------------------------- dose
@dataclass(frozen=True)
class Level:
    label: str
    latency_ms: float | None  # None: the jitter condition
    rate: float
    interval: tuple[float, float]
    per_task: dict[int, tuple[float, tuple[float, float]]]
    p_value: float | None  # McNemar against 0 ms (None for 0 ms itself)
    p_holm: float | None
    starved_share: float
    mean_steps: float
    median_time_to_success_s: float | None


def dose_response(
    runs: Mapping[str, tuple[float | None, Mapping[int, Episode]]],
    reference: str,
    libero: LiberoEnvConfig,
) -> list[Level]:
    """Success per condition (Wilson 95 %, pooled and per task) and McNemar against ``reference`` (Holm)."""
    if reference not in runs:
        raise StepError(f"reference condition {reference!r} missing")
    base = successes(runs[reference][1])
    others = [label for label in runs if label != reference]
    raw_p = [paired_comparison(base, successes(runs[label][1])).p_value for label in others]
    holm = dict(zip(others, holm_adjust(raw_p), strict=True)) if others else {}
    raw = dict(zip(others, raw_p, strict=True))
    levels = []
    for label, (latency, episodes) in runs.items():
        items = list(episodes.values())
        wins = sum(e.success for e in items)
        per_task: dict[int, tuple[float, tuple[float, float]]] = {}
        for task in libero.task_ids:
            group = [e for e in items if episode_slot(libero, e.seed)[0] == task]
            if group:
                k = sum(e.success for e in group)
                per_task[task] = (k / len(group), wilson_interval(k, len(group)))
        times = [e.time_to_success_s for e in items if e.time_to_success_s is not None]
        levels.append(
            Level(
                label=label,
                latency_ms=latency,
                rate=wins / len(items),
                interval=wilson_interval(wins, len(items)),
                per_task=per_task,
                p_value=raw.get(label),
                p_holm=holm.get(label),
                starved_share=sum(e.starved_ticks for e in items) / max(1, sum(e.steps for e in items)),
                mean_steps=float(np.mean([e.steps for e in items])),
                median_time_to_success_s=float(statistics.median(times)) if times else None,
            )
        )
    return levels


# --------------------------------------------------------------------------------------------- CLI
def _libero(config_path: str) -> tuple[LiberoEnvConfig, Path, float]:
    config = load_config(config_path)
    if config.env.libero is None:
        raise ConfigError(f"{config_path} is not a LIBERO experiment")
    return config.env.libero, Path(config.experiment.results_dir), config.env.fps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-3 decision rules of steps 1 and 2.")
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select-tasks", help="task selection (1.2) from the pilot's sync25 run")
    select.add_argument("--pilot-config", required=True)
    select.add_argument("--profile", default="smolvla_libero")
    select.add_argument("--run", default="sync25")
    select.add_argument("--out", default="results/phase3/step1/task_selection.json")
    compare = sub.add_parser("compare", help="paired comparison of two runs (R1, JPEG, cold vs warm)")
    compare.add_argument("--a", required=True, help="episodes of run a (directory or jsonl)")
    compare.add_argument("--b", default=None, help="episodes of run b (b - a is reported)")
    compare.add_argument(
        "--b-summary",
        action="append",
        default=None,
        help="instead of --b: lerobot-eval summary.json file(s) of kortyn.lerobot_reference (merged)",
    )
    compare.add_argument("--rule", choices=["r1", "jpeg", "none"], default="none")
    compare.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "select-tasks":
            libero, root, fps = _libero(args.pilot_config)
            episodes = load_episodes(root / args.profile / args.run, fps)
            selection = select_tasks(task_stats(episodes, libero))
            document: dict[str, Any] = asdict(selection)
        else:
            a = successes(load_episodes(Path(args.a)))
            if (args.b is None) == (args.b_summary is None):
                raise StepError("give either --b or --b-summary")
            b = (
                summary_successes(args.b_summary)
                if args.b_summary
                else successes(load_episodes(Path(args.b)))
            )
            comparison = paired_comparison(a, b)
            document = asdict(comparison)
            if args.rule == "r1":
                document["verdict"] = reproduction_verdict(comparison)
            elif args.rule == "jpeg":
                document["jpeg_standard"] = jpeg_becomes_standard(comparison)
    except (ConfigError, StepError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(document, indent=1)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
