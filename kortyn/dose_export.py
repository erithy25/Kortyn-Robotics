"""Export a measured dose-response condition as counts for the capacity tool (``kortyn-dose-response-v1``).

Reads the rollouts of one condition of an analysis config (as :mod:`kortyn.phase4_steps` does: final attempt
per seed, every level on the same seeds) and writes successes and episodes per task and latency level, with
the run directories as provenance:

    python -m kortyn.dose_export --analysis kortyn/experiments/phase4/analysis.yaml --condition ohne_rtc \\
        --out examples/capacity/smolvla_libero_long.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kortyn.capacity import DOSE_FORMAT
from kortyn.config import ConfigError
from kortyn.fleet_predict import PredictionError
from kortyn.libero_env import LIBERO_10_TASKS
from kortyn.phase3_steps import StepError
from kortyn.phase4_steps import libero_layout, load_analysis_config, load_condition


def export_condition(analysis: str, condition: str, description: str = "") -> dict[str, Any]:
    config = load_analysis_config(analysis)
    if condition not in config.conditions:
        raise ConfigError(f"unknown condition {condition!r} (have {sorted(config.conditions)})")
    levels = sorted(config.conditions[condition], key=lambda level: level.latency_ms)
    dose = load_condition(levels, libero_layout(config.experiment))
    groups = np.asarray(dose.groups)
    tasks = []
    for task in dose.group_ids:
        mask = groups == task
        tasks.append(
            {
                "name": f"libero_10 task {task}: {LIBERO_10_TASKS[task]}",
                "levels": [
                    {
                        "latency_ms": float(latency),
                        "successes": int(dose.outcomes[index, mask].sum()),
                        "episodes": int(mask.sum()),
                    }
                    for index, latency in enumerate(dose.levels_ms)
                ],
            }
        )
    spec = config.spec
    return {
        "format": DOSE_FORMAT,
        "description": description
        or f"condition {condition!r} of {analysis}: success per task at fixed end-to-end latencies",
        "source": {
            "analysis": analysis,
            "condition": condition,
            "runs": [level.run for level in levels],
            "seeds": [int(dose.seeds[0]), int(dose.seeds[-1])],
        },
        "tick_ms": spec.tick_ms,
        "fit_max_ms": spec.fit_max_ms,
        "zero_from_ms": spec.zero_from_ms,
        "gof_alpha": spec.gof_alpha,
        "tasks": tasks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a dose-response condition for the capacity tool.")
    parser.add_argument(
        "--analysis", required=True, help="analysis config (e.g. kortyn/experiments/phase4/...)"
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        document = export_condition(args.analysis, args.condition, args.description)
    except (ConfigError, StepError, PredictionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
