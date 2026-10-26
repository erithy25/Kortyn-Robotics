"""Phase-4 evaluation steps (kortyn.phase4_steps) on synthetic runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kortyn import dose_curve as dc
from kortyn import phase4_steps as p4
from kortyn.config import ConfigError, LiberoEnvConfig
from kortyn.phase3_steps import StepError

LIBERO = LiberoEnvConfig(task_ids=[8, 0], episodes_per_task=50)
SEEDS = range(1000, 1100)


def write_run(directory: Path, success: dict[int, bool]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "episodes.jsonl").open("w", encoding="utf-8") as out:
        for seed, value in success.items():
            out.write(json.dumps({"seed": seed, "attempt": 0, "success": value, "steps": 300}) + "\n")
    return directory


def synthetic_condition(
    tmp_path: Path, name: str, curve: dict[int, dc.TaskCurve], rng: np.random.Generator
) -> list[p4.LevelSpec]:
    levels = []
    for latency in (0.0, 100.0, 200.0, 300.0, 500.0, 750.0, 1000.0, 1250.0):
        x = float(dc.effective_latency_ms(latency))
        outcome = {}
        for seed in SEEDS:
            task = 8 if seed < 1050 else 0
            outcome[seed] = bool(rng.random() < float(curve[task](x)))
        levels.append(p4.LevelSpec(latency, str(write_run(tmp_path / name / f"{latency:.0f}", outcome))))
    return levels


def test_load_condition_orders_levels_and_maps_tasks(tmp_path: Path) -> None:
    a = write_run(tmp_path / "a", {s: s % 2 == 0 for s in SEEDS})
    b = write_run(tmp_path / "b", {s: s % 3 == 0 for s in SEEDS})
    dose = p4.load_condition([p4.LevelSpec(500.0, str(b)), p4.LevelSpec(0.0, str(a))], LIBERO)
    assert dose.levels_ms.tolist() == [0.0, 500.0]
    assert dose.groups[0] == 8
    assert dose.groups[-1] == 0
    assert dose.outcomes[0, 0] == 1.0
    assert dose.outcomes[1, 1] == 0.0
    short = write_run(tmp_path / "c", {s: True for s in range(1000, 1050)})
    with pytest.raises(StepError, match="missing"):
        p4.load_condition([p4.LevelSpec(0.0, str(a)), p4.LevelSpec(100.0, str(short))], LIBERO)
    with pytest.raises(StepError):
        p4.load_condition([p4.LevelSpec(0.0, str(a))], LIBERO)


def test_curves_effect_and_checks_on_synthetic_conditions(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    slow = {8: dc.TaskCurve(0.55, 400.0, 3.0), 0: dc.TaskCurve(0.45, 350.0, 4.0)}
    fast = {8: dc.TaskCurve(0.55, 900.0, 3.0), 0: dc.TaskCurve(0.45, 800.0, 4.0)}
    without = synthetic_condition(tmp_path, "ohne", slow, rng)
    with_rtc = synthetic_condition(tmp_path, "mit", fast, rng)
    sync = write_run(tmp_path / "sync25", {s: bool(rng.random() < 0.5) for s in SEEDS})
    config = p4.AnalysisConfig(
        bootstrap=200,
        conditions={"ohne_rtc": without, "mit_rtc": with_rtc},
        reference_sync=str(sync),
    )
    doses = {name: p4.load_condition(levels, LIBERO) for name, levels in config.conditions.items()}
    fits = p4.fit_conditions(doses, config.spec, config.bootstrap, config.seed)
    summary = p4.condition_summary(doses["ohne_rtc"], fits["ohne_rtc"], 0.03)
    assert {"levels", "fit", "lstar", "success_at_lstar", "curve", "smooth"} <= set(summary)
    assert len(summary["levels"]) == 8
    assert summary["levels"][0]["effective_ms"] == 50.0
    effect = p4.rtc_effect(fits["ohne_rtc"], fits["mit_rtc"], 0.03)
    assert effect["h1"]["gain"] > 0.05
    assert effect["h1"]["verdict"] == "bestätigt"
    assert effect["h2"]["difference_ms"] > 0
    check = p4.r_rtc_check(Path(with_rtc[0].run), sync)
    assert check["pairs"] == 100
    assert check["verdict"] in ("erfüllt", "nicht erfüllt", "nicht entsprechend (besser)")
    # conditions on other seeds cannot be paired
    other = p4.load_condition(
        [
            p4.LevelSpec(0.0, str(write_run(tmp_path / "x0", {s: True for s in range(2000, 2100)}))),
            p4.LevelSpec(500.0, str(write_run(tmp_path / "x1", {s: False for s in range(2000, 2100)}))),
        ],
        LiberoEnvConfig(task_ids=[8, 0], episodes_per_task=50, seed_base=2000),
    )
    with pytest.raises(StepError, match="same seeds"):
        p4.fit_conditions({"a": doses["ohne_rtc"], "b": other}, config.spec, 100, 1)


def test_cluster_interval_and_bias_verdicts() -> None:
    mean, (low, high) = p4.cluster_mean_interval([1.0, 0.0, 1.0, 0.0], [1, 1, 2, 2], 2000, 3)
    assert mean == 0.5
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)
    _, (low, high) = p4.cluster_mean_interval([1.0, 1.0, 0.0, 0.0], [1, 1, 2, 2], 2000, 3)
    assert low == 0.0
    assert high == 1.0
    with pytest.raises(StepError):
        p4.cluster_mean_interval([], [], 10, 1)
    assert p4.bias_verdict(-0.02) == "verschwunden"
    assert p4.bias_verdict(-0.035) == "verkleinert"
    assert p4.bias_verdict(-0.06) == "nicht erklärt"


def test_analysis_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        p4.AnalysisConfig(conditions={})
    with pytest.raises(ConfigError):
        p4.AnalysisConfig(bootstrap=10, conditions={"a": [p4.LevelSpec(0.0, "x")]})
    with pytest.raises(ConfigError):
        p4.LevelSpec(-1.0, "x")
    path = tmp_path / "analysis.yaml"
    path.write_text("bootstrap: 300\nconditions:\n  a:\n    - {latency_ms: 0, run: r}\n", encoding="utf-8")
    loaded = p4.load_analysis_config(path)
    assert loaded.bootstrap == 300
    assert loaded.conditions["a"][0].run == "r"
