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


def test_validation_compares_rollouts_with_the_stage1_prediction(tmp_path: Path) -> None:
    case = tmp_path / "validation" / "d_n2"
    case.mkdir(parents=True)
    meta = {"network": "buero_wlan", "gpu": "g", "batching": "b", "strategy": "d", "robots": 2}
    (case / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    episodes, requests = [], []
    for index, seed in enumerate((1000, 1050, 1001, 1051)):
        robot, robot_episode = index % 2, index // 2
        episodes.append(
            {
                "seed": seed,
                "robot": robot,
                "robot_episode": robot_episode,
                "counted": True,
                "success": index < 2,
            }
        )
        for request in range(3):
            requests.append(
                {
                    "robot": robot,
                    "robot_episode": robot_episode,
                    "t_obs_s": request * 1.25,
                    "t_resp_s": request * 1.25 + 0.22,
                }
            )
    episodes.append({"seed": 1002, "robot": 0, "robot_episode": 2, "counted": False, "success": True})
    (case / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes) + "\n", encoding="utf-8")
    (case / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in requests) + "\n", encoding="utf-8")
    stage1 = tmp_path / "scan.jsonl"
    record = {
        "key": {**meta, "replication": 0},
        "robots": [{"admitted": True, "histogram": {"22": 10}}, {"admitted": True, "histogram": {"42": 10}}],
    }
    stage1.write_text(json.dumps(record) + "\n", encoding="utf-8")
    curve = dc.DoseCurve(
        spec=dc.CurveSpec(), tasks={8: dc.TaskCurve(0.6, 500.0, 3.0), 0: dc.TaskCurve(0.4, 500.0, 3.0)}
    )
    rows = p4.validation_episodes(case, curve, stage1, LIBERO)
    assert len(rows) == 4  # the uncounted episode is left out
    first = rows[0]  # seed 1000 = task 8, robot 0: stage-1 latency bin 22 -> 225 ms -> 250 ms
    assert first.predicted == pytest.approx(float(curve.task_value(8, 225.0)))
    assert first.predicted_own == pytest.approx(float(curve.task_value(8, 220.0)))
    config = p4.AnalysisConfig(
        conditions={"mit_rtc": [p4.LevelSpec(0.0, "x")]},
        validation=p4.ValidationSpec(stage1=str(stage1), results_dir=str(tmp_path / "validation"), draws=500),
    )
    document = p4.validation_document(config, curve)
    result = document["cases"]["d_n2"]
    assert result["episodes"] == 4
    assert result["measured"] == 0.5
    assert result["interval"][0] <= result["difference"] <= result["interval"][1]
    with pytest.raises(StepError, match="no run"):
        p4.stage1_histograms(stage1, {**meta, "robots": 3, "replication": 0})
    with pytest.raises(ConfigError):
        p4.ValidationSpec(confidence=1.0)


ROOT = Path(__file__).resolve().parents[1]
needs_results = pytest.mark.skipif(
    not (ROOT / "results/phase3/validation/a_n2/episodes.jsonl").exists()
    or not (ROOT / "results/phase4/libero/smolvla_libero/fest_500ms/episodes.jsonl").exists(),
    reason="phase-3 validation or phase-4 runs not in this checkout",
)


@needs_results
def test_retest_predicts_every_phase3_validation_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)  # the analysis and fleet configs name repository paths
    config = p4.load_analysis_config("kortyn/experiments/phase4/analysis.yaml")
    flat = dc.DoseCurve(
        spec=dc.CurveSpec(), tasks={8: dc.TaskCurve(0.4, 1e5, 2.0), 0: dc.TaskCurve(0.2, 1e5, 2.0)}
    )
    predictions = p4.retest_predictions(config, flat)
    assert {p.case for p in predictions} == {"a_n2", "a_n4", "e_n2", "e_n4"}
    assert len(predictions) == 240
    for p in predictions:
        assert p.task in (8, 0)
        assert 0.0 <= p.phase3_own <= 1.0
        if p.case != "a_n4":  # inside the budget the flat curve predicts its task level
            assert p.own == pytest.approx(0.4 if p.task == 8 else 0.2, abs=0.01)
    document = p4.retest_document(config, predictions)
    informative = [p for p in predictions if p.case in config.retest.informative]
    assert document["episodes"] == len(informative) == 180
    expected = float(np.mean([p.measured - p.own for p in informative]))
    assert document["pooled"]["own"]["bias"] == pytest.approx(expected)
    low, high = document["pooled"]["own"]["interval"]
    assert low <= expected <= high
    assert document["verdict"] in ("verschwunden", "verkleinert", "nicht erklärt")


@needs_results
def test_sensitivity_pools_the_phase3_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    config = p4.load_analysis_config("kortyn/experiments/phase4/analysis.yaml")
    document = p4.sensitivity_document(config)
    trials = {(row["task"], row["effective_ms"]): row["trials"] for row in document["levels"]}
    assert trials[(8, 500.0)] == 100.0  # 50 seeds of task 8 in each phase
    assert trials[(8, 300.0)] == 50.0  # measured in phase 4 only
    for name in ("vorab", "gepoolt"):
        assert document[name]["lstar_ms"] % 50.0 == 0.0
        assert 0.0 <= document[name]["success_0ms"] <= 1.0
        assert document[name]["retest_verdict"] in ("verschwunden", "verkleinert", "nicht erklärt")
    with pytest.raises(StepError):
        p4.sensitivity_document(p4.AnalysisConfig(conditions=config.conditions))
