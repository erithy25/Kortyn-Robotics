"""Phase 3, steps 1-2: the pre-registered decision rules (task selection, R1, JPEG, dose-response)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kortyn.config import LiberoEnvConfig
from kortyn.phase3_steps import (
    Episode,
    PairedComparison,
    StepError,
    TaskStats,
    dose_response,
    jpeg_becomes_standard,
    load_episodes,
    main,
    paired_comparison,
    reproduction_verdict,
    select_tasks,
    successes,
    summary_successes,
    task_stats,
)


def episode(seed: int, success: bool, steps: int = 520, starved: int = 0) -> Episode:
    return Episode(seed, success, steps, starved, steps / 20.0 if success else None)


def comparison(p_value: float, interval: tuple[float, float]) -> PairedComparison:
    return PairedComparison(100, 0.5, 0.5, 5, 5, 0.0, interval, p_value)


# --------------------------------------------------------------------------------------------- loading
def test_load_episodes_keeps_the_final_attempt(tmp_path: Path) -> None:
    rows = [
        {"seed": 1000, "attempt": 0, "success": True, "steps": 200, "starved_ticks": 3},
        {"seed": 1000, "attempt": 1, "success": False, "steps": 520, "starved_ticks": 9},
        {"seed": 1001, "attempt": 0, "success": True, "steps": 300},
    ]
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    episodes = load_episodes(tmp_path)
    assert episodes[1000] == Episode(1000, False, 520, 9, None)
    assert episodes[1001].time_to_success_s == pytest.approx(15.0)
    with pytest.raises(StepError):
        load_episodes(tmp_path / "missing")


def test_summary_successes_merges_task_parts(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps({"success_by_seed": {"1000": True, "1001": False}}))
    (tmp_path / "b.json").write_text(json.dumps({"success_by_seed": {"1050": True}}))
    assert summary_successes([tmp_path / "a.json", tmp_path / "b.json"]) == {
        1000: True,
        1001: False,
        1050: True,
    }
    with pytest.raises(StepError, match="more than one"):
        summary_successes([tmp_path / "a.json", tmp_path / "a.json"])


# --------------------------------------------------------------------------------------------- paired
def test_paired_comparison_counts_discordant_pairs() -> None:
    a = {s: s % 2 == 0 for s in range(100)}  # 50 %
    b = {s: s % 4 != 3 for s in range(100)}  # 75 %: gains on 25 seeds, loses none
    result = paired_comparison(a, b)
    assert result.pairs == 100
    assert (result.only_a, result.only_b) == (0, 25)
    assert result.difference == pytest.approx(0.25)
    assert result.interval[0] < 0.25 < result.interval[1]
    assert result.p_value < 0.001
    with pytest.raises(StepError):
        paired_comparison({1: True}, {2: True})


def test_reproduction_verdicts() -> None:
    assert reproduction_verdict(comparison(0.4, (-0.10, 0.12))) == "reproduziert"
    assert reproduction_verdict(comparison(0.4, (-0.18, 0.05))).startswith("nicht unterscheidbar")
    assert reproduction_verdict(comparison(0.01, (-0.05, 0.05))) == "abweichend"
    assert reproduction_verdict(comparison(0.05, (-0.15, 0.15))) == "reproduziert"  # boundaries included


def test_jpeg_rule() -> None:
    assert jpeg_becomes_standard(comparison(0.5, (-0.08, 0.04)))
    assert jpeg_becomes_standard(comparison(0.05, (-0.10, 0.04)))
    assert not jpeg_becomes_standard(comparison(0.5, (-0.11, 0.04)))  # possible loss too large
    assert not jpeg_becomes_standard(comparison(0.04, (-0.05, 0.04)))  # significant change


# --------------------------------------------------------------------------------------------- tasks
def test_task_stats_and_selection() -> None:
    libero = LiberoEnvConfig(task_ids=[0, 1, 2, 3], episodes_per_task=10)
    results: dict[int, Episode] = {}
    plan = {0: (2, 400), 1: (5, 300), 2: (9, 450), 3: (10, 500)}  # task: (successes, length of successes)
    for task, (wins, length) in plan.items():
        for within in range(10):
            seed = 1000 + task * 10 + within
            results[seed] = episode(seed, within < wins, length if within < wins else 520)
    stats = task_stats(results, libero)
    assert [(t.task_id, t.successes, t.median_success_steps) for t in stats] == [
        (0, 2, 400.0),
        (1, 5, 300.0),
        (2, 9, 450.0),
        (3, 10, 500.0),
    ]
    selection = select_tasks(stats)
    assert selection.band == (3, 9)
    assert selection.selected == (2, 1)  # eligible: tasks 1 and 2; longest successful episodes first


def test_selection_ties_and_band_widening() -> None:
    tied = [TaskStats(0, 10, 4, 300.0), TaskStats(1, 10, 7, 300.0), TaskStats(2, 10, 5, 280.0)]
    assert select_tasks(tied).selected == (1, 0)  # equal length: higher success rate first
    narrow = [TaskStats(0, 10, 10, 300.0), TaskStats(1, 10, 2, 400.0), TaskStats(2, 10, 0, None)]
    selection = select_tasks(narrow)
    assert selection.band == (2, 10)
    assert selection.selected == (1, 0)
    with pytest.raises(StepError):
        select_tasks([TaskStats(0, 10, 0, None), TaskStats(1, 10, 1, 200.0)])


# --------------------------------------------------------------------------------------------- dose
def test_dose_response_levels_and_holm() -> None:
    libero = LiberoEnvConfig(task_ids=[3, 7], episodes_per_task=20)
    seeds = range(1000, 1040)
    zero = {s: episode(s, True, 300) for s in seeds}
    slow = {s: episode(s, s % 2 == 0, 400, starved=40) for s in seeds}  # loses 20 of 40
    same = {s: episode(s, True, 310) for s in seeds}
    runs = {"fest_0ms": (0.0, zero), "fest_500ms": (500.0, same), "fest_2000ms": (2000.0, slow)}
    levels = {lv.label: lv for lv in dose_response(runs, "fest_0ms", libero)}
    assert levels["fest_0ms"].rate == 1.0
    assert levels["fest_0ms"].p_value is None
    assert levels["fest_2000ms"].rate == 0.5
    assert levels["fest_2000ms"].per_task[3][0] == 0.5
    assert levels["fest_2000ms"].p_value is not None
    assert levels["fest_2000ms"].p_value < 1e-5
    assert levels["fest_2000ms"].p_holm == pytest.approx(min(1.0, 2 * levels["fest_2000ms"].p_value))
    assert levels["fest_500ms"].p_holm == 1.0
    assert levels["fest_2000ms"].starved_share == pytest.approx(0.1)
    assert levels["fest_2000ms"].median_time_to_success_s == pytest.approx(20.0)
    with pytest.raises(StepError):
        dose_response(runs, "missing", libero)


def test_cli_compare_with_lerobot_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [{"seed": 1000 + i, "attempt": 0, "success": i < 6, "steps": 300} for i in range(10)]
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    summary = {"success_by_seed": {str(1000 + i): i < 5 for i in range(10)}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "r1.json"
    code = main(
        [
            "compare",
            "--a",
            str(tmp_path),
            "--b-summary",
            str(tmp_path / "summary.json"),
            "--rule",
            "r1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["difference"] == pytest.approx(-0.1)
    assert written["verdict"] in {
        "reproduziert",
        "nicht unterscheidbar (Präzision unzureichend)",
        "abweichend",
    }
    assert main(["compare", "--a", str(tmp_path)]) == 2  # neither --b nor --b-summary
    assert successes(load_episodes(tmp_path))[1000]
    capsys.readouterr()
