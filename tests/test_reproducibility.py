"""Reproducibility checks (kortyn.reproducibility): fingerprints, digests, bit-level comparison of probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kortyn import reproducibility as rep

ROOT = Path(__file__).resolve().parent.parent


def _episodes(path: Path, rows: list[tuple[int, bool, int]], attempt: int = 0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        for seed, success, steps in rows:
            handle.write(
                json.dumps({"seed": seed, "attempt": attempt, "success": success, "steps": steps}) + "\n"
            )


# --------------------------------------------------------------------------------------------- fingerprints
def test_load_episodes_keeps_the_final_attempt_and_merges_a_glob(tmp_path: Path) -> None:
    _episodes(tmp_path / "s0" / "run", [(1, True, 300)])
    _episodes(tmp_path / "s0" / "run", [(1, False, 520)], attempt=1)
    _episodes(tmp_path / "s1" / "run", [(2, True, 410)])
    episodes = rep.load_episodes("s*/run", tmp_path)
    assert episodes == {1: rep.Episode(False, 520), 2: rep.Episode(True, 410)}
    with pytest.raises(rep.ReproducibilityError):
        rep.load_episodes("missing", tmp_path)


def test_compare_episodes_counts_identical_episodes_and_success_ticks() -> None:
    a = {1: rep.Episode(True, 300), 2: rep.Episode(False, 520), 3: rep.Episode(True, 400)}
    b = {1: rep.Episode(True, 300), 2: rep.Episode(False, 520), 3: rep.Episode(True, 410)}
    counts = rep.compare_episodes(a, b, 1, 3)
    assert counts == {
        "episodes": 3,
        "successes_a": 2,
        "successes_b": 2,
        "identical": 2,
        "success_in_both": 2,
        "same_success_tick": 1,
    }
    with pytest.raises(rep.ReproducibilityError):
        rep.compare_episodes(a, b, 1, 4)


def test_fingerprints_of_the_recorded_runs() -> None:
    """Task 0 reproduced episode for episode across phases; task 8 did not (LIBERO's sampler accumulation)."""
    document = rep.fingerprints_document()
    reference = document["pairs"][0]
    assert reference["tasks"]["task 0"]["identical"] == 50
    assert reference["tasks"]["task 0"]["successes_a"] == reference["tasks"]["task 0"]["successes_b"] == 25
    assert reference["tasks"]["task 8"]["identical"] < 25
    for pair in document["pairs"]:
        if "task 0" in pair["tasks"]:
            task0 = pair["tasks"]["task 0"]
            assert task0["same_success_tick"] == task0["success_in_both"]


# --------------------------------------------------------------------------------------------- digests
def test_digest_depends_on_values_dtype_and_shape() -> None:
    values = np.arange(6, dtype=np.float32)
    assert rep.digest(values) == rep.digest(values.copy())
    assert rep.digest(values) != rep.digest(values.astype(np.float64))
    assert rep.digest(values) != rep.digest(values.reshape(2, 3))
    changed = values.copy()
    changed[3] = np.nextafter(changed[3], np.float32(10))
    assert rep.digest(values) != rep.digest(changed)


def test_model_and_state_digests_take_array_attributes() -> None:
    class Model:
        body_pos = np.zeros((2, 3))
        geom_size = np.ones(4)
        empty = np.zeros(0)
        name = "scene"
        _private = np.ones(1)

        @property
        def broken(self) -> Any:
            raise RuntimeError("not available for this model")

    assert sorted(rep.model_digests(Model())) == ["body_pos", "geom_size"]

    class Data:
        time = 0.0
        qpos = np.zeros(3)

    assert sorted(rep.state_digests(Data())) == ["qpos", "time"]


def _record(seed: int = 1000, stove: str = "a", image: str = "x", plan: str = "p") -> dict[str, Any]:
    return {
        "seed": seed,
        "success": True,
        "steps": 400,
        "model": {"body_pos": stove, "geom_size": "g"},
        "state": {"qpos": "q", "xpos": stove},
        "requests": [
            {
                "obs_timestep": 0,
                "observation": {"camera": image, "state": "s"},
                "plan": plan,
                "first_action": [0.1],
            },
            {
                "obs_timestep": 25,
                "observation": {"camera": "y", "state": "t"},
                "plan": "q",
                "first_action": [0.2],
            },
        ],
    }


def test_compare_records_finds_the_first_difference() -> None:
    same = rep.compare_records(_record(), _record())
    assert same["identical"] is True
    assert same["first_difference"] is None
    moved = rep.compare_records(_record(), _record(stove="b", image="z", plan="r"))
    assert moved["identical"] is False
    assert moved["model_fields_differing"] == ["body_pos"]
    assert moved["state_fields_differing"] == ["xpos"]
    first = moved["first_difference"]
    assert first["request"] == 0
    assert first["observations_differing"] == ["camera"]
    assert first["plan_identical"] is False
    numerics = rep.compare_records(_record(), _record(plan="r"))  # same inputs, different plan (threads)
    assert numerics["first_difference"]["observations_differing"] == []
    assert numerics["first_difference"]["plan_identical"] is False
    with pytest.raises(rep.ReproducibilityError):
        rep.compare_records(_record(1000), _record(1001))


# --------------------------------------------------------------------------------------------- CLI
def test_seeds_are_parsed_in_order() -> None:
    assert rep._seeds("1001, 1000,1050") == [1001, 1000, 1050]
    for text in ("", "a,b"):
        with pytest.raises(rep.ReproducibilityError):
            rep._seeds(text)


def test_main_compares_two_probes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rep, "OUTPUT", tmp_path)
    settings = {"seeds": [1000], "threads": 1, "deterministic": False, "scene_fix": True}
    for name, record in (("fresh", _record()), ("warm", _record(stove="b", image="z"))):
        path = tmp_path / "probes" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"settings": settings, "episodes": [record]}), encoding="utf-8")
    assert rep.main(["--log-level", "WARNING", "compare", "fresh", "warm"]) == 0
    result = json.loads((tmp_path / "comparisons" / "fresh__warm.json").read_text(encoding="utf-8"))
    assert result["seeds"][0]["model_fields_differing"] == ["body_pos"]
    assert result["a"]["name"] == "fresh"


def test_main_reports_missing_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(rep, "OUTPUT", tmp_path)
    assert rep.main(["compare", "nothing", "here"]) == 2
    assert "error" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------- gate, fixed scenes
def test_merge_fixed_replaces_affected_seeds_only() -> None:
    original = {"sync25": {1: True, 2: False, 3: True}}
    fixed = {"sync25": {1: False, 3: False}}
    merged = rep.merge_fixed(original, fixed, lambda seed: seed in (1, 3))
    assert merged == {"sync25": {1: False, 2: False, 3: False}}
    with pytest.raises(rep.ReproducibilityError):
        rep.merge_fixed(original, {"sync25": {1: False}}, lambda seed: seed in (1, 3))


def test_discordance_is_the_share_of_disagreeing_seeds() -> None:
    a = {1: True, 2: False, 3: True, 4: False}
    b = {1: True, 2: True, 3: False, 4: False}
    assert rep.discordance(a, b, [1, 2, 3, 4]) == 0.5
    assert rep.discordance(a, b, [1, 4]) == 0.0
    with pytest.raises(rep.ReproducibilityError):
        rep.discordance(a, b, [])
