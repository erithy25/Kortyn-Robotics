"""Phase-3 report data: every section from synthetic results; missing inputs are skipped."""

from __future__ import annotations

import json
import random
from pathlib import Path

from kortyn.fleet import load_fleet_config
from kortyn.report_phase3 import capacity_lookup, eur, interval_pp, main, p_text, pct

BASE = Path(__file__).resolve().parents[1] / "kortyn" / "experiments" / "phase3" / "libero_base.yaml"
ECONOMICS = Path(__file__).resolve().parents[1] / "kortyn" / "experiments" / "phase3" / "economics.yaml"


def write_runs(root: Path, runs: dict[str, list[bool]], seed_start: int = 1000) -> None:
    for label, outcomes in runs.items():
        directory = root / "smolvla_libero" / label
        directory.mkdir(parents=True)
        rows = [
            {
                "seed": seed_start + i,
                "attempt": 0,
                "success": ok,
                "steps": 300 if ok else 520,
                "starved_ticks": 3,
            }
            for i, ok in enumerate(outcomes)
        ]
        (directory / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def outcomes(rate: float, n: int, seed: int) -> list[bool]:
    rng = random.Random(seed)
    return [rng.random() < rate for _ in range(n)]


def write_config(path: Path, results: Path, task_ids: list[int], per_task: int) -> None:
    path.write_text(
        f"extends: {BASE}\n"
        f"env: {{libero: {{task_ids: {task_ids}, episodes_per_task: {per_task}}}}}\n"
        f"experiment: {{results_dir: {results}, episodes: {len(task_ids) * per_task}}}\n",
        encoding="utf-8",
    )


def capacity_entry(network: str, gpu: str, batching: str, strategy: str, capacity: int) -> dict[str, object]:
    sizes = []
    for robots in (1, 2, 4, 8):
        ok = robots <= capacity
        sizes.append(
            {
                "robots": robots,
                "miss_share": 0.0 if ok else 0.2,
                "busy_share": min(1.0, 0.2 * robots),
                "criterion_1": ok,
                "criterion_2": ok,
            }
        )
    return {
        "network": network,
        "gpu": gpu,
        "batching": batching,
        "strategy": strategy,
        "capacity": capacity,
        "sizes": sizes,
    }


def write_fleet(tmp_path: Path) -> Path:
    capacity_path = tmp_path / "capacity.json"
    configurations = []
    for gpu, base in (("g1", 2), ("g2", 4)):
        configurations.append(capacity_entry("buero_wlan", gpu, "realistisch", "a", base))
        for batching, extra in (("realistisch", 2), ("optimistisch", 4)):
            configurations.append(capacity_entry("buero_wlan", gpu, batching, "d", base * extra))
    capacity_path.write_text(json.dumps({"configurations": configurations}), encoding="utf-8")
    fleet = tmp_path / "fleet.yaml"
    fleet.write_text(
        "gpus: {g1: {s1_ms: 300}, g2: {s1_ms: 150}}\n"
        "batching: {realistisch: {beta: 0.15}, optimistisch: {beta: 0.05}}\n"
        "strategies: {a: {scheduler: fifo}, d: {scheduler: edf_batch, max_batch: 16}}\n"
        "networks: {buero_wlan: {scenario: buero_wlan, margin_ms: 15}}\n"
        "fleet_sizes: [1, 2, 4, 8]\n"
        "episode_lengths: {values: [300]}\n"
        "prediction:\n"
        f"  experiment: {tmp_path / 'main.yaml'}\n"
        "  levels: [{latency_ms: 0, run: fest_0ms}, {latency_ms: 1000, run: fest_1000ms}]\n"
        f"  output: {capacity_path}\n",
        encoding="utf-8",
    )
    return fleet


def test_report_builds_every_section(tmp_path: Path) -> None:
    pilot_results, main_results = tmp_path / "pilot", tmp_path / "main"
    write_config(tmp_path / "pilot.yaml", pilot_results, list(range(4)), 10)
    write_config(tmp_path / "main.yaml", main_results, [1, 2], 20)
    write_runs(pilot_results, {"sync25": outcomes(0.5, 40, 1), "sync50": outcomes(0.5, 40, 2)})
    summary_dir = pilot_results / "lerobot_eval" / "all"
    summary_dir.mkdir(parents=True)
    lerobot = {str(1000 + i): ok for i, ok in enumerate(outcomes(0.5, 40, 3))}
    (summary_dir / "summary.json").write_text(json.dumps({"success_by_seed": lerobot}), encoding="utf-8")
    write_runs(
        main_results,
        {
            "sync25": outcomes(0.6, 40, 4),
            "sync25_q90": outcomes(0.6, 40, 5),
            "fest_0ms": outcomes(0.6, 40, 6),
            "fest_1000ms": outcomes(0.55, 40, 7),
            "fest_2000ms": outcomes(0.3, 40, 8),
            "jitter_500_1500ms": outcomes(0.5, 40, 9),
            "warm_1000ms": outcomes(0.6, 40, 10),
        },
    )
    fleet = write_fleet(tmp_path)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            [
                {
                    "case": "a_n2",
                    "robots": 2,
                    "admitted": 2,
                    "episodes": 100,
                    "measured": 0.5,
                    "predicted": 0.52,
                    "difference": -0.02,
                    "interval": [-0.12, 0.08],
                    "passes": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "reports"
    code = main(
        [
            "--pilot-config",
            str(tmp_path / "pilot.yaml"),
            "--main-config",
            str(tmp_path / "main.yaml"),
            "--fleet-config",
            str(fleet),
            "--validation",
            str(evaluation),
            "--economics",
            str(ECONOMICS),
            "--out-dir",
            str(out),
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 0
    text = (out / "phase3_data.md").read_text(encoding="utf-8")
    for heading in (
        "## Schritt 1",
        "Reproduktion R1",
        "Aufgabenwahl",
        "## Schritt 2",
        "## Schritt 4",
        "## Validierung",
    ):
        assert heading in text, heading
    assert "Regel 1.4: JPEG q90" in text
    assert "1.3 Kalt- gegen Warmstart" in text
    assert (out / "phase3_dosis_wirkung.png").stat().st_size > 10_000
    assert (out / "phase3_kapazitaet_buero_wlan.png").stat().st_size > 10_000
    assert "a_n2" in text


def test_report_skips_missing_inputs(tmp_path: Path) -> None:
    write_config(tmp_path / "pilot.yaml", tmp_path / "nothing", [0], 10)
    fleet = write_fleet(tmp_path)
    (tmp_path / "capacity.json").unlink()
    out = tmp_path / "reports"
    code = main(
        [
            "--pilot-config",
            str(tmp_path / "pilot.yaml"),
            "--main-config",
            str(tmp_path / "missing.yaml"),
            "--fleet-config",
            str(fleet),
            "--validation",
            str(tmp_path / "missing.json"),
            "--out-dir",
            str(out),
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 0
    text = (out / "phase3_data.md").read_text(encoding="utf-8")
    assert "## Schritt 1" not in text
    assert "## Schritt 4" not in text


def test_capacity_lookup_spreads_unbatched_strategies(tmp_path: Path) -> None:
    fleet = load_fleet_config(write_fleet(tmp_path))
    data = json.loads((tmp_path / "capacity.json").read_text(encoding="utf-8"))
    table = capacity_lookup(data, fleet)
    assert table[("buero_wlan", "g1", "optimistisch", "a")] == 2  # FIFO: same under every profile
    assert table[("buero_wlan", "g2", "optimistisch", "d")] == 16


def test_formatting_helpers() -> None:
    assert pct(0.456) == "46 %"
    assert pct(None) == "–"
    assert interval_pp((-0.12, 0.034)) == "[-12, +3] pp"
    assert p_text(0.0004) == "< 0,001"
    assert p_text(0.25) == "0,250"
    assert eur(1234.5) == "1 234 €"
    assert eur(12.345, 2) == "12.35 €"
