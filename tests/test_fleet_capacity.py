"""Phase-4 capacity with smooth curves (kortyn.fleet_capacity) on synthetic stage-1 records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kortyn import dose_curve as dc
from kortyn import fleet_capacity as fc
from kortyn.config import ConfigError
from kortyn.fleet_predict import PredictionError

SPEC = dc.CurveSpec()


def curve(x50: float) -> dc.DoseCurve:
    return dc.DoseCurve(spec=SPEC, tasks={0: dc.TaskCurve(0.5, x50, 3.0)})


def fit(x50: float) -> dc.BootstrapFit:
    return dc.BootstrapFit(point=curve(x50), draws=(curve(x50 * 0.9), curve(x50), curve(x50 * 1.1)))


def record(
    robots: int,
    replication: int,
    latencies_ms: list[float],
    admitted: int | None = None,
    medium: float | None = None,
) -> dict:
    admitted = robots if admitted is None else admitted
    fleet = []
    for index in range(robots):
        latency = latencies_ms[index % len(latencies_ms)]
        histogram = {str(int(latency // 10)): 100} if index < admitted else {}
        fleet.append({"admitted": index < admitted, "histogram": histogram})
    return {
        "key": {
            "network": "buero_wlan",
            "gpu": "gpu",
            "batching": "b",
            "strategy": "d",
            "robots": robots,
            "replication": replication,
        },
        "requests": 100 * robots,
        "misses": 0,
        "mean_batch": 1.0,
        "busy_share": 0.5,
        "medium_busy_share": medium,
        "robots": fleet,
    }


def write(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_bin_curves_and_predictions(tmp_path: Path) -> None:
    curves = fc.bin_curves(fit(400.0), 60)
    assert curves.point.shape == (60,)
    assert curves.draws.shape == (3, 60)
    # bin 24 covers 240-250 ms: centre 245 ms, effective latency 250 ms
    assert curves.point[24] == pytest.approx(float(curve(400.0).value(250.0)))
    runs = fc.read_runs(write(tmp_path / "s.jsonl", [record(2, 0, [120.0, 320.0], admitted=1)]))
    points, draws = fc.predictions(runs[0], curves)
    assert points[0] == pytest.approx(float(curve(400.0).value(120.0)))
    assert points[1] == 0.0  # rejected
    assert draws.shape == (3, 2)
    with pytest.raises(PredictionError):
        fc.bin_curves(fit(400.0), 0)


def test_capacity_with_point_and_conservative_criterion(tmp_path: Path) -> None:
    # fleet latencies grow with the fleet size: 120 ms alone, 220 ms with 2, 420 ms with 3 robots
    records = []
    for replication in range(2):
        records.append(record(1, replication, [120.0]))
        records.append(record(2, replication, [120.0, 220.0]))
        records.append(record(3, replication, [120.0, 220.0, 420.0]))
    runs = fc.read_runs(write(tmp_path / "s.jsonl", records))
    curves = {"c": fc.bin_curves(fit(600.0), 50)}
    [result] = fc.evaluate(runs, curves, {"gpu": "c"}, 0.03)
    by_size = {s.robots: s for s in result.sizes}
    single = float(curve(600.0).value(120.0))
    assert by_size[2].loss == pytest.approx(single - float(curve(600.0).value(220.0)))
    assert by_size[3].loss == pytest.approx(single - float(curve(600.0).value(420.0)))
    assert by_size[2].criterion_2
    assert not by_size[3].criterion_2
    assert result.capacity == 2
    assert result.capacity_conservative <= result.capacity
    assert by_size[3].loss_upper >= by_size[3].loss - 1e-12 or by_size[3].loss_upper > 0
    # configurations without a curve are skipped; missing single-robot runs are an error
    assert fc.evaluate(runs, curves, {"other": "c"}, 0.03) == []
    no_single = [run for run in runs if run.robots != 1]
    with pytest.raises(PredictionError, match="single-robot"):
        fc.evaluate(no_single, curves, {"gpu": "c"}, 0.03)


def test_capacity_stops_at_the_first_failing_size() -> None:
    def size(robots: int, ok: bool) -> fc.SizeResult:
        return fc.SizeResult(robots, 1, 10, 0, 0.0, True, robots, 1.0, 0.5, 0.4, 0.4, 0.4, 0.0, 0.0, ok, ok)

    assert fc.capacity_of([size(1, True), size(2, True), size(3, False), size(4, True)]) == 2
    assert fc.capacity_of([size(1, False)]) == 0


def test_curve_map_parsing() -> None:
    assert fc.parse_curve_map(["a=ohne_rtc", "b=mit_rtc"]) == {"a": "ohne_rtc", "b": "mit_rtc"}
    for bad in ("a", "=x", "a="):
        with pytest.raises(ConfigError):
            fc.parse_curve_map([bad])


def test_predictions_need_curves_as_long_as_the_histograms(tmp_path: Path) -> None:
    runs = fc.read_runs(write(tmp_path / "s.jsonl", [record(1, 0, [920.0])]))
    with pytest.raises(PredictionError):
        fc.predictions(runs[0], fc.bin_curves(fit(400.0), 10))
    assert np.isclose(
        fc.predictions(runs[0], fc.bin_curves(fit(400.0), 100))[0][0], curve(400.0).value(920.0)
    )


def test_medium_busy_share_is_averaged_over_replications(tmp_path: Path) -> None:
    records = [
        record(1, 0, [120.0], medium=0.1),
        record(1, 1, [120.0], medium=0.3),
        record(2, 0, [120.0, 220.0], medium=0.4),
        record(2, 1, [120.0, 220.0], medium=None),  # a run without the medium: no mean
    ]
    runs = fc.read_runs(write(tmp_path / "s.jsonl", records))
    [result] = fc.evaluate(runs, {"c": fc.bin_curves(fit(600.0), 50)}, {"gpu": "c"}, 0.03)
    by_size = {s.robots: s for s in result.sizes}
    assert by_size[1].medium_busy_share == pytest.approx(0.2)
    assert by_size[2].medium_busy_share is None
    legacy = record(1, 0, [120.0])
    del legacy["medium_busy_share"]  # phase-3 records carry no medium
    assert fc.read_runs(write(tmp_path / "legacy.jsonl", [legacy]))[0].medium_busy_share is None


def test_a_run_given_twice_is_an_error(tmp_path: Path) -> None:
    path = write(tmp_path / "s.jsonl", [record(1, 0, [120.0]), record(2, 0, [120.0, 220.0])])
    runs = fc.read_runs(path) + fc.read_runs(path)
    with pytest.raises(PredictionError, match="more than once"):
        fc.evaluate(runs, {"c": fc.bin_curves(fit(600.0), 50)}, {"gpu": "c"}, 0.03)
