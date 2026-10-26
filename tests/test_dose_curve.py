"""Smooth, monotone dose-response curves of phase 4 (kortyn.dose_curve)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import check_grad

from kortyn import dose_curve as dc
from kortyn.fleet_predict import DoseResponse, PredictionError

LEVELS = np.array([0.0, 100.0, 200.0, 300.0, 500.0, 750.0, 1000.0, 1250.0])
X = dc.effective_latency_ms(LEVELS)


def synthetic_dose(curves: dict[int, dc.TaskCurve], seeds_per_task: int, seed: int) -> DoseResponse:
    """Independent Bernoulli outcomes from known task curves (levels LEVELS)."""
    rng = np.random.default_rng(seed)
    tasks = sorted(curves)
    groups = tuple(task for task in tasks for _ in range(seeds_per_task))
    outcomes = np.array([[float(rng.random() < curves[task](x)) for task in groups] for x in X], dtype=float)
    return DoseResponse(levels_ms=LEVELS, seeds=tuple(range(len(groups))), groups=groups, outcomes=outcomes)


def test_effective_latency_rounds_up_to_whole_ticks() -> None:
    values = dc.effective_latency_ms([0.0, 1.0, 50.0, 50.001, 99.0, 100.0, 460.0, 500.0, 1250.0])
    assert values.tolist() == [50.0, 50.0, 50.0, 100.0, 100.0, 100.0, 500.0, 500.0, 1250.0]
    assert dc.effective_latency_ms(130.0, tick_ms=25.0).tolist() == 150.0
    with pytest.raises(PredictionError):
        dc.effective_latency_ms([-1.0])
    with pytest.raises(PredictionError):
        dc.effective_latency_ms([float("nan")])
    with pytest.raises(PredictionError):
        dc.effective_latency_ms([10.0], tick_ms=0.0)


def test_task_curve_is_strictly_decreasing_and_validated() -> None:
    curve = dc.TaskCurve(p=0.5, x50_ms=400.0, h=2.0)
    values = curve(np.linspace(0.0, 2000.0, 50))
    assert values[0] == pytest.approx(0.5)
    assert np.all(np.diff(values) < 0)
    assert curve(400.0) == pytest.approx(0.25)
    for bad in ((1.2, 400.0, 2.0), (0.5, 0.0, 2.0), (0.5, 400.0, 0.0)):
        with pytest.raises(PredictionError):
            dc.TaskCurve(*bad)


def test_gradient_matches_finite_differences() -> None:
    wins = np.array([25.0, 24.0, 21.0, 18.0, 11.0, 6.0, 2.0, 1.0])
    trials = np.full(8, 50.0)
    log_x = np.log(X)
    for theta in ([0.1, 6.0, 0.5], [-1.0, 5.0, 1.5], [2.0, 7.5, -0.5]):
        error = check_grad(
            lambda t: dc._nll_and_grad(t, log_x, wins, trials)[0],
            lambda t: dc._nll_and_grad(t, log_x, wins, trials)[1],
            np.array(theta),
        )
        assert error < 1e-4


def test_fit_recovers_known_parameters() -> None:
    truth = dc.TaskCurve(p=0.45, x50_ms=520.0, h=2.6)
    trials = np.full(8, 1e6)
    curve, nll = dc.fit_task_curve(X, truth(X) * trials, trials)
    assert curve.p == pytest.approx(truth.p, rel=1e-3)
    assert curve.x50_ms == pytest.approx(truth.x50_ms, rel=1e-3)
    assert curve.h == pytest.approx(truth.h, rel=1e-3)
    _warm, warm_nll = dc.fit_task_curve(X, truth(X) * trials, trials, start=curve)
    assert warm_nll == pytest.approx(nll, rel=1e-9)


def test_fit_rejects_inconsistent_counts() -> None:
    with pytest.raises(PredictionError):
        dc.fit_task_curve(X[:3], [1.0, 2.0], [5.0, 5.0, 5.0])
    with pytest.raises(PredictionError):
        dc.fit_task_curve(X[:3], [6.0, 2.0, 1.0], [5.0, 5.0, 5.0])
    with pytest.raises(PredictionError):
        dc.fit_task_curve(X[:3], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0])


def test_isotonic_regression_pools_adjacent_violators() -> None:
    fitted = dc.isotonic_decreasing([0.5, 0.6, 0.3, 0.35, 0.1], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert fitted.tolist() == pytest.approx([0.55, 0.55, 0.325, 0.325, 0.1])
    weighted = dc.isotonic_decreasing([0.2, 0.4], [3.0, 1.0])
    assert weighted.tolist() == pytest.approx([0.25, 0.25])
    curve = dc.IsotonicCurve(x_ms=(50.0, 100.0, 200.0), values=(0.5, 0.4, 0.1))
    assert curve(np.array([10.0, 50.0, 150.0, 900.0])).tolist()[0] == pytest.approx(0.5)
    assert curve(900.0) == pytest.approx(0.1)
    with pytest.raises(PredictionError):
        dc.IsotonicCurve(x_ms=(50.0, 100.0), values=(0.1, 0.4))


def test_goodness_of_fit_falls_back_to_isotonic_curve() -> None:
    good = synthetic_dose({0: dc.TaskCurve(0.5, 450.0, 2.5), 8: dc.TaskCurve(0.45, 600.0, 1.8)}, 400, 3)
    fitted = dc.fit_dose_curve(good, dc.CurveSpec())
    assert set(fitted.families.values()) == {"loglogistic"}
    # a curve with a hump in the middle cannot be log-logistic: the deviance test sends it to the fallback
    groups = tuple([0] * 400)
    rates = np.array([0.5, 0.5, 0.1, 0.45, 0.45, 0.05, 0.05, 0.0])
    outcomes = np.array([[1.0 if i < int(400 * r) else 0.0 for i in range(400)] for r in rates])
    hump = DoseResponse(levels_ms=LEVELS, seeds=tuple(range(400)), groups=groups, outcomes=outcomes)
    fallback = dc.fit_dose_curve(hump, dc.CurveSpec())
    assert fallback.families == {0: "isotonic"}
    values = fallback.value(LEVELS)
    assert np.all(np.diff(values) <= 1e-12)
    report = dc.fit_report(hump, fallback)
    assert report[0]["family"] == "isotonic"


def test_tail_rule_beyond_the_budget() -> None:
    spec = dc.CurveSpec()
    curve = dc.DoseCurve(spec=spec, tasks={0: dc.TaskCurve(0.5, 600.0, 2.0)})
    at_budget = float(dc.TaskCurve(0.5, 600.0, 2.0)(1250.0))
    assert float(curve.value(1250.0)) == pytest.approx(at_budget)
    assert float(curve.value(1360.0)) == pytest.approx(at_budget * (1500.0 - 1400.0) / 250.0)
    assert float(curve.value(1500.0)) == 0.0
    assert float(curve.value(3000.0)) == 0.0


def test_curve_spec_is_validated() -> None:
    with pytest.raises(PredictionError):
        dc.CurveSpec(tick_ms=0.0)
    with pytest.raises(PredictionError):
        dc.CurveSpec(fit_max_ms=1250.0, zero_from_ms=1000.0)
    with pytest.raises(PredictionError):
        dc.CurveSpec(gof_alpha=1.5)
    assert dc.CurveSpec().budget_ticks == 25


def test_tolerable_latency_on_the_tick_grid() -> None:
    spec = dc.CurveSpec()
    task = dc.TaskCurve(0.5, 400.0, 3.0)
    curve = dc.DoseCurve(spec=spec, tasks={0: task})
    lstar, censored = curve.tolerable_latency(0.03)
    grid = 50.0 * np.arange(1, 26)
    losses = task(50.0) - task(grid)
    expected = grid[np.flatnonzero(losses <= 0.03)[-1]]
    assert lstar == expected
    assert not censored
    assert float(curve.loss(lstar)) <= 0.03 < float(curve.loss(lstar + 1.0))
    # continuous effective latency: g(50) - g(x*) = 0.03
    x_star = curve.tolerable_x(0.03)
    assert task(50.0) - task(x_star) == pytest.approx(0.03, abs=1e-9)
    assert lstar <= x_star < lstar + 50.0
    flat = dc.DoseCurve(spec=spec, tasks={0: dc.TaskCurve(0.5, 1e5, 2.0)})
    assert flat.tolerable_latency(0.03) == (1250.0, True)
    assert flat.tolerable_x(0.03) == 1250.0
    with pytest.raises(PredictionError):
        curve.tolerable_latency(0.0)


def test_pooled_value_and_mean_over_latencies_use_the_tick_rule() -> None:
    spec = dc.CurveSpec()
    a, b = dc.TaskCurve(0.6, 500.0, 2.0), dc.TaskCurve(0.4, 300.0, 1.5)
    curve = dc.DoseCurve(spec=spec, tasks={0: a, 8: b})
    assert float(curve.value(460.0)) == pytest.approx(0.5 * (a(500.0) + b(500.0)))
    assert float(curve.task_value(8, 0.0)) == pytest.approx(float(b(50.0)))
    mean = curve.mean_over([460.0, 120.0], [3.0, 1.0])
    assert mean == pytest.approx((3.0 * float(curve.value(500.0)) + float(curve.value(150.0))) / 4.0)
    assert curve.mean_over([460.0], [1.0], task=0) == pytest.approx(float(a(500.0)))
    with pytest.raises(PredictionError):
        curve.mean_over([1.0, 2.0], [1.0])


def test_bootstrap_fit_intervals_and_paired_differences() -> None:
    truth = {0: dc.TaskCurve(0.45, 420.0, 4.0), 8: dc.TaskCurve(0.5, 560.0, 2.5)}
    dose = synthetic_dose(truth, 50, 11)
    spec = dc.CurveSpec()
    multiplicities = dose.bootstrap_weights(60, 5)
    fit = dc.bootstrap_fit(dose, spec, multiplicities)
    assert len(fit.draws) == 60
    tolerable = fit.tolerable(0.03)
    assert tolerable.conservative_ms <= tolerable.upper_ms
    assert tolerable.point_ms % 50.0 == 0.0
    assert tolerable.conservative_ms % 50.0 == 0.0
    low, high = fit.band(np.array([0.0, 500.0, 1000.0]))
    assert np.all(low <= high)
    point, (lo, hi) = fit.loss_interval(500.0)
    assert lo <= hi
    assert 0.0 < point < 0.5
    # identical conditions: the paired difference is exactly zero in every draw
    same = dc.bootstrap_fit(dose, spec, multiplicities)
    difference, interval = dc.paired_tolerable_difference(fit, same, 0.03)
    assert difference == 0.0
    assert interval == (0.0, 0.0)
    gain, gain_interval = dc.mean_success_difference(fit, same, 50.0 * np.arange(1, 26))
    assert gain == 0.0
    assert gain_interval == (0.0, 0.0)
    with pytest.raises(PredictionError):
        dc.bootstrap_fit(dose, spec, multiplicities[:, :10])
    shorter = dc.bootstrap_fit(dose, spec, multiplicities[:10])
    with pytest.raises(PredictionError):
        dc.paired_tolerable_difference(fit, shorter, 0.03)


def test_fit_needs_distinct_ticks_and_enough_levels() -> None:
    groups = (0, 0)
    two_on_one_tick = DoseResponse(
        levels_ms=np.array([0.0, 50.0, 100.0, 200.0, 300.0]),
        seeds=(1, 2),
        groups=groups,
        outcomes=np.ones((5, 2)),
    )
    with pytest.raises(PredictionError, match="same control tick"):
        dc.fit_dose_curve(two_on_one_tick, dc.CurveSpec())
    few = DoseResponse(
        levels_ms=np.array([0.0, 500.0, 1500.0]), seeds=(1, 2), groups=groups, outcomes=np.ones((3, 2))
    )
    with pytest.raises(PredictionError, match="four levels"):
        dc.fit_dose_curve(few, dc.CurveSpec())


def test_fit_report_lists_parameters() -> None:
    truth = {0: dc.TaskCurve(0.45, 420.0, 4.0)}
    dose = synthetic_dose(truth, 300, 2)
    curve = dc.fit_dose_curve(dose, dc.CurveSpec())
    report = dc.fit_report(dose, curve)
    assert report[0]["family"] == "loglogistic"
    assert float(report[0]["df"]) == 5.0
    assert 0.0 <= float(report[0]["p_value"]) <= 1.0
    assert math.isclose(float(report[0]["h"]), 4.0, rel_tol=0.5)
