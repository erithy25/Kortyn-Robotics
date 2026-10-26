"""Power analysis of phase 4 (kortyn.power_phase4): truths, correlated simulation, summaries, checks."""

from __future__ import annotations

import numpy as np
import pytest

from kortyn import dose_curve as dc
from kortyn import power_phase4 as pw
from kortyn.fleet_predict import DoseResponse

SPEC = dc.CurveSpec()


def phase3_like_dose() -> DoseResponse:
    """A small dose-response on the phase-3 levels (0, 500, 1000, 1250, 1500, 2000 ms), two tasks."""
    levels = np.array([0.0, 500.0, 1000.0, 1250.0, 1500.0, 2000.0])
    rates = {0: [0.42, 0.14, 0.02, 0.0, 0.0, 0.0], 8: [0.5, 0.26, 0.12, 0.04, 0.0, 0.0]}
    seeds_per_task = 50
    groups = tuple(task for task in (0, 8) for _ in range(seeds_per_task))
    outcomes = np.zeros((levels.size, len(groups)))
    for column, task in enumerate(groups):
        index = column % seeds_per_task
        for level, rate in enumerate(rates[task]):
            outcomes[level, column] = float(index < round(rate * seeds_per_task))
    return DoseResponse(levels_ms=levels, seeds=tuple(range(len(groups))), groups=groups, outcomes=outcomes)


def test_dependence_matrix_is_a_valid_correlation_matrix() -> None:
    x = dc.effective_latency_ms(np.asarray(pw.LEVELS_MS))
    sigma = pw.Dependence().matrix(x)
    assert sigma.shape == (16, 16)
    assert np.allclose(sigma, sigma.T)
    assert np.allclose(np.diag(sigma), 1.0)
    assert np.all(np.linalg.eigvalsh(sigma) > 0)
    # neighbouring levels of one condition are more alike than distant ones or the other condition
    assert sigma[0, 1] > sigma[0, 7]
    assert sigma[0, 1] > sigma[0, 9]


def test_linear_truth_stretches_the_latency_axis() -> None:
    truth = pw.LinearTruth(x_ms=(50.0, 500.0, 1000.0), values=(0.5, 0.2, 0.0))
    stretched = pw.LinearTruth(x_ms=truth.x_ms, values=truth.values, k=2.0)
    assert float(stretched(50.0)) == pytest.approx(0.5)
    assert float(stretched(950.0)) == pytest.approx(float(truth(500.0)))
    assert float(truth(2000.0)) == pytest.approx(0.0)


def test_profile_fit_recovers_p_and_x50_for_a_fixed_steepness() -> None:
    truth = dc.TaskCurve(p=0.48, x50_ms=430.0, h=3.0)
    x = np.array([50.0, 100.0, 300.0, 500.0, 1000.0])
    trials = np.full(x.size, 1e6)
    fitted = pw.profile_fit(x, truth(x) * trials, trials, 3.0)
    assert fitted.h == 3.0
    assert fitted.p == pytest.approx(0.48, rel=1e-3)
    assert fitted.x50_ms == pytest.approx(430.0, rel=1e-3)


def test_scenarios_cover_shapes_effects_and_a_misspecified_truth() -> None:
    scenarios = pw.build_scenarios(phase3_like_dose(), SPEC, {"mittel": {0: 5.0, 8: 2.5}})
    names = [s.name for s in scenarios]
    assert names == ["mittel_k1", "mittel_k1.5", "mittel_k2", "linear_k1", "linear_k1.5", "linear_k2"]
    base, doubled = scenarios[0], scenarios[2]
    for task in (0, 8):
        a, b = base.without[task], doubled.with_rtc[task]
        assert isinstance(a, dc.TaskCurve)
        assert isinstance(b, dc.TaskCurve)
        assert b.x50_ms == pytest.approx(2.0 * a.x50_ms)
        assert b.p == a.p
        assert b.h == a.h
    assert pw.true_curve(base.with_rtc, SPEC).tolerable_latency(0.03) == pw.true_curve(
        base.without, SPEC
    ).tolerable_latency(0.03)


def test_simulation_reproduces_the_true_marginals() -> None:
    truth = {0: dc.TaskCurve(0.45, 450.0, 3.0), 8: dc.TaskCurve(0.5, 550.0, 2.0)}
    scenario = pw.Scenario("t", truth, truth, 1.0)
    without, with_rtc = pw.simulate(scenario, SPEC, pw.Dependence(), 2000, np.random.default_rng(1))
    x = dc.effective_latency_ms(np.asarray(pw.LEVELS_MS))
    for dose in (without, with_rtc):
        for task in (0, 8):
            mask = np.asarray(dose.groups) == task
            observed = dose.outcomes[:, mask].mean(axis=1)
            expected = truth[task](x)
            assert np.all(
                np.abs(observed - expected) < 4.0 * np.sqrt(expected * (1 - expected) / 2000) + 1e-9
            )
    # the same seed is correlated across conditions
    flat_a, flat_b = without.outcomes.ravel(), with_rtc.outcomes.ravel()
    assert np.corrcoef(flat_a, flat_b)[0, 1] > 0.2


def test_tolerable_from_grid_matches_the_curve_method() -> None:
    grid = SPEC.tick_ms * np.arange(1, SPEC.budget_ticks + 1)
    curves = [
        dc.DoseCurve(spec=SPEC, tasks={0: dc.TaskCurve(0.5, 400.0, 3.0)}),
        dc.DoseCurve(spec=SPEC, tasks={0: dc.TaskCurve(0.5, 1e5, 2.0)}),
    ]
    values = np.array([c.value(grid) for c in curves])
    result = pw._tolerable_from_grid(values, grid)
    assert result.tolist() == [c.tolerable_latency(pw.TOLERANCE)[0] for c in curves]


def test_one_replicate_and_its_summary() -> None:
    scenarios = pw.build_scenarios(phase3_like_dose(), SPEC, {"mittel": {0: 5.0, 8: 2.5}})
    scenario = scenarios[2]  # RTC doubles x50
    replicates = [pw.run_replicate(scenario, SPEC, pw.Dependence(), 20, 20, seed) for seed in (1, 3)]
    for replicate in replicates:
        assert replicate.conservative_without <= replicate.upper_without
        assert replicate.difference_low <= replicate.difference_high
        assert replicate.gain_low <= replicate.gain_high
        assert replicate.lstar_with % SPEC.tick_ms == 0.0
    summary = pw.summarize(scenario, SPEC, replicates)
    assert summary.replicates == 2
    assert summary.true_lstar_with > summary.true_lstar_without
    assert summary.true_gain > 0.0
    assert 0.0 <= summary.power_gain <= 1.0
    assert 0.0 <= summary.coverage_with <= 1.0


def test_paired_check_power_separates_equal_from_much_worse() -> None:
    equal = pw.paired_check_power(0.5, 0.5, 0.9, 100, 0.15, 200, 200, 3)
    worse = pw.paired_check_power(0.5, 0.25, 0.9, 100, 0.15, 200, 200, 3)
    assert equal > 0.8
    assert worse < 0.1


def test_validation_precision_grows_with_the_bias() -> None:
    small = pw.validation_precision(100, 0.3, 2, 0.05)
    large = pw.validation_precision(100, 0.3, 2, 0.15)
    assert small["half_width"] == pytest.approx(2.2414 * np.sqrt(0.21 / 100), rel=1e-3)
    assert small["power"] < large["power"] < 1.0
    assert pw.validation_precision(60, 0.3, 2, 0.05)["half_width"] > small["half_width"]
