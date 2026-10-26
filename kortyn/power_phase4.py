"""Power analysis of phase 4 (before any new run): can the design resolve the pre-registered thresholds?

Simulates the planned dose-response design many times from plausible true curves and applies the
pre-registered estimators of :mod:`kortyn.dose_curve` to every simulated data set:

* **Design:** 2 tasks x 50 seeds, levels 0, 100, 200, 300, 500, 750, 1000, 1250 ms, once without and once
  with RTC on the same seeds.
* **Truths:** log-logistic curves per task whose steepness ``h`` spans what the phase-3 data allow (profile
  likelihood), with ``p`` and ``x50`` fitted to the phase-3 levels for that ``h``; plus a misspecified truth
  (piecewise linear through the phase-3 level rates). RTC multiplies ``x50`` (or stretches the latency axis
  of the linear truth) by ``k`` = 1 (no effect), 1.5 or 2.
* **Dependence:** outcomes of one seed are correlated through a Gaussian copula: latent correlation
  ``rho0 * exp(-|x_i - x_j| / lambda)`` between levels and an extra factor ``kappa`` between the
  conditions, calibrated on the tetrachoric correlations of the phase-3 conditions.

Reported per scenario: the true and estimated ``L*`` (point, conservative), interval coverage, the bootstrap
spread of the loss at the true ``L*`` (resolution of the 3-pp threshold), the power of the paired ``L*``
difference and of the mean success difference within the budget, and the precision of a capacity-type
contrast (loss between 250 and 400 ms). Separately: the pass probability of the RTC-at-0-ms check and the
precision of the fleet validation.

One BLAS thread per process (several worker processes otherwise oversubscribe the CPU 50-fold)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -m kortyn.power_phase4 --simulations 100 --draws 150 --out results/phase4/power.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import norm

from kortyn import dose_curve as dc
from kortyn.fleet_predict import DoseResponse, PredictionError
from kortyn.metrics import mcnemar_exact

LOG = logging.getLogger("kortyn.power_phase4")

Floats = npt.NDArray[np.float64]

LEVELS_MS = (0.0, 100.0, 200.0, 300.0, 500.0, 750.0, 1000.0, 1250.0)
TOLERANCE = 0.03
TASKS = (0, 8)  # phase-3 task ids: task 0 (KITCHEN_SCENE3 ...) and task 8 (KITCHEN_SCENE8 ...)


# --------------------------------------------------------------------------------------------- truths
@dataclass(frozen=True)
class LinearTruth:
    """Piecewise linear curve through level rates (in effective latency), stretched by ``k`` along x."""

    x_ms: tuple[float, ...]
    values: tuple[float, ...]
    k: float = 1.0

    def __call__(self, x_ms: npt.ArrayLike) -> Floats:
        x = np.asarray(x_ms, dtype=float)
        # stretch around the first level: the 0-ms condition (one tick) stays where it is
        origin = self.x_ms[0]
        return np.interp(origin + (x - origin) / self.k, self.x_ms, self.values)


Truth = dc.TaskCurve | LinearTruth


@dataclass(frozen=True)
class Scenario:
    name: str
    without: dict[int, Truth]
    with_rtc: dict[int, Truth]
    k: float


def profile_fit(x_ms: Floats, successes: Floats, trials: Floats, h: float) -> dc.TaskCurve:
    """Maximum-likelihood ``p`` and ``x50`` for a fixed steepness ``h``."""
    log_x = np.log(x_ms)
    log_h = math.log(h)

    def nll(ab: Floats) -> float:
        return dc._nll_and_grad(np.array([ab[0], ab[1], log_h]), log_x, successes, trials)[0]

    results = [minimize(nll, [0.0, math.log(start)], method="Nelder-Mead") for start in (200.0, 500.0, 900.0)]
    best = min(results, key=lambda r: float(r.fun))
    return dc.TaskCurve(p=1.0 / (1.0 + math.exp(-float(best.x[0]))), x50_ms=math.exp(float(best.x[1])), h=h)


def build_scenarios(
    dose: DoseResponse, spec: dc.CurveSpec, shapes: dict[str, dict[int, float]]
) -> list[Scenario]:
    """Truth scenarios calibrated on the phase-3 dose-response (levels inside the budget)."""
    x = dc.effective_latency_ms(dose.levels_ms, spec.tick_ms)
    inside = x <= spec.fit_max_ms
    counts = {task: dc._task_counts(dose, None, task) for task in dose.group_ids}
    scenarios: list[Scenario] = []
    for name, steepness in shapes.items():
        base = {
            task: profile_fit(x[inside], counts[task][0][inside], counts[task][1][inside], steepness[task])
            for task in dose.group_ids
        }
        for k in (1.0, 1.5, 2.0):
            rtc: dict[int, Truth] = {
                t: dc.TaskCurve(p=c.p, x50_ms=c.x50_ms * k, h=c.h) for t, c in base.items()
            }
            scenarios.append(Scenario(f"{name}_k{k:g}", dict(base), rtc, k))
    linear = {
        task: LinearTruth(
            x_ms=tuple(float(v) for v in x[inside]),
            values=tuple(float(v) for v in (counts[task][0] / counts[task][1])[inside]),
        )
        for task in dose.group_ids
    }
    for k in (1.0, 1.5, 2.0):
        rtc_linear: dict[int, Truth] = {
            t: LinearTruth(x_ms=c.x_ms, values=c.values, k=k) for t, c in linear.items()
        }
        scenarios.append(Scenario(f"linear_k{k:g}", dict(linear), rtc_linear, k))
    return scenarios


def true_curve(truths: dict[int, Truth], spec: dc.CurveSpec) -> dc.DoseCurve:
    return dc.DoseCurve(spec=spec, tasks=dict(truths), families={})


# --------------------------------------------------------------------------------------------- simulation
@dataclass(frozen=True)
class Dependence:
    rho0: float = 0.7  # latent correlation of one seed between neighbouring levels
    length_ms: float = 1000.0  # decay of that correlation with the latency distance
    kappa: float = 0.85  # extra factor between the conditions (without / with RTC)

    def matrix(self, x_ms: Floats) -> Floats:
        level = np.exp(-np.abs(x_ms[:, None] - x_ms[None, :]) / self.length_ms)
        condition = np.array([[1.0, self.kappa], [self.kappa, 1.0]])
        sigma = self.rho0 * np.kron(condition, level)
        sigma[np.diag_indices_from(sigma)] = 1.0
        return sigma


def simulate(
    scenario: Scenario,
    spec: dc.CurveSpec,
    dependence: Dependence,
    seeds_per_task: int,
    rng: np.random.Generator,
) -> tuple[DoseResponse, DoseResponse]:
    """One data set: outcomes without and with RTC on the same seeds."""
    levels = np.asarray(LEVELS_MS)
    x = dc.effective_latency_ms(levels, spec.tick_ms)
    chol = np.linalg.cholesky(dependence.matrix(x))
    tasks = sorted(scenario.without)
    seeds = tuple(range(len(tasks) * seeds_per_task))
    groups = tuple(task for task in tasks for _ in range(seeds_per_task))
    outcomes = np.zeros((2, levels.size, len(seeds)))
    for column, task in enumerate(groups):
        latent = chol @ rng.standard_normal(2 * levels.size)
        for condition, truths in enumerate((scenario.without, scenario.with_rtc)):
            probability = np.clip(truths[task](x), 1e-9, 1.0 - 1e-9)
            cell = latent[condition * levels.size : (condition + 1) * levels.size]
            outcomes[condition, :, column] = (cell < norm.ppf(probability)).astype(float)
    return (
        DoseResponse(levels_ms=levels, seeds=seeds, groups=groups, outcomes=outcomes[0]),
        DoseResponse(levels_ms=levels, seeds=seeds, groups=groups, outcomes=outcomes[1]),
    )


@dataclass
class Replicate:
    lstar_without: float
    lstar_with: float
    conservative_without: float
    conservative_with: float
    upper_without: float
    upper_with: float
    loss_sd_at_true_without: float
    loss_sd_at_true_with: float
    capacity_contrast_sd: float
    difference_low: float  # paired L* difference (with - without): interval
    difference_high: float
    gain: float  # mean success difference (with - without) over the budget grid
    gain_low: float
    gain_high: float
    isotonic_fallbacks: int


def run_replicate(
    scenario: Scenario,
    spec: dc.CurveSpec,
    dependence: Dependence,
    seeds_per_task: int,
    draws: int,
    seed: int,
) -> Replicate:
    rng = np.random.default_rng(seed)
    without, with_rtc = simulate(scenario, spec, dependence, seeds_per_task, rng)
    multiplicities = without.bootstrap_weights(draws, seed + 1)
    fit_without = dc.bootstrap_fit(without, spec, multiplicities)
    fit_with = dc.bootstrap_fit(with_rtc, spec, multiplicities)
    true_without = true_curve(scenario.without, spec).tolerable_latency(TOLERANCE)[0]
    true_with = true_curve(scenario.with_rtc, spec).tolerable_latency(TOLERANCE)[0]
    # one pass over the draws: pooled curve on the tick grid, from which L*, losses and gains follow
    grid: Floats = spec.tick_ms * np.arange(1, spec.budget_ticks + 1, dtype=float)
    curves_without = np.array([d.value(grid) for d in fit_without.draws])  # (draws, ticks); grid[0] = 0 ms
    curves_with = np.array([d.value(grid) for d in fit_with.draws])
    lstar_without = _tolerable_from_grid(curves_without, grid)
    lstar_with = _tolerable_from_grid(curves_with, grid)
    a_point = fit_without.point.tolerable_latency(TOLERANCE)[0]
    b_point = fit_with.point.tolerable_latency(TOLERANCE)[0]
    a_low, a_high = dc._interval(lstar_without, 0.95, discrete=True)
    b_low, b_high = dc._interval(lstar_with, 0.95, discrete=True)
    d_low, d_high = dc._interval(lstar_with - lstar_without, 0.95, discrete=True)
    gains = (curves_with - curves_without).mean(axis=1)
    gain = float(np.mean(fit_with.point.value(grid) - fit_without.point.value(grid)))
    g_low, g_high = dc._interval(gains, 0.95)
    index_true_without = round(true_without / spec.tick_ms) - 1
    index_true_with = round(true_with / spec.tick_ms) - 1
    index_250, index_400 = 4, 7  # 250 ms and 400 ms on the tick grid
    return Replicate(
        lstar_without=a_point,
        lstar_with=b_point,
        conservative_without=a_low,
        conservative_with=b_low,
        upper_without=a_high,
        upper_with=b_high,
        loss_sd_at_true_without=float(np.std(curves_without[:, 0] - curves_without[:, index_true_without])),
        loss_sd_at_true_with=float(np.std(curves_with[:, 0] - curves_with[:, index_true_with])),
        capacity_contrast_sd=float(np.std(curves_without[:, index_250] - curves_without[:, index_400])),
        difference_low=d_low,
        difference_high=d_high,
        gain=gain,
        gain_low=g_low,
        gain_high=g_high,
        isotonic_fallbacks=sum(f == "isotonic" for f in fit_without.point.families.values())
        + sum(f == "isotonic" for f in fit_with.point.families.values()),
    )


def _tolerable_from_grid(curves: Floats, grid: Floats) -> Floats:
    """L* per draw from pooled curves on the tick grid (column 0 = one tick = 0 ms); censored at the end."""
    losses = curves[:, :1] - curves
    above = losses > TOLERANCE + 1e-12
    first = np.where(above.any(axis=1), above.argmax(axis=1), grid.size)
    return grid[np.maximum(first - 1, 0)]  # type: ignore[no-any-return]


def _task(args: tuple[Scenario, dc.CurveSpec, Dependence, int, int, int]) -> Replicate:
    return run_replicate(*args)


@dataclass
class ScenarioSummary:
    name: str
    k: float
    true_lstar_without: float
    true_lstar_with: float
    true_gain: float
    replicates: int
    point_without_median: float
    point_without_mad_ms: float  # median absolute deviation from the truth
    point_with_median: float
    point_with_mad_ms: float
    conservative_without_median: float
    conservative_with_median: float
    coverage_without: float  # share of [conservative, upper] intervals containing the true L*
    coverage_with: float
    loss_sd_at_true_without: float  # median bootstrap SD of the loss at the true L* (pp resolution)
    loss_sd_at_true_with: float
    capacity_contrast_sd: float  # median bootstrap SD of g(250) - g(400)
    power_lstar_difference: float  # share with the lower bound of the paired L* difference > 0
    power_gain: float  # share with the lower bound of the mean gain > 0
    isotonic_share: float


def summarize(scenario: Scenario, spec: dc.CurveSpec, replicates: Sequence[Replicate]) -> ScenarioSummary:
    true_without = true_curve(scenario.without, spec).tolerable_latency(TOLERANCE)[0]
    true_with = true_curve(scenario.with_rtc, spec).tolerable_latency(TOLERANCE)[0]
    grid = spec.tick_ms * np.arange(1, spec.budget_ticks + 1)
    true_gain = float(
        np.mean(
            true_curve(scenario.with_rtc, spec).value(grid) - true_curve(scenario.without, spec).value(grid)
        )
    )

    def column(name: str) -> Floats:
        return np.array([getattr(r, name) for r in replicates], dtype=float)

    def coverage(low: Floats, high: Floats, truth: float) -> float:
        return float(np.mean((low <= truth) & (truth <= high)))

    return ScenarioSummary(
        name=scenario.name,
        k=scenario.k,
        true_lstar_without=true_without,
        true_lstar_with=true_with,
        true_gain=true_gain,
        replicates=len(replicates),
        point_without_median=float(np.median(column("lstar_without"))),
        point_without_mad_ms=float(np.median(np.abs(column("lstar_without") - true_without))),
        point_with_median=float(np.median(column("lstar_with"))),
        point_with_mad_ms=float(np.median(np.abs(column("lstar_with") - true_with))),
        conservative_without_median=float(np.median(column("conservative_without"))),
        conservative_with_median=float(np.median(column("conservative_with"))),
        coverage_without=coverage(column("conservative_without"), column("upper_without"), true_without),
        coverage_with=coverage(column("conservative_with"), column("upper_with"), true_with),
        loss_sd_at_true_without=float(np.median(column("loss_sd_at_true_without"))),
        loss_sd_at_true_with=float(np.median(column("loss_sd_at_true_with"))),
        capacity_contrast_sd=float(np.median(column("capacity_contrast_sd"))),
        power_lstar_difference=float(np.mean(column("difference_low") > 0)),
        power_gain=float(np.mean(column("gain_low") > 0)),
        isotonic_share=float(np.mean(column("isotonic_fallbacks"))) / 4.0,
    )


# --------------------------------------------------------------------------------------------- level checks
def paired_check_power(
    p_reference: float,
    p_test: float,
    latent_rho: float,
    episodes: int,
    margin: float,
    simulations: int,
    draws: int,
    seed: int,
) -> float:
    """Pass rate of "McNemar p >= 0.05 and lower 95 % bootstrap bound of (test - reference) >= -margin"."""
    rng = np.random.default_rng(seed)
    z_ref, z_test = norm.ppf(p_reference), norm.ppf(p_test)
    passes = 0
    for _ in range(simulations):
        common = rng.standard_normal(episodes)
        a = math.sqrt(latent_rho) * common + math.sqrt(1.0 - latent_rho) * rng.standard_normal(episodes)
        b = math.sqrt(latent_rho) * common + math.sqrt(1.0 - latent_rho) * rng.standard_normal(episodes)
        reference = (a < z_ref).astype(float)
        test = (b < z_test).astype(float)
        only_ref = int(np.sum((reference == 1) & (test == 0)))
        only_test = int(np.sum((reference == 0) & (test == 1)))
        differences = test - reference
        picks = rng.integers(0, episodes, size=(draws, episodes))
        means = differences[picks].mean(axis=1)
        low = float(np.percentile(means, 2.5))
        if mcnemar_exact(only_ref, only_test) >= 0.05 and low >= -margin:
            passes += 1
    return passes / simulations


def validation_precision(episodes: int, success: float, cases: int, bias: float) -> dict[str, float]:
    """Half-width of the Bonferroni interval of measured - predicted and the power to detect ``bias``."""
    confidence = 1.0 - 0.05 / cases
    z = float(norm.ppf(0.5 + confidence / 2.0))
    se = math.sqrt(success * (1.0 - success) / episodes)
    power = float(norm.cdf(abs(bias) / se - z) + norm.cdf(-abs(bias) / se - z))
    return {"episodes": float(episodes), "half_width": z * se, "power": power, "bias": bias}


# --------------------------------------------------------------------------------------------- CLI
SHAPES = {  # steepness h per task; spans the range the phase-3 data allow (profile likelihood, p >= 0.01)
    "frueh": {0: 1.6, 8: 1.2},
    "mittel": {0: 5.0, 8: 2.5},
    "spaet": {0: 8.0, 8: 3.5},
}


def load_phase3_dose() -> DoseResponse:
    from kortyn.fleet import load_fleet_config
    from kortyn.fleet_predict import load_dose_response, prediction_config

    return load_dose_response(prediction_config(load_fleet_config("kortyn/experiments/phase3/fleet.yaml")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-4 power analysis (simulation, before any new run).")
    parser.add_argument("--out", default="results/phase4/power.json")
    parser.add_argument("--simulations", type=int, default=200, help="data sets per scenario")
    parser.add_argument("--draws", type=int, default=200, help="bootstrap draws per data set")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument(
        "--check-simulations", type=int, default=2000, help="simulations of the RTC-at-0-ms check"
    )
    parser.add_argument("--check-draws", type=int, default=2000, help="bootstrap draws of that check")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    if (
        args.simulations < 1
        or args.draws < 10
        or args.workers < 1
        or min(args.check_simulations, args.check_draws) < 1
    ):
        LOG.error("need simulations >= 1, draws >= 10, workers >= 1 and positive check sizes")
        return 2
    spec = dc.CurveSpec()
    dependence = Dependence()
    try:
        dose = load_phase3_dose()
        scenarios = build_scenarios(dose, spec, SHAPES)
    except PredictionError as exc:
        LOG.error("%s", exc)
        return 2
    summaries: list[dict[str, Any]] = []
    truths: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, scenario in enumerate(scenarios):
            jobs = [
                (scenario, spec, dependence, 50, args.draws, args.seed + 1000 * index + 2 * r)
                for r in range(args.simulations)
            ]
            replicates = list(pool.map(_task, jobs, chunksize=4))
            summary = summarize(scenario, spec, replicates)
            summaries.append(asdict(summary))
            truths[scenario.name] = {str(task): asdict(curve) for task, curve in scenario.without.items()}
            LOG.info(
                "%s: L* true %.0f/%.0f, point %.0f/%.0f, conservative %.0f/%.0f, power dL* %.2f, gain %.2f",
                scenario.name,
                summary.true_lstar_without,
                summary.true_lstar_with,
                summary.point_without_median,
                summary.point_with_median,
                summary.conservative_without_median,
                summary.conservative_with_median,
                summary.power_lstar_difference,
                summary.power_gain,
            )
    checks = {
        "rtc0_vs_sync": {
            f"p_rtc0={p:.2f}": paired_check_power(
                0.53, p, 0.7, 100, 0.15, args.check_simulations, args.check_draws, args.seed + 7
            )
            for p in (0.53, 0.46, 0.38)
        },
        "validation_100": [validation_precision(100, 0.3, 2, b) for b in (0.05, 0.07, 0.10, 0.15)],
        "validation_60": [validation_precision(60, 0.3, 2, b) for b in (0.05, 0.07, 0.10, 0.15)],
    }
    document = {
        "design": {
            "levels_ms": list(LEVELS_MS),
            "seeds_per_task": 50,
            "tasks": list(TASKS),
            "tolerance": TOLERANCE,
        },
        "spec": asdict(spec),
        "dependence": asdict(dependence),
        "simulations": args.simulations,
        "draws": args.draws,
        "seed": args.seed,
        "truths_without": truths,
        "scenarios": summaries,
        "checks": checks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    LOG.info("written %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
