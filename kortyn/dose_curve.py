"""Smooth, monotone dose-response curves (phase 4): success against end-to-end latency.

**Mechanism (checked on the phase-3 logs).** A plan whose observation was taken at tick ``k`` is merged at the
first tick that starts at or after its arrival, so an end-to-end latency ``L`` delays it by
``d(L) = max(1, ceil(L / dt))`` control ticks (``dt`` = 50 ms; ``n_stale`` of every phase-3 request is exactly
``d``). Success therefore depends on ``L`` only through the *effective latency* ``x(L) = dt * d(L)``: 0 ms and
50 ms are the same condition (``x`` = 50 ms) and 460 ms behaves like 500 ms. Phase 3 interpolated linearly in
``L``, which is optimistic between the levels by up to one tick.

**Model.** Per task ``t`` a smooth, strictly decreasing curve in ``x`` (log-logistic decay)::

    g_t(x) = p_t / (1 + (x / x50_t) ** h_t)        p_t in (0, 1), x50_t > 0, h_t > 0

``h < 1``: steepest at small latencies (convex); ``h > 1``: a flat start that bends down later. Only levels
inside the latency budget (``x <= fit_max_ms``) enter the fit: beyond it the robot starves in every cycle,
a different regime. There the curve falls linearly from ``g(fit_max)`` to 0 at ``zero_from_ms`` and stays 0
(phase 3 measured 0/100 at 1 500 and 2 000 ms). The pooled curve weights the tasks equally, like phase 3.

* **Fit:** binomial maximum likelihood per task on the (weighted) successes and trials of every level, with
  an analytic gradient and box bounds on (logit p, log x50, log h); several starting points.
* **Goodness of fit:** deviance against the saturated model (one rate per level), chi-square with
  ``levels - 3`` degrees of freedom. If ``p < gof_alpha`` the task falls back to a monotone, C1-smooth
  curve: PCHIP through the isotonic (pool-adjacent-violators) level rates. The family is decided once on all
  data; bootstrap draws reuse it.
* **Uncertainty:** paired bootstrap over seeds, stratified by task (multiplicities of
  :meth:`kortyn.fleet_predict.DoseResponse.bootstrap_weights`). Conditions measured on the same seeds (with
  and without RTC) use the same multiplicities, so their difference is paired as well.
* **Tolerable latency** ``L*``: the largest end-to-end latency whose predicted loss against 0 ms is at most
  ``tolerance``. Because ``x(L)`` moves in whole ticks, ``L*`` is a multiple of ``dt`` (at least ``dt``:
  everything up to one tick equals 0 ms). Point estimate from the fit on all data; conservative value = lower
  bound of the bootstrap interval. If the loss stays within the tolerance up to the budget, ``L*`` is
  censored at ``fit_max_ms``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.stats import chi2

from kortyn.fleet_predict import DoseResponse, PredictionError

Floats = npt.NDArray[np.float64]

_STARTS = ((150.0, 0.8), (400.0, 2.0), (800.0, 5.0), (1500.0, 1.2))  # (x50 ms, h)
_BOUNDS = ((-10.0, 10.0), (math.log(10.0), math.log(1e5)), (math.log(0.2), math.log(20.0)))


# --------------------------------------------------------------------------------------------- spec
@dataclass(frozen=True)
class CurveSpec:
    """Fixed settings of the curve (pre-registered in reports/phase4_preregistration.md)."""

    tick_ms: float = 50.0  # control period: latencies act in whole ticks
    fit_max_ms: float = 1250.0  # latency budget: the smooth fit uses levels with x <= fit_max_ms
    zero_from_ms: float = 1500.0  # beyond the budget: linear to 0 at this latency, 0 afterwards
    gof_alpha: float = 0.01  # deviance test; below this the task uses the isotonic PCHIP curve

    def __post_init__(self) -> None:
        if self.tick_ms <= 0 or self.fit_max_ms < self.tick_ms or self.zero_from_ms <= self.fit_max_ms:
            raise PredictionError(f"invalid curve spec {self}")
        if not 0.0 < self.gof_alpha < 1.0:
            raise PredictionError("gof_alpha must lie in (0, 1)")

    @property
    def budget_ticks(self) -> int:
        return round(self.fit_max_ms / self.tick_ms)


def effective_latency_ms(latency_ms: npt.ArrayLike, tick_ms: float = 50.0) -> Floats:
    """``x(L) = tick * max(1, ceil(L / tick))``: the delay in whole ticks until a plan is used."""
    if tick_ms <= 0:
        raise PredictionError("tick must be > 0")
    latency = np.asarray(latency_ms, dtype=float)
    if np.any(~np.isfinite(latency)) or np.any(latency < 0):
        raise PredictionError("latencies must be finite and >= 0")
    ticks = np.maximum(1.0, np.ceil(latency / tick_ms - 1e-9))  # 1e-9: exact multiples stay on their tick
    return ticks * tick_ms  # type: ignore[no-any-return]


# --------------------------------------------------------------------------------------------- task curves
class TaskFunction(Protocol):
    """A strictly or weakly decreasing success curve of one task over the effective latency (ms)."""

    def __call__(self, x_ms: npt.ArrayLike) -> Floats: ...


@dataclass(frozen=True)
class TaskCurve:
    """``g(x) = p / (1 + (x / x50_ms) ** h)``, strictly decreasing for ``h > 0``."""

    p: float
    x50_ms: float
    h: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0 or self.x50_ms <= 0.0 or self.h <= 0.0:
            raise PredictionError(f"invalid curve parameters {self}")

    def __call__(self, x_ms: npt.ArrayLike) -> Floats:
        x = np.maximum(np.asarray(x_ms, dtype=float), 0.0)
        return self.p / (1.0 + np.power(x / self.x50_ms, self.h))  # type: ignore[no-any-return]

    @property
    def theta(self) -> Floats:
        return np.array([math.log(self.p / (1.0 - self.p)), math.log(self.x50_ms), math.log(self.h)])


@dataclass(frozen=True)
class IsotonicCurve:
    """Fallback: PCHIP through the isotonic (non-increasing) level rates; constant outside the levels."""

    x_ms: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.x_ms) != len(self.values) or len(self.x_ms) < 2:
            raise PredictionError("isotonic curve needs at least two levels")
        if np.any(np.diff(self.x_ms) <= 0) or np.any(np.diff(self.values) > 1e-12):
            raise PredictionError("isotonic curve needs increasing levels and non-increasing values")

    def __call__(self, x_ms: npt.ArrayLike) -> Floats:
        x = np.clip(np.asarray(x_ms, dtype=float), self.x_ms[0], self.x_ms[-1])
        return np.asarray(PchipInterpolator(self.x_ms, self.values, extrapolate=False)(x), dtype=float)


def isotonic_decreasing(values: Sequence[float] | Floats, weights: Sequence[float] | Floats) -> Floats:
    """Weighted least-squares non-increasing fit (pool adjacent violators)."""
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.shape != w.shape or y.ndim != 1 or np.any(w < 0):
        raise PredictionError("need matching 1-D values and non-negative weights")
    blocks: list[list[float]] = []  # [mean, weight, count]
    for value, weight in zip(y, w, strict=True):
        blocks.append([float(value), float(weight), 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            mean_b, weight_b, count_b = blocks.pop()
            mean_a, weight_a, count_a = blocks[-1]
            total = weight_a + weight_b
            mean = (mean_a * weight_a + mean_b * weight_b) / total if total > 0 else 0.5 * (mean_a + mean_b)
            blocks[-1] = [mean, total, count_a + count_b]
    return np.concatenate([np.full(int(count), mean) for mean, _, count in blocks])


# --------------------------------------------------------------------------------------------- fitting
def _nll_and_grad(theta: Floats, log_x: Floats, wins: Floats, trials: Floats) -> tuple[float, Floats]:
    """Binomial negative log-likelihood of the log-logistic curve; gradient in (logit p, log x50, log h)."""
    a, b, c = (float(v) for v in theta)
    h = math.exp(c)
    t = h * (log_x - b)  # log u, u = (x / x50) ** h
    log_p = -np.logaddexp(0.0, -a)
    log_q = -np.logaddexp(0.0, a)  # log(1 - p)
    p, q = math.exp(log_p), math.exp(log_q)
    softplus_t = np.logaddexp(0.0, t)  # log(1 + u)
    log_f = log_p - softplus_t
    log_one_minus_f = np.logaddexp(log_q, t) - softplus_t  # log((q + u) / (1 + u))
    fails = trials - wins
    nll = -float(np.sum(wins * log_f + fails * log_one_minus_f))
    sigma = np.exp(t - softplus_t)  # u / (1 + u)
    sigma_q = np.exp(t - np.logaddexp(log_q, t))  # u / (q + u)
    d_t = wins * sigma - fails * (sigma_q - sigma)  # d nll / d t
    d_a = -float(np.sum(wins * q - fails * p * q * np.exp(-np.logaddexp(log_q, t))))
    grad = np.array([d_a, float(np.sum(d_t * -h)), float(np.sum(d_t * t))])
    return nll, grad


def _check_counts(x_ms: Floats, wins: Floats, trials: Floats) -> None:
    if x_ms.ndim != 1 or x_ms.shape != wins.shape or x_ms.shape != trials.shape:
        raise PredictionError("levels, successes and trials must be 1-D arrays of equal length")
    if np.any(x_ms <= 0) or np.any(trials < 0) or np.any(wins < 0) or np.any(wins > trials + 1e-9):
        raise PredictionError("need latencies > 0 and 0 <= successes <= trials")
    if np.count_nonzero(trials) < 2:
        raise PredictionError("a curve needs trials at two latencies at least")


def fit_task_curve(
    x_ms: Sequence[float] | Floats,
    successes: Sequence[float] | Floats,
    trials: Sequence[float] | Floats,
    start: TaskCurve | None = None,
) -> tuple[TaskCurve, float]:
    """Maximum-likelihood log-logistic curve for one task and its negative log-likelihood.

    ``start`` (a warm start, e.g. the fit on all data for a bootstrap draw) is tried before the default grid.
    """
    x = np.asarray(x_ms, dtype=float)
    wins = np.asarray(successes, dtype=float)
    counts = np.asarray(trials, dtype=float)
    _check_counts(x, wins, counts)
    used = counts > 0
    x, wins, counts = x[used], wins[used], counts[used]
    log_x = np.log(x)
    first = int(np.argmin(x))
    p0 = float(np.clip(wins[first] / counts[first], 0.02, 0.98))
    starts = (
        [] if start is None else [np.clip(start.theta, [lo for lo, _ in _BOUNDS], [hi for _, hi in _BOUNDS])]
    )
    starts += [np.array([math.log(p0 / (1.0 - p0)), math.log(x50), math.log(h)]) for x50, h in _STARTS]
    best: tuple[float, Floats] | None = None
    for theta0 in starts:
        result = minimize(
            _nll_and_grad, theta0, args=(log_x, wins, counts), jac=True, method="L-BFGS-B", bounds=_BOUNDS
        )
        value = float(result.fun)
        if best is None or value < best[0] - 1e-10:
            best = (value, np.asarray(result.x, dtype=float))
    assert best is not None
    a, b, c = best[1]
    curve = TaskCurve(p=1.0 / (1.0 + math.exp(-a)), x50_ms=math.exp(b), h=math.exp(c))
    return curve, best[0]


def goodness_of_fit(
    curve: TaskFunction, x_ms: Floats, successes: Floats, trials: Floats, parameters: int = 3
) -> tuple[float, int, float]:
    """Deviance against one rate per level, degrees of freedom and chi-square p-value."""
    used = trials > 0
    x, wins, counts = x_ms[used], successes[used], trials[used]
    f = np.clip(curve(x), 1e-12, 1.0 - 1e-12)
    rate = wins / counts
    with np.errstate(divide="ignore", invalid="ignore"):
        term_s = np.where(wins > 0, wins * np.log(rate / f), 0.0)
        term_f = np.where(counts - wins > 0, (counts - wins) * np.log((1.0 - rate) / (1.0 - f)), 0.0)
    deviance = float(2.0 * np.sum(term_s + term_f))
    df = int(x.size - parameters)
    p_value = float(chi2.sf(deviance, df)) if df > 0 else 1.0
    return deviance, df, p_value


def isotonic_task_curve(x_ms: Floats, successes: Floats, trials: Floats) -> IsotonicCurve:
    used = trials > 0
    x, wins, counts = x_ms[used], successes[used], trials[used]
    values = isotonic_decreasing(wins / counts, counts)
    return IsotonicCurve(x_ms=tuple(float(v) for v in x), values=tuple(float(v) for v in values))


# --------------------------------------------------------------------------------------------- pooled curve
@dataclass(frozen=True)
class DoseCurve:
    """Per-task curves over the effective latency plus the tick and tail rules; tasks weighted equally."""

    spec: CurveSpec
    tasks: dict[int, TaskFunction]
    families: dict[int, str] = field(default_factory=dict)  # "loglogistic" or "isotonic"

    def task_value(self, task: int, latency_ms: npt.ArrayLike) -> Floats:
        """Predicted success of one task at end-to-end latencies (tick rule and tail rule applied)."""
        x = effective_latency_ms(latency_ms, self.spec.tick_ms)
        return self._value_x(self.tasks[task], x)

    def value(self, latency_ms: npt.ArrayLike) -> Floats:
        """Pooled predicted success at end-to-end latencies."""
        x = effective_latency_ms(latency_ms, self.spec.tick_ms)
        return np.mean([self._value_x(curve, x) for curve in self.tasks.values()], axis=0)  # type: ignore[no-any-return]

    def value_x(self, x_ms: npt.ArrayLike) -> Floats:
        """Pooled curve over the effective latency itself (smooth; for plots and the continuous ``x*``)."""
        x = np.asarray(x_ms, dtype=float)
        return np.mean([self._value_x(curve, x) for curve in self.tasks.values()], axis=0)  # type: ignore[no-any-return]

    def _value_x(self, curve: TaskFunction, x: Floats) -> Floats:
        spec = self.spec
        inside = curve(np.minimum(x, spec.fit_max_ms))
        at_budget = curve(np.array([spec.fit_max_ms]))[0]
        share = np.clip((spec.zero_from_ms - x) / (spec.zero_from_ms - spec.fit_max_ms), 0.0, 1.0)
        return np.where(x <= spec.fit_max_ms, inside, at_budget * share)

    def mean_over(
        self, latencies_ms: Sequence[float], counts: Sequence[float], task: int | None = None
    ) -> float:
        """Mean prediction over a latency distribution (one robot's requests); one task or pooled."""
        values = np.asarray(latencies_ms, dtype=float)
        weight = np.asarray(counts, dtype=float)
        if values.shape != weight.shape or values.size == 0 or np.any(weight < 0) or weight.sum() <= 0:
            raise PredictionError("need matching, non-negative counts with a positive total")
        f = self.value(values) if task is None else self.task_value(task, values)
        return float(np.sum(f * weight) / weight.sum())

    def loss(self, latency_ms: npt.ArrayLike) -> Floats:
        """Predicted loss against 0 ms (pooled)."""
        return self.value(0.0) - self.value(latency_ms)

    def tolerable_latency(self, tolerance: float) -> tuple[float, bool]:
        """``(L*, censored)``: largest latency (a multiple of the tick) with loss <= ``tolerance``."""
        if tolerance <= 0:
            raise PredictionError("tolerance must be > 0")
        spec = self.spec
        grid = spec.tick_ms * np.arange(1, spec.budget_ticks + 1)
        losses = self.value(0.0) - self.value(grid)
        above = np.flatnonzero(losses > tolerance + 1e-12)
        if above.size == 0:
            return float(grid[-1]), True
        if above[0] == 0:  # cannot happen (grid[0] equals 0 ms), kept for safety
            return float(spec.tick_ms), False
        return float(grid[above[0] - 1]), False

    def tolerable_x(self, tolerance: float) -> float:
        """Continuous effective latency where the smooth curve has lost ``tolerance`` (budget if never)."""
        spec = self.spec
        base = float(self.value_x(spec.tick_ms))
        low, high = spec.tick_ms, spec.fit_max_ms
        if base - float(self.value_x(high)) <= tolerance:
            return high
        for _ in range(60):
            middle = 0.5 * (low + high)
            if base - float(self.value_x(middle)) <= tolerance:
                low = middle
            else:
                high = middle
        return low


def _task_counts(dose: DoseResponse, weights: Floats | None, task: int) -> tuple[Floats, Floats]:
    mask = np.asarray(dose.groups) == task
    w = np.ones(int(mask.sum())) if weights is None else weights[mask]
    return dose.outcomes[:, mask] @ w, np.full(dose.levels_ms.size, float(w.sum()))


def _fit_levels(dose: DoseResponse, spec: CurveSpec) -> tuple[Floats, npt.NDArray[np.bool_]]:
    x = effective_latency_ms(dose.levels_ms, spec.tick_ms)
    inside = x <= spec.fit_max_ms + 1e-9
    if np.unique(x[inside]).size != int(inside.sum()):
        raise PredictionError("two dose-response levels fall on the same control tick")
    if int(inside.sum()) < 4:
        raise PredictionError("the smooth fit needs at least four levels inside the budget")
    return x, inside


def fit_dose_curve(
    dose: DoseResponse,
    spec: CurveSpec,
    weights: Floats | None = None,
    families: Mapping[int, str] | None = None,
    warm: DoseCurve | None = None,
) -> DoseCurve:
    """Fit every task. ``families`` fixes the model family per task (bootstrap draws), otherwise the
    goodness-of-fit rule decides; ``warm`` provides an extra starting point (the full grid always runs)."""
    x, inside = _fit_levels(dose, spec)
    tasks: dict[int, TaskFunction] = {}
    chosen: dict[int, str] = {}
    for task in dose.group_ids:
        wins, trials = _task_counts(dose, weights, task)
        xs, ws, ts = x[inside], wins[inside], trials[inside]
        family = None if families is None else families[task]
        if family == "isotonic":
            tasks[task], chosen[task] = isotonic_task_curve(xs, ws, ts), "isotonic"
            continue
        start = warm.tasks.get(task) if warm is not None else None
        curve, _ = fit_task_curve(xs, ws, ts, start=start if isinstance(start, TaskCurve) else None)
        if family is None and goodness_of_fit(curve, xs, ws, ts)[2] < spec.gof_alpha:
            tasks[task], chosen[task] = isotonic_task_curve(xs, ws, ts), "isotonic"
        else:
            tasks[task], chosen[task] = curve, "loglogistic"
    return DoseCurve(spec=spec, tasks=tasks, families=chosen)


def pooled_counts(doses: Sequence[DoseResponse], spec: CurveSpec) -> dict[int, tuple[Floats, Floats, Floats]]:
    """Per task: the effective latencies inside the fit range with successes and trials summed over samples.

    Each sample keeps its own seeds; levels of different samples on the same control tick are pooled.
    """
    if not doses:
        raise PredictionError("no dose-response sample to pool")
    table: dict[int, dict[float, list[float]]] = {}
    for dose in doses:
        x = effective_latency_ms(dose.levels_ms, spec.tick_ms)
        if np.unique(x).size != x.size:
            raise PredictionError("two levels of one sample fall on the same control tick")
        for task in dose.group_ids:
            wins, trials = _task_counts(dose, None, task)
            levels = table.setdefault(task, {})
            for latency, win, trial in zip(x, wins, trials, strict=True):
                if latency <= spec.fit_max_ms + 1e-9:
                    entry = levels.setdefault(float(latency), [0.0, 0.0])
                    entry[0] += float(win)
                    entry[1] += float(trial)
    counts: dict[int, tuple[Floats, Floats, Floats]] = {}
    for task in sorted(table):
        xs = np.array(sorted(table[task]), dtype=float)
        if xs.size < 4:
            raise PredictionError("the smooth fit needs at least four levels inside the budget")
        counts[task] = (
            xs,
            np.array([table[task][float(v)][0] for v in xs]),
            np.array([table[task][float(v)][1] for v in xs]),
        )
    return counts


def fit_pooled_curve(doses: Sequence[DoseResponse], spec: CurveSpec) -> DoseCurve:
    """The rule of :func:`fit_dose_curve` on several samples pooled level by level (sensitivity analyses)."""
    tasks: dict[int, TaskFunction] = {}
    chosen: dict[int, str] = {}
    for task, (xs, ws, ts) in pooled_counts(doses, spec).items():
        curve, _ = fit_task_curve(xs, ws, ts)
        if goodness_of_fit(curve, xs, ws, ts)[2] < spec.gof_alpha:
            tasks[task], chosen[task] = isotonic_task_curve(xs, ws, ts), "isotonic"
        else:
            tasks[task], chosen[task] = curve, "loglogistic"
    return DoseCurve(spec=spec, tasks=tasks, families=chosen)


def fit_report(dose: DoseResponse, curve: DoseCurve) -> dict[int, dict[str, float | str]]:
    """Parameters and goodness of fit of every task of a fit on all data."""
    x, inside = _fit_levels(dose, curve.spec)
    report: dict[int, dict[str, float | str]] = {}
    for task, function in curve.tasks.items():
        wins, trials = _task_counts(dose, None, task)
        params = 3 if isinstance(function, TaskCurve) else int(inside.sum())
        deviance, df, p_value = goodness_of_fit(function, x[inside], wins[inside], trials[inside], params)
        entry: dict[str, float | str] = {
            "family": curve.families.get(task, "loglogistic"),
            "deviance": deviance,
            "df": float(df),
            "p_value": p_value,
        }
        if isinstance(function, TaskCurve):
            entry.update({"p": function.p, "x50_ms": function.x50_ms, "h": function.h})
        report[task] = entry
    return report


# --------------------------------------------------------------------------------------------- bootstrap
def _interval(values: Floats, confidence: float, discrete: bool = False) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0 * 100.0
    if discrete:
        low = np.percentile(values, tail, method="lower")
        high = np.percentile(values, 100.0 - tail, method="higher")
    else:
        low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


@dataclass(frozen=True)
class BootstrapFit:
    """Fit on all seeds plus refits on stratified bootstrap draws of the seeds."""

    point: DoseCurve
    draws: tuple[DoseCurve, ...]

    def band(self, latency_ms: npt.ArrayLike, confidence: float = 0.95) -> tuple[Floats, Floats]:
        """Pointwise percentile band of the pooled curve at end-to-end latencies."""
        values = np.array([draw.value(latency_ms) for draw in self.draws])
        tail = (1.0 - confidence) / 2.0 * 100.0
        low, high = np.percentile(values, [tail, 100.0 - tail], axis=0)
        return np.asarray(low, dtype=float), np.asarray(high, dtype=float)

    def band_x(self, x_ms: npt.ArrayLike, confidence: float = 0.95) -> tuple[Floats, Floats]:
        values = np.array([draw.value_x(x_ms) for draw in self.draws])
        tail = (1.0 - confidence) / 2.0 * 100.0
        low, high = np.percentile(values, [tail, 100.0 - tail], axis=0)
        return np.asarray(low, dtype=float), np.asarray(high, dtype=float)

    def tolerable_draws(self, tolerance: float) -> Floats:
        return np.array([draw.tolerable_latency(tolerance)[0] for draw in self.draws])

    def tolerable(self, tolerance: float, confidence: float = 0.95) -> TolerableLatency:
        point, censored = self.point.tolerable_latency(tolerance)
        low, high = _interval(self.tolerable_draws(tolerance), confidence, discrete=True)
        return TolerableLatency(
            point_ms=point,
            conservative_ms=low,
            upper_ms=high,
            censored=censored,
            x_point_ms=self.point.tolerable_x(tolerance),
        )

    def loss_interval(self, latency_ms: float, confidence: float = 0.95) -> tuple[float, tuple[float, float]]:
        """Loss against 0 ms at ``latency_ms`` (point) and its bootstrap interval."""
        point = float(self.point.loss(latency_ms))
        values = np.array([float(draw.loss(latency_ms)) for draw in self.draws])
        return point, _interval(values, confidence)


@dataclass(frozen=True)
class TolerableLatency:
    point_ms: float
    conservative_ms: float  # lower bound of the bootstrap interval
    upper_ms: float
    censored: bool  # point estimate at the budget: L* >= budget
    x_point_ms: float  # continuous effective latency where the smooth curve has lost the tolerance


def bootstrap_fit(dose: DoseResponse, spec: CurveSpec, multiplicities: Floats) -> BootstrapFit:
    """Point fit plus one refit per row of ``multiplicities`` (shape (draws, seeds)).

    Every refit starts from the point fit and from the full grid of starting points.
    """
    if multiplicities.ndim != 2 or multiplicities.shape[1] != len(dose.seeds) or multiplicities.shape[0] < 1:
        raise PredictionError("multiplicities must have shape (draws >= 1, seeds)")
    point = fit_dose_curve(dose, spec)
    draws = tuple(
        fit_dose_curve(dose, spec, weights=row, families=point.families, warm=point) for row in multiplicities
    )
    return BootstrapFit(point=point, draws=draws)


def paired_tolerable_difference(
    first: BootstrapFit, second: BootstrapFit, tolerance: float, confidence: float = 0.95
) -> tuple[float, tuple[float, float]]:
    """``L*(second) - L*(first)`` and its interval; both fits must use the same bootstrap multiplicities."""
    if len(first.draws) != len(second.draws):
        raise PredictionError("paired fits need the same number of bootstrap draws")
    point = second.point.tolerable_latency(tolerance)[0] - first.point.tolerable_latency(tolerance)[0]
    values = second.tolerable_draws(tolerance) - first.tolerable_draws(tolerance)
    return point, _interval(values, confidence, discrete=True)


def mean_success_difference(
    first: BootstrapFit, second: BootstrapFit, latencies_ms: Floats, confidence: float = 0.95
) -> tuple[float, tuple[float, float]]:
    """Mean of ``second - first`` over ``latencies_ms`` (pooled curves) with its paired interval."""
    if len(first.draws) != len(second.draws):
        raise PredictionError("paired fits need the same number of bootstrap draws")
    point = float(np.mean(second.point.value(latencies_ms) - first.point.value(latencies_ms)))
    values = np.array(
        [
            float(np.mean(b.value(latencies_ms) - a.value(latencies_ms)))
            for a, b in zip(first.draws, second.draws, strict=True)
        ]
    )
    return point, _interval(values, confidence)
