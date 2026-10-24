"""Freshness engine (phase 2): keep the age of executed plans low without overloading the Brain.

Plan age = time between the observation an action was computed from and the execution of that action.
Phase 1 showed that quality degrades once the mean plan age exceeds ~0.75 s, whether network or compute
causes it. This module holds the agent-side decisions, all pure and unit-tested (no torch, no I/O):

* **When to request** (:class:`ThresholdPolicy`, :class:`AdaptivePolicy`). With one request in flight and
  a request period of ``P`` ticks, a plan that becomes usable ``D`` ticks after its observation is executed
  at ages ``D .. D+P-1``, i.e. a mean plan age of ``D + (P - 1) / 2`` ticks. The threshold policy fixes
  ``P`` indirectly (phase 1: refill at <= 6 actions -> ``P = 8``). The adaptive policy derives ``P`` from
  the online latency estimate so that the expected mean age stays at a target, and additionally requests
  early enough that the next plan arrives before the queue runs dry.
* **How to align a new plan** (:func:`placement_step`). Actions are keyed by control step; steps that passed
  while the plan was computed are skipped instead of executed late (phase-1 behaviour, now explicit).
* **How to smooth the transition** (:func:`build_guide`). For inpainting, the agent tells the Brain which
  actions of the old plan will certainly be executed before the new plan arrives (frozen) and which part
  of the old plan should softly guide the overlap. The weights follow LeRobot's real-time-chunking
  schedules (``RTCProcessor.get_prefix_weights``).
* **When an action is unusable** (:func:`plan_is_usable`): older than the guard's hard limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from kortyn.config import GuardConfig, RequestConfig, SmoothingConfig
from kortyn.latency import LatencyEstimator
from kortyn.typing_utils import FloatArray


def ticks_until_usable(latency_s: float, send_offset_s: float, dt: float) -> int:
    """Ticks from the sending tick until a chunk with this latency is merged (first tick start after it)."""
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    ticks = math.ceil((max(0.0, send_offset_s) + max(0.0, latency_s)) / dt - 1e-9)
    return max(1, ticks)


# --------------------------------------------------------------------------------------------- requests
@dataclass(frozen=True)
class RequestContext:
    """What the agent knows at tick ``step`` after choosing the action for that tick."""

    step: int
    available_after: int  # queued actions for steps after ``step``
    in_flight: int  # requests sent and not answered yet
    last_obs_step: int | None  # observation step of the newest plan requested so far (warm-up: 0)
    send_offset_s: float  # time since the start of this tick (a request now is sent this much later)


@dataclass(frozen=True)
class RequestDecision:
    send: bool
    reason: str  # "threshold" | "deadline" | "freshness" | "" (no request)
    expected_delay_ticks: int  # ticks until the plan requested now is usable (median latency)
    safe_delay_ticks: int  # the same for the latency quantile the deadline is planned with
    period_ticks: int | None = None  # adaptive: request period that keeps the expected mean age on target


class RequestPolicy(Protocol):
    def decide(self, ctx: RequestContext, latency: LatencyEstimator, dt: float) -> RequestDecision: ...


class ThresholdPolicy:
    """Phase 1: request when no request is in flight and at most ``threshold`` actions remain."""

    def __init__(self, threshold: int, latency_quantile_z: float = 1.645) -> None:
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        self.threshold = threshold
        self._z = latency_quantile_z

    def decide(self, ctx: RequestContext, latency: LatencyEstimator, dt: float) -> RequestDecision:
        expected = ticks_until_usable(latency.median_s, ctx.send_offset_s, dt)
        safe = ticks_until_usable(latency.quantile_s(self._z), ctx.send_offset_s, dt)
        send = ctx.in_flight == 0 and ctx.available_after <= self.threshold
        return RequestDecision(send, "threshold" if send else "", expected, safe)


class AdaptivePolicy:
    """Request on time (deadline) and often enough for the plan-age target (freshness), but not more often.

    * Deadline: the plan requested now must arrive before the queue runs dry. Sending at tick ``k`` it is
      usable at ``k + D_safe`` (latency quantile), so the queue must still cover ``k+1 .. k+D_safe-1``:
      request once ``available_after <= D_safe - 1 + safety_margin_ticks``.
    * Freshness: with expected delay ``D`` and request period ``P`` the mean plan age is
      ``D + (P - 1) / 2`` ticks; keeping it at or below the target ``A`` allows
      ``P <= 2 (A - D) + 1``. Request once the newest requested observation is ``P`` ticks old.

    With one request in flight the period cannot drop below ``D`` (requests then go back to back); the
    target is unreachable in that case, which the decision reports via ``period_ticks < expected_delay``.
    """

    def __init__(
        self, target_age_s: float, latency_quantile_z: float, safety_margin_ticks: int, dt: float
    ) -> None:
        if target_age_s <= 0 or dt <= 0:
            raise ValueError("target_age_s and dt must be > 0")
        self.target_age_ticks = target_age_s / dt
        self._z = latency_quantile_z
        self._margin = safety_margin_ticks

    def period_ticks(self, expected_delay_ticks: int) -> int:
        return max(1, math.floor(2.0 * (self.target_age_ticks - expected_delay_ticks) + 1.0 + 1e-9))

    def decide(self, ctx: RequestContext, latency: LatencyEstimator, dt: float) -> RequestDecision:
        expected = ticks_until_usable(latency.median_s, ctx.send_offset_s, dt)
        safe = ticks_until_usable(latency.quantile_s(self._z), ctx.send_offset_s, dt)
        period = self.period_ticks(expected)
        if ctx.in_flight > 0:
            return RequestDecision(False, "", expected, safe, period)
        if ctx.available_after <= safe - 1 + self._margin:
            return RequestDecision(True, "deadline", expected, safe, period)
        if ctx.last_obs_step is None or ctx.step - ctx.last_obs_step >= period:
            return RequestDecision(True, "freshness", expected, safe, period)
        return RequestDecision(False, "", expected, safe, period)


def make_request_policy(config: RequestConfig, default_threshold: int, dt: float) -> RequestPolicy:
    if config.mode == "threshold":
        threshold = default_threshold if config.refill_threshold is None else config.refill_threshold
        return ThresholdPolicy(threshold, config.latency_quantile_z)
    return AdaptivePolicy(config.target_age_s, config.latency_quantile_z, config.safety_margin_ticks, dt)


# --------------------------------------------------------------------------------------------- alignment
def placement_step(obs_step: int, merge_step: int, alignment: str) -> int:
    """Control step at which a chunk computed from ``obs_step`` starts in the queue.

    ``timestep``: actions keep their own control steps; those before ``merge_step`` already passed and are
    skipped. ``shift`` (naive): the chunk is replayed from its first action at the arrival tick, so stale
    actions are executed.
    """
    if alignment == "timestep":
        return obs_step
    if alignment == "shift":
        return max(obs_step, merge_step)
    raise ValueError(f"unknown alignment {alignment!r}")


# --------------------------------------------------------------------------------------------- smoothing
def overlap_weights(n_frozen: int, n_soft: int, schedule: str) -> FloatArray:
    """Weights ``1`` for the frozen part, then a decay over the soft overlap (LeRobot RTC schedules).

    ``linear``: ``linspace(1, 0, n_soft + 2)[1:-1]``; ``exp``: the linear weights scaled by
    ``expm1(w) / (e - 1)``; ``zeros``: no soft overlap (hard inpainting of the frozen part only).
    """
    if n_frozen < 0 or n_soft < 0:
        raise ValueError("n_frozen and n_soft must be >= 0")
    decay: npt.NDArray[np.float64]
    if schedule == "zeros":
        decay = np.zeros(n_soft, dtype=np.float64)
    elif schedule in ("linear", "exp"):
        decay = np.linspace(1.0, 0.0, n_soft + 2, dtype=np.float64)[1:-1]
        if schedule == "exp":
            decay = decay * np.expm1(decay) / (math.e - 1.0)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return np.concatenate([np.ones(n_frozen, dtype=np.float32), decay.astype(np.float32)])


@dataclass(frozen=True)
class Guide:
    """Actions for consecutive control steps from ``start_step`` with weights in [0, 1] (1 = frozen)."""

    start_step: int
    actions: FloatArray  # (G, action_dim), env units
    weights: FloatArray  # (G,)
    frozen: int  # number of leading weight-1 entries (executed before the new plan can take over)

    @property
    def soft(self) -> int:
        return len(self.weights) - self.frozen


def build_guide(
    *,
    step: int,
    previous_action: FloatArray | None,
    current_action: FloatArray,
    queued: list[FloatArray],
    expected_delay_ticks: int,
    config: SmoothingConfig,
) -> Guide:
    """Guide for the plan requested at tick ``step`` (observation ``o_step``).

    Frozen: the action executed at ``step - 1`` (if any), the one executed at ``step``, and the queued
    actions for ``step+1 .. step+D-1`` that will run before the new plan is expected (``D`` ticks). Soft: the
    following queued actions of the old plan, at most ``max_overlap_steps``. ``queued`` must be contiguous
    and start at ``step + 1``.
    """
    if expected_delay_ticks < 1:
        raise ValueError(f"expected_delay_ticks must be >= 1, got {expected_delay_ticks}")
    committed = queued[: expected_delay_ticks - 1]
    soft = queued[expected_delay_ticks - 1 : expected_delay_ticks - 1 + config.max_overlap_steps]
    head = [] if previous_action is None else [previous_action]
    frozen_actions = [*head, current_action, *committed]
    actions = np.asarray([*frozen_actions, *soft], dtype=np.float32).reshape(
        len(frozen_actions) + len(soft), -1
    )
    weights = overlap_weights(len(frozen_actions), len(soft), config.schedule)
    start = step - 1 if previous_action is not None else step
    return Guide(start_step=start, actions=actions, weights=weights, frozen=len(frozen_actions))


# --------------------------------------------------------------------------------------------- guard
def plan_is_usable(age_ticks: int, dt: float, guard: GuardConfig) -> bool:
    """False if the action is older than the guard's hard limit (an emergency: the robot holds instead)."""
    if guard.max_plan_age_s is None:
        return True
    return age_ticks * dt <= guard.max_plan_age_s + 1e-9
