"""Measurement records, JSONL persistence and the statistics used by the report.

Definitions (used consistently by agent, report and README):

* **E2E latency**     ``t_resp - t_obs``: observation captured -> action chunk available on the robot.
* **Reaction time**   ``t_first_exec - t_obs``: observation captured -> first action from that chunk executed
                      (includes waiting for the next control tick).
* **Network RTT**     ``(t_resp - t_send) - server_total``: time on the wire (proxy delays, stalls, stack).
* **Jitter**          mean ``|ΔRTT|`` of consecutive requests within an episode (IPDV, RFC 5481), reported
                      next to the standard deviation of the network RTT.
* **Starvation**      a control tick with no fresh action; the robot repeats its last action. A *starvation
                      event* is a maximal run of consecutive starved ticks.
* **Starvation cause** ``compute`` if the pending request would have been late even with zero network time
                      (send time + server time beyond the tick), otherwise ``network``; ``none_pending`` if no
                      request was in flight (only after failed requests).
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------------- records


@dataclass
class RequestRecord:
    """One Infer call as seen by the agent. Times are seconds relative to the episode clock start t0."""

    run: str
    profile: str
    seed: int
    attempt: int
    request_id: int
    phase: str  # "warmup" | "realtime" | "sync"
    status: str  # "ok" | "timeout" | "unavailable" | "error" | "cancelled"
    obs_timestep: int
    t_obs_s: float
    t_send_s: float
    request_bytes: int
    queue_at_send: int
    error: str | None = None
    t_resp_s: float | None = None
    t_first_exec_s: float | None = None
    response_bytes: int | None = None
    server_queue_ms: float | None = None
    server_inference_ms: float | None = None
    server_total_ms: float | None = None
    queue_at_merge: int | None = None
    n_actions: int = 0
    n_stale: int = 0
    n_used: int = 0
    starved_ticks_while_pending: int = 0

    @property
    def e2e_ms(self) -> float | None:
        return None if self.t_resp_s is None else 1000.0 * (self.t_resp_s - self.t_obs_s)

    @property
    def rtt_ms(self) -> float | None:
        return None if self.t_resp_s is None else 1000.0 * (self.t_resp_s - self.t_send_s)

    @property
    def network_ms(self) -> float | None:
        rtt = self.rtt_ms
        if rtt is None or self.server_total_ms is None:
            return None
        return rtt - self.server_total_ms

    @property
    def reaction_ms(self) -> float | None:
        if self.t_first_exec_s is None:
            return None
        return 1000.0 * (self.t_first_exec_s - self.t_obs_s)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            e2e_ms=self.e2e_ms, rtt_ms=self.rtt_ms, network_ms=self.network_ms, reaction_ms=self.reaction_ms
        )
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RequestRecord:
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclass
class TickSeries:
    """Per control tick (index = control step) of one episode."""

    queue_len: list[int] = field(default_factory=list)  # actions available for this or later steps
    starved: list[int] = field(default_factory=list)  # 1 if the last action had to be repeated
    lateness_ms: list[float] = field(default_factory=list)  # tick start minus its deadline
    action_age: list[int] = field(default_factory=list)  # control step minus obs step of the chunk
    pending_request: list[int] = field(default_factory=list)  # request id in flight (-1 = none)


@dataclass
class EpisodeRecord:
    run: str
    profile: str
    scenario: str
    mode: str
    seed: int
    attempt: int
    success: bool
    steps: int
    fps: float
    wall_duration_s: float
    max_coverage: float
    sum_reward: float
    n_requests: int
    n_failed_requests: int
    starved_ticks: int
    starvation_events: int
    starved_ticks_compute: int
    starved_ticks_network: int
    starved_ticks_none_pending: int
    tick_overruns: int
    max_tick_lateness_ms: float
    valid: bool
    invalid_reasons: list[str] = field(default_factory=list)
    warmup_ms: float | None = None
    cpu_steal_fraction: float | None = None
    proxy: dict[str, Any] | None = None
    ticks: TickSeries = field(default_factory=TickSeries)

    @property
    def duration_s(self) -> float:
        """Simulated episode duration (equals wall-clock time in real-time mode)."""
        return self.steps / self.fps

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_s"] = self.duration_s
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EpisodeRecord:
        fields = cls.__dataclass_fields__
        kwargs = {key: value for key, value in data.items() if key in fields}
        kwargs["ticks"] = TickSeries(**data.get("ticks", {}))
        return cls(**kwargs)


# --------------------------------------------------------------------------------------------- JSONL


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield record


# --------------------------------------------------------------------------------------------- statistics


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy default); ``None`` for an empty sequence."""
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean(values: Sequence[float]) -> float | None:
    return None if len(values) == 0 else float(np.mean(np.asarray(values, dtype=np.float64)))


def std(values: Sequence[float]) -> float | None:
    return None if len(values) < 2 else float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def ipdv(values: Sequence[float]) -> list[float]:
    """Absolute differences of consecutive values (instantaneous packet delay variation)."""
    return [abs(b - a) for a, b in itertools.pairwise(values)]


def runs_of_true(flags: Sequence[int | bool]) -> list[tuple[int, int]]:
    """Maximal runs of truthy flags as ``(start_index, length)``."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(flags) - start))
    return runs


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % by default)."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} must be within [0, n={n}]")
    p_hat = successes / n
    denominator = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs (binomial test with p = 0.5)."""
    if only_a < 0 or only_b < 0:
        raise ValueError("counts must be non-negative")
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def paired_outcomes(a: dict[int, bool], b: dict[int, bool]) -> tuple[int, int, int]:
    """For seeds present in both runs: (#pairs, #success only in a, #success only in b)."""
    common = sorted(set(a) & set(b))
    only_a = sum(1 for s in common if a[s] and not b[s])
    only_b = sum(1 for s in common if b[s] and not a[s])
    return len(common), only_a, only_b


def bootstrap_ci(
    groups: Sequence[Sequence[float]],
    statistic: Callable[[Sequence[float]], float | None],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI of ``statistic`` resampling whole groups (episodes) with replacement.

    Requests within an episode are correlated; resampling episodes keeps that structure.
    """
    non_empty = [list(g) for g in groups if len(g) > 0]
    if len(non_empty) < 2:
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(len(non_empty)):
            sample.extend(non_empty[rng.randrange(len(non_empty))])
        value = statistic(sample)
        if value is not None:
            estimates.append(value)
    if not estimates:
        return None
    low = float(np.percentile(estimates, 100 * alpha / 2))
    high = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return (low, high)


# --------------------------------------------------------------------------------------------- summary


@dataclass
class RunSummary:
    """Aggregates of one run (= one scenario label within one Brain profile)."""

    run: str
    profile: str
    scenario: str
    mode: str
    episodes: int
    invalid_attempts: int
    successes: int
    success_rate: float
    success_ci: tuple[float, float]
    requests: int
    e2e_p50_ms: float | None
    e2e_p95_ms: float | None
    e2e_p99_ms: float | None
    e2e_p95_ci_ms: tuple[float, float] | None
    reaction_p50_ms: float | None
    reaction_p95_ms: float | None
    network_p50_ms: float | None
    network_p95_ms: float | None
    network_p99_ms: float | None
    server_inference_p50_ms: float | None
    server_total_p95_ms: float | None
    jitter_ipdv_ms: float | None
    network_std_ms: float | None
    starvation_events_per_min: float
    starved_ticks_per_min: float
    starved_time_fraction: float
    starved_ticks_compute: int
    starved_ticks_network: int
    starved_ticks_none_pending: int
    request_bytes_mean: float | None
    response_bytes_mean: float | None
    wire_bytes_up_per_request: float | None
    wire_bytes_down_per_request: float | None
    requests_per_min: float
    mean_action_age_ticks: float | None
    failed_requests: int
    tick_overruns: int
    max_tick_lateness_ms: float
    episode_seconds: float
    success_by_seed: dict[int, bool] = field(default_factory=dict)


def final_episodes(episodes: Iterable[EpisodeRecord]) -> tuple[list[EpisodeRecord], int]:
    """Pick one record per seed: the first valid attempt (else the last attempt).

    Returns the selected episodes and the number of discarded invalid attempts.
    """
    by_seed: dict[int, list[EpisodeRecord]] = {}
    for episode in episodes:
        by_seed.setdefault(episode.seed, []).append(episode)
    selected: list[EpisodeRecord] = []
    invalid = 0
    for seed in sorted(by_seed):
        attempts = sorted(by_seed[seed], key=lambda e: e.attempt)
        valid = [e for e in attempts if e.valid]
        invalid += sum(1 for e in attempts if not e.valid)
        selected.append(valid[0] if valid else attempts[-1])
    return selected, invalid


def summarize_run(episodes: Sequence[EpisodeRecord], requests: Sequence[RequestRecord]) -> RunSummary:
    if not episodes:
        raise ValueError("cannot summarize a run without episodes")
    chosen, invalid_attempts = final_episodes(episodes)
    keys = {(e.seed, e.attempt) for e in chosen}
    measured = [
        r
        for r in requests
        if (r.seed, r.attempt) in keys and r.phase in ("realtime", "sync") and r.status == "ok"
    ]
    failed = sum(1 for r in requests if (r.seed, r.attempt) in keys and r.status != "ok")

    def values(attribute: str, subset: Sequence[RequestRecord] = measured) -> list[float]:
        out: list[float] = []
        for record in subset:
            value = getattr(record, attribute)
            if value is not None:
                out.append(float(value))
        return out

    e2e = values("e2e_ms")
    by_episode: dict[tuple[int, int], list[RequestRecord]] = {}
    for record in measured:
        by_episode.setdefault((record.seed, record.attempt), []).append(record)
    e2e_groups = [values("e2e_ms", group) for group in by_episode.values()]
    jitter: list[float] = []
    for group in by_episode.values():
        ordered = sorted(group, key=lambda r: r.request_id)
        jitter.extend(ipdv(values("network_ms", ordered)))

    successes = sum(1 for e in chosen if e.success)
    n = len(chosen)
    episode_seconds = sum(e.duration_s for e in chosen)
    minutes = episode_seconds / 60.0 if episode_seconds > 0 else math.nan
    starved_ticks = sum(e.starved_ticks for e in chosen)
    total_ticks = sum(e.steps for e in chosen)
    ages = [float(age) for e in chosen for age in e.ticks.action_age if age >= 0]
    wire_up = [float(e.proxy["bytes_up"]) for e in chosen if e.proxy]
    wire_down = [float(e.proxy["bytes_down"]) for e in chosen if e.proxy]
    n_requests_proxy = sum(e.n_requests for e in chosen if e.proxy)
    first = chosen[0]
    return RunSummary(
        run=first.run,
        profile=first.profile,
        scenario=first.scenario,
        mode=first.mode,
        episodes=n,
        invalid_attempts=invalid_attempts,
        successes=successes,
        success_rate=successes / n,
        success_ci=wilson_interval(successes, n),
        requests=len(measured),
        e2e_p50_ms=percentile(e2e, 50),
        e2e_p95_ms=percentile(e2e, 95),
        e2e_p99_ms=percentile(e2e, 99),
        e2e_p95_ci_ms=bootstrap_ci(e2e_groups, lambda v: percentile(v, 95)),
        reaction_p50_ms=percentile(values("reaction_ms"), 50),
        reaction_p95_ms=percentile(values("reaction_ms"), 95),
        network_p50_ms=percentile(values("network_ms"), 50),
        network_p95_ms=percentile(values("network_ms"), 95),
        network_p99_ms=percentile(values("network_ms"), 99),
        server_inference_p50_ms=percentile(values("server_inference_ms"), 50),
        server_total_p95_ms=percentile(values("server_total_ms"), 95),
        jitter_ipdv_ms=mean(jitter),
        network_std_ms=std(values("network_ms")),
        starvation_events_per_min=sum(e.starvation_events for e in chosen) / minutes,
        starved_ticks_per_min=starved_ticks / minutes,
        starved_time_fraction=starved_ticks / total_ticks if total_ticks else 0.0,
        starved_ticks_compute=sum(e.starved_ticks_compute for e in chosen),
        starved_ticks_network=sum(e.starved_ticks_network for e in chosen),
        starved_ticks_none_pending=sum(e.starved_ticks_none_pending for e in chosen),
        request_bytes_mean=mean(values("request_bytes")),
        response_bytes_mean=mean(values("response_bytes")),
        wire_bytes_up_per_request=sum(wire_up) / n_requests_proxy if n_requests_proxy else None,
        wire_bytes_down_per_request=sum(wire_down) / n_requests_proxy if n_requests_proxy else None,
        requests_per_min=len(measured) / minutes,
        mean_action_age_ticks=mean(ages),
        failed_requests=failed,
        tick_overruns=sum(e.tick_overruns for e in chosen),
        max_tick_lateness_ms=max(e.max_tick_lateness_ms for e in chosen),
        episode_seconds=episode_seconds,
        success_by_seed={e.seed: e.success for e in chosen},
    )
