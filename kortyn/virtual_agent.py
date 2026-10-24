"""Kortyn Agent in simulation time (phase 3a): the real-time control loop on a virtual clock.

:class:`VirtualTimeAgent` runs the episode loop of ``Agent._run_realtime`` statement by statement, but every
"now" is the simulation clock of :mod:`kortyn.virtual_time` instead of ``time.monotonic``:

* tick ``k`` starts at ``t0 + k * dt`` (plus a modelled or replayed lateness) and first merges every plan
  delivered up to that instant, exactly like the real-time loop merges every completion that arrived
  before the tick started;
* a request leaves the robot ``client_overhead`` after its tick started and its plan is delivered whenever
  the channel says (latency model or replay); meanwhile the loop keeps popping actions, repeats the last
  one when the queue runs dry (starvation) and never waits;
* request decisions, alignment, inpainting guides, guards, records and the episode summary are the very same
  code as in real time (``kortyn.freshness``, :class:`~kortyn.agent.AgentCore`).

``mode: sync`` mirrors ``Agent._run_sync`` (the simulation waits for every plan). The policy runs in-process
(:class:`LocalBrain`) with the Brain's validation path; identical inputs give identical plans, so a whole
experiment is reproducible bit for bit.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict, deque
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from kortyn.agent import (
    ActionQueue,
    AgentCore,
    BrainSpec,
    Environment,
    EpisodeContext,
    EpisodeOutcome,
    IncompatibleBrainError,
)
from kortyn.brain import BrainServicer, policy_info_to_proto
from kortyn.codec import CodecError, decode_observation
from kortyn.config import AgentConfig, ConfigError, NetworkConfig, StrategyConfig, TimingConfig
from kortyn.freshness import RequestContext, build_guide, make_request_policy, placement_step, plan_is_usable
from kortyn.latency import LatencyEstimator
from kortyn.metrics import RequestRecord, TickSeries
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.typing_utils import FloatArray
from kortyn.virtual_time import (
    Channel,
    Command,
    Compute,
    ComputeResult,
    Delivery,
    ModelChannel,
    ReplayChannel,
    ReplayLog,
    ReplayTicks,
    SampledTicks,
    Signal,
    Simulation,
    TickTiming,
    TimingSamplers,
    VirtualBrain,
    VirtualLink,
    WakeAt,
    load_delta,
    make_scheduler,
)

if TYPE_CHECKING:
    from kortyn.policy_runner import PolicyRunner

LOG = logging.getLogger("kortyn.virtual")

# Diagnostics in EpisodeRecord.virtual that depend on the machine or on which worker ran which episode
# before (inference cache); everything else of a virtual-time episode is deterministic.
NONDETERMINISTIC_KEYS = ("compute_s", "inference_ms", "cache_hits")


# --------------------------------------------------------------------------------------------- brain
class AgentCompute(Compute, Protocol):
    """The in-process Brain as the agent sees it: computes plans and counts inference-cache hits."""

    cache_hits: int


class LocalBrain:
    """In-process Brain for the virtual-time mode: the Brain's request validation and inference, no gRPC.

    Plans are memoized by request content (observation bytes, sampling seed, chunk length, guide): the
    policy is deterministic for fixed inputs, so episodes of different conditions that share a seed and
    have not diverged yet reuse each other's plans. Results are identical with or without the cache.
    """

    def __init__(self, runner: PolicyRunner, cache_entries: int = 50_000) -> None:
        self._runner = runner
        self._info = policy_info_to_proto(runner.info)
        self._cache: OrderedDict[bytes, FloatArray] = OrderedDict()
        self._cache_entries = cache_entries
        self.inferences = 0
        self.cache_hits = 0

    @property
    def spec(self) -> BrainSpec:
        return BrainSpec.from_proto(self._info)

    @property
    def info(self) -> pb.BrainInfo:
        return self._info

    @staticmethod
    def _key(request: pb.InferRequest, max_actions: int) -> bytes:
        digest = hashlib.sha256()
        for name in sorted(request.observation):
            tensor = request.observation[name]
            digest.update(name.encode("utf-8"))
            digest.update(tensor.dtype.encode("utf-8"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.data)
        digest.update(np.asarray([request.seed, max_actions], dtype=np.uint64).tobytes())
        digest.update(request.task.encode("utf-8"))
        if request.HasField("guide_actions"):
            offset = int(request.guide_start_timestep) - int(request.obs_timestep)
            digest.update(np.asarray([offset], dtype=np.int64).tobytes())
            digest.update(request.guide_actions.SerializeToString(deterministic=True))
            digest.update(np.asarray(request.guide_weights, dtype=np.float32).tobytes())
        return digest.digest()

    def __call__(self, request: pb.InferRequest) -> ComputeResult:
        from kortyn.policy_runner import ObservationError

        started = time.perf_counter()
        runner = self._runner
        try:
            observation = decode_observation(request.observation)
            runner.validate(observation)
            guide = BrainServicer._decode_guide(request)
            if guide is not None:
                runner.check_guide(guide)
        except (CodecError, ObservationError) as exc:
            return ComputeResult(None, f"INVALID_ARGUMENT: {exc}")
        max_actions = int(request.max_actions or self._info.max_actions)
        if max_actions > self._info.max_actions:
            return ComputeResult(
                None, f"INVALID_ARGUMENT: max_actions={max_actions} exceeds {self._info.max_actions}"
            )
        key = self._key(request, max_actions)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return ComputeResult(
                cached.copy(), compute_ms=1000.0 * (time.perf_counter() - started), cache_hit=True
            )
        try:
            result = runner.infer(
                observation,
                seed=int(request.seed),
                max_actions=max_actions,
                guide=guide,
                task=request.task or None,
            )
        except ObservationError as exc:
            return ComputeResult(None, f"INVALID_ARGUMENT: {exc}")
        self.inferences += 1
        actions = np.array(result.actions, dtype=np.float32, copy=True)
        self._cache[key] = actions
        if len(self._cache) > self._cache_entries:
            self._cache.popitem(last=False)
        return ComputeResult(actions.copy(), compute_ms=1000.0 * (time.perf_counter() - started))


# --------------------------------------------------------------------------------------------- agent
class VirtualTimeAgent(AgentCore):
    """Runs episodes in simulation time against an in-process Brain. One episode at a time."""

    def __init__(
        self,
        config: AgentConfig,
        env: Environment,
        spec: BrainSpec,
        compute: AgentCompute,
        timing: TimingConfig,
        samplers: TimingSamplers | None = None,
        replay: ReplayLog | None = None,
        timing_name: str = "",
    ) -> None:
        if timing.latency == "model" and (samplers is None or samplers.model is None):
            raise ConfigError("the model latency needs samplers for the service time and overheads")
        if timing.latency == "replay" and replay is None:
            raise ConfigError("the replay latency needs the logged run (ReplayLog)")
        super().__init__(config, env, spec)
        self.compute = compute
        self.timing = timing
        self.samplers = samplers
        self.replay = replay
        self.timing_name = timing_name
        self._episode_link_stats: dict[str, Any] | None = None

    def _link_stats(self, ctx: EpisodeContext, invalid: list[str]) -> dict[str, Any] | None:
        return self._episode_link_stats if ctx.mode == "realtime" else None

    # ------------------------------------------------------------------ episodes
    def run_episode(
        self,
        ctx: EpisodeContext,
        strategy: StrategyConfig | None = None,
        network: NetworkConfig | None = None,
    ) -> EpisodeOutcome:
        """Run one episode in virtual time; ``network`` is the scenario's link (real-time mode only)."""
        strategy = strategy or StrategyConfig()
        if (
            ctx.mode == "realtime"
            and strategy.smoothing.mode == "inpaint"
            and not self.spec.supports_guidance
        ):
            raise IncompatibleBrainError(
                f"strategy {ctx.strategy!r} inpaints transitions, but the Brain ({self.spec.policy_type}) "
                "does not support guides"
            )
        if ctx.mode not in ("realtime", "sync"):
            raise ValueError(f"unknown mode {ctx.mode!r}")
        started = time.perf_counter()
        hits_before = self.compute.cache_hits
        self._episode_link_stats = None
        # Request ids restart in every episode: a virtual-time episode must not depend on which episodes
        # the same worker happened to run before it.
        self._next_request_id = 1
        brain: VirtualBrain | None = None
        replay_channel: ReplayChannel | None = None
        channel: Channel
        ticks: TickTiming
        if self.timing.latency == "replay":
            assert self.replay is not None
            if ctx.mode != "realtime":
                raise ConfigError(
                    "replay plays back real-time episodes; run sync references with a model timing"
                )
            logged = self.replay.episode(ctx.seed)
            sim = Simulation(start=logged.warmup.t_obs_s if logged.warmup is not None else 0.0)
            replay_channel = ReplayChannel(sim, logged, self.compute)
            channel = replay_channel
            ticks = ReplayTicks(logged.tick_lateness_ms)
        else:
            assert self.samplers is not None
            assert self.samplers.model is not None
            sim = Simulation(start=0.0)
            brain = VirtualBrain(
                sim, self.compute, make_scheduler(self.timing.scheduler), self.timing.brain_servers
            )
            link_network = network if network is not None and ctx.mode == "realtime" else NetworkConfig()
            link = VirtualLink(sim, link_network, ctx.seed)
            channel = ModelChannel(
                sim,
                brain,
                link,
                self.samplers.model,
                ctx.seed,
                self.timing.timing_seed,
                self.config.client_id,
            )
            ticks = SampledTicks(self.samplers.tick, ctx.seed, self.timing.timing_seed, self.config.client_id)
        if ctx.mode == "sync":
            outcome = sim.run_process(self._sync(ctx, sim, channel))
        else:
            outcome = sim.run_process(self._realtime(ctx, strategy, sim, channel, ticks, brain))
        episode = outcome.episode
        episode.clock = "virtual"
        compute_ms = brain.compute_ms if brain is not None else 0.0
        if replay_channel is not None:
            compute_ms = replay_channel.compute_ms
        episode.virtual = {
            "timing": self.timing_name,
            "latency": self.timing.latency,
            "timing_seed": self.timing.timing_seed,
            "hardware": self.timing.hardware,
            "replay_run": str(self.replay.run_dir) if self.replay is not None else None,
            "replay_exact": replay_channel.counts.exact if replay_channel is not None else None,
            "replay_fallback": replay_channel.counts.fallback if replay_channel is not None else None,
            "events": sim.events_processed,
            "compute_s": time.perf_counter() - started,
            "inference_ms": compute_ms,
            "cache_hits": self.compute.cache_hits - hits_before,
        }
        return outcome

    def _realtime(
        self,
        ctx: EpisodeContext,
        strategy: StrategyConfig,
        sim: Simulation,
        channel: Channel,
        tick_timing: TickTiming,
        brain: VirtualBrain | None,
    ) -> Generator[Command, Any, EpisodeOutcome]:
        cfg = self.config
        env = self.env
        dt = self.dt
        max_steps = self._max_steps()
        policy = make_request_policy(strategy.request, cfg.refill_threshold, dt)
        latency = LatencyEstimator(strategy.latency)
        guard = strategy.guard
        inpaint = strategy.smoothing.mode == "inpaint"
        observation = env.reset(seed=ctx.seed)
        history = self._new_history(observation)
        actions = ActionQueue()
        records: dict[int, RequestRecord] = {}
        # Observation release time per request (simulation clock): plan age = execution - release.
        released: dict[int, float] = {}
        invalid: list[str] = []
        warmup_ms: float | None = None
        warmup_id: int | None = None
        last_requested_step: int | None = None
        inbox: deque[Delivery] = deque()
        arrived: dict[int, Signal] = {}
        blocking: set[int] = set()

        def deliver(delivery: Delivery) -> None:
            if delivery.request_id not in blocking:
                inbox.append(delivery)
            arrived[delivery.request_id].trigger(delivery)

        def send(request: pb.InferRequest, t_obs: float, t_send: float, timeout_s: float) -> None:
            arrived[int(request.request_id)] = Signal(sim)
            channel.send(request, t_obs, t_send, timeout_s, deliver)

        if cfg.start_mode == "warm":
            request = self._build_request(history, 0, ctx.seed)
            t_obs = sim.now
            record = self._new_record(ctx, request, "warmup", t_obs, t_obs, queue_len=0)
            records[record.request_id] = record
            blocking.add(record.request_id)
            send(request, t_obs, t_obs, cfg.warmup_timeout_s)
            delivery: Delivery = yield arrived[record.request_id]
            record.t_resp_s = delivery.t_resp
            warmup_ms = 1000.0 * (record.t_resp_s - record.t_obs_s)
            chunk = self._delivered_chunk(record, delivery)
            if chunk is None or delivery.response is None:
                invalid.append(f"warm-up request failed: {record.error}")
                return self._finish(
                    ctx,
                    records=records,
                    ticks=TickSeries(),
                    steps=0,
                    tick_starts=[],
                    max_coverage=0.0,
                    sum_reward=0.0,
                    success=False,
                    wall=0.0,
                    t0=record.t_obs_s,
                    invalid=invalid,
                    warmup_ms=warmup_ms,
                    cpu_before=None,
                    brain_load=None,
                )
            latency.observe(warmup_ms / 1000.0)
            start_step = int(delivery.response.start_timestep)
            actions.merge(
                placement_step(start_step, 0, strategy.alignment),
                chunk,
                record.request_id,
                current_step=0,
                obs_step=start_step,
            )
            warmup_id = record.request_id
            last_requested_step = start_step

        load_before = brain.counters.snapshot(sim.now) if brain is not None else None
        t0 = channel.episode_start(sim.now)
        channel.arm()
        if warmup_id is not None:
            released[warmup_id] = t0  # o_0 belongs to the start of the episode clock
        ticks = TickSeries()
        tick_starts: list[float] = []
        in_flight: int | None = None
        last_action: FloatArray | None = None
        last_chunk_step = 0
        last_source = -1
        previous_action: FloatArray | None = None
        success = False
        max_coverage = 0.0
        sum_reward = 0.0
        max_step_reward: float | None = None
        steps = 0
        for k in range(max_steps):
            deadline = t0 + k * dt
            tick_start = deadline + tick_timing.lateness_s(k)
            yield WakeAt(tick_start)
            tick_starts.append(tick_start)
            # The request in flight at the start of the tick is the one a starved tick waits for (even if
            # it arrives in this very tick with only stale actions).
            pending_id = -1 if in_flight is None else in_flight

            # 1. merge arrived chunks (aligned to their control steps)
            while inbox:
                completion = inbox.popleft()
                done = records.get(completion.request_id)
                if done is None:
                    continue
                done.t_resp_s = completion.t_resp
                if in_flight == completion.request_id:
                    in_flight = None
                if completion.response is None:
                    done.status = completion.status
                    done.error = completion.error
                    continue
                try:
                    chunk = self._fill_response(done, completion.response)
                except CodecError as exc:
                    done.status = "error"
                    done.error = f"bad action tensor: {exc}"
                    continue
                latency.observe(completion.t_resp - done.t_obs_s)
                obs_step = int(completion.response.start_timestep)
                done.queue_at_merge = actions.available(k)
                done.n_stale, _ = actions.merge(
                    placement_step(obs_step, k, strategy.alignment),
                    chunk,
                    done.request_id,
                    current_step=k,
                    obs_step=obs_step,
                )

            # 2. execute the action for step k; without a usable one: emergency (repeat or hold)
            queue_len = actions.available(k)
            entry = actions.pop(k)
            starved = 1 if entry is None else 0
            emergency = 1 if entry is None else 0
            if entry is not None and not plan_is_usable(k - entry[2], dt, guard):
                emergency = 2  # the plan exists but is older than the guard allows
                entry = None
            if entry is not None:
                action, source_request, chunk_step = entry
                source = records[source_request]
                if source.t_first_exec_s is None:
                    source.t_first_exec_s = tick_start
                source.n_used += 1
                last_action = action
                last_chunk_step = chunk_step
                last_source = source_request
                age = k - chunk_step
                plan_age_ms = 1000.0 * (tick_start - released[source_request])
            else:
                if starved and pending_id in records:
                    records[pending_id].starved_ticks_while_pending += 1
                if guard.mode == "repeat_last" and last_action is not None:
                    action = last_action
                    age = k - last_chunk_step
                    plan_age_ms = 1000.0 * (tick_start - released[last_source])
                else:
                    action = env.hold_action(observation)
                    age = -1
                    plan_age_ms = -1.0

            # 3. request a new plan when the strategy says so (the loop never waits for it)
            send_offset = channel.client_overhead_s(k)
            decision = policy.decide(
                RequestContext(
                    step=k,
                    available_after=actions.available(k + 1),
                    in_flight=0 if in_flight is None else 1,
                    last_obs_step=last_requested_step,
                    send_offset_s=send_offset,
                ),
                latency,
                dt,
            )
            if decision.send:
                guide = (
                    build_guide(
                        step=k,
                        previous_action=previous_action,
                        current_action=action,
                        queued=actions.peek(k + 1, cfg.actions_per_chunk),
                        expected_delay_ticks=decision.expected_delay_ticks,
                        config=strategy.smoothing,
                    )
                    if inpaint
                    else None
                )
                request = self._build_request(history, k, ctx.seed, guide)
                t_send = tick_start + send_offset
                record = self._new_record(
                    ctx, request, "realtime", tick_start, t_send, actions.available(k + 1)
                )
                record.trigger = decision.reason
                record.expected_delay_ticks = decision.expected_delay_ticks
                record.latency_median_ms, record.latency_spread_ms = latency.snapshot()
                if guide is not None:
                    record.guide_frozen = guide.frozen
                    record.guide_soft = guide.soft
                records[record.request_id] = record
                released[record.request_id] = tick_start
                last_requested_step = k
                send(request, tick_start, t_send, cfg.request_timeout_s)
                in_flight = record.request_id

            # 4. advance the simulation by exactly one control step
            result = env.step(action)
            previous_action = action
            observation = result.observation
            history.append(observation)
            steps = k + 1
            sum_reward += result.reward
            max_step_reward = (
                result.reward if max_step_reward is None else max(max_step_reward, result.reward)
            )
            if result.coverage is not None:
                max_coverage = max(max_coverage, result.coverage)
            ticks.queue_len.append(queue_len)
            ticks.starved.append(starved)
            ticks.lateness_ms.append(1000.0 * (tick_start - deadline))
            ticks.action_age.append(age)
            ticks.pending_request.append(pending_id)
            ticks.plan_age_ms.append(plan_age_ms)
            ticks.emergency.append(emergency)
            if result.success:
                success = True
                break
            if result.terminated or result.truncated:
                break
        wall = sim.now - t0

        # Let the last request finish so its record is complete (its plan is not executed any more).
        for request_id in sorted(records):
            signal = arrived.get(request_id)
            if records[request_id].status == "pending" and signal is not None and not signal.triggered:
                yield signal
        while inbox:
            late_delivery = inbox.popleft()
            late = records.get(late_delivery.request_id)
            if late is None or late.status != "pending":
                continue
            late.t_resp_s = late_delivery.t_resp
            self._delivered_chunk(late, late_delivery)
        for leftover in records.values():
            if leftover.status == "pending":
                leftover.status = "error"
                leftover.error = "no completion received"

        brain_load = (
            load_delta(load_before, brain.counters.snapshot(sim.now))
            if brain is not None and load_before is not None
            else None
        )
        self._episode_link_stats = channel.link_stats()
        return self._finish(
            ctx,
            records=records,
            ticks=ticks,
            steps=steps,
            tick_starts=tick_starts,
            max_coverage=max_coverage,
            sum_reward=sum_reward,
            success=success,
            wall=wall,
            t0=t0,
            invalid=invalid,
            warmup_ms=warmup_ms,
            cpu_before=None,
            brain_load=brain_load,
            max_step_reward=max_step_reward,
        )

    def _sync(
        self, ctx: EpisodeContext, sim: Simulation, channel: Channel
    ) -> Generator[Command, Any, EpisodeOutcome]:
        """``Agent._run_sync`` in virtual time: the simulation waits for every plan."""
        cfg = self.config
        env = self.env
        max_steps = self._max_steps()
        observation = env.reset(seed=ctx.seed)
        history = self._new_history(observation)
        records: dict[int, RequestRecord] = {}
        invalid: list[str] = []
        ticks = TickSeries()
        arrived: dict[int, Signal] = {}

        def deliver(delivery: Delivery) -> None:
            arrived[delivery.request_id].trigger(delivery)

        t0 = sim.now
        success = False
        max_coverage = 0.0
        sum_reward = 0.0
        max_step_reward: float | None = None
        k = 0
        while k < max_steps and not success:
            request = self._build_request(history, k, ctx.seed)
            t_obs = sim.now
            record = self._new_record(ctx, request, "sync", t_obs, t_obs, 0)
            records[record.request_id] = record
            arrived[record.request_id] = Signal(sim)
            channel.send(request, t_obs, t_obs, cfg.request_timeout_s, deliver)
            delivery: Delivery = yield arrived[record.request_id]
            record.t_resp_s = delivery.t_resp
            chunk = self._delivered_chunk(record, delivery)
            if chunk is None or delivery.response is None:
                invalid.append(f"sync request failed: {record.error}")
                break
            start_step = int(delivery.response.start_timestep)
            record.queue_at_merge = 0
            record.n_stale = 0
            done = False
            for offset in range(min(cfg.sync_actions_per_chunk, len(chunk))):
                if record.t_first_exec_s is None:
                    record.t_first_exec_s = sim.now
                record.n_used += 1
                result = env.step(chunk[offset])
                history.append(result.observation)
                sum_reward += result.reward
                max_step_reward = (
                    result.reward if max_step_reward is None else max(max_step_reward, result.reward)
                )
                if result.coverage is not None:
                    max_coverage = max(max_coverage, result.coverage)
                ticks.queue_len.append(len(chunk) - offset)
                ticks.starved.append(0)
                ticks.lateness_ms.append(0.0)
                ticks.action_age.append(k - start_step)
                ticks.pending_request.append(-1)
                # the simulation waits for the Brain: plan age in simulated time
                ticks.plan_age_ms.append(1000.0 * (k - start_step) * self.dt)
                ticks.emergency.append(0)
                k += 1
                if result.success:
                    success = True
                    done = True
                    break
                if result.terminated or result.truncated or k >= max_steps:
                    done = True
                    break
            if done:
                break
        wall = sim.now - t0
        return self._finish(
            ctx,
            records=records,
            ticks=ticks,
            steps=k,
            tick_starts=[],
            max_coverage=max_coverage,
            sum_reward=sum_reward,
            success=success,
            wall=wall,
            t0=t0,
            invalid=invalid,
            warmup_ms=None,
            cpu_before=None,
            brain_load=None,
            max_step_reward=max_step_reward,
        )

    def _delivered_chunk(self, record: RequestRecord, delivery: Delivery) -> FloatArray | None:
        """Fill ``record`` from a delivery; the action chunk, or ``None`` if the request failed."""
        if delivery.response is None:
            record.status = delivery.status
            record.error = delivery.error
            return None
        try:
            return self._fill_response(record, delivery.response)
        except CodecError as exc:
            record.status = "error"
            record.error = f"bad action tensor: {exc}"
            return None


def deterministic_view(episode: dict[str, Any]) -> dict[str, Any]:
    """An episode record without its machine-dependent diagnostics (for reproducibility checks)."""
    view = dict(episode)
    virtual = view.get("virtual")
    if isinstance(virtual, dict):
        view["virtual"] = {k: v for k, v in virtual.items() if k not in NONDETERMINISTIC_KEYS}
    return view
