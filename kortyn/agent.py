"""Kortyn Agent: the simulated robot that offloads its policy to the Kortyn Brain.

Real-time contract (``mode: realtime``)
---------------------------------------
* The control loop runs at the environment's control rate with **absolute deadlines**
  (``t0 + k * dt``), so lateness never accumulates into drift.
* Every tick executes **exactly one** ``env.step``. The simulation never waits for the Brain.
* Tick ``k``:
  1. Merge chunks that arrived since the last tick. A chunk computed from the observation of step
     ``t`` holds actions for steps ``t .. t+N-1``; steps before ``k`` are stale and dropped, overlapping
     steps are replaced by the newer chunk.
  2. Execute the action for step ``k``. If there is none (**starvation**), repeat the last action
     (PushT: hold the last position target) and count the tick.
  3. If no request is in flight and at most ``refill_threshold`` actions remain, send the observation
     history ``(o_{k-1}, o_k)`` labelled with step ``k`` asynchronously (gRPC future, never blocks).
  4. Step the simulation.
* Observation ``o_k`` is released at the start of tick ``k`` (its physical time), never earlier.

``mode: sync`` is the reference without real-time: the simulation waits for every chunk and executes
``sync_actions_per_chunk`` actions, which is LeRobot's evaluation semantics.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import queue
import socket
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import grpc
import numpy as np

from kortyn.codec import CodecError, decode_tensor, encode_observation
from kortyn.config import DEFAULT_BASE_CONFIG, AgentConfig, ConfigError, NetworkConfig, load_config
from kortyn.metrics import EpisodeRecord, RequestRecord, TickSeries, append_jsonl, runs_of_true
from kortyn.process_utils import read_cpu_times, steal_fraction
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.proto import kortyn_brain_pb2_grpc as pb_grpc
from kortyn.typing_utils import Array, FloatArray

LOG = logging.getLogger("kortyn.agent")

GRPC_MAX_MESSAGE_BYTES = 32 * 1024 * 1024
Observation = dict[str, Array]


class BrainUnavailableError(RuntimeError):
    """The Brain could not be reached (or did not answer GetInfo in time)."""


class IncompatibleBrainError(RuntimeError):
    """The Brain's policy does not fit this environment/agent configuration."""


class ProxyControlError(RuntimeError):
    """The netem proxy control port failed or rejected a command."""


class StepResultLike(Protocol):
    @property
    def observation(self) -> Observation: ...

    @property
    def reward(self) -> float: ...

    @property
    def terminated(self) -> bool: ...

    @property
    def truncated(self) -> bool: ...

    @property
    def success(self) -> bool: ...

    @property
    def coverage(self) -> float | None: ...


class Environment(Protocol):
    action_dim: int

    @property
    def fps(self) -> float: ...

    @property
    def max_episode_steps(self) -> int: ...

    def reset(self, seed: int) -> Observation: ...

    def step(self, action: FloatArray) -> StepResultLike: ...

    def hold_action(self, observation: Observation) -> FloatArray: ...


# --------------------------------------------------------------------------------------------- clients
@dataclass(frozen=True)
class BrainSpec:
    """What the agent needs to know about the Brain's policy (from GetInfo)."""

    n_obs_steps: int
    max_actions: int
    action_dim: int
    features: dict[str, tuple[str, tuple[int, ...]]]
    sampler: str
    device: str
    policy_type: str
    warmup_inference_ms: float
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_proto(cls, info: pb.BrainInfo) -> BrainSpec:
        return cls(
            n_obs_steps=int(info.n_obs_steps),
            max_actions=int(info.max_actions),
            action_dim=int(info.action_dim),
            features={f.key: (f.kind, tuple(int(d) for d in f.shape)) for f in info.input_features},
            sampler=info.sampler,
            device=info.device,
            policy_type=info.policy_type,
            warmup_inference_ms=float(info.warmup_inference_ms),
            raw={
                "policy_type": info.policy_type,
                "checkpoint": info.checkpoint,
                "device": info.device,
                "sampler": info.sampler,
                "n_obs_steps": int(info.n_obs_steps),
                "max_actions": int(info.max_actions),
                "lerobot_version": info.lerobot_version,
                "torch_version": info.torch_version,
                "torch_threads": int(info.torch_threads),
                "warmup_inference_ms": float(info.warmup_inference_ms),
            },
        )


class BrainClient:
    """gRPC client for the Kortyn Brain (optionally through the netem proxy)."""

    def __init__(self, address: str) -> None:
        self.address = address
        self._channel = grpc.insecure_channel(
            address,
            options=[
                # The container/user environment may define HTTP(S)_PROXY; Brain traffic must stay local.
                ("grpc.enable_http_proxy", 0),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
                ("grpc.initial_reconnect_backoff_ms", 100),
                ("grpc.min_reconnect_backoff_ms", 100),
                ("grpc.max_reconnect_backoff_ms", 1000),
            ],
        )
        self._stub = pb_grpc.KortynBrainStub(self._channel)

    def get_info(self, timeout_s: float) -> pb.BrainInfo:
        try:
            info: pb.BrainInfo = self._stub.GetInfo(
                pb.GetInfoRequest(), timeout=timeout_s, wait_for_ready=True
            )
        except grpc.RpcError as exc:
            raise BrainUnavailableError(
                f"Brain at {self.address} not reachable: {_rpc_message(exc)}"
            ) from exc
        return info

    def infer(self, request: pb.InferRequest, timeout_s: float) -> pb.InferResponse:
        response: pb.InferResponse = self._stub.Infer(request, timeout=timeout_s)
        return response

    def infer_future(self, request: pb.InferRequest, timeout_s: float) -> Any:
        return self._stub.Infer.future(request, timeout=timeout_s)

    def close(self) -> None:
        self._channel.close()


class ProxyControl:
    """Blocking client for the netem proxy's JSON-lines control port."""

    def __init__(self, host: str, port: int, timeout_s: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self._buffer = b""

    def _connect(self) -> socket.socket:
        if self._socket is None:
            try:
                self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            except OSError as exc:
                raise ProxyControlError(f"proxy control {self.host}:{self.port} unreachable: {exc}") from exc
            self._buffer = b""
        return self._socket

    def call(self, message: dict[str, Any]) -> dict[str, Any]:
        sock = self._connect()
        try:
            sock.sendall((json.dumps(message) + "\n").encode("utf-8"))
            while b"\n" not in self._buffer:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ProxyControlError("proxy closed the control connection")
                self._buffer += chunk
        except OSError as exc:
            self.close()
            raise ProxyControlError(f"proxy control I/O failed: {exc}") from exc
        line, _, self._buffer = self._buffer.partition(b"\n")
        reply = json.loads(line.decode("utf-8"))
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise ProxyControlError(f"proxy rejected {message.get('cmd')!r}: {reply}")
        return reply

    def ping(self) -> dict[str, Any]:
        return self.call({"cmd": "ping"})

    def configure(self, network: NetworkConfig, seed: int) -> None:
        self.call({"cmd": "configure", "network": network.to_dict(), "seed": seed})

    def arm(self) -> None:
        self.call({"cmd": "arm"})

    def stats(self) -> dict[str, Any]:
        return self.call({"cmd": "stats"})

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None


def _rpc_message(exc: grpc.RpcError) -> str:
    code = exc.code() if hasattr(exc, "code") else None
    details = exc.details() if hasattr(exc, "details") else str(exc)
    return f"{code.name if code is not None else 'UNKNOWN'}: {details}"


def _status_from_rpc(exc: grpc.RpcError) -> str:
    code = exc.code() if hasattr(exc, "code") else None
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return "timeout"
    if code == grpc.StatusCode.UNAVAILABLE:
        return "unavailable"
    if code == grpc.StatusCode.CANCELLED:
        return "cancelled"
    return "error"


# --------------------------------------------------------------------------------------------- queue
class ActionQueue:
    """Actions indexed by control step. Newer chunks replace overlapping steps; past steps are dropped."""

    def __init__(self) -> None:
        self._entries: dict[int, tuple[FloatArray, int, int]] = {}
        self._newest_chunk_step = -1

    def merge(
        self, start_step: int, actions: FloatArray, request_id: int, current_step: int
    ) -> tuple[int, int]:
        """Merge a chunk whose first action is for ``start_step``. Returns ``(n_stale, n_merged)``."""
        if start_step < self._newest_chunk_step:
            return len(actions), 0  # an older chunk must never overwrite a newer one
        self._newest_chunk_step = start_step
        stale = 0
        merged = 0
        for offset, action in enumerate(actions):
            step = start_step + offset
            if step < current_step:
                stale += 1
                continue
            self._entries[step] = (np.asarray(action, dtype=np.float32), request_id, start_step)
            merged += 1
        for step in [s for s in self._entries if s < current_step]:
            del self._entries[step]
        return stale, merged

    def pop(self, step: int) -> tuple[FloatArray, int, int] | None:
        """Remove and return ``(action, request_id, chunk_start_step)`` for ``step``; drop older entries."""
        for old in [s for s in self._entries if s < step]:
            del self._entries[old]
        return self._entries.pop(step, None)

    def available(self, from_step: int) -> int:
        """Number of queued actions for ``from_step`` or later."""
        return sum(1 for s in self._entries if s >= from_step)

    def clear(self) -> None:
        self._entries.clear()
        self._newest_chunk_step = -1


# --------------------------------------------------------------------------------------------- episode
@dataclass(frozen=True)
class EpisodeContext:
    run: str
    profile: str
    scenario: str
    mode: str  # "realtime" | "sync"
    seed: int
    attempt: int = 0


@dataclass
class EpisodeOutcome:
    episode: EpisodeRecord
    requests: list[RequestRecord]


@dataclass
class _Completion:
    request_id: int
    t_resp: float
    response: pb.InferResponse | None
    status: str
    error: str | None


@dataclass
class _Pending:
    record: RequestRecord
    future: Any


def request_seed(episode_seed: int, obs_step: int) -> int:
    """Sampling seed for one request: identical observation step -> identical noise in every scenario."""
    return (episode_seed * 1_000_003 + obs_step) % (2**63)


class Agent:
    """Runs episodes against a Brain. One instance per process; not thread-safe."""

    def __init__(
        self,
        config: AgentConfig,
        env: Environment,
        brain: BrainClient,
        spec: BrainSpec,
        proxy: ProxyControl | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.env = env
        self.brain = brain
        self.spec = spec
        self.proxy = proxy
        self._clock = clock
        self._sleep = sleep
        self._next_request_id = 1
        self._inbox: queue.SimpleQueue[_Completion] = queue.SimpleQueue()
        self.dt = 1.0 / env.fps
        self._check_compatibility()

    # ------------------------------------------------------------------ setup
    def _check_compatibility(self) -> None:
        spec = self.spec
        if spec.action_dim != self.env.action_dim:
            raise IncompatibleBrainError(
                f"Brain policy outputs {spec.action_dim}-D actions, env expects {self.env.action_dim}-D"
            )
        if self.config.actions_per_chunk > spec.max_actions:
            raise IncompatibleBrainError(
                f"agent.actions_per_chunk={self.config.actions_per_chunk} exceeds the Brain maximum "
                f"{spec.max_actions}"
            )
        observation = self.env.reset(seed=0)
        for key, (kind, shape) in spec.features.items():
            if key not in observation:
                raise IncompatibleBrainError(f"policy needs {key!r}, env provides {sorted(observation)}")
            value = observation[key]
            expected = (shape[1], shape[2], shape[0]) if kind == "VISUAL" else shape
            if tuple(value.shape) != expected:
                raise IncompatibleBrainError(
                    f"{key}: env provides shape {tuple(value.shape)}, policy expects {expected}"
                )

    # ------------------------------------------------------------------ requests
    def _build_request(self, history: deque[Observation], step: int, episode_seed: int) -> pb.InferRequest:
        stacked = {key: np.stack([frame[key] for frame in history]) for key in self.spec.features}
        return pb.InferRequest(
            client_id=self.config.client_id,
            episode_id=episode_seed,
            request_id=self._take_request_id(),
            obs_timestep=step,
            seed=request_seed(episode_seed, step),
            max_actions=self.config.actions_per_chunk,
            observation=encode_observation(stacked),
        )

    def _take_request_id(self) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def _new_record(
        self,
        ctx: EpisodeContext,
        request: pb.InferRequest,
        phase: str,
        t_obs: float,
        t_send: float,
        queue_len: int,
    ) -> RequestRecord:
        return RequestRecord(
            run=ctx.run,
            profile=ctx.profile,
            seed=ctx.seed,
            attempt=ctx.attempt,
            request_id=int(request.request_id),
            phase=phase,
            status="pending",
            obs_timestep=int(request.obs_timestep),
            t_obs_s=t_obs,
            t_send_s=t_send,
            request_bytes=request.ByteSize(),
            queue_at_send=queue_len,
        )

    def _on_done(self, request_id: int, future: Any) -> None:
        """gRPC callback thread: only timestamp and hand over; all state changes happen in the loop."""
        t_resp = self._clock()
        try:
            response = future.result()
            self._inbox.put(_Completion(request_id, t_resp, response, "ok", None))
        except grpc.FutureCancelledError:
            self._inbox.put(_Completion(request_id, t_resp, None, "cancelled", "cancelled by agent"))
        except grpc.RpcError as exc:
            self._inbox.put(_Completion(request_id, t_resp, None, _status_from_rpc(exc), _rpc_message(exc)))

    @staticmethod
    def _fill_response(record: RequestRecord, response: pb.InferResponse) -> FloatArray:
        record.status = "ok"
        record.response_bytes = response.ByteSize()
        record.server_queue_ms = float(response.server_queue_ms)
        record.server_inference_ms = float(response.server_inference_ms)
        record.server_total_ms = float(response.server_total_ms)
        actions = decode_tensor(response.actions).astype(np.float32)
        record.n_actions = int(actions.shape[0])
        return actions

    def _blocking_request(
        self,
        ctx: EpisodeContext,
        history: deque[Observation],
        step: int,
        phase: str,
        timeout_s: float,
        queue_len: int,
    ) -> tuple[RequestRecord, FloatArray | None, int]:
        request = self._build_request(history, step, ctx.seed)
        t_obs = self._clock()
        record = self._new_record(ctx, request, phase, t_obs, t_obs, queue_len)
        try:
            response = self.brain.infer(request, timeout_s)
        except grpc.RpcError as exc:
            record.t_resp_s = self._clock()
            record.status = _status_from_rpc(exc)
            record.error = _rpc_message(exc)
            return record, None, step
        record.t_resp_s = self._clock()
        try:
            actions = self._fill_response(record, response)
        except CodecError as exc:
            record.status = "error"
            record.error = f"bad action tensor: {exc}"
            return record, None, step
        return record, actions, int(response.start_timestep)

    # ------------------------------------------------------------------ episodes
    def run_episode(self, ctx: EpisodeContext) -> EpisodeOutcome:
        gc.collect()  # outside the timed loop; the loop itself runs with gc disabled
        if ctx.mode == "sync":
            return self._run_sync(ctx)
        if ctx.mode == "realtime":
            return self._run_realtime(ctx)
        raise ValueError(f"unknown mode {ctx.mode!r}")

    def _new_history(self, observation: Observation) -> deque[Observation]:
        # Like LeRobot's populate_queues: the first observation fills the whole history.
        return deque([observation] * self.spec.n_obs_steps, maxlen=self.spec.n_obs_steps)

    def _run_realtime(self, ctx: EpisodeContext) -> EpisodeOutcome:
        cfg = self.config
        env = self.env
        dt = self.dt
        max_steps = self._max_steps()
        observation = env.reset(seed=ctx.seed)
        history = self._new_history(observation)
        actions = ActionQueue()
        records: dict[int, RequestRecord] = {}
        invalid: list[str] = []
        warmup_ms: float | None = None
        cpu_before = read_cpu_times()

        if cfg.start_mode == "warm":
            record, chunk, start_step = self._blocking_request(
                ctx, history, 0, "warmup", cfg.warmup_timeout_s, queue_len=0
            )
            records[record.request_id] = record
            assert record.t_resp_s is not None
            warmup_ms = 1000.0 * (record.t_resp_s - record.t_obs_s)
            if chunk is None:
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
                    cpu_before=cpu_before,
                )
            actions.merge(start_step, chunk, record.request_id, current_step=0)

        t0 = self._clock()
        if self.proxy is not None:
            self.proxy.arm()
        ticks = TickSeries()
        tick_starts: list[float] = []
        in_flight: _Pending | None = None
        last_action: FloatArray | None = None
        last_chunk_step = 0
        success = False
        max_coverage = 0.0
        sum_reward = 0.0
        steps = 0
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for k in range(max_steps):
                deadline = t0 + k * dt
                remaining_wait = deadline - self._clock()
                if remaining_wait > 0:
                    self._sleep(remaining_wait)
                tick_start = self._clock()
                tick_starts.append(tick_start)
                # The request in flight at the start of the tick is the one a starved tick waits for (even if
                # it arrives in this very tick with only stale actions).
                pending_id = -1 if in_flight is None else in_flight.record.request_id

                # 1. merge arrived chunks
                while True:
                    try:
                        completion = self._inbox.get_nowait()
                    except queue.Empty:
                        break
                    done = records.get(completion.request_id)
                    if done is None:
                        continue  # belongs to an earlier episode (should not happen: we drain per episode)
                    done.t_resp_s = completion.t_resp
                    if in_flight is not None and in_flight.record.request_id == completion.request_id:
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
                    done.queue_at_merge = actions.available(k)
                    done.n_stale, _ = actions.merge(
                        int(completion.response.start_timestep), chunk, done.request_id, current_step=k
                    )

                # 2. execute the action for step k, or starve
                queue_len = actions.available(k)
                entry = actions.pop(k)
                if entry is not None:
                    action, source_request, chunk_step = entry
                    source = records[source_request]
                    if source.t_first_exec_s is None:
                        source.t_first_exec_s = tick_start
                    source.n_used += 1
                    last_action = action
                    last_chunk_step = chunk_step
                    starved = 0
                    age = k - chunk_step
                else:
                    starved = 1
                    if pending_id in records:
                        records[pending_id].starved_ticks_while_pending += 1
                    if last_action is None:
                        action = env.hold_action(observation)
                        age = -1
                    else:
                        action = last_action
                        age = k - last_chunk_step

                # 3. request a new chunk early enough (never blocks the loop)
                if in_flight is None and actions.available(k + 1) <= cfg.refill_threshold:
                    request = self._build_request(history, k, ctx.seed)
                    t_send = self._clock()
                    record = self._new_record(
                        ctx, request, "realtime", tick_start, t_send, actions.available(k + 1)
                    )
                    records[record.request_id] = record
                    future = self.brain.infer_future(request, cfg.request_timeout_s)
                    future.add_done_callback(lambda f, rid=record.request_id: self._on_done(rid, f))
                    in_flight = _Pending(record, future)

                # 4. advance the simulation by exactly one control step
                result = env.step(action)
                observation = result.observation
                history.append(observation)
                steps = k + 1
                sum_reward += result.reward
                if result.coverage is not None:
                    max_coverage = max(max_coverage, result.coverage)
                ticks.queue_len.append(queue_len)
                ticks.starved.append(starved)
                ticks.lateness_ms.append(1000.0 * (tick_start - deadline))
                ticks.action_age.append(age)
                ticks.pending_request.append(pending_id)
                if result.success:
                    success = True
                    break
                if result.terminated or result.truncated:
                    break
        finally:
            if gc_was_enabled:
                gc.enable()
        wall = self._clock() - t0

        # Let the last request finish so no traffic of this episode leaks into the next one.
        if in_flight is not None:
            # The outcome (response or error) is recorded through the completion callback below.
            with contextlib.suppress(Exception):
                in_flight.future.result(timeout=cfg.request_timeout_s + 1.0)
        deadline_drain = self._clock() + 1.0
        while any(r.status == "pending" for r in records.values()) and self._clock() < deadline_drain:
            try:
                completion = self._inbox.get(timeout=0.05)
            except queue.Empty:
                continue
            late = records.get(completion.request_id)
            if late is None:
                continue
            late.t_resp_s = completion.t_resp
            if completion.response is not None:
                try:
                    self._fill_response(late, completion.response)
                except CodecError as exc:
                    late.status = "error"
                    late.error = f"bad action tensor: {exc}"
            else:
                late.status = completion.status
                late.error = completion.error
        for leftover in records.values():
            if leftover.status == "pending":
                leftover.status = "error"
                leftover.error = "no completion received"

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
            cpu_before=cpu_before,
        )

    def _run_sync(self, ctx: EpisodeContext) -> EpisodeOutcome:
        cfg = self.config
        env = self.env
        max_steps = self._max_steps()
        observation = env.reset(seed=ctx.seed)
        history = self._new_history(observation)
        records: dict[int, RequestRecord] = {}
        invalid: list[str] = []
        ticks = TickSeries()
        cpu_before = read_cpu_times()
        t0 = self._clock()
        success = False
        max_coverage = 0.0
        sum_reward = 0.0
        k = 0
        while k < max_steps and not success:
            record, chunk, start_step = self._blocking_request(
                ctx, history, k, "sync", cfg.request_timeout_s, queue_len=0
            )
            records[record.request_id] = record
            if chunk is None:
                invalid.append(f"sync request failed: {record.error}")
                break
            record.queue_at_merge = 0
            record.n_stale = 0
            done = False
            for offset in range(min(cfg.sync_actions_per_chunk, len(chunk))):
                if record.t_first_exec_s is None:
                    record.t_first_exec_s = self._clock()
                record.n_used += 1
                result = env.step(chunk[offset])
                history.append(result.observation)
                sum_reward += result.reward
                if result.coverage is not None:
                    max_coverage = max(max_coverage, result.coverage)
                ticks.queue_len.append(len(chunk) - offset)
                ticks.starved.append(0)
                ticks.lateness_ms.append(0.0)
                ticks.action_age.append(k - start_step)
                ticks.pending_request.append(-1)
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
        wall = self._clock() - t0
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
            cpu_before=cpu_before,
        )

    def _max_steps(self) -> int:
        return int(self.env.max_episode_steps)

    # ------------------------------------------------------------------ bookkeeping
    def _finish(
        self,
        ctx: EpisodeContext,
        *,
        records: dict[int, RequestRecord],
        ticks: TickSeries,
        steps: int,
        tick_starts: list[float],
        max_coverage: float,
        sum_reward: float,
        success: bool,
        wall: float,
        t0: float,
        invalid: list[str],
        warmup_ms: float | None,
        cpu_before: tuple[int, int] | None,
    ) -> EpisodeOutcome:
        cfg = self.config
        steal = steal_fraction(cpu_before, read_cpu_times())
        proxy_stats: dict[str, Any] | None = None
        if self.proxy is not None and ctx.mode == "realtime":
            try:
                proxy_stats = self.proxy.stats()
            except ProxyControlError as exc:
                invalid.append(f"proxy stats unavailable: {exc}")

        compute, network, none_pending = self._attribute_starvation(ticks, tick_starts, records)
        failed = [r for r in records.values() if r.status != "ok"]
        for record in failed:
            if record.status in ("unavailable", "error"):
                invalid.append(f"request {record.request_id} {record.status}: {record.error}")
        max_lateness = max(ticks.lateness_ms, default=0.0)
        if max_lateness > cfg.max_tick_lateness_ms:
            invalid.append(f"tick lateness {max_lateness:.1f} ms > {cfg.max_tick_lateness_ms} ms")
        if proxy_stats is not None:
            lateness = float(proxy_stats.get("max_delivery_lateness_ms", 0.0))
            if lateness > cfg.max_proxy_lateness_ms:
                invalid.append(f"proxy delivery lateness {lateness:.1f} ms > {cfg.max_proxy_lateness_ms} ms")
            if int(proxy_stats.get("connections_opened_since_configure", 0)) > 0:
                invalid.append("gRPC connection was (re)opened during the episode")
        if steal is not None and steal > cfg.max_cpu_steal_fraction:
            invalid.append(f"CPU steal {steal:.1%} > {cfg.max_cpu_steal_fraction:.0%}")

        # Convert absolute monotonic times to seconds relative to the episode clock start.
        requests: list[RequestRecord] = []
        for record in sorted(records.values(), key=lambda r: r.request_id):
            record.t_obs_s -= t0
            record.t_send_s -= t0
            if record.t_resp_s is not None:
                record.t_resp_s -= t0
            if record.t_first_exec_s is not None:
                record.t_first_exec_s -= t0
            requests.append(record)

        episode = EpisodeRecord(
            run=ctx.run,
            profile=ctx.profile,
            scenario=ctx.scenario,
            mode=ctx.mode,
            seed=ctx.seed,
            attempt=ctx.attempt,
            success=success,
            steps=steps,
            fps=self.env.fps,
            wall_duration_s=wall,
            max_coverage=max_coverage,
            sum_reward=sum_reward,
            n_requests=len(records),
            n_failed_requests=len(failed),
            starved_ticks=int(sum(ticks.starved)),
            starvation_events=len(runs_of_true(ticks.starved)),
            starved_ticks_compute=compute,
            starved_ticks_network=network,
            starved_ticks_none_pending=none_pending,
            tick_overruns=sum(1 for value in ticks.lateness_ms if value > 1000.0 * self.dt),
            max_tick_lateness_ms=max_lateness,
            valid=not invalid,
            invalid_reasons=invalid,
            warmup_ms=warmup_ms,
            cpu_steal_fraction=steal,
            proxy=proxy_stats,
            ticks=ticks,
        )
        return EpisodeOutcome(episode=episode, requests=requests)

    @staticmethod
    def _attribute_starvation(
        ticks: TickSeries, tick_starts: list[float], records: dict[int, RequestRecord]
    ) -> tuple[int, int, int]:
        """Classify starved ticks: compute (late even without network), network, or nothing pending."""
        compute = network = none_pending = 0
        for index, starved in enumerate(ticks.starved):
            if not starved:
                continue
            request_id = ticks.pending_request[index]
            record = records.get(request_id)
            if record is None:
                none_pending += 1
                continue
            if record.status == "ok" and record.server_total_ms is not None:
                ready_without_network = record.t_send_s + record.server_total_ms / 1000.0
                if ready_without_network >= tick_starts[index]:
                    compute += 1
                else:
                    network += 1
            else:
                network += 1
        return compute, network, none_pending


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    """Run a few episodes against an already running Brain (optionally behind the proxy)."""
    from kortyn.sim_env import SimEnv

    parser = argparse.ArgumentParser(description="Kortyn Agent: real-time simulated robot.")
    parser.add_argument("--config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument(
        "--brain", default=None, help="HOST:PORT to send requests to (default: proxy address)"
    )
    parser.add_argument(
        "--proxy-control", default=None, help="HOST:PORT of the proxy control port (optional)"
    )
    parser.add_argument("--mode", choices=["realtime", "sync"], default="realtime")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default=None, help="directory for episodes.jsonl/requests.jsonl")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        LOG.error("%s", exc)
        return 2
    address = args.brain or config.proxy.address
    brain = BrainClient(address)
    proxy: ProxyControl | None = None
    if args.proxy_control:
        host, _, port = args.proxy_control.rpartition(":")
        proxy = ProxyControl(host, int(port))
    env = SimEnv(config.env)
    try:
        spec = BrainSpec.from_proto(brain.get_info(timeout_s=60.0))
        agent = Agent(config.agent, env, brain, spec, proxy)
        for index in range(args.episodes):
            seed = args.seed + index
            ctx = EpisodeContext(
                run="manual", profile=spec.sampler, scenario="manual", mode=args.mode, seed=seed
            )
            outcome = agent.run_episode(ctx)
            e = outcome.episode
            LOG.info(
                "seed %d: success=%s steps=%d starved=%d requests=%d valid=%s %s",
                seed,
                e.success,
                e.steps,
                e.starved_ticks,
                e.n_requests,
                e.valid,
                "; ".join(e.invalid_reasons),
            )
            if args.out:
                out = Path(args.out)
                append_jsonl(out / "episodes.jsonl", [e.to_json()])
                append_jsonl(out / "requests.jsonl", [r.to_json() for r in outcome.requests])
    except (BrainUnavailableError, IncompatibleBrainError, ProxyControlError) as exc:
        LOG.error("%s", exc)
        return 1
    finally:
        env.close()
        brain.close()
        if proxy is not None:
            proxy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
