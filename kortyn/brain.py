"""Kortyn Brain: gRPC inference server around a LeRobot policy.

Design:

* **Stateless per request.** Every ``Infer`` call carries the observation history; the server keeps no
  per-robot state, so any number of agents can share one Brain (phase 3).
* **One inference thread.** All torch work runs on a single dedicated worker thread (MPS is not
  thread-safe, and one model instance serializes anyway). gRPC handler threads only decode, validate and
  wait; the time a request waits for the worker is reported as ``server_queue_ms``.
* **Abandoned calls are skipped.** If the client's deadline expired while the request was queued, the
  worker does not run the model for it.
* **Explicit errors.** Malformed observations -> ``INVALID_ARGUMENT``; model failures -> ``INTERNAL``.

Run::

    python -m kortyn.brain --profile kalibriert
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import grpc

from kortyn.codec import CodecError, decode_observation, encode_array
from kortyn.config import DEFAULT_BASE_CONFIG, BrainConfig, BrainProfile, ConfigError, load_config
from kortyn.process_utils import set_cpu_affinity
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.proto import kortyn_brain_pb2_grpc as pb_grpc

if TYPE_CHECKING:
    from kortyn.policy_runner import InferenceResult, PolicyInfo, PolicyRunner

LOG = logging.getLogger("kortyn.brain")

GRPC_MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def _abort(context: grpc.ServicerContext, code: grpc.StatusCode, message: str) -> NoReturn:
    context.abort(code, message)
    raise RuntimeError("grpc ServicerContext.abort returned unexpectedly")  # pragma: no cover


class InferenceWorker:
    """Single thread that owns the policy. ``submit`` returns a future with the result."""

    def __init__(self, runner_factory: Any) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kortyn-infer"
        )
        # Build the runner on the worker thread itself (device contexts are thread-affine on some backends).
        self.runner: PolicyRunner = self._executor.submit(runner_factory).result()

    def submit(self, fn: Any, *args: Any) -> concurrent.futures.Future[Any]:
        return self._executor.submit(fn, *args)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


@dataclass
class _Window:
    """Rolling statistics for periodic log lines."""

    inference_ms: deque[float]
    total_ms: deque[float]
    count: int = 0


class BrainServicer(pb_grpc.KortynBrainServicer):  # type: ignore[misc]  # generated base is untyped
    def __init__(self, worker: InferenceWorker, info: PolicyInfo, log_every: int = 200) -> None:
        self._worker = worker
        self._info = info
        self._log_every = log_every
        self._lock = threading.Lock()
        self._window = _Window(deque(maxlen=log_every), deque(maxlen=log_every))

    def GetInfo(self, request: pb.GetInfoRequest, context: grpc.ServicerContext) -> pb.BrainInfo:
        info = self._info
        return pb.BrainInfo(
            policy_type=info.policy_type,
            checkpoint=info.checkpoint,
            device=info.device,
            n_obs_steps=info.n_obs_steps,
            max_actions=info.max_actions,
            action_dim=info.action_dim,
            input_features=[
                pb.FeatureSpec(key=f.key, kind=f.kind, shape=list(f.shape)) for f in info.input_features
            ],
            sampler=info.sampler,
            lerobot_version=info.lerobot_version,
            torch_version=info.torch_version,
            warmup_inference_ms=info.warmup_inference_ms,
            torch_threads=info.torch_threads,
        )

    def Infer(self, request: pb.InferRequest, context: grpc.ServicerContext) -> pb.InferResponse:
        from kortyn.policy_runner import ObservationError

        entered = time.perf_counter()
        runner = self._worker.runner
        try:
            observation = decode_observation(request.observation)
            runner.validate(observation)
        except (CodecError, ObservationError) as exc:
            LOG.warning("rejecting request %d from %s: %s", request.request_id, request.client_id, exc)
            _abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        max_actions = request.max_actions or self._info.max_actions
        if max_actions > self._info.max_actions:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"max_actions={max_actions} exceeds the Brain maximum of {self._info.max_actions}",
            )

        def job() -> tuple[float, InferenceResult] | None:
            started = time.perf_counter()
            if not context.is_active():
                return None
            return started, runner.infer(observation, seed=int(request.seed), max_actions=int(max_actions))

        try:
            outcome = self._worker.submit(job).result()
        except Exception as exc:
            LOG.exception("inference failed for request %d from %s", request.request_id, request.client_id)
            _abort(context, grpc.StatusCode.INTERNAL, f"inference failed: {exc}")
        if outcome is None:
            LOG.info(
                "request %d from %s abandoned by the client before inference",
                request.request_id,
                request.client_id,
            )
            _abort(context, grpc.StatusCode.CANCELLED, "client went away before inference started")
        started, result = outcome
        total_ms = (time.perf_counter() - entered) * 1000.0
        self._record(result.inference_ms, total_ms)
        return pb.InferResponse(
            request_id=request.request_id,
            start_timestep=request.obs_timestep,
            actions=encode_array(result.actions),
            server_queue_ms=(started - entered) * 1000.0,
            server_inference_ms=result.inference_ms,
            server_total_ms=total_ms,
        )

    def _record(self, inference_ms: float, total_ms: float) -> None:
        with self._lock:
            window = self._window
            window.count += 1
            window.inference_ms.append(inference_ms)
            window.total_ms.append(total_ms)
            if window.count % self._log_every == 0:
                inference = sorted(window.inference_ms)
                total = sorted(window.total_ms)
                LOG.info(
                    "%d requests served; last %d: inference p50 %.1f ms p95 %.1f ms, total p95 %.1f ms",
                    window.count,
                    len(inference),
                    inference[len(inference) // 2],
                    inference[int(0.95 * (len(inference) - 1))],
                    total[int(0.95 * (len(total) - 1))],
                )


def configure_process(brain: BrainConfig) -> int:
    """Set thread/env limits; must run before torch is imported. Returns the intra-op thread count."""
    threads = brain.resolved_torch_threads()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(variable, str(threads))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    set_cpu_affinity(brain.cpu_affinity, LOG)
    return threads


def build_server(
    brain: BrainConfig, profile: BrainProfile, checkpoint_dir: Path | None = None
) -> tuple[grpc.Server, InferenceWorker, PolicyInfo]:
    """Load the policy (downloading/migrating the checkpoint if needed), warm it up, and bind the server."""
    from kortyn.checkpoint import ensure_checkpoint
    from kortyn.policy_runner import PolicyRunner

    threads = configure_process(brain)
    checkpoint = checkpoint_dir or ensure_checkpoint(brain.checkpoint)

    def factory() -> PolicyRunner:
        return PolicyRunner(checkpoint, profile, device=brain.device, torch_threads=threads)

    worker = InferenceWorker(factory)
    if brain.warmup_inferences > 0:
        median_ms = worker.submit(worker.runner.warmup, brain.warmup_inferences).result()
        LOG.info("warm-up: %d inferences, median %.1f ms", brain.warmup_inferences, median_ms)
    info = worker.runner.info
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=brain.grpc_max_workers, thread_name_prefix="kortyn-rpc"
        ),
        options=[
            ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
            # Fail loudly if the port is taken instead of silently sharing it with another process.
            ("grpc.so_reuseport", 0),
        ],
    )
    pb_grpc.add_KortynBrainServicer_to_server(BrainServicer(worker, info), server)
    try:
        bound = server.add_insecure_port(brain.address)
    except RuntimeError as exc:
        worker.shutdown()
        raise OSError(f"cannot bind Brain to {brain.address}: {exc}") from exc
    if bound == 0:
        worker.shutdown()
        raise OSError(f"cannot bind Brain to {brain.address} (port in use?)")
    return server, worker, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kortyn Brain: LeRobot policy inference over gRPC.")
    parser.add_argument("--config", default=str(DEFAULT_BASE_CONFIG), help="base YAML")
    parser.add_argument(
        "--profile", required=True, help="Brain profile name from the config (sampler settings)"
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto | cpu | mps | cuda[:N]")
    parser.add_argument("--threads", type=int, default=None, help="CPU intra-op threads (0 = cpu_count-1)")
    parser.add_argument(
        "--checkpoint-dir", default=None, help="use this processor-format checkpoint directly"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        config = load_config(args.config)
        profile = config.profile(args.profile)
    except ConfigError as exc:
        LOG.error("%s", exc)
        return 2
    brain = config.brain
    if args.host is not None:
        brain.host = args.host
    if args.port is not None:
        brain.port = args.port
    if args.device is not None:
        brain.device = args.device
    if args.threads is not None:
        brain.torch_threads = args.threads

    try:
        server, worker, info = build_server(
            brain, profile, Path(args.checkpoint_dir) if args.checkpoint_dir else None
        )
    except (ConfigError, OSError, RuntimeError) as exc:
        LOG.error("Brain failed to start: %s", exc)
        return 1

    stop = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        LOG.info("received signal %d, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    server.start()
    LOG.info(
        "Brain ready on %s | %s %s on %s | n_obs_steps=%d chunk=%d | warm-up median %.1f ms | threads=%d",
        brain.address,
        info.policy_type,
        info.sampler,
        info.device,
        info.n_obs_steps,
        info.max_actions,
        info.warmup_inference_ms,
        info.torch_threads,
    )
    stop.wait()
    server.stop(grace=2.0).wait()
    worker.shutdown()
    LOG.info("Brain stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
