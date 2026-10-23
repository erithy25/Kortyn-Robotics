"""Platform-independent TCP proxy that emulates a bad network between Kortyn Agent and Kortyn Brain.

Model (per direction, see ``LinkModel.schedule``):

1. **Stall**: the link accepts nothing during a stall window; data arriving inside a window is sent at
   the window's end. Data whose delivery would fall inside a window is held until the window ends.
2. **Bandwidth**: optional serialization delay ``bytes * 8 / rate`` on a FIFO link.
3. **Delay + jitter**: one-way delay ``rtt/2`` plus jitter (uniform ``±rtt_jitter/2`` like netem's default,
   or normal with sigma ``rtt_jitter/2``). Reads arriving within ``burst_gap_ms`` of each other share one
   delay sample, so a message split across several TCP reads is not delayed by the max of several samples.
4. **In-order delivery**: a byte stream cannot overtake itself (TCP head-of-line blocking).

The proxy is controlled over a separate JSON-lines TCP port (``ping``, ``configure``, ``arm``, ``stats``)
so that scenarios can be switched between episodes without restarting the proxy or breaking the gRPC
connection, and so that stall timing is anchored to the episode start (``arm``).

Run standalone::

    python -m kortyn.netem_proxy --scenario schlecht --upstream 127.0.0.1:50051
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import random
import signal
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kortyn.config import (
    DEFAULT_BASE_CONFIG,
    ConfigError,
    NetworkConfig,
    ProxyConfig,
    decode,
    load_config,
    load_scenario,
)
from kortyn.process_utils import set_cpu_affinity

LOG = logging.getLogger("kortyn.proxy")

UP = "up"  # agent -> brain
DOWN = "down"  # brain -> agent


# --------------------------------------------------------------------------------------------- model
class StallSchedule:
    """Periodic stall windows ``[anchor + phase + n*interval, ... + duration)`` for n >= 0.

    The schedule is disarmed (no stalls) until ``arm`` is called with the episode start time.
    """

    def __init__(self, interval_s: float | None, duration_s: float, phase_s: float) -> None:
        if interval_s is not None and not 0 < duration_s < interval_s:
            raise ValueError("stall duration must be in (0, interval)")
        self.interval_s = interval_s
        self.duration_s = duration_s
        self.phase_s = phase_s
        self.anchor: float | None = None

    @property
    def enabled(self) -> bool:
        return self.interval_s is not None

    def arm(self, anchor: float) -> None:
        self.anchor = anchor

    def window_containing(self, t: float) -> tuple[float, float] | None:
        if self.interval_s is None or self.anchor is None:
            return None
        first_start = self.anchor + self.phase_s
        if t < first_start:
            return None
        n = math.floor((t - first_start) / self.interval_s)
        start = first_start + n * self.interval_s
        end = start + self.duration_s
        return (start, end) if start <= t < end else None

    def windows_started_before(self, t: float) -> list[tuple[float, float]]:
        if self.interval_s is None or self.anchor is None:
            return []
        first_start = self.anchor + self.phase_s
        windows: list[tuple[float, float]] = []
        start = first_start
        while start < t:
            windows.append((start, start + self.duration_s))
            start += self.interval_s
        return windows


@dataclass
class LinkParams:
    one_way_delay_s: float = 0.0
    one_way_jitter_s: float = 0.0
    distribution: str = "uniform"
    bandwidth_bps: float | None = None
    burst_gap_s: float = 0.002

    @classmethod
    def from_network(cls, network: NetworkConfig) -> LinkParams:
        return cls(
            one_way_delay_s=network.one_way_delay_ms / 1000.0,
            one_way_jitter_s=network.one_way_jitter_ms / 1000.0,
            distribution=network.jitter_distribution,
            bandwidth_bps=None if network.bandwidth_mbps is None else network.bandwidth_mbps * 1e6,
            burst_gap_s=network.burst_gap_ms / 1000.0,
        )


class LinkModel:
    """Computes delivery times for one direction. Pure and deterministic given the RNG seed."""

    def __init__(self, params: LinkParams, rng: random.Random, stalls: StallSchedule) -> None:
        self.params = params
        self.rng = rng
        self.stalls = stalls
        self._link_free_at = -math.inf
        self._last_delivery = -math.inf
        self._last_arrival: float | None = None
        self._burst_delay = 0.0

    def reconfigure(self, params: LinkParams, rng: random.Random, stalls: StallSchedule) -> None:
        """Swap parameters but keep ordering state, so data already queued is never reordered."""
        self.params = params
        self.rng = rng
        self.stalls = stalls
        self._last_arrival = None
        self._burst_delay = 0.0

    def sample_delay(self) -> float:
        p = self.params
        if p.one_way_jitter_s <= 0:
            return max(0.0, p.one_way_delay_s)
        if p.distribution == "normal":
            value = self.rng.gauss(p.one_way_delay_s, p.one_way_jitter_s)
        else:
            value = self.rng.uniform(
                p.one_way_delay_s - p.one_way_jitter_s, p.one_way_delay_s + p.one_way_jitter_s
            )
        return max(0.0, value)

    def schedule(self, arrival: float, nbytes: int) -> float:
        """Return the delivery time for ``nbytes`` read from the source at time ``arrival``."""
        # 1. a stalled link does not accept data: sending starts when the window ends
        send = arrival
        window = self.stalls.window_containing(send)
        if window is not None:
            send = window[1]
        # 2. FIFO serialization at the configured rate
        tx_start = max(send, self._link_free_at)
        tx_time = 0.0 if self.params.bandwidth_bps is None else nbytes * 8.0 / self.params.bandwidth_bps
        tx_end = tx_start + tx_time
        self._link_free_at = tx_end
        # 3. propagation delay, one sample per burst
        if self._last_arrival is None or arrival - self._last_arrival > self.params.burst_gap_s:
            self._burst_delay = self.sample_delay()
        self._last_arrival = arrival
        delivery = tx_end + self._burst_delay
        # 4. a stalled link does not deliver either
        window = self.stalls.window_containing(delivery)
        if window is not None:
            delivery = window[1]
        # 5. in-order delivery
        delivery = max(delivery, self._last_delivery)
        self._last_delivery = delivery
        return delivery


@dataclass
class DirectionStats:
    bytes: int = 0
    reads: int = 0
    max_lateness_s: float = 0.0


@dataclass
class EpisodeStats:
    """Counters since the last ``configure`` (= per episode when driven by the runner)."""

    configured_at: float
    seed: int
    up: DirectionStats = field(default_factory=DirectionStats)
    down: DirectionStats = field(default_factory=DirectionStats)
    connections_opened: int = 0


class NetworkState:
    """Shared link state for all connections: one physical link per direction, one stall schedule."""

    def __init__(self, network: NetworkConfig, seed: int, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.network = network
        self.seed = seed
        self.stalls = self._make_stalls(network, seed)
        params = LinkParams.from_network(network)
        self.links = {
            UP: LinkModel(params, random.Random(self._direction_seed(seed, UP)), self.stalls),
            DOWN: LinkModel(params, random.Random(self._direction_seed(seed, DOWN)), self.stalls),
        }
        self.stats = EpisodeStats(configured_at=clock(), seed=seed)
        self.connections_total = 0
        self.active_connections = 0

    @staticmethod
    def _direction_seed(seed: int, direction: str) -> int:
        return seed * 2 + (0 if direction == UP else 1)

    @staticmethod
    def _make_stalls(network: NetworkConfig, seed: int) -> StallSchedule:
        if network.stall is None:
            return StallSchedule(None, 0.0, 0.0)
        interval = network.stall.interval_s
        phase = random.Random(seed * 7919 + 17).uniform(0.0, interval)
        return StallSchedule(interval, network.stall.duration_ms / 1000.0, phase)

    def configure(self, network: NetworkConfig, seed: int) -> None:
        self.network = network
        self.seed = seed
        self.stalls = self._make_stalls(network, seed)
        params = LinkParams.from_network(network)
        for direction, link in self.links.items():
            link.reconfigure(params, random.Random(self._direction_seed(seed, direction)), self.stalls)
        self.stats = EpisodeStats(configured_at=self._clock(), seed=seed)

    def arm(self) -> float:
        now = self._clock()
        self.stalls.arm(now)
        return now

    def stats_dict(self) -> dict[str, Any]:
        now = self._clock()
        anchor = self.stalls.anchor
        windows = [] if anchor is None else self.stalls.windows_started_before(now)
        return {
            "seed": self.stats.seed,
            "since_configure_s": now - self.stats.configured_at,
            "armed": anchor is not None,
            "since_arm_s": None if anchor is None else now - anchor,
            "stall_windows_s": [] if anchor is None else [[s - anchor, e - anchor] for s, e in windows],
            "bytes_up": self.stats.up.bytes,
            "bytes_down": self.stats.down.bytes,
            "reads_up": self.stats.up.reads,
            "reads_down": self.stats.down.reads,
            "max_delivery_lateness_ms": 1000.0
            * max(self.stats.up.max_lateness_s, self.stats.down.max_lateness_s),
            "connections_opened_since_configure": self.stats.connections_opened,
            "connections_total": self.connections_total,
            "active_connections": self.active_connections,
            "network": self.network.to_dict(),
        }


# --------------------------------------------------------------------------------------------- proxy
_EOF = object()


class NetemProxy:
    def __init__(
        self,
        config: ProxyConfig,
        upstream_host: str,
        upstream_port: int,
        network: NetworkConfig,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self._loop: asyncio.AbstractEventLoop | None = None
        self.state = NetworkState(network, seed, time.monotonic)
        self._servers: list[asyncio.Server] = []
        self._connection_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        data_server = await asyncio.start_server(self._handle_client, self.config.host, self.config.port)
        control_server = await asyncio.start_server(
            self._handle_control, self.config.host, self.config.control_port
        )
        self._servers = [data_server, control_server]
        LOG.info(
            "proxy listening on %s:%d -> %s:%d (control :%d)",
            self.config.host,
            self.config.port,
            self.upstream_host,
            self.upstream_port,
            self.config.control_port,
        )

    async def close(self) -> None:
        for server in self._servers:
            server.close()
        for server in self._servers:
            with contextlib.suppress(Exception):
                await server.wait_closed()
        for task in list(self._connection_tasks):
            task.cancel()
        if self._connection_tasks:
            await asyncio.gather(*self._connection_tasks, return_exceptions=True)

    # ------------------------------------------------------------------ data path
    async def _handle_client(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._connection_tasks.add(task)
        peer = client_writer.get_extra_info("peername")
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                self.upstream_host, self.upstream_port
            )
        except OSError as exc:
            LOG.error(
                "upstream %s:%d unreachable for client %s: %s",
                self.upstream_host,
                self.upstream_port,
                peer,
                exc,
            )
            client_writer.close()
            with contextlib.suppress(Exception):
                await client_writer.wait_closed()
            if task is not None:
                self._connection_tasks.discard(task)
            return
        for writer in (client_writer, upstream_writer):
            _set_nodelay(writer)
        self.state.connections_total += 1
        self.state.active_connections += 1
        self.state.stats.connections_opened += 1
        LOG.info("connection opened from %s (total %d)", peer, self.state.connections_total)
        try:
            await asyncio.gather(
                self._pump(client_reader, upstream_writer, UP),
                self._pump(upstream_reader, client_writer, DOWN),
            )
        finally:
            self.state.active_connections -= 1
            for writer in (client_writer, upstream_writer):
                writer.close()
            for writer in (client_writer, upstream_writer):
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
            LOG.info("connection from %s closed", peer)
            if task is not None:
                self._connection_tasks.discard(task)

    async def _pump(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, direction: str) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        buffered = 0
        drained = asyncio.Event()
        drained.set()
        max_buffer = self.config.max_buffer_bytes

        async def read_side() -> None:
            nonlocal buffered
            try:
                while True:
                    while buffered >= max_buffer:
                        drained.clear()
                        await drained.wait()
                    data = await reader.read(self.config.read_chunk_bytes)
                    if not data:
                        break
                    now = loop.time()
                    deliver_at = self.state.links[direction].schedule(now, len(data))
                    stats = self.state.stats.up if direction == UP else self.state.stats.down
                    stats.bytes += len(data)
                    stats.reads += 1
                    buffered += len(data)
                    queue.put_nowait((deliver_at, data))
            except (ConnectionError, OSError) as exc:
                LOG.info("%s read side closed: %s", direction, exc)
            finally:
                queue.put_nowait(_EOF)

        async def write_side() -> None:
            nonlocal buffered
            try:
                while True:
                    item = await queue.get()
                    if item is _EOF:
                        if writer.can_write_eof():
                            with contextlib.suppress(OSError):
                                writer.write_eof()
                        break
                    deliver_at, data = item
                    wait = deliver_at - loop.time()
                    if wait > 0:
                        await asyncio.sleep(wait)
                    lateness = loop.time() - deliver_at
                    stats = self.state.stats.up if direction == UP else self.state.stats.down
                    stats.max_lateness_s = max(stats.max_lateness_s, lateness)
                    writer.write(data)
                    await writer.drain()
                    buffered -= len(data)
                    if buffered < max_buffer:
                        drained.set()
            except (ConnectionError, OSError) as exc:
                LOG.info("%s write side closed: %s", direction, exc)

        await asyncio.gather(read_side(), write_side())

    # ------------------------------------------------------------------ control path
    async def _handle_control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                response = self._dispatch(line)
                writer.write((json.dumps(response) + "\n").encode("utf-8"))
                await writer.drain()
        except (ConnectionError, OSError) as exc:
            LOG.debug("control connection closed: %s", exc)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def _dispatch(self, line: bytes) -> dict[str, Any]:
        try:
            message = json.loads(line.decode("utf-8"))
            if not isinstance(message, dict):
                raise ValueError("control message must be a JSON object")
            command = message.get("cmd")
            if command == "ping":
                return {"ok": True, "pid": os.getpid()}
            if command == "configure":
                network = decode(NetworkConfig, message.get("network") or {}, "control.configure.network")
                seed = int(message.get("seed", 0))
                self.state.configure(network, seed)
                LOG.info("configured seed=%d network=%s", seed, network.to_dict())
                return {"ok": True}
            if command == "arm":
                armed_at = self.state.arm()
                return {"ok": True, "armed_at": armed_at}
            if command == "stats":
                return {"ok": True, **self.state.stats_dict()}
            raise ValueError(f"unknown command {command!r}")
        except (ValueError, ConfigError, TypeError) as exc:
            LOG.warning("control error: %s", exc)
            return {"ok": False, "error": str(exc)}


def _set_nodelay(writer: asyncio.StreamWriter) -> None:
    sock = writer.get_extra_info("socket")
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def parse_hostport(value: str) -> tuple[str, int]:
    host, sep, port = value.rpartition(":")
    if not sep or not host:
        raise argparse.ArgumentTypeError(f"expected HOST:PORT, got {value!r}")
    try:
        return host, int(port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid port in {value!r}") from exc


async def _serve(proxy: NetemProxy) -> None:
    await proxy.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    LOG.info("shutting down")
    await proxy.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kortyn network impairment proxy (netem-like, TCP level).")
    parser.add_argument("--config", default=str(DEFAULT_BASE_CONFIG), help="base YAML (proxy/brain sections)")
    parser.add_argument(
        "--scenario", default="gut", help="scenario name or YAML path for the initial network"
    )
    parser.add_argument("--upstream", type=parse_hostport, default=None, help="Brain address HOST:PORT")
    parser.add_argument("--host", default=None, help="listen host (default from config)")
    parser.add_argument("--port", type=int, default=None, help="listen port (default from config)")
    parser.add_argument("--control-port", type=int, default=None, help="control port (default from config)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        config = load_config(args.config)
        scenario = load_scenario(args.scenario)
    except ConfigError as exc:
        LOG.error("%s", exc)
        return 2
    proxy_cfg = config.proxy
    if args.host is not None:
        proxy_cfg.host = args.host
    if args.port is not None:
        proxy_cfg.port = args.port
    if args.control_port is not None:
        proxy_cfg.control_port = args.control_port
    upstream_host, upstream_port = args.upstream or (config.brain.host, config.brain.port)
    set_cpu_affinity(proxy_cfg.cpu_affinity, LOG)
    proxy = NetemProxy(proxy_cfg, upstream_host, upstream_port, scenario.network, seed=args.seed)
    try:
        asyncio.run(_serve(proxy))
    except OSError as exc:
        LOG.error("proxy failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
