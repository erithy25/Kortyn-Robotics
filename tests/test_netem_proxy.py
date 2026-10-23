"""End-to-end tests of the netem proxy over real sockets (echo server upstream)."""

import asyncio
import json
import os
import socket

import pytest

from kortyn.config import NetworkConfig, ProxyConfig, StallConfig
from kortyn.netem_proxy import NetemProxy


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def start_echo_server() -> tuple[asyncio.Server, int]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, int(server.sockets[0].getsockname()[1])


async def start_proxy(network: NetworkConfig, upstream_port: int, seed: int = 1) -> NetemProxy:
    config = ProxyConfig(host="127.0.0.1", port=free_port(), control_port=free_port())
    proxy = NetemProxy(config, "127.0.0.1", upstream_port, network, seed=seed)
    await proxy.start()
    return proxy


async def control(proxy: NetemProxy, *messages: object) -> list[dict]:
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.config.control_port)
    replies = []
    for message in messages:
        raw = message if isinstance(message, bytes) else (json.dumps(message) + "\n").encode()
        writer.write(raw)
        await writer.drain()
        replies.append(json.loads(await reader.readline()))
    writer.close()
    await writer.wait_closed()
    return replies


def test_payload_integrity_and_round_trip_delay():
    async def scenario() -> None:
        echo, echo_port = await start_echo_server()
        proxy = await start_proxy(NetworkConfig(rtt_ms=100.0), echo_port)
        loop = asyncio.get_running_loop()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy.config.port)
            payload = os.urandom(1_000_000)
            writer.write(payload)
            await writer.drain()
            assert await asyncio.wait_for(reader.readexactly(len(payload)), 10) == payload

            rtts = []
            for _ in range(5):
                start = loop.time()
                writer.write(b"x")
                await writer.drain()
                await asyncio.wait_for(reader.readexactly(1), 5)
                rtts.append(loop.time() - start)
            assert min(rtts) >= 0.099
            assert max(rtts) < 0.160
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.close()
            echo.close()

    asyncio.run(scenario())


def test_stall_freezes_the_link_until_the_window_ends():
    async def scenario() -> None:
        echo, echo_port = await start_echo_server()
        network = NetworkConfig(stall=StallConfig(interval_s=1.0, duration_ms=300.0))
        proxy = await start_proxy(network, echo_port, seed=5)
        loop = asyncio.get_running_loop()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy.config.port)
            (armed,) = await control(proxy, {"cmd": "arm"})
            window_start = armed["armed_at"] + proxy.state.stalls.phase_s
            window_end = window_start + 0.3
            await asyncio.sleep(max(0.0, window_start + 0.05 - loop.time()))
            writer.write(b"ping")
            await writer.drain()
            await asyncio.wait_for(reader.readexactly(4), 5)
            assert loop.time() >= window_end - 0.002
            assert loop.time() < window_end + 0.08
            (stats,) = await control(proxy, {"cmd": "stats"})
            assert stats["stall_windows_s"][0] == pytest.approx(
                [proxy.state.stalls.phase_s, proxy.state.stalls.phase_s + 0.3]
            )
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.close()
            echo.close()

    asyncio.run(scenario())


def test_control_protocol_commands_and_errors():
    async def scenario() -> None:
        echo, echo_port = await start_echo_server()
        proxy = await start_proxy(NetworkConfig(), echo_port)
        try:
            ping, configured, bad_value, unknown_key, unknown_cmd, garbage, stats = await control(
                proxy,
                {"cmd": "ping"},
                {"cmd": "configure", "seed": 1000, "network": {"rtt_ms": 20, "rtt_jitter_ms": 10}},
                {"cmd": "configure", "network": {"rtt_ms": -5}},
                {"cmd": "configure", "network": {"rtt": 5}},
                {"cmd": "explode"},
                b"not json\n",
                {"cmd": "stats"},
            )
            assert ping["ok"] is True
            assert configured == {"ok": True}
            assert bad_value["ok"] is False
            assert "rtt_ms" in bad_value["error"]
            assert unknown_key["ok"] is False
            assert unknown_cmd["ok"] is False
            assert "unknown command" in unknown_cmd["error"]
            assert garbage["ok"] is False
            assert stats["ok"] is True
            assert stats["seed"] == 1000
            assert stats["network"]["rtt_ms"] == 20
            assert stats["armed"] is False
            assert stats["stall_windows_s"] == []
        finally:
            await proxy.close()
            echo.close()

    asyncio.run(scenario())


def test_eof_propagates_and_connections_are_counted():
    async def scenario() -> None:
        echo, echo_port = await start_echo_server()
        proxy = await start_proxy(NetworkConfig(rtt_ms=10.0), echo_port)
        try:
            for _ in range(2):
                reader, writer = await asyncio.open_connection("127.0.0.1", proxy.config.port)
                writer.write(b"hello")
                writer.write_eof()  # client half-closes -> echo server finishes -> client sees EOF
                assert await asyncio.wait_for(reader.read(), 5) == b"hello"
                writer.close()
                await writer.wait_closed()
            await asyncio.sleep(0.05)
            (stats,) = await control(proxy, {"cmd": "stats"})
            assert stats["connections_total"] == 2
            assert stats["active_connections"] == 0
            assert stats["bytes_up"] == 10
            assert stats["bytes_down"] == 10
        finally:
            await proxy.close()
            echo.close()

    asyncio.run(scenario())


def test_unreachable_upstream_closes_client_connection():
    async def scenario() -> None:
        proxy = await start_proxy(NetworkConfig(), upstream_port=free_port())
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy.config.port)
            assert await asyncio.wait_for(reader.read(), 5) == b""
            writer.close()
        finally:
            await proxy.close()

    asyncio.run(scenario())
