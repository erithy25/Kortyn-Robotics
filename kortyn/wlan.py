"""Shared WLAN medium for the fleet simulation (phase 4, step 4): one radio channel for all robots.

Every message of every robot occupies the medium for ``bytes * 8 / rate`` seconds, one message at a time
in the order the messages reach it (FIFO), in both directions (WLAN is half-duplex). The network scenario's
latency model of each robot's link (delay, jitter, stalls) applies on top, after the message has left the
medium.

The rate is the effective goodput the shared channel sustains (after MAC/PHY overhead, retransmissions and
contention); the model has no rate adaptation and no airtime fairness. Since the timing robots of stage 1 send
no images, the payload of a request and of a response is given explicitly (the measured bytes of the image
codec plus the rest of the request).
"""

from __future__ import annotations

from typing import Any

from kortyn.config import ConfigError, NetworkConfig
from kortyn.netem_proxy import UP
from kortyn.virtual_time import Simulation, VirtualLink


class SharedMedium:
    """FIFO radio channel: a message entering at ``t`` leaves when all earlier messages and it are sent."""

    def __init__(self, rate_mbit_s: float) -> None:
        if not rate_mbit_s > 0:
            raise ConfigError(f"WLAN rate must be > 0 Mbit/s, got {rate_mbit_s}")
        self.rate_bit_s = rate_mbit_s * 1e6
        self._free_at = 0.0
        self.busy_s = 0.0
        self.messages = 0
        self.wait_s = 0.0  # total time messages waited for the medium

    def transmit(self, t: float, nbytes: int) -> float:
        """Time the message of ``nbytes`` entering at ``t`` has left the medium."""
        if nbytes < 0:
            raise ConfigError(f"message size must be >= 0 bytes, got {nbytes}")
        start = max(t, self._free_at)
        airtime = nbytes * 8.0 / self.rate_bit_s
        self._free_at = start + airtime
        self.busy_s += airtime
        self.wait_s += start - t
        self.messages += 1
        return self._free_at

    def utilization(self, duration_s: float) -> float:
        """Share of ``duration_s`` the medium was busy."""
        if duration_s <= 0:
            raise ConfigError("duration must be > 0")
        return self.busy_s / duration_s


class SharedMediumLink(VirtualLink):
    """A robot's link whose messages first cross the shared medium (payload sizes given explicitly)."""

    def __init__(
        self,
        sim: Simulation,
        network: NetworkConfig,
        seed: int,
        medium: SharedMedium,
        request_bytes: int,
        response_bytes: int,
    ) -> None:
        if request_bytes < 0 or response_bytes < 0:
            raise ConfigError("payload sizes must be >= 0 bytes")
        super().__init__(sim, network, seed)
        self._medium = medium
        self._payload = {UP: request_bytes}
        self._response_bytes = response_bytes

    def transmit(self, direction: str, t: float, nbytes: int) -> tuple[float, bool]:
        payload = self._payload.get(direction, self._response_bytes)
        on_air = self._medium.transmit(t, payload)
        delivery, _held = super().transmit(direction, on_air, payload)
        return delivery, delivery > t

    def stats(self) -> dict[str, Any]:
        return {**super().stats(), "shared_medium_mbit_s": self._medium.rate_bit_s / 1e6}
