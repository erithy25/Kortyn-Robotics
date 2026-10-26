"""Shared WLAN medium of the phase-4 fleet simulation (kortyn.wlan)."""

from __future__ import annotations

import pytest

from kortyn.config import ConfigError, NetworkConfig
from kortyn.netem_proxy import DOWN, UP
from kortyn.virtual_time import Simulation
from kortyn.wlan import SharedMedium, SharedMediumLink


def test_medium_sends_one_message_at_a_time_in_arrival_order() -> None:
    medium = SharedMedium(rate_mbit_s=8.0)  # 1 byte per microsecond
    assert medium.transmit(0.0, 1_000_000) == pytest.approx(1.0)
    assert medium.transmit(0.5, 500_000) == pytest.approx(1.5)  # waits 0.5 s for the first message
    assert medium.transmit(3.0, 0) == pytest.approx(3.0)
    assert medium.busy_s == pytest.approx(1.5)
    assert medium.wait_s == pytest.approx(0.5)
    assert medium.messages == 3
    assert medium.utilization(3.0) == pytest.approx(0.5)
    with pytest.raises(ConfigError):
        SharedMedium(0.0)
    with pytest.raises(ConfigError):
        medium.transmit(4.0, -1)
    with pytest.raises(ConfigError):
        medium.utilization(0.0)


def test_links_of_two_robots_share_the_medium() -> None:
    sim = Simulation()
    network = NetworkConfig()  # no delay, no jitter: only the medium
    medium = SharedMedium(rate_mbit_s=80.0)  # 10 bytes per microsecond
    a = SharedMediumLink(sim, network, 1, medium, request_bytes=100_000, response_bytes=1_000)
    b = SharedMediumLink(sim, network, 2, medium, request_bytes=100_000, response_bytes=1_000)
    first, held = a.transmit(UP, 0.0, 10)  # the payload size, not the stage-1 message size, counts
    assert first == pytest.approx(0.01)
    assert held
    second, _ = b.transmit(UP, 0.0, 10)
    assert second == pytest.approx(0.02)  # queued behind robot a
    reply, _ = a.transmit(DOWN, 0.02, 10)
    assert reply == pytest.approx(0.0201)
    assert a.stats()["shared_medium_mbit_s"] == pytest.approx(80.0)
    with pytest.raises(ConfigError):
        SharedMediumLink(sim, network, 3, medium, request_bytes=-1, response_bytes=0)
