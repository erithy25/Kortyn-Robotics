"""Unit tests for the pure network model of the netem proxy (no sockets, no real time)."""

import random
import statistics

import pytest

from kortyn.config import NetworkConfig, StallConfig
from kortyn.netem_proxy import DOWN, UP, LinkModel, LinkParams, NetworkState, StallSchedule

NO_STALLS = StallSchedule(None, 0.0, 0.0)


def make_link(seed=0, stalls=NO_STALLS, **kwargs):
    return LinkModel(LinkParams(**kwargs), random.Random(seed), stalls)


def test_zero_impairment_is_pass_through():
    link = make_link()
    for t in (0.0, 0.5, 0.5, 1.25):
        assert link.schedule(t, 1000) == pytest.approx(t)


def test_fixed_delay_is_added():
    link = make_link(one_way_delay_s=0.025)
    assert link.schedule(1.0, 10) == pytest.approx(1.025)
    assert link.schedule(2.0, 10) == pytest.approx(2.025)


def test_uniform_jitter_stays_within_bounds_and_is_centered():
    link = make_link(seed=3, one_way_delay_s=0.040, one_way_jitter_s=0.020)
    delays = []
    for i in range(5000):
        arrival = i * 1.0  # far apart: every read is its own burst, in-order clamp never binds
        delays.append(link.schedule(arrival, 100) - arrival)
    assert min(delays) >= 0.020 - 1e-12
    assert max(delays) <= 0.060 + 1e-12
    assert statistics.fmean(delays) == pytest.approx(0.040, abs=0.001)


def test_round_trip_matches_rtt_spec():
    """rtt 80 ± 40 ms (user spec 'schlecht'): RTT = up + down stays inside [40, 120] ms."""
    network = NetworkConfig(rtt_ms=80.0, rtt_jitter_ms=40.0)
    state = NetworkState(network, seed=11, clock=lambda: 0.0)
    rtts = []
    for i in range(4000):
        t = i * 1.0
        rtts.append(state.links[UP].schedule(t, 100) - t + state.links[DOWN].schedule(t, 100) - t)
    assert min(rtts) >= 0.040 - 1e-12
    assert max(rtts) <= 0.120 + 1e-12
    assert statistics.fmean(rtts) == pytest.approx(0.080, abs=0.002)


def test_negative_delays_are_clipped_to_zero():
    link = make_link(seed=1, one_way_delay_s=0.001, one_way_jitter_s=0.010)
    for i in range(1000):
        assert link.schedule(i * 1.0, 1) >= i * 1.0


def test_normal_distribution_uses_jitter_as_sigma():
    link = make_link(seed=5, one_way_delay_s=0.100, one_way_jitter_s=0.010, distribution="normal")
    delays = [link.schedule(i * 1.0, 1) - i * 1.0 for i in range(5000)]
    assert statistics.fmean(delays) == pytest.approx(0.100, abs=0.001)
    assert statistics.stdev(delays) == pytest.approx(0.010, rel=0.1)


def test_reads_in_one_burst_share_one_delay_sample():
    link = make_link(seed=9, one_way_delay_s=0.050, one_way_jitter_s=0.040, burst_gap_s=0.002)
    first = link.schedule(10.0000, 65536)
    second = link.schedule(10.0010, 65536)  # 1 ms later: same burst
    third = link.schedule(10.0015, 1000)  # 0.5 ms after the previous read: still the same burst
    assert second - first == pytest.approx(0.0010)
    assert third - second == pytest.approx(0.0005)


def test_delivery_is_in_order_even_with_large_jitter():
    link = make_link(seed=2, one_way_delay_s=0.050, one_way_jitter_s=0.049, burst_gap_s=0.0)
    rng = random.Random(4)
    t = 0.0
    last = float("-inf")
    for _ in range(2000):
        t += rng.uniform(0.0, 0.010)
        delivery = link.schedule(t, 100)
        assert delivery >= last
        assert delivery >= t
        last = delivery


def test_bandwidth_serializes_bytes_fifo():
    link = make_link(bandwidth_bps=1_000_000.0)  # 1 Mbit/s
    assert link.schedule(0.0, 125_000) == pytest.approx(1.0)  # 1 Mbit takes 1 s
    assert link.schedule(0.5, 125_000) == pytest.approx(2.0)  # queued behind the first chunk
    assert link.schedule(5.0, 12_500) == pytest.approx(5.1)  # idle link again


def test_stall_schedule_windows():
    stalls = StallSchedule(10.0, 0.3, phase_s=2.0)
    assert stalls.window_containing(5.0) is None  # disarmed
    stalls.arm(100.0)
    assert stalls.window_containing(101.9) is None
    assert stalls.window_containing(102.1) == pytest.approx((102.0, 102.3))
    assert stalls.window_containing(102.3) is None
    assert stalls.window_containing(112.15) == pytest.approx((112.0, 112.3))
    assert stalls.windows_started_before(112.1) == pytest.approx([(102.0, 102.3), (112.0, 112.3)])
    assert stalls.windows_started_before(101.0) == []


def test_stall_blocks_sending_then_adds_delay():
    stalls = StallSchedule(10.0, 1.0, phase_s=0.0)
    stalls.arm(0.0)  # windows [0,1), [10,11), ...
    link = make_link(one_way_delay_s=0.075, stalls=stalls)
    # data arriving inside the window is sent at the window end and then propagates
    assert link.schedule(10.2, 100) == pytest.approx(11.075)


def test_delivery_falling_into_a_stall_is_held_until_its_end():
    stalls = StallSchedule(10.0, 0.3, phase_s=5.0)
    stalls.arm(0.0)  # windows [5,5.3), [15,15.3), ...
    link = make_link(one_way_delay_s=0.150, stalls=stalls)
    # sent before the stall, would arrive at 5.05 (inside the window) -> delivered at 5.3
    assert link.schedule(4.9, 100) == pytest.approx(5.3)
    # outside any window: plain delay
    assert link.schedule(6.0, 100) == pytest.approx(6.150)


def test_same_seed_gives_identical_sequences_and_other_seed_differs():
    def run(seed):
        link = make_link(seed=seed, one_way_delay_s=0.04, one_way_jitter_s=0.02)
        return [link.schedule(i * 1.0, 100) for i in range(50)]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_reconfigure_keeps_ordering_state():
    link = make_link(one_way_delay_s=1.0)
    first = link.schedule(0.0, 100)
    assert first == pytest.approx(1.0)
    link.reconfigure(LinkParams(), random.Random(0), NO_STALLS)
    # without delay this read would be delivered at 0.1, but it must not overtake the queued data
    assert link.schedule(0.1, 100) == pytest.approx(1.0)
    assert link.schedule(2.0, 100) == pytest.approx(2.0)


def test_network_state_configure_resets_stats_and_draws_phase_from_seed():
    clock_value = [0.0]
    state = NetworkState(NetworkConfig(), seed=1, clock=lambda: clock_value[0])
    network = NetworkConfig(rtt_ms=150, rtt_jitter_ms=80, stall=StallConfig(interval_s=10, duration_ms=1000))
    state.stats.up.bytes = 123
    clock_value[0] = 50.0
    state.configure(network, seed=1000)
    assert state.stats.up.bytes == 0
    assert state.stats.seed == 1000
    assert 0.0 <= state.stalls.phase_s < 10.0
    assert state.stalls.anchor is None  # stalls only start after arm()
    phase = state.stalls.phase_s
    state.configure(network, seed=1000)
    assert state.stalls.phase_s == phase  # deterministic per seed
    state.configure(network, seed=1001)
    assert state.stalls.phase_s != phase
    clock_value[0] = 60.0
    assert state.arm() == 60.0
    clock_value[0] = 85.0
    stats = state.stats_dict()
    assert stats["armed"] is True
    assert stats["since_arm_s"] == pytest.approx(25.0)
    first = state.stalls.phase_s
    assert stats["stall_windows_s"][0] == pytest.approx([first, first + 1.0])
    assert all(end - start == pytest.approx(1.0) for start, end in stats["stall_windows_s"])
