import math

import pytest

from kortyn.config import LatencyEstimatorConfig
from kortyn.latency import MAD_TO_SIGMA, LatencyEstimator


def estimator(**overrides) -> LatencyEstimator:
    return LatencyEstimator(LatencyEstimatorConfig(**{"window": 16, **overrides}))


def test_prior_is_used_until_samples_exist():
    est = estimator(initial_ms=400.0, initial_spread_ms=50.0)
    assert est.count == 0
    assert est.median_s == pytest.approx(0.4)
    assert est.spread_s == pytest.approx(0.05)
    est.observe(0.3)
    est.observe(0.31)
    assert est.median_s == pytest.approx(0.305)
    assert est.spread_s == pytest.approx(0.05)  # fewer than three samples: spread still from the prior


def test_median_and_mad_of_a_known_sample():
    est = estimator(min_spread_ms=0.0)
    for value in (0.30, 0.32, 0.34, 0.36, 0.38):
        assert est.observe(value)
    assert est.median_s == pytest.approx(0.34)
    assert est.spread_s == pytest.approx(MAD_TO_SIGMA * 0.02)
    assert est.quantile_s(2.0) == pytest.approx(0.34 + 2.0 * MAD_TO_SIGMA * 0.02)
    assert est.snapshot() == pytest.approx((340.0, 1000.0 * MAD_TO_SIGMA * 0.02))


def test_single_stall_does_not_move_the_estimate():
    est = estimator(min_spread_ms=5.0)
    for _ in range(15):
        est.observe(0.35)
    est.observe(1.6)  # one request caught in a 1 s link freeze
    assert est.median_s == pytest.approx(0.35)
    assert est.spread_s == pytest.approx(0.005)  # MAD is 0 -> floor
    assert est.quantile_s(1.645) < 0.36


def test_window_slides_and_adapts_to_a_new_level():
    est = estimator(window=8)
    for _ in range(8):
        est.observe(0.35)
    for _ in range(4):
        est.observe(0.55)
    assert est.median_s == pytest.approx(0.45)  # half of the window is new: median between the levels
    for _ in range(4):
        est.observe(0.55)
    assert est.count == 8
    assert est.median_s == pytest.approx(0.55)


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_invalid_samples_are_rejected(bad):
    est = estimator()
    assert not est.observe(bad)
    assert est.count == 0


def test_quantile_grows_with_z_and_reset_restores_the_prior():
    est = estimator(min_spread_ms=0.0)
    for value in (0.40, 0.45, 0.50, 0.55, 0.60, 0.48):
        est.observe(value)
    assert est.quantile_s(0.0) < est.quantile_s(1.0) < est.quantile_s(2.0)
    est.reset()
    assert est.count == 0
    assert est.median_s == pytest.approx(0.4)
