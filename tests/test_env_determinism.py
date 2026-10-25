"""PushT reproducibility diagnostic: runs without a checkpoint (scripted pusher), differences stay tiny."""

from __future__ import annotations

import numpy as np
import pytest

from kortyn.config import load_config
from kortyn.env_determinism import checksum, compare, main, run


def test_compare_and_checksum():
    a = np.array([0.1, 0.5, 0.9])
    b = a.copy()
    b[1] = np.nextafter(b[1], 1.0)
    same, close = compare(a, a.copy(), "x"), compare(a, b, "y")
    assert (same.identical, same.steps_differing, same.max_abs_diff) == (True, 0, 0.0)
    assert (close.identical, close.steps_differing) == (False, 1)
    assert 0.0 < close.max_abs_diff < 1e-15
    assert checksum(a) == checksum(a.copy()) != checksum(b)


@pytest.mark.timeout(300)
def test_pusher_trajectories_agree_up_to_the_last_float_bits():
    reference, results = run(load_config(), seed=1000, steps=40, other_episodes=1, source="pusher")
    assert reference.shape == (40,)
    assert np.all(np.isfinite(reference))
    assert [r.situation for r in results] == [
        "same env instance again",
        "after 1 other episodes",
        "second env instance",
    ]
    # pymunk is not guaranteed to be bitwise reproducible across instances; any difference is rounding-sized
    assert all(r.max_abs_diff < 1e-12 for r in results)


def test_cli_rejects_invalid_arguments():
    with pytest.raises(SystemExit):
        main(["--steps", "0", "--source", "pusher"])
