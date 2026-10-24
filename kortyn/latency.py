"""Online end-to-end latency estimate for one robot (phase 2, freshness engine).

The latency that matters to the robot is the **end-to-end latency**: from capturing the observation until
the action chunk computed from it is available on the robot (network both ways + queueing + inference).
Every completed request is one sample.

The estimate must be robust: a single link freeze (stall) produces one sample several times larger than the
rest, and a mean or a maximum would make the agent pessimistic for a long time afterwards. The estimator
therefore keeps a sliding window of the last ``window`` samples and reports

* the **median** as the typical latency, and
* the **MAD** (median absolute deviation) scaled by 1.4826 as a robust standard deviation,

both with a 50 % breakdown point: up to half of the window may be outliers before the estimate moves.
``quantile_s(z) = median + z * spread`` is the latency the agent plans with when it must be on time.
"""

from __future__ import annotations

import math
import statistics
from collections import deque

from kortyn.config import LatencyEstimatorConfig

# For normally distributed data, 1.4826 * MAD is a consistent estimator of the standard deviation.
MAD_TO_SIGMA = 1.4826


class LatencyEstimator:
    """Sliding-window median/MAD estimate of the end-to-end latency in seconds."""

    def __init__(self, config: LatencyEstimatorConfig) -> None:
        self._config = config
        self._samples: deque[float] = deque(maxlen=config.window)

    def reset(self) -> None:
        self._samples.clear()

    def observe(self, latency_s: float) -> bool:
        """Add one completed request's latency; negative or non-finite values are rejected."""
        if not math.isfinite(latency_s) or latency_s < 0:
            return False
        self._samples.append(float(latency_s))
        return True

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def median_s(self) -> float:
        if not self._samples:
            return self._config.initial_ms / 1000.0
        return float(statistics.median(self._samples))

    @property
    def spread_s(self) -> float:
        """Robust standard deviation (1.4826 * MAD), never below ``min_spread_ms``."""
        floor = self._config.min_spread_ms / 1000.0
        if len(self._samples) < 3:
            return max(floor, self._config.initial_spread_ms / 1000.0)
        center = statistics.median(self._samples)
        mad = statistics.median(abs(value - center) for value in self._samples)
        return max(floor, MAD_TO_SIGMA * float(mad))

    def quantile_s(self, z: float) -> float:
        """Latency to plan with when being late is costly: ``median + z * spread``."""
        return self.median_s + z * self.spread_s

    def snapshot(self) -> tuple[float, float]:
        """(median, spread) in milliseconds, for logging."""
        return 1000.0 * self.median_s, 1000.0 * self.spread_s
