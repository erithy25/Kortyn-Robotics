"""Kortyn: offloaded robot policy ("Brain"), real-time simulated robot ("Agent"), network impairment proxy."""

import os

# numpy/scipy in Kortyn work on tiny matrices (curve fits, queue statistics). A multithreaded OpenBLAS next to
# the rollout workers oversubscribes the CPU fifty-fold (measured in phase 4); one thread unless set otherwise.
# Every ``python -m kortyn.<module>`` imports this package before numpy, so the default takes effect.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.0"
