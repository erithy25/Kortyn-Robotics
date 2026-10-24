"""Process-level helpers shared by Brain, proxy and agent (CPU pinning, CPU steal measurement)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_PROC_STAT = Path("/proc/stat")


def set_cpu_affinity(cores: list[int], logger: logging.Logger) -> None:
    """Pin the current process to ``cores`` (Linux only; skipped with a log line elsewhere)."""
    if not cores:
        return
    if not hasattr(os, "sched_setaffinity"):
        logger.info("CPU affinity not supported on this platform; ignoring %s", cores)
        return
    available = os.sched_getaffinity(0)
    usable = [c for c in cores if c in available]
    if not usable:
        logger.warning(
            "none of the requested cores %s are available (%s); not pinning", cores, sorted(available)
        )
        return
    os.sched_setaffinity(0, usable)
    logger.info("pinned to CPU cores %s", usable)


def read_cpu_times() -> tuple[int, int] | None:
    """Return ``(steal_jiffies, total_jiffies)`` from /proc/stat, or ``None`` where unavailable."""
    try:
        first_line = _PROC_STAT.read_text(encoding="ascii").splitlines()[0]
    except (OSError, IndexError):
        return None
    parts = first_line.split()
    if not parts or parts[0] != "cpu":
        return None
    values = [int(v) for v in parts[1:]]
    steal = values[7] if len(values) > 7 else 0
    # guest/guest_nice (indices 8, 9) are already contained in user/nice
    total = sum(values[:8])
    return steal, total


def steal_fraction(before: tuple[int, int] | None, after: tuple[int, int] | None) -> float | None:
    """Share of CPU time stolen by the hypervisor between two ``read_cpu_times`` samples."""
    if before is None or after is None:
        return None
    total = after[1] - before[1]
    if total <= 0:
        return None
    return max(0.0, (after[0] - before[0]) / total)
