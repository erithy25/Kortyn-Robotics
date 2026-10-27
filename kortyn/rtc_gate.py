"""Phase 5: the 0-ms gate of real-time chunking with short guidance.

Pre-registered in ``reports/phase5_preregistration.md`` (section 3). RTC with short guidance
(``H = d + 4``) at 0 ms is compared with the synchronous reference ``sync25``, measured in the same block and
paired per episode seed. Every LIBERO init state is played twice (seeds ``s`` and ``s + 100``: same task, init
state and simulator seed, different policy noise); the two episodes of an init state form one cluster, and the
interval of the difference resamples whole clusters within each task.

The gate passes if (1) the exact McNemar test on all pairs gives ``p >= 0.05`` and (2) the lower bound of the
95-% interval of ``test - reference`` is at least ``-margin`` (15 pp): the criterion R-RTC of phase 4. A test
condition that is significantly *better* than the reference passes as well (flagged): the gate guards against
harm at zero latency, and a gain there is no integration fault.

The same functions serve the power analysis (:mod:`kortyn.power_phase5`) and the evaluation of the real runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from kortyn.metrics import mcnemar_exact

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]

GATE_MARGIN = 0.15  # lower interval bound of (test - reference) must be >= -15 pp (phase 4, R-RTC)
GATE_ALPHA = 0.05  # McNemar exact p must be >= 0.05
GATE_CONFIDENCE = 0.95
GATE_DRAWS = 10_000

PASSED = "bestanden"
PASSED_BETTER = "bestanden (signifikant besser)"
FAILED = "nicht bestanden"


class GateError(ValueError):
    """Raised for inconsistent gate data (no common seeds, mismatched arrays, invalid settings)."""


@dataclass(frozen=True)
class PairedSample:
    """Outcomes (0/1) of the reference and the test condition on the same episodes, with their clusters."""

    reference: Floats  # (n,)
    test: Floats  # (n,)
    clusters: Ints  # (n,) cluster id of every pair (init state)
    strata: Ints  # (n,) stratum (task) of every pair; a cluster lies in exactly one stratum

    def __post_init__(self) -> None:
        n = self.reference.shape[0]
        if n == 0:
            raise GateError("the gate needs at least one pair")
        for name, array in (("test", self.test), ("clusters", self.clusters), ("strata", self.strata)):
            if array.shape != (n,):
                raise GateError(f"{name} must have shape ({n},), got {array.shape}")
        values = np.concatenate([self.reference, self.test])
        if not np.all((values == 0.0) | (values == 1.0)):
            raise GateError("outcomes must be 0 or 1")
        for cluster in np.unique(self.clusters):
            if np.unique(self.strata[self.clusters == cluster]).size != 1:
                raise GateError(f"cluster {cluster} spans several strata")


def paired_sample(
    reference: Mapping[int, bool],
    test: Mapping[int, bool],
    cluster_of: Callable[[int], tuple[int, int]],
) -> PairedSample:
    """Pairs of the seeds both runs hold; ``cluster_of(seed)`` gives ``(task, init state)``."""
    common = sorted(set(reference) & set(test))
    if not common:
        raise GateError("the two runs have no seed in common")
    slots = [cluster_of(seed) for seed in common]
    keys = {slot: index for index, slot in enumerate(sorted(set(slots)))}
    return PairedSample(
        reference=np.array([float(bool(reference[s])) for s in common]),
        test=np.array([float(bool(test[s])) for s in common]),
        clusters=np.array([keys[slot] for slot in slots], dtype=np.int64),
        strata=np.array([slot[0] for slot in slots], dtype=np.int64),
    )


def cluster_bootstrap_means(
    differences: Floats, clusters: Ints, strata: Ints, draws: int, rng: np.random.Generator
) -> Floats:
    """Mean of ``differences`` over ``draws`` resamples of whole clusters (with replacement, per stratum)."""
    if draws < 1:
        raise GateError("need at least one bootstrap draw")
    total = np.zeros(draws)
    count = np.zeros(draws)
    for stratum in np.unique(strata):
        members = clusters[strata == stratum]
        ids = np.unique(members)
        sums = np.array([differences[clusters == cid].sum() for cid in ids])
        sizes = np.array([float(np.sum(clusters == cid)) for cid in ids])
        picks = rng.multinomial(ids.size, np.full(ids.size, 1.0 / ids.size), size=draws).astype(float)
        total += picks @ sums
        count += picks @ sizes
    means: Floats = total / count
    return means


@dataclass(frozen=True)
class GateResult:
    pairs: int
    clusters: int
    reference_rate: float
    test_rate: float
    difference: float  # test - reference
    interval: tuple[float, float]
    only_reference: int  # pairs with success only in the reference
    only_test: int
    p_value: float  # exact McNemar
    margin: float
    passed: bool
    verdict: str

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["interval"] = list(self.interval)
        return data


def evaluate_gate(
    sample: PairedSample,
    *,
    margin: float = GATE_MARGIN,
    alpha: float = GATE_ALPHA,
    confidence: float = GATE_CONFIDENCE,
    draws: int = GATE_DRAWS,
    rng: np.random.Generator | None = None,
    seed: int = 0,
) -> GateResult:
    """Apply the gate to one paired sample (``rng`` overrides ``seed``, e.g. inside a power simulation)."""
    if not 0.0 < margin < 1.0 or not 0.0 < alpha < 1.0 or not 0.5 < confidence < 1.0:
        raise GateError("need margin and alpha in (0, 1) and confidence in (0.5, 1)")
    generator = rng if rng is not None else np.random.default_rng(seed)
    differences = sample.test - sample.reference
    means = cluster_bootstrap_means(differences, sample.clusters, sample.strata, draws, generator)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = (float(v) for v in np.percentile(means, [tail, 100.0 - tail]))
    only_reference = int(np.sum((sample.reference == 1.0) & (sample.test == 0.0)))
    only_test = int(np.sum((sample.reference == 0.0) & (sample.test == 1.0)))
    p_value = mcnemar_exact(only_reference, only_test)
    difference = float(differences.mean())
    if p_value >= alpha and low >= -margin - 1e-12:
        passed, verdict = True, PASSED
    elif p_value < alpha and difference > 0:
        passed, verdict = True, PASSED_BETTER
    else:
        passed, verdict = False, FAILED
    return GateResult(
        pairs=int(differences.size),
        clusters=int(np.unique(sample.clusters).size),
        reference_rate=float(sample.reference.mean()),
        test_rate=float(sample.test.mean()),
        difference=difference,
        interval=(low, high),
        only_reference=only_reference,
        only_test=only_test,
        p_value=p_value,
        margin=margin,
        passed=passed,
        verdict=verdict,
    )


def subset(sample: PairedSample, keep: Sequence[bool] | npt.NDArray[np.bool_]) -> PairedSample:
    """The pairs selected by a boolean mask (e.g. one task, or the first episode of every init state)."""
    mask = np.asarray(keep, dtype=bool)
    if mask.shape != sample.reference.shape or not mask.any():
        raise GateError("the mask must select at least one pair and match the sample")
    return PairedSample(
        reference=sample.reference[mask],
        test=sample.test[mask],
        clusters=sample.clusters[mask],
        strata=sample.strata[mask],
    )
