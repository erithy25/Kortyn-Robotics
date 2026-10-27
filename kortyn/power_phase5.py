"""Power analysis of phase 5 (before any new run): how often does the 0-ms gate pass?

The gate (:mod:`kortyn.rtc_gate`, reports/phase5_preregistration.md section 3) compares RTC with short
guidance at 0 ms with the synchronous reference ``sync25`` of the same block, paired per episode seed. This
module simulates the planned design many times and applies exactly the gate's code to every simulated data
set.

* **Design:** 2 tasks x 50 init states; one episode per init state (100 pairs, phase-4 design) or two with
  different policy noise (200 pairs, phase-5 design).
* **Truths:** the reference succeeds with ``p_reference``; the test condition with ``p_test`` (equal to the
  reference, at the level of asynchronous execution without RTC, or 10 and 15 pp below the reference).
* **Dependence:** Gaussian copula. A latent cluster effect (task x init state) is shared by every episode of
  that init state in both conditions, a latent seed effect by both conditions of one episode; the rest is
  independent per condition. ``rho_total`` = cluster share + seed share is the latent correlation of the two
  conditions on one seed. Phases 3 and 4 measured phi coefficients of 0.17-0.58 between conditions on the same
  seeds (tetrachoric roughly 0.3-0.8), hence the grid 0.3 / 0.5 / 0.7, half of it on the cluster.

    OMP_NUM_THREADS=1 python -m kortyn.power_phase5 --out results/phase5/power.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from kortyn.rtc_gate import GATE_MARGIN, GateError, PairedSample, evaluate_gate

LOG = logging.getLogger("kortyn.power_phase5")

TASKS = 2
INIT_STATES = 50


@dataclass(frozen=True)
class GateScenario:
    """One cell of the power grid."""

    replicates: int  # episodes per init state (1: 100 pairs, 2: 200 pairs)
    p_reference: float
    p_test: float
    rho_total: float
    cluster_share: float = 0.5  # part of rho_total that lies on the init state

    def __post_init__(self) -> None:
        if self.replicates < 1:
            raise GateError("need at least one episode per init state")
        if not (0.0 < self.p_reference < 1.0 and 0.0 < self.p_test < 1.0):
            raise GateError("success rates must lie in (0, 1)")
        if not 0.0 <= self.rho_total < 1.0 or not 0.0 <= self.cluster_share <= 1.0:
            raise GateError("need 0 <= rho_total < 1 and 0 <= cluster_share <= 1")

    @property
    def pairs(self) -> int:
        return TASKS * INIT_STATES * self.replicates


def simulate_sample(scenario: GateScenario, rng: np.random.Generator) -> PairedSample:
    """One data set of the design: reference and test outcomes of every episode."""
    clusters = TASKS * INIT_STATES
    n = clusters * scenario.replicates
    rho_cluster = scenario.rho_total * scenario.cluster_share
    rho_seed = scenario.rho_total - rho_cluster
    cluster_ids = np.repeat(np.arange(clusters), scenario.replicates)
    shared = math.sqrt(rho_cluster) * rng.standard_normal(clusters)[cluster_ids] + math.sqrt(
        rho_seed
    ) * rng.standard_normal(n)
    rest = math.sqrt(1.0 - scenario.rho_total)
    reference = (shared + rest * rng.standard_normal(n) < norm.ppf(scenario.p_reference)).astype(float)
    test = (shared + rest * rng.standard_normal(n) < norm.ppf(scenario.p_test)).astype(float)
    return PairedSample(
        reference=reference,
        test=test,
        clusters=cluster_ids.astype(np.int64),
        strata=(cluster_ids // INIT_STATES).astype(np.int64),
    )


def pass_rate(scenario: GateScenario, simulations: int, draws: int, seed: int) -> float:
    """Share of simulated data sets on which the gate passes."""
    if simulations < 1:
        raise GateError("need at least one simulation")
    rng = np.random.default_rng(seed)
    passes = 0
    for _ in range(simulations):
        result = evaluate_gate(simulate_sample(scenario, rng), margin=GATE_MARGIN, draws=draws, rng=rng)
        passes += int(result.passed)
    return passes / simulations


def _job(args: tuple[GateScenario, int, int, int]) -> float:
    scenario, simulations, draws, seed = args
    return pass_rate(scenario, simulations, draws, seed)


def grid(p_reference: float, async_gap: float) -> list[GateScenario]:
    """The pre-registered power grid: both designs x three dependence levels x four truths."""
    truths = (p_reference, p_reference - async_gap, p_reference - 0.10, p_reference - 0.15)
    return [
        GateScenario(replicates=replicates, p_reference=p_reference, p_test=p_test, rho_total=rho)
        for replicates in (1, 2)
        for rho in (0.3, 0.5, 0.7)
        for p_test in truths
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-5 power analysis of the 0-ms gate (simulation).")
    parser.add_argument("--out", default="results/phase5/power.json")
    parser.add_argument("--simulations", type=int, default=2000, help="data sets per scenario")
    parser.add_argument("--draws", type=int, default=2000, help="cluster-bootstrap draws per data set")
    parser.add_argument("--p-reference", type=float, default=0.53, help="sync25 (phase 3: 53 %%)")
    parser.add_argument(
        "--async-gap", type=float, default=0.06, help="asynchronous 0 ms below sync25 (phases 3-4: 6-7 pp)"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        if args.simulations < 1 or args.draws < 100 or args.workers < 1:
            raise GateError("need simulations >= 1, draws >= 100 and workers >= 1")
        scenarios = grid(args.p_reference, args.async_gap)
    except GateError as exc:
        LOG.error("%s", exc)
        return 2
    jobs = [(s, args.simulations, args.draws, args.seed + 7919 * i) for i, s in enumerate(scenarios)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rates = list(pool.map(_job, jobs))
    rows: list[dict[str, Any]] = []
    for scenario, rate in zip(scenarios, rates, strict=True):
        rows.append({**asdict(scenario), "pairs": scenario.pairs, "pass_rate": rate})
        LOG.info(
            "%3d pairs, rho %.1f, p_test %.2f: gate passes in %.1f %%",
            scenario.pairs,
            scenario.rho_total,
            scenario.p_test,
            100.0 * rate,
        )
    document = {
        "gate": {"margin": GATE_MARGIN, "alpha": 0.05, "confidence": 0.95},
        "simulations": args.simulations,
        "draws": args.draws,
        "seed": args.seed,
        "p_reference": args.p_reference,
        "async_gap": args.async_gap,
        "scenarios": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    LOG.info("written %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
