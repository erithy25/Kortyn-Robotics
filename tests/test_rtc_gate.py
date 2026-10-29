"""Phase-5 gate (kortyn.rtc_gate) and its power simulation (kortyn.power_phase5)."""

from __future__ import annotations

import numpy as np
import pytest

from kortyn.power_phase5 import GateScenario, grid, pass_rate, simulate_sample
from kortyn.rtc_gate import (
    FAILED,
    PASSED,
    PASSED_BETTER,
    GateError,
    PairedSample,
    cluster_bootstrap_means,
    evaluate_gate,
    paired_sample,
    subset,
)


def _sample(reference: list[int], test: list[int], clusters: list[int], strata: list[int]) -> PairedSample:
    return PairedSample(
        reference=np.array(reference, dtype=float),
        test=np.array(test, dtype=float),
        clusters=np.array(clusters, dtype=np.int64),
        strata=np.array(strata, dtype=np.int64),
    )


def _slots(seed: int) -> tuple[int, int]:
    """Phase-5 layout: seeds 1000-1049 task 8, 1050-1099 task 0, the same again from 1100."""
    index = (seed - 1000) % 100
    return (8 if index < 50 else 0, index % 50)


def test_paired_sample_pairs_common_seeds_and_clusters_replicates() -> None:
    reference = {1000: True, 1001: False, 1100: True, 1050: False, 1999: True}
    test = {1000: False, 1001: False, 1100: True, 1050: True}
    sample = paired_sample(reference, test, _slots)
    assert sample.reference.tolist() == [1.0, 0.0, 0.0, 1.0]  # seeds 1000, 1001, 1050, 1100
    assert sample.test.tolist() == [0.0, 0.0, 1.0, 1.0]
    # seeds 1000 and 1100 share task 8 / init state 0
    assert sample.clusters[0] == sample.clusters[3]
    assert len(set(sample.clusters.tolist())) == 3
    assert sample.strata.tolist() == [8, 8, 0, 8]


def test_paired_sample_without_common_seeds_fails() -> None:
    with pytest.raises(GateError):
        paired_sample({1: True}, {2: True}, _slots)


def test_sample_validation() -> None:
    with pytest.raises(GateError):
        _sample([1, 0], [1], [0, 1], [0, 0])
    with pytest.raises(GateError):
        _sample([1, 2], [1, 0], [0, 1], [0, 0])
    with pytest.raises(GateError):  # one cluster in two strata
        _sample([1, 0], [1, 0], [0, 0], [0, 1])
    with pytest.raises(GateError):
        _sample([], [], [], [])


def test_cluster_bootstrap_keeps_constant_differences() -> None:
    differences = np.array([0.5, 0.5, 0.5, 0.5])
    means = cluster_bootstrap_means(
        differences, np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), 200, np.random.default_rng(0)
    )
    assert np.allclose(means, 0.5)


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    # cluster 0 holds +1 twice, cluster 1 holds -1 twice: every resample mean is -1, 0 or +1
    differences = np.array([1.0, 1.0, -1.0, -1.0])
    means = cluster_bootstrap_means(
        differences, np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), 500, np.random.default_rng(1)
    )
    assert set(np.round(means, 9).tolist()) <= {-1.0, 0.0, 1.0}


def test_cluster_bootstrap_stratifies() -> None:
    # stratum 0 always +1, stratum 1 always -1, equal sizes: the mean never moves
    differences = np.array([1.0, 1.0, -1.0, -1.0])
    means = cluster_bootstrap_means(
        differences, np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1]), 300, np.random.default_rng(2)
    )
    assert np.allclose(means, 0.0)


def test_cluster_bootstrap_needs_draws() -> None:
    with pytest.raises(GateError):
        cluster_bootstrap_means(np.zeros(2), np.array([0, 1]), np.array([0, 0]), 0, np.random.default_rng(0))


def _balanced(n: int, only_reference: int, only_test: int, both: int) -> PairedSample:
    reference = [1] * only_reference + [0] * only_test + [1] * both
    test = [0] * only_reference + [1] * only_test + [1] * both
    reference += [0] * (n - len(reference))
    test += [0] * (n - len(test))
    clusters = [i // 2 for i in range(n)]
    strata = [0 if c < n // 4 else 1 for c in clusters]
    return _sample(reference, test, clusters, strata)


def test_gate_passes_on_equal_conditions() -> None:
    result = evaluate_gate(_balanced(200, 20, 20, 80), draws=2000, seed=3)
    assert result.verdict == PASSED
    assert result.passed
    assert result.difference == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.interval[0] < 0.0 < result.interval[1]
    assert result.pairs == 200
    assert result.clusters == 100


def test_gate_fails_on_a_large_loss() -> None:
    result = evaluate_gate(_balanced(200, 50, 10, 60), draws=2000, seed=4)
    assert result.verdict == FAILED
    assert not result.passed
    assert result.difference == pytest.approx(-0.2)
    assert result.p_value < 0.05


def test_gate_fails_when_only_the_interval_is_too_wide() -> None:
    # 24 / 12 discordant pairs of 100: p = 0.065 (not significant), but the interval reaches below -15 pp
    result = evaluate_gate(_balanced(100, 24, 12, 30), draws=4000, seed=5)
    assert result.p_value >= 0.05
    assert result.interval[0] < -0.15
    assert result.verdict == FAILED


def test_gate_passes_when_significantly_better() -> None:
    result = evaluate_gate(_balanced(200, 10, 50, 60), draws=2000, seed=6)
    assert result.p_value < 0.05
    assert result.difference > 0
    assert result.verdict == PASSED_BETTER
    assert result.passed


def test_gate_settings_are_checked() -> None:
    sample = _balanced(20, 2, 2, 6)
    with pytest.raises(GateError):
        evaluate_gate(sample, margin=0.0)
    with pytest.raises(GateError):
        evaluate_gate(sample, confidence=0.4)


def test_gate_result_json_and_determinism() -> None:
    sample = _balanced(200, 30, 20, 70)
    first = evaluate_gate(sample, draws=1000, seed=7)
    second = evaluate_gate(sample, draws=1000, seed=7)
    assert first == second
    data = first.to_json()
    assert isinstance(data["interval"], list)
    assert data["verdict"] == first.verdict


def test_subset_selects_pairs() -> None:
    sample = _balanced(8, 2, 1, 2)
    part = subset(sample, [True, False] * 4)
    assert part.reference.size == 4
    with pytest.raises(GateError):
        subset(sample, [False] * 8)


def test_simulated_sample_has_the_design_shape() -> None:
    scenario = GateScenario(replicates=2, p_reference=0.5, p_test=0.5, rho_total=0.5)
    sample = simulate_sample(scenario, np.random.default_rng(8))
    assert sample.reference.size == 200 == scenario.pairs
    assert np.unique(sample.clusters).size == 100
    assert np.unique(sample.strata).tolist() == [0, 1]
    # both episodes of an init state share the cluster
    assert np.all(sample.clusters[0::2] == sample.clusters[1::2])


def test_simulated_rates_match_the_truth() -> None:
    scenario = GateScenario(replicates=2, p_reference=0.53, p_test=0.38, rho_total=0.5)
    rng = np.random.default_rng(9)
    reference = np.concatenate([simulate_sample(scenario, rng).reference for _ in range(60)])
    test = np.concatenate([simulate_sample(scenario, rng).test for _ in range(60)])
    assert reference.mean() == pytest.approx(0.53, abs=0.02)
    assert test.mean() == pytest.approx(0.38, abs=0.02)


def test_power_is_high_for_equal_and_low_for_much_worse_conditions() -> None:
    equal = pass_rate(GateScenario(replicates=2, p_reference=0.53, p_test=0.53, rho_total=0.5), 150, 400, 10)
    worse = pass_rate(GateScenario(replicates=2, p_reference=0.53, p_test=0.30, rho_total=0.5), 150, 400, 11)
    assert equal > 0.85
    assert worse < 0.05


def test_scenario_validation_and_grid() -> None:
    with pytest.raises(GateError):
        GateScenario(replicates=0, p_reference=0.5, p_test=0.5, rho_total=0.5)
    with pytest.raises(GateError):
        GateScenario(replicates=1, p_reference=1.0, p_test=0.5, rho_total=0.5)
    with pytest.raises(GateError):
        GateScenario(replicates=1, p_reference=0.5, p_test=0.5, rho_total=1.0)
    with pytest.raises(GateError):
        pass_rate(GateScenario(replicates=1, p_reference=0.5, p_test=0.5, rho_total=0.5), 0, 100, 0)
    cells = grid(0.53, 0.06)
    assert len(cells) == 24
    assert {cell.pairs for cell in cells} == {100, 200}
