import json

import numpy as np
import pytest

from kortyn.metrics import (
    EpisodeRecord,
    RequestRecord,
    TickSeries,
    append_jsonl,
    bootstrap_ci,
    final_episodes,
    ipdv,
    max_reward,
    mcnemar_exact,
    paired_mean_difference,
    paired_outcomes,
    percentile,
    read_jsonl,
    runs_of_true,
    summarize_run,
    wilson_interval,
)


def episode(seed, success=True, attempt=0, valid=True, steps=100, starved=(), **overrides):
    flags = [1 if i in starved else 0 for i in range(steps)]
    runs = runs_of_true(flags)
    defaults = dict(
        run="gut",
        profile="p",
        scenario="gut",
        mode="realtime",
        seed=seed,
        attempt=attempt,
        success=success,
        steps=steps,
        fps=10.0,
        wall_duration_s=steps / 10.0,
        max_coverage=1.0,
        sum_reward=10.0,
        n_requests=10,
        n_failed_requests=0,
        starved_ticks=sum(flags),
        starvation_events=len(runs),
        starved_ticks_compute=0,
        starved_ticks_network=sum(flags),
        starved_ticks_none_pending=0,
        tick_overruns=0,
        max_tick_lateness_ms=1.0,
        valid=valid,
        ticks=TickSeries(
            queue_len=[5] * steps,
            starved=flags,
            lateness_ms=[0.1] * steps,
            action_age=[2] * steps,
            pending_request=[-1] * steps,
        ),
    )
    defaults.update(overrides)
    return EpisodeRecord(**defaults)


def request(seed, request_id, t_obs, e2e_ms, server_ms, attempt=0, phase="realtime", status="ok"):
    t_resp = t_obs + e2e_ms / 1000.0
    return RequestRecord(
        run="gut",
        profile="p",
        seed=seed,
        attempt=attempt,
        request_id=request_id,
        phase=phase,
        status=status,
        obs_timestep=int(t_obs * 10),
        t_obs_s=t_obs,
        t_send_s=t_obs,
        request_bytes=55_400,
        queue_at_send=6,
        t_resp_s=t_resp if status == "ok" else None,
        t_first_exec_s=t_resp + 0.05 if status == "ok" else None,
        response_bytes=160 if status == "ok" else None,
        server_total_ms=server_ms if status == "ok" else None,
        server_inference_ms=server_ms - 1 if status == "ok" else None,
    )


def test_percentile_matches_numpy_and_handles_empty():
    values = [5.0, 1.0, 9.0, 3.0]
    assert percentile(values, 50) == pytest.approx(float(np.percentile(values, 50)))
    assert percentile([], 99) is None


def test_ipdv_and_runs():
    assert ipdv([10.0, 12.0, 9.0]) == [2.0, 3.0]
    assert ipdv([1.0]) == []
    assert runs_of_true([0, 1, 1, 0, 1, 0, 0, 1, 1, 1]) == [(1, 2), (4, 1), (7, 3)]
    assert runs_of_true([]) == []
    assert runs_of_true([True, True]) == [(0, 2)]


def test_wilson_interval_known_values():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4)
    assert high == pytest.approx(0.9433, abs=1e-4)
    assert wilson_interval(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(11, 10)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(11, 5) == pytest.approx(0.2101, abs=1e-4)
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0


def test_paired_outcomes_only_counts_common_seeds():
    a = {1: True, 2: True, 3: False, 4: True}
    b = {1: True, 2: False, 3: True, 5: False}
    assert paired_outcomes(a, b) == (3, 1, 1)


def test_bootstrap_ci_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(0)
    groups = [list(rng.normal(100, 10, size=20)) for _ in range(30)]
    ci1 = bootstrap_ci(groups, lambda v: percentile(v, 50), n_boot=300, seed=1)
    ci2 = bootstrap_ci(groups, lambda v: percentile(v, 50), n_boot=300, seed=1)
    assert ci1 == ci2
    assert ci1 is not None
    assert ci1[0] < 100 < ci1[1]
    assert bootstrap_ci([[1.0]], lambda v: percentile(v, 50)) is None


def test_request_record_derived_latencies():
    record = request(1000, 3, t_obs=2.0, e2e_ms=450.0, server_ms=400.0)
    record.t_send_s = 2.001
    assert record.e2e_ms == pytest.approx(450.0)
    assert record.rtt_ms == pytest.approx(449.0)
    assert record.network_ms == pytest.approx(49.0)
    assert record.reaction_ms == pytest.approx(500.0)
    failed = request(1000, 4, t_obs=3.0, e2e_ms=0.0, server_ms=0.0, status="timeout")
    assert failed.e2e_ms is None
    assert failed.network_ms is None


def test_records_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "x" / "episodes.jsonl"
    original = episode(1000, starved=(3, 4, 9))
    append_jsonl(path, [original.to_json()])
    (loaded,) = [EpisodeRecord.from_json(d) for d in read_jsonl(path)]
    assert loaded == original
    assert loaded.duration_s == pytest.approx(10.0)
    req = request(1000, 1, 0.5, 300.0, 250.0)
    append_jsonl(tmp_path / "r.jsonl", [req.to_json()])
    (raw,) = list(read_jsonl(tmp_path / "r.jsonl"))
    assert raw["e2e_ms"] == pytest.approx(300.0)
    assert RequestRecord.from_json(raw) == req


def test_read_jsonl_reports_corrupt_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"a": 1}) + "\n{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        list(read_jsonl(path))
    assert list(read_jsonl(tmp_path / "missing.jsonl")) == []


def test_final_episodes_prefers_first_valid_attempt():
    records = [
        episode(1, attempt=0, valid=False, success=False),
        episode(1, attempt=1, valid=True, success=True),
        episode(2, attempt=0, valid=False, success=False),
        episode(2, attempt=1, valid=False, success=True),
        episode(3, attempt=0, valid=True, success=False),
    ]
    chosen, invalid = final_episodes(records)
    assert [(e.seed, e.attempt) for e in chosen] == [(1, 1), (2, 1), (3, 0)]
    assert invalid == 3


def test_summarize_run_aggregates_success_latency_jitter_and_starvation():
    episodes = [
        episode(1000, success=True, steps=100, starved=(10, 11, 50)),  # 2 events, 3 ticks
        episode(1001, success=False, steps=200, starved=(), max_coverage=0.76),
        episode(1002, success=True, steps=300, starved=(5,)),
    ]
    requests = [
        request(1000, 1, 0.0, 400.0, 350.0, phase="warmup"),  # excluded from latency stats
        request(1000, 2, 1.0, 400.0, 350.0),
        request(1000, 3, 2.0, 420.0, 350.0),
        request(1000, 4, 3.0, 390.0, 350.0),
        request(1001, 2, 1.0, 600.0, 350.0),
        request(1001, 3, 2.0, 0.0, 0.0, status="timeout"),
        request(1002, 2, 1.0, 500.0, 350.0),
    ]
    summary = summarize_run(episodes, requests)
    assert summary.episodes == 3
    assert summary.successes == 2
    assert summary.success_rate == pytest.approx(2 / 3)
    assert summary.requests == 5
    assert summary.failed_requests == 1
    assert summary.e2e_p50_ms == pytest.approx(420.0)
    # network part = e2e - server: [50, 70, 40] in episode 1000 -> |Δ| = [20, 30]
    assert summary.jitter_ipdv_ms == pytest.approx(25.0)
    minutes = (10 + 20 + 30) / 60
    assert summary.starvation_events_per_min == pytest.approx(3 / minutes)
    assert summary.starved_ticks_per_min == pytest.approx(4 / minutes)
    assert summary.starved_time_fraction == pytest.approx(4 / 600)
    assert summary.request_bytes_mean == pytest.approx(55_400)
    assert summary.success_by_seed == {1000: True, 1001: False, 1002: True}
    assert summary.max_reward_by_seed == pytest.approx({1000: 1.0, 1001: 0.8, 1002: 1.0})
    assert summary.max_reward_mean == pytest.approx(2.8 / 3)


def test_max_reward_follows_the_pusht_reward_definition():
    assert max_reward(0.475) == pytest.approx(0.5)
    assert max_reward(0.95) == pytest.approx(1.0)
    assert max_reward(0.99) == 1.0  # capped like gym-pusht's clipped reward
    assert max_reward(-0.1) == 0.0


def test_paired_mean_difference_uses_common_seeds_only():
    a = {1: 0.5, 2: 1.0, 3: 0.9}
    b = {1: 0.7, 2: 0.8, 4: 0.0}
    result = paired_mean_difference(a, b)
    assert result is not None
    delta, bounds = result
    assert delta == pytest.approx(0.0)  # seeds 1 and 2: +0.2 and -0.2
    assert bounds is not None
    assert bounds[0] == pytest.approx(-0.2)
    assert bounds[1] == pytest.approx(0.2)
    assert paired_mean_difference({1: 1.0}, {2: 1.0}) is None
    shifted = paired_mean_difference({s: 0.9 for s in range(30)}, {s: 0.8 for s in range(30)})
    assert shifted is not None
    assert shifted[0] == pytest.approx(-0.1)
    assert shifted[1] == pytest.approx((-0.1, -0.1))


def test_summarize_run_rejects_empty_input():
    with pytest.raises(ValueError, match="without episodes"):
        summarize_run([], [])
