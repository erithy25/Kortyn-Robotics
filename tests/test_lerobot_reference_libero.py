"""lerobot-eval on LIBERO (phase 3): arguments for a task subset and the mapping back to Kortyn seeds."""

from __future__ import annotations

from typing import Any

import pytest

from kortyn.config import ConfigError, LiberoEnvConfig
from kortyn.lerobot_reference import libero_env_args, libero_success_by_seed


def eval_info(per_task: dict[int, list[bool]]) -> dict[str, Any]:
    return {
        "per_task": [
            {"task_id": task_id, "metrics": {"successes": successes}}
            for task_id, successes in per_task.items()
        ]
    }


def test_env_args_name_only_the_evaluated_tasks() -> None:
    libero = LiberoEnvConfig(task_ids=[0, 3, 7], episodes_per_task=2)
    args = libero_env_args(libero, [3, 7])
    assert "--env.task_ids=[3,7]" in args
    assert "--env.task=libero_10" in args
    assert "--env.observation_height=360" in args
    assert "--env.control_mode=relative" in args


def test_success_maps_episodes_to_kortyn_seeds_of_the_full_config() -> None:
    libero = LiberoEnvConfig(task_ids=[0, 3, 7], episodes_per_task=2, seed_base=1000)
    info = eval_info({3: [True, False], 7: [False, True]})
    # task 3 is the second task of the config: Kortyn seeds 1002, 1003; task 7: 1004, 1005
    assert libero_success_by_seed(info, libero, [3, 7]) == {
        "1002": True,
        "1003": False,
        "1004": False,
        "1005": True,
    }


def test_success_of_all_tasks_by_default() -> None:
    libero = LiberoEnvConfig(task_ids=[1, 2], episodes_per_task=1, seed_base=1000)
    info = eval_info({1: [True], 2: [False]})
    assert libero_success_by_seed(info, libero) == {"1000": True, "1001": False}


@pytest.mark.parametrize("per_task", [{0: [True]}, {0: [True, True], 3: [False]}])
def test_success_rejects_missing_or_incomplete_tasks(per_task: dict[int, list[bool]]) -> None:
    libero = LiberoEnvConfig(task_ids=[0, 3], episodes_per_task=2)
    with pytest.raises(ConfigError, match="expected 2"):
        libero_success_by_seed(eval_info(per_task), libero)
