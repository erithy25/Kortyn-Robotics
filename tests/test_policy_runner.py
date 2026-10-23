from collections import deque

import numpy as np
import pytest
import torch

from kortyn.config import BrainProfile, ConfigError
from kortyn.policy_runner import ObservationError, PolicyRunner, resolve_device

PROFILE = BrainProfile(noise_scheduler_type="DDIM", num_inference_steps=4)


@pytest.fixture(scope="module")
def runner(tiny_checkpoint):
    return PolicyRunner(tiny_checkpoint, PROFILE, device="cpu", torch_threads=1)


def random_history(rng, steps=2):
    return {
        "observation.image": rng.integers(0, 256, size=(steps, 32, 32, 3), dtype=np.uint8),
        "observation.state": rng.uniform(0, 512, size=(steps, 2)).astype(np.float32),
    }


def test_info_reflects_checkpoint_and_profile(runner):
    info = runner.info
    assert info.policy_type == "diffusion"
    assert info.device == "cpu"
    assert info.n_obs_steps == 2
    assert info.max_actions == 15  # horizon 16 - n_obs_steps 2 + 1
    assert info.action_dim == 2
    assert info.sampler == "DDIM-4"
    features = {f.key: (f.kind, f.shape) for f in info.input_features}
    assert features == {"observation.image": ("VISUAL", (3, 32, 32)), "observation.state": ("STATE", (2,))}


def test_chunk_longer_than_the_horizon_allows_is_rejected(tiny_checkpoint):
    with pytest.raises(ConfigError, match="at most 15"):
        PolicyRunner(tiny_checkpoint, BrainProfile(n_action_steps=16), device="cpu")


def test_unknown_or_unavailable_device_is_rejected():
    with pytest.raises(ConfigError, match="unsupported device"):
        resolve_device("tpu")
    if not torch.backends.mps.is_available():
        with pytest.raises(ConfigError, match="MPS is not available"):
            resolve_device("mps")
    assert resolve_device("cpu") == "cpu"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda o: o.pop("observation.state"), "missing observation keys"),
        (lambda o: o.update(extra=np.zeros(1, dtype=np.float32)), "unexpected observation keys"),
        (lambda o: o.update({"observation.state": np.zeros((1, 2), np.float32)}), r"expected shape \(2, 2\)"),
        (lambda o: o.update({"observation.image": np.zeros((2, 32, 32, 3), np.float32)}), "uint8"),
        (lambda o: o.update({"observation.image": np.zeros((2, 96, 96, 3), np.uint8)}), r"\(2, 32, 32, 3\)"),
        (lambda o: o.update({"observation.state": np.zeros((2, 2), np.int64)}), "float32"),
    ],
)
def test_validate_rejects_malformed_observations(runner, mutate, message):
    observation = runner.dummy_observation()
    mutate(observation)
    with pytest.raises(ObservationError, match=message):
        runner.validate(observation)


def test_same_seed_gives_identical_chunk_and_other_seed_differs(runner):
    observation = random_history(np.random.default_rng(1))
    first = runner.infer(observation, seed=42, max_actions=15)
    second = runner.infer(observation, seed=42, max_actions=15)
    third = runner.infer(observation, seed=43, max_actions=15)
    assert first.actions.shape == (15, 2)
    assert first.actions.dtype == np.float32
    np.testing.assert_array_equal(first.actions, second.actions)
    assert not np.allclose(first.actions, third.actions)
    assert first.inference_ms > 0


def test_chunk_prefix_does_not_depend_on_requested_length(runner):
    observation = random_history(np.random.default_rng(2))
    long = runner.infer(observation, seed=7, max_actions=15).actions
    short = runner.infer(observation, seed=7, max_actions=5).actions
    assert short.shape == (5, 2)
    np.testing.assert_allclose(short, long[:5], rtol=1e-6, atol=1e-5)


def test_actions_are_unnormalized_into_the_dataset_range(runner):
    actions = runner.infer(random_history(np.random.default_rng(3)), seed=1, max_actions=15).actions
    # clip_sample keeps the normalized sample in [-1, 1] -> MIN_MAX un-normalization maps into [min, max]
    assert np.all(actions[:, 0] >= 10.0 - 1e-3)
    assert np.all(actions[:, 0] <= 500.0 + 1e-3)
    assert np.all(actions[:, 1] >= 20.0 - 1e-3)
    assert np.all(actions[:, 1] <= 480.0 + 1e-3)


def test_invalid_max_actions_is_rejected(runner):
    with pytest.raises(ObservationError, match="max_actions"):
        runner.infer(runner.dummy_observation(), seed=0, max_actions=16)


def test_matches_lerobot_select_action(tiny_checkpoint, runner):
    """Golden test: Kortyn's stateless history path == LeRobot's own select_action rollout semantics.

    Reference: LeRobot eval feeds one preprocessed observation per step into ``select_action``, which keeps
    the observation/action queues internally and re-plans every ``n_action_steps`` (8) steps. Kortyn sends
    the explicit 2-frame history (padded at the start) and executes the first 8 actions of each chunk.
    With identical initial noise per chunk, both must produce the same actions.
    """
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    reference = DiffusionPolicy.from_pretrained(tiny_checkpoint)
    assert reference.config.n_action_steps == 8
    assert reference.config.num_inference_steps == 4
    pre, post = make_pre_post_processors(reference.config, pretrained_path=str(tiny_checkpoint))
    reference.reset()

    rng = np.random.default_rng(123)
    frames = rng.integers(0, 256, size=(20, 32, 32, 3), dtype=np.uint8)
    states = rng.uniform(0, 512, size=(20, 2))  # float64 like gym-pusht's agent_pos
    history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=2)
    chunk = np.empty((0, 2), dtype=np.float32)
    for t in range(20):
        seed = 1000 + t // 8
        noise = torch.randn((1, 16, 2), generator=torch.Generator().manual_seed(seed))
        raw = {"pixels": frames[t][None], "agent_pos": states[t][None]}
        expected = post(reference.select_action(pre(preprocess_observation(raw)), noise=noise))

        if t == 0:
            history.extend([(frames[0], states[0])] * 2)
        else:
            history.append((frames[t], states[t]))
        if t % 8 == 0:
            observation = {
                "observation.image": np.stack([f for f, _ in history]),
                "observation.state": np.stack([s for _, s in history]).astype(np.float32),
            }
            chunk = runner.infer(observation, seed=seed, max_actions=8).actions
        np.testing.assert_allclose(chunk[t % 8], expected[0].numpy(), rtol=1e-5, atol=1e-4)
