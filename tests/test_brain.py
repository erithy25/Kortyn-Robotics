"""The Brain gRPC server end to end (in-process server, real channel, tiny policy)."""

import socket

import grpc
import numpy as np
import pytest

from kortyn.brain import build_server
from kortyn.codec import decode_tensor, encode_array, encode_observation
from kortyn.config import BrainConfig, BrainProfile
from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.proto import kortyn_brain_pb2_grpc as pb_grpc

PROFILE = BrainProfile(noise_scheduler_type="DDIM", num_inference_steps=4)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def brain(tiny_checkpoint):
    config = BrainConfig(host="127.0.0.1", port=free_port(), torch_threads=1, warmup_inferences=1)
    server, worker, info = build_server(config, PROFILE, checkpoint_dir=tiny_checkpoint)
    server.start()
    channel = grpc.insecure_channel(config.address, options=[("grpc.enable_http_proxy", 0)])
    grpc.channel_ready_future(channel).result(timeout=10)
    yield pb_grpc.KortynBrainStub(channel), info, config
    channel.close()
    server.stop(0).wait()
    worker.shutdown()


def make_request(observation, seed=1, max_actions=0, request_id=7, timestep=40):
    return pb.InferRequest(
        client_id="test",
        episode_id=1000,
        request_id=request_id,
        obs_timestep=timestep,
        seed=seed,
        max_actions=max_actions,
        observation=encode_observation(observation),
    )


def valid_observation(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observation.image": rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8),
        "observation.state": rng.uniform(0, 512, size=(2, 2)).astype(np.float32),
    }


def test_get_info(brain):
    stub, info, _ = brain
    reply = stub.GetInfo(pb.GetInfoRequest(), timeout=5)
    assert reply.policy_type == "diffusion"
    assert reply.n_obs_steps == 2
    assert reply.max_actions == 15
    assert reply.action_dim == 2
    assert reply.sampler == "DDIM-4"
    assert reply.warmup_inference_ms > 0
    assert {f.key: tuple(f.shape) for f in reply.input_features} == {
        "observation.image": (3, 32, 32),
        "observation.state": (2,),
    }
    assert info.max_actions == reply.max_actions
    assert reply.supports_guidance


def test_infer_round_trip_with_timings(brain):
    stub, _, _ = brain
    reply = stub.Infer(make_request(valid_observation()), timeout=10)
    actions = decode_tensor(reply.actions)
    assert actions.shape == (15, 2)
    assert actions.dtype == np.float32
    assert reply.request_id == 7
    assert reply.start_timestep == 40
    assert reply.server_inference_ms > 0
    assert reply.server_queue_ms >= 0
    assert reply.server_total_ms >= reply.server_inference_ms


def test_infer_is_deterministic_per_seed_and_honours_max_actions(brain):
    stub, _, _ = brain
    first = decode_tensor(stub.Infer(make_request(valid_observation(), seed=9), timeout=10).actions)
    second = decode_tensor(
        stub.Infer(make_request(valid_observation(), seed=9, max_actions=4), timeout=10).actions
    )
    assert second.shape == (4, 2)
    np.testing.assert_allclose(second, first[:4], rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize(
    ("observation", "max_actions", "message"),
    [
        ({"observation.state": np.zeros((2, 2), np.float32)}, 0, "missing observation keys"),
        (
            {
                "observation.image": np.zeros((1, 32, 32, 3), np.uint8),
                "observation.state": np.zeros((1, 2), np.float32),
            },
            0,
            "n_obs_steps=2",
        ),
        (None, 99, "exceeds the Brain maximum"),
    ],
)
def test_invalid_requests_are_rejected_with_invalid_argument(brain, observation, max_actions, message):
    stub, _, _ = brain
    request = make_request(
        observation if observation is not None else valid_observation(), max_actions=max_actions
    )
    with pytest.raises(grpc.RpcError) as error:
        stub.Infer(request, timeout=10)
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert message in error.value.details()


def test_corrupt_tensor_is_rejected(brain):
    stub, _, _ = brain
    request = make_request(valid_observation())
    request.observation["observation.state"].data = b"\x00" * 3
    with pytest.raises(grpc.RpcError) as error:
        stub.Infer(request, timeout=10)
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "observation.state" in error.value.details()


def test_occupied_port_fails_loudly(tiny_checkpoint, brain):
    _, _, running = brain
    clash = BrainConfig(host="127.0.0.1", port=running.port, torch_threads=1, warmup_inferences=0)
    with pytest.raises(OSError, match="cannot bind"):
        build_server(clash, PROFILE, checkpoint_dir=tiny_checkpoint)


def test_guided_inference_reproduces_the_frozen_actions(brain):
    stub, _, _ = brain
    frozen = np.array([[100.0, 120.0], [110.0, 130.0], [120.0, 140.0], [130.0, 150.0]], dtype=np.float32)
    request = make_request(valid_observation(4), seed=3, timestep=40)
    request.guide_start_timestep = 39  # the previous tick's action, then steps 40, 41, 42
    request.guide_actions.CopyFrom(encode_array(frozen))
    request.guide_weights.extend([1.0, 1.0, 1.0, 1.0])
    actions = decode_tensor(stub.Infer(request, timeout=10).actions)
    np.testing.assert_allclose(actions[:3], frozen[1:], atol=1e-2)


def test_malformed_guides_are_rejected_with_invalid_argument(brain):
    stub, _, _ = brain
    request = make_request(valid_observation(), timestep=40)
    request.guide_start_timestep = 39
    request.guide_actions.CopyFrom(encode_array(np.zeros((2, 2), np.float32)))
    request.guide_weights.extend([1.0])
    with pytest.raises(grpc.RpcError) as error:
        stub.Infer(request, timeout=10)
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "one weight per action" in error.value.details()


def test_stats_count_requests_and_inference_time(brain):
    stub, _, _ = brain
    before = stub.GetStats(pb.GetStatsRequest(), timeout=5)
    for seed in range(3):
        stub.Infer(make_request(valid_observation(), seed=seed), timeout=10)
    after = stub.GetStats(pb.GetStatsRequest(), timeout=5)
    assert after.requests - before.requests == 3
    assert after.inferences - before.inferences == 3
    assert after.busy_ms > before.busy_ms
    assert after.queue_wait_ms >= before.queue_wait_ms
    assert after.uptime_ms > before.uptime_ms
    assert after.abandoned == before.abandoned
