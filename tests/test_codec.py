import numpy as np
import pytest

from kortyn.codec import (
    MAX_TENSOR_BYTES,
    CodecError,
    decode_observation,
    decode_tensor,
    encode_array,
    encode_observation,
)
from kortyn.proto import kortyn_brain_pb2 as pb


@pytest.mark.parametrize(
    "array",
    [
        np.arange(2 * 96 * 96 * 3, dtype=np.uint8).reshape(2, 96, 96, 3),
        np.array([[1.5, -2.25], [3.0, 4.0]], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float64),
        np.array(7, dtype=np.int64),
        np.zeros((0, 2), dtype=np.float32),
    ],
)
def test_round_trip_preserves_values_dtype_and_shape(array):
    message = encode_array(array)
    decoded = decode_tensor(pb.Tensor.FromString(message.SerializeToString()))
    assert decoded.dtype == array.dtype
    assert decoded.shape == array.shape
    np.testing.assert_array_equal(decoded, array)


def test_non_contiguous_input_is_encoded_correctly():
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    view = base[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(decode_tensor(encode_array(view)), view)


def test_unsupported_dtype_is_rejected_on_encode():
    with pytest.raises(CodecError, match="not supported"):
        encode_array(np.array([1 + 2j]))


def test_unsupported_dtype_is_rejected_on_decode():
    with pytest.raises(CodecError, match="not supported"):
        decode_tensor(pb.Tensor(dtype="object", shape=[1], data=b"x"))


def test_size_mismatch_is_rejected():
    tensor = pb.Tensor(dtype="float32", shape=[2, 2], data=b"\x00" * 15)
    with pytest.raises(CodecError, match="require 16 bytes"):
        decode_tensor(tensor)


def test_negative_dimension_is_rejected():
    with pytest.raises(CodecError, match="negative"):
        decode_tensor(pb.Tensor(dtype="uint8", shape=[-1], data=b""))


def test_absurd_shape_is_rejected_before_allocation():
    side = int((MAX_TENSOR_BYTES // 4) ** 0.5) + 10
    with pytest.raises(CodecError, match="exceeds"):
        decode_tensor(pb.Tensor(dtype="float32", shape=[side, side], data=b""))


def test_observation_mapping_reports_offending_key():
    good = encode_observation({"observation.state": np.zeros((2, 2), dtype=np.float32)})
    good["observation.image"] = pb.Tensor(dtype="uint8", shape=[2, 2], data=b"\x00")
    with pytest.raises(CodecError, match=r"^observation\.image: "):
        decode_observation(good)
