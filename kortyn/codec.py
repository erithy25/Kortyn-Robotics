"""Conversion between numpy arrays and the protobuf ``Tensor`` message.

Arrays travel as raw little-endian bytes plus dtype and shape. Only a small set of dtypes is accepted so
that a malformed or malicious request can never make the receiver allocate arbitrary objects (no pickle).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from kortyn.proto import kortyn_brain_pb2 as pb
from kortyn.typing_utils import Array

# dtype name -> little-endian numpy dtype
_ALLOWED_DTYPES: dict[str, np.dtype[np.generic]] = {
    "uint8": np.dtype("<u1"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "int64": np.dtype("<i8"),
}

# Hard upper bound for a single decoded tensor (protects against absurd shapes in requests).
MAX_TENSOR_BYTES = 64 * 1024 * 1024


class CodecError(ValueError):
    """Raised when a tensor message cannot be decoded into a valid array."""


def encode_array(array: Array) -> pb.Tensor:
    """Encode a numpy array as a ``Tensor`` message (C-contiguous, little-endian)."""
    name = array.dtype.name
    if name not in _ALLOWED_DTYPES:
        raise CodecError(f"dtype {name!r} is not supported; allowed: {sorted(_ALLOWED_DTYPES)}")
    # np.asarray keeps 0-d arrays 0-d (np.ascontiguousarray would promote them to 1-d).
    contiguous = np.asarray(array, dtype=_ALLOWED_DTYPES[name], order="C")
    return pb.Tensor(dtype=name, shape=list(contiguous.shape), data=contiguous.tobytes(order="C"))


def decode_tensor(tensor: pb.Tensor) -> Array:
    """Decode a ``Tensor`` message into a read-only numpy array, validating dtype, shape and size."""
    dtype = _ALLOWED_DTYPES.get(tensor.dtype)
    if dtype is None:
        raise CodecError(f"dtype {tensor.dtype!r} is not supported; allowed: {sorted(_ALLOWED_DTYPES)}")
    shape = tuple(int(d) for d in tensor.shape)
    if any(d < 0 for d in shape):
        raise CodecError(f"negative dimension in shape {shape}")
    n_elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
    expected_bytes = n_elements * dtype.itemsize
    if expected_bytes > MAX_TENSOR_BYTES:
        raise CodecError(f"tensor of shape {shape} ({expected_bytes} bytes) exceeds {MAX_TENSOR_BYTES} bytes")
    if len(tensor.data) != expected_bytes:
        raise CodecError(
            f"payload has {len(tensor.data)} bytes but dtype {tensor.dtype} and shape {shape} "
            f"require {expected_bytes} bytes"
        )
    return np.frombuffer(tensor.data, dtype=dtype).reshape(shape)


def encode_observation(observation: Mapping[str, Array]) -> dict[str, pb.Tensor]:
    """Encode a mapping of named arrays (e.g. an observation history)."""
    return {key: encode_array(value) for key, value in observation.items()}


def decode_observation(tensors: Mapping[str, pb.Tensor]) -> dict[str, Array]:
    """Decode a mapping of named ``Tensor`` messages, reporting the offending key on failure."""
    decoded: dict[str, Array] = {}
    for key, tensor in tensors.items():
        try:
            decoded[key] = decode_tensor(tensor)
        except CodecError as exc:
            raise CodecError(f"{key}: {exc}") from exc
    return decoded
