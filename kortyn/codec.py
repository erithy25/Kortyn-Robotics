"""Conversion between numpy arrays and the protobuf ``Tensor`` message.

Arrays travel as raw little-endian bytes plus dtype and shape. Only a small set of dtypes is accepted so
that a malformed or malicious request can never make the receiver allocate arbitrary objects (no pickle).

Camera images (uint8 ``(T, H, W, C)``) can optionally travel scaled down and/or compressed (phase 3a,
:func:`encode_images`): the frames are stacked vertically into one image, encoded as JPEG, PNG or raw bytes,
and the receiver decodes them and scales them back to the original resolution, so the policy always gets
the input size it was trained on.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import cv2
import numpy as np

from kortyn.config import ImageTransportConfig
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
    """Decode a ``Tensor`` message into a numpy array, validating dtype, shape and size.

    Raw tensors become read-only views on the message bytes; transported images become new arrays.
    """
    if tensor.encoding:
        return _decode_images(tensor)
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


def encode_observation(
    observation: Mapping[str, Array],
    image_transport: ImageTransportConfig | None = None,
    image_keys: Collection[str] = (),
) -> dict[str, pb.Tensor]:
    """Encode a mapping of named arrays (e.g. an observation history); ``image_keys`` use the transport."""
    transport = image_transport or ImageTransportConfig()
    return {
        key: encode_images(value, transport) if key in image_keys else encode_array(value)
        for key, value in observation.items()
    }


# --------------------------------------------------------------------------------------------- images
_IMAGE_EXTENSIONS = {"jpeg": ".jpg", "png": ".png"}


def _check_frames(frames: Array) -> tuple[int, int, int, int]:
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[3] not in (1, 3):
        raise CodecError(
            f"images must be uint8 (T, H, W, C) with C in (1, 3), got {frames.dtype} {frames.shape}"
        )
    t, h, w, c = (int(d) for d in frames.shape)
    if min(t, h, w) < 1:
        raise CodecError(f"empty image tensor {frames.shape}")
    return t, h, w, c


def _resize(frames: Array, height: int, width: int, interpolation: int) -> Array:
    if frames.shape[1:3] == (height, width):
        return frames
    resized = [cv2.resize(frame, (width, height), interpolation=interpolation) for frame in frames]
    return np.stack([r.reshape(height, width, -1) for r in resized]).astype(np.uint8, copy=False)


def encode_images(frames: Array, transport: ImageTransportConfig) -> pb.Tensor:
    """Encode uint8 frames ``(T, H, W, C)`` for the wire: scaled to the transport size, then compressed."""
    if transport.is_identity:
        return encode_array(frames)
    t, h, w, c = _check_frames(frames)
    height, width = transport.height or h, transport.width or w
    scaled = _resize(np.ascontiguousarray(frames), height, width, cv2.INTER_AREA)
    stacked = scaled.reshape(t * height, width, c)
    if transport.codec == "raw":
        data = stacked.tobytes(order="C")
    else:
        image = stacked[..., ::-1] if c == 3 else stacked[..., 0]  # OpenCV expects BGR
        params = [cv2.IMWRITE_JPEG_QUALITY, transport.quality] if transport.codec == "jpeg" else []
        ok, buffer = cv2.imencode(_IMAGE_EXTENSIONS[transport.codec], np.ascontiguousarray(image), params)
        if not ok:
            raise CodecError(f"{transport.codec} encoding of {frames.shape} failed")
        data = buffer.tobytes()
    return pb.Tensor(
        dtype="uint8",
        shape=[t, h, w, c],
        data=data,
        encoding=transport.codec,
        encoded_shape=[t, height, width, c],
    )


def _decode_images(tensor: pb.Tensor) -> Array:
    if tensor.dtype != "uint8":
        raise CodecError(f"image encoding {tensor.encoding!r} needs dtype uint8, got {tensor.dtype!r}")
    shape = tuple(int(d) for d in tensor.shape)
    encoded = tuple(int(d) for d in tensor.encoded_shape)
    if len(shape) != 4 or len(encoded) != 4 or shape[0] != encoded[0] or shape[3] != encoded[3]:
        raise CodecError(f"image shapes do not match: shape {shape}, encoded_shape {encoded}")
    if min(*shape, *encoded) < 1 or shape[3] not in (1, 3):
        raise CodecError(f"invalid image shapes {shape} / {encoded}")
    for name, dims in (("shape", shape), ("encoded_shape", encoded)):
        if int(np.prod(dims, dtype=np.int64)) > MAX_TENSOR_BYTES:
            raise CodecError(f"image {name} {dims} exceeds {MAX_TENSOR_BYTES} bytes")
    t, height, width, c = encoded
    frames: Array
    if tensor.encoding == "raw":
        if len(tensor.data) != t * height * width * c:
            raise CodecError(
                f"raw image payload has {len(tensor.data)} bytes, encoded_shape {encoded} needs more"
            )
        frames = np.frombuffer(tensor.data, dtype=np.uint8).reshape(encoded)
    elif tensor.encoding in _IMAGE_EXTENSIONS:
        image = cv2.imdecode(np.frombuffer(tensor.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise CodecError(f"{tensor.encoding} payload could not be decoded")
        expected = (t * height, width, c) if c == 3 else (t * height, width)
        if image.shape != expected:
            raise CodecError(f"decoded {tensor.encoding} image has shape {image.shape}, expected {expected}")
        if c == 3:
            image = image[..., ::-1]  # BGR -> RGB
        frames = np.ascontiguousarray(image, dtype=np.uint8).reshape(encoded)
    else:
        raise CodecError(f"unknown image encoding {tensor.encoding!r}; known: raw, jpeg, png")
    return _resize(frames, shape[1], shape[2], cv2.INTER_LINEAR)


def decode_observation(tensors: Mapping[str, pb.Tensor]) -> dict[str, Array]:
    """Decode a mapping of named ``Tensor`` messages, reporting the offending key on failure."""
    decoded: dict[str, Array] = {}
    for key, tensor in tensors.items():
        try:
            decoded[key] = decode_tensor(tensor)
        except CodecError as exc:
            raise CodecError(f"{key}: {exc}") from exc
    return decoded
