import numpy as np
import pytest

from kortyn.codec import (
    MAX_TENSOR_BYTES,
    CodecError,
    decode_observation,
    decode_tensor,
    encode_array,
    encode_images,
    encode_observation,
)
from kortyn.config import ImageTransportConfig
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


# --------------------------------------------------------------------------------------------- images
def smooth_frames(t: int = 2, h: int = 48, w: int = 64) -> np.ndarray:
    """Camera-like frames: smooth gradients and waves (JPEG-friendly, but not constant)."""
    y, x = np.mgrid[0:h, 0:w]
    frames = []
    for i in range(t):
        r = (x * 255 / w + 20 * i) % 256
        g = y * 255 / h
        b = 128 + 100 * np.sin(x / 7.0) * np.cos(y / 9.0)
        frames.append(np.stack([r, g, b], axis=-1))
    return np.asarray(frames, dtype=np.uint8)


def test_default_image_transport_is_the_phase1_wire_format():
    frames = smooth_frames()
    identity = encode_images(frames, ImageTransportConfig())
    assert identity == encode_array(frames)
    assert identity.encoding == ""
    assert encode_observation({"img": frames}) == encode_observation(
        {"img": frames}, ImageTransportConfig(), image_keys={"img"}
    )


def test_png_is_lossless_and_jpeg_is_close_and_smaller():
    frames = smooth_frames()
    png = encode_images(frames, ImageTransportConfig(codec="png"))
    np.testing.assert_array_equal(decode_tensor(png), frames)
    jpeg = encode_images(frames, ImageTransportConfig(codec="jpeg", quality=90))
    decoded = decode_tensor(jpeg)
    assert decoded.shape == frames.shape
    assert decoded.dtype == np.uint8
    mse = np.mean((decoded.astype(np.float64) - frames) ** 2)
    assert 10 * np.log10(255.0**2 / mse) > 30.0  # PSNR
    assert jpeg.ByteSize() < frames.nbytes / 3
    assert list(jpeg.encoded_shape) == list(frames.shape)


def test_downscaled_images_are_restored_to_the_camera_resolution():
    frames = smooth_frames(t=1, h=96, w=96)
    for codec in ("raw", "png", "jpeg"):
        tensor = encode_images(frames, ImageTransportConfig(codec=codec, height=48, width=48))
        assert list(tensor.encoded_shape) == [1, 48, 48, 3]
        decoded = decode_tensor(tensor)
        assert decoded.shape == frames.shape
        assert np.abs(decoded.astype(np.int16) - frames).mean() < 12.0, codec
    raw_small = encode_images(frames, ImageTransportConfig(codec="raw", height=48, width=48))
    assert len(raw_small.data) == frames.nbytes // 4
    gray = np.ascontiguousarray(frames[..., :1])
    np.testing.assert_array_equal(decode_tensor(encode_images(gray, ImageTransportConfig(codec="png"))), gray)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"encoding": "webp"}, "unknown image encoding"),
        ({"dtype": "float32"}, "needs dtype uint8"),
        ({"encoded_shape": [2, 48, 64, 3]}, "do not match"),
        ({"data": b"not an image"}, "could not be decoded"),
        ({"encoded_shape": [1, 24, 64, 3]}, "expected"),
    ],
)
def test_malformed_image_tensors_are_rejected(change, message):
    tensor = encode_images(smooth_frames(t=1), ImageTransportConfig(codec="png"))
    if "encoded_shape" in change:
        del tensor.encoded_shape[:]
        tensor.encoded_shape.extend(change["encoded_shape"])
    else:
        for field, value in change.items():
            setattr(tensor, field, value)
    with pytest.raises(CodecError, match=message):
        decode_tensor(tensor)
    with pytest.raises(CodecError, match="uint8"):
        encode_images(np.zeros((1, 8, 8, 3), dtype=np.float32), ImageTransportConfig(codec="jpeg"))
