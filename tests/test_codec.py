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
        ({"encoding": "bmp"}, "unknown image encoding"),
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


# --------------------------------------------------------------------------------------------- phase 4
LOSSLESS = [
    ImageTransportConfig(codec="raw"),
    ImageTransportConfig(codec="png"),
    ImageTransportConfig(codec="png", level=6),
    ImageTransportConfig(codec="webp", level=0),
    ImageTransportConfig(codec="webp", level=4),
    ImageTransportConfig(codec="zstd"),
    ImageTransportConfig(codec="zstd", level=9),
]


def phase4_images() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(4)
    noise = rng.integers(0, 256, size=(1, 64, 48, 3), dtype=np.uint8)
    return {
        "noise": noise,
        "smooth_two_frames": smooth_frames(t=2, h=40, w=56),
        "black": np.zeros((1, 32, 32, 3), dtype=np.uint8),
        "white": np.full((1, 32, 32, 3), 255, dtype=np.uint8),
        "one_channel": np.ascontiguousarray(noise[..., :1]),
    }


@pytest.mark.parametrize("transport", LOSSLESS, ids=lambda t: f"{t.codec}-{t.level}")
def test_lossless_codecs_return_the_frames_bit_for_bit(transport):
    assert transport.is_lossless
    for name, frames in phase4_images().items():
        message = pb.Tensor.FromString(encode_images(frames, transport).SerializeToString())
        decoded = decode_tensor(message)
        assert decoded.dtype == np.uint8, name
        assert decoded.shape == frames.shape, name
        assert np.array_equal(decoded, frames), name


def test_lossless_codecs_compress():
    # PNG and WebP predict pixels from their neighbours: smooth frames shrink a lot. zstd sees only bytes:
    # it compresses repetition (a flat frame), not gradients.
    frames = smooth_frames(t=1, h=96, w=96)
    flat = np.full((1, 96, 96, 3), 90, dtype=np.uint8)
    for transport in LOSSLESS[1:]:
        if transport.codec == "zstd":
            assert encode_images(flat, transport).ByteSize() < flat.nbytes / 20, transport
        else:
            assert encode_images(frames, transport).ByteSize() < frames.nbytes / 2, transport


def test_codec_levels_are_validated_and_lossless_flag_is_strict():
    from kortyn.config import ConfigError

    for codec, level in (("png", 10), ("webp", 7), ("zstd", 0), ("zstd", 23), ("jpeg", 3), ("raw", 1)):
        with pytest.raises(ConfigError, match="level"):
            ImageTransportConfig(codec=codec, level=level)
    assert not ImageTransportConfig(codec="jpeg").is_lossless
    assert not ImageTransportConfig(codec="png", height=48, width=48).is_lossless
    assert not ImageTransportConfig(codec="raw", height=48).is_lossless


def test_malformed_zstd_and_webp_payloads_are_rejected():
    import zstandard

    frames = smooth_frames(t=1)
    zstd = encode_images(frames, ImageTransportConfig(codec="zstd"))
    truncated = pb.Tensor()
    truncated.CopyFrom(zstd)
    truncated.data = zstd.data[: len(zstd.data) // 2]
    with pytest.raises(CodecError, match="zstd"):
        decode_tensor(truncated)
    # a payload that inflates beyond the declared shape is refused (no decompression bomb)
    bomb = pb.Tensor()
    bomb.CopyFrom(zstd)
    bomb.data = zstandard.ZstdCompressor().compress(bytes(frames.nbytes * 4))
    with pytest.raises(CodecError, match="zstd"):
        decode_tensor(bomb)
    webp = encode_images(frames, ImageTransportConfig(codec="webp"))
    wrong = pb.Tensor()
    wrong.CopyFrom(webp)
    del wrong.encoded_shape[:]
    wrong.encoded_shape.extend([1, frames.shape[1] // 2, frames.shape[2], 3])
    del wrong.shape[:]
    wrong.shape.extend([1, frames.shape[1] // 2, frames.shape[2], 3])
    with pytest.raises(CodecError, match="webp"):
        decode_tensor(wrong)
    garbage = pb.Tensor()
    garbage.CopyFrom(webp)
    garbage.data = b"RIFF0000WEBPnot really"
    with pytest.raises(CodecError, match="webp"):
        decode_tensor(garbage)
