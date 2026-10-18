"""Image encoding: shapes, dtypes, downscaling and size accounting."""

from __future__ import annotations

import base64

import pytest

from conftest import imaging_module

np = pytest.importorskip("numpy")


def test_torch_batch_tensor_is_encoded_as_jpeg(image_tensor):
    encoded = imaging_module.encode_image(image_tensor(64, 48))
    assert encoded.mime == "image/jpeg"
    assert (encoded.width, encoded.height) == (48, 64)
    raw = base64.b64decode(encoded.data_b64)
    assert raw[:3] == b"\xff\xd8\xff"  # JPEG SOI
    assert encoded.byte_count == len(raw)
    assert not encoded.summary()["downscaled"]


def test_png_encoding_has_png_signature(image_tensor):
    encoded = imaging_module.encode_image(image_tensor(32, 32), fmt="PNG")
    assert encoded.mime == "image/png"
    assert base64.b64decode(encoded.data_b64)[:8] == b"\x89PNG\r\n\x1a\n"


def test_long_side_is_capped_and_aspect_ratio_kept(image_tensor):
    encoded = imaging_module.encode_image(image_tensor(1024, 512), max_side=256)
    assert (encoded.width, encoded.height) == (128, 256)
    assert (encoded.source_width, encoded.source_height) == (512, 1024)
    assert encoded.summary()["downscaled"] is True


def test_zero_max_side_keeps_original_resolution(image_tensor):
    encoded = imaging_module.encode_image(image_tensor(300, 200), max_side=0)
    assert (encoded.width, encoded.height) == (200, 300)


def test_small_images_are_never_upscaled(image_tensor):
    encoded = imaging_module.encode_image(image_tensor(40, 20), max_side=1024)
    assert (encoded.width, encoded.height) == (20, 40)


def test_chw_and_uint8_inputs_are_accepted():
    chw = np.zeros((3, 8, 12), dtype=np.float32)
    encoded = imaging_module.encode_image(chw)
    assert (encoded.width, encoded.height) == (12, 8)

    uint8 = np.zeros((8, 12, 3), dtype=np.uint8)
    assert imaging_module.encode_image(uint8).mime == "image/jpeg"


def test_grayscale_is_promoted_to_rgb():
    gray = np.zeros((8, 8, 1), dtype=np.float32)
    assert (imaging_module.encode_image(gray).width, imaging_module.encode_image(gray).height) == (8, 8)


def test_data_uri_carries_the_media_type(image_tensor):
    encoded = imaging_module.encode_image(image_tensor(8, 8))
    assert encoded.data_uri().startswith("data:image/jpeg;base64,")


def test_zero_sized_image_is_rejected():
    with pytest.raises(ValueError):
        imaging_module.encode_image(np.zeros((0, 10, 3), dtype=np.uint8))


def test_token_estimate_is_positive_for_a_real_image(image_tensor):
    encoded = imaging_module.encode_image(image_tensor(1024, 1024), max_side=1024)
    assert imaging_module.estimate_tokens(encoded) > 0
    assert imaging_module.estimate_tokens(None) == 0
