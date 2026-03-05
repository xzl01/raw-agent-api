"""Unit tests for the RAW engine and image tools."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from core.image_tools import (
    adjust_contrast,
    adjust_exposure,
    adjust_saturation,
    adjust_white_balance,
    apply_adjustments,
    image_to_jpeg_bytes,
)
from core.raw_engine import RawMetadata, _extract_metadata, decode_raw_bytes


# ---------------------------------------------------------------------------
# image_tools tests
# ---------------------------------------------------------------------------

class TestAdjustExposure:
    def test_brighter(self, sample_image: Image.Image) -> None:
        result = adjust_exposure(sample_image, 2.0)
        assert result.size == sample_image.size
        arr = np.array(result)
        # Pixels should be brighter
        assert arr.mean() > np.array(sample_image).mean()

    def test_darker(self, sample_image: Image.Image) -> None:
        result = adjust_exposure(sample_image, 0.5)
        arr = np.array(result)
        assert arr.mean() < np.array(sample_image).mean()

    def test_neutral(self, sample_image: Image.Image) -> None:
        result = adjust_exposure(sample_image, 1.0)
        np.testing.assert_array_equal(np.array(result), np.array(sample_image))


class TestAdjustContrast:
    def test_increased_contrast(self, sample_image: Image.Image) -> None:
        result = adjust_contrast(sample_image, 2.0)
        assert result.size == sample_image.size

    def test_neutral_contrast(self, sample_image: Image.Image) -> None:
        result = adjust_contrast(sample_image, 1.0)
        np.testing.assert_array_equal(np.array(result), np.array(sample_image))


class TestAdjustSaturation:
    def test_desaturate(self, sample_image: Image.Image) -> None:
        result = adjust_saturation(sample_image, 0.0)
        arr = np.array(result)
        # R, G, B channels should be equal (grayscale)
        assert arr[..., 0].mean() == pytest.approx(arr[..., 1].mean(), abs=2)
        assert arr[..., 1].mean() == pytest.approx(arr[..., 2].mean(), abs=2)

    def test_neutral_saturation(self, sample_image: Image.Image) -> None:
        result = adjust_saturation(sample_image, 1.0)
        np.testing.assert_array_equal(np.array(result), np.array(sample_image))


class TestAdjustWhiteBalance:
    def test_red_scale(self) -> None:
        arr = np.full((4, 4, 3), 100, dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        result = adjust_white_balance(image, red_scale=1.5, blue_scale=1.0)
        out = np.array(result)
        assert out[0, 0, 0] == 150  # red channel scaled
        assert out[0, 0, 1] == 100  # green unchanged
        assert out[0, 0, 2] == 100  # blue unchanged

    def test_blue_scale(self) -> None:
        arr = np.full((4, 4, 3), 100, dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        result = adjust_white_balance(image, red_scale=1.0, blue_scale=0.5)
        out = np.array(result)
        assert out[0, 0, 2] == 50  # blue channel scaled

    def test_clip_does_not_overflow(self) -> None:
        arr = np.full((4, 4, 3), 200, dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        result = adjust_white_balance(image, red_scale=2.0, blue_scale=2.0)
        out = np.array(result)
        assert out.max() == 255


class TestApplyAdjustments:
    def test_all_neutral(self, sample_image: Image.Image) -> None:
        result = apply_adjustments(sample_image)
        np.testing.assert_array_equal(np.array(result), np.array(sample_image))

    def test_combined(self, sample_image: Image.Image) -> None:
        result = apply_adjustments(
            sample_image,
            exposure=1.2,
            contrast=1.1,
            saturation=1.3,
            white_balance_red=1.1,
            white_balance_blue=0.9,
        )
        assert result.size == sample_image.size


class TestImageToJpegBytes:
    def test_returns_bytes(self, sample_image: Image.Image) -> None:
        data = image_to_jpeg_bytes(sample_image)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_valid_jpeg_header(self, sample_image: Image.Image) -> None:
        data = image_to_jpeg_bytes(sample_image)
        assert data[:2] == b"\xff\xd8"  # JPEG magic bytes


# ---------------------------------------------------------------------------
# raw_engine tests
# ---------------------------------------------------------------------------

class TestDecodeRawBytes:
    def test_decode_returns_image_and_metadata(self, mock_rawpy) -> None:
        _, mock_raw = mock_rawpy
        dummy_bytes = b"FAKE_ARW_DATA"
        image, metadata = decode_raw_bytes(dummy_bytes)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert isinstance(metadata, RawMetadata)

    def test_postprocess_called(self, mock_rawpy) -> None:
        _, mock_raw = mock_rawpy
        decode_raw_bytes(b"FAKE")
        mock_raw.postprocess.assert_called_once()

    def test_metadata_fields(self, mock_rawpy) -> None:
        _, mock_raw = mock_rawpy
        _, metadata = decode_raw_bytes(b"FAKE")
        assert metadata.iso == 400
        assert metadata.shutter_speed == pytest.approx(0.01)
        assert metadata.aperture == pytest.approx(5.6)
        assert metadata.focal_length == pytest.approx(50.0)


class TestRawMetadata:
    def test_defaults(self) -> None:
        m = RawMetadata()
        assert m.iso == 0
        assert m.shutter_speed == 0.0
        assert m.aperture == 0.0
        assert m.focal_length == 0.0
        assert m.camera_make == ""
        assert m.camera_model == ""

    def test_custom_values(self) -> None:
        m = RawMetadata(iso=800, shutter_speed=0.002, aperture=2.8, focal_length=85.0)
        assert m.iso == 800
        assert m.focal_length == pytest.approx(85.0)
