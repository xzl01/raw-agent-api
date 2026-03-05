"""RAW engine: decode .ARW files with rawpy, extract metadata, render to Pillow Image."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import rawpy
from PIL import Image


@dataclass
class RawMetadata:
    """Basic EXIF-style metadata extracted from a RAW file."""

    iso: int = 0
    shutter_speed: float = 0.0
    aperture: float = 0.0
    focal_length: float = 0.0
    camera_make: str = ""
    camera_model: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def decode_raw_bytes(raw_bytes: bytes) -> tuple[Image.Image, RawMetadata]:
    """Decode RAW bytes into a Pillow Image and extract metadata.

    Args:
        raw_bytes: Raw binary content of an .ARW (or compatible) file.

    Returns:
        A tuple of (PIL Image in RGB mode, RawMetadata).

    Raises:
        rawpy.LibRawError: If the bytes cannot be decoded as a RAW image.
    """
    with rawpy.imread(io.BytesIO(raw_bytes)) as raw:
        metadata = _extract_metadata(raw)
        rgb_array: np.ndarray = raw.postprocess(
            use_camera_wb=True,
            output_bps=8,
        )

    image = Image.fromarray(rgb_array, mode="RGB")
    return image, metadata


def _extract_metadata(raw: rawpy.RawPy) -> RawMetadata:
    """Pull basic metadata fields from a rawpy.RawPy object."""
    other_data = raw.other_params if hasattr(raw, "other_params") else {}

    iso = int(other_data.get("iso_speed", 0)) if isinstance(other_data, dict) else 0
    shutter = (
        float(other_data.get("shutter", 0.0)) if isinstance(other_data, dict) else 0.0
    )
    aperture = (
        float(other_data.get("aperture", 0.0)) if isinstance(other_data, dict) else 0.0
    )
    focal_length = (
        float(other_data.get("focal_len", 0.0))
        if isinstance(other_data, dict)
        else 0.0
    )

    return RawMetadata(
        iso=iso,
        shutter_speed=shutter,
        aperture=aperture,
        focal_length=focal_length,
    )
