"""Image adjustment functions using Pillow ImageEnhance."""

from __future__ import annotations

import io

from PIL import Image, ImageEnhance


def adjust_exposure(image: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness/exposure of an image.

    Args:
        image: Source Pillow Image.
        factor: Multiplier – 1.0 is original, >1 brighter, <1 darker.

    Returns:
        Adjusted Pillow Image.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast of an image.

    Args:
        image: Source Pillow Image.
        factor: Multiplier – 1.0 is original.

    Returns:
        Adjusted Pillow Image.
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def adjust_saturation(image: Image.Image, factor: float) -> Image.Image:
    """Adjust color saturation of an image.

    Args:
        image: Source Pillow Image.
        factor: Multiplier – 1.0 is original, 0.0 is grayscale.

    Returns:
        Adjusted Pillow Image.
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def adjust_white_balance(
    image: Image.Image,
    red_scale: float = 1.0,
    blue_scale: float = 1.0,
) -> Image.Image:
    """Adjust white balance by scaling the red and blue channels.

    Args:
        image: Source Pillow Image (RGB).
        red_scale: Multiplier for the red channel.
        blue_scale: Multiplier for the blue channel.

    Returns:
        Adjusted Pillow Image.
    """
    import numpy as np

    arr = np.array(image, dtype=np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * red_scale, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * blue_scale, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def apply_adjustments(
    image: Image.Image,
    exposure: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    white_balance_red: float = 1.0,
    white_balance_blue: float = 1.0,
) -> Image.Image:
    """Apply all adjustments in sequence.

    Args:
        image: Source Pillow Image.
        exposure: Exposure multiplier.
        contrast: Contrast multiplier.
        saturation: Saturation multiplier.
        white_balance_red: Red channel scale for white balance.
        white_balance_blue: Blue channel scale for white balance.

    Returns:
        Fully adjusted Pillow Image.
    """
    image = adjust_exposure(image, exposure)
    image = adjust_contrast(image, contrast)
    image = adjust_saturation(image, saturation)
    image = adjust_white_balance(image, white_balance_red, white_balance_blue)
    return image


def image_to_jpeg_bytes(image: Image.Image, quality: int = 90) -> bytes:
    """Encode a Pillow Image to JPEG bytes.

    Args:
        image: Source Pillow Image.
        quality: JPEG quality (1-95).

    Returns:
        JPEG-encoded bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
