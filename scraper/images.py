"""Image helpers: content hashing, validation, normalisation to RGB JPEG."""

from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageFile

# Some Commons JPEGs are truncated a few bytes early; allow loading them.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_jpeg(
    data: bytes,
    *,
    max_size: int | None = None,
    quality: int = 90,
    min_width: int = 0,
) -> bytes | None:
    """Validate raw bytes as an image and return normalised JPEG bytes.

    Returns ``None`` if the data is not a usable photo (corrupt, too small,
    or not a raster image).
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return None

    if img.width < min_width:
        return None

    # Flatten transparency / palettes / CMYK onto white, force RGB.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if max_size and max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()
