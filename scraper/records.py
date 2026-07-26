"""Shared record type produced by every image source."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhotoRecord:
    """A single downloadable photo plus its attribution metadata."""

    source: str            # "commons" | "planespotters"
    ident: str             # stable id within the source (title / photo id)
    download_url: str      # URL to fetch the pixels from
    source_url: str = ""   # human page for attribution / license verification
    author: str = ""       # photographer / uploader
    license: str = ""      # short license name, e.g. "CC BY-SA 4.0"
    license_url: str = ""
    width: int = 0
    height: int = 0
    extra: dict = field(default_factory=dict)
