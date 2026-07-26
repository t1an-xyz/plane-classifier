"""Optional Planespotters supplement (per-registration photo API).

IMPORTANT LIMITATIONS -- read before using:
  * The public API returns at most ONE photo per aircraft registration, and
    only a ~280px thumbnail. It has no "search by type" endpoint.
  * Photos remain copyrighted by the photographer. The API terms allow display
    with attribution + a link back and are non-commercial. They are fine for
    private model training but MUST NOT be redistributed.
  * Because of the above, Wikimedia Commons is the recommended primary source;
    this module only exists to top up rare variants.

To use it you supply a mapping of {class_name: [registrations...]} as JSON, e.g.
    {"Bombardier CRJ 100-200": ["N8928A", "N875AS", ...]}
API docs: https://www.planespotters.net/photo/api
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from . import config
from .http import ThrottledSession
from .records import PhotoRecord

API_REG = "https://api.planespotters.net/pub/photos/reg/{reg}"
_HOST = "api.planespotters.net"


class PlanespottersSource:
    def __init__(self, session: ThrottledSession) -> None:
        self.s = session
        # be gentle with the free API
        self.s.set_gap(_HOST, 1.0)

    def photos_for_registrations(self, regs: list[str]) -> Iterator[PhotoRecord]:
        for reg in regs:
            reg = reg.strip().upper()
            if not reg:
                continue
            data = self.s.get_json(API_REG.format(reg=reg), params={})
            for p in data.get("photos", []):
                thumb = p.get("thumbnail_large") or p.get("thumbnail") or {}
                src = thumb.get("src")
                if not src:
                    continue
                size = thumb.get("size", {}) or {}
                yield PhotoRecord(
                    source="planespotters",
                    ident=str(p.get("id", reg)),
                    download_url=src,
                    source_url=p.get("link", ""),
                    author=p.get("photographer", ""),
                    license="Planespotters (all rights reserved; attribution + link required)",
                    license_url=p.get("link", ""),
                    width=int(size.get("width", 0) or 0),
                    height=int(size.get("height", 0) or 0),
                    extra={"registration": reg},
                )

    @staticmethod
    def load_registration_map(path: str | Path) -> dict[str, list[str]]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
