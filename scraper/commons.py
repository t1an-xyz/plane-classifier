"""Wikimedia Commons image source.

Uses CirrusSearch's server-side ``deepcat:`` operator to expand each root
category's whole sub-tree in a single paginated query (thousands of hits in
well under a second), then resolves each File page to a scaled download URL
plus license/author metadata via the ``imageinfo`` API.

API docs: https://www.mediawiki.org/wiki/Help:CirrusSearch
          https://www.mediawiki.org/wiki/API:Imageinfo
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterator

from . import config
from .http import ThrottledSession
from .records import PhotoRecord

API = "https://commons.wikimedia.org/w/api.php"

# raster image mimes we accept (skip svg / video / pdf / tiff)
_ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    """Strip HTML tags/entities that Commons uses in Artist/Credit fields."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _excluded(title: str, exclude: tuple[str, ...]) -> bool:
    low = title.lower()
    return any(kw in low for kw in exclude)


class CommonsSource:
    def __init__(
        self,
        session: ThrottledSession,
        *,
        max_depth: int = config.DEFAULT_MAX_DEPTH,  # kept for CLI compat (unused)
        download_width: int = config.DEFAULT_DOWNLOAD_WIDTH,
        exclude: tuple[str, ...] = config.EXCLUDE_KEYWORDS,
    ) -> None:
        self.s = session
        self.download_width = download_width
        self.exclude = exclude

    # -- title discovery -----------------------------------------------------
    def _search(self, srsearch: str, limit: int) -> Iterator[str]:
        """Yield File titles matching a CirrusSearch query (paginated)."""
        cont: dict = {}
        seen = 0
        while seen < limit:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": srsearch,
                "srnamespace": "6",          # File:
                "srlimit": "500",
                "srprop": "",                # titles only -> lighter responses
                "maxlag": "5",
            }
            params.update(cont)
            data = self.s.get_json(API, params)
            hits = data.get("query", {}).get("search", [])
            if not hits:
                break
            for r in hits:
                yield r["title"]
                seen += 1
            if "continue" in data:
                cont = data["continue"]
            else:
                break

    def collect_titles(
        self,
        include: list[str],
        exclude: list[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Return unique, noise-filtered File titles for a class spec.

        ``include``/``exclude`` are Commons category names (without the
        "Category:" prefix); each include tree is expanded and each exclude
        tree subtracted server-side via deepcat.
        """
        cap = limit or 10_000
        # spread the budget across include roots so no single variant dominates
        per_root = max(cap // max(len(include), 1), 50)
        neg = "".join(
            f' -deepcat:"{e[len("Category:"):] if e.startswith("Category:") else e}"'
            for e in (exclude or [])
        )
        titles: list[str] = []
        seen: set[str] = set()
        for root in include:
            cat = root[len("Category:"):] if root.startswith("Category:") else root
            query = f'deepcat:"{cat}"{neg} filetype:bitmap'
            for title in self._search(query, per_root):
                if title in seen or _excluded(title, self.exclude):
                    continue
                seen.add(title)
                titles.append(title)
                if len(titles) >= cap:
                    return titles
        return titles

    # -- metadata resolution -------------------------------------------------
    def resolve(
        self, titles: list[str], min_width: int = config.DEFAULT_MIN_WIDTH
    ) -> Iterator[PhotoRecord]:
        """Resolve File titles (in batches of 50) to PhotoRecords."""
        for i in range(0, len(titles), 50):
            batch = titles[i : i + 50]
            params = {
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "titles": "|".join(batch),
                "iiprop": "url|size|mime|extmetadata|user",
                "iiurlwidth": str(self.download_width),
                "maxlag": "5",
            }
            data = self.s.get_json(API, params)
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                info = (page.get("imageinfo") or [None])[0]
                if not info:
                    continue
                if info.get("mime") not in _ALLOWED_MIME:
                    continue
                if info.get("width", 0) < min_width:
                    continue
                meta = info.get("extmetadata", {})

                def mv(key: str) -> str:
                    return _clean(meta.get(key, {}).get("value", ""))

                yield PhotoRecord(
                    source="commons",
                    ident=page.get("title", info.get("descriptionurl", "")),
                    # Prefer the ORIGINAL file: it is served straight from cache
                    # and does NOT hit Wikimedia's thumbnail-render rate limiter
                    # (which will 403/429-block your IP under load). The scaled
                    # thumbnail is kept only as a fallback for oversized files.
                    download_url=info["url"],
                    source_url=info.get("descriptionurl", ""),
                    author=mv("Artist") or info.get("user", ""),
                    license=mv("LicenseShortName"),
                    license_url=mv("LicenseUrl"),
                    width=info.get("width", 0),
                    height=info.get("height", 0),
                    extra={
                        "credit": mv("Credit"),
                        "usage_terms": mv("UsageTerms"),
                        "thumb_url": info.get("thumburl", ""),
                        "size": info.get("size", 0),
                    },
                )

    # -- convenience ---------------------------------------------------------
    def photos_for(
        self,
        include: list[str],
        *,
        exclude: list[str] | None = None,
        min_width: int = config.DEFAULT_MIN_WIDTH,
        title_limit: int | None = None,
    ) -> Iterator[PhotoRecord]:
        titles = self.collect_titles(include, exclude, limit=title_limit)
        yield from self.resolve(titles, min_width=min_width)
