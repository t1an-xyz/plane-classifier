"""Configuration for the aircraft image scraper.

Each target CLASS maps to one or more Wikimedia Commons *root* categories.
The Commons engine walks each root category and its sub-categories
(recursively, up to ``max_depth``) collecting file pages, then downloads a
scaled copy of every unique photo.

Class folder names are intentionally filesystem-friendly (no slashes) and are
meant to sit alongside the existing folders in
``Commercial aircraft classification/``.

The user's requested new classes were:
    - Embraer E-Jet family
    - Bombardier CRJ700 / 705 / 900 / 1000
    - Bombardier CRJ 100 / 200 / 440
    - Embraer ERJ family
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Target classes  ->  Wikimedia Commons categories.
# ---------------------------------------------------------------------------
# Each class maps to a spec:
#     {"include": [root categories...], "exclude": [categories to subtract]}
# The Commons engine expands every include category's whole sub-tree server
# side via  deepcat:"X"  and subtracts every exclude category via -deepcat:"X".
#
# Category names were verified against the live Commons API. Notes:
#   * E-Jets live under "Embraer 170/175/190/195" (+ "-E2").
#   * Regional jets live under short "CRJxxx" names (not "Bombardier CRJxxx").
#   * CRJ705 is filed under CRJ900, CRJ440 under CRJ200 -- rare, kept anyway.


def _spec(include: list[str], exclude: list[str] | None = None) -> dict:
    return {"include": include, "exclude": exclude or []}


# --- NEW classes requested by the user (regional jets) ---------------------
NEW_CLASSES: dict[str, dict] = {
    "Embraer E-Jet": _spec([
        "Embraer 170", "Embraer 175", "Embraer 190", "Embraer 195",
        "Embraer 175-E2", "Embraer 190-E2", "Embraer 195-E2",
    ]),
    "Bombardier CRJ 700-1000": _spec(["CRJ700", "CRJ705", "CRJ900", "CRJ1000"]),
    "Bombardier CRJ 100-200": _spec(["CRJ100", "CRJ200", "CRJ440"]),
    "Embraer ERJ": _spec(["Embraer ERJ 135", "Embraer ERJ 140", "Embraer ERJ 145"]),
}

# --- EXISTING dataset classes (folder names match the dataset EXACTLY, so
#     scraped output can be merged straight into
#     "Commercial aircraft classification/") -------------------------------
EXISTING_CLASSES: dict[str, dict] = {
    "Airbus A220": _spec(["Airbus A220", "Bombardier CSeries"]),
    "Airbus A318": _spec(["Airbus A318"]),
    "Airbus A319": _spec(["Airbus A319"]),
    "Airbus A320": _spec(["Airbus A320", "Airbus A320neo"],
                         exclude=["Airbus A318", "Airbus A319", "Airbus A321"]),
    "Airbus a321": _spec(["Airbus A321", "Airbus A321neo"]),
    "Airbus A330": _spec(["Airbus A330"]),
    "Airbus A350": _spec(["Airbus A350"]),
    "Airbus A380": _spec(["Airbus A380"]),
    "ATR 42": _spec(["ATR 42"]),
    "ATR 72": _spec(["ATR 72"]),
    "Boeing 737 max": _spec(["Boeing 737 MAX"]),
    "Boeing 737 NG": _spec(["Boeing 737 Next Generation"]),
    "Boeing 747": _spec(["Boeing 747"]),
    "Boeing 767": _spec(["Boeing 767"]),
    "Boeing 777": _spec(["Boeing 777"], exclude=["Boeing 777X"]),
    "Boeing 777X": _spec(["Boeing 777X"]),
    "Boeing 787": _spec(["Boeing 787"]),
}

# Everything the scraper knows about. ``--group`` on the CLI selects a subset.
CLASSES: dict[str, dict] = {**NEW_CLASSES, **EXISTING_CLASSES}

# ---------------------------------------------------------------------------
# Sub-categories / files we never want (wreckage, cockpits, cabins, drawings...)
# Matched as case-insensitive substrings against category and file titles.
# ---------------------------------------------------------------------------
EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "accident",
    "incident",
    "crash",
    "wreck",
    "cockpit",
    "flight deck",
    "interior",
    "cabin",
    "seat",           # seats / seating / seat map
    "galley",
    "lavatory",
    "diagram",
    "drawing",
    "blueprint",
    "schematic",
    "planform",
    "silhouette",
    "livery detail",
    "safety card",
    "boarding pass",
    "registration document",
    "scale model",
    "model aircraft",
    "airfix",
    "revell",
)

# ---------------------------------------------------------------------------
# Defaults (overridable on the CLI)
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR = "scraped_data"
DEFAULT_PER_CLASS = 300      # max images to keep per class
DEFAULT_MAX_DEPTH = 3        # sub-category recursion depth
DEFAULT_MIN_WIDTH = 800      # skip source images narrower than this (px)
DEFAULT_DOWNLOAD_WIDTH = 1024  # width for the thumbnail fallback (oversized files)
DEFAULT_MAX_SIZE = 1280      # downscale saved images so the longest side <= this
DEFAULT_JPEG_QUALITY = 90

# Polite request pacing (seconds between requests to the same host).
API_THROTTLE = 1.0           # api.php  -- Wikimedia asks callers to go easy
DOWNLOAD_THROTTLE = 0.5      # upload.wikimedia.org thumbnails

# Sent on every request. Wikimedia REQUIRES a descriptive User-Agent with a
# contact.  Please edit CONTACT before running large jobs.
CONTACT = "plane-classifier dataset builder (set your contact in scraper/config.py)"
USER_AGENT = f"plane-classifier-scraper/{__import__('scraper').__version__} ({CONTACT})"

# upload.wikimedia.org (static media) now returns 403 to obvious bot/script
# User-Agents. It serves browser User-Agents normally, so downloads use this.
# The descriptive USER_AGENT above is still used for the api.php requests.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
