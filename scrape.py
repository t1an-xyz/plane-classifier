#!/usr/bin/env python3
"""Collect license-clean aircraft photos for the plane-classifier project.

Primary source is Wikimedia Commons; Planespotters is an optional supplement.

Examples
--------
    # everything, defaults (Commons, 300 imgs/class -> ./scraped_data)
    python scrape.py

    # just two classes, 150 each, downscaled to 800px, into a custom dir
    python scrape.py --classes "Embraer ERJ" "Bombardier CRJ 100-200" \
        --per-class 150 --max-size 800 --out scraped_data

    # also emit an 80/20 train/test split ready to merge into the dataset
    python scrape.py --split 0.2

    # top up from Planespotters using a registrations file
    python scrape.py --source planespotters --registrations regs.json
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm

from scraper import config
from scraper.commons import CommonsSource
from scraper.http import ThrottledSession
from scraper.images import content_hash, to_jpeg
from scraper.planespotters import PlanespottersSource
from scraper.records import PhotoRecord

ATTR_COLUMNS = [
    "filename", "source", "ident", "author", "license",
    "license_url", "source_url", "download_url", "width", "height",
]

MAX_DOWNLOAD_BYTES = 40 * 1024 * 1024   # hard cap on any single download (40 MB)
PREFER_THUMB_ABOVE = 15 * 1024 * 1024   # for originals bigger than this, use the thumb


def fetch_bytes(session: ThrottledSession, url: str, tries: int = 2) -> bytes | None:
    """Download a URL body with a hard read timeout and size cap.

    Returns None on timeout / network error / oversized payload so the caller
    can simply skip and move on instead of stalling the whole run.
    """
    try:
        resp = session.get(url, stream=True, tries=tries, timeout=(10, 30))
        try:
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    return None
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            resp.close()
    except Exception:
        return None


def fetch_photo(session: ThrottledSession, rec: PhotoRecord) -> bytes | None:
    """Fetch a photo.

    Downloads the ORIGINAL file by default (bypasses Wikimedia's thumbnail
    render limiter). Only for rare oversized originals do we request the
    pre-scaled thumbnail, and we fall back to it if the original download fails.
    """
    thumb = rec.extra.get("thumb_url")
    size = rec.extra.get("size") or 0

    if thumb and size and size > PREFER_THUMB_ABOVE:
        data = fetch_bytes(session, thumb, tries=2)
        if data is not None:
            return data

    data = fetch_bytes(session, rec.download_url, tries=2)
    if data is not None:
        return data
    if thumb and thumb != rec.download_url:
        return fetch_bytes(session, thumb, tries=1)
    return None


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "class"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape license-clean aircraft images (Wikimedia Commons + Planespotters).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", choices=["commons", "planespotters", "all"],
                   default="commons", help="which image source(s) to use")
    p.add_argument("--group", choices=["new", "existing", "all"], default="new",
                   help="which set of classes to fetch when --classes is omitted "
                        "(new = the 4 regional jets; existing = the 17 dataset "
                        "classes; all = both)")
    p.add_argument("--classes", nargs="+", default=None,
                   help="explicit subset of class names (overrides --group)")
    p.add_argument("--out", default=config.DEFAULT_OUT_DIR, help="output directory")
    p.add_argument("--per-class", type=int, default=config.DEFAULT_PER_CLASS,
                   help="max images to keep per class")
    p.add_argument("--max-depth", type=int, default=config.DEFAULT_MAX_DEPTH,
                   help="Commons sub-category recursion depth")
    p.add_argument("--min-width", type=int, default=config.DEFAULT_MIN_WIDTH,
                   help="skip source images narrower than this (px)")
    p.add_argument("--download-width", type=int, default=config.DEFAULT_DOWNLOAD_WIDTH,
                   help="thumbnail width to request from Commons")
    p.add_argument("--max-size", type=int, default=config.DEFAULT_MAX_SIZE,
                   help="downscale saved images so the longest side <= this px "
                        "(0 keeps full resolution)")
    p.add_argument("--quality", type=int, default=config.DEFAULT_JPEG_QUALITY,
                   help="output JPEG quality")
    p.add_argument("--split", type=float, default=None, metavar="VAL_FRACTION",
                   help="also write train/ and test/ subfolders with this val fraction")
    p.add_argument("--registrations", default=None,
                   help="JSON {class: [regs...]} for the Planespotters source")
    p.add_argument("--seed", type=int, default=42, help="shuffle seed")
    p.add_argument("--api-throttle", type=float, default=config.API_THROTTLE,
                   help="seconds between api.php requests")
    p.add_argument("--download-throttle", type=float, default=config.DOWNLOAD_THROTTLE,
                   help="seconds between image downloads (raise this if you get "
                        "rate-limited / 403-blocked)")
    p.add_argument("--contact", default=None,
                   help="contact string for the User-Agent (recommended)")
    return p.parse_args(argv)


def records_for_class(
    cls: str,
    args: argparse.Namespace,
    commons: CommonsSource | None,
    planespotters: PlanespottersSource | None,
    reg_map: dict[str, list[str]] | None,
    rng: random.Random,
) -> Iterator[PhotoRecord]:
    """Yield (shuffled, deduped-by-source-id) candidate records for one class."""
    seen_ids: set[str] = set()

    if commons is not None:
        spec = config.CLASSES.get(cls)
        if not spec:
            print(f"  ! no Commons categories configured for {cls!r}", file=sys.stderr)
        else:
            # collect a bounded pool of titles, then shuffle for airline variety
            title_cap = max(args.per_class * 6, 300)
            titles = commons.collect_titles(
                spec["include"], spec.get("exclude"), limit=title_cap
            )
            rng.shuffle(titles)
            for rec in commons.resolve(titles, min_width=args.min_width):
                if rec.ident in seen_ids:
                    continue
                seen_ids.add(rec.ident)
                yield rec

    if planespotters is not None and reg_map is not None:
        regs = reg_map.get(cls, [])
        for rec in planespotters.photos_for_registrations(regs):
            if rec.ident in seen_ids:
                continue
            seen_ids.add(rec.ident)
            yield rec


def download_class(
    cls: str,
    args: argparse.Namespace,
    session: ThrottledSession,
    records: Iterator[PhotoRecord],
    global_hashes: set[str],
) -> int:
    slug = slugify(cls)
    out_dir = Path(args.out) / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    attr_path = out_dir / "_attribution.csv"
    new_file = not attr_path.exists()

    saved = 0
    idx = 0
    with open(attr_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ATTR_COLUMNS)
        if new_file:
            writer.writeheader()

        bar = tqdm(total=args.per_class, desc=cls, unit="img", leave=True)
        for rec in records:
            if saved >= args.per_class:
                break
            data = fetch_photo(session, rec)
            if data is None:
                continue

            digest = content_hash(data)
            if digest in global_hashes:
                continue  # exact duplicate already saved (any class)

            jpeg = to_jpeg(
                data,
                max_size=args.max_size,
                quality=args.quality,
                min_width=args.min_width,
            )
            if jpeg is None:
                continue

            global_hashes.add(digest)
            idx += 1
            fname = f"{slug}_{idx:04d}.jpg"
            (out_dir / fname).write_bytes(jpeg)
            writer.writerow({
                "filename": fname,
                "source": rec.source,
                "ident": rec.ident,
                "author": rec.author,
                "license": rec.license,
                "license_url": rec.license_url,
                "source_url": rec.source_url,
                "download_url": rec.download_url,
                "width": rec.width,
                "height": rec.height,
            })
            fh.flush()
            saved += 1
            bar.update(1)
        bar.close()
    return saved


def make_split(out_dir: Path, cls: str, val_fraction: float, rng: random.Random) -> None:
    """Copy a class folder into train/<cls> and test/<cls> subfolders."""
    import shutil

    src = out_dir / cls
    imgs = sorted(p for p in src.glob("*.jpg"))
    rng.shuffle(imgs)
    n_val = int(len(imgs) * val_fraction)
    val, train = imgs[:n_val], imgs[n_val:]
    for split, files in (("train", train), ("test", val)):
        dst = out_dir / split / cls
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dst / f.name)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    # resolve which classes to fetch: explicit --classes wins, else --group
    if args.classes:
        classes = args.classes
    elif args.group == "new":
        classes = list(config.NEW_CLASSES)
    elif args.group == "existing":
        classes = list(config.EXISTING_CLASSES)
    else:
        classes = list(config.CLASSES)

    ua = config.USER_AGENT
    if args.contact:
        ua = f"plane-classifier-scraper/{config.__dict__.get('__version__', '0.1')} ({args.contact})"
    session = ThrottledSession(user_agent=ua)
    session.set_gap("commons.wikimedia.org", args.api_throttle)
    session.set_gap("upload.wikimedia.org", args.download_throttle)
    session.set_gap("t.plnspttrs.net", args.download_throttle)
    # static media hosts reject bot UAs -> use a browser UA for downloads only
    session.set_ua("upload.wikimedia.org", config.BROWSER_UA)
    session.set_ua("t.plnspttrs.net", config.BROWSER_UA)

    commons = None
    planespotters = None
    reg_map = None
    if args.source in ("commons", "all"):
        commons = CommonsSource(
            session, max_depth=args.max_depth, download_width=args.download_width
        )
    if args.source in ("planespotters", "all"):
        planespotters = PlanespottersSource(session)
        if not args.registrations:
            print("ERROR: --source planespotters requires --registrations FILE",
                  file=sys.stderr)
            return 2
        reg_map = PlanespottersSource.load_registration_map(args.registrations)

    unknown = [c for c in classes if c not in config.CLASSES]
    if unknown and commons is not None:
        print(f"WARNING: classes not in config (Commons): {unknown}", file=sys.stderr)

    out_dir = Path(args.out)
    totals: dict[str, int] = {}
    global_hashes: set[str] = set()

    for cls in classes:
        print(f"\n=== {cls} ===")
        recs = records_for_class(cls, args, commons, planespotters, reg_map, rng)
        totals[cls] = download_class(cls, args, session, recs, global_hashes)
        if args.split is not None:
            make_split(out_dir, cls, args.split, rng)

    print("\n---- summary ----")
    for cls, n in totals.items():
        print(f"  {cls:28s} {n:4d} images")
    print(f"  TOTAL: {sum(totals.values())} images -> {out_dir}/")
    if args.split is not None:
        print(f"  train/test split written under {out_dir}/train and {out_dir}/test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
