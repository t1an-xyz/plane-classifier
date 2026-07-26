# Aircraft image scraper (Wikimedia Commons + Planespotters)

Collects **license-clean** aircraft photos to augment the
`Commercial aircraft classification/` dataset for the plane-classifier project.

The intended workflow:

1. Base-train a model on the FGVC-Aircraft dataset (done separately, on Colab).
2. Fine-tune on `Commercial aircraft classification/`.
3. **Use this tool** to (a) add new regional-jet classes the dataset lacks and
   (b) top up the existing classes with more images.

Primary source is **Wikimedia Commons** (CC / public-domain, redistributable,
rich: thousands of photos per type). **Planespotters** is an optional
supplement (see caveats below).

---

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Before any large run, set a contact string so Wikimedia can reach you
(their policy asks for it). Either edit `CONTACT` in `scraper/config.py`
or pass `--contact "you@example.com"`.

---

## Usage

```bash
# The 4 NEW regional-jet classes (default), 300 imgs each -> ./scraped_data
.venv/bin/python scrape.py --contact "you@example.com"

# Top up the 17 EXISTING dataset classes instead
.venv/bin/python scrape.py --group existing --per-class 150

# Everything (new + existing)
.venv/bin/python scrape.py --group all

# A specific subset
.venv/bin/python scrape.py --classes "Embraer ERJ" "Boeing 777"

# Also emit an 80/20 train/test split ready to merge into the dataset
.venv/bin/python scrape.py --split 0.2

# If you get rate-limited / 403-blocked, slow down and resume
.venv/bin/python scrape.py --download-throttle 1.5 --api-throttle 1.5
```

Output layout (folder names match the dataset exactly, so merging is trivial):

```
scraped_data/
  Embraer ERJ/
    embraer_erj_0001.jpg
    ...
    _attribution.csv        # photographer + license + source URL per image
  train/ Embraer ERJ/ ...   # only if --split was given
  test/  Embraer ERJ/ ...
```

Merge into the dataset when you're happy with a class, e.g.:

```bash
cp -r scraped_data/train/"Embraer ERJ" "Commercial aircraft classification/train/"
cp -r scraped_data/test/"Embraer ERJ"  "Commercial aircraft classification/test/"
```

### Key options

| flag | default | meaning |
|------|---------|---------|
| `--group {new,existing,all}` | `new` | which class set to fetch |
| `--classes ...` | — | explicit class names (overrides `--group`) |
| `--per-class N` | 300 | max images kept per class |
| `--max-size PX` | 1280 | downscale longest side (0 = keep full res) |
| `--min-width PX` | 800 | skip source images narrower than this |
| `--split FRAC` | off | also write `train/`+`test/` with this val fraction |
| `--download-throttle S` | 0.5 | seconds between downloads (raise if blocked) |
| `--source {commons,planespotters,all}` | `commons` | image source(s) |

---

## How it works

- **Discovery** uses CirrusSearch's server-side `deepcat:` operator, which
  expands a category's entire sub-tree in one paginated query (thousands of
  hits per type in well under a second). Class → category mappings and
  per-class exclusions (e.g. keep 777X out of 777) live in `scraper/config.py`.
- **Downloads fetch the ORIGINAL file**, not an on-the-fly thumbnail. This is
  deliberate: Wikimedia's thumbnail-render service rate-limits hard and will
  temporarily **403-block your IP** if you request many distinct renders
  quickly. Originals are served from cache and avoid this. Files are then
  downscaled locally (`--max-size`) and re-encoded to RGB JPEG.
- **Filtering** drops cockpits/cabins/interiors/diagrams/models/accidents
  (see `EXCLUDE_KEYWORDS`), non-photos, and images below `--min-width`.
- **Dedup** is by SHA-256 of image bytes, across the whole run, so the same
  photo never lands in two classes.
- **Attribution** for every saved image is logged to `_attribution.csv`
  (author, license, license URL, Commons page). Keep this: most Commons
  images are CC-BY/CC-BY-SA and require credit if you redistribute.

### Data availability (verified `deepcat` hit counts)

| class | approx. candidate photos |
|-------|--------------------------|
| Embraer E-Jet | ~11,500 |
| Bombardier CRJ 700-1000 | ~4,400 |
| Bombardier CRJ 100-200 | ~3,300 |
| Embraer ERJ | ~3,550 |
| existing classes (A319, A320, 737NG, 747, 777, …) | 10k–30k each |

Commons alone is more than enough for a few hundred clean images per class.

---

## Planespotters (optional, limited)

The public Planespotters photo API only returns **one ~280px thumbnail per
aircraft registration** and has no "search by type" endpoint. Photos remain
**copyright the photographer** (attribution + link required, non-commercial) —
fine for private model training, **not** for redistribution. It's only worth
using to top up genuinely rare variants.

```bash
# regs.json:  {"Bombardier CRJ 100-200": ["N8928A", "N875AS", ...]}
.venv/bin/python scrape.py --source planespotters --registrations regs.json
```

---

## Licensing note

Wikimedia Commons images are freely licensed (mostly CC-BY / CC-BY-SA / public
domain) and are safe to train on and redistribute **with attribution** — that's
why `_attribution.csv` is written alongside every class. Review it before
publishing any dataset or model artifacts built from these images.
