"""Dataset and augmentation classes for the plane classifier.

These live in a module (not the notebook) so PyTorch ``DataLoader`` worker processes can
import them. On Windows the ``spawn`` start method re-imports code in each worker, and
classes defined inside a notebook can't be reconstructed there -- which is why
``num_workers > 0`` fails when everything lives in the notebook. Keeping them here lets us
use several workers so image decoding/augmentation overlaps GPU compute (otherwise the GPU
sits idle waiting on a single-threaded data pipeline).
"""

import io
import os
import zlib

import numpy as np
from PIL import Image, ImageEnhance, ImageFile
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

# Some web-scraped JPEGs are slightly truncated; let PIL decode them instead of raising.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def stable_seed(key: str) -> int:
    """Deterministic 32-bit seed from a string (crc32 -- unlike ``hash()`` it isn't salted,
    so field-sim corruptions are identical across kernel restarts)."""
    return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF


class FieldCorruptions:
    """Simulate the gap between pristine planespotter photos (training domain) and a phone
    pointed at a distant landing aircraft (deployment domain).

    The training set is sharp, telephoto, side-on, well-exposed. The field is not: a phone's
    digital zoom is soft, the subject is small and hazy, there's motion/camera shake, the sky
    backlights the airframe, and everything is re-compressed. A model tuned only on pristine
    images exploits fine detail that simply isn't present in the field.

    This class applies that family of corruptions. It is used two ways:

    * **Evaluation** (``deterministic``): each held-out val image is corrupted once, with a
      severity fixed by its filename, giving a stable "field-sim" macro-F1 that actually
      tracks deployment -- unlike pristine-val F1, which measures the wrong distribution.
    * **Training** (randomised each epoch, wider ranges): teaches invariance to these effects.

    To avoid "grading our own homework", the eval instance uses a *narrower, fixed* severity
    band while the train augmenter randomises widely -- same corruption families, different
    realisations (as in ImageNet-C's train/test severity split).
    """

    def __init__(self, severity=1.0, probs=None, max_side=512):
        self.severity = severity
        self.max_side = max_side
        # Per-corruption application probabilities. Overridable so train (aggressive) and
        # eval (moderate, always-on-ish) can differ.
        self.probs = probs or dict(
            perspective=0.6, zoom_soft=0.8, motion_blur=0.5,
            haze=0.6, lighting=0.6, jpeg=0.8,
        )

    def _bound_size(self, img):
        """Downscale so the longest side <= max_side BEFORE corrupting. Without this, a
        full-res original (e.g. a loose-crop training sample) would make motion blur's
        per-pixel np.roll loop take seconds -- and the classifier only needs ~256px anyway."""
        w, h = img.size
        s = self.max_side / max(w, h)
        if s < 1.0:
            img = img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                             Image.BILINEAR)
        return img

    # -- individual corruptions -------------------------------------------------
    def _perspective(self, img, rng):
        """Approximate a viewpoint change (e.g. side-profile training image -> the belly/
        head-on look of an aircraft on approach) by warping the image corners.

        Two effects are combined: (a) random corner jitter (general 3D-ish wobble), and
        (b) a trapezoidal foreshortening where one edge is squeezed toward the centre --
        a crude stand-in for looking UP at an angled underside (the deployment view).
        This is still a 2D warp and cannot truly synthesise a belly view from a side
        profile; real below-angle data is the only full fix."""
        w, h = img.size
        d = 0.28 * self.severity
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        # (a) corner jitter
        ex = [rng.uniform(0, d * w) for _ in range(4)]
        ey = [rng.uniform(0, d * h) for _ in range(4)]
        endpoints = [
            (ex[0], ey[0]),
            (w - ex[1], ey[1]),
            (w - ex[2], h - ey[2]),
            (ex[3], h - ey[3]),
        ]
        # (b) trapezoidal foreshortening: pull one randomly-chosen edge inward.
        squeeze = rng.uniform(0.10, 0.45) * self.severity
        edge = rng.integers(0, 4)
        if edge == 0:      # top edge narrows (looking up at the far top)
            endpoints[0] = (endpoints[0][0] + squeeze * w, endpoints[0][1])
            endpoints[1] = (endpoints[1][0] - squeeze * w, endpoints[1][1])
        elif edge == 1:    # bottom edge narrows
            endpoints[3] = (endpoints[3][0] + squeeze * w, endpoints[3][1])
            endpoints[2] = (endpoints[2][0] - squeeze * w, endpoints[2][1])
        elif edge == 2:    # left edge shortens vertically
            endpoints[0] = (endpoints[0][0], endpoints[0][1] + squeeze * h)
            endpoints[3] = (endpoints[3][0], endpoints[3][1] - squeeze * h)
        else:              # right edge shortens vertically
            endpoints[1] = (endpoints[1][0], endpoints[1][1] + squeeze * h)
            endpoints[2] = (endpoints[2][0], endpoints[2][1] - squeeze * h)
        return TF.perspective(img, corners, endpoints,
                              interpolation=TF.InterpolationMode.BILINEAR)

    def _zoom_soft(self, img, rng):
        """Digital-zoom softness: throw away real detail by down- then up-sampling.

        Kept mild: a modern phone camera is sharp, so even a zoomed-in far-away
        subject only loses a little detail. Factor ~1.1..2.5 at severity=1 (was 2.0..7.0)."""
        w, h = img.size
        factor = rng.uniform(1.1, 1.3 + 1.2 * self.severity)
        sw, sh = max(1, int(round(w / factor))), max(1, int(round(h / factor)))
        return img.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.BILINEAR)

    def _motion_blur(self, img, rng):
        """Directional blur from hand shake / a moving aircraft: average shifted copies."""
        length = int(rng.integers(5, 6 + int(18 * self.severity)))
        angle = rng.uniform(0, np.pi)
        dx, dy = np.cos(angle), np.sin(angle)
        arr = np.asarray(img, dtype=np.float32)
        acc = np.zeros_like(arr)
        for i in range(length):
            t = i - length / 2.0
            acc += np.roll(np.roll(arr, int(round(dy * t)), axis=0),
                           int(round(dx * t)), axis=1)
        acc /= length
        return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))

    def _haze(self, img, rng):
        """Atmospheric haze on a distant subject: blend toward a bright sky tone and flatten
        contrast.

        Kept mild: a good phone camera captures a fairly clear image, so this is a light
        veil rather than thick fog. Strength ~0.05..0.28 at severity=1 (was 0.1..0.55)."""
        a = rng.uniform(0.05, 0.08 + 0.2 * self.severity)
        tone = np.array([rng.uniform(185, 225), rng.uniform(190, 228),
                         rng.uniform(200, 235)], dtype=np.float32)   # slightly blue
        arr = np.asarray(img, dtype=np.float32)
        out = arr * (1 - a) + tone[None, None, :] * a
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def _lighting(self, img, rng):
        """Bright sky behind the plane: either wash the whole frame out (overexposure) or
        darken + flatten it (a backlit silhouette)."""
        if rng.random() < 0.5:                     # overexposed
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(1.2, 1.6))
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 0.9))
        else:                                      # backlit / silhouetted
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.5, 0.8))
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.5, 0.8))
        return img

    def _jpeg(self, img, rng):
        """Re-encode as a low-quality JPEG (phone capture + messaging recompression)."""
        q = int(rng.integers(18, 60))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    # -- driver -----------------------------------------------------------------
    def __call__(self, img, seed=None):
        """Corrupt ``img``. If ``seed`` is given the result is deterministic (used to build a
        stable field-sim eval set); otherwise a fresh random draw (training)."""
        rng = np.random.default_rng(seed)
        img = self._bound_size(img)
        order = [
            ("perspective", self._perspective),
            ("zoom_soft", self._zoom_soft),
            ("motion_blur", self._motion_blur),
            ("haze", self._haze),
            ("lighting", self._lighting),
            ("jpeg", self._jpeg),
        ]
        for name, fn in order:
            if rng.random() < self.probs.get(name, 0.0):
                img = fn(img, rng)
        return img


class AircraftAugmenter:
    """Aspect-preserving letterbox + normalise (eval) and light photometric+geometric
    augmentation (train).

    ``yolo_crop_fn`` is an optional ``callable(PIL.Image) -> PIL.Image`` used only at
    inference time to crop un-cropped photos to the aircraft. It's left as ``None`` for
    training because images are already pre-cropped, and because the YOLO model isn't
    picklable into worker processes.
    """

    def __init__(self, img_height, img_width, norm_mean, norm_std,
                 zoom_blur_prob=0.3, zoom_blur_range=(2.0, 4.0), yolo_crop_fn=None,
                 pad_fill=(0, 0, 0), field_corruptions=None):
        self.img_height = img_height
        self.img_width = img_width
        self.zoom_blur_prob = zoom_blur_prob
        self.zoom_blur_range = zoom_blur_range
        self.yolo_crop_fn = yolo_crop_fn
        self.pad_fill = pad_fill
        # Optional FieldCorruptions applied on the TRAIN path only, to close the gap to the
        # phone/landing-approach deployment domain (viewpoint, haze, motion blur, backlight,
        # digital-zoom softness, recompression). Randomised each call. Left None -> the old
        # light-augmentation behaviour, so the two can be compared.
        self.field_corruptions = field_corruptions

        # NOTE: we deliberately do NOT use ``T.Resize((h, w))`` anywhere. Aircraft crops are
        # wide (a fuselage is ~3:1), and squashing them into a square destroys the very
        # aspect-ratio cue that distinguishes families (an A380 vs a CRJ). ``_letterbox``
        # below rescales preserving aspect and pads the remainder, so the whole airframe is
        # visible undistorted. Both the eval and train paths start from a letterboxed image.
        self.zoom_crop = T.RandomResizedCrop(
            size=(img_height, img_width), scale=(0.85, 1.0), ratio=(0.9, 1.1)
        )
        self.augment_pipeline = T.Compose([
            T.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))], p=0.2),
        ])
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
        ])

    def apply_digital_zoom_blur(self, img_pil):
        """Simulate the softness of digital zoom on a far-away subject: downscale the image
        (discarding real detail) and upscale it back to full size. Framing is unchanged;
        only sharpness drops."""
        if np.random.rand() > self.zoom_blur_prob:
            return img_pil
        w, h = img_pil.size
        factor = np.random.uniform(*self.zoom_blur_range)
        small_w, small_h = max(1, int(round(w / factor))), max(1, int(round(h / factor)))
        down = img_pil.resize((small_w, small_h), Image.BILINEAR)
        return down.resize((w, h), Image.BILINEAR)

    def apply_obstruction(self, img_pil):
        """Randomly paint a small black rectangle to simulate obstructions."""
        if np.random.rand() > 0.2:
            return img_pil
        img_np = np.array(img_pil).copy()
        h, w = img_np.shape[:2]
        obs_h, obs_w = np.random.randint(10, 50), np.random.randint(10, 50)
        y = np.random.randint(0, max(1, h - obs_h))
        x = np.random.randint(0, max(1, w - obs_w))
        img_np[y:y + obs_h, x:x + obs_w, :] = 0
        return Image.fromarray(img_np)

    def _letterbox(self, img_pil):
        """Resize preserving aspect ratio to fit (img_width, img_height), padding the
        leftover strips with ``pad_fill``. Only the single scale + paste happens here, so
        this is the *one* geometric resample on the eval path (no redundant pre-resize)."""
        w, h = img_pil.size
        scale = min(self.img_width / w, self.img_height / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img_pil.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.img_width, self.img_height), self.pad_fill)
        canvas.paste(resized, ((self.img_width - nw) // 2, (self.img_height - nh) // 2))
        return canvas

    def __call__(self, img_pil, augment=False):
        if self.yolo_crop_fn is not None:
            img_pil = self.yolo_crop_fn(img_pil)
        if not augment:
            # Eval path: any field-sim corruption is injected by the Dataset (deterministic)
            # BEFORE this call, so here we only letterbox + normalise.
            img_pil = self._letterbox(img_pil)
            return self.to_tensor(img_pil)
        # Train path. Domain-matched corruptions first (on the full crop, before letterbox so
        # perspective/blur see true pixel geometry), then the lighter photometric/geometric
        # augmentation. When field_corruptions is set it supersedes the old single zoom-blur.
        if self.field_corruptions is not None:
            img_pil = self.field_corruptions(img_pil)   # random seed -> fresh each epoch
        img_pil = self._letterbox(img_pil)
        img_pil = self.zoom_crop(img_pil)
        if self.field_corruptions is None:
            img_pil = self.apply_digital_zoom_blur(img_pil)
        img_pil = self.augment_pipeline(img_pil)
        img_pil = self.apply_obstruction(img_pil)
        return self.to_tensor(img_pil)


class AircraftDataset(Dataset):
    """Reads the pre-cropped image for each (path, label) sample and applies the augmenter.

    ``crop_dir`` holds the YOLO-cropped images (keyed by original filename); the original
    extracted image is used as a fallback if a crop is missing.

    ``field_corruptions`` (optional): a ``FieldCorruptions`` used to build a *deterministic*
    "field-sim" evaluation set -- each image is corrupted once, seeded by its filename, so the
    resulting macro-F1 is stable across runs and reflects the phone/approach deployment domain
    rather than the pristine training domain. Only meaningful with ``augment=False``.

    ``loose_crop_prob`` (train only): with this probability, use the *un-cropped* original
    image instead of the tight YOLO crop. In the field YOLO often returns a loose box or
    misses a small/distant plane entirely, so the classifier must tolerate lots of background;
    training exclusively on tight crops bakes in a train/serve skew.
    """

    def __init__(self, samples, augmenter, augment, crop_dir,
                 field_corruptions=None, loose_crop_prob=0.0):
        self.samples = samples
        self.augmenter = augmenter
        self.augment = augment
        self.crop_dir = crop_dir
        self.field_corruptions = field_corruptions
        self.loose_crop_prob = loose_crop_prob

    def __len__(self):
        return len(self.samples)

    def _crop_path(self, orig_path):
        return os.path.join(self.crop_dir, os.path.basename(orig_path))

    def __getitem__(self, idx):
        # Skip any unreadable (truncated/corrupt) file by walking forward, so one bad
        # image never crashes an epoch.
        for offset in range(len(self.samples)):
            orig_path, label = self.samples[(idx + offset) % len(self.samples)]
            base = os.path.basename(orig_path)
            crop_path = self._crop_path(orig_path)
            # Train-time loose-crop simulation: sometimes feed the full original image.
            use_loose = (self.augment and self.loose_crop_prob > 0
                         and np.random.rand() < self.loose_crop_prob)
            if use_loose and os.path.exists(orig_path):
                path = orig_path
            else:
                path = crop_path if os.path.exists(crop_path) else orig_path
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                continue
            if self.field_corruptions is not None:
                # Deterministic per-image corruption for a stable field-sim eval metric.
                image = self.field_corruptions(image, seed=stable_seed(base))
            return self.augmenter(image, augment=self.augment), label
        raise RuntimeError("No readable images found in dataset.")
