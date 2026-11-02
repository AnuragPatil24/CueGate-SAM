"""Image corruptions used to simulate poor-quality cameras.

Every function takes a uint8 RGB image and a severity from 1 to 5. Severities
for the first seven corruptions are the COD10K-C values (arXiv 2606.02603), so
results can be compared. COD10K-C gives no exact rain settings, so rain is our
own design.

Three combined conditions (drone, night, cctv) chain single corruptions in
physical order: blur, then darkening, then noise, then JPEG.

Two protocols decide where the damage happens (see plan.md section 4.5):

  input  : resize to the model's input size, then corrupt (low-resolution camera)
  native : corrupt at full image size, then resize (high-resolution camera)
"""
import zlib

import cv2
import numpy as np
from PIL import Image

INPUT_SIZE = 352

NOISE_SIGMA = [8, 16, 24, 32, 40]
MOTION_KERNEL = [3, 5, 7, 9, 11]
BLUR_RADIUS = [0.8, 1.2, 1.6, 2.0, 2.5]
DIM_FACTOR = [0.85, 0.70, 0.55, 0.40, 0.30]
FOG_ALPHA = [0.12, 0.20, 0.28, 0.36, 0.44]
JPEG_QUALITY = [70, 55, 40, 30, 20]
RAIN_DENSITY = [0.002, 0.004, 0.006, 0.008, 0.010]
RAIN_LENGTH = [8, 12, 16, 20, 24]


def _clip(img):
    """Round a float image back to valid uint8 pixel values."""
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def gaussian_noise(img, s, rng):
    return _clip(img.astype(np.float32) + rng.normal(0, NOISE_SIGMA[s - 1], img.shape))


def gaussian_blur(img, s, rng):
    return cv2.GaussianBlur(img, (0, 0), BLUR_RADIUS[s - 1])


def motion_blur(img, s, rng):
    # A horizontal line kernel, the same as a camera panning sideways.
    k = MOTION_KERNEL[s - 1]
    return cv2.filter2D(img, -1, np.ones((1, k), np.float32) / k, borderType=cv2.BORDER_REFLECT)


def brightness(img, s, rng):
    return _clip(img.astype(np.float32) * DIM_FACTOR[s - 1])


def contrast(img, s, rng):
    # Pull each pixel towards the image mean.
    mean = img.mean(axis=(0, 1), keepdims=True)
    return _clip(mean + (img.astype(np.float32) - mean) * DIM_FACTOR[s - 1])


def fog(img, s, rng):
    a = FOG_ALPHA[s - 1]
    return _clip(img.astype(np.float32) * (1 - a) + 255 * a)


def jpeg(img, s, rng):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY[s - 1]])
    assert ok, "JPEG encoding failed"
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def rain(img, s, rng):
    # Draw random bright slanted streaks on a blank layer, then add it on top.
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.uint8)
    length = RAIN_LENGTH[s - 1]
    n = int(RAIN_DENSITY[s - 1] * h * w)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    dx = int(length * np.tan(np.deg2rad(rng.uniform(-20, 20))))
    for x, y in zip(xs, ys):
        cv2.line(layer, (int(x), int(y)), (int(x) + dx, int(y) + length), 200, 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0).astype(np.float32)[..., None]
    return _clip(img.astype(np.float32) * 0.9 + layer * 0.5)


def drone(img, rng):
    # Moving camera, heavily compressed video link.
    return jpeg(motion_blur(img, 3, rng), 3, rng)


def night(img, rng):
    # Dark scene, high sensor gain, moderate compression.
    return jpeg(gaussian_noise(brightness(img, 4, rng), 3, rng), 2, rng)


def cctv(img, rng):
    # Soft lens, noisy sensor, low bitrate.
    return jpeg(gaussian_noise(gaussian_blur(img, 3, rng), 2, rng), 4, rng)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise, "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur, "brightness": brightness, "contrast": contrast,
    "fog": fog, "jpeg": jpeg, "rain": rain,
}
COMBINED = {"drone": drone, "night": night, "cctv": cctv}


def image_seed(name, corruption, severity):
    """Stable seed, so every model and every rerun sees identical damage."""
    return zlib.crc32(f"{name}|{corruption}|{severity}".encode())


def corrupt(img, corruption, severity, seed):
    """Apply one corruption or combined condition to a uint8 RGB image."""
    rng = np.random.default_rng(seed)
    if corruption == "clean":
        return img
    if corruption in COMBINED:
        return COMBINED[corruption](img, rng)
    return CORRUPTIONS[corruption](img, severity, rng)


def resize(img, size=INPUT_SIZE):
    """Resize the way the official SAM2-UNet and SINet-V2 test code does.

    PIL bilinear smooths the image while shrinking it. OpenCV's INTER_LINEAR
    does not, and would keep more noise, so the two must not be mixed here.
    """
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def apply_protocol(img, corruption, severity, seed, protocol, size=INPUT_SIZE):
    """Return the image a model receives, at its input size."""
    if protocol == "input":
        return corrupt(resize(img, size), corruption, severity, seed)
    if protocol == "native":
        return resize(corrupt(img, corruption, severity, seed), size)
    raise ValueError(f"unknown protocol {protocol}")
