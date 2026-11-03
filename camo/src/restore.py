"""Rule-based classical restoration, the arm R baseline. Nothing here is learned.

The quality vector q decides which course filter each image gets: a median filter
for noise, gamma correction for dark images, and unsharp masking for blur. Median
filtering comes first because gamma and sharpening would both amplify noise.
TAU and MEDIAN were chosen on validation by scripts/tune_restore.py.
"""
import math

import torch
from kornia.filters import gaussian_blur2d, median_blur

from quality import NAMES, quality_vector

TAU = 1.5           # act when a measure is this many standard deviations from the mean
MEDIAN = 3
TARGET_MEAN = 0.45
UNSHARP_SIGMA = 1.5
UNSHARP_AMOUNT = 1.0


def restore(rgb, q_mean, q_std, tau=None, median=None):
    """Return restored images in [0, 1]. Images that pass every rule stay unchanged."""
    tau = TAU if tau is None else tau
    median = median or MEDIAN
    z = (quality_vector(rgb) - q_mean) / q_std
    gray_mean = rgb.mean((1, 2, 3)).clamp(0.02, 0.98)
    noisy = z[:, NAMES.index("noise")] > tau
    dark = z[:, NAMES.index("brightness")] < -tau
    blurry = (z[:, NAMES.index("sharpness")] < -tau) & ~noisy

    def where(mask, new, old):
        return torch.where(mask.view(-1, 1, 1, 1), new, old)

    out = where(noisy, median_blur(rgb, median), rgb)
    gamma = (math.log(TARGET_MEAN) / gray_mean.log()).clamp(max=1).view(-1, 1, 1, 1)
    out = where(dark, out.clamp(min=1e-4) ** gamma, out)
    soft = gaussian_blur2d(out, (9, 9), (UNSHARP_SIGMA, UNSHARP_SIGMA))
    out = where(blurry, out + UNSHARP_AMOUNT * (out - soft), out)
    return out.clamp(0, 1)
