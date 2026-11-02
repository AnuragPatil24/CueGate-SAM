"""Four classical cue maps, computed from the input image with no learning.

Each cue returns a tensor with shape (B, C, H, W). Fixed scale factors bring the
channels to roughly unit range. They are constants, not per-image stretches,
because the size of a cue is information: a blurred image should have a weak
detail cue. plan.md section 4.3 explains the choices.
"""
import kornia
import torch
import torch.nn.functional as F

from quality import gradients, laplacian, to_gray

CHANNELS = {"detail": 2, "colour": 2, "texture": 8, "structure": 2}
LBP_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def gaussian(x, sigma):
    size = 2 * int(3 * sigma) + 1      # three sigma on each side covers almost all the kernel
    return kornia.filters.gaussian_blur2d(x, (size, size), (sigma, sigma), border_type="replicate")


def detail_cue(gray):
    """Absolute Laplacian and absolute difference of Gaussians."""
    dog = gaussian(gray, 1.0) - gaussian(gray, 2.0)
    return torch.cat([laplacian(gray).abs() * 8, dog.abs() * 16], dim=1).clamp(max=4)


def colour_cue(rgb, window=31):
    """Lab a and b, minus their local means, so it shows colour contrast."""
    ab = kornia.color.rgb_to_lab(rgb)[:, 1:]
    return (ab - F.avg_pool2d(F.pad(ab, (window // 2,) * 4, mode="replicate"), window, stride=1)) / 16


def texture_cue(gray, window=7):
    """Local Binary Pattern sign bits, each averaged over a small box.

    The eight bits stay as separate channels instead of one LBP code, because
    codes are labels and can't be averaged or resized.
    """
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    h, w = gray.shape[2:]
    bits = [(padded[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w] >= gray).float()
            for dy, dx in LBP_NEIGHBOURS]
    bits = torch.cat(bits, dim=1)
    return F.avg_pool2d(F.pad(bits, (window // 2,) * 4, mode="replicate"), window, stride=1) - 0.5


def structure_cue(gray, sigma=2.0):
    """Coherence and smaller eigenvalue of the structure tensor.

    Gaussian smoothing at sigma = 2 keeps this cue at a coarse scale, so it should
    survive fine damage better than the detail cue.
    """
    ix, iy = gradients(gray)
    jxx, jyy, jxy = gaussian(ix * ix, sigma), gaussian(iy * iy, sigma), gaussian(ix * iy, sigma)
    half_trace = (jxx + jyy) / 2
    root = torch.sqrt(((jxx - jyy) / 2) ** 2 + jxy ** 2 + 1e-10)
    coherence = root / (half_trace + 1e-6)
    smaller = (half_trace - root).clamp(min=0)
    return torch.cat([coherence, torch.log1p(smaller * 1e4)], dim=1)


@torch.no_grad()
def compute_cues(rgb):
    """Return every cue for a batch of RGB images in [0, 1]."""
    gray = to_gray(rgb)
    return {"detail": detail_cue(gray), "colour": colour_cue(rgb),
            "texture": texture_cue(gray), "structure": structure_cue(gray)}
