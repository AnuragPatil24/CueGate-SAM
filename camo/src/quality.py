"""Six classical image-quality measures. Nothing here is learned.

Each measure takes a batch of RGB images in [0, 1] with shape (B, 3, H, W) and
returns one number per image. quality_vector() stacks them into q, the input of
the gate. plan.md section 4.2 lists which lecture each measure comes from.
"""
import torch
import torch.nn.functional as F

NAMES = ["sharpness", "directionality", "noise", "brightness", "contrast", "blockiness"]
LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8
EPS = 1e-8


def to_gray(rgb):
    """Rec. 601 luma, with shape (B, 1, H, W)."""
    w = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (rgb * w).sum(dim=1, keepdim=True)


def laplacian(gray):
    return F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), LAPLACIAN.to(gray), padding=0)


def gradients(gray):
    """Sobel derivatives (Ix, Iy)."""
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    sobel_x = SOBEL_X.to(gray)
    return F.conv2d(padded, sobel_x), F.conv2d(padded, sobel_x.transpose(2, 3))


def sharpness(gray):
    """Log variance of the Laplacian. Blur removes high frequencies, so it falls."""
    return torch.log(laplacian(gray).var(dim=(1, 2, 3)) + EPS)


def directionality(gray):
    """Log ratio of the image-wide structure tensor's eigenvalues.

    Motion blur smooths along one direction only, so gradients become one-sided
    and the larger eigenvalue pulls away from the smaller one.
    """
    ix, iy = gradients(gray)
    jxx, jyy, jxy = (ix * ix).mean((1, 2, 3)), (iy * iy).mean((1, 2, 3)), (ix * iy).mean((1, 2, 3))
    half_trace = (jxx + jyy) / 2
    root = torch.sqrt(((jxx - jyy) / 2) ** 2 + jxy ** 2 + EPS)
    return torch.log((half_trace + root + EPS) / ((half_trace - root).clamp(min=0) + EPS))


def noise(gray, patch=8, share=0.10):
    """Log of the median variance in the flattest 10% of 8 x 8 patches.

    Flat patches hold almost no image detail, so what varies there is mostly noise.
    """
    mean = F.avg_pool2d(gray, patch, stride=patch)
    variance = (F.avg_pool2d(gray * gray, patch, stride=patch) - mean * mean).clamp(min=0)
    variance = variance.flatten(1).sort(dim=1).values
    keep = max(1, int(variance.shape[1] * share))
    return torch.log(variance[:, :keep].median(dim=1).values + EPS)


def brightness(gray):
    """Mean of the intensity histogram."""
    return gray.mean(dim=(1, 2, 3))


def contrast(gray):
    """Spread between the 5th and 95th intensity percentiles."""
    low, high = torch.quantile(gray.flatten(1), gray.new_tensor([0.05, 0.95]), dim=1)
    return high - low


def blockiness(gray):
    """Gradient across 8 x 8 block edges, divided by the gradient elsewhere.

    JPEG works in 8 x 8 blocks, so compression leaves steps at those edges. This
    only sees the grid if the image was not resized after JPEG.
    """
    def ratio(diff, axis):
        at_edge = torch.zeros(diff.shape[axis], dtype=torch.bool, device=diff.device)
        at_edge[7::8] = True
        dims = tuple(d for d in (1, 2, 3))
        edge = diff.index_select(axis, at_edge.nonzero().squeeze(1)).mean(dims)
        rest = diff.index_select(axis, (~at_edge).nonzero().squeeze(1)).mean(dims)
        return (edge + EPS) / (rest + EPS)

    across_cols = (gray[..., 1:] - gray[..., :-1]).abs()
    across_rows = (gray[..., 1:, :] - gray[..., :-1, :]).abs()
    return torch.log((ratio(across_cols, 3) + ratio(across_rows, 2)) / 2)


@torch.no_grad()
def quality_vector(rgb):
    """Return q with shape (B, 6), in the order of NAMES."""
    gray = to_gray(rgb)
    return torch.stack([sharpness(gray), directionality(gray), noise(gray), brightness(gray),
                        contrast(gray), blockiness(gray)], dim=1)
