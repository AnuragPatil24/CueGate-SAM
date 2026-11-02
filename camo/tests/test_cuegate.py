"""Smoke test: a new CueGate model must equal SAM2-UNet, and only new parts train.

Run from the project root: python camo/tests/test_cuegate.py
"""
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cuegate import CueGateSAM, trainable_parameters
from models import new_sam2unet
from quality import quality_vector

Q_MEAN, Q_STD = [0.0] * 6, [1.0] * 6


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn(2, 3, 352, 352, device=device)
    base = new_sam2unet().to(device).eval()
    with torch.inference_mode():
        expected = base(x)[0]
    del base

    for arm in "BCDE":
        model = CueGateSAM(new_sam2unet(), arm, Q_MEAN, Q_STD).to(device).eval()
        with torch.inference_mode():
            out = model(x)[0]
        gap = (out - expected).abs().max().item()
        assert gap == 0, f"arm {arm} differs from SAM2-UNet by {gap}"
        trained = sum(p.numel() for p in trainable_parameters(model)) / 1e6
        total = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"arm {arm}: equal to SAM2-UNet, {trained:.2f}M of {total:.1f}M parameters train")
        del model
        torch.cuda.empty_cache()

    start = time.time()
    q = quality_vector(torch.rand(6, 3, 352, 352, device=device))
    assert q.shape == (6, 6) and torch.isfinite(q).all()
    print(f"quality vector finite, {time.time() - start:.3f}s for 6 images")


if __name__ == "__main__":
    main()
