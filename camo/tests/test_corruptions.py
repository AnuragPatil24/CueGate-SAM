"""Checks that the corruptions behave. Run with: python camo/tests/test_corruptions.py"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from corruptions import COMBINED, CORRUPTIONS, apply_protocol, corrupt, image_seed
from metrics import binary_scores, stretch

rng = np.random.default_rng(0)
smooth = np.tile(np.linspace(0, 255, 500, dtype=np.uint8)[None, :, None], (400, 1, 3))
image = np.clip(smooth.astype(int) + rng.integers(-20, 20, smooth.shape), 0, 255).astype(np.uint8)


def test_every_corruption_changes_image_and_keeps_format():
    for name in list(CORRUPTIONS) + list(COMBINED):
        out = corrupt(image, name, 3, seed=1)
        assert out.dtype == np.uint8 and out.shape == image.shape, name
        assert (out != image).mean() > 0.05, name


def test_same_seed_gives_same_damage():
    for name in list(CORRUPTIONS) + list(COMBINED):
        assert np.array_equal(corrupt(image, name, 3, 7), corrupt(image, name, 3, 7)), name


def test_seed_is_stable_across_runs():
    assert image_seed("a.jpg", "fog", 3) == 4165992957


def test_severity_makes_noise_stronger():
    low = np.abs(corrupt(image, "gaussian_noise", 1, 1).astype(int) - image).mean()
    high = np.abs(corrupt(image, "gaussian_noise", 5, 1).astype(int) - image).mean()
    assert high > low * 2


def test_shrinking_removes_noise_under_native_protocol():
    # This is what Gate 1 tests on real images: noise added before a shrink fades.
    clean = apply_protocol(smooth, "clean", 0, 1, "input")
    native = apply_protocol(smooth, "gaussian_noise", 5, 1, "native")
    direct = apply_protocol(smooth, "gaussian_noise", 5, 1, "input")
    assert np.abs(native.astype(int) - clean).mean() < np.abs(direct.astype(int) - clean).mean()


def test_stretch_inflates_a_weak_prediction():
    prob = np.full((64, 64), 0.1)
    prob[20:44, 20:44] = 0.2
    gt = prob > 0.15
    assert binary_scores(prob, gt)[0] == 0.0
    assert binary_scores(stretch(prob), gt)[0] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
