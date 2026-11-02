"""Training data: images resized, randomly flipped, and usually corrupted.

Corruption follows the input protocol (resize, then corrupt), because that's the
protocol with real damage. Validation uses a fixed set of conditions with stable
seeds, so scores can be compared across epochs and arms.
"""
import os
import zlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from corruptions import COMBINED, CORRUPTIONS, corrupt, image_seed, resize

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
VAL_SIZE = 400
CLEAN_SHARE = 0.2
VAL_CONDITIONS = [("clean", 0), ("motion_blur", 3), ("gaussian_noise", 3), ("jpeg", 4),
                  ("drone", 0), ("night", 0), ("cctv", 0)]


def split_names(root):
    """Return (train, val) file names, with a seeded validation hold-out."""
    names = sorted(os.listdir(os.path.join(root, "Imgs")))
    order = np.random.default_rng(604).permutation(len(names))
    val = {names[i] for i in order[:VAL_SIZE]}
    return [n for n in names if n not in val], sorted(val)


def read_pair(root, name):
    """Read one image and mask, resized to the model's input size."""
    rgb = cv2.cvtColor(cv2.imread(os.path.join(root, "Imgs", name)), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(os.path.join(root, "GT", os.path.splitext(name)[0] + ".png"), cv2.IMREAD_GRAYSCALE)
    return resize(rgb), resize(gt)


def to_tensors(rgb, gt):
    x = (rgb.transpose(2, 0, 1).astype(np.float32) / 255 - MEAN) / STD
    y = (gt.astype(np.float32) / 255)[None]
    return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


class TrainSet(Dataset):
    """Each sample is clean, or has one corruption at severity 1-4, or a combined one.

    Severity 5 stays out of training, so it remains a harder test. With augment
    False, every image is returned clean except for the flip.
    """

    def __init__(self, root, names, augment=True):
        self.root, self.names, self.augment = root, names, augment
        self.choices = list(CORRUPTIONS) + list(COMBINED)
        self.epoch = 0

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        # Seeded from the name and epoch, so a resumed run sees the same damage.
        rng = np.random.default_rng(zlib.crc32(f"{name}|{self.epoch}".encode()))
        rgb, gt = read_pair(self.root, name)
        if rng.random() < 0.5:
            rgb, gt = rgb[:, ::-1], gt[:, ::-1]
        if self.augment and rng.random() >= CLEAN_SHARE:
            kind = self.choices[rng.integers(len(self.choices))]
            rgb = corrupt(rgb.copy(), kind, int(rng.integers(1, 5)), int(rng.integers(2 ** 31)))
        return to_tensors(rgb, gt)


class ValSet(Dataset):
    """Hold-out images under one fixed condition."""

    def __init__(self, root, names, corruption, severity):
        self.root, self.names = root, names
        self.corruption, self.severity = corruption, severity

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        rgb, gt = read_pair(self.root, name)
        seed = image_seed(name, self.corruption, self.severity)
        return to_tensors(corrupt(rgb, self.corruption, self.severity, seed), gt)
