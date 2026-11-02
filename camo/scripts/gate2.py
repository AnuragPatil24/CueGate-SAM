"""Gate 2: does each quality measure track the damage it is meant to detect?

Also saves the z-score statistics that the gate needs, measured on training images
under the same random damage the model sees in training.

    python camo/scripts/gate2.py
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from corruptions import COMBINED, CORRUPTIONS, corrupt
from data import read_pair, split_names
from quality import NAMES, quality_vector

TRAIN_ROOT = os.path.join(ROOT, "data", "COD-TrainDataset")
TESTS = [("sharpness", "gaussian_blur"), ("sharpness", "motion_blur"),
         ("directionality", "motion_blur"), ("noise", "gaussian_noise"),
         ("brightness", "brightness"), ("contrast", "contrast"), ("contrast", "fog"),
         ("blockiness", "jpeg")]
IMAGES = 300


def measure(images, device):
    batch = np.stack(images).transpose(0, 3, 1, 2).astype(np.float32) / 255
    return quality_vector(torch.from_numpy(batch).to(device)).cpu().numpy()


def main():
    device = torch.device("cuda")
    train, val = split_names(TRAIN_ROOT)
    rng = np.random.default_rng(604)

    pool = [train[i] for i in rng.permutation(len(train))[:IMAGES]]
    images = [read_pair(TRAIN_ROOT, n)[0] for n in pool]
    choices = list(CORRUPTIONS) + list(COMBINED)
    damaged = [img if rng.random() < 0.2 else
               corrupt(img, choices[rng.integers(len(choices))], int(rng.integers(1, 5)),
                       int(rng.integers(2 ** 31))) for img in images]
    q = np.concatenate([measure(damaged[i:i + 32], device) for i in range(0, IMAGES, 32)])
    stats = {"names": NAMES, "mean": q.mean(0).tolist(), "std": q.std(0).tolist()}
    with open(os.path.join(ROOT, "results", "q_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    images = [read_pair(TRAIN_ROOT, n)[0] for n in val[:100]]
    print(f"{'measure':16s}{'corruption':16s}{'spearman':>10s}  rho on each image, then averaged")
    passed = True
    for measure_name, corruption in TESTS:
        k = NAMES.index(measure_name)
        scores = [measure(images, device)[:, k]]
        for s in range(1, 6):
            batch = [corrupt(img, corruption, s, 604 + i) for i, img in enumerate(images)]
            scores.append(measure(batch, device)[:, k])
        scores = np.stack(scores, axis=1)
        rho = np.mean([spearmanr(range(6), row).statistic for row in scores])
        passed &= abs(rho) >= 0.7
        print(f"{measure_name:16s}{corruption:16s}{rho:10.3f}  {'pass' if abs(rho) >= 0.7 else 'FAIL'}")
    print("Gate 2 passed" if passed else "Gate 2 FAILED on the rows above")


if __name__ == "__main__":
    main()
