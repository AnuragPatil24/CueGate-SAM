"""Choose arm R's threshold and median size on validation images only.

    python camo/scripts/tune_restore.py
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cuegate import MEAN, STD, load_q_stats
from data import VAL_CONDITIONS, ValSet, split_names
from metrics import binary_scores
from models import new_sam2unet
from restore import restore

TRAIN_ROOT = os.path.join(ROOT, "data", "COD-TrainDataset")


@torch.inference_mode()
def main():
    device = torch.device("cuda")
    net = new_sam2unet().to(device).eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    q_mean, q_std = (torch.tensor(v, device=device)
                     for v in load_q_stats(os.path.join(ROOT, "results", "q_stats.json")))
    names = split_names(TRAIN_ROOT)[1][:150]
    for tau, median in [(t, m) for t in (0.0, 0.5, 1.0, 1.5) for m in (3, 5)]:
        scores = {}
        for corruption, severity in VAL_CONDITIONS:
            dice = []
            for x, y in DataLoader(ValSet(TRAIN_ROOT, names, corruption, severity), 8, num_workers=2):
                rgb = restore((x.to(device) * std + mean).clamp(0, 1), q_mean, q_std, tau, median)
                prob = net((rgb - mean) / std)[0].sigmoid().cpu().numpy()
                dice += [binary_scores(p[0], g[0] > 0.5)[0] for p, g in zip(prob, y.numpy())]
            scores[f"{corruption}{severity or ''}"] = round(float(np.mean(dice)), 4)
        print(tau, median, round(float(np.mean(list(scores.values()))), 4), scores, flush=True)


if __name__ == "__main__":
    main()
