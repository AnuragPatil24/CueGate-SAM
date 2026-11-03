"""Explain arm E, and measure the quality of the real footage.

    python camo/scripts/gate_analysis.py

Writes three files to camo/results:
  gate_response.csv : mean gate weights for each kind and severity of damage
  cue_dropout.csv   : Dice with each cue switched off, per condition
  moca_quality.csv  : the quality vector of every MoCA-Mask frame
"""
import csv
import ctypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from benchmark import ROOT, list_images, load_sample, run_condition

sys.path.insert(0, os.path.join(ROOT, "src"))
from cuegate import CUE_NAMES, build_model
from quality import NAMES, quality_vector

RESULTS = os.path.join(ROOT, "results")
COD10K = os.path.join(ROOT, "data", "COD-TestDataset", "COD10K")
MOCA = os.path.join(ROOT, "data", "MoCA_Video")
SUBSET = 500
BATCH = 32
RESPONSE = [("clean", 0), ("drone", 0), ("night", 0), ("cctv", 0)] + [
    (c, s) for c in ("motion_blur", "gaussian_noise", "jpeg", "brightness") for s in range(1, 6)]
DROPOUT = [("clean", 0), ("motion_blur", 3), ("gaussian_noise", 3), ("jpeg", 3), ("brightness", 3),
           ("drone", 0), ("night", 0), ("cctv", 0)]


def images(items, corruption, severity):
    """Yield batches of damaged images in [0, 1], and their names."""
    with ThreadPoolExecutor(6) as pool:
        samples = list(pool.map(lambda i: load_sample(i, corruption, severity, "input")[:2], items))
    for i in range(0, len(samples), BATCH):
        names, rgbs = zip(*samples[i:i + BATCH])
        yield names, torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float().cuda() / 255.0


def write(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


@torch.inference_mode()
def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    device = torch.device("cuda")
    model = build_model("E", device, os.path.join(ROOT, "weights", "runs", "E", "best.pth")).eval()
    items = list_images(COD10K, SUBSET)

    rows = []
    for corruption, severity in RESPONSE:
        weights = torch.cat([model.gate(quality_vector(x)) for _, x in images(items, corruption, severity)])
        rows.append([corruption, severity, *weights.mean(0).cpu().numpy().round(4)])
    write("gate_response.csv", ["corruption", "severity", *CUE_NAMES], rows)

    rows = []
    for corruption, severity in DROPOUT:
        row = [corruption, severity]
        for drop in [None] + CUE_NAMES:
            def net(x):
                return model(x, drop_cue=drop)[0]
            scored, _ = run_condition(net, items, corruption, severity, "input", device, 6, 8)
            row.append(round(float(np.mean([r["dice_raw"] for r in scored])), 4))
        rows.append(row)
        print(row, flush=True)
    write("cue_dropout.csv", ["corruption", "severity", "all_cues", *[f"no_{c}" for c in CUE_NAMES]], rows)

    rows = []
    frames = list_images(MOCA, 0)
    for start in range(0, len(frames), 256):
        for names, x in images(frames[start:start + 256], "clean", 0):
            rows += [[n, *q] for n, q in zip(names, quality_vector(x).cpu().numpy().round(4))]
    write("moca_quality.csv", ["image", *NAMES], rows)


if __name__ == "__main__":
    main()
