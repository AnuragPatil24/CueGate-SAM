"""Benchmark detectors on clean and corrupted images under both protocols.

One forward pass gives raw and stretched scores. Finished conditions are skipped
on restart. Per-image scores go to <out>_images/ for paired statistics.

Gate 1 from plan.md:
    python camo/scripts/benchmark.py --models sam2unet sinet --out camo/results/gate1.csv

Quick speed test:
    python camo/scripts/benchmark.py --models sam2unet --subset 100 \
        --corruptions --out camo/results/speed.csv
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from corruptions import COMBINED, apply_protocol, image_seed
from metrics import image_scores
from models import LOADERS

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SCORES = ["dice_raw", "iou_raw", "mae_raw", "dice_stretched", "iou_stretched",
          "mae_stretched", "sm", "peak_prob"]
FIELDS = ["model", "protocol", "corruption", "severity", "images"] + SCORES + ["img_per_s"]


def list_images(data_root, subset):
    """Return sorted (image, mask) paths, optionally a fixed random subset."""
    img_dir, gt_dir = os.path.join(data_root, "Imgs"), os.path.join(data_root, "GT")
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        gt = os.path.join(gt_dir, os.path.splitext(f)[0] + ".png")
        assert os.path.exists(gt), f"missing mask for {f}"
        pairs.append((os.path.join(img_dir, f), gt))
    if subset and subset < len(pairs):
        keep = np.random.default_rng(604).choice(len(pairs), subset, replace=False)
        pairs = [pairs[i] for i in sorted(keep)]
    return pairs


def load_sample(pair, corruption, severity, protocol):
    """Read one pair from disk and apply the damage. Runs in a worker thread."""
    img_path, gt_path = pair
    rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 127
    name = os.path.basename(img_path)
    seed = image_seed(name, corruption, severity)
    return name, apply_protocol(rgb, corruption, severity, seed, protocol), gt


@torch.inference_mode()
def predict(net, rgb, size, device):
    """Follow the official test code: upsample logits, then apply the sigmoid."""
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255.0
    x = ((x - MEAN) / STD).to(device)
    logits = F.interpolate(net(x), size=size, mode="bilinear", align_corners=False)
    return logits.sigmoid().squeeze().float().cpu().numpy()


def run_condition(net, pairs, corruption, severity, protocol, device, workers):
    """Score every image under one condition. Returns per-image rows and speed."""
    rows = []
    start = time.time()
    with ThreadPoolExecutor(workers) as pool:
        jobs = (pool.submit(load_sample, p, corruption, severity, protocol) for p in pairs)
        pending = []
        for job in jobs:
            pending.append(job)
            if len(pending) < workers * 2:
                continue
            name, rgb, gt = pending.pop(0).result()
            rows.append({"image": name, **image_scores(predict(net, rgb, gt.shape, device), gt)})
        for job in pending:
            name, rgb, gt = job.result()
            rows.append({"image": name, **image_scores(predict(net, rgb, gt.shape, device), gt)})
    return rows, len(rows) / (time.time() - start)


def finished_conditions(out):
    """Conditions already in the summary, so a restart can skip them."""
    if not os.path.exists(out):
        return set()
    with open(out, newline="") as f:
        return {(r["model"], r["protocol"], r["corruption"], int(r["severity"]))
                for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=sorted(LOADERS))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "COD-TestDataset", "COD10K"))
    ap.add_argument("--protocols", nargs="+", default=["input", "native"])
    ap.add_argument("--corruptions", nargs="*", default=["motion_blur", "gaussian_noise"])
    ap.add_argument("--severities", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--combined", action="store_true", help="also run drone, night and cctv")
    ap.add_argument("--subset", type=int, default=0, help="0 means all images")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    pairs = list_images(args.data, args.subset)
    conditions = [("clean", 0)] + [(c, s) for c in args.corruptions for s in args.severities]
    if args.combined:
        conditions += [(c, 0) for c in COMBINED]
    os.makedirs(args.out[:-4] + "_images", exist_ok=True)
    done = finished_conditions(args.out)
    new_file = not os.path.exists(args.out)

    with open(args.out, "a", newline="") as f:
        summary = csv.DictWriter(f, FIELDS)
        if new_file:
            summary.writeheader()
        for model in args.models:
            net = LOADERS[model](device)
            for protocol in args.protocols:
                for corruption, severity in conditions:
                    if (model, protocol, corruption, severity) in done:
                        continue
                    rows, speed = run_condition(net, pairs, corruption, severity, protocol,
                                                device, args.workers)
                    tag = f"{model}_{protocol}_{corruption}_{severity}"
                    with open(os.path.join(args.out[:-4] + "_images", tag + ".csv"), "w",
                              newline="") as g:
                        detail = csv.DictWriter(g, ["image"] + SCORES)
                        detail.writeheader()
                        detail.writerows(rows)
                    means = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in SCORES}
                    summary.writerow({"model": model, "protocol": protocol,
                                      "corruption": corruption, "severity": severity,
                                      "images": len(rows), **means,
                                      "img_per_s": round(speed, 1)})
                    f.flush()
                    print(f"{tag:45s} raw={means['dice_raw']:.3f} "
                          f"stretched={means['dice_stretched']:.3f} sm={means['sm']:.3f} "
                          f"peak={means['peak_prob']:.2f} ({speed:.1f} img/s)", flush=True)
            del net
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
