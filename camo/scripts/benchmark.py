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
import ctypes
import glob
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
          "mae_stretched", "sm", "peak_prob", "top1"]
FOLDERS = {"Imgs": "GT", "Image": "GT_Object"}     # image folder: mask folder
FIELDS = ["model", "protocol", "corruption", "severity", "images"] + SCORES + ["img_per_s"]


def list_images(data_root, subset):
    """Return sorted (name, image, mask) triples, optionally a fixed random subset.

    Finds every image folder under data_root, so per-video datasets work too.
    Their names keep the video, because frame names repeat between videos.
    """
    items = []
    for img_folder, gt_folder in FOLDERS.items():
        for img_dir in sorted(glob.glob(os.path.join(data_root, "**", img_folder), recursive=True)):
            parent = os.path.dirname(img_dir)
            video = os.path.relpath(parent, data_root).replace(os.sep, "/")
            for f in sorted(os.listdir(img_dir)):
                gt = os.path.join(parent, gt_folder, os.path.splitext(f)[0] + ".png")
                assert os.path.exists(gt), f"missing mask for {f}"
                name = f if video == "." else f"{video}/{f}"
                items.append((name, os.path.join(img_dir, f), gt))
    if subset and subset < len(items):
        keep = np.random.default_rng(604).choice(len(items), subset, replace=False)
        items = [items[i] for i in sorted(keep)]
    return items


def load_sample(item, corruption, severity, protocol):
    """Read one image from disk and apply the damage. Runs in a worker thread."""
    name, img_path, gt_path = item
    rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 127
    seed = image_seed(name, corruption, severity)
    return name, apply_protocol(rgb, corruption, severity, seed, protocol), gt


@torch.inference_mode()
def predict(net, rgbs, sizes, device):
    """Follow the official test code: upsample logits, then apply the sigmoid."""
    x = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float() / 255.0
    logits = net(((x - MEAN) / STD).to(device))
    return [F.interpolate(l[None], size=s, mode="bilinear", align_corners=False)
            .sigmoid().squeeze().float().cpu().numpy() for l, s in zip(logits, sizes)]


def run_condition(net, items, corruption, severity, protocol, device, workers, batch):
    """Score every image under one condition. Returns per-image rows and speed."""
    rows = []
    start = time.time()
    with ThreadPoolExecutor(workers) as pool:
        def flush(samples):
            names, rgbs, gts = zip(*samples)
            probs = predict(net, rgbs, [g.shape for g in gts], device)
            rows.extend({"image": n, **s} for n, s in zip(names, pool.map(image_scores, probs, gts)))

        jobs = (pool.submit(load_sample, i, corruption, severity, protocol) for i in items)
        pending = []
        for job in jobs:
            pending.append(job)
            if len(pending) >= batch * 2:
                flush([j.result() for j in pending[:batch]])
                del pending[:batch]
        while pending:
            flush([j.result() for j in pending[:batch]])
            del pending[:batch]
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
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Ask Windows not to sleep while this process runs. It ends when the run ends.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    device = torch.device("cuda")
    items = list_images(args.data, args.subset)
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
                    rows, speed = run_condition(net, items, corruption, severity, protocol,
                                                device, args.workers, args.batch)
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
