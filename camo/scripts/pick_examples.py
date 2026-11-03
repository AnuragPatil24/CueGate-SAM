"""Choose the demo images from test results, and draw the qualitative figure.

    python camo/scripts/pick_examples.py

Per habitat, takes the COD10K image that CueGate-SAM gains most on across the three
combined conditions, among images it already segments well when clean. The image
where it loses most is added too, so the demo also shows a failure. One background
and one blurry MoCA-Mask frame complete the set.
"""
import glob
import os
import shutil
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from corruptions import apply_protocol, image_seed
from metrics import binary_scores
from models import LOADERS

RESULTS = os.path.join(ROOT, "results")
SCORES = os.path.join(RESULTS, "test_cod10k_images")
COD10K = os.path.join(ROOT, "data", "COD-TestDataset", "COD10K")
EXAMPLES = os.path.join(ROOT, "demo_examples")
COMBINED = ["drone", "night", "cctv"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dice(model, condition):
    return pd.read_csv(os.path.join(SCORES, f"{model}_input_{condition}.csv"), index_col="image").dice_raw


def choose():
    gain = sum(dice("arm_E", f"{c}_0") - dice("sam2unet", f"{c}_0") for c in COMBINED) / 3
    gain = gain[dice("arm_E", "clean_0") > 0.85]
    habitat = gain.index.str.split("-").str[3]
    picks = {h.lower(): gain[habitat == h].idxmax() for h in sorted(set(habitat) - {"Other"})}
    picks["failure"] = gain.idxmin()
    return picks


def copy(picks):
    os.makedirs(EXAMPLES, exist_ok=True)
    for label, name in picks.items():
        stem = os.path.splitext(name)[0]
        shutil.copy(os.path.join(COD10K, "Imgs", name), os.path.join(EXAMPLES, label + ".jpg"))
        shutil.copy(os.path.join(COD10K, "GT", stem + ".png"), os.path.join(EXAMPLES, label + ".png"))
    backgrounds = sorted(glob.glob(os.path.join(ROOT, "data", "COD10K-v3", "Test", "Image", "*Background*.jpg")))
    shutil.copy(backgrounds[0], os.path.join(EXAMPLES, "no_animal.jpg"))
    quality = pd.read_csv(os.path.join(RESULTS, "moca_quality.csv"), index_col="image")
    frame = quality.sharpness.idxmin()
    video_dir = os.path.join(ROOT, "data", "MoCA_Video", os.path.dirname(frame))
    shutil.copy(os.path.join(video_dir, "Imgs", os.path.basename(frame)), os.path.join(EXAMPLES, "real_video.jpg"))
    shutil.copy(os.path.join(video_dir, "GT", os.path.splitext(os.path.basename(frame))[0] + ".png"),
                os.path.join(EXAMPLES, "real_video.png"))


@torch.inference_mode()
def figure(picks):
    """Rows are images under the night condition; columns are the input, mask and three arms."""
    rows = [(label, name) for label, name in picks.items() if label != "amphibian"]
    models = {"A: SAM2-UNet": "sam2unet", "B: augmentation": "arm_B", "E: CueGate-SAM": "arm_E"}
    samples = []
    for _, name in rows:
        rgb = cv2.cvtColor(cv2.imread(os.path.join(COD10K, "Imgs", name)), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(os.path.join(COD10K, "GT", os.path.splitext(name)[0] + ".png"), 0) > 127
        samples.append((apply_protocol(rgb, "night", 0, image_seed(name, "night", 0), "input"), gt))
    predictions = {}
    for title, key in models.items():
        net = LOADERS[key](torch.device("cuda"))
        for i, (seen, gt) in enumerate(samples):
            x = torch.from_numpy(seen.copy()).permute(2, 0, 1)[None].float() / 255.0
            logits = net(((x - MEAN) / STD).cuda())
            predictions[title, i] = torch.nn.functional.interpolate(
                logits, size=gt.shape, mode="bilinear").sigmoid().squeeze().cpu().numpy()
        del net
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(len(rows), 5, figsize=(6.9, 1.35 * len(rows)))
    for i, ((label, _), (seen, gt)) in enumerate(zip(rows, samples)):
        panels = [("Input, night", cv2.resize(seen, gt.shape[::-1]), None), ("Ground truth", gt, None)]
        panels += [(t, predictions[t, i] >= 0.5, binary_scores(predictions[t, i], gt)[0]) for t in models]
        for ax, (title, image, score) in zip(axes[i], panels):
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=7)
            if score is not None:
                ax.set_xlabel(f"Dice {score:.2f}", fontsize=7)
        axes[i][0].set_ylabel(label.capitalize(), fontsize=7)
    fig.tight_layout(pad=0.3)
    for extension in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, "figures", f"qualitative.{extension}"), dpi=200, bbox_inches="tight")


def main():
    picks = choose()
    print(picks)
    copy(picks)
    figure(picks)
    positives = pd.read_csv(os.path.join(SCORES, "arm_E_input_clean_0.csv")).top1
    print("presence threshold at 95% detection:", round(float(np.percentile(positives, 5)), 4))


if __name__ == "__main__":
    main()
