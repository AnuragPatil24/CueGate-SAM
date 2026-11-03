"""Interactive CueGate-SAM demo.

    python camo/app.py

Pick an image, damage it with the sliders, and compare the published SAM2-UNet with
CueGate-SAM. The chart shows which classical cues the quality gate trusted.
"""
import os
import sys

import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from corruptions import brightness, gaussian_blur, gaussian_noise, jpeg, motion_blur, resize
from cuegate import CUE_NAMES, MEAN, STD, build_model
from metrics import binary_scores, presence_score
from models import new_sam2unet

EXAMPLES = os.path.join(ROOT, "demo_examples")
# Top-1% score that keeps 95% detection of clean COD10K test images (scripts/pick_examples.py).
PRESENCE_THRESHOLD = 0.7841
PRESETS = {"Clean": (0, 0, 0, 0, 0), "Drone": (3, 0, 0, 0, 3), "Night": (0, 0, 4, 3, 2), "CCTV": (0, 3, 0, 2, 4)}
CAMERAS = {"Low-resolution camera": "input", "High-resolution camera": "native"}
CUE_COLOURS = {"detail": "#2a78d6", "colour": "#eb6834", "texture": "#1baf7a", "structure": "#6250d6"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = new_sam2unet().to(device).eval()
ours = build_model("E", device, os.path.join(ROOT, "weights", "runs", "E", "best.pth")).eval()
mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
std = torch.tensor(STD, device=device).view(1, 3, 1, 1)


def damage(rgb, motion, soft, dark, noise, compress):
    """Apply the chosen damage in physical order: blur, darkening, noise, then JPEG."""
    rng = np.random.default_rng(604)
    if motion:
        rgb = motion_blur(rgb, motion, rng)
    if soft:
        rgb = gaussian_blur(rgb, soft, rng)
    if dark:
        rgb = brightness(rgb, dark, rng)
    if noise:
        rgb = gaussian_noise(rgb, noise, rng)
    return jpeg(rgb, compress, rng) if compress else rgb


@torch.inference_mode()
def segment(model, rgb, size):
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(device) / 255.0
    logits = model((x - mean) / std)[0]
    prob = torch.nn.functional.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return prob.sigmoid().squeeze().cpu().numpy()


def overlay(rgb, prob):
    """Tint the found object and outline it."""
    view = cv2.resize(rgb, prob.shape[::-1], interpolation=cv2.INTER_LINEAR)
    mask = (prob >= 0.5).astype(np.uint8)
    view[mask > 0] = (0.55 * view[mask > 0] + 0.45 * np.array([235, 104, 52])).astype(np.uint8)
    edges, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(view, edges, -1, (255, 255, 255), max(1, prob.shape[1] // 300))


def gate_chart(weights):
    fig, ax = plt.subplots(figsize=(4.6, 1.8))
    rows = np.arange(len(CUE_NAMES))[::-1]
    ax.barh(rows, weights, color=[CUE_COLOURS[c] for c in CUE_NAMES], height=0.6)
    for row, weight in zip(rows, weights):
        ax.text(weight + 0.01, row, f"{weight:.0%}", va="center", color="#0b0b0b", fontsize=9)
    ax.set_yticks(rows, [c.capitalize() for c in CUE_NAMES])
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of attention from the quality gate", color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def verdict(prob, gt):
    found = presence_score(prob) >= PRESENCE_THRESHOLD
    text = "object found" if found else "no object found"
    return f"{text}, Dice {binary_scores(prob, gt)[0]:.3f}" if gt is not None else text


def run(path, motion, soft, dark, noise, compress, camera):
    if path is None:
        return None, None, None, None, "Choose an image first."
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    levels = (motion, soft, dark, noise, compress)
    seen = resize(damage(rgb, *levels)) if CAMERAS[camera] == "native" else damage(resize(rgb), *levels)
    stem = os.path.splitext(os.path.basename(path))[0]
    gt_path = os.path.join(EXAMPLES, stem + ".png")
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 127 if os.path.exists(gt_path) else None
    size = gt.shape if gt is not None else rgb.shape[:2]

    published = segment(base, seen, size)
    cuegate = segment(ours, seen, size)
    weights = ours.weights[0].cpu().numpy()
    report = (f"**SAM2-UNet:** {verdict(published, gt)}\n\n**CueGate-SAM:** {verdict(cuegate, gt)}\n\n"
              f"**Most trusted cue:** {CUE_NAMES[int(weights.argmax())]}")
    return seen, overlay(seen, published), overlay(seen, cuegate), gate_chart(weights), report


def apply_preset(name):
    return [gr.update(value=v) for v in PRESETS[name]]


with gr.Blocks(title="CueGate-SAM") as demo:
    gr.Markdown("# CueGate-SAM\nFinding camouflaged animals in poor footage. EE604, IIT Kanpur.")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="filepath", label="Image", height=260)
            preset = gr.Radio(list(PRESETS), value="Clean", label="Camera preset")
            camera = gr.Radio(list(CAMERAS), value="Low-resolution camera", label="Where the damage happens")
            sliders = [gr.Slider(0, 5, 0, step=1, label=label) for label in
                       ("Motion blur", "Soft focus", "Darkness", "Sensor noise", "Compression")]
        with gr.Column(scale=2):
            with gr.Row():
                seen = gr.Image(label="What the model sees", interactive=False)
                published = gr.Image(label="SAM2-UNet (published)", interactive=False)
                cuegate = gr.Image(label="CueGate-SAM (ours)", interactive=False)
            with gr.Row():
                chart = gr.Plot(label="Quality gate")
                report = gr.Markdown()
    names = sorted(f for f in os.listdir(EXAMPLES) if f.endswith(".jpg")) if os.path.isdir(EXAMPLES) else []
    if names:
        gr.Examples([[os.path.join(EXAMPLES, f)] for f in names], inputs=image, examples_per_page=8)

    inputs, outputs = [image, *sliders, camera], [seen, published, cuegate, chart, report]
    preset.input(apply_preset, preset, sliders).then(run, inputs, outputs)
    image.change(run, inputs, outputs)
    camera.input(run, inputs, outputs)
    for slider in sliders:
        slider.release(run, inputs, outputs)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
