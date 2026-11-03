"""Draw the report figures from the tables that analyze.py and gate_analysis.py wrote.

    python camo/scripts/make_figures.py

Saves PDF and PNG to camo/results/figures. A figure is skipped if its data is missing.
Each arm keeps one colour and one marker shape, so nothing relies on colour alone.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
FIGURES = os.path.join(RESULTS, "figures")

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
ARMS = {"sam2unet": ("A: SAM2-UNet", "#2a78d6", "o"), "arm_B": ("B: augmentation", "#eb6834", "s"),
        "arm_E": ("E: CueGate-SAM", "#1baf7a", "^")}
CUES = {"detail": ("#2a78d6", "o"), "colour": ("#eb6834", "s"), "texture": ("#1baf7a", "^"),
        "structure": ("#6250d6", "D")}
LABELS = {"gaussian_noise": "Noise", "gaussian_blur": "Gaussian blur", "motion_blur": "Motion blur",
          "brightness": "Darkening", "contrast": "Low contrast", "fog": "Fog", "jpeg": "JPEG", "rain": "Rain",
          "drone": "Drone", "night": "Night", "cctv": "CCTV", "clean": "Clean"}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})


def nice(condition):
    """Turn motion_blur_3 into 'Motion blur s3'."""
    name, _, severity = condition.rpartition("_")
    return f"{LABELS[name]} s{severity}" if severity.isdigit() else LABELS[condition]


def save(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{extension}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def table(name):
    path = os.path.join(TABLES, name + ".csv")
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None


def conditions():
    """Dice per condition, hardest first, for the three main arms."""
    data = table("cod10k_input_dice")
    if data is None:
        return
    data = data.drop(index=["mean_corrupted", "retention"]).sort_values("sam2unet")
    fig, ax = plt.subplots(figsize=(3.3, 3.6))
    rows = np.arange(len(data))
    ax.hlines(rows, data[list(ARMS)].min(axis=1), data[list(ARMS)].max(axis=1), color=GRID, linewidth=2)
    for arm, (label, colour, marker) in ARMS.items():
        ax.scatter(data[arm], rows, s=26, color=colour, marker=marker, label=label, zorder=3,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(rows, [nice(c) for c in data.index])
    ax.set_xlabel("Dice at threshold 0.5, COD10K")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1), ncol=3, handletextpad=0, columnspacing=0.7, fontsize=7)
    save(fig, "conditions")


def gate_response():
    """Mean cue weight as each kind of damage gets stronger."""
    path = os.path.join(RESULTS, "gate_response.csv")
    if not os.path.exists(path):
        return
    data = pd.read_csv(path)
    clean = data[data.corruption == "clean"].iloc[0]
    kinds = ["motion_blur", "gaussian_noise", "jpeg", "brightness"]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.8), sharey=True)
    for ax, kind in zip(axes, kinds):
        rows = pd.concat([clean.to_frame().T, data[data.corruption == kind]])
        for cue, (colour, marker) in CUES.items():
            ax.plot(rows.severity.astype(int), rows[cue].astype(float), color=colour, marker=marker,
                    markersize=4, linewidth=1.5, label=cue.capitalize())
        ax.set_title(LABELS[kind], fontsize=8, color=INK)
        ax.set_xlabel("Severity (0 = clean)")
        ax.set_xticks(range(6))
    axes[0].set_ylabel("Mean gate weight")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=4)
    fig.tight_layout()
    save(fig, "gate_response")


def cue_dropout():
    """Dice lost when one cue is switched off at test time."""
    path = os.path.join(RESULTS, "cue_dropout.csv")
    if not os.path.exists(path):
        return
    data = pd.read_csv(path)
    names = [nice(f"{c}_{s}") if s else nice(c) for c, s in zip(data.corruption, data.severity)]
    cues = list(CUES)
    change = np.stack([data[f"no_{c}"] - data.all_cues for c in cues], axis=1)
    limit = max(np.abs(change).max(), 1e-3)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    image = ax.imshow(change, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "loss", ["#d03b3b", "#f0efec", "#2a78d6"]), norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")
    for (i, j), value in np.ndenumerate(change):
        ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=7,
                color="white" if abs(value) > 0.65 * limit else INK)
    ax.set_xticks(range(len(cues)), [c.capitalize() for c in cues])
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Dice change without the cue", shrink=0.9)
    save(fig, "cue_dropout")


def protocol():
    """Retention of the two published detectors under each place the damage can happen."""
    data = table("cod10k_input_dice")
    native = table("cod10k_native_dice")
    if data is None or native is None or "sinet" not in data or "sinet" not in native:
        return
    shared = [c for c in native.index if c in data.index and c not in ("clean", "mean_corrupted", "retention")]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharex=True, sharey=True)
    for ax, model, title in zip(axes, ("sam2unet", "sinet"), ("SAM2-UNet", "SINet-V2")):
        low = data.loc[shared, model] / data.loc["clean", model]
        high = native.loc[shared, model] / native.loc["clean", model]
        rows = np.arange(len(shared))
        ax.hlines(rows, low, high, color=GRID, linewidth=2)
        ax.scatter(low, rows, s=24, color="#eb6834", marker="s", label="Resize, then damage", zorder=3)
        ax.scatter(high, rows, s=24, color="#2a78d6", marker="o", label="Damage, then resize", zorder=3)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("Retention: damaged Dice / clean Dice")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(shared)), [nice(c) for c in shared])
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
    fig.tight_layout()
    save(fig, "protocol")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    conditions()
    gate_response()
    cue_dropout()
    protocol()


if __name__ == "__main__":
    main()
