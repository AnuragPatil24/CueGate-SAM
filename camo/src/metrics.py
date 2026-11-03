"""Per-image scores for a prediction, raw and min-max stretched.

SAM2-UNet's test.py and py_sod_metrics both stretch every prediction to the full
0 to 1 range. That turns a weak, unsure map into a confident-looking mask, so
the raw scores are the honest ones and the stretched scores are only for
comparing with papers (Descion.md, D-008).
"""
import numpy as np
from py_sod_metrics import Smeasure


def stretch(prob):
    """Per-image min-max stretch, as in the official test scripts."""
    return (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)


def binary_scores(prob, gt):
    """Dice, IoU and MAE, thresholding at 0.5. gt is a boolean mask."""
    pred = prob >= 0.5
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    union = total - inter
    dice = 1.0 if total == 0 else 2 * inter / total
    iou = 1.0 if union == 0 else inter / union
    return float(dice), float(iou), float(np.abs(prob - gt).mean())


def smeasure(prob, gt):
    """S-measure of one image. The library stretches internally."""
    metric = Smeasure()
    metric.step(pred=(prob * 255).astype(np.uint8), gt=gt.astype(np.uint8) * 255)
    return float(metric.get_results()["sm"])


def presence_score(prob):
    """Mean of the top 1% of pixels: high when the model is sure of some region."""
    k = max(1, prob.size // 100)
    return float(np.partition(prob.ravel(), -k)[-k:].mean())


def image_scores(prob, gt):
    """All scores for one image, plus how confident the raw prediction was."""
    raw = binary_scores(prob, gt)
    fit = binary_scores(stretch(prob), gt)
    return {
        "dice_raw": raw[0], "iou_raw": raw[1], "mae_raw": raw[2],
        "dice_stretched": fit[0], "iou_stretched": fit[1], "mae_stretched": fit[2],
        "sm": smeasure(prob, gt), "peak_prob": float(prob.max()),
        "top1": presence_score(prob),
    }
