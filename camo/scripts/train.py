"""Train one arm, starting from the published SAM2-UNet camouflage weights.

    python camo/scripts/train.py --arm E --epochs 8 --batch 6
    python camo/scripts/train.py --arm E --speed-test

Each epoch saves last.pth, so a stopped run resumes where it left off. best.pth
is chosen on the validation hold-out only, never on a test set.
"""
import argparse
import csv
import ctypes
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cuegate import build_model, trainable_parameters
from data import VAL_CONDITIONS, TrainSet, ValSet, split_names
from metrics import binary_scores

TRAIN_ROOT = os.path.join(ROOT, "data", "COD-TrainDataset")


def structure_loss(pred, mask):
    """SAM2-UNet's own loss, which weights pixels near object edges more."""
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weight * wbce).sum((2, 3)) / weight.sum((2, 3))
    prob = torch.sigmoid(pred)
    inter = (prob * mask * weight).sum((2, 3))
    union = ((prob + mask) * weight).sum((2, 3))
    return (wbce + 1 - (inter + 1) / (union - inter + 1)).mean()


@torch.inference_mode()
def validate(model, names, device, batch, workers):
    """Mean Dice over the fixed validation conditions."""
    model.eval()
    scores = {}
    for corruption, severity in VAL_CONDITIONS:
        loader = DataLoader(ValSet(TRAIN_ROOT, names, corruption, severity), batch, num_workers=workers)
        dice = []
        for x, y in loader:
            with torch.autocast("cuda"):
                logits = model(x.to(device, non_blocking=True))[0]
            prob = logits.sigmoid().float().cpu().numpy()
            dice += [binary_scores(p[0], g[0] > 0.5)[0] for p, g in zip(prob, y.numpy())]
        scores[f"{corruption}{severity or ''}"] = float(np.mean(dice))
    return float(np.mean(list(scores.values()))), scores


def train_epoch(model, loader, optim, scaler, device, limit=None):
    model.train()
    total, count, start = 0.0, 0, time.time()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            out, out1, out2 = model(x)
            loss = structure_loss(out, y) + structure_loss(out1, y) + structure_loss(out2, y)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        total, count = total + loss.item(), count + 1
        if limit and count >= limit:
            break
    return total / count, count / (time.time() - start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list("BCDE"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--speed-test", action="store_true", help="time 40 iterations and stop")
    args = ap.parse_args()

    # Ask Windows not to sleep while this process runs. It ends when training ends.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    device = torch.device("cuda")
    torch.manual_seed(604)
    out_dir = os.path.join(ROOT, "weights", "runs", args.arm)
    os.makedirs(out_dir, exist_ok=True)
    last_path = os.path.join(out_dir, "last.pth")

    train_names, val_names = split_names(TRAIN_ROOT)
    model = build_model(args.arm, device)
    params = trainable_parameters(model)
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda")
    train_set = TrainSet(TRAIN_ROOT, train_names)
    # Workers restart each epoch on purpose: persistent ones would keep an old
    # copy of train_set.epoch and repeat the same damage on every epoch.
    loader = DataLoader(train_set, args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True)

    if args.speed_test:
        loss, speed = train_epoch(model, loader, optim, scaler, device, limit=40)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        hours = len(loader) * args.epochs / speed / 3600
        print(f"arm {args.arm}: batch {args.batch}, {speed:.2f} it/s, peak {peak:.2f} GB, "
              f"{args.epochs} epochs about {hours:.1f} h without validation, loss {loss:.3f}")
        return

    first, best = 0, -1.0
    if os.path.exists(last_path):
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        first, best = state["epoch"] + 1, state["best"]
        print(f"resumed after epoch {state['epoch']}, best {best:.4f}", flush=True)

    log_path = os.path.join(ROOT, "results", f"train_{args.arm}.csv")

    def log_row(epoch, loss, mean_dice, scores, start):
        row = {"epoch": epoch, "loss": round(loss, 4), "val_dice": round(mean_dice, 4),
               **{k: round(v, 4) for k, v in scores.items()},
               "minutes": round((time.time() - start) / 60, 1)}
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, list(row))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(row, flush=True)

    if first == 0:      # epoch -1 is the untrained model, which equals SAM2-UNet
        start = time.time()
        log_row(-1, 0.0, *validate(model, val_names, device, args.batch, args.workers), start)

    for epoch in range(first, args.epochs):
        train_set.epoch = epoch
        start = time.time()
        loss, speed = train_epoch(model, loader, optim, scaler, device)
        sched.step()
        mean_dice, scores = validate(model, val_names, device, args.batch, args.workers)
        log_row(epoch, loss, mean_dice, scores, start)
        if mean_dice > best:
            best = mean_dice
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best}, last_path)


if __name__ == "__main__":
    main()
