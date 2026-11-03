"""Turn the benchmark CSVs into the report's tables, with confidence intervals.

    python camo/scripts/analyze.py

Reads per-image scores from camo/results/test_*_images and writes camo/results/tables.
Missing runs are skipped, so this can be used while the evaluation is still going.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
ARMS = ["sinet", "sam2unet", "arm_R", "arm_B", "arm_C", "arm_D", "arm_E"]
PAIRS = [("arm_B", "sam2unet"), ("arm_R", "sam2unet"), ("arm_C", "arm_B"), ("arm_D", "arm_B"),
         ("arm_E", "arm_B"), ("arm_E", "arm_C"), ("arm_E", "arm_D")]
RESAMPLES = 10000
rng = np.random.default_rng(604)


def load(dataset, score="dice_raw"):
    """Return {(model, protocol, condition): per-image Series} for one test set."""
    runs = {}
    for path in glob.glob(os.path.join(RESULTS, f"test_{dataset}_images", "*.csv")):
        tag = os.path.basename(path)[:-4]
        for protocol in ("input", "native"):
            if f"_{protocol}_" in tag:
                model, rest = tag.split(f"_{protocol}_")
                condition = rest[:-2] if rest.endswith("_0") else rest
                frame = pd.read_csv(path, index_col="image")
                if len(frame) == {"cod10k": 2026, "camo": 250, "nc4k": 4121}.get(dataset, len(frame)):
                    runs[(model, protocol, condition)] = frame[score]
    return runs


def bootstrap(diff):
    """Mean and 95% interval of paired per-image differences."""
    diff = np.asarray(diff)
    picks = rng.integers(len(diff), size=(RESAMPLES, len(diff)))
    means = diff[picks].mean(axis=1)
    return diff.mean(), *np.percentile(means, [2.5, 97.5])


def dice_table(runs, protocol):
    """One row per condition and one column per model, plus summary rows."""
    table = {}
    for (model, proto, condition), scores in runs.items():
        if proto == protocol:
            table.setdefault(condition, {})[model] = scores.mean()
    table = pd.DataFrame(table).T
    table = table[[a for a in ARMS if a in table]]
    # Average only conditions every model has, so the means stay comparable.
    corrupted = table.drop(index="clean", errors="ignore").dropna()
    table.loc["mean_corrupted"] = corrupted.mean()
    if "clean" in table.index:
        table.loc["retention"] = table.loc["mean_corrupted"] / table.loc["clean"]
    return table.round(4)


def paired_table(runs, protocol):
    """Bootstrap the gain of each comparison, per condition and over all damage."""
    rows = []
    conditions = sorted({c for m, p, c in runs if p == protocol and c != "clean"})
    for better, base in PAIRS:
        shared = [c for c in conditions if (better, protocol, c) in runs and (base, protocol, c) in runs]
        if not shared:
            continue
        diffs = {c: runs[(better, protocol, c)] - runs[(base, protocol, c)] for c in shared}
        if len(shared) == len(conditions):
            diffs["mean_corrupted"] = pd.concat(diffs.values(), axis=1).mean(axis=1)
        if (better, protocol, "clean") in runs and (base, protocol, "clean") in runs:
            diffs["clean"] = runs[(better, protocol, "clean")] - runs[(base, protocol, "clean")]
        for condition, diff in diffs.items():
            mean, low, high = bootstrap(diff)
            rows.append({"comparison": f"{better} - {base}", "condition": condition, "gain": round(mean, 4),
                         "low": round(low, 4), "high": round(high, 4), "excludes_zero": low > 0 or high < 0})
    return pd.DataFrame(rows)


def habitat_table(runs):
    """Mean corrupted Dice per habitat, using the habitat in COD10K file names."""
    rows = []
    for (model, protocol, condition), scores in runs.items():
        if protocol == "input" and condition != "clean":
            rows.append(scores.groupby(scores.index.str.split("-").str[3]).mean().rename(model))
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, axis=1)
    return frame.T.groupby(level=0).mean().T.round(4)[[a for a in ARMS if a in {r.name for r in rows}]]


def auroc(positive, negative):
    ranks = rankdata(np.concatenate([positive, negative]))
    return (ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))


def presence_table():
    """Can the top-1% score separate camouflaged scenes from scenes without one?"""
    found, absent = load("cod10k", "top1"), load("absent", "top1")
    rows = []
    for (model, protocol, condition), negatives in absent.items():
        positives = found.get((model, protocol, condition))
        if positives is None:
            continue
        background = negatives.index.str.contains("Background")
        threshold = np.percentile(positives, 5)     # keeps 95% detection
        for group, scores in (("backgrounds", negatives[background]), ("visible animals", negatives[~background])):
            rows.append({"model": model, "condition": condition, "negatives": group,
                         "auroc": round(auroc(positives.values, scores.values), 4),
                         "false_alarm_at_95": round(float((scores >= threshold).mean()), 4)})
    return pd.DataFrame(rows)


def moca_table():
    """Compare models on the worst real frames, resampling whole videos."""
    runs = load("moca")
    path = os.path.join(RESULTS, "moca_quality.csv")
    if not runs or not os.path.exists(path):
        return pd.DataFrame()
    quality = pd.read_csv(path, index_col="image")
    rows = []
    for measure, worst_low in (("sharpness", True), ("brightness", True), ("noise", False)):
        ranks = quality[measure].rank(pct=True, ascending=worst_low)
        groups = {"all frames": ranks.index, "worst quartile": ranks[ranks <= 0.25].index,
                  "best quartile": ranks[ranks > 0.75].index}
        for group, frames in groups.items():
            row = {"measure": measure, "frames": group, "count": len(frames)}
            for (model, _, _), scores in runs.items():
                row[model] = round(scores.reindex(frames).mean(), 4)
            for better, base in (("arm_B", "sam2unet"), ("arm_E", "sam2unet"), ("arm_E", "arm_B"), ("arm_E", "arm_C")):
                if (better, "input", "clean") not in runs or (base, "input", "clean") not in runs:
                    continue
                diff = (runs[(better, "input", "clean")] - runs[(base, "input", "clean")]).reindex(frames)
                video = diff.index.str.rsplit("/", n=1).str[0]
                by_video = diff.groupby(video).agg(["sum", "count"]).values
                picks = rng.integers(len(by_video), size=(RESAMPLES, len(by_video)))
                means = by_video[picks, 0].sum(1) / by_video[picks, 1].sum(1)
                low, high = np.percentile(means, [2.5, 97.5])
                row[f"{better} - {base}"] = f"{diff.mean():.4f} [{low:.4f}, {high:.4f}]"
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(TABLES, exist_ok=True)
    outputs = {}
    for dataset in ("cod10k", "camo", "nc4k"):
        runs = load(dataset)
        if not runs:
            continue
        for protocol in ("input", "native"):
            if any(p == protocol for _, p, _ in runs):
                outputs[f"{dataset}_{protocol}_dice"] = dice_table(runs, protocol)
                outputs[f"{dataset}_{protocol}_gains"] = paired_table(runs, protocol)
        if dataset == "cod10k":
            outputs["cod10k_habitat"] = habitat_table(runs)
    outputs["presence"] = presence_table()
    outputs["moca"] = moca_table()
    for name, table in outputs.items():
        if len(table):
            table.to_csv(os.path.join(TABLES, name + ".csv"))
            print(f"\n{name}\n{table.to_string()}")


if __name__ == "__main__":
    main()
