# CueGate-SAM: camouflaged animal detection in poor-quality footage

EE604 Image Processing course project, IIT Kanpur, 2026. Code and results: https://github.com/AnuragPatil24/CueGate-SAM

Camouflage detectors are tested on clean photos, but real cameras such as drones, CCTV and night cameras give blurry, noisy, dark and compressed images. This project does two things:

- It measures how much SAM2-UNet, a strong current camouflage detector, loses on such images.
- It builds **CueGate-SAM**, a small adapter for SAM2-UNet. The adapter adds four classical image cues: detail, colour, texture and structure. A quality gate decides, for each image, how much to trust each cue, using six classical image-quality measures.

## Start here

Read these in order:

1. **[explaination.md](explaination.md)** covers everything: the problem, method, experiments, results, comparison with related work, limitations and likely questions.
2. **[camo/results/figures](camo/results/figures)** holds the five report figures.
3. **[camo/results/tables](camo/results/tables)** holds every result table with 95% confidence intervals.
4. **[Descion.md](Descion.md)** records every decision with its reason and evidence, numbered D-001 to D-039.
5. **[plan.md](plan.md)** is the original plan, with each step's status.

## Main results

| Comparison (mean Dice on damaged images, 95% CI) | COD10K | CAMO | NC4K |
|---|---|---|---|
| Trained on damaged images vs published SAM2-UNet | +0.050 [0.047, 0.052] | +0.029 [0.021, 0.036] | +0.064 [0.061, 0.067] |
| CueGate-SAM vs trained without cues | +0.005 [0.004, 0.006] | +0.006 [0.003, 0.008] | +0.006 [0.005, 0.008] |
| Classical restoration first vs published | −0.002 [−0.003, −0.000] | −0.001 [−0.003, 0.002] | not run |

On clean images there's no accuracy cost: 0.833 against 0.832 Dice on COD10K. Section 7 of `explaination.md` explains these results, including where the method does **not** help.

## Folder layout

```text
explaination.md      full explanation of the project
Descion.md           decision log
plan.md              plan and status
camo/
  app.py             interactive Gradio demo
  src/               quality measures, cues, gate and adapter, damage functions, metrics, data
  scripts/           training, evaluation, analysis and figure scripts
  tests/             tests (run each file directly)
  results/           training logs, test summaries, tables, figures
  demo_examples/     images used by the demo
```

## Not included, because of size

| Item | Size | Where to get it |
|---|---|---|
| Datasets: COD10K, CAMO, NC4K, MoCA-Mask, COD10K non-camouflaged | about 10 GB | Official sources, listed in `plan.md` Section 10 and `Descion.md` D-017 and D-018. Place them under `camo/data/` |
| Published SAM2-UNet camouflage weights | 827 MB | [SAM2-UNet checkpoints](https://drive.google.com/drive/folders/1w2fK8kLhtEmMWZ6G6w9_J17xwgfm3lev), saved as `camo/weights/sam2unet_cod/SAM2UNet-COD.pth` |
| SINet-V2 weights | 202 MB | [SINet-V2](https://github.com/GewelsJI/SINet-V2), saved under `camo/weights/sinetv2/` |
| Our trained weights (B, C, D, E) | 3.3 GB | Ask the author. They go in `camo/weights/runs/<arm>/best.pth` |
| Third-party code | small | Clone [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet) and [SINet-V2](https://github.com/GewelsJI/SINet-V2) into `camo/third_party/` |

Every number and figure can be read without any of these. They're only needed to rerun the code.

## Running it

You need Python 3.11 and an NVIDIA GPU with 8 GB or more. The results were produced on an RTX 4060 Laptop GPU.

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python camo/app.py                         # demo, after the weights are in place
python camo/scripts/train.py --arm E       # train one arm
python camo/scripts/benchmark.py --models sam2unet arm_E --out camo/results/my_run.csv
python camo/scripts/analyze.py             # rebuild the tables
python camo/scripts/make_figures.py        # rebuild the figures
```

## Credits

- **Built on:** [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet) and [SINet-V2](https://github.com/GewelsJI/SINet-V2).
- **Metrics:** computed with [PySODMetrics](https://github.com/lartpang/PySODMetrics).
- **Corruption settings:** follow COD10K-C.
- **Related prior work:** [VFM-CV](https://github.com/Neel-Ronak-Org/VFM-CV) already benchmarked SAM2-UNet under corruption.
