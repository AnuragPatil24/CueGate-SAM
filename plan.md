# CueGate-SAM: Implementation Plan

EE604 course project: camouflaged object detection on poor-quality images.

| Deadline | What |
|---|---|
| **Fri Nov 6, 2026** | Write-up on HelloIITK (CVPR 2022 template). Every member submits, with their own contributions listed |
| **Nov 7-8, 2026** | Presentation and code/demo exam |

Plan written late on Mon Nov 2. That leaves three working days, so every step below has a fallback.

**Revised Nov 2 after verifying every paper.** Two of the three original novelty claims turned out to be false and were withdrawn. The evidence and reasons for every decision are in [Descion.md](Descion.md).

---

## 1. Your idea, with four corrections

Your outline is right: test the best models on poor-quality images, compare against clean images, then build an adapter to fix the drop. These four changes make it defensible.

| # | Your idea | Correction | Why |
|---|---|---|---|
| 1 | Make an adapter so "those best models" work on bad images | Benchmark several models, but **build the adapter for one: SAM2-UNet** | An adapter plugs into a specific encoder. It can't be attached to every model. SINet-V2 is a CNN with no such slots |
| 2 | Testing models on bad images | This step alone is **not novel**, even for SAM2 | COD10K-C did it for SINet-v2, PFNet and ZoomNet. The public VFM-CV repo did it for SAM2-UNet and SAM-Adapter. RUN++ and NUN already test combined degradations. What's new is **questioning how robustness is measured** (Section 2) |
| 3 | Make models "work well" on bad images | You must **beat plain corruption augmentation**, not just the original model | COD10K-C found augmentation was "the main source of robustness". Removing it cost 0.041 Dice, while removing their frequency prior cost 0.018. Your professor will ask this first |
| 4 | Low quality means blur, compression, low light | Also test **combined** degradations, and model low light as darkening **plus noise** | Darkening alone barely hurts: SINet-v2 scores 0.731 vs 0.734 clean. Noise and blur do the damage |

### Corrections to what I told you earlier
- **"Leading adapters rely on high-frequency detail" is only partly true.** It holds for SAM2-Adapter and SAM3-Adapter. SAM2-UNet's adapter uses no hand-made prior at all. Section 2 restates the problem accordingly.
- **We can't use SAM2-Adapter itself.** It runs at 1024 px, was trained on 3-4 A100 GPUs, and I found no published SAM2 camouflage weights for it. Arm C below stands in for it.
- **Wavelets are not really in your lectures.** They appear only in Lec 1 and as Haar filters in Lec 9. This plan uses Laplacian and DoG instead, which were taught.

---

## 2. Project definition

### 2.1 Problem statement

**In one line:** build a camouflage detector that keeps working on poor-quality images, without being told what is wrong with the image, and that can show which visual cues it trusted.

Camouflaged object detection means finding an object that blends into its background, such as an insect on bark or a frog on leaves. The best detectors score well on sharp photographs. Real cameras are different: moving drones blur the image, video links compress it, and dark scenes add sensor noise.

Camouflage makes this harder than ordinary detection, because the object differs from its background only slightly. Our reasoning is that a small loss of image quality can therefore remove much of the difference that was left. Gate 1 measures how true this is.

**Example.** A drone photographs a deer in dry grass. The image is blurred by motion and compressed for the radio link. We want the deer's mask, and a report such as "sharpness is low, so fine detail was trusted less and colour and structure were trusted more". That wording is illustrative: nothing has been trained yet.

### 2.2 Input and output

| | What it is |
|---|---|
| **Input** | One RGB image of any size, resized to 352 x 352 |
| **Not given** | No click, box or text prompt. No clean copy of the image. No label saying which degradation is present |
| **Output 1: mask** | A probability for each pixel, and a binary mask at threshold 0.5 |
| **Output 2: quality report** | Six numbers: sharpness, directionality, noise, brightness, contrast, blockiness |
| **Output 3: trust report** | Four gate weights that sum to 1, one each for detail, colour, texture and structure, plus a one-line explanation |
| **Output 4: presence score** | How likely the image is to contain a camouflaged object at all. See Section 2.7 |

**Not in scope:**
- Naming the object (it gives a mask, not a class).
- Producing a cleaned-up image.
- Video as video. MoCA-Mask frames are used as a real test set, but each frame is handled on its own.
- UAV footage as a target domain. No suitable public dataset was found, see Section 2.7.

**Constraints:** frozen SAM2 encoder, only small parts trained, one 8 GB laptop GPU, three working days. Training uses synthetic corruptions. Real footage is used for testing only.

### 2.3 What we are trying to solve

1. **How much do the best detectors really degrade?** Published answers disagree, so we measure it under two clearly defined protocols.
2. **Why do the published numbers disagree?** With the same model and severity, COD10K-C reports SINet-v2 at 0.538 Dice under motion blur (severity 3), while VFM-CV reports 0.682.
3. **Where the damage is real, can an adapter reduce it beyond plain augmentation?** This is the main engineering question.
4. **Can the model explain itself?** A readable gate should show which cues it relied on, and cue dropout tests whether that explanation is true.

### 2.4 Why we are doing this

- **Wildlife monitoring is the target use:** cameras in the field don't produce studio-quality images, and 99% of COD10K's test images show animals. MoCA-Mask adds real video of wildlife. That is still not field deployment, so the report must not claim the model works on real monitoring systems.
- **Clean benchmark rank may not carry over:** COD10K-C makes this point directly. SINet-v2 beats its small model on clean images, but loses by 6.5 Dice points under motion blur.
- **Failures are silent:** a normal detector gives a mask either way, so the user can't tell whether image quality affected it. Our quality and trust reports are meant to make that visible.
- **Adapters are the active research direction:** SAM2-Adapter, SAM3-Adapter and your professor's SAMWave all adapt foundation models, and SAMWave's authors argue high-frequency detail is crucial. It's natural to ask what happens when poor image quality damages that detail.
- **It fits the course:** Laplacian, DoG, histograms, LBP, structure tensor and Lab colour become working parts of a modern model.

### 2.5 Current accuracy of existing models

Camouflage papers rarely report plain "accuracy", because most pixels are background and a model that predicts nothing could still score highly. These are the metrics used instead:

| Metric | Meaning | Better |
|---|---|---|
| Dice | Overlap between mask and ground truth, 0 to 1 | Higher |
| Sα (S-measure) | Structural similarity to the ground truth | Higher |
| MAE | Average per-pixel error | Lower |
| Retention | Corrupted Dice divided by clean Dice | Closer to 1 |

**Clean images, COD10K test set (2,026 images)**

| Model | Sα | MAE | Dice | Note |
|---|---|---|---|---|
| SINet-V2 | 0.815 | 0.037 | 0.734 | Classic CNN baseline, 352 px |
| ZoomNet | 0.838 | 0.029 | 0.759 | |
| SAM-Adapter | 0.884 | 0.025 | 0.836 | |
| **SAM2-UNet (our base model)** | **0.880** | **0.021** | **0.832** | 352 px, weights available, fits 8 GB |
| NUN (PVT V2) | 0.886 | 0.019 | Not reported | Paper's own figures |
| SAM2-Adapter | 0.899 | 0.018 | Not reported | 1024 px, trained on multiple A100 GPUs |
| SAM3-Adapter | 0.927 | 0.015 | Not reported | Paper's own figures |

Sources: the first four rows come from VFM-CV's `benchmark.csv`. SAM2-UNet's own paper also reports Sα 0.880 and MAE 0.021. The SAM2-Adapter row was read from a badly extracted PDF table and matched across two papers, so check it against the PDF before citing it.

**Poor-quality images: mean over 8 corruptions and 5 severities, COD10K**

| Model | Clean Dice | Mean corrupted Dice | Retention | Source |
|---|---|---|---|---|
| SINet-v2 | 0.734 | 0.644 | 0.877 | COD10K-C |
| SINet-V2 | 0.734 | 0.702 | 0.956 | VFM-CV |
| ZoomNet | 0.759 | 0.696 | 0.918 | VFM-CV |
| PFNet | 0.710 | 0.665 | 0.937 | VFM-CV |
| SAM-Adapter | 0.836 | 0.783 | 0.937 | VFM-CV |
| **SAM2-UNet** | **0.832** | **0.809** | **0.973** | VFM-CV |

Values are rounded from the source files, so they can differ by 0.001 from Section 6. VFM-CV is an unreviewed repo, and its numbers are not yet reproduced for corrupted images.

**Harsher degradations in a published paper: Sα on COD10K, from NUN**

| Model | Clean | Low light | Haze | Combined (low resolution, low light, haze) |
|---|---|---|---|---|
| FEDER (ResNet50) | 0.810 | 0.670 | 0.728 | 0.539 |
| RUN (ResNet50) | 0.827 | 0.684 | 0.731 | 0.548 |
| NUN (ResNet50) | 0.829 | 0.751 | 0.743 | 0.643 |

These models were retrained on degraded data, and they still drop heavily. NUN is the strongest published competitor found for degraded camouflage, but it needs restoration and language-model components.

**Our measurement: Gate 1, 500 fixed COD10K images (Nov 3).** Dice on raw probabilities, with retention in brackets.

| Condition | SAM2-UNet, input | SAM2-UNet, native | SINet-V2, input | SINet-V2, native |
|---|---|---|---|---|
| Clean | 0.819 | 0.819 | 0.728 | 0.728 |
| Motion blur, s=3 | 0.705 (0.86) | 0.795 (0.97) | 0.529 (0.73) | 0.674 (0.93) |
| Motion blur, s=5 | 0.612 (0.75) | 0.764 (0.93) | 0.434 (0.60) | 0.614 (0.84) |
| Gaussian noise, s=3 | 0.656 (0.80) | 0.787 (0.96) | 0.559 (0.77) | 0.684 (0.94) |
| Gaussian noise, s=5 | 0.528 (0.64) | 0.753 (0.92) | 0.471 (0.65) | 0.639 (0.88) |
| Drone | 0.584 (0.71) | 0.773 (0.94) | 0.473 (0.65) | 0.659 (0.91) |
| Night | 0.448 (0.55) | 0.690 (0.84) | 0.406 (0.56) | 0.585 (0.80) |
| CCTV | 0.524 (0.64) | 0.752 (0.92) | 0.430 (0.59) | 0.632 (0.87) |

**Why the two benchmarks disagree: SINet-V2 Dice**

| Condition | Ours, input | COD10K-C | Ours, native | VFM-CV |
|---|---|---|---|---|
| Motion blur, s=3 | 0.529 | 0.538 | 0.674 | 0.682 |
| Motion blur, s=5 | 0.434 | 0.439 | 0.614 | 0.627 |
| Gaussian noise, s=3 | 0.559 | 0.558 | 0.684 | 0.688 |
| Gaussian noise, s=5 | 0.471 | 0.466 | 0.639 | 0.647 |

What this shows:
- **Gate 1 passes.** SAM2-UNet's retention at s=3 is 0.86 for motion blur and 0.80 for noise under the input protocol, below the 0.95 limit. The adapter has real damage to fix.
- **Corruption location explains the disagreement.** Input-protocol results land within 0.009 of COD10K-C, and native results within 0.013 of VFM-CV. This is the project's strongest finding so far. COD10K-C doesn't state its order and trains at 384 px, so write that our input protocol *reproduces* its numbers, not that it used this protocol.
- **Stretching doesn't matter.** Raw and stretched Dice differ by at most 0.0005 across all 32 conditions, so that explanation is dropped.
- **SAM2-UNet is more robust than SINet-V2, but not as robust as it looked.** Under the input protocol, it falls from 0.819 to 0.448 at night.
- **Neither protocol is "the true one".** A fixed pixel blur hurts a low-resolution camera more, because it covers more of the scene. The report must state which camera each number represents.

These figures come from a sample of 500 images, so final tables must use all 2,026.

There is no agreed standard for this task: benchmarks use different corruptions, metrics and protocols, so numbers across tables above can't be compared directly. There's also no accepted target score, so ours is relative: beat arm B on corrupted images, with a confidence interval above zero, and stay within 0.005 Sα of A on clean images.

### 2.6 Why this approach is good

**Method in one line.** Measure image quality with classical formulas, and use those numbers to decide how much to trust four classical cues inside a SAM2 adapter.

| Alternative | Its weakness | What we do instead |
|---|---|---|
| Use the model as it is | Drops under the conditions above | Add adapters that receive cue information |
| Augmentation only | Strong and essential, but gives no explanation of what the model relied on | Keep it as baseline B, and add readable cues and a gate on top |
| Restore the image first | Restoration tends to target how images look, not detection. RUN++ needs a separate network per degradation | Skip the extra network, and test classical restoration as arm R |
| Retrain the whole model | Hiera-L has 214M encoder parameters, too heavy for this laptop in 3 days | Keep the encoder frozen and train small parts |
| Learned quality gate | Harder to read or check | Six classical formulas that can each be tested alone (Gate 2) |

**Strengths:**
- **It needs no degradation label.** Quality is measured from the image, so it suits mixed or unknown conditions.
- **It's safe to train.** Zero-initialised projections mean the model starts exactly equal to SAM2-UNet.
- **It's explainable and testable.** Gate weights can be plotted, and cue dropout checks them against real usefulness.
- **It's feasible.** It reuses published weights and trains only small parts.
- **It's a clean research test.** Arms B, D and E separate the effect of augmentation, cues and the gate.

**Weaknesses, stated honestly:**
- The gain beyond augmentation may be small or zero, and that's a plausible result (Section 4.8).
- Classical cues are computed from the damaged image, so they may be damaged too.
- One seed per arm limits how small a gain can be trusted.
- Controlled severity numbers come from synthetic corruptions. MoCA-Mask is real, but it's YouTube video, not footage from field cameras.

### 2.7 Scope changes from your Nov 3 suggestions

Each suggestion was checked against the data that actually exists. Reasons and proof are in D-028 to D-030 in [Descion.md](Descion.md).

| Suggestion | Decision | What changes |
|---|---|---|
| 1. Detect whether a camouflaged object exists | **Adopted, as a small add-on** | Add a presence score, and test it on clean and degraded images |
| 2. Use real footage instead of synthetic corruption | **Adopted for testing only** | MoCA-Mask becomes the real test set. Synthetic corruption stays for training and severity curves |
| 3. Target one image type | **Adopted as wildlife. UAV rejected** | The project is scoped to wildlife, with results per habitat. No separate adapter is needed |

**Presence check**
- **Data:** 4,934 non-camouflaged COD10K images with empty masks: 2,960 train and 1,974 test. The test images are 784 pure backgrounds and 1,190 clearly visible animals.
- **Step 1, no training:** score each image from its probability map, using the mean of its top 1% of pixels. Run this on A first.
- **Step 2, only if step 1 is weak:** train a small head on frozen encoder features after the adapter is fixed, so the arm comparison isn't affected.
- **Metrics:** AUROC, and false alarm rate at 95% detection. Report backgrounds and visible animals separately, because visible animals are the harder case.
- **Stretch goal:** three answers instead of two: object present, no object, or image too poor to tell, using the quality vector q.
- **Claim:** presence detection itself is not new, because OPC16K and CFCamo do it on clean images. Testing it on degraded images is the only possible new part. Two targeted searches found nothing, so the wording stays "to our knowledge".

**Real footage: MoCA-Mask**
- **Data:** 87 real wildlife videos with 4,691 masked frames, mostly 1280 x 720, in `camo/data/MoCA_Video`. SAM2-UNet wasn't trained on it, so all 87 videos can be test data. Never train on it.
- **Why synthetic stays:** training needs known, controllable damage, and real footage has no severity labels. Without synthetic data there are no severity curves and no Gate 2 test.
- **Method:** measure q on every frame, and sort frames into sharpness and brightness quartiles. Compare A, B and E on the worst quartile.
- **Compare models within a quartile, not across quartiles,** because blurry frames may also be harder scenes for other reasons.
- **Statistics:** bootstrap by video, not by frame, because frames from one video are strongly related.

**Wildlife scope**
- COD10K test images are 2,011 of 2,026 animals, and train images 3,011 of 3,040, so the existing data already fits. A separate wildlife adapter would be nearly the same model.
- Report results by habitat: Aquatic 474, Terrestrial 699, Flying 714, Amphibian 124 test images. Amphibian is small, so treat it as a rough guide.
- CAMO's 1,000 training images stay, as they're part of the published checkpoint's training set.

**Why UAV was rejected:** I found no public dataset of camouflaged wildlife from drones with pixel masks. WildLive's animals aren't camouflaged, and its masks come from SAM2. CPD-UAV shows people rather than wildlife, and I couldn't confirm that it can be downloaded. The `drone` condition stays as a simulation only.

**Warning from Gate 1:** on full-resolution footage, SAM2-UNet loses little except at night, with native retention of 0.92-0.97 for most conditions and 0.84 for `night`. MoCA-Mask is mostly 720p, so it may show little gain from the adapter. That would be a real result, and it must be reported.

### Novelty claims, to defend in the presentation

| # | Claim | Status |
|---|---|---|
| 1 | Measurement study: camouflage robustness scores depend heavily on whether corruption is applied before or after resizing, which reproduces both conflicting benchmarks | **Supported by Gate 1 on 500 images.** No camouflage paper found. A histopathology paper compares corruption order for blur only. VFM-CV tests resolution on its own model only |
| 2 | An adapter that weights multiple classical cues using unlearned, readable quality measures | **Open, narrow claim.** Nothing found doing this. Closest work below |

Withdrawn claims, which must not appear in the report:
- ~~First corruption benchmark that includes SAM2-based detectors~~: VFM-CV already did this.
- ~~Combined degradations are new for camouflage~~: RUN++ and NUN already do this.

| Closest work | How this differs |
|---|---|
| VFM-CV (public repo, no published paper found) | Prompt-free MobileSAM adapter with augmentation and a consistency loss. No cues, no quality gate. It keeps per-image stretching and corrupts before resizing. **Must be cited** |
| NUN | Uses learned quality scores (TOPIQ, Q-Align, MUSIQ) to choose between restored images, plus a vision-language model. Ours uses classical formulas to weight cues inside an adapter, with no restoration network |
| RUN++ | One specialised restoration network per degradation type |
| GaRA-SAM | Learned gate that picks adapter rank. Uses prompts and general datasets, no camouflage set |
| RobustSAM | Prompted, general segmentation. Only cites camouflage, doesn't evaluate it |
| PGE-SAM | Interactive and prompted. No camouflage content |
| RobustCODLite (in COD10K-C) | A small CNN with one fixed frequency prior and no quality-based selection |
| SAMWave (your professor's) | Improves the high-frequency prior. No corruption evaluation |

**Not novel, so don't claim it:** SAM2-UNet, corruption augmentation, combined degradations, adapters on foundation models, or the individual cues.

Keep the wording "to our knowledge". Before submitting, repeat the search once.

---

## 3. Current state

| Item | Status |
|---|---|
| Datasets | Done and verified: `camo/data/COD-TrainDataset` and `camo/data/COD-TestDataset`. All counts match, see D-017 in [Descion.md](Descion.md) |
| MoCA-Mask | Extracted to `camo/data/MoCA_Video`: 87 videos, 4,691 frames, all paired. Not yet run |
| COD10K non-camouflaged | Extracted to `camo/data/COD10K-v3`: 2,960 train and 1,974 test. Sampled masks are empty |
| GPU | RTX 4060 Laptop, 8 GB. CUDA confirmed working in `.venv` |
| RAM / disk | 23.7 GB RAM; about 370 GB free on D: |
| `.venv` | Python 3.11.9, torch 2.11.0+cu128, `pysodmetrics` 1.6.2 installed |
| `SAM2UNet-COD.pth` | In `camo/weights/sam2unet_cod/`. File reads correctly, not yet run |
| SINet-V2 | Repo cloned, weights in `camo/weights/sinetv2/` |
| Git, LaTeX | Local git repo in the project root. Data, weights, third-party code and lecture notes are ignored. No local LaTeX, so use Overleaf |

`SAM2UNet-COD.pth` is the most important file: with it, baseline A needs no training. See D-033 for what was deleted, and how to get it back.

**Both benchmark models now run.** Inference is 7-10 images/s. Training speed and memory are still estimates until step 1.5.

---

## 4. Method

### 4.1 Overview

```text
                     Input image (352 x 352)
                       |                |
             classical formulas    classical filters (no learning)
                       v                v
          Quality vector q (6)     4 cue maps: detail, colour, texture, structure
                       |                |
                  small MLP        small conv encoders
                       v                v
           Gate weights g (4)  --->  weighted sum of cue features F
                                        |
                                 resized per stage, zero-init projection
                                        v
   Frozen SAM2 Hiera blocks, each with Adapter: x + MLP(x) + proj(F)
                                        v
                              U-Net decoder ---> mask
```

### 4.2 Quality vector q (no learning)

| # | Measure | Formula | Detects | Lecture |
|---|---|---|---|---|
| 1 | Sharpness | log variance of Laplacian | Blur | Lec 4 |
| 2 | Directionality | Eigenvalue ratio of the image-wide structure tensor | Motion blur | Lec 9 |
| 3 | Noise | Median local variance in the flattest 10% of patches, using integral images | Sensor noise | Lec 5, 9 |
| 4 | Brightness | Mean of the intensity histogram | Darkness | Lec 3 |
| 5 | Contrast | 95th minus 5th intensity percentile | Fog, low contrast | Lec 3 |
| 6 | Blockiness | Gradient at 8 x 8 block edges divided by gradient elsewhere | JPEG | Lec 8 feature ideas |

Z-score each value using statistics from the training split.

### 4.3 Cue maps

| Cue | Computation | Channels | Lecture |
|---|---|---|---|
| Detail | Absolute Laplacian and DoG of intensity | 2 | Lec 4 |
| Colour | Lab a, b minus their local means | 2 | Lec 7, 12 |
| Texture | LBP sign bits, each averaged over a 7 x 7 box | 8 | Lec 9 |
| Structure | Coherence and smaller eigenvalue of the structure tensor, at Gaussian scale sigma = 2 | 2 | Lec 9 |

Use Lab rather than the HSI taught in class, because hue is unstable at low saturation, which is what dark images have. This also makes a good answer in the viva.

Compute cues with `kornia` on the GPU, inside `torch.no_grad()`.

### 4.4 Gate and adapter

```python
class QualityGate(nn.Module):        # q (6) -> g (4), weights sum to 1
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 4))
    def forward(self, q):
        return self.mlp(q).softmax(dim=-1)

class CueAdapter(nn.Module):         # wraps one existing SAM2-UNet Adapter
    def __init__(self, adapter, ctx, proj):
        super().__init__()
        self.adapter, self.ctx, self.proj = adapter, ctx, proj
    def forward(self, x):            # Hiera blocks use x = (B, H, W, C)
        F = interpolate(self.ctx.features, size=x.shape[1:3], mode="bilinear")
        cue = self.proj(F.permute(0, 2, 3, 1))
        return self.adapter.block(x + self.adapter.prompt_learn(x) + cue)
```

Implementation rules:
- **Don't edit `third_party/`.** Put everything in `camo/src/`, and wrap the blocks after loading the checkpoint.
- **Load order:** build `SAM2UNet()`, load `SAM2UNet-COD.pth` with `strict=True`, then wrap `encoder.blocks`.
- **Passing cues in:** Hiera calls `blk(x)` with only `x`, so store `F` on a shared `ctx` object before calling the encoder.
- **Zero-initialise each `proj`.** The model then starts exactly equal to baseline A, which makes training safe and gives a simple unit test.
- **Share one `proj` per stage,** 4 in total instead of 48. Hiera-L has `stages: [2, 6, 36, 4]`.
- The gate gives one set of weights per image. This keeps it explainable.

### 4.5 Corruptions

Two protocols, because Gate 1 compares them:

- **Input protocol (P1):** resize to 352 x 352, then corrupt, so every model sees the same pixels. This keeps the JPEG 8 x 8 grid aligned with blockiness measure 6. It represents a low-resolution camera.
- **Native protocol (P2):** corrupt at full image size, then let each model resize. This matches VFM-CV, and it represents a high-resolution camera.

Seed each image with `zlib.crc32(name + condition)`, not Python's `hash()`, so every model and rerun sees identical images.

Severity values are copied from COD10K-C:

| Corruption | Severity 1 to 5 |
|---|---|
| Gaussian noise (sigma) | 8, 16, 24, 32, 40 |
| Motion blur (horizontal kernel) | 3, 5, 7, 9, 11 |
| Gaussian blur (radius) | 0.8, 1.2, 1.6, 2.0, 2.5 |
| Brightness factor | 0.85, 0.70, 0.55, 0.40, 0.30 |
| Contrast factor | 0.85, 0.70, 0.55, 0.40, 0.30 |
| Fog (blend with white) | 0.12, 0.20, 0.28, 0.36, 0.44 |
| JPEG quality | 70, 55, 40, 30, 20 |
| Rain | Our own definition: the paper gives no exact values |

Combined camera conditions. These are test conditions only, not a novelty claim. Apply in physical order: blur, then darkening, then noise, then JPEG.

| Name | Recipe |
|---|---|
| Drone | Motion blur k=7, JPEG 40 |
| Night | Brightness 0.40, noise sigma=24, JPEG 55 |
| CCTV | Gaussian blur r=1.6, noise sigma=16, JPEG 30 |

COD10K-C's code isn't released, it trains at 384 px, and it doesn't say whether it corrupts before or after resizing, so expect differences from its numbers.

`camo/src/corruptions.py` and `camo/src/benchmark.py` are untested drafts from a separate session. They use 384 px and one `mixed` condition, so update them to match this section.

### 4.6 Experiment arms

All arms start from `SAM2UNet-COD.pth`, so the only difference is the listed change.

| Arm | Description | Training | Question it answers |
|---|---|---|---|
| **A** | SAM2-UNet as published | None | How much does it degrade? |
| **R** | Rule-based classical restoration, then A | None | Is it enough to clean the image first? |
| **B** | A + corruption augmentation | Yes | **The main comparison** |
| **C** | B + detail cue only | Yes | Stand-in for a single high-frequency prior |
| **D** | B + 4 cues, equal weights | Yes | Do multiple cues help? |
| **E** | B + 4 cues + quality gate | Yes | **Our model.** Does the gate help? |

**R uses course methods only:** unsharp masking for blur (Lec 4), median or adaptive filtering for noise (Lec 4, 5), gamma or histogram equalisation for dark and low-contrast images (Lec 2, 3), all chosen by rules on q.

**Training priority:** B, then E, D, C. If time runs out, cut C first.

**Settings**
- Train on `TrainValDataset` (4,040 images) with a seeded 400-image validation hold-out.
- Each sample is clean with probability 0.2. Otherwise apply one random corruption at severity 1-4, or a combined condition. Keep severity 5 for testing only.
- Loss: the repo's `structure_loss` on all three outputs. Optimiser: AdamW with cosine decay.
- Start with lr 1e-4 and 8 epochs. The trained parts are the adapters, decoder, and in C-E the cue encoders, gate and projections.
- Use mixed precision. Start at batch size 6 and reduce if out of memory.
- Use `num_workers=4`. Windows starts workers slowly, so use `persistent_workers=True` and keep the `__main__` guard.
- Choose the best epoch on validation only.

**Limitation to state in the report:** one seed per arm. If time allows, add a second seed for B and E.

### 4.7 Evaluation

**Test sets**
- COD10K, 2,026 images: main results.
- CAMO, 250 images: secondary.
- NC4K, 4,121 images: a separate set, clean and severity 3 only.
- MoCA-Mask, 4,691 real frames: quartile analysis for A, B and E (Section 2.7).
- COD10K non-camouflaged, 1,974 images: presence check, clean and severity 3.

**Benchmark grid**
- Input protocol: clean, 8 corruptions at severity 3, motion blur, noise and JPEG at severity 5, and 3 combined conditions, giving **15 conditions** on COD10K for A, R, B, C, D, E and SINet-V2. The original 28-condition grid would take about 45 hours (D-035).
- Native protocol: 10 conditions for A, SINet-V2, B and E.
- `camo/scripts/evaluate_all.ps1` runs everything, most important first. It's safe to rerun.
- BiRefNet-COD was dropped and its files deleted (D-033).

**Metrics**
- **Score two versions of every prediction: raw probabilities, and min-max stretched.** SAM2-UNet's `test.py` line 41 stretches every map, so stretched scores are only for comparing with papers. Dice, IoU and MAE on raw probabilities are the honest numbers.
- Clean: Sα, Fβw, Eφ, MAE from `pysodmetrics`, plus Dice.
- Corrupted: Dice, Sα, Fβw, MAE. Eφ is slow, so skip it here.
- Summary: mean corrupted Dice, and **retention** = corrupted Dice / clean Dice, matching COD10K-C.
- Save per-image Dice to CSV. Report Dice at threshold 0.5, and state that threshold because COD10K-C doesn't.

**Statistics.** Use a paired bootstrap on per-image Dice differences (10,000 resamples, 95% CI) for E vs D, E vs B, and D vs B. Report CIs, because small gains are likely.

**Extra analyses**
1. **Gate response:** plot g against severity for each corruption. This is the main explanation figure.
2. **Cue dropout:** in E, zero one cue at test time, and show the Dice change as a heatmap of cue vs corruption. Check whether high gate weight really means high usefulness. If they disagree, report it.
3. **Held-out corruptions (stretch goal):** train B and E without motion blur and JPEG, then test on those.

**Test-set rule:** tune everything on validation. Run the test set only on final checkpoints.

### 4.8 Realistic expectations

The gain beyond augmentation will probably be small. COD10K-C found about 0.02 Dice from a frequency prior, compared with 0.04 from augmentation. VFM-CV's numbers are the bigger warning: SAM2-UNet goes from 0.831 clean to about 0.81 averaged over corruptions, which leaves about 0.02 Dice to win back under that protocol. So the adapter is only worth building for conditions where Gate 1 shows real damage.

| Outcome | What it means | How to present it |
|---|---|---|
| E > D > B with CIs above zero | Full success | Quality-gated cues add robustness beyond augmentation |
| D > B, but E is about equal to D | Cues help, the gate doesn't | Honest result. The gate still gives explanation at no accuracy cost |
| E, D and B about equal | Augmentation is enough for SAM2 | Benchmark and combined conditions remain new. The gate is a readable diagnostic |
| A barely drops under every protocol | Premise fails | See Gate 1 |

Clean-image cost target: Sα no more than 0.005 below A.

---

## 5. Repository layout

```text
Project/
  plan.md
  Notes/
  camo/
    src/
      corruptions.py   # all corruptions, severities, combined conditions
      quality.py       # 6 quality measures
      cues.py          # 4 cue maps
      cuegate.py       # QualityGate, CueAdapter, CueGateSAM
      data.py          # train/val split, augmentation, test loader
      restore.py       # classical restoration for arm R
      metrics.py       # pysodmetrics wrapper, per-image CSV
    scripts/
      prepare_data.py  benchmark.py  train.py
      analyze.py       make_figures.py
      app.py           # Gradio demo
    tests/             # smoke tests
    results/           # CSVs, figures (commit these)
    data/  weights/    # never commit these
    third_party/
```

Run `git init` at the start. Add `camo/data/`, `camo/weights/`, `.venv/` and checkpoints to `.gitignore`. Keep third-party licence files.

---

## 6. Day-by-day plan

### Tonight (Mon Nov 2)
- [x] Download and verify the datasets and model weights.
- [x] Confirm the topic with your professor. Approved.

### Tue Nov 3: prove the premise

| # | Task | Done when |
|---|---|---|
| 1.1 | `git init`, `.gitignore` | First commit made |
| 1.2 | Confirm GPU in `.venv`; run the `SAM2UNet.py` smoke test | `cuda True`; outputs have the expected shapes |
| 1.3 | Fix version issues. Repo pins torch 2.4, you have 2.11 | Forward pass works |
| 1.4 | Dataset counts | Done Nov 2 |
| 1.5 | Measure inference img/s, training it/s, and peak VRAM at batch 6 | Numbers recorded; epochs and grid finalised |
| 1.6 | Reproduce clean A on COD10K | Close to the repo's `Pre-computed maps.zip` |
| 1.7 | Write `corruptions.py` and save a visual grid | Every corruption looks right |
| 1.8 | Run SINet-V2 clean, noise and motion blur under P1 and P2 | One protocol lands near COD10K-C and the other near VFM-CV, as in the table below |
| 1.9 | **Run the 2 x 2 test on A and SINet-V2:** P1 and P2, raw and stretched, for motion blur and noise at severities 3 and 5 | **Gate 1 decided** |
| 1.10 | Attend office hours, 15:30-16:30, ACES 302, with the Gate 1 result if ready | Topic confirmed |

SINet-v2 reference, Dice at severity 1 / 3 / 5. Both sources give 0.734 clean:

| Condition | COD10K-C | VFM-CV |
|---|---|---|
| Gaussian noise | 0.679 / 0.558 / 0.466 | 0.726 / 0.687 / 0.647 |
| Motion blur | 0.680 / 0.538 / 0.439 | 0.729 / 0.681 / 0.626 |

SAM2-UNet reference from VFM-CV: 0.831 clean, noise 0.827 / 0.802 / 0.768, motion blur 0.829 / 0.808 / 0.781. These come from an unpublished repo and use its protocol, so treat them as a guide until reproduced.

### Wed Nov 4: build it

| # | Task | Done when |
|---|---|---|
| 2.1 | Start the benchmark for SINet-V2 and A | Running in the background |
| 2.2 | `quality.py`; test against severity on validation images | **Gate 2 passes** |
| 2.3 | `cues.py`; save a figure of cues under each corruption | Cues look sensible |
| 2.4 | `cuegate.py` | Zero-init test passes: output equals A at start |
| 2.5 | `train.py` for arms B-E; run 50 iterations of each | Loss falls, no memory error |
| 2.6 | `restore.py`; evaluate arm R | Results saved |
| 2.6a | Run A on MoCA-Mask and on the non-camouflaged images, with no training | Real-footage baseline and presence score saved |
| 2.7 | **Start overnight training: B, E, D, then C** | Saves each epoch, and can resume |

### Thu Nov 5: results and demo

| # | Task | Done when |
|---|---|---|
| 3.1 | Check runs on validation; choose checkpoints | Final checkpoints fixed |
| 3.2 | Test all arms on the grid, CAMO and NC4K | CSVs saved |
| 3.2a | Test B and E on MoCA-Mask quartiles and the presence check | CSVs saved |
| 3.3 | `analyze.py`: tables, CIs, retention, habitat breakdown | Tables generated |
| 3.4 | `make_figures.py`, gate response and cue dropout | Figures generated |
| 3.5 | Gradio demo | Works from a fresh start |
| 3.6 | Photograph 10 real test scenes | Added to demo examples |
| 3.7 | Fill the Overleaf report with real numbers | Full draft |

### Fri Nov 6: submit
- [ ] Rerun the novelty search; finalise the wording.
- [ ] Finish the report and proofread every number against the CSVs.
- [ ] Add code appendix, GitHub link, datasets and repo links, individual contributions.
- [ ] Push to GitHub and check the repo from a fresh clone.
- [ ] **Submit by evening.** Confirm the exact cut-off on HelloIITK. Every member must submit.

### Sat Nov 7: present
- [ ] Build slides and rehearse, including the viva questions in Section 9.
- [ ] Record a backup demo video.
- [ ] Your slot could be Nov 7 itself, so try to finish slides on Nov 6 night if possible.

---

## 7. Decision gates

| Gate | When | Test | If it passes | If it fails |
|---|---|---|---|---|
| **1. Premise** | Nov 3 | Raw-probability retention of A, under P1 or P2, is below 0.95 on motion blur or noise at s=3 | Build CueGate and target the conditions that really hurt | Check s=5 and combined conditions. If A still holds up, the measurement study becomes the main result: why SAM2-UNet is robust, and how much of that is real. Tell your professor the same day |
| **2. Quality meter** | Nov 4 | Relevant measure has Spearman \|rho\| of at least 0.7 with severity | Build the gate | Replace that measure. If time is short, drop it. The gate works with fewer inputs |
| **3. Memory and speed** | Nov 3 | One training arm fits in about 2.5 hours | Keep 8 epochs | Reduce epochs, batch size, or train a smaller grid of arms |

**Status, Nov 3:** your professor approved the topic. All three gates **passed**:
- **Gate 1:** see Section 2.5.
- **Gate 2:** every measure tracked its damage on 100 validation images, with |rho| of 0.84 to 1.00.
- **Gate 3:** arm E uses 5.3 GB at batch 6 and 1.0 it/s, so 8 epochs take about 1.7 hours with validation.

Training finished at 05:53 on Nov 3 (D-034). The final test evaluation started at 06:14 and should take about 12.5 hours (D-035). Rerun `camo/scripts/evaluate_all.ps1` to resume if it stops. This is ahead of the day plan in Section 6.

### If you fall behind, cut in this order
1. Held-out corruption test
2. Three-answer presence output, then the trained presence head
3. Arm C
4. NC4K, keeping CAMO and MoCA-Mask
5. Severity levels, keeping s=3 and combined conditions
6. Arm D last, because E vs B alone can't show whether the cues or the gate caused any gain

**Never cut:** the 2 x 2 measurement test, A vs B vs E, the gate figure, the demo.

---

## 8. Deliverables, mapped to the rubric

### Write-up (5%)
| Required | Where it comes from |
|---|---|
| Abstract: problem, novelty, results | Written last, with real numbers |
| Introduction: why it matters, novelty | Sections 1 and 2 |
| Proposed method with overview figure | Section 4.1, redrawn properly |
| Experiments: tables and figures | Items below |
| Appendix: entire code, GitHub link | `\lstinputlisting` of `camo/src` and `camo/scripts`. Link third-party code |
| Dataset and repository links | Section 10 |

**Tables:** clean results; per-corruption Dice for all models; arms A, R, B-E with retention and CIs; NC4K and CAMO.

**Figures:** overview; corruption samples; Dice vs severity curves; gate response; cue dropout heatmap; qualitative rows of input, ground truth, A, B and E, including failures.

### Presentation (5%)
Rough outline of 12 slides: problem and constraints; why it matters; literature; gap and novelty; method overview; quality meter; cues and gate; datasets and setup; benchmark result; arm comparison; explanation figures; limits and conclusion.

### Code/demo (5%)
The Gradio app (`camo/scripts/app.py`) needs:
- Upload an image or pick an example.
- Sliders for blur, noise, darkness and JPEG, plus Drone, Night and CCTV presets.
- A and E masks side by side.
- Quality readings, a live gate weight bar chart, and a one-line explanation.
- Pre-cached outputs and a video, in case the GPU fails on the day.

Comment all your code, with file-level and function-level docstrings, and be ready to explain each module.

### Contribution log
Keep one line per person per day from day one. You need it for the report. If you have teammates, a natural split is: (1) corruptions and benchmark, (2) quality, cues and adapter, (3) training and analysis, (4) demo, (5) report and slides.

---

## 9. Risks and viva questions

### Risks

| Risk | Plan |
|---|---|
| Download links fail, or Drive quota runs out | Start tonight. Use `gdown`, or download manually in a browser |
| Out of memory at batch 6 | Mixed precision, reduce batch size, keep frozen layers without gradients where possible |
| torch 2.11 breaks vendored SAM2 code | Create a second venv with torch 2.4, as the repo pins |
| Laptop overheats or sleeps overnight | Mains power, disable sleep, save each epoch, resume support |
| Blockiness fails on real photos, because the JPEG grid shifts on resize | Measure on the original image in the demo. Say this is a known limit |
| Results may not hold on real footage | All quantitative data is synthetic. Real photos are qualitative only, and the report must say so |
| Plagiarism penalty: zero for the component | Cite every paper and repo. Keep your own code clearly separate |

### Questions to prepare for
- **"Hasn't this been benchmarked already?"** Yes, by COD10K-C and VFM-CV. Say so first, then show why their numbers disagree.
- **"How is this different from NUN?"** NUN restores the image using learned quality scores and a language model. This adapter uses classical formulas to weight cues, with no restoration network.
- **"Isn't this just data augmentation?"** Show arm B vs E, with CIs.
- **"Why not restore the image first?"** Show arm R.
- **"Why hand-crafted quality measures instead of learned ones?"** They're readable, need no labels, and can be verified on their own. Gate 2 shows they track severity.
- **"Is your gate explanation true?"** Show cue dropout vs gate weight.
- **"How is this different from SAMWave or GaRA-SAM?"** See Section 2.
- **"Why 352 px?"** It's SAM2-UNet's native size, and it fits 8 GB.
- **"What's the weakest part of your work?"** Synthetic corruptions and one seed per arm.

---

## 10. Sources and resources

The verification result for each source is in [Descion.md](Descion.md).

**Papers**
- [COD10K-C](https://arxiv.org/abs/2606.02603)
- [NUN](https://arxiv.org/abs/2511.18164)
- [RUN++](https://arxiv.org/abs/2508.15027)
- [SAM2-UNet](https://arxiv.org/abs/2408.08870)
- [SAM2-Adapter](https://arxiv.org/abs/2408.04579)
- [SAM3-Adapter](https://arxiv.org/abs/2511.19425)
- [GaRA-SAM](https://arxiv.org/abs/2506.02882)
- [RobustSAM](https://arxiv.org/abs/2406.09627)
- [PGE-SAM](https://arxiv.org/abs/2606.30477)
- [SAMWave](https://arxiv.org/abs/2507.20186)
- [Blur order in histopathology](https://arxiv.org/abs/2603.04081)

**Prior work to cite**
- [VFM-CV](https://github.com/Neel-Ronak-Org/VFM-CV): public repo, no published paper found. Cite it as a code repository with its access date.

**Code and data**
- [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet) and [its checkpoints](https://drive.google.com/drive/folders/1w2fK8kLhtEmMWZ6G6w9_J17xwgfm3lev)
- [SINet-V2](https://github.com/GewelsJI/SINet-V2)
- [BiRefNet-COD](https://huggingface.co/ZhengPeng7/BiRefNet-COD)
- [PySODMetrics](https://github.com/lartpang/PySODMetrics)
- [COD dataset mirror](https://huggingface.co/datasets/cod-tdq/cod): licence is marked "other", so cite the original datasets
