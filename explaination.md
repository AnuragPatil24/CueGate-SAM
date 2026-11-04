# CueGate-SAM: what we did, how, and why

This file explains the whole project from start to finish. It covers what we built, how it works, what we found, and how it differs from Prof. Koteswar Rao Jerripothula's work. Read it before writing the report or facing questions in the viva.

The other project files:
- [plan.md](plan.md) is the plan.
- [Descion.md](Descion.md) holds every decision with its proof. Its entries are referred to below as D-xxx.
- `camo/` holds the code and results.

---

## 1. The project in one paragraph

Camouflage detectors are tested on clean, high-quality photos. Real cameras, such as drones, trail cameras, CCTV and night footage, give blurry, noisy, dark and compressed images. We measured how much the best current detector, SAM2-UNet, loses on such images. We found the answer depends on *where* the damage happens, which explains why two published benchmarks disagree. We then built **CueGate-SAM**, a small adapter for the frozen SAM2 encoder. It feeds four hand-crafted image cues from the course (detail, colour, texture, structure) into the model. A **quality gate** decides, for each image, how much to trust each cue. The gate is driven by six classical, unlearned image-quality measures. Seven model versions were tested on COD10K, CAMO, NC4K, real wildlife video (MoCA-Mask) and images with no animal at all. Every comparison has bootstrap confidence intervals.

**Main findings:**
1. Training the adapter on damaged images recovers most of the lost accuracy: +0.050 Dice on COD10K.
2. The classical cues add a small but statistically significant extra gain on all three synthetic test sets, about +0.005.
3. The gate beats the ungated versions only on CAMO, so its main value is **explaining** which cue the model relies on under each kind of damage.
4. Cleaning images with classical filters first does not help.

---

## 2. The problem

### Input and output
- **Input:** one RGB image that may be degraded. Its quality is unknown and no damage label is given.
- **Output:** a per-pixel mask of the camouflaged animal, plus a readable report of the image's quality and which cues the model trusted. A presence score also says whether any animal seems to be there.

### Why it matters
- Camouflage detection is used for wildlife monitoring, search and rescue, pest detection and medical imaging. These rarely come with studio-quality images.
- Detectors fail **silently**: they still output a mask, and the user can't tell whether poor image quality ruined it.
- A detector's ranking on clean images doesn't carry over to damaged ones. COD10K-C showed SINet-V2 losing its lead under motion blur.

### What was already known (verified, D-001 to D-022)
| Work | What it did | What it left open |
|---|---|---|
| **SAM2-UNet** | Frozen SAM2 Hiera-L encoder with small adapters and a U-Net decoder. A top camouflage detector, with public weights | Only tested on clean images |
| **COD10K-C** | Corruption benchmark for camouflage detection | Its SAM2-UNet numbers disagree with VFM-CV's |
| **VFM-CV** (public repo) | Already benchmarks SAM2-UNet under corruption | Damages the image *before* resizing. No classical cues, no quality gate |
| **NUN, RUN++** | Already use combined degradations and restoration networks | Use learned quality scores or separate restoration models, not cues inside an adapter |

Because of this, two claims I made early on were **withdrawn**: "the first SAM2 corruption benchmark" and "combined degradations are new". They must never appear in the report (D-021). VFM-CV must be cited.

---

## 3. What we did, in order

| When | Step | Result |
|---|---|---|
| Nov 2 | Checked every paper and novelty claim, downloaded data and weights | Two claims withdrawn. Data counts verified |
| Nov 3, 00:59 | **Gate 1:** benchmarked SAM2-UNet and SINet-V2 on 500 COD10K images under damage | Premise confirmed. Found the protocol effect (Section 4) |
| Nov 3, ~01:10 | **Gate 2:** checked that each quality measure tracks its damage | Spearman \|rho\| 0.84 to 1.00 on 100 images |
| Nov 3, ~01:10 | **Gate 3:** checked memory and speed | 5.3 GB, about 1.7 hours per version, so it fits the 8 GB laptop GPU |
| Nov 3, 01:12 to 05:53 | Trained four versions (B, C, D, E) | All improved on validation |
| Nov 3, ~06:00 | Tuned the restoration baseline R on validation | Chose the gentlest settings (D-036) |
| Nov 3, 06:14 to 19:45 | Final test evaluation: 7 models, 5 test sets, 2 protocols | 259 runs, all complete |
| Nov 3, evening | Tables, bootstrap CIs, figures, demo examples | Everything in `camo/results/` |

Each step had a pass/fail **gate** set in advance, so a failed idea would have been caught early rather than at the deadline.

---

## 4. Finding 1: where the damage happens changes the answer

### The puzzle
COD10K-C and VFM-CV report very different robustness for the same model. COD10K-C shows large losses, while VFM-CV shows SAM2-UNet losing almost nothing.

### What we tested
A model sees a 352 × 352 image, so damage can be applied at two points:
- **Input protocol:** resize to 352 first, then damage. This is like a **low-resolution camera**, where the damage is at the scale the model sees.
- **Native protocol:** damage the full-size photo, then resize. This is like a **high-resolution camera**. Shrinking the image averages pixels together, which removes much of the noise and blur. This is the low-pass filtering and sampling taught in class.

### Result, all 2,026 COD10K test images
Retention is damaged Dice divided by clean Dice. It is averaged over the 9 conditions both protocols share: motion blur, noise and JPEG at severities 3 and 5, plus drone, night and CCTV.

| Model | Input protocol | Native protocol |
|---|---|---|
| SAM2-UNet | 0.74 | **0.96** |
| SINet-V2 | 0.69 | 0.92 |

The input protocol reproduces COD10K-C's numbers to within 0.009, and the native protocol reproduces VFM-CV's to within 0.013 (Gate 1, D-031). **So the two benchmarks aren't contradicting each other; they're measuring different cameras.** At night, even the native protocol loses 15%, so darkness hurts at any resolution.

We also tested whether min-max stretching of predictions explained the gap. It doesn't: it changes Dice by at most 0.0005.

Figure: `camo/results/figures/protocol.pdf`.

---

## 5. Finding 2: the method, CueGate-SAM

### 5.1 Overview

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

### 5.2 Quality vector q: six measures, nothing learned (`camo/src/quality.py`)
| Measure | How it's computed | Damage it detects | Course topic |
|---|---|---|---|
| Sharpness | Log variance of the Laplacian | Blur | Lec 4, sharpening and the Laplacian |
| Directionality | Eigenvalue ratio of the image-wide structure tensor | Motion blur, which smooths in only one direction | Lec 9, structure tensor |
| Noise | Median variance in the flattest 10% of 8 × 8 patches | Sensor noise | Lec 5, 9 |
| Brightness | Mean of the intensity histogram | Darkness | Lec 3, histograms |
| Contrast | 95th minus 5th intensity percentile | Fog, low contrast | Lec 3 |
| Blockiness | Gradient at 8 × 8 block edges divided by gradient elsewhere | JPEG compression | Lec 8, JPEG's 8 × 8 DCT blocks |

Each value is z-scored using statistics from damaged training images, stored in `camo/results/q_stats.json`. Gate 2 confirmed that every measure tracks its damage.

### 5.3 Four cue maps (`camo/src/cues.py`)
| Cue | Computation | Channels | Why it helps camouflage | Course topic |
|---|---|---|---|---|
| Detail | Absolute Laplacian and Difference of Gaussians | 2 | Edges and fine detail at the animal's outline | Lec 4 |
| Colour | Lab a and b, minus their local means | 2 | Small colour mismatches between animal and background | Lec 7, 12 |
| Texture | 8 LBP sign bits, each averaged over a 7 × 7 box | 8 | Texture that differs slightly from the surroundings | Lec 9 |
| Structure | Structure-tensor coherence and smaller eigenvalue | 2 | Shape and orientation patterns | Lec 9 |

**Why Lab and not HSI:** hue is unstable at low saturation, and dark images have low saturation.
**Why Laplacian and DoG and not wavelets:** they're taught in the course, and wavelets would sit too close to SAMWave (D-015, Section 8).

### 5.4 Gate and adapter (`camo/src/cuegate.py`)
- **Gate:** a tiny MLP (6 → 32 → 4) with a softmax. It gives one weight per cue, per image, and the weights sum to 1, so they can be read as "the model trusted colour 45%".
- **Cue encoders:** small convolutional networks turn each cue map into a 32-channel feature map.
- **Injection:** the weighted sum of cue features is resized to each encoder stage, then added inside SAM2-UNet's existing adapters: `x + prompt_learn(x) + proj(F)`.
- **Zero initialisation:** every projection starts at zero, so at step 0 the model is *exactly* the published SAM2-UNet. Training can only move away from that starting point if the cues help. A unit test checks this for every version.
- **What trains:** the adapters, decoder, cue encoders, gate and projections, about **4.5M of 216.7M parameters (2%)**. The SAM2 encoder stays frozen.

### 5.5 Training (`camo/scripts/train.py`)
- 3,640 training images from COD10K and CAMO train. 400 are held out for validation, chosen with seed 604.
- 20% of training images are clean. The rest get one random damage at severity 1 to 4, or a combined condition. Severity 5 is never used in training.
- 8 epochs, AdamW, learning rate 1e-4 with cosine decay, batch 6, mixed precision. The loss is SAM2-UNet's own structure loss on all three outputs.
- The best epoch was chosen on validation only. Test images were never used to tune anything.

### 5.6 The seven versions ("arms") we compared
| Arm | What it is | Question it answers |
|---|---|---|
| SINet-V2 | Older, widely used detector | How does a pre-SAM model compare? |
| **A** | SAM2-UNet as published | How much does damage hurt? |
| **R** | Rule-based classical restoration (median filter, gamma, unsharp masking), then A | Is it enough to clean the image first? |
| **B** | A trained on damaged images | How much does training alone give? **The main baseline** |
| **C** | B plus the detail cue only | Does one high-frequency cue help? |
| **D** | B plus all four cues with equal weights | Do several cues help? |
| **E** | B plus four cues plus the quality gate | **Our model.** Does the gate help? |

Every trained arm starts from the same checkpoint with the same settings, so each comparison isolates one change.

---

## 6. How we tested

### Test data
| Set | Images | Use |
|---|---|---|
| COD10K test | 2,026 | Main results, plus results per habitat |
| CAMO test | 250 | Second test set |
| NC4K | 4,121 | Third test set: clean plus 3 combined conditions |
| MoCA-Mask | 4,691 frames from 87 real videos | **Real footage**, not synthetic damage |
| COD10K non-camouflaged | 1,974: 784 backgrounds and 1,190 visible animals | Presence check |

### Damage conditions (`camo/src/corruptions.py`)
- **Single damages, at severity 3:** noise, Gaussian blur, motion blur, darkening, low contrast, fog, JPEG and rain. The severity settings match COD10K-C.
- **Severity 5, never seen in training:** motion blur, noise and JPEG.
- **Combined conditions,** applied in physical order (blur, then darkening, then noise, then JPEG):
  - **Drone:** motion blur, then strong JPEG.
  - **Night:** dark, then noisy, then JPEG.
  - **CCTV:** soft focus, then noise, then very strong JPEG.

The full 28-condition grid would have taken about 45 hours, so a prioritised 15-condition grid was used (D-035).

### Metrics and statistics
- **Dice at threshold 0.5** on raw probabilities is the headline number. S-measure, IoU and MAE are also stored.
- **Paired bootstrap:** 10,000 resamples of per-image differences, giving a 95% CI. A gain counts only if its interval excludes zero.
- **MoCA:** whole videos are resampled, not single frames, because frames from one video are strongly related.

---

## 7. Results

### 7.1 COD10K, input protocol (Dice)
| Condition | SINet-V2 | A | R | B | C | D | **E** |
|---|---|---|---|---|---|---|---|
| Clean | 0.734 | 0.832 | 0.830 | 0.835 | 0.835 | 0.835 | 0.833 |
| Motion blur s3 | 0.538 | 0.721 | 0.722 | 0.763 | 0.767 | 0.766 | 0.764 |
| Noise s3 | 0.557 | 0.680 | 0.678 | 0.718 | 0.723 | 0.721 | 0.724 |
| JPEG s3 | 0.640 | 0.715 | 0.712 | 0.759 | 0.762 | 0.763 | 0.763 |
| Noise s5 (unseen) | 0.468 | 0.552 | 0.562 | 0.623 | 0.636 | 0.632 | 0.636 |
| JPEG s5 (unseen) | 0.580 | 0.648 | 0.645 | 0.719 | 0.723 | 0.725 | 0.726 |
| Drone | 0.480 | 0.602 | 0.598 | 0.692 | 0.701 | 0.700 | 0.702 |
| Night | 0.414 | 0.465 | 0.427 | 0.572 | 0.589 | 0.585 | 0.589 |
| CCTV | 0.430 | 0.541 | 0.537 | 0.631 | 0.641 | 0.642 | 0.645 |
| **Mean, 14 damaged** | 0.568 | 0.679 | 0.678 | 0.729 | 0.734 | 0.734 | 0.734 |

All 15 conditions are in `camo/results/tables/cod10k_input_dice.csv`.

### 7.2 Gains with 95% confidence intervals (mean over damaged conditions)
| Comparison | COD10K | CAMO | NC4K |
|---|---|---|---|
| B − A (training on damage) | **+0.050** [0.047, 0.052] | **+0.029** [0.021, 0.036] | **+0.064** [0.061, 0.067] |
| R − A (restore first) | −0.002 [−0.003, −0.000] | −0.001 [−0.003, 0.002] | not run |
| C − B (detail cue) | **+0.005** [0.004, 0.006] | +0.000, not significant | not run |
| D − B (four cues, equal weights) | **+0.005** [0.004, 0.006] | +0.002, not significant | not run |
| E − B (four cues, gated) | **+0.005** [0.004, 0.006] | **+0.006** [0.003, 0.008] | **+0.006** [0.005, 0.008] |
| E − C | −0.000, not significant | **+0.005** [0.002, 0.008] | not run |
| E − D | +0.000, not significant | **+0.004** [0.001, 0.007] | not run |

Clean-image cost: on clean COD10K, E is 0.833 against A's 0.832. **Robustness costs no clean accuracy.**

### 7.3 What the numbers mean
1. **Training on damaged images is most of the gain.** It holds on all three test sets and costs nothing on clean images.
2. **The classical cues add a small, real extra gain:** +0.005 on COD10K, and +0.006 for E on CAMO and NC4K. The gain is largest under the heaviest damage: +0.017 at night, +0.014 for CCTV and +0.013 at severity-5 noise, comparing E with B.
3. **The gate is mixed.** On COD10K, gated, equal-weight and detail-only cues all perform the same. On CAMO, only the gated version beats B, C and D. So we can't claim the gate generally improves accuracy.
4. **Restoration doesn't help.** Median filtering, gamma and unsharp masking only help Gaussian blur, by +0.023, and hurt at night, by −0.038. Overall they slightly hurt. The filters remove the faint evidence camouflage detection depends on. Restoration settings were tuned on validation (D-036), but only two of them, so the claim is "simple rule-based restoration doesn't help", not "restoration can't help".
5. **Full resolution, native protocol:** B − A is +0.016, but E − B is −0.001. When the camera is high resolution, the damage is mostly averaged away and the cues add nothing.
6. **Per habitat:** all habitats gain. Terrestrial is hardest, 0.683 for E against 0.624 for A.

### 7.4 The gate explains the model (Figures `gate_response` and `cue_dropout`)
- **Gate response:** as JPEG gets stronger, the gate moves trust to **colour**, from 20% to 45%. Under darkening and motion blur it moves trust to **detail**.
- **Cue dropout:** switching one cue off at test time shows how much it was really used.
  - Removing **colour** costs −0.015 under JPEG, −0.021 under drone, −0.025 at night and −0.027 under CCTV, the largest losses in the table.
  - Removing **structure** costs the most on clean images (−0.004) and under noise (−0.008).
  - Removing **detail** costs almost nothing, even where the gate trusts it.
- **Conclusion:** the gate's weights match real usefulness for colour and structure, but not for detail. This honest, checkable explanation is what the gate contributes.

### 7.5 Real footage: MoCA-Mask (bootstrapped by video)
| Frames | A | B | E | B − A | E − B |
|---|---|---|---|---|---|
| All 4,691 | 0.747 | 0.767 | 0.767 | **+0.020** [0.007, 0.034] | +0.001, not significant |
| Blurriest 25% | 0.772 | 0.807 | 0.810 | **+0.035** [0.015, 0.058] | +0.003, not significant |
| Darkest 25% | 0.714 | 0.748 | 0.748 | **+0.035** [0.012, 0.064] | −0.001, not significant |

- Training on *synthetic* damage carries over to *real* video, and helps most on the worst frames.
- The cues add nothing on real video. MoCA is mostly 720p, which is like the native protocol, so this fits Finding 5.
- **Compare models within a quartile, not across quartiles.** The blurriest frames score higher than the sharpest ones because they come from easier scenes, not because blur helps.

### 7.6 Is there an animal at all? (presence check)
The score is the mean of the top 1% of predicted probabilities.

| Negatives | Condition | A: AUROC | E: AUROC |
|---|---|---|---|
| Empty backgrounds | Clean | 0.923 | 0.933 |
| Empty backgrounds | Night | 0.628 | **0.762** |
| Empty backgrounds | CCTV | 0.739 | **0.841** |
| Clearly visible animals | Clean | 0.281 | 0.324 |

- Against empty backgrounds the score works well, and our model makes **far fewer false alarms in poor footage**.
- Against clearly visible animals it fails: AUROC below 0.5 means visible animals score *higher* than camouflaged ones. **The models detect animals, not camouflage.** This is an honest limitation and worth saying aloud in the viva.

---

## 8. How this differs from Prof. Jerripothula's work

These are the professor's two most relevant papers. Read both abstracts before the viva.

### 8.1 SAMWave (BMVC 2025; Yadav, Gupta and Jerripothula; arXiv 2507.20186)
**What it does, from the paper:** SAMWave adapts SAM and SAM2 to low-level segmentation tasks. It uses the **wavelet transform**, a real-valued DWT plus complex-valued variants with "complex-valued adapters", to extract "richer, multi-scale high-frequency features". These go into adapters inside SAM's image encoder. The authors argue that the Fourier-based high-frequency features of earlier adapters give "limited benefits". It's evaluated at 1024 × 1024 on four tasks:
- Camouflage: COD10K, CAMO, CHAMELEON.
- Shadow detection: SBU, ISTD.
- Defocus blur detection: CUHK, DUT.
- Polyp segmentation: Kvasir, ClinicDB, ColonDB, ETIS.

All of these are **clean** benchmarks.

| | SAMWave (professor) | CueGate-SAM (ours) |
|---|---|---|
| **Goal** | Better adaptation of SAM on clean images, across 4 tasks | Keep camouflage detection working when **image quality is poor** |
| **Extra signal** | One kind: wavelet high-frequency bands | **Four kinds** of classical cues: detail, colour (Lab), texture (LBP), structure (structure tensor) |
| **How the signal is weighted** | Learned inside the adapters | A **separate gate** driven by **six unlearned quality measures**, giving one readable weight per cue per image |
| **Explanation** | Not a stated goal | A core output: "this image looks compressed, so the model trusted colour" |
| **Backbone** | SAM and SAM2, 1024 px | SAM2-UNet (Hiera-L) at 352 px, from its published camouflage checkpoint, with zero-initialised projections |
| **Evaluation** | Clean benchmarks | 15 damage conditions, 2 protocols, real video, presence check, paired bootstrap CIs |

**How the two connect.** SAMWave's premise is that **high-frequency detail is crucial** for adapting SAM. Our project asks the next question: *what happens when poor image quality destroys that detail?*
- Noise, blur and compression all attack high frequencies.
- Our cue dropout shows that under compression and night conditions the model shifts to **colour** information, which survives better.
- Removing our high-frequency detail cue costs almost nothing.

This **extends** the professor's line of work rather than contradicting it. Our detail cue is a simple Laplacian and DoG, not his richer wavelet features, so his result is not tested here. A natural follow-up would be to plug SAMWave's wavelet branch in as a fifth cue and see whether the gate turns it down when it is unreliable.

**Why we didn't use wavelets:** wavelets appear only briefly in the course (Lec 1, and Haar in Lec 9), while the Laplacian and DoG are core material (Lec 4). Using wavelets would also make the project look like a re-implementation of SAMWave (D-015).

### 8.2 CIRCOD (WACV 2025; Gupta, Jerripothula and Tillo)
**What it is:** *Co-Saliency Inspired Referring Camouflaged Object Discovery*. In referring camouflage detection, the model receives **reference images** of the kind of object to find, then looks for that object camouflaged in a target image. The method borrows ideas from **co-saliency**, which finds the object common to a group of images, a long-running theme in the professor's work. CIRCOD has no image-degradation content (checked in D-020). Our network couldn't open its paper page, so check the exact method details from the paper before quoting them.

| | CIRCOD (professor) | CueGate-SAM (ours) |
|---|---|---|
| Extra input | Reference images that say *what* to look for | None. One image, with quality judged from the image itself |
| Question answered | Which object does the user mean? | Can we still find the animal when the camera is poor? |
| Image quality | Clean benchmarks | The central variable |

**Link to our presence check:** we found that our models can't tell a camouflaged animal from a clearly visible one, because they find "an animal". A reference-guided setting like CIRCOD's targets *which* object is wanted. Testing referring camouflage detection on degraded images would be a natural joint direction. Say this as an idea only, since we haven't tested it.

### 8.3 The professor's wider research themes
The professor's earlier paper titles show a long interest in **fusing multiple cues adaptively**:
- *Appearance Fusion of Multiple Cues for Video Co-localization*
- *Comprehensive Saliency Fusion for Object Co-segmentation*
- *ASOC: Adaptive Self-aware Object Co-localization*

Our gate is also an adaptive fusion of several cues, but its weights come from **measured image quality**. This is a good framing for the presentation, but **we only know these titles, not the papers.** Read their abstracts before saying anything specific.

---

## 9. What is novel, and what is not

**What we claim:**
1. **Measurement study:** whether damage is applied before or after resizing explains the conflict between camouflage-robustness benchmarks. It reproduces both, on all 2,026 COD10K images with two detectors. No camouflage paper we found does this. One histopathology paper studies blur order only.
2. **Method:** an adapter that weights several classical cues using unlearned, readable quality measures, verified with cue dropout. To our knowledge this hasn't been done. The claim is narrow, so say "to our knowledge".
3. **Evidence that's useful even where it's negative:**
   - Simple restoration doesn't help.
   - Cues help on low-resolution damage but not at full resolution or on real video.
   - Presence scores fail against visible animals.

**Not novel, so never claim it:** SAM2-UNet, adapters on foundation models, training with corruption augmentation, combined degradations, a corruption benchmark including SAM2, or any single cue.

**Limitations to state openly:**
- Training damage is synthetic.
- Each version was trained once, with one seed.
- Some conditions overlap between training and test (severities 1 to 4). Severity 5 and real video are the truly unseen tests.
- Validation images were part of the published checkpoint's training set, so they were used only to choose epochs.
- The novelty search should be repeated before submission.

---

## 10. Where everything is

| What | Where |
|---|---|
| Quality measures, cues, gate, adapter | `camo/src/quality.py`, `cues.py`, `cuegate.py` |
| Damage functions and protocols | `camo/src/corruptions.py` |
| Restoration baseline R | `camo/src/restore.py`, tuned by `camo/scripts/tune_restore.py` |
| Training | `camo/scripts/train.py`, `train_all.ps1` |
| Evaluation | `camo/scripts/benchmark.py`, `evaluate_all.ps1` |
| Gate response, cue dropout, MoCA quality | `camo/scripts/gate_analysis.py` |
| Tables with CIs | `camo/scripts/analyze.py`, writing to `camo/results/tables/` |
| Figures | `camo/scripts/make_figures.py`, `pick_examples.py`, writing to `camo/results/figures/` |
| Demo | `camo/app.py`, with examples in `camo/demo_examples/` |
| Tests | `camo/tests/` |
| Trained weights | `camo/weights/runs/{B,C,D,E}/best.pth`, not in git because of size |

**Running the demo:** `.venv\Scripts\python.exe camo\app.py`. It opens in the browser. Pick an example, choose a camera preset (Drone, Night, CCTV) or move the damage sliders, then compare the published model with ours and read the gate chart.

---

## 11. Questions to expect in the viva

| Question | Answer |
|---|---|
| Why not just clean the image first? | We tested it: arm R loses 0.002 on average and 0.038 at night. Filters remove the faint evidence camouflage depends on |
| Isn't this just data augmentation? | Augmentation gives most of the gain, and we say so. The cues add a significant extra +0.005 to +0.006 on three test sets, and the gate explains the model's behaviour |
| Does the gate help? | On CAMO, yes. On COD10K, it matches the ungated versions. Its main value is explanation, checked by cue dropout |
| Is the explanation true? | For colour and structure, yes: the cue the gate trusts most is the one whose removal hurts most. For detail, no, and we report that |
| Why hand-crafted quality measures? | They need no labels, are readable, and can be checked on their own. Gate 2 showed they track severity |
| How is this different from SAMWave? | See Section 8.1: one wavelet signal on clean images, against four classical cues weighted by measured quality on damaged images |
| Why 352 px? | It's SAM2-UNet's native size, and it fits an 8 GB laptop GPU |
| Why do the benchmarks disagree? | Where the damage happens. See Section 4 |
| Does it work on real footage? | Training on damage helps (+0.020, and +0.035 on the blurriest frames). The cues don't add more there |
| What's the weakest part? | Synthetic training damage, a single seed, and the presence check failing on visible animals |

---

## 12. What's left
1. **The report,** in the CVPR 2022 template, due **Nov 6**. All tables and figures are ready.
2. **The slides,** for Nov 7 and 8.
3. **A repeat novelty search,** and checking the CIRCOD details against its paper.
4. **GitHub:** create the repository and push it yourself. The write-up needs the link.
5. **Practise the demo.**
