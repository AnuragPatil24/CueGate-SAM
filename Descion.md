# Decision Log: CueGate-SAM

Every project decision is recorded here with its reason, proof and necessity. Add new entries at the end and never rewrite old ones. If a decision changes, add a new entry that replaces it, so the viva trail stays honest.

**Status labels:** Active, Replaced, Pending.

**Evidence labels:**
- **Measured:** we ran it ourselves.
- **Read:** taken from the source text.
- **Reported:** someone else's number that we haven't reproduced.
- **Hypothesis:** a suspicion, not yet tested.

---

## Summary of where things stand (Nov 2, 2026)

| Question | Answer |
|---|---|
| Are the cited papers real and accurately described? | Yes, with the corrections in Section B |
| Does the original novelty survive? | **Only 1 of 3 claims survives, and in a narrower form** |
| New novelty | A measurement study of why robustness scores disagree. Open, but unproven until measured |
| Biggest risk | SAM2-UNet may already be robust, which leaves little for an adapter to improve |
| Next critical step | The 2 x 2 test in D-007 |

---

## A. Novelty verification

| # | Original claim | Verdict | Proof |
|---|---|---|---|
| 1 | First corruption benchmark including SAM2-based detectors | **False. Withdrawn** | [VFM-CV](https://github.com/Neel-Ronak-Org/VFM-CV) benchmarks SAM2-UNet and SAM-Adapter under 8 corruptions x 5 severities on 4 datasets. `results/benchmark.csv` has the numbers (Read) |
| 2 | Combined degradations are new for camouflage | **False. Withdrawn** | [RUN++](https://arxiv.org/abs/2508.15027) has a "blind degradation" setting, a random combination of one to three, on COD10K. [NUN](https://arxiv.org/abs/2511.18164) has a "Combined Degradation" table of low-resolution, low-light and hazy (Read) |
| 3 | Adapter weighting several classical cues using unlearned quality measures | **Survives, narrowed** | No match in several targeted searches. Nearest neighbours are NUN and GaRA-SAM below. Absence from searches isn't proof, so keep "to our knowledge" |
| - | My earlier statement: "no work joins adapter-based camouflage detection with degradation handling" | **False. Withdrawn** | VFM-CV trains a prompt-free MobileSAM adapter with corruption augmentation (Read) |
| - | Plan's statement: SAM2 models "have not been tested this way" | **False. Withdrawn** | Same VFM-CV benchmark |

### Why claim 3 still stands
- **NUN** uses *learned* quality scorers (TOPIQ, Q-Align, MUSIQ) to pick the best *restored image*, guided by a vision-language model. Ours uses classical formulas to weight *cues* inside an adapter, with no restoration network.
- **GaRA-SAM** uses a *learned* gate to choose *adapter rank*. It needs prompts, and its text has no camouflage dataset.
- **VFM-CV's adapter** has no cues and no quality gate.

Claim 3 is narrow, and a claim of being different isn't a claim of being better. Only experiments can show it helps.

### What VFM-CV is
- Public repo, MIT licence, last pushed Oct 30, 2026.
- A course project. Its history mentions `CV_mini_project_I015_I032.pptx`.
- Its README refers to "the paper", but the paper was removed from the repo, and I found no published version.
- It isn't peer reviewed, so its numbers are Reported, not trusted facts.

---

## B. Source verification

Papers were downloaded and their text searched. "Key sections" means I read the parts that matter for this project, not every page.

| Source | Exists | Checked | Verdict on what we said |
|---|---|---|---|
| [COD10K-C](https://arxiv.org/abs/2606.02603) | Yes, single-author preprint | Key sections and tables | Accurate. 0.734, 0.549, 0.730, 0.632 and 0.614 all confirmed. No SAM model tested. Code not released |
| [SAM2-UNet](https://arxiv.org/abs/2408.08870) | Yes | Key sections, repo code | Accurate. Its adapter is a plain MLP with **no** hand-made prior |
| [SAM2-Adapter](https://arxiv.org/abs/2408.04579) | Yes | Key sections, repo | Accurate. Fpe + Fhfc prior, Hiera-L, 20 epochs, A100 GPUs. Repo config uses 1024 px |
| [SAM3-Adapter](https://arxiv.org/abs/2511.19425) | Yes | Text search | Accurate. No corruption evaluation |
| [SAM-Adapter](https://arxiv.org/abs/2304.09148) | Yes | Text search | Fpe + Fhfc prior confirmed |
| [GaRA-SAM](https://arxiv.org/abs/2506.02882) | Yes | Key sections | Accurate. Learned gate, prompts, LVIS and similar. No camouflage hits |
| [RobustSAM](https://arxiv.org/abs/2406.09627) | Yes | Text search | Accurate. Prompted. Camouflage appears only as a citation |
| [PGE-SAM](https://arxiv.org/abs/2606.30477) | Yes | Text search | Accurate. Zero camouflage hits |
| [SAMWave](https://arxiv.org/abs/2507.20186) | Yes, co-authored by your professor | Text search | Accurate. High-frequency prior, no corruption robustness study |
| [CIRCOD](https://openaccess.thecvf.com/content/WACV2025/html/Gupta_CIRCOD_Co-Saliency_Inspired_Referring_Camouflaged_Object_Discovery_WACV_2025_paper.html) | Yes, co-authored by your professor | Text search | Accurate. Referring camouflage. No degradation content |
| [OPC16K](https://arxiv.org/abs/2608.11135) | Yes | Text search | No overlap. "Realistic" means images that may hold no object |
| [NUN](https://arxiv.org/abs/2511.18164) | Yes | Key sections | **Missed earlier.** Direct competitor. Includes real degraded subsets PCOD-LQ and MCOD-LQ |
| [RUN++](https://arxiv.org/abs/2508.15027) | Yes | Key sections | **Missed earlier.** Direct competitor. Restoration network per degradation |
| [Robust Promptable VOS](https://arxiv.org/abs/2605.12006) | Yes | Text search | SAM2 degrades under corruption on prompted video. Supports the premise only weakly, since it's a different task |
| [Blur order in histopathology](https://arxiv.org/abs/2603.04081) | Yes | Search summary only | Compares blur before and after resizing. Read before citing |

**TellTale sources, no longer used in the project:** Making and Breaking of Camouflage and CFCamo titles confirmed on arXiv. The SeamCam page loads, but I read only its summary. The Geirhos arXiv ID did not resolve through the API, so don't cite it without checking.

**Corrections to what I told you earlier:**
- "Leading adapters rely on high-frequency detail" is true for SAM2-Adapter and SAM3-Adapter, not SAM2-UNet.
- I missed NUN, RUN++ and VFM-CV in my first searches, and stated gaps more confidently than the searches justified.
- `camo/third_party/SAM2-Adapter` isn't empty any more. A separate session filled it.

---

## C. Decisions

### D-001: Project direction is CueGate-SAM
- **Status:** Active, approved by you on Nov 2. Your professor's approval is still pending.
- **Decision:** Benchmark detectors on poor-quality images, then build a quality-gated adapter. TellTale is dropped.
- **Reason:** You asked for a real field problem, course concepts and trending topics.
- **Proof:** Real cameras degrade images (COD10K-C, NUN). Adapters are a current method (SAM2-Adapter, SAM3-Adapter, SAMWave).
- **Necessity:** The topic must be fixed now, given the Nov 6 deadline.

### D-002: Build the adapter for one model only, SAM2-UNet
- **Status:** Active
- **Decision:** Benchmark several models, adapt only SAM2-UNet.
- **Reason:** An adapter wraps specific encoder blocks, so it can't be attached to every model.
- **Proof (Read):** `SAM2UNet.py:51-67` wraps each Hiera block. It runs at 352 px. A public `SAM2UNet-COD.pth` exists, and we now have it.
- **Necessity:** There's no time to build several. 8 GB of VRAM rules out heavier models.

### D-003: Don't train SAM2-Adapter
- **Status:** Active
- **Reason:** Repo config uses 1024 px with batch size 2. Its authors used 3-4 A100 GPUs. I found no SAM2 camouflage weights for it.
- **Proof (Read):** `configs/demo.yaml`; repo README; paper Section 4.2.
- **Necessity:** It won't fit your laptop in the time left. Arm C stands in for a single high-frequency prior.

### D-004: Fine-tune from `SAM2UNet-COD.pth`, don't train from scratch
- **Status:** Active
- **Reason:** Baseline A then needs no training, and every arm starts equal.
- **Proof (Measured):** The checkpoint reads correctly with `weights_only=True`: 1,192 tensors, 216.5 M parameters. It hasn't yet been loaded into the model or run.
- **Necessity:** Saves many hours, and makes the comparison between arms fair.

### D-005: The adapter must beat plain augmentation
- **Status:** Active
- **Reason:** Otherwise the adapter's gain could simply be augmentation's gain.
- **Proof (Reported):** COD10K-C: removing augmentation costs 0.041 Dice, removing its frequency prior costs 0.018. VFM-CV: augmentation raised corrupted Sα from 0.799 to 0.810.
- **Necessity:** This is the first question a professor will ask. Arm B exists for this.

### D-006: Withdraw two novelty claims and narrow the third
- **Status:** Active
- **Decision:** See Section A.
- **Reason:** Stating a false "first" in the report would be wrong, and it could be challenged in the viva.
- **Necessity:** Novelty is graded, and you must defend it.

### D-007: Add a 2 x 2 measurement test as the new Gate 1
- **Status:** Active
- **Decision:** On SAM2-UNet and SINet-V2, cross **where corruption is applied** (P1 at model input, P2 at full image size) with **how predictions are scored** (raw or stretched).
- **Reason:** Two benchmarks disagree strongly on the same model and settings, and SAM2-UNet looks nearly immune. We need to know what's real before building anything.
- **Proof:**

| SINet-v2, Dice (both clean = 0.734) | COD10K-C | VFM-CV |
|---|---|---|
| Motion blur, severity 3 | 0.538 | 0.681 |
| Gaussian noise, severity 3 | 0.558 | 0.687 |

| SAM2-UNet on VFM-CV (Reported) | Dice | Retention |
|---|---|---|
| Clean | 0.831 | - |
| Motion blur, severity 3 | 0.808 | 0.972 |
| Gaussian noise, severity 3 | 0.802 | 0.965 |
| Gaussian blur, severity 5 | 0.759 | 0.913 |
| Rain, severity 5 | 0.741 | 0.892 |
| Mean over all corruptions | | 0.973 |

  - VFM-CV corrupts at full size, and its own notes say that 352 px models shrink images about 2.8x, averaging away noise and blur (Read).
  - SAM-Adapter at 1024 px retains less (0.937) than SAM2-UNet at 352 px (0.973). Whether that's architecture or resolution is a Hypothesis.
  - VFM-CV tested resolution only on its own model, and only on CHAMELEON in the published file (Read).
  - COD10K-C doesn't state whether it corrupts before or after resizing (Read). So resolution as the reason for the disagreement is a Hypothesis.
- **Necessity:** Under VFM-CV's protocol, SAM2-UNet would **fail** the original Gate 1, since its retention at severity 3 is above 0.95. This test decides whether an adapter is worth building, and it needs no training.

### D-008: Score raw probabilities as well as stretched ones
- **Status:** Active
- **Reason:** Per-image stretching turns a weak, unsure prediction into a confident-looking mask, which could hide the effect of degradation.
- **Proof:**
  - `third_party/SAM2-UNet/test.py:41` stretches every map (Read).
  - VFM-CV's SAM2-UNet wrapper keeps this step (Read).
  - `py_sod_metrics.utils.prepare_data` also stretches. In my test, a prediction ranging 0.10 to 0.20 became 0.0 to 1.0 (Measured).
- **Necessity:** Otherwise we'd repeat a possible flaw. Whether this changes results is a Hypothesis until measured. I found no camouflage paper studying it.
- **Credit:** A separate session found this.

### D-009: Cite VFM-CV, and don't copy its code or results
- **Status:** Active
- **Reason:** It's prior work on a very similar project, from another course. Copying risks plagiarism, and the course penalty is a zero for the whole project component.
- **Proof (Read):** Lec 1 course policy.
- **Necessity:** Ignoring it would be dishonest, and your professor could find it. Citing it first makes your position stronger.

### D-010: Use COD10K-C's severity values, and two corruption protocols
- **Status:** Active
- **Reason:** Using the same values means results can be compared directly, and the two protocols are what D-007 tests.
- **Proof (Read):** COD10K-C Section 3.2.
- **Necessity:** P1 at 352 px also keeps the JPEG grid aligned for the blockiness measure.
- **Note:** The drafts `camo/src/corruptions.py` and `benchmark.py` use 384 px and one `mixed` condition. They must be updated.

### D-011: Combined conditions stay, but aren't a novelty claim
- **Status:** Active
- **Reason:** Real cameras combine degradations, and these may be where SAM2-UNet really struggles.
- **Proof (Read):** RUN++ and NUN already do combinations, though not blur + noise + JPEG on a SAM2 model.
- **Necessity:** They give the adapter a chance at conditions with more room to improve.

### D-012: Model low light as darkening plus noise
- **Status:** Active
- **Reason:** Darkening alone barely hurts.
- **Proof (Reported):** COD10K-C: SINet-v2 scores 0.731 on brightness vs 0.734 clean. VFM-CV: SAM2-UNet goes from 0.831 to 0.807 at the strongest brightness setting.
- **Necessity:** A darkening-only test would make "low light" look solved when it isn't.

### D-013: Include a restoration-first baseline (arm R)
- **Status:** Active
- **Reason:** NUN and RUN++ restore images first, so we must show whether simple restoration is enough.
- **Proof (Read):** Both papers. Unsharp masking, median filtering, gamma and histogram equalisation are all in your notes.
- **Necessity:** Answers "why not just clean the image first?" It uses course methods and needs no training.

### D-014: Use Lab instead of HSI for the colour cue
- **Status:** Active
- **Reason:** Hue is unstable at low saturation, which is what dark images have.
- **Proof (Read):** HSI is taught in Lec 7. Lab isn't in the notes, so explain it in the report.
- **Necessity:** Colour must stay usable on dark images, or the gate has nothing to switch to.

### D-015: Use Laplacian and DoG instead of wavelets
- **Status:** Active
- **Reason:** They're taught in the course, and wavelets are close to your professor's SAMWave.
- **Proof (Read):** Laplacian and DoG are in Lec 4. "Wavelet" appears only in Lec 1 and as Haar in Lec 9.
- **Necessity:** Keeps the method defensible and clearly different from SAMWave.

### D-016: Use classical quality measures, not learned ones
- **Status:** Active
- **Reason:** They're readable, need no labels, use course methods, and they're what separates this from NUN and GaRA-SAM.
- **Proof:** Gate 2 will test whether they track severity. Not yet measured.
- **Necessity:** This is the core of the surviving novelty claim.

### D-017: Download the dataset from official Drive zips
- **Status:** Active. Verified Nov 2
- **Reason:** The Hugging Face route was far too slow, so it was stopped and its partial folder removed.
- **Proof (Measured):** Hugging Face: about 68 files per minute, roughly 7 hours in total. Drive: about 8 MB/s.

| Set | Images | Masks | Paired | Unreadable |
|---|---|---|---|---|
| Train (COD10K + CAMO) | 4,040 | 4,040 | Yes | 0 |
| COD10K test | 2,026 | 2,026 | Yes | 0 |
| CAMO test | 250 | 250 | Yes | 0 |
| CHAMELEON | 76 | 76 | Yes | 0 |
| NC4K | 4,121 | 4,121 | Yes | 0 |

- **Necessity:** Experiments start tomorrow.
- **Notes:** 2 CHAMELEON images differ in size from their masks, so always resize the prediction to the mask. An empty `example` folder in CAMO's zip was removed. `camo/data/nc4k_drive/` is a spare copy of NC4K. The zips can be deleted to free 3 GB.

### D-018: Keep each download in its own folder, and pin BiRefNet
- **Status:** Active
- **Reason:** Downloads are untrusted, and BiRefNet's model code runs through `trust_remote_code`.
- **Proof (Measured):** The local copy is revision `6aa747035f7b70caa1c66cd5c128d6e9c654ddd0`.
- **Necessity:** Prevents one bad file or changed revision from affecting the rest. Load `.pth` files with `weights_only=True` where possible.

### D-019: BiRefNet is optional
- **Status:** Active
- **Reason:** It's a useful modern comparison, but the project doesn't depend on it.
- **Proof:** Input size and speed on your GPU aren't measured yet.
- **Necessity:** First thing to cut, after the held-out test, if time runs out.

### D-020: Tune on validation only, and run one seed per arm
- **Status:** Active
- **Reason:** Using the test set to choose settings would inflate results. Several seeds won't fit the time.
- **Proof:** About 3.5 days remain, and training speed is unmeasured.
- **Necessity:** Keeps results honest. State the one-seed limit in the report.

### D-021: Alternative project ideas searched
- **Status:** Pending your choice. `plan.md` is unchanged until you decide.
- **Reason:** You asked for other ideas on Nov 3. CueGate may have little to gain if SAM2-UNet is already robust.
- **Method:** Web searches, plus arXiv API and GitHub repository searches meant to find existing versions of each idea. Semantic Scholar refused the requests (rate limit), so it wasn't checked.
- **Proof:**

| Idea | Closest existing work | Verdict |
|---|---|---|
| Guaranteed search zone, using conformal prediction | arXiv abstract search for conformal + camouflaged returned only physics and thermal papers. GitHub: 0 repos. Nearest are [morphological prediction sets](https://arxiv.org/abs/2503.05618), which are medical, and [Macro Mondrian risk control](https://arxiv.org/abs/2610.35035), a general method | **Recommended.** The maths is known, the camouflage application and quality grouping look new |
| Camouflage-tuned JPEG | [J4D](https://arxiv.org/abs/2606.16185) covers classification and Pascal VOC segmentation. None found for camouflage | Viable, but DCT isn't in your notes, and J4D could cover it |
| Reveal hidden objects to people | [RIDE](https://arxiv.org/abs/2605.15450) enhances inside a network. A [CLBP paper](https://link.springer.com/chapter/10.1007/978-981-95-5021-0_11) enhances for detectors. No human-facing study found | Viable, but it depends on a user study in 3 days |
| Test-time adaptation | [GenSAM](https://arxiv.org/abs/2312.07374), [SAM-TTT](https://arxiv.org/abs/2509.11884) | Rejected, too close |
| SAM3 text prompts | [IJCAI 2026 paper](https://www.ijcai.org/proceedings/2026/109) | Rejected, already done |
| Images with no object | [OPC16K](https://arxiv.org/abs/2608.11135), [CFCamo](https://arxiv.org/abs/2606.11231) | Rejected, already done |
| Tiny objects at high resolution | HitNet, MSRNet and similar | Rejected, crowded |
| Multispectral | [MCOD](https://arxiv.org/abs/2509.15753) | Rejected, no data or time |

- **Necessity:** A project that can't be undercut by one repo result is safer with 3 days left.

### D-022: Stay with CueGate-SAM, conditional on Gate 1
- **Status:** Active, on my reading of your Nov 3 message that "the previous idea" means CueGate. Replaces the pending choice in D-021.
- **Decision:** Keep `plan.md` as it is, and run the Gate 1 test before building the adapter.
- **Reason:** You prefer it. It's a model-building project, and adapters are the trending topic you asked for and close to your professor's own work.
- **Proof:** The quality-gated cue adapter still has no match in my searches (Section A). Its gain is unproven: VFM-CV's numbers leave about 0.02 Dice under its protocol (Reported).
- **Necessity:** Gate 1 needs no training, and it shows whether four training runs are worth the time.

### D-023: CueGate-SAM confirmed, alternatives closed
- **Status:** Active. You confirmed on Nov 3 that the project stays with the adapter idea. This resolves D-021, and the search zone, JPEG and Reveal ideas are closed.
- **Decision:** `plan.md` stands. Gate 1 still runs first, because it tells us where the adapter has real damage to fix.
- **Reason:** Your choice, after hearing the risks in D-022.
- **Proof:** See D-022 and Section A.
- **Necessity:** With 3 days left, one fixed direction avoids splitting the time.

### D-024: Resize with PIL bilinear, as the official test code does
- **Status:** Active
- **Decision:** All resizing uses PIL bilinear. This replaces the OpenCV resize in the earlier drafts.
- **Reason:** PIL bilinear smooths while shrinking, and OpenCV's `INTER_LINEAR` doesn't. Gate 1 asks whether shrinking hides damage, so using the wrong resize would change the answer.
- **Proof (Read):** SAM2-UNet `dataset.py` and SINet-V2 use torchvision or PIL resize on PIL images.
- **Necessity:** Results must match how these models were really tested.

### D-025: Models reproduce published clean scores
- **Status:** Active, on a 60-image sample only
- **Proof (Measured):** SAM2-UNet: Sα 0.879, against 0.880 reported by VFM-CV on all of COD10K. SINet-V2: Dice 0.730 against 0.734, and Sα 0.816 against 0.815. Both load with `strict=True`.
- **Reason:** Confirms the code path before trusting corrupted results.
- **Necessity:** Without this, a low corrupted score could just be a bug.

### D-026: Decide Gate 1 on a fixed 500-image subset
- **Status:** Active
- **Decision:** Use 500 COD10K images, chosen with a fixed seed, for the go or no-go decision. Final tables must use all 2,026.
- **Reason:** Speed is 7-10 images/s, so the 32 Gate 1 conditions would take about 3 hours on the full set, and about 30-40 minutes on 500.
- **Proof (Measured):** 60-image speed test.
- **Necessity:** The answer is needed today, before the adapter work starts.
- **Warning:** The planned full grid at this speed would take far too long, so evaluation must be sped up, or the grid cut, before Nov 4.

### D-027: Expand the problem statement in `plan.md` and restate its evidence
- **Status:** Active
- **Decision:** Section 2 of `plan.md` now gives the problem, input, output, purpose, existing benchmarks and reasons for the approach. The headline is "keep working on poor images, without a degradation label, and show which cues were trusted". The earlier claim that stretching hides weak predictions is now marked as looking unimportant.
- **Reason:** You asked for this on Nov 3. The early Gate 1 rows also changed the evidence, so repeating the old wording would have been wrong.
- **Proof (Measured, preliminary):** SAM2-UNet, input protocol, 500 images, as of 00:29: clean 0.819 Dice, motion blur 0.705 (s=3) and 0.612 (s=5), noise 0.656 (s=3). Raw and stretched Dice differ by 0.0001, with peak probability at least 0.979.
- **Proof (Reported):** VFM-CV `benchmark.csv`, COD10K-C Table 3, SAM2-UNet Table 3, NUN Tables 1 and 6, and SAM3-Adapter Table 1.
- **Caveat:** Native-protocol rows weren't finished, so Gate 1 isn't decided. The SAM2-Adapter row came from a garbled PDF table and needs checking against the PDF.
- **Necessity:** The team needs one clear statement of the project for the report, the slides and your professor.

### D-028: Add a presence check as a small add-on
- **Status:** Active, on your Nov 3 suggestion
- **Decision:** Add an image-level presence score. Start from the probability map with no training, and train a small head only if that's weak. Keep it separate from arms A to E.
- **Reason:** A detector that always draws a mask gives false alarms on empty scenes, which matters for monitoring. It's a clean extension, but it mustn't disturb the main comparison.
- **Proof (Measured):** `COD10K-v3.zip` holds 4,934 non-camouflaged images, 2,960 train and 1,974 test, of which 784 test images are pure background. All 200 masks sampled from each split were empty. Its 2,026 camouflaged test names match our test set exactly.
- **Proof (Reported):** OPC16K and CFCamo already handle absent objects on clean images, and OPC16K's data isn't released yet. The CFCamo abstract and the OPC16K full text show no corrupted-image evaluation. Two targeted searches found no work testing presence under degradation.
- **Caveat:** Not novel on clean images. The degraded angle is "to our knowledge" only.
- **Necessity:** Answers a real weakness of the standard benchmark assumption, at low cost.

### D-029: Real footage is for testing only
- **Status:** Active, modifying your Nov 3 suggestion
- **Decision:** MoCA-Mask becomes a real test set, analysed by quality quartile. Synthetic corruption stays for training, severity curves and Gate 2.
- **Reason:** Real footage has no severity labels, so it can't drive controlled augmentation or a gate test. Real-only would lose the core experiments.
- **Proof (Measured):** MoCA-Mask has 87 videos and 4,691 paired frames, with 74 of 87 videos at 1280 x 720.
- **Proof (Searched):** PCOD-LQ and MCOD-LQ, NUN's real subsets of 160 images, are built by NUN's annotators from multi-modal sets, and I found no public release. CPD-UAV's page returned 403, so access is unconfirmed.
- **Caveat:** MoCA-Mask is YouTube video described as high quality by its sources. Real doesn't mean heavily degraded. Quartiles may mix quality with scene difficulty, so compare models only within a quartile.
- **Necessity:** Gives one honest real-world result, which synthetic data can't provide.

### D-030: Scope to wildlife, and reject UAV
- **Status:** Active, modifying your Nov 3 suggestion
- **Decision:** State the target as wildlife, and report by habitat. No separate wildlife adapter. No UAV training.
- **Reason:** The data is already wildlife, so the scope is honest and costs almost nothing. UAV would need data that I couldn't find.
- **Proof (Measured):** Filenames show 2,011 of 2,026 COD10K test images and 3,011 of 3,040 COD10K train images in animal classes.
- **Proof (Searched):** No public camouflaged-wildlife drone dataset with pixel masks found. WildLive isn't camouflage and uses SAM2 masks. CPD-UAV shows people.
- **Necessity:** A narrower claim is easier to defend, and it needs no new training data.

### D-031: Gate 1 passed, so build the adapter
- **Status:** Active, on 500 images. This closes the Gate 1 tests in D-007 and D-008.
- **Decision:** Build the adapter. Keep the input protocol and night conditions as the main targets, because that's where damage is large. Make corruption location the main measurement finding. Drop min-max stretching as an explanation.
- **Reason:** Gate 1 passes under the input protocol. Native-protocol damage is smaller, so a gain there may be small.
- **Proof (Measured, `camo/results/gate1.csv`, 32 conditions):** SAM2-UNet clean 0.819. Input: motion blur 0.705 (s=3), noise 0.656 (s=3), night 0.448. Native: 0.795, 0.787, 0.690. SINet-V2 input results land within 0.009 of COD10K-C, and native results within 0.013 of VFM-CV. Largest raw vs stretched Dice gap: 0.0005. Lowest `peak_prob` value in the file: 0.9795.
- **Caveat:** COD10K-C doesn't state its corruption order and trains at 384 px, so our match suggests the cause but doesn't prove what it did. Final tables must use all 2,026 images.
- **Necessity:** This was the go or no-go decision for the training work.

---

## D. Open questions

| Question | How it gets answered | When |
|---|---|---|
| Does SAM2-UNet really degrade? | **Answered in D-031:** yes under the input protocol, far less under the native one | Done, on 500 images |
| Does corruption location explain the benchmark disagreement? | **Answered in D-031:** it reproduces both sets of numbers | Done, on 500 images |
| Does stretching change scores? | **Answered in D-031:** no | Done, on 500 images |
| Has your professor approved the topic? | Office hours, 15:30-16:30, ACES 302 | Nov 3 |
| Does a published version of VFM-CV exist? | Repeat the search | Before submission |
| Do the quality measures track severity? | Gate 2 | Nov 4 |
