# CONTINUE HERE — Project Handover

**Project:** HRV Feature Extraction and Context-Aware False Positive Analysis for
Atrial Fibrillation Detection
**Dataset:** CACHET-CADB
**Last updated:** 6 August 2026
**Status:** Full pipeline built and validated. Phases 1–8 run end to end.
Next task is the signal-quality gate experiment (Section 6).

---

## HOW TO USE THIS FILE

If you are starting a **new chat session**, copy everything inside the code
block in Section 1 and paste it as your first message. That gives the assistant
the full context in one go, so you do not have to re-explain the project.

Everything after Section 1 is reference material for you.

---

## 1. THE PROMPT TO PASTE INTO A NEW SESSION

```
I am continuing my B.Tech final year project. Please read
AFib_Project/CONTINUE_HERE.md in my project folder first — it has the full
context, the corrections already made, and the current results.

Project: AF detection from 10-second ECG segments using HRV features, plus a
false-positive analysis against contextual sensor data (CACHET-CADB dataset).

Current state: Phases 1-8 all run successfully. SVM AUC 0.932, XGBoost AUC
0.923, using 10 MRMR-selected features on 1381 segments from 23 subjects with
subject-wise GroupKFold.

Key finding so far: the detector fires AF on 78-84% of motion-corrupted "Noise"
segments versus 8-11% on clean NSR — a 7-10x risk ratio. The mechanism is that
noise-corrupted segments have HRV features statistically indistinguishable from
AF, because artifact breaks R-peak detection and erratic RR intervals are the
AF signature.

What I want to do next: build the signal-quality gate experiment described in
Section 6 of CONTINUE_HERE.md. Please explain each step as you go — I am still
learning, so do not skip the reasoning.

IMPORTANT: Section 3 of that file lists facts that were verified the hard way
and must not be "corrected" back. Please read it before changing any config.
```

---

## 2. WHAT THIS PROJECT DOES, IN PLAIN TERMS

We are trying to answer two questions.

**Question 1 — can we detect atrial fibrillation from heart rhythm alone?**
We take 10 seconds of ECG, find each heartbeat, measure the gaps between beats,
and compute statistics describing how irregular those gaps are. Atrial
fibrillation makes the heart beat irregularly, so those statistics should give
it away. We train two classifiers (SVM and XGBoost) to make the call.

**Question 2 — when the detector is wrong, why is it wrong?**
The dataset is unusual because it also records what the person was physically
doing: movement, activity type, body position, step count. So when the model
raises a false alarm, we can look up what was happening at that moment. This is
the research contribution — most AF papers only report accuracy, not the
circumstances of failure.

---

## 3. FACTS VERIFIED THE HARD WAY — DO NOT "FIX" THESE BACK

Each of these looks wrong at first glance. Each was checked against the data.
Changing any of them back will silently corrupt your results.

### 3.1 The ECG sample rate is 1024 Hz, not 256 Hz

An early version of `config.py` said 256 Hz. Every `unisens.xml` in the dataset
declares `sampleRate="1024"` for the ECG channel. A wrong sample rate makes
every heart-rate and interval calculation wrong by a factor of four.

We downsample to 256 Hz for analysis (`TARGET_SAMPLING_FREQ`), which is where
the number 256 legitimately belongs — as a target, not the source.

### 3.2 Class 1 = AF and class 2 = NSR — NOT the other way round

The intuitive guess is that 1 means "normal". It does not. Verified three ways:

1. **RR irregularity.** Class 1 segments had RMSSD 290 ms and RR coefficient of
   variation 0.236. Class 2 had 42 ms and 0.042. AF is irregular by definition,
   so class 1 is AF — a 7x gap with no other explanation.
2. **The device's own HRV column** in `context.xlsx`, computed by the sensor
   firmware independently of our code: class 1 mean RMSSD 86.5 ms, class 2
   15.0 ms. Same direction.
3. **The control cohort settles it.** Subjects PNSR-1, PNSR-3, PNSR-4 are
   healthy volunteers — "NSR" is in the folder name. They have **zero** class-1
   segments and 65 class-2 segments. A healthy control cannot be in AF.

Getting this backwards is the most dangerous bug possible here: the pipeline
runs perfectly, the metrics look plausible, and every prediction is inverted.

### 3.3 Segments are 10 seconds, not 5 minutes

Every row of every `annotation.csv` spans exactly 10,240 samples at 1024 Hz =
10 seconds. That is the dataset's own labelled unit. Merging segments into
5-minute windows would combine segments with conflicting labels.

**Consequence:** 10 seconds holds only ~10–15 beats. That is enough for
short-term HRV (RMSSD, pNN50, SD1, entropy) but **not** for frequency-domain
features. Resolving LF power at 0.04 Hz requires at least 25 seconds of signal.
`USE_FREQUENCY_DOMAIN` is therefore `False`, and the write-up must say so
rather than report meaningless LF/HF numbers.

### 3.4 The ectopic beat filter is OFF, deliberately

Standard HRV practice removes beats deviating more than ~20% from a local
median. That is correct for sinus rhythm and catastrophic for AF, because in AF
consecutive intervals genuinely do differ by more than 20% — that irregularity
*is* the diagnosis.

Measured damage when it was enabled:

| Class | Beats removed | Segments destroyed |
|-------|---------------|--------------------|
| AF    | 48.5%         | 25.7% (192 of 747) |
| NSR   | 15.4%         | 0.5% (3 of 615)    |

Worse, the deleted AF segments were **1.42× more irregular** than the survivors
— it was removing the most unambiguous AF. That is selection on the label, and
it inflates every reported metric.

Turning it off improved pooled F1 from 0.795 to 0.887 and AUC from 0.897 to
0.947, and collapsed per-fold F1 variance from ±0.332 to ±0.071.

Reproduce the evidence any time:
```powershell
.\venv\Scripts\python.exe tests\test_ectopic_filter_bias.py
```

### 3.5 Subject numbering has gaps, and there are two cohorts

Real folder names: P1–P19, P21, P23, PNSR-1, PNSR-3, PNSR-4. **There is no P20
or P22.** Never generate subject IDs with a loop. P16 exists but has no usable
labels — disclose this in the methodology.

PNSR-* are healthy controls. If one lands entirely in a test fold, that fold's
accuracy is inflated. This is why splitting is by **subject**, not by segment.

---

## 4. CURRENT RESULTS

Setup: 1381 segments, 23 subjects, 10 MRMR-selected features, subject-wise
GroupKFold (5 folds), SMOTE applied inside each training fold only.

| Metric | SVM | XGBoost |
|--------|-----|---------|
| Accuracy | 0.877 ± 0.087 | 0.887 ± 0.061 |
| F1 | 0.880 ± 0.071 | 0.891 ± 0.053 |
| AUC | 0.932 ± 0.072 | 0.923 ± 0.086 |
| Sensitivity | 0.847 | 0.884 |
| Specificity | 0.908 | 0.883 |

**Report pooled out-of-fold metrics, not the mean of per-fold metrics.** Folds
differ wildly in class composition, so averaging them is misleading.

**Selected features (MRMR, 10 of 20):** pNN50, MCVNN, CVNN, ShannonEn, pNN20,
IQRNN, MadNN, SD1, RMSSD, SDNN. AUC was unchanged versus using all 20 — every
survivor is a dispersion measure of RR intervals, and all fragmentation
features (PAS, PIP, IALS, PSS) were dropped. That redundancy is itself a
reportable result: a wearable needs far fewer features than the literature uses.

### 4.1 The main finding

| Population | False alarm rate |
|------------|------------------|
| Clean NSR segments | 8–11% |
| Motion-corrupted (Noise) segments | **78–84%** |
| Risk ratio | **7–10×** |

**Mechanism, verified:** noise segments look like AF to any HRV-based model.

| Class | RMSSD | CVNN | pNN50 |
|-------|-------|------|-------|
| NSR | 96.6 | 0.10 | 43.8 |
| **Noise** | **212.1** | **0.30** | **71.4** |
| AF | 240.8 | 0.24 | 77.8 |

On CVNN, noise is *more irregular than AF itself*. Corrupted signal breaks
R-peak detection, which produces erratic RR intervals, which is precisely the
AF signature. The classifier cannot separate artifact-induced irregularity from
fibrillation-induced irregularity because on every feature they are identical.

### 4.2 What NOT to claim

The movement hypothesis was **not** supported (Spearman rho = −0.066, p = 0.33;
quartile alarm rates flat at 78/76/85/73%).

Do **not** write "movement does not cause false positives". When 80% of
segments alarm, the outcome is nearly constant and no covariate can correlate
with it — the test is **saturated**, not informative. Write that the effect
could not be tested because the false-alarm rate was near-ceiling at all
movement levels.

The null result on clean NSR (chi-square p = 0.65, Mann-Whitney p = 0.14) is
useful as your **control condition**: on readable signal, context genuinely does
not matter; on unreadable signal, nothing else does.

---

## 5. HOW TO RUN THE PIPELINE

Always activate the virtual environment and set UTF-8 output first:

```powershell
cd C:\vaishnav\projects\Main_Project\AFib_Project
.\venv\Scripts\activate
$env:PYTHONIOENCODING="utf-8"
```

**Run everything:**
```powershell
.\venv\Scripts\python.exe main.py
```

**Run specific phases:**
```powershell
.\venv\Scripts\python.exe main.py --phases 4 5 6 7 8
```

| Phase | What it does | Time |
|-------|--------------|------|
| 1 | Build/validate the segment index | 15 s |
| 2 | Preprocessing demo figure | 2 s |
| 3 | HRV feature extraction (1602 segments) | 20 s |
| 4 | MRMR feature selection | 7 s |
| 5 | SVM training + cross-validation | 5 s |
| 6 | XGBoost training + cross-validation | 3 s |
| 7 | Evaluation figures | 2 s |
| 8 | Context-aware FP analysis (clean NSR) | 70 s |

**The Noise false-alarm analysis (separate module):**
```powershell
.\venv\Scripts\python.exe -m src.context_analysis.noise_fp_analysis
```

### 5.1 Caching gotchas — these will waste your time

Several phases skip work if output already exists. When you change a setting,
**delete the cached file or the change will not take effect**:

| Changed | Delete first |
|---------|--------------|
| Anything affecting features (filters, peak detection) | run `run_feature_pipeline(resume=False)` — see below |
| `N_FEATURES_TO_SELECT` or MRMR settings | `data\features\selected_features.json` |
| Dataset paths | `data\processed\segment_index.csv` |

Force feature re-extraction:
```powershell
.\venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from src.feature_extraction.feature_pipeline import run_feature_pipeline; run_feature_pipeline(resume=False)"
```

---

## 6. NEXT TASK — THE SIGNAL-QUALITY GATE

**Why:** Section 4.1 shows the detector false-alarms on ~80% of corrupted
segments. That diagnoses a problem. This experiment shows the problem is
**fixable** and quantifies the cost of fixing it — which turns a finding into an
engineering recommendation and makes the paper self-contained.

**The idea in one sentence:** before the classifier is allowed to make a call,
check whether the signal is good enough to be worth judging, and refuse to
answer if it is not.

**The trade-off being measured:** a strict gate rejects more noise (fewer false
alarms) but also rejects some real AF (lower sensitivity). Nobody can tell you
the right threshold without seeing that curve. Producing the curve is the work.

### Step by step

**Step 1 — Build a quality score.**
You already extract everything needed; no new signal processing required. Good
candidates already sitting in `hrv_features.csv`:
- `n_beats_detected` — implausibly high or low counts indicate bad detection
- `ectopic_fraction` — how many intervals violate physiological limits
- Beat-rate plausibility — does implied heart rate sit in a believable range?

Combine them into a single 0–1 score. Start simple; a weighted sum is fine and
is easier to defend than something opaque.

**Step 2 — Verify the score actually separates classes.**
Before using it, check that Noise segments score worse than AF and NSR
segments. If it does not separate them, the score is useless and you must fix
it before continuing. **Do not skip this check** — it is the difference between
an experiment and wishful thinking.

**Step 3 — Sweep the threshold.**
For thresholds from 0 (accept everything) to 1 (accept almost nothing), record:
- how many Noise segments were gated out
- how many AF segments were lost
- the false-alarm rate on the segments that survived
- sensitivity on the segments that survived

**Step 4 — Plot the operating curve.**
X-axis: AF sensitivity retained. Y-axis: false-alarm rate on noise. Each point
is one threshold. This is the figure a device manufacturer actually needs.

**Step 5 — State a recommended operating point.**
Something like: "rejecting the noisiest 20% of segments cuts the false-alarm
rate from 80% to X% at a cost of Y% sensitivity." Pick a defensible point and
justify the choice — for screening, sensitivity usually matters more than
precision, because a missed AF is a stroke risk and a false alarm is an
inconvenience.

**Expected output:** a new module `src/evaluation/quality_gate.py`, a figure in
`reports/figures/`, and a results table.

---

## 7. OUTSTANDING WORK & COMPLETED INVESTIGATIONS

1. ✅ **Fold 1 Investigation (COMPLETE)**:
   - **Root Cause Confirmed**: Patient **P1** accounts for 184 of 185 AF segments in Fold 1.
   - P1's AF is unusually organized (RMSSD = 124.4 ms vs 300.9 ms training average; delta = -176.0 ms).
   - Simultaneously, P1's NSR segments exhibit high respiratory sinus arrhythmia (RMSSD = 161.3 ms), exceeding P1's own AF segments.
   - This inverts the expected feature relationship, collapsing the RMSSD effect size (|rank-biserial r|) from 0.77+ to 0.059 in Fold 1.
   - **Artifacts Created**:
     - Script: `src/evaluation/fold1_analysis.py`
     - Figures: `reports/figures/fold1_subject_profiles.png`, `fold1_overlap_heatmap.png`, `fold1_auc_comparison.png`
     - Comprehensive PDF Report: Section 11 added with full discussion and figures.
2. ⏳ **External Validation**:
   - MIT-BIH AFDB is located in `Dataset/files` (23 records, 250 Hz).
   - Requires the raw dataset drive/pendrive to be mounted to extract signals and test cross-database generalization.
3. 📝 **Paper & Thesis Write-up**:
   - Everything needed for a high-impact paper is complete: reproducible 8-phase pipeline, 7x noise false-alarm empirical discovery, Pareto quality gate, and the Fold 1 cohort heterogeneity diagnostic.

### Things the methodology section must disclose

- P16 has no usable labels — 23 subjects, not 24
- 221 Noise segments excluded from training, retained for the FP analysis
- 19 "Other" arrhythmia segments grouped with non-AF (too few to model)
- Only 7 of 23 subjects contain **both** AF and NSR; most are almost entirely
  one class, which is why fold composition varies so much
- Frequency-domain HRV excluded — the 10-second window cannot resolve LF
- 33 overlapping segments exist in the index (consecutive annotated runs)

---

## 8. PROJECT LAYOUT

```
AFib_Project/
├── config/config.py                      ALL settings. Read Section 3 first.
├── src/
│   ├── preprocessing/
│   │   ├── unisens_reader.py             raw .bin -> millivolts
│   │   ├── dataset_index.py              builds segment_index.csv
│   │   └── ecg_filter.py                 bandpass, notch, downsample
│   ├── peak_detection/rpeak_detector.py  R-peaks -> RR intervals
│   ├── feature_extraction/
│   │   ├── hrv_features.py               the 20 HRV features
│   │   └── feature_pipeline.py           runs it over all segments
│   ├── feature_selection/mrmr_selector.py
│   ├── models/train_svm.py, train_xgboost.py
│   ├── evaluation/metrics.py
│   └── context_analysis/
│       ├── fp_analysis.py                FP analysis on clean NSR
│       └── noise_fp_analysis.py          FP analysis on Noise  <- main finding
├── tests/test_ectopic_filter_bias.py     reproduces the Section 3.4 evidence
├── notebooks/Phase04_Dataset_EDA.py      EDA on the real data
├── data/
│   ├── processed/segment_index.csv       one row per labelled segment
│   └── features/hrv_features.csv         one row per segment, 20 features
├── models/                               trained .joblib files
├── reports/figures/                      all figures
└── main.py                               orchestrator, --phases 1-8
```

Dataset lives **outside** the project at `Main_Project/Dataset/` and is
referenced by path — not copied — to avoid duplicating ~10 GB.

---

## 9. IF SOMETHING BREAKS

| Symptom | Cause |
|---------|-------|
| `KeyError: 'subject'` | Merging a frame with `hrv_features.csv`, which has its own `subject` column. Drop overlapping columns before merging. |
| Changes have no effect | Cached output. See Section 5.1. |
| Unicode errors in PowerShell | Run `$env:PYTHONIOENCODING="utf-8"` |
| Metrics suspiciously perfect | Check for data leakage — splits must be by subject |
| Metrics suspiciously inverted | Check the class mapping. See Section 3.2. |
| Repeated openpyxl style warnings | Harmless. The context.xlsx files lack a default style block. |
