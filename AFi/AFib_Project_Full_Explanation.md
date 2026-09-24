# Comprehensive Research Explanation: HRV Feature Extraction and Context-Aware False Positive Analysis for Atrial Fibrillation Detection Using CACHET-CADB

**B.Tech Final Year Main Project — Complete Technical Reference**

---

## 1. Project Overview

This project builds an end-to-end machine learning pipeline that detects Atrial Fibrillation (AF) from raw electrocardiogram (ECG) signals recorded in real-world, free-living conditions. What sets it apart from typical AF detection studies is a second, equally important goal: it systematically investigates **why** AF detectors produce false alarms by correlating misclassifications with the patient's physical activity, body posture, movement intensity, and stress level at the moment of each error.

The project uses the **CACHET-CADB** (Context-Annotated Cardiac Database), a unique dataset from the Technical University of Denmark that provides both raw ECG recordings and synchronized contextual metadata from 24 patients over a combined 259 days of ambulatory monitoring. This database is the only publicly available ECG dataset that pairs every 10-second ECG segment with information about what the patient was physically doing at that exact moment.

The pipeline is divided into eight sequential stages spread across two semesters (8 months), progressing from raw signal processing through classification and finally to a novel context-correlation analysis that no published paper has fully explored on this specific dataset.

### What This Project Does NOT Do

It is important to note that this project works entirely with **numerical time-series data and computed features**. There is no ECG-to-image conversion involved. Students work with raw ECG signals (numerical arrays), RR intervals (numbers), HRV features (calculated statistics), and context labels. The classifiers used are traditional machine learning models (SVM and XGBoost), not deep learning or convolutional neural networks. ECG-to-image conversion and CNN/BiLSTM approaches are a separate, more advanced line of research.

---

## 2. Clinical Background: Why This Matters

### 2.1 What Is Atrial Fibrillation?

Atrial Fibrillation (AF) is the most common sustained cardiac arrhythmia worldwide, affecting an estimated 33.5 million people globally. In AF, the heart's upper chambers (atria) fire electrical impulses chaotically at rates of 350 to 600 times per minute, instead of the normal coordinated contraction. The atrioventricular (AV) node randomly allows some of these impulses through to the ventricles, producing an **irregularly irregular** heartbeat — the defining clinical signature of AF.

AF is dangerous primarily because it increases the risk of stroke by approximately 5-fold. Blood pools in the fibrillating atria, forming clots that can travel to the brain. AF is also associated with heart failure, reduced quality of life, and increased mortality.

There are three clinical subtypes:

- **Paroxysmal AF (PAF):** Episodes that start and stop spontaneously, typically within 7 days. This is the hardest to detect because it may not be present during a brief clinical ECG recording.
- **Persistent AF:** Episodes lasting longer than 7 days, requiring medical intervention (cardioversion or medication) to restore normal rhythm.
- **Permanent AF:** Continuous AF where rhythm control has been abandoned and the clinical strategy shifts to rate control and anticoagulation.

The detection challenge lies in paroxysmal AF. Because it comes and goes unpredictably, a standard 12-lead ECG recorded during a brief hospital visit often misses it entirely. This is why wearable, ambulatory ECG monitors that record continuously over days or weeks are increasingly important — and why automated detection algorithms are needed to sift through the massive volume of recorded data.

### 2.2 How AF Appears in an ECG Signal

A normal ECG heartbeat cycle consists of three main components:

- **P wave** — Atrial depolarization: the atria contracting in a coordinated manner. This appears as a small, rounded bump before the main spike.
- **QRS complex** — Ventricular depolarization: the sharp spike representing the main heartbeat contraction. The R-peak is the tallest point of this complex.
- **T wave** — Ventricular repolarization: the ventricles resetting electrically. This appears as a broad, gentle bump after the QRS complex.

In normal sinus rhythm (NSR), the P wave is clearly visible before each QRS complex, and the intervals between consecutive R-peaks (the RR interval) are relatively regular — varying slightly with breathing and autonomic nervous system activity, but following a predictable pattern.

In AF, two characteristic changes occur:

1. **The P wave disappears entirely**, replaced by low-amplitude, chaotic fibrillatory (f) waves because the atria are firing randomly rather than contracting in an organized way.
2. **The RR intervals become irregularly irregular** — there is no discernible pattern to the spacing between beats. The AV node randomly allows through some of the 350–600 atrial impulses per minute, so the ventricles contract at seemingly random intervals.

It is this second feature — the chaotic RR interval pattern — that Heart Rate Variability (HRV) features are designed to capture and quantify. This is the mathematical foundation of the entire project.

### 2.3 The False Positive Problem in Wearable Monitoring

When AF detection algorithms are deployed on wearable devices in real-world settings, they face a critical reliability problem: **false positives**. A false positive occurs when the algorithm flags a segment as AF when the patient actually has a normal rhythm.

Research has shown that only about 34% of irregular pulse alerts on smartwatches correspond to confirmed AF on follow-up ECG monitoring (Apple Heart Study, Perez et al., 2019, *New England Journal of Medicine*). The primary culprits are:

- **Motion artifacts:** When a patient walks, jogs, or cycles, body movement introduces electrical noise into the ECG signal that can distort the RR interval pattern, making it mimic the irregularity characteristic of AF.
- **Ectopic beats:** Premature atrial contractions (PACs) and premature ventricular contractions (PVCs) produce irregular intervals that resemble AF-induced irregularity on algorithmic analysis.
- **Poor sensor contact:** In ambulatory settings, electrodes may lose contact with the skin, producing voltage spikes and signal dropouts.

This is not merely an academic problem — false alerts cause patient anxiety, unnecessary emergency visits, inappropriate anticoagulation therapy, healthcare resource waste, and erosion of physician trust in digital health tools. One study found that false AF alerts from smartwatches are associated with decreased perceived physical well-being and confidence in chronic symptom management.

Understanding **which specific activities and conditions** trigger these false positives is the unique research contribution of this project. By using CACHET-CADB's synchronized context data, the project can answer questions no published study has systematically answered using this specific dataset.

---

## 3. The CACHET-CADB Database

### 3.1 What Makes It Unique

CACHET-CADB (Context-Annotated Cardiac Database) was published by Kumar, Puthusserypady, Dominguez, Sharma, and Bardram from the Department of Health Technology at the Technical University of Denmark (DTU) in *Frontiers in Cardiovascular Medicine* (July 2022). It is a 259-day-long contextualized single-channel ECG arrhythmia database recorded from 24 patients under free-living ambulatory conditions using a Movisens EcgMove4 chest-worn sensor.

What makes CACHET-CADB fundamentally different from other ECG databases is the **contextual metadata**. Other widely-used databases — MIT-BIH Arrhythmia Database (MIT-BIH AFDB), CPSC 2021, PhysioNet Challenge datasets — provide raw ECG signals with rhythm annotations, but no information about what the patient was doing when the recording was made. Most of these databases were recorded in controlled clinical settings (hospitals, monitoring labs) where patients are relatively still.

CACHET-CADB provides, for every 10-second window of ECG data:

- **ActivityClass** (0–11): sitting, walking, jogging, cycling, lying, standing, etc.
- **BodyPosition** (0–7): supine, prone, left side, right side, upright, standing, etc.
- **Movement Acceleration Index (MAI):** a continuous measure of body movement intensity in g-force units.
- **Step Count:** number of steps in the 10-second window.
- **Stress Level (1–5):** patient self-reported.
- **Sleep Quality (1–5):** patient self-reported.
- **Food Intake and Symptom Diary:** patient-recorded unusual events.

This contextual data is the key that enables the false-positive correlation analysis that forms the project's most original contribution.

### 3.2 Dataset Statistics

| Parameter | Value |
|---|---|
| Number of patients | 24 |
| Total recording duration | 259 days (24 hours to 3 weeks per patient) |
| ECG sampling frequency | 1024 Hz, 12-bit resolution |
| Total annotated segments | 1,602 (each 10 seconds long) |
| AF segments | 747 (46.6%) |
| NSR segments | 615 (38.4%) |
| Noise segments | 221 (13.8%) |
| Other rhythm segments | 19 (1.2%) |
| Noisy data proportion | ~11% of total ECG data |
| Annotation method | Two independent qualified cardiologists |

### 3.3 File Structure

Each patient folder in CACHET-CADB contains:

**Signal folder (raw data):**

- `ecg.bin` — Raw ECG in binary format (NOT directly readable as CSV)
- `acc.bin` — Accelerometer data
- `angularrate.bin` — Gyroscope data
- `press.bin` — Pressure sensor data
- `unisens.xml` — Metadata (patient demographics, recording start timestamp, sampling frequency)

**Annotation folder:**

- `context.xlsx` — Context data every 10 seconds (activity, position, MAI, stress, etc.)
- `annotation.csv` — ECG rhythm labels (AF=1, NSR=2, Noise=3, Others=4) with start and end indices

The ECG data is stored in **Unisens binary format**, requiring the `pyunisens` Python library to parse using metadata from `unisens.xml`. This is a non-trivial data engineering step that requires understanding the binary structure, sample encoding, and timestamp alignment.

### 3.4 Class Imbalance Challenge

The class distribution is moderately imbalanced. AF and NSR are reasonably balanced, but the Noise class is smaller, and the Others class has only 19 samples — too few for reliable training. This imbalance must be handled explicitly (using SMOTE or class weighting) to prevent the classifier from ignoring minority classes.

---

## 4. The Complete Pipeline: Stage-by-Stage Explanation

### Stage 1: ECG Preprocessing (Month 3)

#### 4.1.1 Why Preprocessing Is Necessary

Raw ECG signals from ambulatory recordings contain substantial noise from three main sources:

**Baseline wander** is a slow, undulating drift in the signal baseline caused by breathing and electrode movement, typically below 0.5 Hz. The entire signal appears to slowly rise and fall, making it impossible to establish a consistent baseline for peak detection.

**High-frequency noise** comes from electrical interference (power line at 50/60 Hz) and muscle contractions (electromyographic noise). This appears as rapid, fine oscillations superimposed on the ECG waveform.

**Motion artifacts** are transient distortions caused by body movement, electrode shifting, or device contact issues. In CACHET-CADB's free-living data, about 11% of recordings are classified as noisy, and even "clean" segments contain varying degrees of artifact. Motion artifacts are particularly insidious because they can mimic the appearance of cardiac events.

#### 4.1.2 Bandpass Filtering

The solution is a **Butterworth bandpass filter** with cutoff frequencies of 0.5 Hz (high-pass, to remove baseline wander) and 50 Hz (low-pass, to remove high-frequency noise).

**Why Butterworth?** A 4th-order Butterworth filter provides a "maximally flat" passband response — it preserves signal amplitude without distortion or ripple within the pass band. Other filter types (Chebyshev, Elliptic) achieve sharper cutoffs but introduce amplitude ripples that can distort ECG morphology.

**Why filtfilt()?** The `scipy.signal.filtfilt()` function applies the filter in both forward and reverse directions, producing **zero phase distortion**. This is critical because phase shifts would move the apparent position of R-peaks in time, corrupting RR interval measurements. A forward-only filter introduces a phase delay proportional to frequency, which would distort the temporal relationships between ECG components.

**Implementation:**

```python
from scipy.signal import butter, filtfilt

def bandpass_filter(ecg, fs=1024, low=0.5, high=50, order=4):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, ecg)
```

After bandpass filtering, an optional **Savitzky-Golay smoothing** step can further reduce residual noise while preserving peak shapes. This fits a polynomial within a sliding window (window_length=11, polyorder=3) and replaces each point with the polynomial value.

#### 4.1.3 Validation

Every preprocessing step must be visually validated by plotting raw versus filtered ECG side-by-side. The key checks: Has baseline wander been eliminated? Are QRS peaks still sharp and clearly defined? Does the filter work on both clean and noisy segments?

### Stage 2: R-Peak Detection — Pan-Tompkins Algorithm (Month 3)

#### 4.2.1 What Are R-Peaks?

The R-peak is the tallest, sharpest point of the QRS complex in each heartbeat cycle. It represents the moment of maximum ventricular depolarization. Detecting R-peaks is the foundational step for all subsequent analysis because the time between consecutive R-peaks (the **RR interval**) is the raw material from which all HRV features are computed.

Accurate R-peak detection is critical — a single missed or falsely detected peak corrupts two adjacent RR intervals, which cascades into errors in every feature computed from those intervals.

#### 4.2.2 The Pan-Tompkins Algorithm

Published by Jiapu Pan and Willis Tompkins in 1985, this remains the most widely referenced algorithm for real-time QRS complex detection. Despite being nearly 40 years old, it is still the benchmark against which newer methods are compared. It works in five stages:

**Step 1 — Differentiation:** The filtered ECG is differentiated (first derivative calculated). This accentuates the steep slopes of the QRS complex while suppressing the gentler slopes of P and T waves. The QRS complex has the steepest slope of any ECG component because ventricular depolarization is the most rapid electrical event in the cardiac cycle. A five-point derivative approximation is used for numerical stability.

**Step 2 — Squaring:** Each differentiated value is squared. This serves two purposes: (a) it makes all values positive (the downstroke of the QRS would otherwise produce negative values), and (b) it nonlinearly amplifies larger peaks relative to smaller ones. Because squaring is a nonlinear operation, a peak that is twice as tall in the derivative becomes four times as prominent after squaring. This helps separate QRS complexes from T waves.

**Step 3 — Moving Window Integration:** A 150-millisecond sliding window averages the squared signal. This produces a smooth envelope around each QRS complex, converting the multi-peaked QRS morphology into a single smooth hump. The 150ms window width is critical: too wide and adjacent QRS and T wave humps merge into one; too narrow and a single QRS complex produces multiple separate peaks.

**Step 4 — Adaptive Thresholding:** The algorithm maintains two dynamic thresholds that adapt as the signal changes. Peaks above the "signal" threshold are classified as QRS complexes; peaks between the signal and "noise" thresholds are classified as noise. After each detected peak, both thresholds are updated using exponential moving averages, allowing the algorithm to track changes in signal amplitude (e.g., when the patient changes position and electrode contact quality shifts).

**Step 5 — Search-Back and Artifact Rules:** Several heuristic rules handle edge cases:

- If no QRS is detected within 166% of the average RR interval (~1.83 seconds at normal heart rate), the algorithm searches back through the noise-classified peaks for a missed QRS.
- If a detected RR interval is shorter than (mean - 0.8×SD), the detection is likely a false positive (perhaps a T wave mis-identified as a QRS) and is rejected.
- Intervals deviating from the mean by more than ±1.9 standard deviations are flagged as artifacts.

**Practical implementation** uses the `neurokit2` library:

```python
import neurokit2 as nk
signals, info = nk.ecg_process(ecg_filtered, sampling_rate=1024)
r_peaks = info['ECG_R_Peaks']
```

### Stage 3: RR Interval Extraction and Cleaning (Month 3)

Once R-peaks are detected, RR intervals are calculated:

```python
rr_samples = np.diff(r_peaks)
rr_ms = rr_samples * (1000 / 1024)  # Convert to milliseconds
```

Three cleaning steps follow:

**Physiological bounds:** RR intervals below 300ms (heart rate >200 bpm, physiologically impossible in most contexts) or above 2000ms (heart rate <30 bpm, extremely unlikely) are removed.

**Ectopic beat removal:** If an RR interval differs from the local mean by more than 20%, it likely represents a premature atrial contraction (PAC) or premature ventricular contraction (PVC) rather than a normal or AF beat. These are removed because ectopic beats introduce variability that is unrelated to the underlying rhythm.

**Minimum count threshold:** For each annotated 10-second ECG segment, all clean RR intervals within that window are extracted. A typical 10-second window at normal heart rate contains 8 to 15 RR intervals. Segments with fewer than 5 clean RR intervals are excluded because too few data points produce unreliable HRV statistics.

---

### Stage 4: HRV Feature Extraction (Month 4)

Heart Rate Variability (HRV) is the variation in duration between consecutive cardiac cycles. In healthy individuals, RR intervals vary slightly and predictably, modulated by the autonomic nervous system (breathing causes a regular speeding and slowing called respiratory sinus arrhythmia). In AF, RR intervals vary massively and randomly because the chaotic atrial firing produces unpredictable ventricular responses.

HRV features are mathematical tools that quantify different aspects of this variability. The project extracts 14–16 features from four categories:

#### 4.4.1 Time-Domain Features

These are computed directly from the RR interval values:

**MeanNN** — The arithmetic mean of all RR intervals. Provides a basic measure of heart rate (MeanNN of 800ms = 75 bpm) but is unreliable as an AF indicator alone because AF can occur at any average heart rate.

**SDNN (Standard Deviation of NN intervals)** — Measures overall spread of RR intervals around their mean. Formula: `SDNN = sqrt(mean((RR - mean(RR))^2))`. Higher SDNN indicates greater variability. In AF, SDNN is significantly elevated because beats are spaced randomly. Normal range: 50–100ms.

**RMSSD (Root Mean Square of Successive Differences)** — The **most important single HRV feature** for AF detection. Formula: `RMSSD = sqrt(mean(diff(RR)^2))`. Unlike SDNN which measures total spread, RMSSD specifically captures **beat-to-beat** irregularity — the hallmark of AF. In NSR, consecutive beats are similar (low RMSSD). In AF, each beat is wildly different from the next (high RMSSD).

**SDSD** — Standard deviation of successive RR interval differences. Closely related to RMSSD but measures the spread of the differences rather than their root mean square.

**pNN50** — The percentage of consecutive RR intervals differing by more than 50ms. In NSR: typically 10–20%. In AF: typically 50–80%. A simple, powerful AF indicator.

**pNN20** — Same threshold lowered to 20ms. More sensitive, catches subtler irregularities. Useful for detecting early or mild AF.

**MCVNN (Median Coefficient of Variation)** — Standard deviation divided by median of RR intervals. Normalizes variability by heart rate. Hasan & Motin (2025) found this to be the **single most important feature** for AF detection (XGBoost importance: 0.37). Uses median instead of mean for robustness to outliers.

**MadNN (Median Absolute Deviation)** — Median of absolute deviations from the median RR value. More robust than SDNN against motion artifact outliers. Second most important feature (importance: 0.23).

**IQRNN (Interquartile Range)** — Difference between 75th and 25th percentiles. In NSR: small IQR (tight clustering). In AF: large IQR (wide spread). Third most important feature (importance: 0.06).

**HTI (HRV Triangular Index)** — Total number of RR intervals divided by histogram peak height. NSR produces a tall narrow histogram (high HTI); AF produces a short wide histogram (low HTI).

#### 4.4.2 Entropy Features

**Sample Entropy (SampEn)** — Measures unpredictability of the RR sequence. The algorithm: (1) take patterns of m consecutive values (m=2), (2) count how often each pattern repeats elsewhere within tolerance r=0.2×SD, (3) calculate how much repeatability drops when pattern length increases to m+1. High SampEn = patterns never repeat = unpredictable = AF. Low SampEn = patterns recur = structured = NSR.

Important caveat: SampEn becomes unreliable for very short sequences (<60 beats). **CosEn** (Coefficient of Sample Entropy) is a modified version designed for short recordings, accounting for window length and normalizing by heart rate. For 10-second segments from CACHET-CADB, CosEn may be more appropriate.

**Shannon Entropy** — Measures uncertainty in the probability distribution of RR values. Build a histogram, compute bin probabilities, apply H = -Σ p(x) log₂ p(x). NSR intervals cluster narrowly (low uncertainty). AF intervals spread widely (high uncertainty).

#### 4.4.3 Nonlinear Features (Poincaré Plot)

A Poincaré plot graphs each RR interval against the next: RR(n) on x-axis, RR(n+1) on y-axis. The resulting scatter forms an ellipse:

**SD1** — Width (minor axis) of the ellipse. Measures short-term, beat-to-beat variability. Formula: `SD1 = sqrt(Var(diff(RR)) / 2)`. In AF: large, diffuse cloud (high SD1). In NSR: compact cluster (low SD1).

**SD2** — Length (major axis) of the ellipse. Measures longer-term variability. Formula: `SD2 = sqrt(2*Var(RR) - Var(diff(RR))/2)`. Both SD1 and SD2 are elevated in AF.

The visual difference is striking: a normal Poincaré plot shows a tight elliptical cluster along the identity line (RR(n) ≈ RR(n+1) because consecutive beats are similar), while an AF Poincaré plot shows a large, shapeless cloud scattered across the plot (consecutive beats bear no relationship to each other).

#### 4.4.4 Fragmentation Features

Computed using `neurokit2.hrv_fragmentation()`:

**PIP (Percentage of Inflection Points)** — How often the RR series changes direction (increasing to decreasing or vice versa). High PIP = fragmented, chaotic rhythm = AF.

**IALS (Inverse of Average Length of Acceleration/Deceleration Segments)** — How short the runs of consistently increasing or decreasing RR intervals are. Short runs = more fragmentation = AF.

**PSS (Percentage of Short Segments)** — Fraction of RR intervals in very short acceleration/deceleration runs.

**PAS (Percentage of Alternating Segments)** — How often RR intervals alternate between increasing and decreasing.

#### 4.4.5 Feature Validation

After computing all features, each must be validated:

- **Direction check:** AF vs NSR box plots must show differences in the expected direction (RMSSD higher in AF, SampEn higher in AF, etc.). If any feature shows an unexpected direction, there is a bug.
- **Statistical significance:** Mann-Whitney U test between AF and NSR distributions. Features with p > 0.05 may not be useful.
- **Noise differentiation:** Three-way box plot (AF vs NSR vs Noise) verifies that noise segments produce distinct values.

The final output is a **feature matrix**: 1,602 rows × 14–16 feature columns + class label + context columns.

---

### Stage 5: mRMR Feature Selection and Classification (Month 5)

#### 4.5.1 mRMR Feature Selection

With 14–16 features, some carry redundant information (e.g., SDNN and RMSSD both measure variability). **mRMR (Minimum Redundancy Maximum Relevance)** selects features that are maximally relevant to the class label while being minimally redundant with each other. This is superior to simpler methods like correlation-based selection because it explicitly penalizes redundancy.

```python
from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X, y=y, K=10)
```

Published benchmarks suggest 5 features may be optimal (Islam & Motin, 2023). The project tests performance with top 5, 8, 10, and all features to find the optimal count.

#### 4.5.2 SMOTE for Class Imbalance

**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic samples for underrepresented classes by interpolating between existing minority samples in feature space. Critical rule: SMOTE is applied **only to training data**, never test data, to prevent information leakage.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```

#### 4.5.3 SVM Classifier

Support Vector Machine with RBF kernel, using a Pipeline for feature standardization:

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1, probability=True, random_state=42))
])
```

**Subject-wise 5-fold cross-validation** (GroupKFold with patient IDs) ensures no patient's data appears in both training and test sets — preventing data leakage that would artificially inflate accuracy. This is non-negotiable for clinical ML evaluation.

#### 4.5.4 XGBoost Classifier

```python
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                     subsample=0.8, eval_metric='mlogloss', random_state=42)
```

XGBoost provides built-in feature importance scores based on split frequency and gain. These rankings are compared with mRMR for consistency.

---

### Stage 6: Context-False Positive Correlation Analysis (Month 6)

**This is the most original and unique contribution of the project.**

#### 4.6.1 Identifying False Positives

All test segments where the model predicted AF but the true label was not AF are collected. For each, the corresponding context metadata is retrieved:

```python
false_positives = []
for i in range(len(y_true)):
    if y_pred[i] == 'AF' and y_true[i] != 'AF':
        false_positives.append(i)
```

#### 4.6.2 Activity-Wise FPR Analysis

The central hypothesis: False Positive Rate (FPR = FP/(FP+TN)) differs significantly across activity classes. Expected pattern:

| Activity | Expected FPR | Reason |
|---|---|---|
| Jogging | HIGH | Vigorous movement generates motion artifacts mimicking AF irregularity |
| Cycling | HIGH | Rhythmic upper-body movement distorts ECG |
| Walking | MODERATE | Moderate movement, some artifact |
| Sitting | LOW | Minimal movement |
| Lying | VERY LOW | No movement, clean signal |
| Sleeping | VERY LOW | No movement, physiologically stable |

A **chi-square test** determines statistical significance. If p < 0.05, activity significantly affects FPR — proving context matters for AF detection.

#### 4.6.3 MAI Threshold Analysis

The Movement Acceleration Index (MAI) analysis is particularly valuable. By plotting FPR against continuous MAI values, the project identifies a **threshold** above which false positives spike. This is found using ROC analysis.

Example finding: *"When MAI exceeds 0.8g, false positive rate increases by 65% compared to segments where MAI is below 0.3g."*

This is an original, quantifiable finding specific to this project.

#### 4.6.4 Stress and Sleep Correlation

Additional analyses examine whether self-reported stress level and sleep quality correlate with FPR. Stress affects autonomic HRV independently of rhythm, potentially causing false AF detection. These analyses demonstrate the contextual richness of CACHET-CADB.

---

### Stage 7: Evaluation and Comparison (Month 7)

#### 4.7.1 Metrics

For each classifier, per-class metrics are reported:

- **Sensitivity** (Se = TP/(TP+FN)) — Did we catch all AF episodes?
- **Specificity** (Sp = TN/(TN+FP)) — Did we avoid false alarms?
- **Precision** (Pr = TP/(TP+FP)) — Of our AF predictions, how many were correct?
- **F1 Score** — Harmonic mean of precision and recall
- **Balanced Accuracy** — (Se+Sp)/2
- **AUC-ROC** — Area under Receiver Operating Characteristic curve (per class, one-vs-rest)

The emphasis on sensitivity and specificity rather than raw accuracy prevents the **accuracy paradox**: in an imbalanced dataset, a classifier that always predicts the majority class achieves high accuracy while being clinically useless.

#### 4.7.2 Cross-Paper Comparison

| Study | Dataset | Accuracy | Sensitivity | F1 |
|---|---|---|---|---|
| Andersen et al. (2017) | MIT-BIH AFDB | — | 96.81% | — |
| Islam & Motin (2023) | CPSC 2021 | 92.34% | 88.05% | 0.91 |
| Hasan & Motin (2025) | CPSC 2021 | 95.4% | — | 0.956 |
| This Project (SVM) | CACHET-CADB | TBD | TBD | TBD |
| This Project (XGBoost) | CACHET-CADB | TBD | TBD | TBD |

The expected finding: performance on CACHET-CADB will be lower than on MIT-BIH or CPSC 2021 because free-living ambulatory data is inherently noisier and more challenging than controlled clinical recordings. **This performance gap is itself a meaningful finding** — it quantifies how much real-world conditions degrade AF detection.

#### 4.7.3 Failure Mode Analysis

Five specific false positive cases are documented with: the ECG waveform plot, the context at that moment, the HRV feature values that led to the wrong prediction, and an explanation of why the model was fooled. Example:

> *"False Positive Case 3: ECG segment at timestamp 14:32:10. Patient was jogging (ActivityClass=5, MAI=1.8g). Model predicted AF. True label: NSR. Reason: Jogging caused rapid, slightly irregular RR intervals. RMSSD=87ms and SampEn=1.23 were both elevated to levels similar to AF. Context (jogging) could have prevented this false positive."*

---

### Stage 8: Report, Dashboard, and Defense (Month 8)

#### Dashboard

An interactive Plotly dashboard with 10 panels:

1. FPR vs Activity Class (bar chart)
2. FPR vs Body Position (bar chart)
3. FPR vs MAI (scatter with trend line)
4. Confusion matrix heatmap
5. ROC curves for SVM and XGBoost
6. Feature importance (mRMR and XGBoost)
7. Class distribution (before and after SMOTE)
8. Example ECG: true positive AF detection
9. Example ECG: false positive (motion artifact)
10. HRV feature box plots: AF vs NSR vs Noise

#### Report Structure

The final report (60–80 pages) follows standard B.Tech thesis format: Introduction, Literature Review, Methodology, Results and Analysis, Discussion and Conclusion, References, Appendices.

---

## 5. Month-by-Month Timeline Summary

| Month | Semester | Focus | Difficulty | Key Deliverable |
|---|---|---|---|---|
| 1 | S7 | Theory study + setup | Moderate | Concepts report; Python env |
| 2 | S7 | Data loading | HIGH | Working data loader; visualizations |
| 3 | S7 | Preprocessing | HIGH | Filter + R-peak + RR extraction |
| 4 | S7 | Feature extraction | HIGH | Feature matrix; validation plots |
| 5 | S8 | mRMR + Classification | Moderate-High | Trained models; comparison table |
| 6 | S8 | Context-FP analysis | Moderate | FPR analysis; MAI threshold |
| 7 | S8 | Evaluation | Moderate | Metrics; cross-paper comparison |
| 8 | S8 | Report + viva | Varies | 60-80 page report; demo; defense |

---

## 6. Why This Project Is Rigorous Despite AI Availability

The document explicitly addresses the concern that AI tools make the project too easy. The key argument: AI can generate boilerplate code, but it **cannot** do the actual research work:

- **Real data engineering:** CACHET-CADB's Unisens binary format breaks AI-generated code that assumes clean CSV inputs.
- **Signal quality debugging:** Real ambulatory ECG contains artifacts that require domain understanding to diagnose and fix.
- **Physiological validation:** Checking that computed features match known clinical expectations requires medical knowledge.
- **Methodological understanding:** Why subject-wise cross-validation, why SMOTE only on training data, why sensitivity over accuracy — these require conceptual understanding.
- **Original analytical thinking:** The context-correlation analysis requires hypothesis formulation, statistical design, and coherent interpretation.
- **Viva defense:** Oral examination tests understanding that cannot be outsourced.

---

## 7. Required Tools and Libraries

| Library | Purpose |
|---|---|
| numpy, pandas | Numerical computation and data manipulation |
| matplotlib, seaborn | Static visualization and box plots |
| scipy | Signal filtering (Butterworth), statistical tests |
| pyunisens | Reading CACHET-CADB Unisens binary format |
| neurokit2 | ECG processing, R-peak detection, HRV features |
| scikit-learn | SVM, StandardScaler, GroupKFold, metrics |
| xgboost | XGBoost classifier |
| imbalanced-learn | SMOTE for class balancing |
| mrmr-selection | mRMR feature selection algorithm |
| plotly | Interactive visualization dashboard |
| wfdb | PhysioNet database access (for MIT-BIH comparison) |

---

## 8. Key References

[1] Kumar D, Puthusserypady S, Dominguez H, Sharma K, Bardram JE. "CACHET-CADB: A Contextualized Ambulatory Electrocardiography Arrhythmia Dataset." *Frontiers in Cardiovascular Medicine*, 9:893090, July 2022.

[2] Pan J, Tompkins WJ. "A Real-Time QRS Detection Algorithm." *IEEE Transactions on Biomedical Engineering*, BME-32(3):230–236, March 1985.

[3] Andersen RS, Poulsen ES, Puthusserypady S. "A Novel Approach for Automatic Detection of Atrial Fibrillation Based on Inter Beat Intervals and Support Vector Machine." *Proc. IEEE EMBC*, 2017:2039–2042, 2017.

[4] Islam MS, Motin MA. "Short-term atrial fibrillation detection using electrocardiograms: A comparison of machine learning approaches." *International Journal of Medical Informatics*, 2023.

[5] Hasan MI, Motin MA. "HRV Feature Extraction and AF Detection Using XGBoost." 2025.

[6] Michel PO et al. "Using Minimum Redundancy Maximum Relevance Algorithm to Select Minimal Sets of Heart Rate Variability Parameters for Atrial Fibrillation Detection." *Journal of Clinical Medicine*, 11(14):4004, 2022.

[7] Richman JS, Moorman JR. "Physiological time-series analysis using approximate entropy and sample entropy." *American Journal of Physiology*, 278(6):H2039–H2049, 2000.

[8] Perez MV et al. "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation." *New England Journal of Medicine*, 381:1909–1917, 2019.

---

## 9. Summary

This project produces: a working ECG preprocessing pipeline, a validated 14–16 feature HRV extraction tool, trained SVM and XGBoost classifiers with subject-wise cross-validation, a complete context-false positive correlation analysis with statistical tests and MAI threshold identification, an interactive 10-panel dashboard, cross-paper comparison with three published studies, five documented failure case analyses, and a comprehensive project report.

Its distinguishing strength is that it does not just classify AF — it explains **why** the classifier fails under specific real-world conditions. This transforms it from a standard classification exercise into a genuine research contribution bridging signal processing, machine learning, and clinical context understanding.
