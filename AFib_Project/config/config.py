"""
=============================================================================
FILE: config/config.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis
         for Atrial Fibrillation Detection using CACHET-CADB Dataset

PYTHON VERSION: 3.13.7  (Required)
=============================================================================

PURPOSE:
    This is the central configuration file for the entire project.
    Think of it like the "settings panel" of your application.
    Instead of hardcoding numbers (like sampling frequency = 256) 
    inside every file, we define them ONCE here.
    
    WHY THIS IS IMPORTANT:
    - If you need to change a value, you only change it in ONE place.
    - All other files automatically use the updated value.
    - This prevents bugs caused by inconsistent values.
    - This is called the "Single Source of Truth" principle.

AUTHOR: [Your Name]
DATE: 2026
INSTITUTION: [Your College Name]
SUPERVISOR: [Your Guide Name]
=============================================================================
"""

import os
import sys  # For runtime Python version checking

# =============================================================================
# PYTHON VERSION REQUIREMENT
# =============================================================================
# This project is developed and tested on Python 3.13.7.
# We enforce this at runtime so you get a clear error immediately
# instead of a confusing crash later inside scientific code.

PYTHON_REQUIRED = (3, 10)      # (major, minor) - hard MINIMUM
PYTHON_PREFERRED = (3, 13, 7)  # The version this project was developed on

# sys.version_info gives the current interpreter version as a tuple.
# Example: Python 3.13.7 → sys.version_info = (3, 13, 7, 'final', 0)
#
# WHY A WARNING AND NOT A CRASH?
#     The original version of this file raised an error on anything below
#     3.13.7. That makes the project impossible to run on a lab machine,
#     a college server, or a grader's laptop with 3.11 installed - even
#     though every library we use works fine there. A hard minimum plus a
#     warning is the honest engineering trade-off.
if sys.version_info[:2] < PYTHON_REQUIRED:
    raise RuntimeError(
        f"\n{'='*60}\n"
        f"  PYTHON VERSION TOO OLD!\n"
        f"{'='*60}\n"
        f"  Minimum  : Python {'.'.join(map(str, PYTHON_REQUIRED))}\n"
        f"  Detected : Python {sys.version.split()[0]}\n"
        f"\n"
        f"  How to fix:\n"
        f"  1. Install Python 3.13.7 (recommended)\n"
        f"  2. Recreate venv: python -m venv venv\n"
        f"  3. Activate: .\\venv\\Scripts\\activate\n"
        f"{'='*60}"
    )
elif sys.version_info[:3] < PYTHON_PREFERRED:
    import warnings
    warnings.warn(
        f"Running Python {sys.version.split()[0]}; this project was developed "
        f"on {'.'.join(map(str, PYTHON_PREFERRED))}. Results should be "
        f"identical, but report your actual version in the thesis.",
        RuntimeWarning,
    )

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# os.path.dirname(__file__) = the folder where THIS file is located
# os.path.join() = safely joins folder names (works on Windows AND Linux)
# ".." = go one folder UP from current location

# The root of the entire project (the AFib_Project folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR    = os.path.join(DATA_DIR, "raw")          # Original dataset files
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")    # After preprocessing
FEATURES_DIR    = os.path.join(DATA_DIR, "features")     # Extracted HRV features

# Output paths
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
FIGURES_DIR     = os.path.join(REPORTS_DIR, "figures")   # Publication-quality plots
IMAGES_DIR      = os.path.join(BASE_DIR, "images")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
DASHBOARD_DIR   = os.path.join(BASE_DIR, "dashboard")
MODELS_DIR      = os.path.join(BASE_DIR, "models")       # Saved trained models

# -----------------------------------------------------------------------------
# WHERE THE ACTUAL CACHET-CADB DATASET LIVES
# -----------------------------------------------------------------------------
# The dataset was NOT copied into AFib_Project/data/raw - it sits one level up
# in the workspace, next to the project folder:
#
#   Main_Project/
#   ├── AFib_Project/          <- BASE_DIR (the code)
#   └── Dataset/
#       └── cachet cadb/
#           └── CACHET-CADB/
#               ├── signal/       <- Unisens recordings (ecg.bin, acc.bin, ...)
#               └── annotations/  <- annotation.csv + context.xlsx per session
#
# We point at it directly instead of duplicating ~10 GB of binary data.
# Override with the CADB_ROOT environment variable if your copy lives elsewhere.

WORKSPACE_DIR   = os.path.dirname(BASE_DIR)             # Main_Project/

CADB_ROOT = os.environ.get(
    "CADB_ROOT",
    os.path.join(WORKSPACE_DIR, "Dataset", "cachet cadb", "CACHET-CADB"),
)
CADB_SIGNAL_DIR      = os.path.join(CADB_ROOT, "signal")
CADB_ANNOTATION_DIR  = os.path.join(CADB_ROOT, "annotations")

# =============================================================================
# ECG SIGNAL CONFIGURATION
# =============================================================================
# These values are specific to the CACHET-CADB dataset.
# We will explain these in detail during Phase 2 (Dataset Study).

# !! CORRECTED !! An earlier draft of this file said 256 Hz. That was wrong.
# The truth is written by the recording device itself, inside every
# signal/<subject>/<session>/<part>/unisens.xml file:
#
#   <signalEntry contentClass="ecg" id="ecg.bin" sampleRate="1024"
#                dataType="int16" adcResolution="16" baseline="2048"
#                lsbValue="0.0026858184230029595" unit="mV" />
#
# ALWAYS trust the metadata file over any hardcoded number. The loader in
# src/preprocessing/unisens_reader.py re-reads this per recording and will
# raise if a recording disagrees with the value below.

SAMPLING_FREQUENCY  = 1024     # Hz - ECG channel (EcgMove4 sensor, chest)
                                # 1024 samples of the ECG waveform every second.

ACC_SAMPLING_FREQ   = 64       # Hz - 3-axis accelerometer (accX/Y/Z), unit g
GYRO_SAMPLING_FREQ  = 64       # Hz - 3-axis angular rate, unit dps
PRESS_SAMPLING_FREQ = 8        # Hz - barometric pressure, unit Pa
LIVE_SAMPLING_FREQ  = 1 / 60   # Hz - on-device HR, RMSSD and movement
                                # acceleration: ONE value per minute.

# Should we downsample the ECG before processing?
#   1024 Hz is far more than ECG needs (useful content is under ~40 Hz), and
#   it makes every filter and peak-detection pass 4x slower. Downsampling to
#   256 Hz keeps R-peak timing accurate to ~4 ms, which is well below the
#   beat-to-beat variation HRV measures. This is where the "256" from the old
#   config actually belongs - as a TARGET, not as the source rate.
TARGET_SAMPLING_FREQ = 256     # Hz - rate used for all downstream analysis
DOWNSAMPLE_ECG       = True    # Set False to work at the native 1024 Hz

NYQUIST_FREQUENCY   = TARGET_SAMPLING_FREQ / 2  # = 128 Hz
                                # THEORY: You must sample at LEAST 2x the highest
                                # frequency you want to capture (Nyquist theorem).
                                # Human ECG has frequencies up to ~40 Hz,
                                # so 128 Hz is more than enough.

LEAD_CONFIGURATION  = "single_lead"   # "ECG I", chest-worn, single channel
SIGNAL_UNIT         = "mV"            # ECG voltage measured in millivolts
ECG_BASELINE        = 2048            # ADC offset subtracted before scaling
ECG_LSB_VALUE       = 0.0026858184230029595  # mV per ADC step (fallback only;
                                             # the real value is read per file)

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================
# These are the filter parameters we will use to clean the ECG signal.
# Full explanation comes in Phase 5 (ECG Preprocessing).

# Bandpass filter: keeps only frequencies BETWEEN low and high cutoffs
BANDPASS_LOW_HZ     = 0.5      # Hz - removes baseline wander (slow drift)
BANDPASS_HIGH_HZ    = 40.0     # Hz - removes high-frequency noise

# Notch filter: removes a SPECIFIC frequency (powerline noise)
POWERLINE_FREQ_HZ   = 50       # Hz - India uses 50 Hz power grid
                                # (USA uses 60 Hz)

# Filter order: higher = sharper cutoff, but more computation
FILTER_ORDER        = 4

# =============================================================================
# R-PEAK DETECTION CONFIGURATION
# =============================================================================
# Parameters for detecting the "R peak" (the tall spike in ECG)
# Full explanation in Phase 7.

MIN_RR_MS           = 300      # milliseconds - minimum time between two heartbeats
                                # = 200 BPM max heart rate (unrealistic to exceed this)
                                
MAX_RR_MS           = 2000     # milliseconds - maximum time between two heartbeats
                                # = 30 BPM min heart rate (below this = abnormal pause)

# =============================================================================
# HRV FEATURE EXTRACTION CONFIGURATION  
# =============================================================================
# How we segment the ECG into windows for feature extraction.
# Full explanation in Phase 9.

# !! CORRECTED !! An earlier draft used 300-second (5-minute) windows.
# We do NOT choose the window ourselves - CACHET-CADB has already chosen it.
# Every row of every annotation.csv looks like:
#
#   Start,End,Class
#   32196064,32206304,1
#
# Start and End are SAMPLE INDICES into ecg.bin at 1024 Hz.
# End - Start = 10240 samples = exactly 10 seconds, for every single row.
#
# So the labelled unit of this dataset is a 10-second segment. Inventing
# 5-minute windows would mean merging segments whose labels disagree, which
# destroys the very labels we are trying to learn.
#
# CONSEQUENCE FOR HRV: 10 seconds holds only ~10-15 beats. That is enough for
# short-term HRV (RMSSD, pNN50, SD1, entropy measures - all beat-to-beat) but
# NOT enough for frequency-domain features. LF power sits at 0.04-0.15 Hz;
# resolving 0.04 Hz needs at least 25 seconds of signal. We therefore EXCLUDE
# LF/HF/LF-HF-ratio and say so explicitly in the paper, rather than reporting
# numbers that are mathematically meaningless.

SEGMENT_DURATION_SEC = 10      # Native labelled segment length in CACHET-CADB
SEGMENT_SAMPLES_RAW  = SEGMENT_DURATION_SEC * SAMPLING_FREQUENCY      # 10240
SEGMENT_SAMPLES      = SEGMENT_DURATION_SEC * TARGET_SAMPLING_FREQ    # 2560

# Kept as aliases so older phase scripts still import cleanly.
WINDOW_SIZE_SECONDS = SEGMENT_DURATION_SEC
WINDOW_OVERLAP      = 0.0      # No overlap: segments are pre-defined, and
                                # overlapping them would leak the same beats
                                # into both train and test sets.

# -----------------------------------------------------------------------------
# ECTOPIC BEAT FILTERING - READ THIS BEFORE ENABLING IT
# -----------------------------------------------------------------------------
# Standard HRV practice removes "ectopic" beats: intervals that deviate more
# than ~20% from a local median, on the assumption that such a jump is a
# premature beat or a detection error rather than genuine autonomic variation.
# That assumption is correct for SINUS RHYTHM. It is catastrophic for AF.
#
# In atrial fibrillation there is no sinus pacemaker. Ventricular activation is
# driven by chaotic atrial impulses, so consecutive RR intervals genuinely do
# differ by far more than 20%. That irregularity is not an artifact to clean -
# IT IS THE DIAGNOSTIC SIGNAL. Filtering it out means deleting the evidence and
# then asking a classifier to find it.
#
# We measured the damage on this dataset with the filter enabled at 20%:
#
#     class   beats removed   segments destroyed (dropped for too few beats)
#     AF          48.5%              25.7%   (192 of 747)
#     NSR         15.4%               0.5%   (3 of 615)
#
# Two separate harms, both severe:
#   1. SURVIVING AF segments lost 35% of their beats on average, flattening
#      their measured RMSSD/SDNN toward NSR values and blunting every feature.
#   2. A quarter of all AF segments were deleted outright - and not at random.
#      The ones deleted are the MOST irregular, i.e. the most unambiguous AF.
#      Training on what remains is selecting on the label, which inflates every
#      reported metric. A detector validated this way looks good on paper and
#      fails on exactly the cases that matter clinically.
#
# So the filter is OFF by default. We keep only the physiological range check
# (MIN_RR_MS / MAX_RR_MS), which removes impossible intervals without assuming
# the rhythm is regular. This matches AF-detection literature, where RR
# irregularity is the feature rather than the noise.

ECTOPIC_FILTER_ENABLED = False   # See above. Do not enable for AF detection.
ECTOPIC_THRESHOLD      = 0.20    # Only used when the filter is enabled.
ECTOPIC_WINDOW         = 5       # Local median window, in beats.

MIN_RR_BEATS        = 5        # Minimum beats in a 10 s segment to compute HRV.
                                # Below ~5 beats (=30 BPM) the segment is either
                                # a genuine pause or, far more likely, noise.

USE_FREQUENCY_DOMAIN = False   # See the explanation above. Do not flip this to
                                # True without extending the window length.

# =============================================================================
# MACHINE LEARNING CONFIGURATION
# =============================================================================

RANDOM_STATE        = 42       # Fixed seed for reproducibility
                                # Using 42 ensures same results every run.
                                # (A famous number in computer science!)
                                
TEST_SIZE           = 0.2      # 20% of data used for testing
VALIDATION_SIZE     = 0.1      # 10% for validation

# Cross-validation
N_FOLDS             = 5        # 5-fold cross validation
CV_STRATEGY         = "GroupKFold"  # We use subject-wise splitting
                                    # to prevent data leakage!

# -----------------------------------------------------------------------------
# CLASS LABELS
# -----------------------------------------------------------------------------
# CACHET-CADB writes FOUR classes in the "Class" column of annotation.csv.
# These are the RAW codes as they appear on disk:
#
#   1 = AF     (Atrial Fibrillation)
#   2 = NSR    (Normal Sinus Rhythm)
#   3 = Noise  (signal too corrupted to be readable)
#   4 = Other  (other arrhythmia / non-AF abnormal rhythm)
#
# !! READ THIS BEFORE YOU "FIX" THE ORDER !!
# The intuitive guess is 1=NSR, 2=AF, because normal usually comes first.
# THAT GUESS IS WRONG, and it is wrong in the most dangerous possible way:
# the pipeline runs perfectly, the metrics look reasonable, and every single
# prediction is inverted. We verified the true mapping with three independent
# checks before writing these constants down:
#
#   1. RR-INTERVAL IRREGULARITY (the definition of AF). Over 160 randomly
#      sampled segments, class 1 had RMSSD 290 ms and RR coefficient of
#      variation 0.236; class 2 had RMSSD 42 ms and CV 0.042. AF is
#      "irregularly irregular" by definition, so class 1 is AF - a 7x gap
#      that cannot be explained any other way.
#
#   2. THE DEVICE'S OWN HRV COLUMN in context.xlsx, computed by the sensor
#      firmware and entirely independent of our code, agrees: class 1 mean
#      RMSSD 86.5 ms vs class 2 15.0 ms.
#
#   3. THE CONTROL COHORT SETTLES IT. Subjects PNSR-1, PNSR-3 and PNSR-4 are
#      healthy volunteers - "NSR" is literally in the folder name. They carry
#      ZERO class-1 segments and 65 class-2 segments. A healthy control cannot
#      be in atrial fibrillation, so class 2 must be NSR.
#
# Actual counts in this copy of the dataset (all subjects, all sessions):
#   AF 747  |  NSR 615  |  Noise 221  |  Other 19   (1602 total)
#
# Note this is a small dataset by ML standards - which is exactly why
# subject-wise cross-validation (below) is non-negotiable.

RAW_CLASS_AF        = 1
RAW_CLASS_NSR       = 2
RAW_CLASS_NOISE     = 3
RAW_CLASS_OTHER     = 4

RAW_CLASS_NAMES = {
    RAW_CLASS_AF:    "AF",
    RAW_CLASS_NSR:   "NSR",
    RAW_CLASS_NOISE: "Noise",
    RAW_CLASS_OTHER: "Other",
}

# A cheap runtime guard against anyone silently reordering the constants.
assert RAW_CLASS_AF == 1 and RAW_CLASS_NSR == 2, (
    "Class codes were changed! CACHET-CADB uses 1=AF, 2=NSR. "
    "See the verification notes above before touching this."
)

# Our task is BINARY: AF vs not-AF. We map the raw codes to model labels.
CLASS_NORMAL        = 0        # Label for Normal sinus rhythm
CLASS_AFIB          = 1        # Label for Atrial Fibrillation

# Which raw classes become which model label. "Other" (only 19 segments) is
# grouped with non-AF because that is the clinically meaningful question a
# screening device asks: "is this AF, yes or no?" With 19 samples we could not
# learn a separate "Other" class anyway.
LABEL_MAP = {
    RAW_CLASS_NSR:   CLASS_NORMAL,
    RAW_CLASS_AF:    CLASS_AFIB,
    RAW_CLASS_OTHER: CLASS_NORMAL,
}

# Noise segments are DROPPED from training. They carry no rhythm label - the
# signal is unreadable, so there is nothing to learn. But we keep them on disk
# and reuse them in Phase 14: a detector that fires on noise is precisely the
# false positive our context analysis is about.
DROP_NOISE_FROM_TRAINING = True

# =============================================================================
# FEATURE SELECTION CONFIGURATION
# =============================================================================

# !! CORRECTED !! This was 20, and we extract exactly 20 HRV features - so
# MRMR was "selecting" all of them and the entire feature-selection stage was
# a no-op that quietly did nothing. A selection step that selects everything
# is not a selection step; reporting it as one would be misleading.
#
# 10 makes the stage meaningful and gives a result worth reporting: if AUC
# holds with half the features, that is evidence the HRV feature set is highly
# redundant (which it is - RMSSD, SD1 and SDSD are near-algebraic transforms
# of one another) and that a wearable could compute far fewer of them.
N_FEATURES_TO_SELECT = 10      # How many features to keep after MRMR selection
MRMR_METHOD          = "MIQ"   # Mutual Information Quotient method

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

FIGURE_DPI          = 150      # Dots per inch - image quality for saved plots
FIGURE_SIZE         = (14, 6)  # Default figure size (width, height) in inches
PLOT_STYLE          = "seaborn-v0_8-darkgrid"  # Plot theme
COLOR_NORMAL        = "#2ecc71"   # Green for normal rhythm
COLOR_AFIB          = "#e74c3c"   # Red for AF rhythm
COLOR_ECG           = "#2980b9"   # Blue for ECG signal

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL           = "INFO"   # How much detail to log
                                # DEBUG < INFO < WARNING < ERROR < CRITICAL
LOG_FORMAT          = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
LOG_DATE_FORMAT     = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# DATASET SUBJECTS
# =============================================================================
# CACHET-CADB has 24 subjects. We list them for reference.
# Full explanation in Phase 2.

# !! CORRECTED !! The old code generated ['p01' ... 'p24'], which matches
# NOTHING on disk. The real folder names under signal/ and annotations/ are:

SUBJECT_IDS = [
    "P1",  "P2",  "P3",  "P4",  "P5",  "P6",  "P7",  "P8",
    "P9",  "P10", "P11", "P12", "P13", "P14", "P15", "P16",
    "P17", "P18", "P19", "P21", "P23",          # AF-patient cohort
    "PNSR-1", "PNSR-3", "PNSR-4",               # healthy control cohort
]
TOTAL_SUBJECTS = len(SUBJECT_IDS)   # = 24

# Two things to notice, both of which matter for the results section:
#
# 1. THE NUMBERING HAS GAPS. There is no P20 or P22 - those recordings were
#    excluded by the dataset authors. Never generate subject IDs with a loop;
#    always list the directories that actually exist. The loader does this.
#
# 2. THERE ARE TWO COHORTS. P1-P23 are cardiac patients; PNSR-1/3/4 are
#    healthy controls who contribute normal rhythm only. If a control subject
#    lands entirely in your test fold, accuracy on that fold is inflated and
#    means very little. This is the main reason we split by SUBJECT, not by
#    segment - see CV_STRATEGY above.

CONTROL_SUBJECT_IDS = ["PNSR-1", "PNSR-3", "PNSR-4"]

# Each subject folder contains one or more SESSIONS (e.g. "a2b3c4@cachet.dk1"),
# and each session is split into numbered PARTS ("0", "1", ... "4-last")
# because the sensor rotates its files. Sample indices in annotation.csv are
# relative to the part they sit in, so part identity must be tracked.

# =============================================================================
# CONTEXT DATA COLUMNS
# =============================================================================
# These are the contextual variables we will use in False Positive Analysis.
# Full explanation in Phase 14.

# !! CORRECTED !! These are the REAL column names in context.xlsx, units and
# all. The old guessed names ("activity", "stress", ...) match nothing.
#
# context.xlsx sits next to each annotation.csv and has 32 columns sampled on
# a 10-SECOND GRID ("Time rel [s]" = 0, 10, 20, ...). That grid lines up
# exactly with the 10-second labelled segments - a lucky and very convenient
# fact: each ECG segment maps to exactly one context row, no interpolation.

CONTEXT_TIME_COLUMN = "Time rel [s]"   # Seconds since recording start
CONTEXT_GRID_SEC    = 10               # One context row per 10 seconds

# The context variables used in the Phase 14 false-positive analysis:
CONTEXT_COLUMNS = [
    "ActivityClass []",           # Coded activity (lying/sitting/walking/...)
    "BodyPosition []",            # Trunk orientation from the accelerometer
    "MovementAcceleration [g]",   # THE key variable - motion artifact proxy
    "StepCount [steps]",          # Steps in the interval (pedometer)
    "BaevskyStressIndex []",      # Autonomic stress index
    "NonWearSleepWake []",        # Sleep/wake/non-wear state
    "MET []",                     # Metabolic equivalent - exertion intensity
    "Hr [1/min]",                 # Device-computed heart rate
    "InclinationForward [deg]",   # Posture detail
    "InclinationRight [deg]",
]

# DEVICE-COMPUTED HRV COLUMNS - DO NOT USE AS MODEL FEATURES.
# context.xlsx also ships HrvRmssd, HrvSdnn, HrvSd1, HrvPnn50 etc., computed
# on-device by the sensor firmware. Training on these would be circular: we
# would be predicting AF from someone else's HRV algorithm instead of from the
# ECG. We keep them ONLY to sanity-check our own feature extraction in Phase 9
# (our RMSSD should correlate strongly with theirs - a free correctness test).
CONTEXT_DEVICE_HRV_COLUMNS = [
    "HrvRmssd [ms]", "HrvSdnn [ms]", "HrvSd1 [ms]", "HrvSd2 [ms]",
    "HrvPnn50 [%]", "HrvSdsd [ms]", "HrvLf [ms^2]", "HrvHf [ms^2]",
    "HrvLfHf []",
]

# Movement Acceleration Index threshold separating "at rest" from "in motion".
# Provisional - Phase 14 re-derives it from the data distribution.
MAI_MOTION_THRESHOLD = 0.1     # g

# =============================================================================
# PRINT CONFIGURATION SUMMARY (useful for debugging)
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Base Directory    : {BASE_DIR}")
    print(f"CADB Root         : {CADB_ROOT}")
    print(f"  exists?         : {os.path.isdir(CADB_ROOT)}")
    print(f"Native ECG rate   : {SAMPLING_FREQUENCY} Hz")
    print(f"Analysis rate     : {TARGET_SAMPLING_FREQ} Hz "
          f"(downsample={DOWNSAMPLE_ECG})")
    print(f"Bandpass Filter   : {BANDPASS_LOW_HZ} - {BANDPASS_HIGH_HZ} Hz")
    print(f"Segment           : {SEGMENT_DURATION_SEC} s "
          f"= {SEGMENT_SAMPLES_RAW} raw / {SEGMENT_SAMPLES} analysis samples")
    print(f"Freq-domain HRV   : {USE_FREQUENCY_DOMAIN} (window too short)")
    print(f"Subjects          : {TOTAL_SUBJECTS} "
          f"({len(CONTROL_SUBJECT_IDS)} healthy controls)")
    print(f"Raw classes       : {RAW_CLASS_NAMES}")
    print(f"CV Strategy       : {CV_STRATEGY}, {N_FOLDS} folds")
    print(f"Random State      : {RANDOM_STATE}")
    print("=" * 60)
