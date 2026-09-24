"""
=============================================================================
FILE: notebooks/Phase01_Biomedical_Fundamentals.py
      (Run as a script OR copy into a Jupyter Notebook cell by cell)

PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis
         for AF Detection using CACHET-CADB

PHASE 1: BIOMEDICAL FUNDAMENTALS
=============================================================================

WHAT YOU WILL LEARN IN THIS PHASE:
    1. What is the heart and how does it work?
    2. What is an ECG (Electrocardiogram)?
    3. Understanding ECG waves: P, QRS, T
    4. What is an RR Interval?
    5. What is Heart Rate and Heart Rate Variability (HRV)?
    6. What is Normal Sinus Rhythm vs Atrial Fibrillation?
    7. What is noise and why does it hurt our analysis?
    8. What is Sampling Frequency?
    9. Basic Signal Processing concepts

=============================================================================
MENTOR NOTE:
    Read this file like a textbook chapter.
    Every concept builds on the previous one.
    Do NOT skip sections.
    After reading each section, try the mini assignment.
=============================================================================
"""

# =============================================================================
# IMPORTS - Every library we need for this phase
# =============================================================================

import numpy as np          # For numerical computations (arrays, math)
import matplotlib.pyplot as plt  # For creating plots
import matplotlib.patches as mpatches  # For drawing shapes in plots
from matplotlib.gridspec import GridSpec  # For complex subplot layouts
import warnings             # To suppress unnecessary warning messages
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")


# =============================================================================
# SECTION 1: THE HEART - ANATOMY AND ELECTRICAL CONDUCTION
# =============================================================================
"""
THEORY: The Heart as an Electrical Pump
========================================

Your heart is a PUMP made of muscle (cardiac muscle).
It has 4 chambers:

    ┌─────────────────────────────────┐
    │        RIGHT SIDE               │
    │  Right Atrium (RA) ─receives    │
    │     deoxygenated blood          │
    │  Right Ventricle (RV) ─pumps   │
    │     blood to LUNGS              │
    ├─────────────────────────────────┤
    │        LEFT SIDE                │
    │  Left Atrium (LA) ─receives    │
    │     oxygenated blood from lungs │
    │  Left Ventricle (LV) ─pumps   │
    │     blood to BODY               │
    └─────────────────────────────────┘

ELECTRICAL CONDUCTION SYSTEM:
The heart has its OWN electrical system that controls when to beat.
It works like a carefully timed electrical circuit.

THE PATH OF ELECTRICAL SIGNAL:
    
    1. SA Node (Sinoatrial Node) → "The Pacemaker"
       Located in the RIGHT ATRIUM.
       Generates an electrical impulse spontaneously.
       Normal rate: 60-100 beats per minute.
       
    2. AV Node (Atrioventricular Node) → "The Gate"
       Located between atria and ventricles.
       DELAYS the signal by ~0.12 seconds.
       WHY? To let the atria FINISH squeezing before ventricles start.
       
    3. Bundle of His → "The Highway"
       A fast pathway that carries signal down.
       
    4. Left & Right Bundle Branches → "The Roads"
       Splits into left and right, to reach both ventricles.
       
    5. Purkinje Fibers → "The Delivery System"
       Spread the signal throughout ventricular muscle.
       Makes the ventricle squeeze from BOTTOM to TOP
       (most efficient pumping direction).

REAL WORLD INTUITION:
Think of a stadium crowd doing "The Wave."
The SA Node is like the first person who stands up.
The wave spreads in an organized, predictable pattern.
In AF, MANY random people start waves simultaneously - CHAOS!
"""


# =============================================================================
# SECTION 2: WHAT IS AN ECG?
# =============================================================================
"""
THEORY: Electrocardiogram (ECG / EKG)
======================================

ECG = Electrocardiogram
- "Electro" = Electrical
- "Cardio" = Heart  
- "Gram" = Recording

DEFINITION:
An ECG is a graphical recording of the electrical activity of the heart 
over time, captured by electrodes placed on the skin.

HOW IT WORKS:
1. Electrodes (small sticky pads) are placed on skin
2. The heart's electrical signals travel through the body
3. Electrodes detect these tiny voltage changes
4. An amplifier makes the signal readable
5. The signal is plotted as Voltage (mV) vs Time (seconds)

WHY CAN WE DETECT HEART ELECTRICITY FROM SKIN?
The body is like a bag of salt water (conductive!).
When millions of heart cells fire together, the electrical field is 
strong enough to reach the skin surface.
It's like feeling vibrations from a concert outside the venue.

UNITS:
- Y-axis: Voltage in millivolts (mV) → 1 mV = 0.001 Volts
- X-axis: Time in seconds

SAMPLING FREQUENCY (CRUCIAL CONCEPT):
When a computer records an ECG, it doesn't record CONTINUOUSLY.
It takes a SNAPSHOT (sample) at regular time intervals.

CACHET-CADB samples at 256 Hz (Hz = samples per second)
This means: 256 snapshots per second
= 1 snapshot every (1/256) = 0.0039 seconds = ~3.9 milliseconds

WHY 256 Hz?
The highest important frequency in ECG is about 40 Hz.
By Nyquist theorem: sampling rate must be ≥ 2 × max_frequency
= 2 × 40 = 80 Hz minimum
256 Hz gives us comfortable headroom (3.2× the minimum).
"""


# =============================================================================
# SECTION 3: ECG WAVES - P, QRS, T
# =============================================================================
"""
THEORY: The ECG Waveform Components
=====================================

A normal ECG shows repeating patterns called "cardiac cycles."
Each cycle has distinct waves, each representing a specific electrical event.

THE PQRST COMPLEX:

P WAVE:
- Represents: Atrial depolarization (atria contracting)
- Duration: 80-120 ms (milliseconds)
- Amplitude: 0.1-0.3 mV
- Shape: Small, smooth, rounded bump
- Clinical significance: 
  * Absent P wave = possible AF!
  * Irregular P waves = atrial problems

PR INTERVAL:
- From start of P to start of QRS
- Duration: 120-200 ms
- Represents the delay at the AV Node
- Too short = Wolff-Parkinson-White syndrome
- Too long = heart block

QRS COMPLEX:
- Represents: Ventricular depolarization (ventricles contracting)
- Duration: 80-120 ms
- Amplitude: 0.5-2.0 mV (TALLEST peak on ECG)
- Q wave: Small negative deflection (downward)
- R wave: LARGE positive deflection (upward) ← This is what we detect!
- S wave: Small negative deflection after R
- Clinical significance:
  * Wide QRS = bundle branch block
  * ST elevation = heart attack!

QT INTERVAL:
- From start of QRS to end of T
- Duration: 350-440 ms
- Long QT = risk of dangerous arrhythmia

T WAVE:
- Represents: Ventricular repolarization (ventricles recovering)
- Duration: 160-320 ms
- Amplitude: 0.1-0.5 mV
- Shape: Broad, smooth, rounded bump

U WAVE (sometimes visible):
- Small wave after T
- Represents: Purkinje fiber repolarization
- Not always visible
- Prominent U wave = low potassium levels

ISOELECTRIC LINE (Baseline):
- The flat line between cardiac cycles
- Represents electrical silence (no activity)
- Our "zero reference" for measuring amplitudes

REAL WORLD INTUITION:
Imagine a pump cycle:
P = Someone fills the pump (atria fill with blood from previous beat)
QRS = The main pump FIRES (ventricles push blood out)
T = The pump resets and reloads (heart relaxes)
Then the cycle repeats.
"""


# =============================================================================
# SECTION 4: SIMULATING AND VISUALIZING AN ECG WAVE
# =============================================================================

def simulate_ecg_wave(duration_seconds: float = 3.0, 
                       heart_rate_bpm: float = 70.0,
                       sampling_freq: float = 256.0,
                       rhythm: str = "normal") -> tuple:
    """
    Simulates a synthetic ECG signal for educational visualization.
    
    WHY SYNTHETIC? 
    Before touching real patient data, we learn with simulated data.
    It's like a medical student practicing on a mannequin before a real patient.
    
    PARAMETERS:
    -----------
    duration_seconds : float
        How many seconds of ECG to generate
    heart_rate_bpm : float
        Heart rate in beats per minute
    sampling_freq : float
        Sampling frequency in Hz (samples per second)
    rhythm : str
        "normal" = regular rhythm, "afib" = simulated AF rhythm
    
    RETURNS:
    --------
    tuple : (time_array, ecg_signal)
        time_array: Array of time points in seconds
        ecg_signal: Corresponding ECG voltage values in mV
    
    MATHEMATICS:
    ------------
    Heart rate of 70 BPM means:
    - 70 beats per 60 seconds
    - = 1 beat every (60/70) = 0.857 seconds
    - = Beat period = 857 ms
    
    Sampling: 256 Hz means:
    - Total samples = duration × sampling_freq = 3 × 256 = 768 samples
    - Time array: [0, 1/256, 2/256, ..., 767/256]
    """
    
    # Total number of samples
    # Example: 3 seconds × 256 samples/second = 768 samples
    n_samples = int(duration_seconds * sampling_freq)
    
    # Create time array: evenly spaced from 0 to duration
    # np.linspace(start, stop, num_points)
    # endpoint=False means we DON'T include the last point
    # This avoids overlap if we concatenate signals
    time = np.linspace(0, duration_seconds, n_samples, endpoint=False)
    
    # Initialize ECG signal with zeros
    ecg = np.zeros(n_samples)
    
    # Beat period: time between consecutive heartbeats
    # BPM → seconds per beat = 60 / BPM
    beat_period = 60.0 / heart_rate_bpm  # in seconds
    
    # For AF: add random variation to beat timing (irregular RR intervals)
    if rhythm == "afib":
        # AF has HIGHLY irregular RR intervals
        # We simulate this by randomly varying the beat period ±30%
        beat_jitter_std = 0.30  # 30% standard deviation
    else:
        # Normal: small natural variation ±5% (normal sinus arrhythmia)
        beat_jitter_std = 0.05
    
    # Generate individual heartbeat positions
    current_time = 0.1  # Start first beat at 0.1 seconds
    beat_times = []
    
    while current_time < duration_seconds - 0.2:
        beat_times.append(current_time)
        # Add random variation to next beat timing
        next_period = beat_period * (1 + np.random.normal(0, beat_jitter_std))
        next_period = np.clip(next_period, beat_period * 0.5, beat_period * 2.0)
        current_time += next_period
    
    # For each beat, add the PQRST waveform
    for beat_time in beat_times:
        
        # === P WAVE ===
        # A Gaussian (bell) curve centered at (beat_time - 0.18 seconds)
        # Width controlled by sigma
        p_center = beat_time - 0.18  # P wave is 180ms before R peak
        p_amplitude = 0.15           # 0.15 mV tall
        p_sigma = 0.025              # Width of bell curve
        
        # Gaussian formula: A × exp(-(t - center)² / (2 × sigma²))
        p_wave = p_amplitude * np.exp(-((time - p_center) ** 2) / (2 * p_sigma ** 2))
        
        # In AF: P wave is absent (no organized atrial activity)
        if rhythm == "afib":
            p_amplitude_factor = 0.0   # No P wave in AF
            # Instead, add fibrillatory (f) waves - tiny chaotic oscillations
            f_wave = 0.08 * np.sin(2 * np.pi * 6 * time + np.random.uniform(0, 2*np.pi))
            # Only add f-waves near this beat's region
            f_mask = (np.abs(time - beat_time) < beat_period * 0.4)
            ecg += f_wave * f_mask * 0.3
            p_wave = p_wave * 0.0  # Remove P wave
        
        ecg += p_wave
        
        # === Q WAVE (small negative deflection before R) ===
        q_center = beat_time - 0.03
        q_amplitude = -0.08    # Negative (downward)
        q_sigma = 0.010
        q_wave = q_amplitude * np.exp(-((time - q_center) ** 2) / (2 * q_sigma ** 2))
        ecg += q_wave
        
        # === R WAVE (the MAIN SPIKE - what we detect!) ===
        r_center = beat_time   # R peak IS the beat time
        r_amplitude = 1.2      # 1.2 mV tall (tallest feature!)
        r_sigma = 0.012        # Very narrow (sharp spike)
        r_wave = r_amplitude * np.exp(-((time - r_center) ** 2) / (2 * r_sigma ** 2))
        ecg += r_wave
        
        # === S WAVE (negative deflection after R) ===
        s_center = beat_time + 0.035
        s_amplitude = -0.15
        s_sigma = 0.015
        s_wave = s_amplitude * np.exp(-((time - s_center) ** 2) / (2 * s_sigma ** 2))
        ecg += s_wave
        
        # === T WAVE (broader recovery wave) ===
        t_center = beat_time + 0.22   # T wave is 220ms after R
        t_amplitude = 0.3             # 0.3 mV
        t_sigma = 0.045               # Broader than P wave
        t_wave = t_amplitude * np.exp(-((time - t_center) ** 2) / (2 * t_sigma ** 2))
        ecg += t_wave
    
    # Add a tiny amount of realistic noise (always present in real ECGs)
    # np.random.normal(mean, std, size)
    noise_std = 0.02  # 20 microvolts of noise
    ecg += np.random.normal(0, noise_std, n_samples)
    
    return time, ecg, beat_times


# =============================================================================
# SECTION 5: RR INTERVAL - THE HEART OF HRV ANALYSIS
# =============================================================================
"""
THEORY: RR Interval
=====================

DEFINITION:
The RR interval is the time distance between two consecutive R peaks.

WHY R PEAKS?
Because R peaks are the TALLEST, SHARPEST feature of the ECG.
They are the easiest to detect accurately and reliably.

FORMULA:
RR interval (ms) = (Sample index of R2 - Sample index of R1) / Sampling Frequency × 1000

EXAMPLE:
- R1 detected at sample 512
- R2 detected at sample 769
- Sampling frequency = 256 Hz

RR = (769 - 512) / 256 × 1000
   = 257 / 256 × 1000
   = 1003.9 ms
   ≈ 1004 ms ≈ 1 second

CLINICAL INTERPRETATION:
RR = 1000ms → 60 BPM (1 beat per second) — Normal resting
RR = 600ms  → 100 BPM — Upper limit of normal at rest
RR = 400ms  → 150 BPM — Fast heart rate (exercise or arrhythmia)
RR = 300ms  → 200 BPM — Very fast (dangerous, ventricular tachycardia risk)
RR = 2000ms → 30 BPM  — Very slow (needs medical attention)

NORMAL RANGE for our filter: 300ms to 2000ms
(Anything outside = artifact or extreme pathology)

IN ATRIAL FIBRILLATION:
- RR intervals are HIGHLY IRREGULAR
- Variation can be 200-600ms between consecutive beats
- No organized P waves = no controlled atrial activity
- The AV node receives random impulses from the chaotic atria
- This irregularity is the SIGNATURE of AF

THIS IS WHY HRV FEATURES WORK FOR AF DETECTION:
HRV measures the VARIATION in RR intervals.
Normal: small, rhythmic variation
AF: LARGE, chaotic, unpredictable variation
"""


# =============================================================================
# SECTION 6: HEART RATE VARIABILITY (HRV) - OUR KEY SIGNAL
# =============================================================================
"""
THEORY: Heart Rate Variability (HRV)
======================================

DEFINITION:
HRV is the variation in time between successive heartbeats.
It is NOT the heart rate itself, but how MUCH the rate changes.

REAL WORLD INTUITION:
Imagine two runners who both run 100 meters in 10 seconds (same average speed).
Runner A has perfectly uniform strides.
Runner B's stride length varies randomly.

A healthy heart behaves like Runner B - it has NATURAL variability.
This variability is GOOD and comes from:
1. Autonomic Nervous System (ANS) control
2. Breathing (Respiratory Sinus Arrhythmia)
3. Blood pressure regulation (Baroreflex)

WHY IS HRV CLINICALLY IMPORTANT?
- HIGH HRV = healthy ANS, heart adapts well to changes = GOOD
- LOW HRV = poor ANS function, heart is too "rigid" = BAD
  (associated with: heart failure, diabetes, anxiety, poor recovery)
- CHAOTIC/HIGHLY IRREGULAR HRV = AF signature

THE AUTONOMIC NERVOUS SYSTEM (ANS):
The ANS has two branches that constantly "battle" over heart rate:

  SYMPATHETIC NERVOUS SYSTEM:
  - "Fight or Flight" response
  - Increases heart rate, decreases HRV
  - Active during: stress, exercise, fear
  - Effect on RR intervals: SHORTER and MORE UNIFORM

  PARASYMPATHETIC NERVOUS SYSTEM (Vagal):
  - "Rest and Digest" response
  - Decreases heart rate, increases HRV
  - Active during: relaxation, sleep
  - Effect on RR intervals: LONGER and MORE VARIABLE

IN AF:
- The normal ANS control of HRV is OVERWHELMED
- The chaotic atrial firing creates EXTREME irregularity
- HRV features capture this pathological irregularity
- This is what our model will learn to classify!

HRV FEATURES WE WILL EXTRACT (Preview):
TIME DOMAIN:
  MeanNN  = Average RR interval (ms) → baseline heart rate
  SDNN    = Standard Deviation of RR intervals → overall variability
  RMSSD   = Root Mean Square of Successive Differences → beat-to-beat variability
  pNN50   = % beats where consecutive difference > 50ms → high variability marker
  
NONLINEAR:
  SD1, SD2  = Poincaré plot parameters → short/long-term variability
  SampEn    = Sample Entropy → signal complexity/regularity
  
(Full implementation in Phase 9)
"""


# =============================================================================
# SECTION 7: NORMAL RHYTHM vs ATRIAL FIBRILLATION
# =============================================================================
"""
THEORY: Normal Sinus Rhythm (NSR) vs Atrial Fibrillation (AF)
================================================================

NORMAL SINUS RHYTHM (NSR):
- Regular P waves: one before each QRS
- Regular QRS complexes
- Regular RR intervals (with small natural variation)
- Heart rate: 60-100 BPM
- P wave → QRS delay: 120-200ms (AV node delay)
- Rhythm: REGULAR

ECG SIGNATURE OF NSR:
  P-QRS-T | P-QRS-T | P-QRS-T | P-QRS-T
  (regular, organized, predictable)
  
ATRIAL FIBRILLATION (AF):
- ABSENT P waves (replaced by chaotic "f" waves)
- IRREGULARLY IRREGULAR QRS complexes
- Highly variable RR intervals (no consistent pattern)
- Heart rate: 100-180 BPM typically (uncontrolled)
- AV node randomly passes impulses from chaotic atria
- Rhythm: CHAOTICALLY IRREGULAR

ECG SIGNATURE OF AF:
  f-QRS-T | f-QRS-T | f---QRS | fQRS-T | f--QRS
  (no P waves, QRS at random times)

WHAT CAUSES AF?
- Age (most common risk factor)
- High blood pressure
- Heart disease
- Hyperthyroidism
- Sleep apnea
- Alcohol ("Holiday Heart Syndrome")
- Genetic predisposition
- Electrolyte imbalances

CLINICAL DANGER:
In AF, the atria quiver instead of pump.
Blood POOLS in the left atrium.
Blood clots form in the pool.
Clot can travel to the BRAIN → STROKE!
AF patients have 5× higher stroke risk.

DETECTION CHALLENGE:
- AF is often PAROXYSMAL (comes and goes unpredictably)
- Standard 12-lead ECG catches it only if AF is happening RIGHT NOW
- Ambulatory (wearable) monitors like CACHET-CADB are needed
- Our project uses context data to make these monitors smarter!

KEY ECG DIFFERENCES TABLE:
┌─────────────────┬──────────────────┬──────────────────────┐
│ Feature         │ Normal Rhythm    │ Atrial Fibrillation  │
├─────────────────┼──────────────────┼──────────────────────┤
│ P waves         │ Present, regular │ Absent / Fibrillatory│
│ QRS             │ Regular          │ Irregular             │
│ RR intervals    │ Regular (±5%)    │ Highly irregular      │
│ HRV (SDNN)     │ 30-100 ms        │ Often >100 ms        │
│ HRV (pNN50)    │ 5-30%            │ Often >40%           │
│ RR pattern      │ Rhythmic         │ Random               │
└─────────────────┴──────────────────┴──────────────────────┘
"""


# =============================================================================
# SECTION 8: NOISE AND ARTIFACTS
# =============================================================================
"""
THEORY: ECG Noise Types and Their Sources
==========================================

Real ECGs are NEVER perfectly clean. They contain various types of noise.
Understanding noise is crucial because noise can MIMIC disease!

TYPE 1: BASELINE WANDER
- Cause: Breathing, electrode movement, sweating
- Frequency range: 0.05 - 0.5 Hz (very slow oscillations)
- Visual appearance: The entire ECG signal drifts slowly up and down
  like a wave on the ocean
- Effect: Makes amplitude measurements inaccurate
- Our filter: High-pass filter at 0.5 Hz removes this

TYPE 2: POWERLINE INTERFERENCE (EMI)
- Cause: Electrical noise from wall outlets (50 Hz in India)
- Frequency: Exactly 50 Hz (or 60 Hz in USA)
- Visual appearance: Regular "buzzing" pattern superimposed on ECG
- Effect: Obscures small features, especially P and T waves
- Our filter: Notch filter at 50 Hz removes this

TYPE 3: MUSCLE ARTIFACTS (EMG Noise)
- Cause: Skeletal muscle activity near electrodes
  (especially when patient is moving or exercising)
- Frequency range: 20-2000 Hz (broad spectrum)
- Visual appearance: Spiky, irregular pattern that looks like ECG peaks
- Effect: Can create FALSE R peaks or hide real ones!
- THIS IS DIRECTLY RELATED TO OUR FALSE POSITIVE ANALYSIS!

TYPE 4: MOTION ARTIFACTS
- Cause: Electrode moves relative to skin during physical movement
- Frequency range: 0.1 - 10 Hz
- Visual appearance: Large, sudden amplitude changes
- Effect: Can make signal completely unreadable
- In CACHET-CADB: Movement Acceleration Index (MAI) captures this!

TYPE 5: ELECTRODE CONTACT NOISE
- Cause: Poor electrode contact (dried gel, hair, etc.)
- Frequency: Various
- Visual appearance: Random spikes and dropouts

WHY THIS MATTERS FOR OUR PROJECT:
When our model makes a FALSE POSITIVE (predicts AF when it's actually normal),
the MOST COMMON CAUSE is motion artifact!
Why? Because motion creates irregular RR-like patterns that LOOK LIKE AF to the model.
Our context analysis will quantify exactly this!

CACHET-CADB ADVANTAGE:
The dataset includes sensor data (accelerometer, step count, activity labels)
alongside the ECG. This lets us correlate "when did the model fail?" with
"what was the patient doing?"
This is our research novelty!
"""


# =============================================================================
# SECTION 9: SIGNAL PROCESSING BASICS
# =============================================================================
"""
THEORY: Digital Signal Processing Fundamentals
================================================

WHAT IS A SIGNAL?
A signal is any quantity that varies with time.
ECG = voltage varying with time.

ANALOG vs DIGITAL:
- Analog: continuous, infinite values (real world)
- Digital: discrete, finite values (computer)
- ADC (Analog-to-Digital Converter) converts analog ECG to digital

SAMPLING:
Taking measurements at regular time intervals.
fs = 256 Hz → 256 measurements per second
T = 1/fs = 1/256 = 3.9 milliseconds between samples

NYQUIST THEOREM:
To accurately represent a signal with maximum frequency f_max,
we must sample at LEAST at fs = 2 × f_max.

ECG max frequency ≈ 40 Hz
Required minimum fs = 2 × 40 = 80 Hz
We use 256 Hz → very safe!

ALIASING (Bad thing that happens if Nyquist is violated):
If you sample too slowly, high frequencies appear as LOW frequencies.
Like a spinning wheel appearing to spin backwards in a movie.
We avoid this by using adequate sampling rate.

FREQUENCY DOMAIN:
Any signal can be expressed as a SUM of sine waves.
Fourier Transform converts: time domain → frequency domain
This lets us see WHICH frequencies exist in the signal.

ECG FREQUENCY CONTENT:
- Baseline wander: 0 - 0.5 Hz
- P wave: 0.5 - 10 Hz
- QRS complex: 5 - 50 Hz
- T wave: 1 - 7 Hz
- Muscle noise: 20 - 2000 Hz
- Powerline noise: 50 Hz (single spike)

FILTERING:
A filter KEEPS certain frequencies and REMOVES others.

Types relevant to us:
1. LOW-PASS: Keeps LOW frequencies, removes HIGH (cuts noise)
2. HIGH-PASS: Keeps HIGH frequencies, removes LOW (cuts drift)
3. BAND-PASS: Keeps a RANGE, removes rest (keeps 0.5-40 Hz for ECG)
4. NOTCH: Removes ONE specific frequency (removes 50 Hz powerline)

BUTTERWORTH FILTER:
The most common filter type in ECG processing.
Properties:
- "Maximally flat" passband (doesn't distort kept frequencies)
- Smooth roll-off (gradual transition, not abrupt)
- No ripples in passband
- Order N controls sharpness (higher = sharper)

We will implement this in detail in Phase 5.
"""


# =============================================================================
# VISUALIZATION: CREATE THE COMPLETE EDUCATIONAL ECG FIGURE
# =============================================================================

def create_educational_ecg_visualization():
    """
    Creates a comprehensive educational ECG visualization showing:
    1. Labeled ECG waveform components (P, QRS, T, waves)
    2. Normal vs AF rhythm comparison
    3. RR interval demonstration
    4. Simulated noise types
    
    This plot will be saved as an image for your project report.
    """
    
    np.random.seed(42)  # Fixed seed for reproducibility
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor('#1a1a2e')  # Dark background
    
    # Define grid layout: 5 rows, 2 columns
    gs = GridSpec(5, 2, figure=fig, hspace=0.5, wspace=0.3)
    
    # Color palette
    colors = {
        'ecg': '#00d4ff',        # Cyan for ECG signal
        'normal': '#00ff88',     # Green for normal
        'afib': '#ff4757',       # Red for AF
        'annotation': '#ffd700', # Gold for text
        'grid': '#2d3436',       # Dark gray for grid
        'background': '#16213e', # Subplot background
        'peak': '#ff6b35',       # Orange for peaks
    }
    
    # =========================================================================
    # SUBPLOT 1: Labeled ECG Beat (Full Page Width, Top)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])  # First row, both columns
    ax1.set_facecolor(colors['background'])
    
    # Generate one clean beat
    time_1beat, ecg_1beat, beats_1beat = simulate_ecg_wave(
        duration_seconds=1.2, heart_rate_bpm=75, rhythm="normal"
    )
    
    ax1.plot(time_1beat * 1000, ecg_1beat, color=colors['ecg'], linewidth=2.5, zorder=3)
    ax1.axhline(y=0, color='white', linewidth=0.5, alpha=0.3, linestyle='--')
    ax1.grid(True, color=colors['grid'], alpha=0.5, linewidth=0.5)
    
    # Label the waveform components with arrows
    label_style = dict(fontsize=11, color=colors['annotation'], fontweight='bold')
    arrow_style = dict(arrowstyle='->', color=colors['annotation'], lw=1.5)
    
    # Find R peak in the signal for reference
    r_idx = np.argmax(ecg_1beat)
    r_time = time_1beat[r_idx] * 1000  # Convert to ms
    r_val = ecg_1beat[r_idx]
    
    # P wave label
    ax1.annotate('P Wave\n(Atrial depolarization)\n~80-120ms',
                xy=(r_time - 180, 0.15), xytext=(r_time - 350, 0.6),
                arrowprops=arrow_style, **label_style, ha='center')
    
    # Q wave label  
    ax1.annotate('Q\n(small)', xy=(r_time - 30, -0.08),
                xytext=(r_time - 150, -0.5), arrowprops=arrow_style,
                **label_style, ha='center')
    
    # R wave label
    ax1.annotate('R Wave (R Peak)\n← We detect this!\nAmplitude: 1-2mV',
                xy=(r_time, r_val), xytext=(r_time + 80, 1.5),
                arrowprops=arrow_style, **label_style, ha='center')
    
    # S wave label
    ax1.annotate('S\n(small)', xy=(r_time + 35, -0.15),
                xytext=(r_time + 180, -0.6), arrowprops=arrow_style,
                **label_style, ha='center')
    
    # T wave label
    ax1.annotate('T Wave\n(Ventricular repolarization)\n~160-320ms',
                xy=(r_time + 220, 0.3), xytext=(r_time + 380, 0.8),
                arrowprops=arrow_style, **label_style, ha='center')
    
    # Add interval markers
    # PR interval
    ax1.annotate('', xy=(r_time - 30, -0.85), xytext=(r_time - 210, -0.85),
                arrowprops=dict(arrowstyle='<->', color='#a29bfe', lw=2))
    ax1.text(r_time - 120, -1.0, 'PR Interval\n(120-200ms)', ha='center',
            fontsize=9, color='#a29bfe')
    
    # QRS duration
    ax1.annotate('', xy=(r_time + 65, -1.2), xytext=(r_time - 35, -1.2),
                arrowprops=dict(arrowstyle='<->', color='#fd79a8', lw=2))
    ax1.text(r_time + 15, -1.35, 'QRS\n(80-120ms)', ha='center',
            fontsize=9, color='#fd79a8')
    
    ax1.set_xlim(0, 1200)
    ax1.set_ylim(-1.5, 2.3)
    ax1.set_xlabel('Time (milliseconds)', color='white', fontsize=11)
    ax1.set_ylabel('Amplitude (mV)', color='white', fontsize=11)
    ax1.set_title('ECG Waveform: PQRST Complex with Clinical Labels',
                 color=colors['annotation'], fontsize=13, fontweight='bold', pad=10)
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#4a4a7a')
    
    # =========================================================================
    # SUBPLOT 2: Normal Sinus Rhythm (3 seconds)
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor(colors['background'])
    
    time_n, ecg_n, beats_n = simulate_ecg_wave(
        duration_seconds=4.0, heart_rate_bpm=72, rhythm="normal"
    )
    ax2.plot(time_n, ecg_n, color=colors['normal'], linewidth=2, label='Normal ECG')
    ax2.axhline(y=0, color='white', linewidth=0.5, alpha=0.3, linestyle='--')
    ax2.grid(True, color=colors['grid'], alpha=0.4)
    
    # Mark R peaks
    for bt in beats_n:
        ax2.axvline(x=bt, color='yellow', alpha=0.4, linewidth=1, linestyle=':')
    
    # Annotate RR interval between first two beats
    if len(beats_n) >= 2:
        rr_ms = (beats_n[1] - beats_n[0]) * 1000
        mid = (beats_n[0] + beats_n[1]) / 2
        ax2.annotate('', xy=(beats_n[1], 1.5), xytext=(beats_n[0], 1.5),
                    arrowprops=dict(arrowstyle='<->', color='yellow', lw=2))
        ax2.text(mid, 1.65, f'RR = {rr_ms:.0f}ms\n({60000/rr_ms:.0f} BPM)',
                ha='center', fontsize=10, color='yellow', fontweight='bold')
    
    ax2.set_title('Normal Sinus Rhythm — Regular, Organized, Predictable',
                 color=colors['normal'], fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', color='white', fontsize=10)
    ax2.set_ylabel('Amplitude (mV)', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.set_ylim(-0.8, 2.0)
    for spine in ax2.spines.values():
        spine.set_color('#4a4a7a')
    
    # =========================================================================
    # SUBPLOT 3: Atrial Fibrillation (3 seconds)
    # =========================================================================
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_facecolor(colors['background'])
    
    time_af, ecg_af, beats_af = simulate_ecg_wave(
        duration_seconds=4.0, heart_rate_bpm=90, rhythm="afib"
    )
    ax3.plot(time_af, ecg_af, color=colors['afib'], linewidth=2, label='AF ECG')
    ax3.axhline(y=0, color='white', linewidth=0.5, alpha=0.3, linestyle='--')
    ax3.grid(True, color=colors['grid'], alpha=0.4)
    
    # Annotate RR intervals to show irregularity
    for bt in beats_af:
        ax3.axvline(x=bt, color='#ffeaa7', alpha=0.4, linewidth=1, linestyle=':')
    
    # Show multiple RR intervals to demonstrate irregularity
    for i in range(min(3, len(beats_af)-1)):
        rr_ms = (beats_af[i+1] - beats_af[i]) * 1000
        mid = (beats_af[i] + beats_af[i+1]) / 2
        y_pos = 1.3 - (i * 0.1)
        ax3.annotate('', xy=(beats_af[i+1], y_pos), xytext=(beats_af[i], y_pos),
                    arrowprops=dict(arrowstyle='<->', color='#ffeaa7', lw=1.5))
        ax3.text(mid, y_pos + 0.12, f'RR={rr_ms:.0f}ms',
                ha='center', fontsize=9, color='#ffeaa7', fontweight='bold')
    
    ax3.set_title('Atrial Fibrillation — No P Waves, IRREGULAR RR Intervals',
                 color=colors['afib'], fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (seconds)', color='white', fontsize=10)
    ax3.set_ylabel('Amplitude (mV)', color='white', fontsize=10)
    ax3.tick_params(colors='white')
    ax3.set_ylim(-0.8, 2.0)
    for spine in ax3.spines.values():
        spine.set_color('#4a4a7a')
    
    # =========================================================================
    # SUBPLOT 4: RR Interval Comparison (Tachogram)
    # =========================================================================
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.set_facecolor(colors['background'])
    
    # Compute RR intervals for both
    rr_normal = np.diff(beats_n) * 1000   # Convert to ms
    rr_afib   = np.diff(beats_af) * 1000
    
    # Plot tachogram (RR interval over time)
    ax4.plot(range(len(rr_normal)), rr_normal, 'o-',
            color=colors['normal'], linewidth=2, markersize=6, label='Normal')
    ax4.plot(range(len(rr_afib)), rr_afib, 'o-',
            color=colors['afib'], linewidth=2, markersize=6, label='AF')
    
    ax4.axhline(y=np.mean(rr_normal), color=colors['normal'], 
               linewidth=1.5, linestyle='--', alpha=0.7, 
               label=f'Mean Normal: {np.mean(rr_normal):.0f}ms')
    ax4.axhline(y=np.mean(rr_afib), color=colors['afib'], 
               linewidth=1.5, linestyle='--', alpha=0.7, 
               label=f'Mean AF: {np.mean(rr_afib):.0f}ms')
    
    ax4.set_title('RR Interval Tachogram\n(Heartbeat-by-Heartbeat Variation)',
                 color=colors['annotation'], fontsize=11, fontweight='bold')
    ax4.set_xlabel('Beat Number', color='white')
    ax4.set_ylabel('RR Interval (ms)', color='white')
    ax4.legend(fontsize=8, loc='upper right', 
              facecolor='#2d3436', labelcolor='white')
    ax4.tick_params(colors='white')
    ax4.grid(True, color=colors['grid'], alpha=0.4)
    for spine in ax4.spines.values():
        spine.set_color('#4a4a7a')
    
    # =========================================================================
    # SUBPLOT 5: HRV Distribution Comparison
    # =========================================================================
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.set_facecolor(colors['background'])
    
    # Histogram of RR intervals
    ax5.hist(rr_normal, bins=10, color=colors['normal'], 
            alpha=0.6, label=f'Normal (SDNN={np.std(rr_normal):.1f}ms)',
            edgecolor='white', linewidth=0.5)
    ax5.hist(rr_afib, bins=10, color=colors['afib'], 
            alpha=0.6, label=f'AF (SDNN={np.std(rr_afib):.1f}ms)',
            edgecolor='white', linewidth=0.5)
    
    ax5.set_title('RR Interval Distribution\n(AF is MORE SPREAD = Higher HRV)',
                 color=colors['annotation'], fontsize=11, fontweight='bold')
    ax5.set_xlabel('RR Interval (ms)', color='white')
    ax5.set_ylabel('Count', color='white')
    ax5.legend(fontsize=9, facecolor='#2d3436', labelcolor='white')
    ax5.tick_params(colors='white')
    ax5.grid(True, color=colors['grid'], alpha=0.4)
    for spine in ax5.spines.values():
        spine.set_color('#4a4a7a')
    
    # =========================================================================
    # SUBPLOT 6: Noise Types Visualization
    # =========================================================================
    ax6 = fig.add_subplot(gs[4, :])
    ax6.set_facecolor(colors['background'])
    
    # Create a clean ECG
    t, clean_ecg, _ = simulate_ecg_wave(duration_seconds=3.0, 
                                         heart_rate_bpm=72, rhythm="normal")
    
    # Remove default noise from our simulation
    clean_ecg_no_noise = clean_ecg.copy()
    
    # Add different noise types
    n = len(t)
    
    # Baseline wander (0.15 Hz slow oscillation)
    baseline_wander = 0.4 * np.sin(2 * np.pi * 0.15 * t)
    
    # Powerline interference (50 Hz)
    powerline = 0.15 * np.sin(2 * np.pi * 50 * t)
    
    # Muscle artifact (random high-freq noise, only in segment 1.5-2.5s)
    muscle_noise = np.zeros(n)
    mask = (t > 1.5) & (t < 2.5)
    muscle_noise[mask] = 0.3 * np.random.normal(0, 1, np.sum(mask))
    
    # Combined noisy ECG
    noisy_ecg = clean_ecg_no_noise + baseline_wander + powerline + muscle_noise
    
    # Plot
    ax6.plot(t, noisy_ecg, color='#e17055', linewidth=1.5, label='Noisy ECG', alpha=0.9)
    ax6.plot(t, clean_ecg_no_noise, color=colors['normal'], linewidth=2, 
            label='Clean ECG', alpha=0.8, linestyle='--')
    
    # Annotate noise regions
    ax6.annotate('Baseline\nWander', xy=(0.3, -0.3), xytext=(0.3, -0.9),
                arrowprops=dict(arrowstyle='->', color='#fdcb6e'),
                fontsize=9, color='#fdcb6e', ha='center')
    
    ax6.annotate('Powerline\nNoise (50Hz)', xy=(0.8, 1.4), xytext=(0.8, 1.9),
                arrowprops=dict(arrowstyle='->', color='#74b9ff'),
                fontsize=9, color='#74b9ff', ha='center')
    
    ax6.annotate('Muscle Artifact\n(Motion-related!)', xy=(2.0, 1.2), 
                xytext=(2.0, 1.9),
                arrowprops=dict(arrowstyle='->', color='#ff7675'),
                fontsize=9, color='#ff7675', ha='center')
    
    ax6.set_title('ECG Noise Types: Baseline Wander + Powerline Interference + Muscle Artifact',
                 color=colors['annotation'], fontsize=11, fontweight='bold')
    ax6.set_xlabel('Time (seconds)', color='white')
    ax6.set_ylabel('Amplitude (mV)', color='white')
    ax6.legend(fontsize=10, facecolor='#2d3436', labelcolor='white', loc='lower right')
    ax6.tick_params(colors='white')
    ax6.grid(True, color=colors['grid'], alpha=0.4)
    for spine in ax6.spines.values():
        spine.set_color('#4a4a7a')
    
    # Main figure title
    fig.suptitle('PHASE 1: Biomedical Fundamentals\nECG Signal Analysis for AF Detection',
                color='white', fontsize=16, fontweight='bold', y=0.98)
    
    # Save the figure
    import os
    save_dir = r"c:\vaishnav\projects\Main_Project\AFib_Project\reports\figures"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "Phase01_ECG_Fundamentals.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
               facecolor=fig.get_facecolor())
    print(f"\nFigure saved to: {save_path}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
    return save_path


# =============================================================================
# SECTION 10: NUMERICAL DEMONSTRATION OF HRV STATISTICS
# =============================================================================

def demonstrate_hrv_basics():
    """
    Demonstrates basic HRV statistics on simulated RR intervals.
    This gives you an intuition for what our features MEAN numerically.
    """
    
    print("\n" + "="*65)
    print("DEMONSTRATION: HRV STATISTICS ON SIMULATED RR INTERVALS")
    print("="*65)
    
    np.random.seed(42)
    
    # Simulate Normal RR intervals (ms)
    # Mean ~857ms = 70 BPM, small variation (std = 35ms)
    rr_normal = np.array([
        842, 856, 871, 845, 863, 851, 869, 840, 858, 875,
        849, 862, 854, 867, 843, 858, 872, 847, 861, 855
    ], dtype=float)
    
    # Simulate AF RR intervals (ms)
    # Higher mean ~700ms = 86 BPM, MUCH larger variation (std = 180ms)
    rr_afib = np.array([
        612, 845, 534, 923, 445, 1102, 678, 512, 891, 423,
        756, 1034, 589, 812, 467, 943, 623, 788, 512, 867
    ], dtype=float)
    
    for label, rr in [("NORMAL RHYTHM", rr_normal), ("ATRIAL FIBRILLATION", rr_afib)]:
        print(f"\n{'='*30}")
        print(f"  {label}")
        print(f"{'='*30}")
        
        # Basic statistics
        mean_nn = np.mean(rr)
        sdnn = np.std(rr)        # Standard deviation of NN intervals
        
        # Successive differences
        diff_rr = np.diff(rr)   # Differences between consecutive RR intervals
        rmssd = np.sqrt(np.mean(diff_rr**2))  # Root Mean Square of Successive Differences
        
        # pNN50: percentage of consecutive RR differences > 50ms
        pnn50 = (np.sum(np.abs(diff_rr) > 50) / len(diff_rr)) * 100
        
        # pNN20: percentage of consecutive RR differences > 20ms
        pnn20 = (np.sum(np.abs(diff_rr) > 20) / len(diff_rr)) * 100
        
        print(f"  RR Intervals (ms): {rr[:6]}...")
        print(f"  \n  --- TIME DOMAIN FEATURES ---")
        print(f"  MeanNN (ms) : {mean_nn:.2f}  → {60000/mean_nn:.1f} BPM average")
        print(f"  SDNN   (ms) : {sdnn:.2f}   → Overall variability")
        print(f"  RMSSD  (ms) : {rmssd:.2f}   → Beat-to-beat variability")
        print(f"  pNN50  (%)  : {pnn50:.1f}    → Beats with diff > 50ms")
        print(f"  pNN20  (%)  : {pnn20:.1f}    → Beats with diff > 20ms")
        
        print(f"\n  --- CLINICAL INTERPRETATION ---")
        if sdnn > 80:
            print(f"  ⚠ SDNN = {sdnn:.1f}ms >> HIGH variability → Possible AF")
        else:
            print(f"  ✓ SDNN = {sdnn:.1f}ms → Normal variability range")
            
        if pnn50 > 30:
            print(f"  ⚠ pNN50 = {pnn50:.1f}% → Very high! Suspicious for AF")
        else:
            print(f"  ✓ pNN50 = {pnn50:.1f}% → Normal range")
    
    print("\n" + "="*65)
    print("KEY INSIGHT:")
    print("  Normal SDNN: ~35ms | AF SDNN: much higher")
    print("  This is why SDNN is a powerful AF detection feature!")
    print("="*65)
    
    return rr_normal, rr_afib


# =============================================================================
# SECTION 11: SAMPLING FREQUENCY DEMONSTRATION
# =============================================================================

def demonstrate_sampling():
    """
    Visually demonstrates what sampling frequency means.
    Shows how different sampling rates capture (or miss) signal details.
    """
    
    print("\n" + "="*60)
    print("SAMPLING FREQUENCY DEMONSTRATION")
    print("="*60)
    
    # Create a "true" continuous signal (simulated at very high rate)
    t_true = np.linspace(0, 0.1, 10000)  # 0 to 100ms, very dense
    
    # Simulate a QRS complex (sharp peak)
    qrs_true = 1.5 * np.exp(-((t_true - 0.05)**2) / (2 * 0.005**2))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    sampling_rates = [50, 128, 256]  # Hz
    titles = ['50 Hz (Too Low!)', '128 Hz (Minimum)', '256 Hz (CACHET-CADB)']
    colors_sr = ['#ff4757', '#ffa502', '#2ed573']
    
    for ax, fs, title, color in zip(axes, sampling_rates, titles, colors_sr):
        ax.set_facecolor('#16213e')
        
        # Sample the signal at this rate
        sample_times = np.arange(0, 0.1, 1.0/fs)
        sample_values = 1.5 * np.exp(-((sample_times - 0.05)**2) / (2 * 0.005**2))
        
        # Plot true signal
        ax.plot(t_true*1000, qrs_true, color='white', linewidth=2, 
               alpha=0.5, label='True signal', zorder=1)
        
        # Plot sampled version
        ax.stem(sample_times*1000, sample_values, 
               linefmt=color, markerfmt=f'o', basefmt='gray',
               label=f'Sampled ({fs} Hz)')
        
        # Connect samples to show reconstruction
        ax.plot(sample_times*1000, sample_values, color=color, 
               linewidth=1.5, alpha=0.7, linestyle='--', zorder=2)
        
        ax.set_title(f'{title}\n{fs} samples in 1 second',
                    color=color, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (ms)', color='white')
        ax.set_ylabel('Amplitude (mV)', color='white')
        ax.legend(fontsize=8, facecolor='#2d3436', labelcolor='white')
        ax.tick_params(colors='white')
        ax.grid(True, color='#2d3436', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color('#4a4a7a')
        
        n_samples = len(sample_times)
        print(f"Sampling at {fs:3d} Hz: {n_samples:3d} samples in 100ms | "
              f"Time resolution: {1000/fs:.1f} ms/sample")
    
    fig.suptitle('Effect of Sampling Frequency on QRS Peak Capture',
                color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    save_path = r"c:\vaishnav\projects\Main_Project\AFib_Project\reports\figures\Phase01_Sampling.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nSampling demo saved to: {save_path}")
    plt.show()


# =============================================================================
# MAIN: RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("PHASE 1: BIOMEDICAL FUNDAMENTALS")
    print("HRV Feature Extraction for Atrial Fibrillation Detection")
    print("=" * 70)
    print()
    
    print("Step 1: Creating ECG Educational Visualization...")
    create_educational_ecg_visualization()
    
    print("\nStep 2: Demonstrating HRV Statistics...")
    rr_normal, rr_afib = demonstrate_hrv_basics()
    
    print("\nStep 3: Demonstrating Sampling Frequency...")
    demonstrate_sampling()
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE!")
    print("=" * 70)
    print("\nWhat you learned:")
    print("  ✓ Heart anatomy and electrical conduction system")
    print("  ✓ ECG waves: P, Q, R, S, T and their clinical meaning")
    print("  ✓ RR interval and its measurement")
    print("  ✓ Heart Rate Variability (HRV)")
    print("  ✓ Normal vs AF rhythm differences")
    print("  ✓ ECG noise types and their sources")
    print("  ✓ Sampling frequency and Nyquist theorem")
    print("  ✓ Basic HRV statistics (MeanNN, SDNN, RMSSD, pNN50)")
    print("\nSaved figures in: reports/figures/")
    print("\nNext: Phase 2 - Dataset Study (CACHET-CADB)")
    print("Type 'confirm' to proceed to Phase 2 when ready!")
