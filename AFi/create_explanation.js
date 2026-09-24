const docx = require('docx');
const fs = require('fs');

const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ShadingType, PageBreak,
  Header, Footer, PageNumber, NumberFormat, TabStopPosition, TabStopType
} = docx;

// Color palette
const PRIMARY = "1B4F72";
const SECONDARY = "2E86C1";
const ACCENT = "E74C3C";
const DARK = "2C3E50";
const LIGHT_BG = "EBF5FB";
const WHITE = "FFFFFF";
const BORDER_COLOR = "BDC3C7";

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 200 },
    children: [new TextRun({ text, bold: true, size: 32, color: PRIMARY, font: "Calibri" })],
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 300, after: 150 },
    children: [new TextRun({ text, bold: true, size: 26, color: SECONDARY, font: "Calibri" })],
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, bold: true, size: 22, color: DARK, font: "Calibri" })],
  });
}

function para(...runs) {
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    children: runs.map(r => {
      if (typeof r === 'string') return new TextRun({ text: r, size: 22, font: "Calibri", color: "333333" });
      return r;
    }),
  });
}

function bold(text) {
  return new TextRun({ text, bold: true, size: 22, font: "Calibri", color: "333333" });
}

function italic(text) {
  return new TextRun({ text, italics: true, size: 22, font: "Calibri", color: "555555" });
}

function tableCell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.shading ? { type: ShadingType.CLEAR, color: "auto", fill: opts.shading } : undefined,
    children: [
      new Paragraph({
        alignment: opts.align || AlignmentType.LEFT,
        spacing: { before: 40, after: 40 },
        children: [new TextRun({
          text,
          bold: !!opts.bold,
          size: opts.size || 20,
          font: "Calibri",
          color: opts.color || "333333",
        })],
      }),
    ],
  });
}

function headerCell(text, width) {
  return tableCell(text, { bold: true, shading: PRIMARY, color: WHITE, width, size: 20 });
}

const noBorder = { style: BorderStyle.NONE, size: 0, color: WHITE };
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: BORDER_COLOR };

const doc = new Document({
  styles: {
    paragraphStyles: [],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1200, bottom: 1200, left: 1200, right: 1200 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "HRV-Based AFib Detection Using CACHET-CADB", italics: true, size: 18, color: "999999", font: "Calibri" })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", size: 18, color: "999999", font: "Calibri" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "999999", font: "Calibri" }),
          ],
        })],
      }),
    },
    children: [
      // ============ TITLE PAGE ============
      new Paragraph({ spacing: { before: 2000 }, children: [] }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
        children: [new TextRun({ text: "COMPREHENSIVE RESEARCH EXPLANATION", bold: true, size: 40, color: PRIMARY, font: "Calibri" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 100 },
        children: [new TextRun({ text: "HRV Feature Extraction and Context-Aware", bold: true, size: 30, color: SECONDARY, font: "Calibri" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 100 },
        children: [new TextRun({ text: "False Positive Analysis for Atrial Fibrillation Detection", bold: true, size: 30, color: SECONDARY, font: "Calibri" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        children: [new TextRun({ text: "Using CACHET-CADB Free-Living Ambulatory ECG Database", bold: true, size: 30, color: SECONDARY, font: "Calibri" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 100 },
        children: [new TextRun({ text: "B.Tech Final Year Main Project", size: 24, color: DARK, font: "Calibri" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 800 },
        children: [new TextRun({ text: "Complete Technical Reference Document", size: 22, color: "777777", font: "Calibri" })],
      }),

      // ============ SECTION 1: PROJECT OVERVIEW ============
      new Paragraph({ children: [new TextRun({ children: [new PageBreak()] })] }),
      heading1("1. Project Overview"),

      para("This project builds an end-to-end machine learning pipeline that detects Atrial Fibrillation (AF) from raw electrocardiogram (ECG) signals recorded in real-world, free-living conditions. What sets it apart from typical AF detection studies is a second, equally important goal: it systematically investigates ", bold("why"), " AF detectors produce false alarms by correlating misclassifications with the patient's physical activity, body posture, movement intensity, and stress level at the moment of each error."),

      para("The project uses the ", bold("CACHET-CADB"), " (Context-Annotated Cardiac Database), a unique dataset from the Technical University of Denmark that provides both raw ECG recordings and synchronized contextual metadata from 24 patients over a combined 259 days of ambulatory monitoring. This database is the only publicly available ECG dataset that pairs every 10-second ECG segment with information about what the patient was physically doing at that exact moment."),

      para("The pipeline is divided into eight sequential stages spread across two semesters (8 months), progressing from raw signal processing through classification and finally to a novel context-correlation analysis that no published paper has fully explored on this specific dataset."),

      // ============ SECTION 2: CLINICAL BACKGROUND ============
      heading1("2. Clinical Background: Why This Matters"),

      heading2("2.1 What Is Atrial Fibrillation?"),
      para("Atrial Fibrillation (AF) is the most common sustained cardiac arrhythmia worldwide, affecting an estimated 33.5 million people globally. In AF, the heart's upper chambers (atria) fire electrical impulses chaotically at rates of 350 to 600 times per minute, instead of the normal coordinated contraction. The atrioventricular (AV) node randomly allows some of these impulses through to the ventricles, producing an ", bold("irregularly irregular"), " heartbeat — the defining clinical signature of AF."),

      para("AF is dangerous primarily because it increases the risk of stroke by approximately 5-fold. Blood pools in the fibrillating atria, forming clots that can travel to the brain. AF is also associated with heart failure, reduced quality of life, and increased mortality. There are three clinical subtypes: ", bold("Paroxysmal AF"), " (episodes that start and stop spontaneously, typically within 7 days), ", bold("Persistent AF"), " (episodes lasting longer than 7 days, requiring intervention to restore normal rhythm), and ", bold("Permanent AF"), " (continuous AF where rhythm control has been abandoned)."),

      para("The detection challenge lies in paroxysmal AF. Because it comes and goes unpredictably, a standard 12-lead ECG recorded during a brief hospital visit often misses it entirely. This is why wearable, ambulatory ECG monitors that record continuously over days or weeks are increasingly important — and why automated detection algorithms are needed to sift through the massive volume of recorded data."),

      heading2("2.2 How AF Appears in an ECG Signal"),
      para("A normal ECG heartbeat cycle consists of three main components: the ", bold("P wave"), " (atrial depolarization — the atria contracting in a coordinated manner), the ", bold("QRS complex"), " (ventricular depolarization — the sharp spike representing the main heartbeat contraction), and the ", bold("T wave"), " (ventricular repolarization — the ventricles resetting). In normal sinus rhythm (NSR), the P wave is clearly visible before each QRS complex, and the intervals between consecutive R-peaks (the tallest point of each QRS complex) are relatively regular."),

      para("In AF, two characteristic changes occur. First, the P wave disappears entirely, replaced by low-amplitude, chaotic ", bold("fibrillatory (f) waves"), " because the atria are firing randomly rather than contracting in an organized way. Second, the RR intervals (time between consecutive heartbeats) become ", bold("irregularly irregular"), " — there is no discernible pattern to the spacing between beats. It is this second feature, the chaotic RR interval pattern, that Heart Rate Variability (HRV) features are designed to capture and quantify."),

      heading2("2.3 The False Positive Problem in Wearable Monitoring"),
      para("When AF detection algorithms are deployed on wearable devices in real-world settings, they face a critical reliability problem: ", bold("false positives"), ". A false positive occurs when the algorithm flags a segment as AF when the patient actually has a normal rhythm. Research has shown that only about 34% of irregular pulse alerts on smartwatches correspond to confirmed AF on follow-up ECG monitoring (Apple Heart Study, 2019)."),

      para("The primary culprit is ", bold("motion artifacts"), ". When a patient walks, jogs, cycles, or even shifts position, body movement introduces electrical noise into the ECG signal. This noise can distort the RR interval pattern, making it mimic the irregularity characteristic of AF. The result: the algorithm raises an alarm for a rhythm that is actually normal but corrupted by movement. This is not merely an academic problem — false alerts cause patient anxiety, unnecessary emergency visits, inappropriate anticoagulation therapy, and erosion of physician trust in digital health tools."),

      para("Understanding ", bold("which specific activities and conditions"), " trigger these false positives is the unique research contribution of this project. By using CACHET-CADB's synchronized context data, the project can answer questions like: 'Does jogging cause a higher false positive rate than sitting?' and 'Is there a movement acceleration threshold above which false positives spike?' No published study has systematically answered these questions using this specific dataset."),

      // ============ SECTION 3: THE CACHET-CADB DATABASE ============
      new Paragraph({ children: [new TextRun({ children: [new PageBreak()] })] }),
      heading1("3. The CACHET-CADB Database"),

      heading2("3.1 What Makes It Unique"),
      para("CACHET-CADB (Context-Annotated Cardiac Database) was published by Kumar, Puthusserypady, Dominguez, Sharma, and Bardram from the Department of Health Technology at DTU in Frontiers in Cardiovascular Medicine (July 2022). It is a 259-day-long contextualized single-channel ECG arrhythmia database recorded from 24 patients under free-living ambulatory conditions using a Movisens EcgMove4 chest-worn sensor."),

      para("What makes CACHET-CADB fundamentally different from other ECG databases (like MIT-BIH AFDB or CPSC 2021) is the ", bold("contextual metadata"), ". For every 10-second window of ECG data, the database also records what the patient was doing: their activity class (sitting, walking, jogging, cycling, lying, etc.), body position (supine, prone, left side, right side, upright, standing), movement acceleration index (MAI — a continuous measure of body movement intensity), step count, self-reported stress level (1–5 scale), and self-reported sleep quality (1–5 scale). This contextual data is the key that enables the false-positive correlation analysis."),

      heading2("3.2 Dataset Statistics"),

      new Table({
        width: { size: 9600, type: WidthType.DXA },
        rows: [
          new TableRow({ children: [
            headerCell("Parameter", 4800),
            headerCell("Value", 4800),
          ]}),
          new TableRow({ children: [
            tableCell("Number of patients", { width: 4800 }),
            tableCell("24", { width: 4800 }),
          ]}),
          new TableRow({ children: [
            tableCell("Total recording duration", { width: 4800 }),
            tableCell("259 days (24 hours to 3 weeks per patient)", { width: 4800 }),
          ]}),
          new TableRow({ children: [
            tableCell("ECG sampling frequency", { width: 4800 }),
            tableCell("1024 Hz, 12-bit resolution", { width: 4800 }),
          ]}),
          new TableRow({ children: [
            tableCell("Total annotated segments", { width: 4800 }),
            tableCell("1,602 (each 10 seconds long)", { width: 4800 }),
          ]}),
          new TableRow({ children: [
            tableCell("AF segments", { width: 4800 }),
            tableCell("747 (46.6%)", { width: 4800 }),
          ]}),
          new TableRow({ children: [
            tableCell("NSR (Normal Sinus Rhythm) segments", { width: 4800 }),
            tableCell("615 (38.4%)", { width: 4800 }),
          ]}),
          new TableRow({ children: [
            tableCell("Noise segments", { width: 4800 }),
            tableCell("221 (13.8%)", { width: 4800 }),
          ]}),
          new TableRow({ children: [
            tableCell("Other rhythm segments", { width: 4800 }),
            tableCell("19 (1.2%)", { width: 4800 }),
          ]}),
          new TableRow({ children: [
            tableCell("Noisy data proportion", { width: 4800 }),
            tableCell("~11% of total ECG data", { width: 4800 }),
          ]}),
          new TableRow({ children: [
            tableCell("Annotation method", { width: 4800 }),
            tableCell("Two independent qualified cardiologists", { width: 4800 }),
          ]}),
        ],
      }),

      heading2("3.3 File Structure"),
      para("Each patient folder in CACHET-CADB contains a ", bold("Signal folder"), " with binary sensor files (ecg.bin for raw ECG, acc.bin for accelerometer data, angularrate.bin for gyroscope data, press.bin for pressure sensor data, and a unisens.xml metadata file containing patient demographics and recording parameters), and an ", bold("Annotation folder"), " with context.xlsx (contextual labels every 10 seconds) and annotation.csv (ECG rhythm labels: AF=1, NSR=2, Noise=3, Others=4)."),

      para("The ECG data is stored in ", bold("Unisens binary format"), ", not as a readable CSV. Loading it requires the pyUnisens Python library that can parse the binary file using the metadata from unisens.xml. This is a non-trivial data engineering step that introduces genuine technical challenge early in the project."),

      // ============ SECTION 4: THE COMPLETE PIPELINE ============
      new Paragraph({ children: [new TextRun({ children: [new PageBreak()] })] }),
      heading1("4. The Complete Pipeline: Stage-by-Stage Explanation"),

      heading2("Stage 1: ECG Preprocessing (Month 3)"),

      heading3("4.1.1 Why Preprocessing Is Necessary"),
      para("Raw ECG signals from ambulatory recordings contain substantial noise from three main sources. ", bold("Baseline wander"), " is a slow, undulating drift in the signal baseline caused by breathing and electrode movement, typically below 0.5 Hz. ", bold("High-frequency noise"), " comes from electrical interference (power line at 50/60 Hz) and muscle contractions. ", bold("Motion artifacts"), " are transient distortions caused by body movement, electrode shifting, or device contact issues. In CACHET-CADB's free-living data, about 11% of recordings are classified as noisy, and even 'clean' segments contain varying degrees of artifact."),

      heading3("4.1.2 Bandpass Filtering"),
      para("The solution is a ", bold("Butterworth bandpass filter"), " with cutoff frequencies of 0.5 Hz (high-pass, to remove baseline wander) and 50 Hz (low-pass, to remove high-frequency noise). A 4th-order Butterworth filter is used because it provides a maximally flat passband response, meaning it preserves the signal amplitude without distortion within the pass band. The filtfilt() function from SciPy applies the filter in both forward and reverse directions, producing zero phase distortion — critical because phase shifts would move the apparent position of R-peaks, corrupting RR interval measurements."),

      para("After bandpass filtering, an optional ", bold("Savitzky-Golay smoothing"), " step can further smooth the signal while preserving peak shapes. This uses a polynomial fit within a sliding window (typically window_length=11, polyorder=3) to reduce residual high-frequency noise without blunting the sharp QRS peaks."),

      heading3("4.1.3 Validation"),
      para("Every preprocessing step must be visually validated by plotting raw versus filtered ECG side-by-side. The key checks are: has baseline wander been eliminated? Are QRS peaks still sharp and clearly defined? Does the filter work on both clean and noisy segments? Segments where filtering fails to produce clean output should be flagged for exclusion."),

      heading2("Stage 2: R-Peak Detection — Pan-Tompkins Algorithm (Month 3)"),

      heading3("4.2.1 What Are R-Peaks?"),
      para("The R-peak is the tallest, sharpest point of the QRS complex in each heartbeat cycle. Detecting R-peaks is the foundational step for all subsequent analysis because the time between consecutive R-peaks (the ", bold("RR interval"), ") is the raw material from which all HRV features are computed. Accurate R-peak detection is therefore critical — a single missed or falsely detected peak corrupts two adjacent RR intervals."),

      heading3("4.2.2 The Pan-Tompkins Algorithm"),
      para("Published by Jiapu Pan and Willis Tompkins in 1985, this remains the standard algorithm for real-time QRS complex detection. It works in five stages:"),

      para(bold("Differentiation: "), "The filtered ECG is differentiated (the first derivative is calculated). This accentuates the steep slopes of the QRS complex while suppressing the gentler slopes of P and T waves. The QRS complex has the steepest slope of any ECG component because ventricular depolarization is the most rapid electrical event in the cardiac cycle."),

      para(bold("Squaring: "), "Each differentiated value is squared. This serves two purposes: it makes all values positive (the downstroke of the QRS would otherwise produce negative values), and it nonlinearly amplifies larger peaks relative to smaller ones. Because squaring is a nonlinear operation, a peak that is twice as tall in the derivative becomes four times as prominent after squaring. This helps separate QRS complexes from T waves, which can sometimes have significant slopes."),

      para(bold("Moving Window Integration: "), "A 150-millisecond sliding window averages the squared signal. This produces a smooth envelope around each QRS complex, converting the sharp multi-peaked QRS morphology into a single smooth hump. The 150ms window width is important: too wide and adjacent QRS and T wave humps merge; too narrow and a single QRS complex produces multiple separate peaks."),

      para(bold("Adaptive Thresholding: "), "The algorithm maintains two dynamic thresholds: a signal-level threshold and a noise-level threshold. Peaks above the signal threshold are classified as QRS complexes; peaks between the two thresholds are classified as noise. After each detected peak, both thresholds are updated based on the amplitude of the detected peak, allowing the algorithm to adapt to changing signal amplitude throughout the recording."),

      para(bold("Search-Back and Artifact Rules: "), "Several heuristic rules handle edge cases. If no QRS is detected within 166% of the average RR interval (1.83 seconds at a normal heart rate), the algorithm searches back through the noise-classified peaks for a missed QRS. If a detected RR interval is shorter than 80% of the recent mean minus one standard deviation, the detection is likely a false positive (perhaps a T wave) and is rejected. Intervals deviating from the mean by more than 1.9 standard deviations are flagged as artifacts."),

      para("In practice, the neurokit2 library's ecg_process() function provides a robust implementation that handles these steps automatically, though students are expected to understand each step conceptually and validate the output visually on multiple recording types (AF patients, NSR patients, and noisy segments)."),

      heading2("Stage 3: RR Interval Extraction and Cleaning (Month 3)"),
      para("Once R-peaks are detected, RR intervals are calculated as the time difference between consecutive peaks, converted from samples to milliseconds: RR_ms = (R_peak[i+1] - R_peak[i]) × (1000 / 1024). Three cleaning steps follow."),
      para(bold("Physiological bounds: "), "RR intervals below 300ms (heart rate above 200 bpm, physiologically impossible in most contexts) or above 2000ms (heart rate below 30 bpm, extremely unlikely) are removed."),
      para(bold("Ectopic beat removal: "), "If an RR interval differs from the local mean by more than 20%, it likely represents a premature atrial contraction (PAC) or premature ventricular contraction (PVC) rather than a normal or AF beat. These are removed because ectopic beats introduce variability that is not related to the underlying rhythm and would contaminate HRV features."),
      para(bold("Minimum count threshold: "), "For each annotated 10-second ECG segment, all clean RR intervals within that window are extracted. A typical 10-second window at normal heart rate contains 8 to 15 RR intervals. Segments with fewer than 5 clean RR intervals are excluded because too few data points produce unreliable HRV statistics."),

      // ============ HRV FEATURES ============
      new Paragraph({ children: [new TextRun({ children: [new PageBreak()] })] }),
      heading2("Stage 4: HRV Feature Extraction (Month 4)"),
      para("Heart Rate Variability (HRV) is the variation in the duration of consecutive cardiac cycles. In healthy individuals, RR intervals vary slightly and predictably, modulated by the autonomic nervous system (breathing, posture changes, emotional state). In AF, RR intervals vary massively and randomly because the chaotic atrial firing produces unpredictable ventricular responses. HRV features are mathematical tools that quantify different aspects of this variability, allowing a classifier to distinguish the organized variability of NSR from the chaotic variability of AF."),
      para("This project extracts 14 to 16 features from four categories:"),

      heading3("4.4.1 Time-Domain Features"),
      para(bold("MeanNN: "), "The arithmetic mean of all RR intervals in the segment. Provides a basic measure of heart rate but is unreliable as an AF indicator alone because AF can occur at any average heart rate."),
      para(bold("SDNN (Standard Deviation of NN intervals): "), "Measures overall spread of RR intervals around their mean. Higher SDNN indicates greater variability. In AF, SDNN is significantly elevated because beats are spaced randomly. Normal range is 50–100ms; AF values are typically much higher."),
      para(bold("RMSSD (Root Mean Square of Successive Differences): "), "The most important single HRV feature for AF detection. It computes the square root of the mean of the squared differences between consecutive RR intervals. Unlike SDNN which measures total spread, RMSSD specifically captures ", bold("beat-to-beat"), " irregularity — exactly the hallmark of AF. In NSR, consecutive beats are similar, so RMSSD is low to moderate. In AF, each beat is wildly different from the next, so RMSSD is very high."),
      para(bold("SDSD: "), "Standard deviation of successive RR interval differences. Closely related to RMSSD but measures the spread of the differences rather than their magnitude."),
      para(bold("pNN50 and pNN20: "), "The percentage of consecutive RR intervals differing by more than 50ms (or 20ms). In NSR, most consecutive beats are similar, so pNN50 is typically 10–20%. In AF, many consecutive beats differ substantially, pushing pNN50 to 50–80%. pNN20 is more sensitive and catches subtler irregularities."),
      para(bold("MCVNN (Median Coefficient of Variation): "), "The standard deviation of RR intervals divided by their median. Normalizes variability by heart rate, making it comparable across patients with different baseline rates. Hasan and Motin (2025) found this to be the ", bold("single most important feature"), " for AF detection using XGBoost, with an importance score of 0.37. The use of median rather than mean makes it robust to outliers caused by motion artifacts."),
      para(bold("MadNN (Median Absolute Deviation): "), "The median of the absolute deviations of RR intervals from their median. A robust alternative to SDNN that is not influenced by extreme outlier values. Ranked as the second most important feature (importance: 0.23)."),
      para(bold("IQRNN (Interquartile Range): "), "The difference between the 75th and 25th percentiles of RR intervals. In NSR, intervals cluster tightly (small IQR); in AF, they spread widely (large IQR). Third most important feature (importance: 0.06)."),
      para(bold("HTI (HRV Triangular Index): "), "Total number of RR intervals divided by the peak height of the RR interval histogram. Measures overall HRV through the shape of the distribution. NSR produces a tall, narrow histogram (consistent beats, high HTI); AF produces a short, wide histogram (random beats, low HTI)."),

      heading3("4.4.2 Entropy Features"),
      para(bold("Sample Entropy (SampEn): "), "Measures the unpredictability of the RR interval sequence. The algorithm takes patterns of m consecutive values (typically m=2), counts how often each pattern repeats elsewhere in the sequence (within a tolerance r = 0.2 × standard deviation), and calculates how much that repeatability drops when the pattern length increases to m+1. High SampEn means patterns rarely repeat — the signal is complex and unpredictable (characteristic of AF). Low SampEn means patterns recur frequently — the signal has structure (characteristic of NSR). An important caveat: standard SampEn becomes unreliable for very short sequences (fewer than about 60 beats), which is why CosEn (Coefficient of Sample Entropy) may be preferred for 10-second windows."),
      para(bold("Shannon Entropy: "), "Measures uncertainty in the probability distribution of RR interval values. A histogram of RR values is constructed, the probability of each bin is calculated, and the Shannon formula H = -Σ p(x) log2 p(x) is applied. NSR intervals cluster in a narrow range (low uncertainty, low Shannon Entropy); AF intervals spread across a wide range (high uncertainty, high Shannon Entropy)."),

      heading3("4.4.3 Nonlinear Features (Poincaré Plot)"),
      para("A Poincaré plot graphs each RR interval against the next one: RR(n) on the x-axis, RR(n+1) on the y-axis. The resulting scatter plot forms an ellipse whose shape encodes rhythm regularity."),
      para(bold("SD1: "), "The width (minor axis) of the Poincaré ellipse, measuring short-term, beat-to-beat variability. Mathematically, SD1 = sqrt(Var(diff(RR))/2). In AF, the Poincaré plot is a large, diffuse cloud (high SD1); in NSR, it is a compact cluster (low SD1)."),
      para(bold("SD2: "), "The length (major axis) of the Poincaré ellipse, measuring longer-term variability. SD2 = sqrt(2×Var(RR) - Var(diff(RR))/2). Both SD1 and SD2 are elevated in AF."),

      heading3("4.4.4 Fragmentation Features"),
      para("These features, typically computed using neurokit2's hrv_fragmentation() function, measure how often and how rapidly the heart rhythm changes direction:"),
      para(bold("PIP (Percentage of Inflection Points): "), "How often the RR series changes from increasing to decreasing or vice versa. A fragmented, chaotic rhythm (AF) produces many direction changes (high PIP)."),
      para(bold("IALS (Inverse of Average Length of Acceleration/Deceleration Segments): "), "Measures how short the runs of consistently increasing or decreasing RR intervals are. Short runs mean more fragmentation."),
      para(bold("PSS (Percentage of Short Segments): "), "What fraction of RR intervals belong to very short acceleration/deceleration runs."),
      para(bold("PAS (Percentage of Alternating Segments): "), "How often the RR interval alternates between going up and going down in consecutive intervals."),

      heading3("4.4.5 Feature Validation"),
      para("After computing all features, each one must be validated against physiological expectations. For every feature, box plots comparing AF versus NSR distributions should show the difference in the expected direction (e.g., RMSSD higher in AF than NSR). A Mann-Whitney U test confirms statistical significance (p < 0.05). If any feature shows an unexpected direction, there is a bug in the implementation that must be found and fixed. A three-way comparison (AF vs. NSR vs. Noise) further verifies that noise segments produce distinct feature values from both rhythm classes."),

      para("The final output is a ", bold("feature matrix"), ": 1,602 rows (one per annotated segment) × 14–16 columns (HRV features) plus a class label column and separate context columns (ActivityClass, BodyPosition, MAI, StressLevel, SleepQuality) for the later correlation analysis."),

      // ============ FEATURE SELECTION AND CLASSIFICATION ============
      new Paragraph({ children: [new TextRun({ children: [new PageBreak()] })] }),
      heading2("Stage 5: mRMR Feature Selection and Classification (Month 5)"),

      heading3("4.5.1 Why Feature Selection Matters"),
      para("With 14–16 HRV features, some carry redundant information. SDNN and RMSSD, for example, both measure variability but in slightly different ways. Feeding redundant features into a classifier can hurt performance (overfitting, increased computation) and obscure which features are genuinely informative. The ", bold("mRMR (Minimum Redundancy Maximum Relevance)"), " algorithm addresses this by selecting features that are maximally relevant to the class label (AF vs. NSR) while being minimally redundant with each other."),

      para("The implementation uses the mrmr_classif() function from the mrmr-selection Python package. The output is a ranked list of features. Published benchmarks (Islam and Motin, 2023) suggest that 5 features may be optimal; Hasan and Motin (2025) found MCVNN ranks highest. Students test the classifier with the top 5, 8, 10, and all features to find the inflection point where adding more features stops improving performance."),

      heading3("4.5.2 Handling Class Imbalance with SMOTE"),
      para("The class distribution in CACHET-CADB is moderately imbalanced: AF (747 segments, 46.6%), NSR (615, 38.4%), Noise (221, 13.8%), Others (19, 1.2%). The 'Others' class is severely underrepresented. A naive classifier would ignore it entirely. ", bold("SMOTE (Synthetic Minority Over-sampling Technique)"), " generates synthetic samples for minority classes by interpolating between existing minority samples in feature space. Crucially, SMOTE is applied ", bold("only to the training data"), ", never to the test data, to prevent information leakage that would artificially inflate performance metrics."),

      heading3("4.5.3 SVM Classifier"),
      para("A Support Vector Machine with an RBF (Radial Basis Function) kernel is trained using a scikit-learn Pipeline that first standardizes features (StandardScaler) and then fits the SVM (C=1, probability=True). The key methodological requirement is ", bold("subject-wise 5-fold cross-validation"), " using GroupKFold with patient IDs as the group variable. This ensures that all segments from a given patient appear in either the training set or the test set, never both. Segment-wise cross-validation (where segments from the same patient can appear in both sets) artificially inflates accuracy because patient-specific characteristics leak across the train/test boundary."),

      heading3("4.5.4 XGBoost Classifier"),
      para("XGBoost (eXtreme Gradient Boosting) is trained with 200 estimators, learning_rate=0.1, max_depth=6, and subsample=0.8. XGBoost naturally provides feature importance scores based on how frequently and how effectively each feature is used in tree splits. These importance rankings are compared with the mRMR rankings for consistency. The comparison between SVM and XGBoost examines which achieves better AF sensitivity, lower false positive rate, and faster execution."),

      // ============ CONTEXT CORRELATION ============
      heading2("Stage 6: Context-False Positive Correlation Analysis (Month 6)"),
      para(bold("This is the most original and unique contribution of the project.")),

      heading3("4.6.1 Identifying False Positives"),
      para("After training, all test segments where the model predicted AF but the true label was not AF are identified as false positives. For each false positive, the corresponding context data row is retrieved: what activity was the patient performing? What body position were they in? What was the movement acceleration index? What was their stress level? This produces a ", bold("false-positive context dataframe"), " that is the basis for all subsequent analysis."),

      heading3("4.6.2 Activity-Wise False Positive Rate Analysis"),
      para("The central hypothesis, supported by the CACHET-CADB paper's observations, is that false positive rate (FPR = FP / (FP + TN)) differs significantly across activity classes. The expected pattern: FPR is highest during jogging and cycling (vigorous movement generates motion artifacts that mimic AF's irregular RR intervals), moderate during walking, and lowest during lying and sleeping (minimal movement, minimal artifact). A chi-square test of independence determines whether the observed differences are statistically significant (p < 0.05)."),

      heading3("4.6.3 Body Position and MAI Analysis"),
      para("The same analysis is repeated for body position (upright vs. supine vs. standing) and for the Movement Acceleration Index. The MAI analysis is particularly valuable: by plotting FPR against continuous MAI values, the project can identify a ", bold("threshold MAI value"), " above which false positives begin to spike. This threshold is found using ROC analysis. An example finding might be: 'When MAI exceeds 0.8g, the false positive rate increases by 65% compared to segments where MAI is below 0.3g.' This is an original, quantifiable finding."),

      heading3("4.6.4 Stress and Sleep Correlation"),
      para("Additional analyses examine whether self-reported stress level (1–5) and sleep quality correlate with FPR. Stress is relevant because it affects autonomic nervous system activity, which modulates HRV independently of rhythm. High stress could increase HRV even during normal rhythm, potentially causing false AF detection."),

      // ============ EVALUATION ============
      new Paragraph({ children: [new TextRun({ children: [new PageBreak()] })] }),
      heading2("Stage 7: Evaluation and Comparison (Month 7)"),

      heading3("4.7.1 Metrics"),
      para("For each classifier, the project reports per-class Sensitivity (Se = TP/(TP+FN)), Specificity (Sp = TN/(TN+FP)), Precision (Pr = TP/(TP+FP)), F1 Score (harmonic mean of precision and recall), Balanced Accuracy ((Se+Sp)/2), AUC-ROC (per class, one-vs-rest), and AUC-PRC (Precision-Recall Curve). The emphasis on sensitivity and specificity rather than raw accuracy is deliberate: in a moderately imbalanced dataset, a classifier that always predicts the majority class achieves deceptively high accuracy while being clinically useless (the ", bold("accuracy paradox"), ")."),

      heading3("4.7.2 Cross-Paper Comparison"),

      new Table({
        width: { size: 9600, type: WidthType.DXA },
        rows: [
          new TableRow({ children: [
            headerCell("Study", 2400),
            headerCell("Dataset", 2400),
            headerCell("Accuracy", 1600),
            headerCell("Sensitivity", 1600),
            headerCell("F1", 1600),
          ]}),
          new TableRow({ children: [
            tableCell("Andersen et al. (2017)", { width: 2400 }),
            tableCell("MIT-BIH AFDB", { width: 2400 }),
            tableCell("—", { width: 1600 }),
            tableCell("96.81%", { width: 1600 }),
            tableCell("—", { width: 1600 }),
          ]}),
          new TableRow({ children: [
            tableCell("Islam & Motin (2023)", { width: 2400 }),
            tableCell("CPSC 2021", { width: 2400 }),
            tableCell("92.34%", { width: 1600 }),
            tableCell("88.05%", { width: 1600 }),
            tableCell("0.91", { width: 1600 }),
          ]}),
          new TableRow({ children: [
            tableCell("Hasan & Motin (2025)", { width: 2400 }),
            tableCell("CPSC 2021", { width: 2400 }),
            tableCell("95.4%", { width: 1600 }),
            tableCell("—", { width: 1600 }),
            tableCell("0.956", { width: 1600 }),
          ]}),
          new TableRow({ children: [
            tableCell("This Project (SVM)", { width: 2400, bold: true }),
            tableCell("CACHET-CADB", { width: 2400, bold: true }),
            tableCell("TBD", { width: 1600, bold: true }),
            tableCell("TBD", { width: 1600, bold: true }),
            tableCell("TBD", { width: 1600, bold: true }),
          ]}),
          new TableRow({ children: [
            tableCell("This Project (XGBoost)", { width: 2400, bold: true }),
            tableCell("CACHET-CADB", { width: 2400, bold: true }),
            tableCell("TBD", { width: 1600, bold: true }),
            tableCell("TBD", { width: 1600, bold: true }),
            tableCell("TBD", { width: 1600, bold: true }),
          ]}),
        ],
      }),

      para("The expected outcome is that performance on CACHET-CADB will be lower than on MIT-BIH AFDB or CPSC 2021 because free-living ambulatory data is inherently noisier and more challenging than controlled clinical recordings. This performance gap is itself a meaningful finding: it quantifies how much real-world conditions degrade AF detection and motivates the need for context-aware algorithms."),

      heading3("4.7.3 Failure Mode Analysis"),
      para("Five specific false positive cases must be documented with the ECG waveform plot, the context at that moment, the HRV feature values that led to the wrong prediction, and an explanation of why the model was fooled. This failure analysis is the bridge connecting the classifier results to the context-correlation findings and demonstrates genuine analytical understanding."),

      // ============ SECTION 5: TIMELINE ============
      new Paragraph({ children: [new TextRun({ children: [new PageBreak()] })] }),
      heading1("5. Month-by-Month Timeline"),

      new Table({
        width: { size: 9600, type: WidthType.DXA },
        rows: [
          new TableRow({ children: [
            headerCell("Month", 1000),
            headerCell("Focus Area", 2800),
            headerCell("Key Deliverable", 3400),
            headerCell("Difficulty", 2400),
          ]}),
          new TableRow({ children: [
            tableCell("1", { width: 1000 }),
            tableCell("Theory study + environment setup", { width: 2800 }),
            tableCell("ECG/AF/HRV concepts report; Python env working", { width: 3400 }),
            tableCell("Moderate", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("2", { width: 1000 }),
            tableCell("CACHET-CADB data loading", { width: 2800 }),
            tableCell("Working data loader; AF vs NSR visualizations", { width: 3400 }),
            tableCell("HIGH (binary format)", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("3", { width: 1000 }),
            tableCell("ECG preprocessing pipeline", { width: 2800 }),
            tableCell("Filtering + R-peak detection + RR extraction", { width: 3400 }),
            tableCell("HIGH (real data noise)", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("4", { width: 1000 }),
            tableCell("HRV feature extraction", { width: 2800 }),
            tableCell("Feature matrix CSV; validation boxplots", { width: 3400 }),
            tableCell("HIGH", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("5", { width: 1000 }),
            tableCell("mRMR + SVM + XGBoost", { width: 2800 }),
            tableCell("Trained classifiers; comparison table", { width: 3400 }),
            tableCell("Moderate-High", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("6", { width: 1000 }),
            tableCell("Context-FP correlation", { width: 2800 }),
            tableCell("Activity/position FPR analysis; MAI threshold", { width: 3400 }),
            tableCell("Moderate (analysis-heavy)", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("7", { width: 1000 }),
            tableCell("Evaluation + comparison", { width: 2800 }),
            tableCell("Metrics tables; cross-paper comparison; failure cases", { width: 3400 }),
            tableCell("Moderate", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("8", { width: 1000 }),
            tableCell("Report + presentation + viva", { width: 2800 }),
            tableCell("60-80 page report; working demo; slides", { width: 3400 }),
            tableCell("Varies", { width: 2400 }),
          ]}),
        ],
      }),

      // ============ SECTION 6: WHY THIS PROJECT IS RIGOROUS ============
      heading1("6. Why This Project Is Rigorous Despite AI Availability"),

      para("A legitimate concern about modern B.Tech projects is that AI tools can generate boilerplate code for standard pipelines. This project is designed so that the coding is only a fraction of the actual work. The genuinely challenging parts include:"),

      para(bold("Real data engineering: "), "CACHET-CADB uses Unisens binary format. AI-generated code assumes clean CSV inputs and will fail on this binary data without significant debugging and understanding of the file structure."),

      para(bold("Signal quality handling: "), "Real ambulatory ECG contains baseline wander, electrode pops, inverted polarity, signal gaps, and motion artifacts that break naive R-peak detection. Students must understand why their code fails on specific segments and fix it."),

      para(bold("Physiological validation: "), "Every computed feature must be validated against known physiological expectations. If SampEn is lower in AF than NSR, the implementation has a bug. This validation requires domain understanding, not just coding ability."),

      para(bold("Methodological rigor: "), "Understanding why subject-wise cross-validation prevents data leakage, why SMOTE is applied only on training data, and why sensitivity matters more than accuracy in clinical AF detection — these are conceptual requirements that cannot be delegated to code generation."),

      para(bold("Original analytical thinking: "), "The context-correlation analysis (Month 6) requires formulating hypotheses, designing statistical tests, interpreting results, and writing coherent explanations. This is research work that demands thinking, not prompting."),

      para(bold("Viva defense: "), "Students must explain every design decision, interpret every result, and defend their methodology orally. Understanding cannot be outsourced."),

      // ============ SECTION 7: TOOLS AND LIBRARIES ============
      heading1("7. Required Tools and Libraries"),

      new Table({
        width: { size: 9600, type: WidthType.DXA },
        rows: [
          new TableRow({ children: [
            headerCell("Library", 2400),
            headerCell("Purpose", 4800),
            headerCell("Install Command", 2400),
          ]}),
          new TableRow({ children: [
            tableCell("numpy, pandas", { width: 2400 }),
            tableCell("Numerical computation and data manipulation", { width: 4800 }),
            tableCell("pip install numpy pandas", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("matplotlib, seaborn", { width: 2400 }),
            tableCell("Static visualization and box plots", { width: 4800 }),
            tableCell("pip install matplotlib seaborn", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("scipy", { width: 2400 }),
            tableCell("Signal filtering (Butterworth), statistical tests", { width: 4800 }),
            tableCell("pip install scipy", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("pyunisens", { width: 2400 }),
            tableCell("Reading CACHET-CADB Unisens binary format", { width: 4800 }),
            tableCell("pip install pyunisens", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("neurokit2", { width: 2400 }),
            tableCell("ECG processing, R-peak detection, HRV features", { width: 4800 }),
            tableCell("pip install neurokit2", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("scikit-learn", { width: 2400 }),
            tableCell("SVM, StandardScaler, GroupKFold, metrics", { width: 4800 }),
            tableCell("pip install scikit-learn", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("xgboost", { width: 2400 }),
            tableCell("XGBoost classifier", { width: 4800 }),
            tableCell("pip install xgboost", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("imbalanced-learn", { width: 2400 }),
            tableCell("SMOTE for class balancing", { width: 4800 }),
            tableCell("pip install imbalanced-learn", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("mrmr-selection", { width: 2400 }),
            tableCell("mRMR feature selection algorithm", { width: 4800 }),
            tableCell("pip install mrmr-selection", { width: 2400 }),
          ]}),
          new TableRow({ children: [
            tableCell("plotly", { width: 2400 }),
            tableCell("Interactive visualization dashboard", { width: 4800 }),
            tableCell("pip install plotly", { width: 2400 }),
          ]}),
        ],
      }),

      // ============ SECTION 8: KEY REFERENCES ============
      new Paragraph({ children: [new TextRun({ children: [new PageBreak()] })] }),
      heading1("8. Key References"),

      para(bold("[1] "), "Kumar D, Puthusserypady S, Dominguez H, Sharma K, Bardram JE. \"CACHET-CADB: A Contextualized Ambulatory Electrocardiography Arrhythmia Dataset.\" ", italic("Frontiers in Cardiovascular Medicine"), ", 9:893090, July 2022."),
      para(bold("[2] "), "Pan J, Tompkins WJ. \"A Real-Time QRS Detection Algorithm.\" ", italic("IEEE Transactions on Biomedical Engineering"), ", BME-32(3):230–236, March 1985."),
      para(bold("[3] "), "Andersen RS, Poulsen ES, Puthusserypady S. \"A Novel Approach for Automatic Detection of Atrial Fibrillation Based on Inter Beat Intervals and Support Vector Machine.\" ", italic("Proc. IEEE EMBC"), ", 2017:2039–2042, 2017."),
      para(bold("[4] "), "Islam MS, Motin MA. \"Short-term atrial fibrillation detection using electrocardiograms: A comparison of machine learning approaches.\" ", italic("International Journal of Medical Informatics"), ", 2023."),
      para(bold("[5] "), "Hasan MI, Motin MA. \"HRV Feature Extraction and AF Detection Using XGBoost.\" 2025."),
      para(bold("[6] "), "Michel PO et al. \"Using Minimum Redundancy Maximum Relevance Algorithm to Select Minimal Sets of Heart Rate Variability Parameters for Atrial Fibrillation Detection.\" ", italic("Journal of Clinical Medicine"), ", 11(14):4004, 2022."),
      para(bold("[7] "), "Richman JS, Moorman JR. \"Physiological time-series analysis using approximate entropy and sample entropy.\" ", italic("American Journal of Physiology - Heart and Circulatory Physiology"), ", 278(6):H2039–H2049, 2000."),
      para(bold("[8] "), "Perez MV et al. \"Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation.\" ", italic("New England Journal of Medicine"), ", 381:1909–1917, 2019. (Apple Heart Study)"),

      // ============ FINAL SUMMARY ============
      heading1("9. Summary of Final Deliverables"),
      para("Upon completion, the project produces the following concrete outputs: a working ECG preprocessing pipeline (bandpass filtering, R-peak detection, RR interval extraction), a validated HRV feature extraction tool computing 14–16 features across four categories, trained and evaluated SVM and XGBoost classifiers with subject-wise cross-validation, a complete context-false positive correlation analysis with chi-square tests and MAI threshold identification, an interactive Plotly visualization dashboard with 10 panels, a cross-paper comparison with three published studies, five documented failure case analyses with ECG waveform plots, and a 60–80 page project report suitable for B.Tech final-year submission."),

      para("The project's distinguishing strength is that it does not just classify AF — it explains ", bold("why"), " the classifier fails under specific real-world conditions. This transforms it from a standard classification exercise into a genuine research contribution that bridges signal processing, machine learning, and clinical context understanding."),
    ],
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('/sessions/elegant-cool-cerf/mnt/outputs/AFib_Project_Full_Explanation.docx', buffer);
  console.log('Document created successfully');
});
