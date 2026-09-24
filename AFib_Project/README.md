# HRV Feature Extraction and Context-Aware False Positive Analysis for Atrial Fibrillation Detection

> **B.Tech Final Year Project | IEEE-Level Research Implementation**  
> Dataset: CACHET-CADB (Context-Aware Cardiac Holter Dataset)

---

## 📌 Project Overview

This project implements a complete biomedical machine learning pipeline to:

1. **Detect Atrial Fibrillation (AF)** from short-term ECG recordings using Heart Rate Variability (HRV) features
2. **Analyze false positive predictions** by correlating model errors with real-world contextual information (activity, movement, stress, sleep, body position)

The **context-aware false positive analysis** is the primary research contribution — understanding *why* the model fails in real-world conditions.

---

## 🧬 What is Atrial Fibrillation?

Atrial Fibrillation (AF) is the most common cardiac arrhythmia affecting over 37 million people worldwide. It causes:
- Irregular, chaotic electrical activity in the upper chambers of the heart
- Increased risk of stroke (5x higher than normal)
- Often goes undetected without ECG monitoring

---

## 📁 Project Structure

```
AFib_Project/
│
├── data/
│   ├── raw/              ← Original CACHET-CADB dataset files (do NOT modify)
│   ├── processed/        ← Preprocessed ECG signals (cleaned, filtered)
│   └── features/         ← Extracted HRV feature tables (CSV files)
│
├── notebooks/            ← Jupyter notebooks for exploration and teaching
│
├── src/
│   ├── preprocessing/    ← ECG signal filtering and cleaning
│   ├── peak_detection/   ← R-peak detection (Pan-Tompkins + NeuroKit2)
│   ├── feature_extraction/ ← HRV time, frequency, nonlinear features
│   ├── feature_selection/  ← MRMR, correlation, mutual information
│   ├── models/           ← SVM and XGBoost classifiers
│   ├── evaluation/       ← Metrics: ROC, Confusion Matrix, F1, AUC
│   ├── context_analysis/ ← False positive correlation with context
│   ├── visualization/    ← All plot generation functions
│   └── utils/            ← Logger, helpers, file I/O
│
├── dashboard/            ← Interactive Plotly Dashboard (HTML)
├── reports/              ← Research paper, analysis reports
│   └── figures/          ← Publication-quality saved figures
├── images/               ← Architecture diagrams, flowcharts
├── config/
│   └── config.py         ← Central configuration (ALL settings here)
├── tests/                ← Unit tests
├── logs/                 ← Runtime logs
│
├── main.py               ← Main pipeline entry point
├── requirements.txt      ← All Python dependencies
└── README.md             ← This file
```

---

## 🚀 Quick Start

### Step 1: Clone and Navigate
```bash
cd AFib_Project
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
- Visit: https://doi.org/10.5281/zenodo.4244447
- Download CACHET-CADB dataset
- Place files in `data/raw/`

### Step 5: Run Main Pipeline
```bash
python main.py
```

---

## 📊 Dataset: CACHET-CADB

| Property | Value |
|----------|-------|
| Full Name | Context-Aware Cardiac Holter Dataset |
| Subjects | 24 patients |
| ECG Lead | Single-lead ambulatory |
| Sampling Rate | 256 Hz |
| Total Duration | ~122 hours |
| Annotations | AF, Normal, Other |
| Context Data | Activity, Position, Movement, Steps, Stress, Sleep |

---

## 🔬 Methodology

```
Raw ECG → Preprocessing → R-Peak Detection → RR Interval Extraction
     → HRV Feature Extraction → Feature Selection (MRMR)
     → Classification (SVM / XGBoost) → Evaluation
     → FALSE POSITIVE ANALYSIS with Context Data
```

---

## 📈 HRV Features Extracted

**Time Domain:** MeanNN, SDNN, RMSSD, pNN50, pNN20, MCVNN, MadNN, IQRNN, HTI  
**Nonlinear:** SD1, SD2, SD1/SD2, Sample Entropy, Shannon Entropy, Approximate Entropy  
**Fragmentation:** PAS, PIP, IALS, PSS, PAS  

---

## 🎯 Research Contribution

The novel contribution of this work is the **Context-Aware False Positive Analysis**:
- Identifies model errors correlated with physical activity
- Quantifies motion artifact influence using Movement Acceleration Index (MAI)
- Provides clinical insights for improving real-world AF detectors
- Uses Chi-square tests and MAI thresholding for statistical validation

---

## 🏆 Results Summary

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| SVM | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |

*(Results will be updated after training)*

---

## 📚 References

1. Moody GB, Mark RG. "The impact of the MIT-BIH Arrhythmia Database." *IEEE EMB Magazine* (2001)
2. Task Force of ESC. "Heart rate variability: standards." *Circulation* 93.5 (1996): 1043-1065.
3. CACHET-CADB: Context-Aware Cardiac Holter Dataset. Zenodo (2021).
4. Pan J, Tompkins WJ. "A real-time QRS detection algorithm." *IEEE TBME* (1985).

---

## 👤 Author

- **Student:** [Your Name]  
- **Roll No:** [Your Roll Number]  
- **Department:** [Your Department]  
- **Institution:** [Your College]  
- **Guide:** [Your Guide's Name]  
- **Academic Year:** 2025–2026

---

## 📄 License

This project is for academic research purposes only.
