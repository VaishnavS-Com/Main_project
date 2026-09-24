# -*- coding: utf-8 -*-
"""
=============================================================================
FILE: reports/generate_phase01_pdf.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PURPOSE:
    Generates a professional, multi-page PDF report for Phase 1
    (Biomedical Fundamentals) of the AF Detection project.

    Uses matplotlib's PdfPages backend — no external PDF library needed.
    Every page is drawn as a matplotlib figure and saved into one PDF.

OUTPUT:
    C:\\vaishnav\\projects\\Main_Project\\AFib_Project\\Phase01_Biomedical_Fundamentals.pdf

HOW TO RUN:
    cd c:\\vaishnav\\projects\\Main_Project\\AFib_Project
    .\\venv\\Scripts\\activate
    python reports\\generate_phase01_pdf.py

PYTHON VERSION: 3.13.7
=============================================================================
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')                          # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages  # Multi-page PDF support
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# OUTPUT PATH
# =============================================================================
OUTPUT_PDF = r"C:\vaishnav\projects\Main_Project\AFib_Project\Phase01_Biomedical_Fundamentals.pdf"
FIGURES_DIR = r"C:\vaishnav\projects\Main_Project\AFib_Project\reports\figures"

# =============================================================================
# GLOBAL STYLE — Dark premium theme
# =============================================================================
BG_DARK    = '#0d1117'      # Page background
BG_PANEL   = '#161b22'      # Panel / card background
BG_ACCENT  = '#1f2937'      # Slightly lighter panel
COL_CYAN   = '#58a6ff'      # Headers
COL_GREEN  = '#3fb950'      # Normal rhythm / positive
COL_RED    = '#f85149'      # AF / negative / warning
COL_GOLD   = '#d29922'      # Highlight / labels
COL_WHITE  = '#e6edf3'      # Body text
COL_GRAY   = '#8b949e'      # Secondary text
COL_PURPLE = '#bc8cff'      # Formula / math
COL_ORANGE = '#ffa657'      # Code / values

TITLE_FONT  = {'fontsize': 22, 'fontweight': 'bold', 'color': COL_CYAN,  'fontfamily': 'monospace'}
HEAD1_FONT  = {'fontsize': 15, 'fontweight': 'bold', 'color': COL_GOLD}
HEAD2_FONT  = {'fontsize': 12, 'fontweight': 'bold', 'color': COL_CYAN}
BODY_FONT   = {'fontsize': 10, 'color': COL_WHITE, 'linespacing': 1.7}
SMALL_FONT  = {'fontsize': 9,  'color': COL_GRAY}
CODE_FONT   = {'fontsize': 9,  'color': COL_ORANGE, 'fontfamily': 'monospace'}
MATH_FONT   = {'fontsize': 10, 'color': COL_PURPLE, 'fontfamily': 'monospace'}


def new_page(pdf, title=None, subtitle=None, page_num=None, total_pages=None):
    """
    Creates a new blank dark-themed page and returns (fig, ax_main).

    Every page gets:
      - Dark background
      - Top header band with optional title
      - Bottom footer with page number and project name
    """
    fig = plt.figure(figsize=(11.69, 16.54))   # A4 in inches (landscape-friendly)
    fig.patch.set_facecolor(BG_DARK)

    # ── Header band ──────────────────────────────────────────────────────────
    if title:
        fig.text(0.5, 0.965, title,
                 ha='center', va='top',
                 fontsize=18, fontweight='bold',
                 color=COL_CYAN, fontfamily='monospace')
    if subtitle:
        fig.text(0.5, 0.945, subtitle,
                 ha='center', va='top', fontsize=11, color=COL_GRAY)

    # Horizontal divider line under header
    line = plt.Line2D([0.05, 0.95], [0.935, 0.935],
                      transform=fig.transFigure,
                      color=COL_GOLD, linewidth=1.5, alpha=0.7)
    fig.add_artist(line)

    # ── Footer ───────────────────────────────────────────────────────────────
    footer_line = plt.Line2D([0.05, 0.95], [0.038, 0.038],
                              transform=fig.transFigure,
                              color=COL_GRAY, linewidth=0.5, alpha=0.5)
    fig.add_artist(footer_line)

    project = "HRV Feature Extraction & Context-Aware AF Detection  |  CACHET-CADB  |  Python 3.13.7"
    fig.text(0.05, 0.022, project, fontsize=7.5, color=COL_GRAY, va='bottom')

    if page_num and total_pages:
        fig.text(0.95, 0.022, f"Page {page_num} of {total_pages}",
                 ha='right', fontsize=8, color=COL_GRAY, va='bottom')

    return fig


def draw_text_block(fig, x, y, lines, style=None, spacing=0.028):
    """
    Draws multiple lines of text on a figure.

    x, y      : figure-fraction coordinates (0-1)
    lines     : list of strings to draw
    style     : dict of text kwargs
    spacing   : vertical gap between lines (figure fraction)
    """
    if style is None:
        style = BODY_FONT
    cur_y = y
    for line in lines:
        fig.text(x, cur_y, line, va='top', **style)
        cur_y -= spacing
    return cur_y   # returns the y after the last line


def draw_box(fig, x, y, w, h, color=BG_PANEL, alpha=1.0, radius=0.01):
    """Draws a filled rounded-corner rectangle on the figure."""
    from matplotlib.patches import FancyBboxPatch
    ax_dummy = fig.add_axes([x, y, w, h], frameon=False)
    ax_dummy.set_xlim(0, 1)
    ax_dummy.set_ylim(0, 1)
    ax_dummy.axis('off')
    box = FancyBboxPatch((0, 0), 1, 1,
                          boxstyle=f"round,pad=0.02",
                          facecolor=color, edgecolor=COL_GRAY,
                          linewidth=0.5, alpha=alpha,
                          transform=ax_dummy.transAxes, clip_on=False)
    ax_dummy.add_patch(box)
    return ax_dummy


# =============================================================================
# SIMULATE ECG (reused from Phase01 script)
# =============================================================================
def simulate_ecg(duration=4.0, bpm=72, rhythm='normal', seed=42):
    np.random.seed(seed)
    fs = 256
    n  = int(duration * fs)
    t  = np.linspace(0, duration, n, endpoint=False)
    ecg = np.zeros(n)

    period = 60.0 / bpm
    jitter = 0.30 if rhythm == 'afib' else 0.05
    cur = 0.15
    beats = []
    while cur < duration - 0.3:
        beats.append(cur)
        nxt = period * (1 + np.random.normal(0, jitter))
        nxt = np.clip(nxt, period * 0.45, period * 2.0)
        cur += nxt

    for b in beats:
        # P
        if rhythm != 'afib':
            ecg += 0.15 * np.exp(-((t - (b - 0.18))**2) / (2*0.025**2))
        else:
            mask = np.abs(t - b) < period * 0.45
            ecg += 0.07 * np.sin(2*np.pi*6*t + np.random.uniform(0, 6.28)) * mask * 0.4
        # Q
        ecg += -0.08 * np.exp(-((t - (b - 0.03))**2) / (2*0.010**2))
        # R
        ecg +=  1.20 * np.exp(-((t - b)**2)           / (2*0.012**2))
        # S
        ecg += -0.15 * np.exp(-((t - (b + 0.035))**2) / (2*0.015**2))
        # T
        ecg +=  0.30 * np.exp(-((t - (b + 0.22))**2)  / (2*0.045**2))

    ecg += np.random.normal(0, 0.018, n)
    return t, ecg, beats


# =============================================================================
# PAGE 1 — COVER PAGE
# =============================================================================
def page_cover(pdf, pn, tot):
    fig = new_page(pdf, page_num=pn, total_pages=tot)

    # Big centred badge
    ax_badge = fig.add_axes([0.15, 0.55, 0.70, 0.35])
    ax_badge.set_facecolor(BG_PANEL)
    ax_badge.axis('off')
    for spine in ax_badge.spines.values():
        spine.set_visible(False)

    ax_badge.text(0.5, 0.90, 'PHASE 1', ha='center', va='top',
                  fontsize=42, fontweight='bold', color=COL_GOLD,
                  fontfamily='monospace', transform=ax_badge.transAxes)
    ax_badge.text(0.5, 0.62, 'Biomedical Fundamentals', ha='center', va='top',
                  fontsize=24, fontweight='bold', color=COL_WHITE,
                  transform=ax_badge.transAxes)
    ax_badge.text(0.5, 0.42, 'ECG Signal Processing & HRV Theory', ha='center',
                  va='top', fontsize=14, color=COL_GRAY,
                  transform=ax_badge.transAxes)

    divider = plt.Line2D([0.1, 0.9], [0.32, 0.32], transform=ax_badge.transAxes,
                          color=COL_GOLD, linewidth=1, alpha=0.5)
    ax_badge.add_artist(divider)

    ax_badge.text(0.5, 0.22, 'HRV Feature Extraction & Context-Aware', ha='center',
                  va='top', fontsize=11, color=COL_CYAN,
                  transform=ax_badge.transAxes)
    ax_badge.text(0.5, 0.10, 'False Positive Analysis for Atrial Fibrillation Detection', ha='center',
                  va='top', fontsize=11, color=COL_CYAN,
                  transform=ax_badge.transAxes)

    # Info grid
    info = [
        ('Dataset',  'CACHET-CADB  (24 subjects, ~122 hours)'),
        ('Python',   '3.13.7  |  NumPy 2.5.1  |  Matplotlib 3.11.0'),
        ('Phase',    '1 of 18  —  Biomedical Fundamentals'),
        ('Lessons',  '8 lessons  |  30 viva questions  |  4 mini assignments'),
    ]
    y0 = 0.50
    for label, val in info:
        fig.text(0.15, y0, f"  {label:<12}:", fontsize=10,
                 color=COL_GRAY, va='top', fontfamily='monospace')
        fig.text(0.34, y0, val, fontsize=10, color=COL_WHITE, va='top')
        y0 -= 0.035

    # Contents list
    fig.text(0.15, 0.38, 'CONTENTS', fontsize=13, fontweight='bold',
             color=COL_GOLD, va='top')
    contents = [
        '1.1  Heart Anatomy & Electrical Conduction',
        '1.2  ECG: What It Is and How It Works',
        '1.3  ECG Waves: P, QRS, T — Clinical Meaning',
        '1.4  RR Interval: Foundation of HRV',
        '1.5  Heart Rate Variability (HRV)',
        '1.6  Normal Rhythm vs Atrial Fibrillation',
        '1.7  ECG Noise Types & Motion Artifacts',
        '1.8  Signal Processing & Filtering Basics',
        '       + ECG Visualizations (6-panel Figure)',
        '       + HRV Statistics Demonstration',
        '       + Sampling Frequency Demo',
        '       + 30 Viva Questions with Answers',
    ]
    y0 = 0.355
    for item in contents:
        bullet = '▶' if item.startswith('1.') else '  '
        col = COL_WHITE if item.startswith('1.') else COL_GRAY
        fig.text(0.18, y0, f'{bullet}  {item.strip()}', fontsize=10,
                 color=col, va='top')
        y0 -= 0.030

    pdf.savefig(fig, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)


# =============================================================================
# PAGE 2 — HEART ANATOMY + ECG BASICS (Lesson 1.1 + 1.2)
# =============================================================================
def page_heart_and_ecg(pdf, pn, tot):
    fig = new_page(pdf,
                   title='LESSON 1.1 & 1.2  —  Heart Anatomy & The ECG',
                   subtitle='Understanding the source of the ECG signal',
                   page_num=pn, total_pages=tot)

    # ── Left column: Heart conduction ────────────────────────────────────────
    cx = 0.05
    y  = 0.910

    fig.text(cx, y, '1.1  HEART ANATOMY', **HEAD1_FONT, va='top')
    y -= 0.032

    anatomy = [
        ('Right Atrium (RA)', 'Receives deoxygenated blood from body'),
        ('Right Ventricle (RV)', 'Pumps blood to the LUNGS'),
        ('Left Atrium (LA)',  'Receives oxygenated blood from lungs'),
        ('Left Ventricle (LV)', 'Pumps blood to the ENTIRE BODY'),
    ]
    for chamber, func in anatomy:
        fig.text(cx+0.01, y, f'  ■  {chamber}', fontsize=10,
                 color=COL_CYAN, fontweight='bold', va='top')
        fig.text(cx+0.01, y-0.020, f'       → {func}', fontsize=9,
                 color=COL_WHITE, va='top')
        y -= 0.048

    y -= 0.010
    fig.text(cx, y, 'ELECTRICAL CONDUCTION PATH', **HEAD2_FONT, va='top')
    y -= 0.028

    path_steps = [
        ('SA Node',        '(Right Atrium)  →  Natural pacemaker, 60-100 BPM'),
        ('  ↓', ''),
        ('AV Node',        '(Junction)  →  Delays signal 120-200 ms'),
        ('  ↓', ''),
        ('Bundle of His',  '→  Fast pathway down the septum'),
        ('  ↓', ''),
        ('Bundle Branches','Left + Right  →  Each ventricle'),
        ('  ↓', ''),
        ('Purkinje Fibers','→  Fine network, fires every muscle cell'),
        ('  ↓', ''),
        ('Ventricles',     'CONTRACT  (bottom to top — most efficient)'),
    ]
    for label, desc in path_steps:
        if label.strip() == '↓':
            fig.text(cx+0.05, y, '↓', fontsize=14, color=COL_GOLD, va='top')
        else:
            fig.text(cx+0.01, y, f'  {label:<20}', fontsize=9.5,
                     color=COL_GOLD, fontweight='bold', va='top',
                     fontfamily='monospace')
            if desc:
                fig.text(cx+0.22, y, desc, fontsize=9, color=COL_WHITE, va='top')
        y -= 0.028

    # Intuition box
    y -= 0.010
    intuition = [
        '  REAL-WORLD INTUITION:',
        '  Normal heart = Stadium wave: ONE person starts,',
        '  wave spreads in organised, predictable pattern.',
        '  AF = Thousands of people standing randomly',
        '  — complete CHAOS, no effective pump action!',
    ]
    ax_box = fig.add_axes([cx, y-0.085, 0.42, 0.090])
    ax_box.set_facecolor('#1a2332')
    ax_box.axis('off')
    for i, line in enumerate(intuition):
        col = COL_GOLD if i == 0 else COL_WHITE
        ax_box.text(0.02, 0.92 - i*0.19, line, fontsize=9,
                    color=col, va='top', transform=ax_box.transAxes)

    # ── Right column: ECG basics ──────────────────────────────────────────────
    rx = 0.53
    y  = 0.910

    fig.text(rx, y, '1.2  WHAT IS AN ECG?', **HEAD1_FONT, va='top')
    y -= 0.030

    ecg_text = [
        'ECG = Electrocardiogram',
        '"Electro" = Electrical activity',
        '"Cardio"  = Heart',
        '"Gram"    = Recording (graph)',
        '',
        'A graphical recording of the heart\'s',
        'electrical activity over time.',
    ]
    y = draw_text_block(fig, rx, y, ecg_text, spacing=0.026)
    y -= 0.012

    fig.text(rx, y, 'HOW IT WORKS (step by step):', **HEAD2_FONT, va='top')
    y -= 0.028
    steps = [
        '① Heart fires electrical impulse',
        '② Signal spreads through body (like ripples)',
        '③ Skin electrodes detect tiny voltage (mV)',
        '④ Amplifier boosts the signal',
        '⑤ ADC samples at 256 Hz (256×/second)',
        '⑥ Computer stores: [0.02, 0.15, 1.23, ...]',
        '⑦ Plot Voltage (mV) vs Time (s) = ECG!',
    ]
    for step in steps:
        fig.text(rx+0.01, y, step, fontsize=10, color=COL_WHITE, va='top')
        y -= 0.028

    y -= 0.015
    fig.text(rx, y, 'SAMPLING FREQUENCY (Critical!)', **HEAD2_FONT, va='top')
    y -= 0.030

    sampling_lines = [
        'CACHET-CADB uses  fs = 256 Hz',
        '  256 measurements per second',
        '  1 sample every 1/256 = 3.9 ms',
        '  5-min recording = 76,800 data points',
    ]
    for line in sampling_lines:
        col = COL_ORANGE if 'CACHET' in line or '=' in line else COL_WHITE
        fig.text(rx+0.01, y, line, fontsize=10, color=col, va='top',
                 fontfamily='monospace')
        y -= 0.026

    y -= 0.015
    fig.text(rx, y, 'NYQUIST THEOREM:', **HEAD2_FONT, va='top')
    y -= 0.030
    nyquist = [
        '  fs  ≥  2 × f_max',
        '',
        '  ECG max frequency ≈ 40 Hz (QRS)',
        '  Minimum required fs = 2 × 40 = 80 Hz',
        '  CACHET-CADB: 256 Hz → 6.4× safety ✓',
    ]
    for line in nyquist:
        col = COL_PURPLE if '≥' in line or 'fs' == line.strip() else COL_WHITE
        if '≥' in line:
            fig.text(rx+0.01, y, line, fontsize=12, color=COL_PURPLE,
                     va='top', fontfamily='monospace', fontweight='bold')
        else:
            fig.text(rx+0.01, y, line, fontsize=10, color=col, va='top')
        y -= 0.030

    # Key formulas box
    y -= 0.010
    fig.text(rx, y, 'KEY FORMULAS:', **HEAD2_FONT, va='top')
    y -= 0.028
    formulas = [
        'HR (BPM) = 60,000 / RR (ms)',
        'RR (ms)  = 60,000 / HR (BPM)',
        'Time res = 1000 / fs  ms/sample',
    ]
    ax_f = fig.add_axes([rx, y-0.075, 0.42, 0.078])
    ax_f.set_facecolor('#1a1f2e')
    ax_f.axis('off')
    for i, f in enumerate(formulas):
        ax_f.text(0.04, 0.85 - i*0.30, f, fontsize=10, color=COL_PURPLE,
                  va='top', fontfamily='monospace', transform=ax_f.transAxes)

    # Mini assignment
    y -= 0.135
    ma_lines = [
        '  MINI ASSIGNMENT 1.1 & 1.2',
        '  Q1: HR = 85 BPM  →  RR = ?         A: 60,000/85 = 705.9 ms',
        '  Q2: RR = 1200 ms  →  HR = ?         A: 60,000/1200 = 50 BPM',
        '  Q3: 10-min ECG @ 256 Hz  →  samples? A: 10×60×256 = 153,600',
    ]
    ax_ma = fig.add_axes([0.05, y-0.06, 0.90, 0.070])
    ax_ma.set_facecolor('#1c2a1c')
    ax_ma.axis('off')
    for i, line in enumerate(ma_lines):
        col = COL_GREEN if i == 0 else COL_WHITE
        ax_ma.text(0.01, 0.88 - i*0.24, line, fontsize=9.5, color=col,
                   va='top', transform=ax_ma.transAxes,
                   fontfamily='monospace' if i == 0 else 'sans-serif')

    pdf.savefig(fig, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)


# =============================================================================
# PAGE 3 — PQRST WAVES + RR INTERVAL (Lesson 1.3 + 1.4)
# =============================================================================
def page_waves_and_rr(pdf, pn, tot):
    fig = new_page(pdf,
                   title='LESSON 1.3 & 1.4  —  ECG Waves & RR Interval',
                   subtitle='Understanding the PQRST complex and timing measurements',
                   page_num=pn, total_pages=tot)

    # ── ECG waveform diagram (top half) ──────────────────────────────────────
    ax = fig.add_axes([0.07, 0.68, 0.86, 0.23])
    ax.set_facecolor(BG_PANEL)

    t, ecg, beats = simulate_ecg(duration=1.5, bpm=78, rhythm='normal', seed=1)
    ax.plot(t*1000, ecg, color=COL_CYAN, linewidth=2.2, zorder=3)
    ax.axhline(0, color=COL_GRAY, linewidth=0.5, alpha=0.5, linestyle='--')
    ax.set_facecolor(BG_PANEL)
    ax.grid(True, color='#2d3748', alpha=0.6, linewidth=0.4)
    ax.tick_params(colors=COL_GRAY, labelsize=8)
    ax.set_xlabel('Time (ms)', color=COL_GRAY, fontsize=9)
    ax.set_ylabel('Amplitude (mV)', color=COL_GRAY, fontsize=9)
    ax.set_title('PQRST Complex with Clinical Annotations',
                 color=COL_GOLD, fontsize=11, fontweight='bold', pad=6)
    for sp in ax.spines.values():
        sp.set_color('#3d4f6e')

    # Annotations
    if beats:
        b = beats[0]
        bm = b * 1000
        astyle = dict(arrowstyle='->', color=COL_GOLD, lw=1.2)
        tstyle = dict(fontsize=8.5, color=COL_GOLD, ha='center',
                      fontweight='bold')

        ax.annotate('P Wave\n(Atrial depol.)',
                    xy=(bm-178, 0.15), xytext=(bm-320, 0.75),
                    arrowprops=astyle, **tstyle)
        ax.annotate('R Peak\n← Detect this!',
                    xy=(bm, 1.18), xytext=(bm+80, 1.6),
                    arrowprops=astyle, **tstyle)
        ax.annotate('T Wave\n(Ventricular repol.)',
                    xy=(bm+218, 0.28), xytext=(bm+330, 0.9),
                    arrowprops=astyle, **tstyle)
        ax.annotate('Q', xy=(bm-30, -0.08),
                    xytext=(bm-150, -0.55), arrowprops=astyle,
                    fontsize=9, color=COL_GOLD, ha='center')
        ax.annotate('S', xy=(bm+35, -0.14),
                    xytext=(bm+170, -0.6), arrowprops=astyle,
                    fontsize=9, color=COL_GOLD, ha='center')

        # PR interval bracket
        ax.annotate('', xy=(bm-28, -0.85), xytext=(bm-208, -0.85),
                    arrowprops=dict(arrowstyle='<->', color='#a29bfe', lw=1.8))
        ax.text(bm-115, -1.0, 'PR (120-200ms)', ha='center',
                fontsize=8, color='#a29bfe')

        # QRS bracket
        ax.annotate('', xy=(bm+65, -1.25), xytext=(bm-35, -1.25),
                    arrowprops=dict(arrowstyle='<->', color='#fd79a8', lw=1.8))
        ax.text(bm+15, -1.42, 'QRS (80-120ms)', ha='center',
                fontsize=8, color='#fd79a8')

    ax.set_xlim(0, 1450)
    ax.set_ylim(-1.6, 2.2)

    # ── Wave reference table ──────────────────────────────────────────────────
    fig.text(0.07, 0.665, 'WAVE REFERENCE TABLE', **HEAD1_FONT, va='top')

    headers = ['Wave', 'Represents', 'Duration', 'Amplitude', 'Clinical Significance']
    col_x   = [0.07, 0.16, 0.47, 0.58, 0.69]
    col_w   = [0.08, 0.30, 0.10, 0.10, 0.27]

    y_tbl = 0.638
    # Header row
    for hdr, cx in zip(headers, col_x):
        fig.text(cx, y_tbl, hdr, fontsize=9, fontweight='bold',
                 color=COL_GOLD, va='top')
    y_tbl -= 0.002

    # Thin line under header
    line = plt.Line2D([0.07, 0.96], [y_tbl, y_tbl],
                      transform=fig.transFigure,
                      color=COL_GOLD, linewidth=0.8, alpha=0.6)
    fig.add_artist(line)
    y_tbl -= 0.022

    rows = [
        ('P wave',   'Atrial depolarization',             '80-120 ms',  '0.1-0.3 mV', '⚠ ABSENT in AF'),
        ('PR itvl',  'AV node conduction delay',          '120-200 ms', '—',           'Prolonged = AV block'),
        ('Q wave',   'Septal depolarization',              '<40 ms',     '<0.25 mV',   'Deep Q = old infarct'),
        ('R wave',   'Ventricular depol. peak  ★ DETECT', '~10 ms',     '0.5-2.0 mV', 'Our timing reference'),
        ('S wave',   'Late ventricular depol.',            '~20 ms',     'Variable',   '—'),
        ('QRS cplx', 'Full ventricular contraction',      '80-120 ms',  '—',           'Wide = bundle block'),
        ('ST seg',   'Ventricular plateau',                '80-120 ms',  '~0 mV',      '⚠ Elevated = MI!'),
        ('T wave',   'Ventricular repolarization',        '160-320 ms', '0.1-0.5 mV', 'Inverted = ischemia'),
        ('QT itvl',  'Depol. + repol. combined',          '350-440 ms', '—',           'Long QT = arrhythmia risk'),
    ]
    for row in rows:
        cols_data = zip(row, col_x)
        for val, cx in cols_data:
            red_flag = '⚠' in val or 'ABSENT' in val or 'MI' in val
            col = COL_RED if red_flag else (COL_ORANGE if '★' in val else COL_WHITE)
            fig.text(cx, y_tbl, val, fontsize=8.5, color=col, va='top')
        y_tbl -= 0.024

    # ── RR Interval section ───────────────────────────────────────────────────
    y_rr = y_tbl - 0.018
    fig.text(0.07, y_rr, '1.4  RR INTERVAL', **HEAD1_FONT, va='top')
    y_rr -= 0.032

    # Formula box
    ax_f = fig.add_axes([0.07, y_rr - 0.075, 0.54, 0.072])
    ax_f.set_facecolor('#1a1f2e')
    ax_f.axis('off')
    formulas = [
        'RR (ms) = (Sample_R2 - Sample_R1) / fs × 1000',
        'Example:  R1=512, R2=769, fs=256 Hz',
        '  RR = (769-512)/256 × 1000 = 1003.9 ms  ≈  60 BPM',
    ]
    for i, f in enumerate(formulas):
        col = COL_PURPLE if i == 0 else (COL_GRAY if i == 1 else COL_GREEN)
        ax_f.text(0.02, 0.88 - i*0.30, f, fontsize=9.5, color=col,
                  va='top', fontfamily='monospace', transform=ax_f.transAxes)

    # RR range table
    y_rr2 = y_rr - 0.090
    fig.text(0.07, y_rr2, 'VALID RR RANGE:', fontsize=10,
             fontweight='bold', color=COL_CYAN, va='top')
    ranges = [
        ('Min RR = 300 ms', '200 BPM', 'Physiological maximum heart rate'),
        ('Max RR = 2000 ms', '30 BPM', 'Physiological minimum heart rate'),
    ]
    y_rr2 -= 0.028
    for rr, bpm, note in ranges:
        fig.text(0.09, y_rr2, f'  {rr:<22} → {bpm:<10} ({note})',
                 fontsize=9.5, color=COL_WHITE, va='top', fontfamily='monospace')
        y_rr2 -= 0.024

    # Normal vs AF pattern
    fig.text(0.64, y_rr, 'RR PATTERN COMPARISON:', fontsize=10,
             fontweight='bold', color=COL_CYAN, va='top')
    y_rr -= 0.028
    patterns = [
        ('NORMAL:', COL_GREEN, '856 | 862 | 851 | 869 | 845 ms', '(±30-50 ms variation)'),
        ('AF:',     COL_RED,   '612 | 845 | 534 | 923 | 1102 ms', '(±300-600 ms variation!)'),
    ]
    for label, col, vals, note in patterns:
        fig.text(0.64, y_rr, label, fontsize=10, fontweight='bold',
                 color=col, va='top')
        fig.text(0.78, y_rr, vals, fontsize=9, color=COL_WHITE, va='top',
                 fontfamily='monospace')
        fig.text(0.64, y_rr - 0.022, note, fontsize=8.5, color=COL_GRAY, va='top')
        y_rr -= 0.048

    pdf.savefig(fig, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)


# =============================================================================
# PAGE 4 — HRV + NORMAL vs AF (Lesson 1.5 + 1.6)
# =============================================================================
def page_hrv_and_af(pdf, pn, tot):
    fig = new_page(pdf,
                   title='LESSON 1.5 & 1.6  —  HRV & Atrial Fibrillation',
                   subtitle='What HRV measures and how AF breaks the normal pattern',
                   page_num=pn, total_pages=tot)

    # ── Left: HRV theory ─────────────────────────────────────────────────────
    cx, y = 0.05, 0.910
    fig.text(cx, y, '1.5  HEART RATE VARIABILITY (HRV)', **HEAD1_FONT, va='top')
    y -= 0.030

    fig.text(cx, y,
             'HRV = the variation in time between consecutive heartbeats.',
             fontsize=10, color=COL_WHITE, va='top')
    y -= 0.024
    fig.text(cx, y,
             'It is NOT heart rate — it is how much the heart rate CHANGES.',
             fontsize=10, color=COL_ORANGE, va='top', style='italic')
    y -= 0.036

    hrv_table = [
        ('HIGH HRV',    COL_GREEN,  'Healthy ANS, adaptable',     'Athletes, youth, relaxation'),
        ('LOW HRV',     COL_ORANGE, 'Poor ANS function, rigid',   'Heart failure, diabetes, stress'),
        ('CHAOTIC HRV', COL_RED,    'Pathological irregular',     'ATRIAL FIBRILLATION ← Our target'),
    ]
    for label, col, meaning, assoc in hrv_table:
        fig.text(cx+0.01, y, f'  {label:<14}', fontsize=10,
                 fontweight='bold', color=col, va='top', fontfamily='monospace')
        fig.text(cx+0.17, y, meaning, fontsize=10, color=COL_WHITE, va='top')
        fig.text(cx+0.17, y-0.020, f'  → {assoc}', fontsize=9,
                 color=COL_GRAY, va='top')
        y -= 0.050

    # HRV formulas
    y -= 0.010
    fig.text(cx, y, 'KEY HRV FORMULAS:', **HEAD2_FONT, va='top')
    y -= 0.028

    formulas = [
        ('MeanNN', 'mean(RR_i)',                                       'Average RR interval (ms)'),
        ('SDNN',   'std(RR_i) = sqrt[Σ(RR_i - MeanNN)² / (N-1)]',    'Overall variability'),
        ('RMSSD',  'sqrt[mean((RR_{i+1} - RR_i)²)]',                  'Beat-to-beat variability'),
        ('pNN50',  'count(|RR_{i+1}-RR_i|>50ms) / (N-1) × 100 %',    'High variability fraction'),
    ]
    ax_formulas = fig.add_axes([cx, y-0.135, 0.44, 0.135])
    ax_formulas.set_facecolor('#12151c')
    ax_formulas.axis('off')
    for i, (name, expr, meaning) in enumerate(formulas):
        row_y = 0.93 - i * 0.23
        ax_formulas.text(0.01, row_y, f'{name} =', fontsize=10,
                         color=COL_GOLD, fontweight='bold', va='top',
                         fontfamily='monospace',
                         transform=ax_formulas.transAxes)
        ax_formulas.text(0.14, row_y, expr, fontsize=8.5,
                         color=COL_PURPLE, va='top',
                         fontfamily='monospace',
                         transform=ax_formulas.transAxes)
        ax_formulas.text(0.14, row_y - 0.12, f'  # {meaning}',
                         fontsize=8, color=COL_GRAY, va='top',
                         fontfamily='monospace',
                         transform=ax_formulas.transAxes)

    # Actual output table
    y -= 0.155
    fig.text(cx, y, 'ACTUAL CODE OUTPUT (from your script):', **HEAD2_FONT, va='top')
    y -= 0.028

    ax_out = fig.add_axes([cx, y-0.165, 0.44, 0.165])
    ax_out.set_facecolor('#0d1117')
    ax_out.axis('off')
    output_lines = [
        ('NORMAL RHYTHM:', COL_GREEN, True),
        ('  MeanNN : 856.90 ms  →  70.0 BPM', COL_WHITE, False),
        ('  SDNN   :  10.33 ms  (low = regular)', COL_GREEN, False),
        ('  RMSSD  :  18.17 ms', COL_WHITE, False),
        ('  pNN50  :   0.0 %   ✓ Normal', COL_GREEN, False),
        ('', COL_GRAY, False),
        ('ATRIAL FIBRILLATION:', COL_RED, True),
        ('  MeanNN : 717.80 ms  →  83.6 BPM', COL_WHITE, False),
        ('  SDNN   : 200.65 ms  ⚠ 19× HIGHER!', COL_RED, False),
        ('  RMSSD  : 373.17 ms  ⚠ 20× HIGHER!', COL_RED, False),
        ('  pNN50  : 100.0  %   ⚠ All beats differ', COL_RED, False),
    ]
    for i, (line, col, bold) in enumerate(output_lines):
        ax_out.text(0.02, 0.96 - i*0.085, line, fontsize=8.5, color=col,
                    va='top', fontfamily='monospace',
                    fontweight='bold' if bold else 'normal',
                    transform=ax_out.transAxes)

    # ── Right: AF section ────────────────────────────────────────────────────
    rx, ry = 0.55, 0.910
    fig.text(rx, ry, '1.6  ATRIAL FIBRILLATION', **HEAD1_FONT, va='top')
    ry -= 0.030

    comparison = [
        ('Feature',       'Normal Rhythm',          'Atrial Fibrillation'),
        ('P waves',       'Present, regular',       '⚠ ABSENT (f-waves)'),
        ('QRS',           'Regular spacing',        '⚠ Irregular'),
        ('RR intervals',  'Small variation ±5%',   '⚠ CHAOTIC variation'),
        ('Heart rate',    '60-100 BPM',             '100-180 BPM'),
        ('AV node',       'Passes every SA impulse','Randomly passes impulses'),
        ('SDNN',          '30-100 ms',              '⚠ Often >150 ms'),
        ('pNN50',         '5-30%',                  '⚠ Often >50%'),
    ]
    col_rx = [rx, rx+0.18, rx+0.33]
    for i, row in enumerate(comparison):
        is_header = (i == 0)
        fw = 'bold' if is_header else 'normal'
        cols_c = [COL_GOLD, COL_GREEN, COL_RED] if is_header else [COL_GRAY, COL_WHITE, COL_RED if '⚠' in row[2] else COL_WHITE]
        for val, cx_col, cc in zip(row, col_rx, cols_c):
            fig.text(cx_col, ry, val, fontsize=9, color=cc,
                     va='top', fontweight=fw)
        ry -= 0.030

    # Stroke risk box
    ry -= 0.010
    ax_stroke = fig.add_axes([rx, ry-0.115, 0.40, 0.110])
    ax_stroke.set_facecolor('#2a0a0a')
    ax_stroke.axis('off')
    stroke_lines = [
        ('⚠ WHY AF CAUSES STROKES:', COL_RED),
        ('  Atria quiver → blood pools in', COL_WHITE),
        ('  left atrium → clot forms →', COL_WHITE),
        ('  clot travels to brain → STROKE', COL_RED),
        ('  5× HIGHER stroke risk than normal!', COL_ORANGE),
    ]
    for i, (line, col) in enumerate(stroke_lines):
        ax_stroke.text(0.02, 0.90 - i*0.18, line, fontsize=9.5, color=col,
                       va='top', fontweight='bold' if i == 0 else 'normal',
                       transform=ax_stroke.transAxes)

    # ECG mini-plots: normal vs AF side by side
    ry -= 0.130
    # Normal ECG
    ax_n = fig.add_axes([rx, ry-0.150, 0.18, 0.140])
    ax_n.set_facecolor(BG_PANEL)
    t_n, e_n, _ = simulate_ecg(duration=3.0, bpm=70, rhythm='normal', seed=5)
    ax_n.plot(t_n, e_n, color=COL_GREEN, linewidth=1.5)
    ax_n.set_title('Normal', color=COL_GREEN, fontsize=9, fontweight='bold', pad=3)
    ax_n.axis('off')

    # AF ECG
    ax_a = fig.add_axes([rx+0.22, ry-0.150, 0.18, 0.140])
    ax_a.set_facecolor(BG_PANEL)
    t_a, e_a, _ = simulate_ecg(duration=3.0, bpm=90, rhythm='afib', seed=7)
    ax_a.plot(t_a, e_a, color=COL_RED, linewidth=1.5)
    ax_a.set_title('Atrial Fibrillation', color=COL_RED, fontsize=9,
                   fontweight='bold', pad=3)
    ax_a.axis('off')

    pdf.savefig(fig, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)


# =============================================================================
# PAGE 5 — NOISE TYPES + SIGNAL PROCESSING (Lesson 1.7 + 1.8)
# =============================================================================
def page_noise_and_filtering(pdf, pn, tot):
    fig = new_page(pdf,
                   title='LESSON 1.7 & 1.8  —  ECG Noise & Signal Processing',
                   subtitle='Understanding noise types and how filtering cleans the ECG',
                   page_num=pn, total_pages=tot)

    # ── Noise demo plot ───────────────────────────────────────────────────────
    ax_noise = fig.add_axes([0.07, 0.73, 0.86, 0.18])
    ax_noise.set_facecolor(BG_PANEL)

    np.random.seed(42)
    fs = 256
    t, clean, _ = simulate_ecg(duration=3.0, bpm=72, rhythm='normal', seed=42)
    n = len(t)

    baseline = 0.40 * np.sin(2 * np.pi * 0.15 * t)
    powerline = 0.15 * np.sin(2 * np.pi * 50 * t)
    muscle = np.zeros(n)
    m = (t > 1.5) & (t < 2.5)
    muscle[m] = 0.35 * np.random.normal(0, 1, m.sum())
    noisy = clean + baseline + powerline + muscle

    ax_noise.plot(t, noisy, color=COL_RED, linewidth=1.2, alpha=0.85,
                  label='Noisy ECG', zorder=2)
    ax_noise.plot(t, clean, color=COL_GREEN, linewidth=1.8, alpha=0.8,
                  label='Clean ECG', linestyle='--', zorder=3)
    ax_noise.axvspan(0,   0.6,  alpha=0.07, color=COL_GOLD,   label='Baseline wander region')
    ax_noise.axvspan(0.6, 1.5,  alpha=0.07, color=COL_PURPLE, label='Powerline noise region')
    ax_noise.axvspan(1.5, 2.5,  alpha=0.10, color=COL_RED,    label='Muscle artifact region')

    # Annotations
    for xpos, txt, col in [(0.3, 'Baseline\nWander\n(0.05-0.5 Hz)', COL_GOLD),
                            (1.05, 'Powerline\nNoise\n(50 Hz)', COL_PURPLE),
                            (2.0, 'Muscle\nArtifact\n(20-2000 Hz)', COL_RED)]:
        ax_noise.text(xpos, 1.65, txt, ha='center', fontsize=8,
                      color=col, fontweight='bold')
        ax_noise.annotate('', xy=(xpos, 1.35), xytext=(xpos, 1.55),
                          arrowprops=dict(arrowstyle='->', color=col, lw=1.2))

    ax_noise.set_title('ECG Noise Types: Baseline Wander + Powerline + Muscle Artifact',
                        color=COL_GOLD, fontsize=10, fontweight='bold', pad=4)
    ax_noise.set_xlabel('Time (s)', color=COL_GRAY, fontsize=9)
    ax_noise.set_ylabel('Amplitude (mV)', color=COL_GRAY, fontsize=9)
    ax_noise.tick_params(colors=COL_GRAY, labelsize=8)
    ax_noise.set_ylim(-0.9, 2.0)
    ax_noise.legend(fontsize=7.5, facecolor=BG_ACCENT,
                    labelcolor=COL_WHITE, loc='lower right')
    for sp in ax_noise.spines.values():
        sp.set_color('#3d4f6e')

    # ── Noise reference table ─────────────────────────────────────────────────
    fig.text(0.07, 0.720, '1.7  NOISE REFERENCE TABLE', **HEAD1_FONT, va='top')

    noise_hdr  = ['Type', 'Source', 'Frequency', 'Visual', 'Filter Used']
    noise_cx   = [0.07, 0.20, 0.42, 0.54, 0.70]
    noise_rows = [
        ('Baseline Wander', 'Breathing / sweat / movement', '0.05-0.5 Hz', 'Slow drift', 'High-pass @ 0.5 Hz'),
        ('Powerline EMI',   'Electrical outlets (50 Hz India)', '50 Hz exactly', 'Regular buzz', 'Notch @ 50 Hz'),
        ('Muscle (EMG)',    'Skeletal muscle near electrode', '20-2000 Hz', 'Spiky noise', 'Low-pass @ 40 Hz'),
        ('Motion Artifact', 'Electrode moving on skin', '0.1-10 Hz', 'Large jumps', 'Bandpass filter'),
    ]
    y_t = 0.692
    for hdr, cx_h in zip(noise_hdr, noise_cx):
        fig.text(cx_h, y_t, hdr, fontsize=9.5, fontweight='bold',
                 color=COL_GOLD, va='top')
    y_t -= 0.025
    for row in noise_rows:
        for val, cx_h in zip(row, noise_cx):
            fig.text(cx_h, y_t, val, fontsize=8.5, color=COL_WHITE, va='top')
        y_t -= 0.023

    # False positive chain
    y_t -= 0.010
    fig.text(0.07, y_t, 'THE MOTION → FALSE POSITIVE CHAIN  (Our Research!)', **HEAD2_FONT, va='top')
    y_t -= 0.028

    chain = [
        ('Patient walks fast', COL_WHITE),
        ('  ↓  Electrode moves → motion artifact spikes', COL_ORANGE),
        ('  ↓  R-peak detector: "these look like R peaks!"', COL_ORANGE),
        ('  ↓  Spurious random RR intervals created', COL_RED),
        ('  ↓  SDNN↑↑  RMSSD↑↑  pNN50↑↑  (all AF-like!)', COL_RED),
        ('  ↓  Classifier predicts: AF  ← FALSE POSITIVE!', COL_RED),
        ('  →  CACHET-CADB records accelerometer (MAI)', COL_CYAN),
        ('     We prove: high MAI → more false positives', COL_CYAN),
    ]
    cx_c = 0.07
    for line, col in chain:
        bold = 'FALSE POSITIVE' in line or 'Research' in line or 'MAI' in line
        fig.text(cx_c, y_t, line, fontsize=9.5, color=col, va='top',
                 fontweight='bold' if bold else 'normal')
        y_t -= 0.025

    # ── Signal processing section ─────────────────────────────────────────────
    y_sp = y_t - 0.015
    fig.text(0.07, y_sp, '1.8  SIGNAL PROCESSING BASICS', **HEAD1_FONT, va='top')
    y_sp -= 0.030

    # Left: frequency map
    ax_freq = fig.add_axes([0.07, y_sp-0.155, 0.38, 0.150])
    ax_freq.set_facecolor('#12151c')
    ax_freq.axis('off')
    freq_bands = [
        (0.0,  0.5,  'Baseline wander',  COL_RED,    'REMOVE'),
        (0.5,  10.0, 'P wave / T wave',  COL_GREEN,  'KEEP'),
        (5.0,  40.0, 'QRS complex',      COL_CYAN,   'KEEP ★'),
        (50.0, 50.5, 'Powerline noise',  COL_ORANGE, 'REMOVE'),
        (50.5, 60.0, 'Muscle noise',     COL_RED,    'REMOVE'),
    ]
    ax_freq.text(0.5, 0.96, 'ECG Frequency Content', ha='center',
                 fontsize=9, fontweight='bold', color=COL_GOLD,
                 transform=ax_freq.transAxes)
    for i, (flo, fhi, label, col, action) in enumerate(freq_bands):
        row_y = 0.82 - i * 0.17
        ax_freq.text(0.02, row_y,
                     f'{flo:>5.1f} - {fhi:>5.1f} Hz',
                     fontsize=9, color=COL_GRAY, va='top',
                     fontfamily='monospace', transform=ax_freq.transAxes)
        ax_freq.text(0.40, row_y, label, fontsize=9, color=col,
                     va='top', transform=ax_freq.transAxes)
        ax_freq.text(0.78, row_y, action, fontsize=8.5, color=col,
                     va='top', fontweight='bold',
                     transform=ax_freq.transAxes)

    # Right: Butterworth filter summary
    y_sp2 = y_sp - 0.015
    rx_sp = 0.52
    fig.text(rx_sp, y_sp2, 'BUTTERWORTH BANDPASS FILTER:', **HEAD2_FONT, va='top')
    y_sp2 -= 0.028

    butter_lines = [
        'Keeps frequencies: 0.5 Hz  to  40.0 Hz',
        'Removes: baseline wander + high-freq noise',
        'Order N = 4 (steeper cutoff than N=2)',
        '"Maximally flat" → no distortion in passband',
        'Implementation: scipy.signal.butter() + filtfilt()',
    ]
    for line in butter_lines:
        col = COL_ORANGE if 'scipy' in line else COL_WHITE
        fig.text(rx_sp+0.01, y_sp2, f'  • {line}', fontsize=9.5,
                 color=col, va='top')
        y_sp2 -= 0.027

    # Filter pipeline diagram
    y_sp2 -= 0.015
    pipeline = [
        'Raw ECG', '→ High-pass\n  0.5 Hz', '→ Low-pass\n  40 Hz',
        '→ Notch\n  50 Hz', '→ Clean ECG'
    ]
    cols_pipe = [COL_RED, COL_ORANGE, COL_GOLD, COL_CYAN, COL_GREEN]
    ax_pipe = fig.add_axes([rx_sp, y_sp2-0.065, 0.44, 0.062])
    ax_pipe.set_facecolor(BG_ACCENT)
    ax_pipe.axis('off')
    for i, (label, col) in enumerate(zip(pipeline, cols_pipe)):
        x_pos = 0.03 + i * 0.19
        ax_pipe.text(x_pos, 0.55, label, fontsize=8.5, color=col,
                     va='center', fontweight='bold',
                     transform=ax_pipe.transAxes)

    pdf.savefig(fig, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)


# =============================================================================
# PAGE 6 — FULL ECG VISUALIZATION FIGURE
# =============================================================================
def page_ecg_figure(pdf, pn, tot):
    """Embeds the full 6-panel ECG figure generated by Phase01 script."""
    fig = new_page(pdf,
                   title='GENERATED FIGURE  —  ECG Signal Analysis',
                   subtitle='6-panel educational visualization (output of Phase01_Biomedical_Fundamentals.py)',
                   page_num=pn, total_pages=tot)

    ecg_img_path = os.path.join(FIGURES_DIR, 'Phase01_ECG_Fundamentals.png')
    if os.path.exists(ecg_img_path):
        img = plt.imread(ecg_img_path)
        ax_img = fig.add_axes([0.04, 0.055, 0.92, 0.860])
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_facecolor(BG_DARK)
    else:
        fig.text(0.5, 0.5,
                 'Figure not found.\nRun Phase01_Biomedical_Fundamentals.py first.',
                 ha='center', va='center', fontsize=14, color=COL_RED)

    pdf.savefig(fig, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)


# =============================================================================
# PAGE 7 — SAMPLING FREQUENCY FIGURE
# =============================================================================
def page_sampling_figure(pdf, pn, tot):
    fig = new_page(pdf,
                   title='GENERATED FIGURE  —  Sampling Frequency Demo',
                   subtitle='Why CACHET-CADB uses 256 Hz — effect on QRS peak capture',
                   page_num=pn, total_pages=tot)

    samp_img_path = os.path.join(FIGURES_DIR, 'Phase01_Sampling.png')
    if os.path.exists(samp_img_path):
        img = plt.imread(samp_img_path)
        ax_img = fig.add_axes([0.04, 0.35, 0.92, 0.55])
        ax_img.imshow(img)
        ax_img.axis('off')
    else:
        fig.text(0.5, 0.55, 'Figure not found.', ha='center', fontsize=14, color=COL_RED)

    # Explanation below the image
    y_ex = 0.330
    fig.text(0.07, y_ex, 'INTERPRETATION:', **HEAD2_FONT, va='top')
    y_ex -= 0.030
    explanations = [
        ('50 Hz  (Red)',    COL_RED,    'Only 5 samples/100ms → completely misses peak height. 7.5× error in amplitude!'),
        ('128 Hz (Orange)', COL_ORANGE, '13 samples/100ms → close but still misses true maximum. Acceptable minimum.'),
        ('256 Hz (Green)',  COL_GREEN,  '26 samples/100ms → accurately traces the QRS curve. CACHET-CADB uses this. ✓'),
    ]
    for label, col, desc in explanations:
        fig.text(0.07, y_ex, f'  {label}:', fontsize=10.5, color=col,
                 fontweight='bold', va='top', fontfamily='monospace')
        fig.text(0.30, y_ex, desc, fontsize=10, color=COL_WHITE, va='top')
        y_ex -= 0.032

    y_ex -= 0.012
    fig.text(0.07, y_ex,
             'At 50 Hz, the R peak might be detected at 0.2 mV instead of 1.5 mV  →  missed beats or false detections!',
             fontsize=10, color=COL_RED, va='top', style='italic')

    y_ex -= 0.045
    fig.text(0.07, y_ex, 'NYQUIST APPLIED TO ECG:', **HEAD2_FONT, va='top')
    y_ex -= 0.028
    nyq = [
        'ECG maximum frequency: ~40 Hz (QRS complex)',
        'Nyquist minimum:  2 × 40 = 80 Hz',
        'CACHET-CADB:      256 Hz  →  6.4× safety margin',
        'Time resolution:  1000/256 = 3.9 ms per sample',
    ]
    for line in nyq:
        col = COL_ORANGE if '256' in line else COL_WHITE
        fig.text(0.09, y_ex, f'  {line}', fontsize=10, color=col, va='top',
                 fontfamily='monospace')
        y_ex -= 0.028

    pdf.savefig(fig, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)


# =============================================================================
# PAGE 8 — VIVA Q&A BANK
# =============================================================================
def page_viva(pdf, pn, tot):
    fig = new_page(pdf,
                   title='VIVA QUESTION BANK  —  Phase 1  (30 Questions)',
                   subtitle='Prepare these for your final year viva examination',
                   page_num=pn, total_pages=tot)

    viva_qa = [
        # (Q, A)
        ('What does ECG stand for?',
         'Electrocardiogram — records the heart\'s electrical activity over time.'),
        ('Name the 4 chambers of the heart.',
         'Right Atrium, Right Ventricle, Left Atrium, Left Ventricle.'),
        ('What is the SA node?',
         'Sinoatrial node — the natural pacemaker in the right atrium; fires at 60-100 BPM.'),
        ('Why does the AV node delay the signal?',
         'To allow the atria to finish contracting before the ventricles start.'),
        ('What does the P wave represent?',
         'Atrial depolarization — the electrical event causing atria to contract.'),
        ('What does the QRS complex represent?',
         'Ventricular depolarization — the main pumping contraction of the heart.'),
        ('What does the T wave represent?',
         'Ventricular repolarization — the heart muscle recovering and resetting.'),
        ('Why is the R wave used as the timing reference?',
         'It is the tallest and sharpest feature → highest signal-to-noise ratio.'),
        ('What is the RR interval formula?',
         'RR (ms) = (Sample_R2 - Sample_R1) / fs × 1000'),
        ('Calculate: R1=512, R2=769, fs=256. What is RR?',
         'RR = (769-512)/256 × 1000 = 1003.9 ms ≈ 60 BPM'),
        ('What is the valid RR range and why?',
         '300-2000 ms (= 200 BPM to 30 BPM physiological extremes). Outside = artifact.'),
        ('What is Heart Rate Variability (HRV)?',
         'The variation in time between consecutive heartbeats (NOT the heart rate itself).'),
        ('What does SDNN measure?',
         'Standard deviation of NN intervals — reflects overall autonomic variability.'),
        ('What does RMSSD measure?',
         'Root Mean Square of Successive Differences — reflects parasympathetic activity.'),
        ('What does pNN50 mean?',
         '% of consecutive RR differences >50 ms. High value = high variability.'),
        ('What is Atrial Fibrillation (AF)?',
         'A cardiac arrhythmia with chaotic atrial firing, absent P waves, and irregular RR intervals.'),
        ('Name the 3 ECG hallmarks of AF.',
         '1. Absent P waves (replaced by f-waves)  2. Irregular QRS  3. Chaotic RR intervals'),
        ('Why does AF cause strokes?',
         'Atria quiver → blood pools in left atrium → clot forms → travels to brain.'),
        ('What is the stroke risk increase in AF patients?',
         '5× higher stroke risk compared to normal sinus rhythm.'),
        ('What is sampling frequency? State it for CACHET-CADB.',
         'Number of samples per second (Hz). CACHET-CADB: 256 Hz.'),
        ('State the Nyquist theorem.',
         'To capture frequency f_max, sample at fs ≥ 2 × f_max. ECG: 80 Hz min.'),
        ('What is aliasing?',
         'When sampling rate is too low, high-freq signals appear as false low-freq signals.'),
        ('Name 4 ECG noise types and their frequencies.',
         '1. Baseline wander (0.05-0.5 Hz)  2. Powerline (50 Hz)  3. Muscle (20-2000 Hz)  4. Motion (0.1-10 Hz)'),
        ('What is a bandpass filter?',
         'Keeps frequencies within a range (0.5-40 Hz) and removes all others.'),
        ('Why use Butterworth filter for ECG?',
         'Maximally flat passband — does not distort the signal it keeps.'),
        ('What causes motion-related false positives?',
         'Physical movement → electrode motion → artifact peaks → spurious RR → AF-like HRV.'),
        ('What is MAI in CACHET-CADB?',
         'Movement Acceleration Index — quantifies motion from accelerometer data.'),
        ('What is paroxysmal AF?',
         'AF that comes and goes unpredictably; not always present when ECG is recorded.'),
        ('What is a wide QRS complex (>120ms)?',
         'Indicates bundle branch block — abnormal conduction pathway in ventricles.'),
        ('What does elevated ST segment indicate?',
         'Myocardial infarction (heart attack) — acute ischemia of ventricular muscle.'),
    ]

    # Two-column layout
    left_col  = viva_qa[:15]
    right_col = viva_qa[15:]

    def draw_qa_col(x_start, questions, y_start=0.905):
        y = y_start
        for i, (q, a) in enumerate(questions, 1):
            n_offset = 0 if x_start < 0.5 else 15
            fig.text(x_start, y, f'Q{i+n_offset}. {q}', fontsize=8.5,
                     color=COL_CYAN, va='top', fontweight='bold')
            y -= 0.022
            # Word-wrap answer manually at ~55 chars
            words = a.split()
            line, wrapped = '', []
            for w in words:
                if len(line) + len(w) + 1 <= 58:
                    line = (line + ' ' + w).strip()
                else:
                    wrapped.append(line)
                    line = w
            if line:
                wrapped.append(line)
            for wl in wrapped:
                fig.text(x_start + 0.01, y, f'   A: {wl}' if wl == wrapped[0] else f'      {wl}',
                         fontsize=8.2, color=COL_WHITE, va='top')
                y -= 0.019
            y -= 0.008
        return y

    draw_qa_col(0.04, left_col)
    draw_qa_col(0.52, right_col)

    pdf.savefig(fig, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)


# =============================================================================
# PAGE 9 — KEY NUMBERS + SUMMARY TABLE
# =============================================================================
def page_summary(pdf, pn, tot):
    fig = new_page(pdf,
                   title='PHASE 1 SUMMARY  —  Key Numbers & Completion Checklist',
                   subtitle='Everything you must know cold for viva and project implementation',
                   page_num=pn, total_pages=tot)

    # Key numbers table
    fig.text(0.07, 0.905, 'CRITICAL PARAMETERS TO MEMORIZE', **HEAD1_FONT, va='top')

    params = [
        ('Parameter',            'Value',         'Unit',  'Reason'),
        ('Sampling Frequency',   '256',           'Hz',    'CACHET-CADB standard'),
        ('Time Resolution',      '3.9',           'ms',    '1000/256'),
        ('Min Valid RR',         '300',           'ms',    '200 BPM max (physiological)'),
        ('Max Valid RR',         '2000',          'ms',    '30 BPM min (physiological)'),
        ('Bandpass Low',         '0.5',           'Hz',    'Remove baseline wander'),
        ('Bandpass High',        '40.0',          'Hz',    'Remove muscle noise'),
        ('Notch Frequency',      '50',            'Hz',    'India powerline interference'),
        ('HRV Window',           '5',             'min',   'Clinical Task Force standard'),
        ('Filter Order',         '4',             '—',     'Butterworth sharpness'),
        ('Normal SDNN',          '30-100',        'ms',    'Clinical reference range'),
        ('AF SDNN (typical)',    '>150',          'ms',    'Pathological range'),
        ('Normal pNN50',         '5-30',          '%',     'Clinical reference'),
        ('AF pNN50 (typical)',   '>50',           '%',     'Pathological range'),
        ('Total Subjects',       '24',            '—',     'CACHET-CADB dataset'),
        ('Python Version',       '3.13.7',        '—',     'This project requirement'),
    ]

    col_x_s = [0.07, 0.36, 0.52, 0.60]
    y_s = 0.872
    for i, row in enumerate(params):
        is_hdr = (i == 0)
        fw = 'bold' if is_hdr else 'normal'
        cols_c = [COL_GOLD]*4 if is_hdr else [COL_WHITE, COL_ORANGE, COL_GRAY, COL_GRAY]
        for val, cx in zip(row, col_x_s):
            fig.text(cx, y_s, val, fontsize=9.5 if is_hdr else 9,
                     color=cols_c[col_x_s.index(cx)],
                     va='top', fontweight=fw)
        if not is_hdr and i % 2 == 0:
            stripe = plt.Line2D([0.06, 0.95], [y_s+0.005, y_s+0.005],
                                transform=fig.transFigure,
                                color=BG_ACCENT, linewidth=8, alpha=0.4)
            fig.add_artist(stripe)
        y_s -= 0.027

    # Formula quick-reference
    y_f = y_s - 0.020
    fig.text(0.07, y_f, 'FORMULA QUICK-REFERENCE', **HEAD1_FONT, va='top')
    y_f -= 0.030

    ax_fref = fig.add_axes([0.07, y_f-0.110, 0.86, 0.110])
    ax_fref.set_facecolor('#12151c')
    ax_fref.axis('off')
    formula_pairs = [
        ('HR (BPM) = 60,000 / RR (ms)',
         'SDNN  = sqrt[Σ(RR_i - MeanNN)² / (N-1)]'),
        ('RR (ms) = (R2 - R1) / fs × 1000',
         'RMSSD = sqrt[mean((RR_{i+1} - RR_i)²)]'),
        ('Nyquist: fs ≥ 2 × f_max',
         'pNN50 = count(|ΔRR|>50ms) / (N-1) × 100'),
    ]
    for i, (left, right) in enumerate(formula_pairs):
        row_y = 0.88 - i * 0.30
        ax_fref.text(0.01, row_y, left, fontsize=9.5, color=COL_PURPLE,
                     va='top', fontfamily='monospace',
                     transform=ax_fref.transAxes)
        ax_fref.text(0.50, row_y, right, fontsize=9.5, color=COL_CYAN,
                     va='top', fontfamily='monospace',
                     transform=ax_fref.transAxes)

    # Completion checklist
    y_cl = y_f - 0.135
    fig.text(0.07, y_cl, 'PHASE 1 COMPLETION CHECKLIST', **HEAD1_FONT, va='top')
    y_cl -= 0.030

    checklist = [
        'I can explain what an ECG measures (not just "it measures the heart")',
        'I can name ALL PQRST waves and what they represent physiologically',
        'I can calculate RR interval from sample indices using the formula',
        'I can calculate heart rate from RR interval',
        'I can explain WHY the R wave is used (not P or T)',
        'I can explain HRV and why it differs between Normal and AF',
        'I can name 4 types of ECG noise and their frequency ranges',
        'I can state the Nyquist theorem and apply it to ECG',
        'I understand the motion → false positive chain (our research!)',
        'I can explain mechanistically why AF causes strokes',
    ]
    for item in checklist:
        fig.text(0.09, y_cl, f'☐  {item}', fontsize=10,
                 color=COL_WHITE, va='top')
        y_cl -= 0.030

    # Ready for Phase 2
    y_cl -= 0.020
    ax_next = fig.add_axes([0.07, y_cl-0.065, 0.86, 0.062])
    ax_next.set_facecolor('#1a2f1a')
    ax_next.axis('off')
    ax_next.text(0.5, 0.70, '▶  READY FOR PHASE 2: CACHET-CADB Dataset Study',
                 ha='center', va='center', fontsize=12, color=COL_GREEN,
                 fontweight='bold', transform=ax_next.transAxes)
    ax_next.text(0.5, 0.25,
                 'Complete all 10 checklist items and answer 20/30 viva questions before proceeding.',
                 ha='center', va='center', fontsize=9.5, color=COL_GRAY,
                 transform=ax_next.transAxes)

    pdf.savefig(fig, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)


# =============================================================================
# MAIN — ASSEMBLE PDF
# =============================================================================
def main():
    print('=' * 60)
    print('  GENERATING PHASE 1 PDF REPORT')
    print('=' * 60)

    TOTAL = 9  # Total pages

    with PdfPages(OUTPUT_PDF) as pdf:
        # Set PDF metadata
        d = pdf.infodict()
        d['Title']    = 'Phase 1: Biomedical Fundamentals — ECG & HRV'
        d['Author']   = 'AF Detection Project — CACHET-CADB'
        d['Subject']  = 'HRV Feature Extraction for Atrial Fibrillation Detection'
        d['Keywords'] = 'ECG, HRV, AF, Atrial Fibrillation, CACHET-CADB, Python 3.13.7'
        d['Creator']  = 'Python 3.13.7 + Matplotlib 3.11.0'

        pages = [
            ('Cover Page',            page_cover),
            ('Heart & ECG Basics',    page_heart_and_ecg),
            ('Waves & RR Interval',   page_waves_and_rr),
            ('HRV & AF',              page_hrv_and_af),
            ('Noise & Filtering',     page_noise_and_filtering),
            ('ECG Figure',            page_ecg_figure),
            ('Sampling Figure',       page_sampling_figure),
            ('Viva Questions',        page_viva),
            ('Summary & Checklist',   page_summary),
        ]

        for i, (name, fn) in enumerate(pages, 1):
            print(f'  [{i:02d}/{TOTAL}] Generating: {name}...')
            fn(pdf, i, TOTAL)

    size_mb = os.path.getsize(OUTPUT_PDF) / (1024 * 1024)
    print()
    print('=' * 60)
    print('  PDF GENERATED SUCCESSFULLY!')
    print('=' * 60)
    print(f'  File   : {OUTPUT_PDF}')
    print(f'  Pages  : {TOTAL}')
    print(f'  Size   : {size_mb:.2f} MB')
    print('=' * 60)


if __name__ == '__main__':
    main()
