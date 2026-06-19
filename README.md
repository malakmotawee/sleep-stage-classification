# Sleep Stage Classification from Apple Watch Data

Predicting sleep stages (Sleep/Wake and Wake/NREM/REM) from wrist-worn sensor data using machine learning — built on a real clinical dataset of 31 overnight recordings.

---

## Overview

Polysomnography (PSG) is the gold standard for sleep assessment but requires expensive lab equipment and trained technicians. This project investigates whether Apple Watch sensor data alone — accelerometer and heart rate — can reliably classify sleep stages using standard ML classifiers.

**Research question:** Can wrist-worn consumer wearables replace or approximate clinical sleep staging?

---

## Dataset

- **Source:** [sleep-accel dataset (Walch et al., 2019)](https://physionet.org/content/sleep-accel/1.0.0/) via PhysioNet
- **Subjects:** 31 healthy adults, one overnight recording each
- **Total labeled time:** ~223 hours
- **Epochs:** 26,773 labeled 30-second windows
- **Sensors:** Apple Watch Series 2 — tri-axial accelerometer (~50 Hz) + PPG-derived heart rate (~1 Hz)
- **Ground truth:** In-lab PSG labels (Wake, N1, N2, N3, REM)

---

## Features

Six features extracted per 30-second epoch using a ±5-minute rolling window:

| Feature | Source | Description |
|---|---|---|
| `activity_mean` | Accelerometer | Mean body movement in window |
| `activity_std` | Accelerometer | Variability of movement |
| `hr_mean` | Heart rate | Mean HR in window (bpm) |
| `hr_local_std` | Heart rate | Short-term HR variability |
| `hr_trend` | Heart rate | Direction of HR change (slope) |
| `clock_proxy` | Timestamp | Circadian position: cos(2π·t / 28,800) |

Features were z-score standardized per subject to remove inter-subject baseline differences.

Four feature combinations were tested: Motion only, HR only, Motion + HR, and Motion + HR + Clock.

---

## Models

Four scikit-learn classifiers were compared:

- Logistic Regression (L2, C=1.0)
- K-Nearest Neighbors (k=7)
- Random Forest (100 trees, max_depth=10)
- Multi-Layer Perceptron (hidden layers: 64→32, ReLU, early stopping)
- Support Vector Machine (RBF kernel, Platt scaling)

**Validation:** Monte Carlo cross-validation — 50 random 80/20 subject-level splits (binary), 20 splits (3-class).

---

## Results

### Binary Classification — Sleep vs. Wake (Motion + HR + Clock)

| Classifier | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.928 | 0.992 | 0.294 | 0.826 |
| KNN | 0.908 | 0.969 | 0.307 | 0.720 |
| Random Forest | 0.926 | 0.987 | 0.327 | **0.863** |
| MLP | 0.919 | 0.979 | 0.331 | 0.850 |

**Best AUC: Random Forest at 0.863** — within 1–2% of the original paper's neural net (AUC = 0.878).

### Three-Class Classification — Wake / NREM / REM

| Classifier | Accuracy | AUC |
|---|---|---|
| Logistic Regression | **0.718** | 0.754 |
| Random Forest | 0.709 | **0.769** |
| MLP | 0.679 | 0.741 |
| KNN | 0.645 | 0.665 |

Three-class classification is harder due to the challenge of separating NREM and REM from wrist sensors alone. Most errors occur at the NREM/REM boundary.

### Key Findings

- Adding heart rate to motion features consistently improved both accuracy and AUC across all classifiers
- The clock proxy (circadian signal) was the single most impactful feature for wake detection
- Motion-only features cannot distinguish REM from NREM — heart rate is essential for 3-class tasks
- Results closely replicate Walch et al. (2019) despite using simpler hyperparameters

---

## Tech Stack

- **Language:** Python
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn
- **Dataset:** PhysioNet (open access)
- **Validation:** Monte Carlo cross-validation (subject-level splits)

---

## Project Structure

```
sleep-stage-classification/
├── data/               # Instructions for downloading the PhysioNet dataset
├── notebooks/          # Main analysis notebook
├── report/             # Full project report (PDF)
├── figures/            # ROC curves, confusion matrices, hypnograms
└── README.md
```

---

## Authors

Malak Motawee & Judy Grida  
DSCI 3415 — Fundamentals of Machine Learning  
The American University in Cairo, May 2026

---

## Reference

Walch, O., Huang, Y., Forger, D., & Goldstein, C. (2019). Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device. *Sleep, 42*(12). https://doi.org/10.1093/sleep/zsz180
