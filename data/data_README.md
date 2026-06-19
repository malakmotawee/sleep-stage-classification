# Data

The dataset used in this project is the **sleep-accel** dataset by Walch et al. (2019), publicly available on PhysioNet.

## Download Instructions

1. Go to: https://physionet.org/content/sleep-accel/1.0.0/
2. Create a free PhysioNet account if you don't have one
3. Download the dataset and place the files in this folder

## Dataset Contents

Once downloaded, this folder should contain:
- `heart_rate/` — PPG-derived heart rate recordings per subject (`.txt` files)
- `motion/` — Tri-axial accelerometer recordings per subject (`.txt` files)
- `labels/` — PSG sleep stage labels per subject (`.txt` files)
- `demographics.csv` — Subject age, gender, and recording metadata

## Citation

Walch, O., Huang, Y., Forger, D., & Goldstein, C. (2019). Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device. *Sleep, 42*(12). https://doi.org/10.1093/sleep/zsz180

Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation, 101*(23), e215–e220.
