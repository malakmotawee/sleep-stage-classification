import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import OUTPUTS_DIR, FEATURE_NAMES, FEATURE_SETS

WINDOW_HALF = 300


def extract_features_for_epoch(t_epoch, t_hr_1hz, hr_smoothed,
                                t_act_1hz, activity_smoothed, clock_val):
    t_lo = t_epoch - WINDOW_HALF
    t_hi = t_epoch + WINDOW_HALF

    act_mask = (t_act_1hz >= t_lo) & (t_act_1hz <= t_hi)
    act_win = activity_smoothed[act_mask]

    if len(act_win) < 2:
        act_mean = act_std = 0.0
    else:
        act_mean = float(np.mean(act_win))
        act_std  = float(np.std(act_win))

    hr_mask = (t_hr_1hz >= t_lo) & (t_hr_1hz <= t_hi)
    hr_win  = hr_smoothed[hr_mask]
    t_hr_win = t_hr_1hz[hr_mask]

    if len(hr_win) < 2:
        hr_std = hr_mean = hr_trend = 0.0
    else:
        hr_std  = float(np.std(hr_win))
        hr_mean = float(np.mean(hr_win))
        if len(hr_win) >= 3:
            slope, *_ = linregress(t_hr_win - t_epoch, hr_win)
            hr_trend = float(slope)
        else:
            hr_trend = 0.0

    return np.array([act_mean, act_std, hr_std, hr_mean, clock_val, hr_trend],
                    dtype=float)


def build_subject_features(data):
    t_epochs = data['epoch_times']
    t_hr_1hz = data['t_hr_1hz']
    hr_smoothed = data['hr_smoothed']
    t_act_1hz = data['t_activity_1hz']
    act_smoothed = data['activity_smoothed']
    clock_vals = data['clock_proxy']
    stages     = data['stages']

    n = len(t_epochs)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=float)

    for i in range(n):
        X[i] = extract_features_for_epoch(
            t_epochs[i], t_hr_1hz, hr_smoothed,
            t_act_1hz, act_smoothed, clock_vals[i]
        )

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_binary = np.where(stages == 0, 0, 1).astype(int)
    y_3class = np.zeros(n, dtype=int)
    y_3class[stages == 0] = 0
    y_3class[stages == 5] = 2
    y_3class[(stages > 0) & (stages != 5)] = 1

    return X, y_binary, y_3class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    pre_path = OUTPUTS_DIR / 'preprocessed.pkl'
    print(f'Loading {pre_path}...')
    with open(pre_path, 'rb') as f:
        preprocessed = pickle.load(f)

    subject_ids = list(preprocessed.keys())
    if args.quick:
        subject_ids = subject_ids[:5]

    print(f'Extracting features for {len(subject_ids)} subjects...')
    features = {'subjects': {}, 'feature_names': FEATURE_NAMES, 'feature_sets': FEATURE_SETS}

    for sid in subject_ids:
        data = preprocessed[sid]
        print(f'  {sid} ({len(data["epoch_times"])} epochs)', end=' ... ', flush=True)
        X, y_bin, y_3cls = build_subject_features(data)
        features['subjects'][sid] = {
            'X': X,
            'y_binary': y_bin,
            'y_3class': y_3cls,
        }
        print(f'wake={np.sum(y_bin==0)}  sleep={np.sum(y_bin==1)}')

    out_path = OUTPUTS_DIR / 'features.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(features, f)
    print(f'\nSaved to {out_path}')

    all_X = np.vstack([features['subjects'][s]['X'] for s in subject_ids])
    print(f'shape: {all_X.shape}')
    print(f'Feature matrix shape: {all_X.shape}')
    print(f'NaN count: {np.isnan(all_X).sum()}')
    print(f'Inf count: {np.isinf(all_X).sum()}')
    for i, name in enumerate(FEATURE_NAMES):
        print(f'  {name:25s}  mean={all_X[:,i].mean():.4f}  std={all_X[:,i].std():.4f}')


if __name__ == '__main__':
    main()
