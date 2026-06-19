import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_subject_ids, load_heart_rate, load_motion, load_labels, OUTPUTS_DIR


def compute_activity_1hz(t_motion, x, y, z, t_start, t_end):
    n_bins = int(np.ceil(t_end - t_start)) + 1
    t_bins = t_start + np.arange(n_bins, dtype=float)

    mask = (t_motion >= t_start - 1) & (t_motion <= t_end + 1)
    t = t_motion[mask];  vx = x[mask];  vy = y[mask];  vz = z[mask]

    if len(t) < 2:
        return t_bins, np.zeros(n_bins)

    order = np.argsort(t)
    t = t[order];  vx = vx[order];  vy = vy[order];  vz = vz[order]

    diffs = np.abs(np.diff(vx)) + np.abs(np.diff(vy)) + np.abs(np.diff(vz))
    bin_idx = np.floor(t[:-1] - t_start).astype(int)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)

    activity = np.zeros(n_bins)
    np.add.at(activity, bin_idx[valid], diffs[valid])
    return t_bins, activity


def interpolate_hr_1hz(t_hr, bpm, t_start, t_end):
    n_points = int(np.ceil(t_end - t_start)) + 1
    t_out = t_start + np.arange(n_points, dtype=float)

    mask = (t_hr >= t_start - 600) & (t_hr <= t_end + 600)
    t = t_hr[mask];  b = bpm[mask]

    if len(t) < 2:
        return t_out, np.full(n_points, np.nan)

    order = np.argsort(t)
    t = t[order];  b = b[order]
    _, uniq = np.unique(t, return_index=True)
    t = t[uniq];  b = b[uniq]

    hr_out = np.interp(t_out, t, b)
    return t_out, hr_out


def clock_proxy(t_epochs, t_start, T_night=8 * 3600):
    return np.cos(2 * np.pi * (t_epochs - t_start) / T_night)


def preprocess_subject(subject_id):
    print(f'  Subject {subject_id}', end=' ... ', flush=True)

    t_labels, stages = load_labels(subject_id)

    valid = stages != -1
    t_epochs = t_labels[valid].astype(float)
    valid_stages = stages[valid]

    if len(t_epochs) < 30:
        print(f'skip ({len(t_epochs)} valid epochs)')
        return None

    t_start = float(t_epochs[0])
    t_end   = float(t_epochs[-1]) + 30.0

    t_hr, bpm = load_heart_rate(subject_id)
    t_hr_1hz, hr_1hz = interpolate_hr_1hz(t_hr, bpm, t_start, t_end)

    nan_mask = np.isnan(hr_1hz)
    if nan_mask.any():
        ok = np.where(~nan_mask)[0]
        if len(ok) >= 2:
            hr_1hz = np.interp(np.arange(len(hr_1hz)), ok, hr_1hz[ok])
        else:
            hr_1hz[:] = 70.0

    hr_smoothed = gaussian_filter1d(hr_1hz, sigma=120.0)

    t_mot, mx, my, mz = load_motion(subject_id)
    t_act_1hz, activity_1hz = compute_activity_1hz(t_mot, mx, my, mz, t_start, t_end)
    activity_smoothed = gaussian_filter1d(activity_1hz, sigma=50.0)

    clk = clock_proxy(t_epochs, t_start)

    print(f'{len(t_epochs)} epochs', flush=True)
    return {
        'subject_id':        subject_id,
        'epoch_times':       t_epochs,
        'stages':            valid_stages,
        't_hr_1hz':          t_hr_1hz,
        'hr_smoothed':       hr_smoothed,
        't_activity_1hz':    t_act_1hz,
        'activity_smoothed': activity_smoothed,
        'clock_proxy':       clk,
        't_start':           t_start,
        't_end':             t_end,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='only 5 subjects for a fast test run')
    args = parser.parse_args()

    subject_ids = get_subject_ids()
    if args.quick:
        subject_ids = subject_ids[:5]
        print(f'[--quick] Using {len(subject_ids)} subjects')

    print(f'Preprocessing {len(subject_ids)} subjects...')
    results = {}
    skip_count = 0

    for sid in subject_ids:
        data = preprocess_subject(sid)
        if data is not None:
            results[sid] = data
        else:
            skip_count += 1

    print(f'\nDone. {len(results)} subjects processed, {skip_count} skipped.')

    print('\n--- Checkpoint: Per-subject epoch counts and class distribution ---')
    print(f'{"Subject":>12}  {"Epochs":>7}  {"Wake%":>6}  {"N1%":>5}  {"N2%":>5}  {"N3%":>5}  {"REM%":>5}')
    for sid, d in results.items():
        s = d['stages']
        n = len(s)
        def pct(v): return 100.0 * np.sum(s == v) / n
        print(f'{sid:>12}  {n:>7}  {pct(0):>5.1f}%  {pct(1):>4.1f}%  {pct(2):>4.1f}%  {pct(3):>4.1f}%  {pct(5):>4.1f}%')

    out_path = OUTPUTS_DIR / 'preprocessed.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
