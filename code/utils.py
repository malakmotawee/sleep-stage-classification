from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
OUTPUTS_DIR = ROOT / 'outputs'

FEATURE_NAMES = ['activity_mean', 'activity_std', 'hr_local_std', 'hr_mean', 'clock_proxy', 'hr_trend']

FEATURE_SETS = [
    {'name': 'Motion only',         'indices': [0, 1]},
    {'name': 'HR only',             'indices': [2, 3]},
    {'name': 'Motion + HR',         'indices': [0, 1, 2, 3]},
    {'name': 'Motion + HR + Clock', 'indices': [0, 1, 2, 3, 4, 5]},
]

CLASSIFIERS = ['Logistic Regression', 'KNN', 'Random Forest', 'MLP']


def get_subject_ids():
    labels_dir = DATA_DIR / 'labels'
    return sorted(f.stem.replace('_labeled_sleep', '')
                  for f in labels_dir.glob('*_labeled_sleep.csv'))


def load_heart_rate(subject_id):
    path = DATA_DIR / 'heart_rate' / f'{subject_id}_heartrate.csv'
    data = pd.read_csv(path, header=None, names=['t', 'bpm'], dtype=float).values
    return data[:, 0], data[:, 1]


def load_motion(subject_id):
    path = DATA_DIR / 'motion' / f'{subject_id}_acceleration.csv'
    data = pd.read_csv(path, header=None, names=['t', 'x', 'y', 'z'],
                       sep=r'\s+', engine='python', dtype=float).values
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def load_labels(subject_id):
    path = DATA_DIR / 'labels' / f'{subject_id}_labeled_sleep.csv'
    data = pd.read_csv(path, header=None, names=['t', 'stage'],
                       sep=r'\s+', engine='python', dtype=float).values
    return data[:, 0], data[:, 1].astype(int)
