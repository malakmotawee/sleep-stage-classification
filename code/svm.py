import sys
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import joblib

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from utils import OUTPUTS_DIR

RANDOM_STATE = 42
TRAIN_RATIO  = 0.80
FS_INDICES   = [0, 1, 2, 3, 4, 5]   # Motion + HR + Clock
FS_NAME      = 'Motion + HR + Clock'


def make_svm_binary():
    return SVC(kernel='rbf', C=1.0, gamma='scale',
               probability=True, random_state=RANDOM_STATE)


def make_svm_3class():
    return SVC(kernel='rbf', C=1.0, gamma='scale',
               probability=True, decision_function_shape='ovr',
               random_state=RANDOM_STATE)


def pool(features_dict, subject_ids, task):
    X = np.vstack([features_dict[s]['X'][:, FS_INDICES] for s in subject_ids])
    y = np.hstack([features_dict[s]['y_binary' if task == 'binary' else 'y_3class']
                   for s in subject_ids])
    return X, y


def split_subjects(subject_ids, rng):
    ids = list(subject_ids)
    rng.shuffle(ids)
    n_train = max(1, int(len(ids) * TRAIN_RATIO))
    return ids[:n_train], ids[n_train:]


def metrics_binary(y_true, y_pred, y_proba):
    acc  = float(accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    auc  = float(roc_auc_score(y_true, y_proba[:, 1]))
    return acc, sens, spec, auc


def metrics_3class(y_true, y_pred, y_proba):
    acc  = float(accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    aucs = []
    for c in range(3):
        if y_bin[:, c].sum() > 0:
            aucs.append(roc_auc_score(y_bin[:, c], y_proba[:, c]))
    auc = float(np.mean(aucs)) if aucs else float('nan')
    return acc, sens, float('nan'), auc


def run_mc(features_dict, subject_ids, task, n_splits, clf_factory):
    rng  = np.random.default_rng(RANDOM_STATE)
    rows = []

    print(f'  Running {n_splits} Monte Carlo splits...', flush=True)
    for i in range(n_splits):
        if (i + 1) % 10 == 0:
            print(f'    split {i + 1}/{n_splits}', flush=True)

        train_ids, test_ids = split_subjects(subject_ids, rng)

        try:
            X_tr, y_tr = pool(features_dict, train_ids, task)
            X_te, y_te = pool(features_dict, test_ids,  task)
        except Exception:
            continue

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        try:
            clf = clf_factory()
            clf.fit(X_tr, y_tr)
            y_pred  = clf.predict(X_te)
            y_proba = clf.predict_proba(X_te)
        except Exception as e:
            print(f'    [warn] split {i} skipped: {e}')
            continue

        if task == 'binary':
            acc, sens, spec, auc = metrics_binary(y_te, y_pred, y_proba)
        else:
            acc, sens, spec, auc = metrics_3class(y_te, y_pred, y_proba)

        rows.append({
            'task':        task,
            'classifier':  'SVM',
            'feature_set': FS_NAME,
            'split':       i,
            'accuracy':    acc,
            'sensitivity': sens,
            'specificity': spec,
            'kappa':       float('nan'),
            'auc':         auc,
        })

    return pd.DataFrame(rows)


def print_comparison(df_existing, df_new, task):
    metrics = (['accuracy', 'sensitivity', 'specificity', 'auc']
               if task == 'binary'
               else ['accuracy', 'sensitivity', 'auc'])

    combined = pd.concat([df_existing, df_new], ignore_index=True)
    mask = (
        (combined['task']        == task) &
        (combined['feature_set'] == FS_NAME)
    )
    sub = combined[mask]

    if sub.empty:
        print(f'  (no data for task={task})')
        return

    summary = (sub.groupby('classifier')[metrics]
                  .agg(['mean', 'std'])
                  .round(4))

    order = sorted([c for c in summary.index if c != 'SVM']) + ['SVM']
    order = [c for c in order if c in summary.index]
    summary = summary.reindex(order)

    header = f'--- SVM vs Existing Classifiers ({task.upper()}) ---'
    print(f'\n{header}')
    print(summary.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='5 splits only for fast testing')
    args = parser.parse_args()

    feat_path = OUTPUTS_DIR / 'features.pkl'
    print(f'Loading {feat_path}...')
    with open(feat_path, 'rb') as f:
        feat_data = pickle.load(f)

    features_dict = feat_data['subjects']
    subject_ids   = list(features_dict.keys())
    print(f'{len(subject_ids)} subjects available')

    n_bin  = 5 if args.quick else 50
    n_3cls = 5 if args.quick else 20

    results_dir = OUTPUTS_DIR / 'results'
    models_dir  = OUTPUTS_DIR / 'models'

    print('\n[Binary: sleep/wake]')
    df_bin = run_mc(features_dict, subject_ids, 'binary', n_bin, make_svm_binary)

    print('\n[3-class: wake/NREM/REM]')
    df_3cls = run_mc(features_dict, subject_ids, '3class', n_3cls, make_svm_3class)

    bin_csv = results_dir / 'metrics_binary.csv'
    cls_csv = results_dir / 'metrics_3class.csv'

    df_bin_existing  = pd.read_csv(bin_csv)
    df_3cls_existing = pd.read_csv(cls_csv)

    if 'SVM' not in df_bin_existing['classifier'].values:
        pd.concat([df_bin_existing, df_bin], ignore_index=True) \
          .to_csv(bin_csv, index=False)
        print(f'\nAppended SVM → {bin_csv.name}')
    else:
        print(f'\nSVM already present in {bin_csv.name} — skipped')

    if 'SVM' not in df_3cls_existing['classifier'].values:
        pd.concat([df_3cls_existing, df_3cls], ignore_index=True) \
          .to_csv(cls_csv, index=False)
        print(f'Appended SVM → {cls_csv.name}')
    else:
        print(f'SVM already present in {cls_csv.name} — skipped')

    print_comparison(df_bin_existing,  df_bin,  'binary')
    print_comparison(df_3cls_existing, df_3cls, '3class')

    print('\nTraining final SVM models on all subjects...')
    for task, factory, fname in [
        ('binary', make_svm_binary,  'SVM_binary.pkl'),
        ('3class', make_svm_3class, 'SVM_3class.pkl'),
    ]:
        X_all, y_all = pool(features_dict, subject_ids, task)
        clf = factory()
        clf.fit(X_all, y_all)
        out = models_dir / fname
        joblib.dump(clf, out)
        print(f'  Saved {out}')

    print('\nDone.')


if __name__ == '__main__':
    main()
