import sys
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              roc_auc_score, recall_score,
                              confusion_matrix)
from sklearn.preprocessing import label_binarize
import joblib

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from utils import OUTPUTS_DIR, FEATURE_SETS, CLASSIFIERS

N_SPLITS_BINARY = 50
N_SPLITS_3CLASS = 20
TRAIN_RATIO = 0.80
RANDOM_STATE = 42

BEST_PARAMS = {
    'Logistic Regression': {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'},
    'KNN':                 {'n_neighbors': 7},
    'Random Forest':       {'n_estimators': 100, 'max_depth': 10, 'random_state': RANDOM_STATE},
    'MLP':                 {'hidden_layer_sizes': (64, 32), 'max_iter': 500,
                            'random_state': RANDOM_STATE, 'early_stopping': True,
                            'validation_fraction': 0.1},
}


def make_clf(name, n_classes=2):
    params = dict(BEST_PARAMS[name])
    if name == 'Logistic Regression':
        params['multi_class'] = 'multinomial' if n_classes > 2 else 'ovr'
        params['random_state'] = RANDOM_STATE
    classes = {
        'Logistic Regression': LogisticRegression,
        'KNN':                 KNeighborsClassifier,
        'Random Forest':       RandomForestClassifier,
        'MLP':                 MLPClassifier,
    }
    return classes[name](**params)


def pool_subjects(features_dict, subject_ids, fs_indices, task):
    X = np.vstack([features_dict[s]['X'][:, fs_indices] for s in subject_ids])
    key = 'y_binary' if task == 'binary' else 'y_3class'
    y = np.hstack([features_dict[s][key] for s in subject_ids])
    return X, y


def split_subjects(subject_ids, rng):
    ids = list(subject_ids)
    rng.shuffle(ids)
    n_train = max(1, int(len(ids) * TRAIN_RATIO))
    return ids[:n_train], ids[n_train:]


def compute_auc(y_true, y_proba, n_classes):
    if n_classes == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    aucs = []
    for c in range(n_classes):
        if y_bin[:, c].sum() > 0:
            aucs.append(roc_auc_score(y_bin[:, c], y_proba[:, c]))
    return float(np.mean(aucs)) if aucs else 0.0


def run_monte_carlo(features_dict, subject_ids, task, n_splits):
    n_classes = 2 if task == 'binary' else 3
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    confusion_accumulator = {}

    print(f'  Running {n_splits} Monte Carlo splits (task={task})...')
    for split_idx in range(n_splits):
        if (split_idx + 1) % 10 == 0:
            print(f'    split {split_idx + 1}/{n_splits}', flush=True)

        train_ids, test_ids = split_subjects(subject_ids, rng)

        for fs in FEATURE_SETS:
            try:
                X_train, y_train = pool_subjects(features_dict, train_ids, fs['indices'], task)
                X_test,  y_test  = pool_subjects(features_dict, test_ids,  fs['indices'], task)
            except Exception:
                continue

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            for clf_name in CLASSIFIERS:
                try:
                    clf = make_clf(clf_name, n_classes)
                    clf.fit(X_train, y_train)
                    y_pred  = clf.predict(X_test)
                    y_proba = clf.predict_proba(X_test)
                except Exception:
                    continue

                acc  = float(accuracy_score(y_test, y_pred))
                kap  = float(cohen_kappa_score(y_test, y_pred))
                auc  = compute_auc(y_test, y_proba, n_classes)

                if task == 'binary':
                    sens = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
                    spec = float(recall_score(y_test, y_pred, pos_label=0, zero_division=0))
                else:
                    sens = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
                    spec = float('nan')
                    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
                    (confusion_accumulator
                     .setdefault(clf_name, {})
                     .setdefault(fs['name'], [])
                     .append(cm))

                rows.append({'task': task, 'classifier': clf_name,
                             'feature_set': fs['name'], 'split': split_idx,
                             'accuracy': acc, 'sensitivity': sens,
                             'specificity': spec, 'kappa': kap, 'auc': auc})

    return pd.DataFrame(rows), confusion_accumulator


def summarise(df):
    metrics = ['accuracy', 'sensitivity', 'specificity', 'kappa', 'auc']
    return (df.groupby(['classifier', 'feature_set'])[metrics]
              .agg(['mean', 'std']).round(4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    feat_path = OUTPUTS_DIR / 'features.pkl'
    print(f'Loading {feat_path}...')
    with open(feat_path, 'rb') as f:
        feat_data = pickle.load(f)

    features_dict = feat_data['subjects']
    subject_ids   = list(features_dict.keys())
    print(f'{len(subject_ids)} subjects available')

    n_bin  = 10 if args.quick else N_SPLITS_BINARY
    n_3cls = 5  if args.quick else N_SPLITS_3CLASS

    all_rows = []
    all_confusion = {}

    print('\n[Binary: sleep/wake]')
    df_bin, cm_bin = run_monte_carlo(features_dict, subject_ids, 'binary', n_bin)
    all_rows.append(df_bin)

    print('\n[3-class: wake/NREM/REM]')
    df_3cls, cm_3cls = run_monte_carlo(features_dict, subject_ids, '3class', n_3cls)
    all_rows.append(df_3cls)

    df_all = pd.concat(all_rows, ignore_index=True)

    results_dir = OUTPUTS_DIR / 'results'
    results_dir.mkdir(exist_ok=True)
    df_bin.to_csv(results_dir / 'metrics_binary.csv', index=False)
    df_3cls.to_csv(results_dir / 'metrics_3class.csv', index=False)
    print(f'\nSaved CSVs to {results_dir}')

    avg_confusion = {}
    for clf_name, fs_dict in cm_3cls.items():
        avg_confusion[clf_name] = {}
        for fs_name, cm_list in fs_dict.items():
            avg_confusion[clf_name][fs_name] = np.mean(cm_list, axis=0)

    with open(OUTPUTS_DIR / 'confusion_matrices.pkl', 'wb') as f:
        pickle.dump(avg_confusion, f)

    print('\nTraining final models on all data (Motion + HR + Clock)...')
    best_fs = FEATURE_SETS[3]
    models_dir = OUTPUTS_DIR / 'models'
    models_dir.mkdir(exist_ok=True)
    for task in ['binary', '3class']:
        n_classes = 2 if task == 'binary' else 3
        X_all, y_all = pool_subjects(features_dict, subject_ids, best_fs['indices'], task)
        for clf_name in CLASSIFIERS:
            try:
                clf = make_clf(clf_name, n_classes)
                clf.fit(X_all, y_all)
                fname = f'{clf_name.replace(" ","_")}_{task}.pkl'
                joblib.dump(clf, models_dir / fname)
                print(f'  Saved {fname}')
            except Exception as e:
                print(f'  Could not save {clf_name} {task}: {e}')

    print('\n--- Summary Table (mean over splits) ---')
    for task in ['binary', '3class']:
        df_task = df_all[df_all['task'] == task]
        print(f'\nTask: {task}')
        print(summarise(df_task).to_string())

    print('\nTraining complete.')


if __name__ == '__main__':
    main()
