import sys
import argparse
import pickle
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              confusion_matrix)
from sklearn.preprocessing import label_binarize
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
from utils import OUTPUTS_DIR, FEATURE_SETS, FEATURE_NAMES, get_subject_ids

FIGURES_DIR = OUTPUTS_DIR / 'figures'
RESULTS_DIR = OUTPUTS_DIR / 'results'
REPORT_DIR  = Path(__file__).parent.parent / 'report'
DPI = 150
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']   # one colour per classifier
FS_PALETTE = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']  # one colour per feature set
CLASSIFIERS = ['Logistic Regression', 'KNN', 'Random Forest', 'MLP']
RANDOM_STATE = 42


def load_data():
    with open(OUTPUTS_DIR / 'features.pkl', 'rb') as f:
        feat = pickle.load(f)
    df_bin  = pd.read_csv(RESULTS_DIR / 'metrics_binary.csv')
    df_3cls = pd.read_csv(RESULTS_DIR / 'metrics_3class.csv')
    with open(OUTPUTS_DIR / 'confusion_matrices.pkl', 'rb') as f:
        confusion = pickle.load(f)
    return feat, df_bin, df_3cls, confusion


def pool_all(feat_dict, subject_ids, fs_indices, task):
    X = np.vstack([feat_dict[s]['X'][:, fs_indices] for s in subject_ids])
    key = 'y_binary' if task == 'binary' else 'y_3class'
    y = np.hstack([feat_dict[s][key] for s in subject_ids])
    return X, y


def train_clf(name, X, y):
    params = {
        'Logistic Regression': {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs',
                                'multi_class': 'auto', 'random_state': RANDOM_STATE},
        'KNN':                 {'n_neighbors': 7},
        'Random Forest':       {'n_estimators': 100, 'max_depth': 10, 'random_state': RANDOM_STATE},
        'MLP':                 {'hidden_layer_sizes': (64, 32), 'max_iter': 500,
                                'random_state': RANDOM_STATE},
    }
    classes = {
        'Logistic Regression': LogisticRegression,
        'KNN':                 KNeighborsClassifier,
        'Random Forest':       RandomForestClassifier,
        'MLP':                 MLPClassifier,
    }
    clf = classes[name](**params[name])
    clf.fit(X, y)
    return clf


def fig_roc_binary(feat_dict, subject_ids):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ci, clf_name in enumerate(CLASSIFIERS):
        ax = axes[ci]
        for fi, fs in enumerate(FEATURE_SETS):
            X, y = pool_all(feat_dict, subject_ids, fs['indices'], 'binary')
            clf = train_clf(clf_name, X, y)
            y_score = clf.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_score)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=FS_PALETTE[fi], lw=2,
                    label=f'{fs["name"]} (AUC={auc_val:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(clf_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.02])
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.suptitle('ROC Curves — Binary (Sleep/Wake)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = FIGURES_DIR / 'fig_roc_binary.png'
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path.name}')


def fig_roc_3class(feat_dict, subject_ids):
    class_labels = ['Wake', 'NREM', 'REM']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ci, clf_name in enumerate(CLASSIFIERS):
        ax = axes[ci]
        for fi, fs in enumerate(FEATURE_SETS):
            X, y = pool_all(feat_dict, subject_ids, fs['indices'], '3class')
            clf = train_clf(clf_name, X, y)
            y_proba = clf.predict_proba(X)
            y_bin   = label_binarize(y, classes=[0, 1, 2])
            macro_auc = []
            for c in range(3):
                if y_bin[:, c].sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
                macro_auc.append(auc(fpr, tpr))
            avg_auc = np.mean(macro_auc)
            fpr_all = np.linspace(0, 1, 200)
            tpr_all = np.zeros_like(fpr_all)
            for c in range(3):
                if y_bin[:, c].sum() == 0:
                    continue
                fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], y_proba[:, c])
                tpr_all += np.interp(fpr_all, fpr_c, tpr_c)
            tpr_all /= 3
            ax.plot(fpr_all, tpr_all, color=FS_PALETTE[fi], lw=2,
                    label=f'{fs["name"]} (AUC={avg_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(clf_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('False Positive Rate');  ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.02])
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.suptitle('ROC Curves — 3-Class (Wake/NREM/REM), macro-average OvR', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = FIGURES_DIR / 'fig_roc_3class.png'
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path.name}')


def fig_precision_recall(feat_dict, subject_ids):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ci, clf_name in enumerate(CLASSIFIERS):
        ax = axes[ci]
        for fi, fs in enumerate(FEATURE_SETS):
            X, y = pool_all(feat_dict, subject_ids, fs['indices'], 'binary')
            clf = train_clf(clf_name, X, y)
            y_score_wake = clf.predict_proba(X)[:, 0]
            y_wake = (y == 0).astype(int)
            prec, rec, _ = precision_recall_curve(y_wake, y_score_wake)
            ax.plot(rec, prec, color=FS_PALETTE[fi], lw=2, label=fs['name'])

        ax.set_title(clf_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Recall (Wake)');  ax.set_ylabel('Precision (Wake)')
        ax.legend(fontsize=8)
        ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.02])
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.suptitle('Precision-Recall Curves (Wake as Positive Class)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = FIGURES_DIR / 'fig_precision_recall.png'
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path.name}')


def fig_bar_charts(df_bin, df_3cls):
    best_fs = FEATURE_SETS[3]['name']

    for metric, task_df, task_label, fname in [
        ('accuracy', df_bin,  'Binary (Sleep/Wake)', 'fig_accuracy_comparison.png'),
        ('auc',      df_bin,  'Binary (Sleep/Wake)', 'fig_auc_comparison.png'),
    ]:
        sub = task_df[task_df['feature_set'] == best_fs]
        means = sub.groupby('classifier')[metric].mean()
        stds  = sub.groupby('classifier')[metric].std()
        clfs  = [c for c in CLASSIFIERS if c in means.index]
        vals  = [means[c] for c in clfs]
        errs  = [stds[c]  for c in clfs]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(clfs, vals, yerr=errs, capsize=5,
                      color=PALETTE, edgecolor='black', width=0.5)
        ax.set_ylim(max(0, min(vals) - 0.1), 1.0)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.upper()} Comparison — {task_label}\n(Feature set: {best_fs})',
                     fontsize=13, fontweight='bold')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        path = FIGURES_DIR / fname
        plt.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f'  Saved {fname}')


def fig_confusion_matrix(confusion_dict):
    clf_order = CLASSIFIERS
    fig, axes = plt.subplots(1, len(clf_order), figsize=(16, 4))

    labels = ['Wake', 'NREM', 'REM']
    best_fs = FEATURE_SETS[3]['name']

    for ax, clf_name in zip(axes, clf_order):
        cm = None
        for cname, fs_dict in confusion_dict.items():
            if cname == clf_name:
                cm_raw = fs_dict.get(best_fs)
                if cm_raw is None:
                    cm_raw = next(iter(fs_dict.values()), None)
                if cm_raw is not None and len(cm_raw) > 0:
                    cm = np.round(np.array(cm_raw)).astype(int)
        if cm is None:
            cm = np.zeros((3, 3), dtype=int)

        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    cbar=False, linewidths=0.5, linecolor='gray')
        ax.set_title(clf_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    fig.suptitle('Confusion Matrices — 3-Class (Wake/NREM/REM)\n'
                 f'Feature set: {best_fs}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = FIGURES_DIR / 'fig_confusion_matrix.png'
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path.name}')


def fig_hypnogram(feat_dict, subject_ids):
    best_sid = max(subject_ids, key=lambda s: len(feat_dict[s]['y_binary']))
    subj     = feat_dict[best_sid]
    X, y_true = subj['X'][:, FEATURE_SETS[3]['indices']], subj['y_3class']

    clf = train_clf('Random Forest', X, y_true)
    y_pred = clf.predict(X)

    stage_map = {0: 'Wake', 1: 'NREM', 2: 'REM'}
    t = np.arange(len(y_true)) * 30 / 3600   # convert 30-s epochs → hours

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, y, title in [(ax1, y_true, 'True (PSG)'), (ax2, y_pred, 'Predicted (RF)')]:
        ax.step(t, y, where='post', lw=1.5, color='steelblue')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Wake', 'NREM', 'REM'])
        ax.set_ylabel(title, fontsize=11)
        ax.invert_yaxis()
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    ax2.set_xlabel('Time (hours)', fontsize=11)
    fig.suptitle(f'Hypnogram — Subject {best_sid} (Motion+HR+Clock, Random Forest)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = FIGURES_DIR / 'fig_hypnogram_sample.png'
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path.name}')


def fig_feature_importance(feat_dict, subject_ids):
    X, y = pool_all(feat_dict, subject_ids, FEATURE_SETS[3]['indices'], 'binary')
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feat_names  = [FEATURE_NAMES[i] for i in FEATURE_SETS[3]['indices']]
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(importances)), importances[order],
                  color=sns.color_palette('muted', len(importances)), edgecolor='black')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feat_names[i] for i in order], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Importance (MDI)', fontsize=12)
    ax.set_title('Random Forest Feature Importances\n(Motion + HR + Clock, Binary Task)',
                 fontsize=13, fontweight='bold')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    path = FIGURES_DIR / 'fig_feature_importance.png'
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path.name}')

if __name__ == '__main__':
    main()
