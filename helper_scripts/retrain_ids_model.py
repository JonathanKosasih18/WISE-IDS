#!/usr/bin/env python3
"""
retrain_ids_model.py — Day 12 (item 3, option 3, retrain half): combine the
original Week-1 hping3/iperf3 training data with InSDN, retrain, and produce
a fair before/after comparison.

Design decisions (read before rerunning):
  - The original flows_labeled.csv (667 rows: 542 normal / 125 attack) is
    small and precious -- ALL of it goes into training, none held out. It's
    the only source of real Mininet/hping3-specific signal.
  - InSDN (343,889 rows) is split 70/30, STRATIFIED BY THE ORIGINAL MULTICLASS
    LABEL (Normal/DoS/DDoS/Probe/BFA/Web-Attack/BOTNET/U2R) so each attack
    type keeps its ratio in both halves. 70% joins training, 30% is held out
    and NEVER trained on.
  - Fairness fix vs. the earlier eval_insdn.py run: that script evaluated the
    baseline model against ALL of InSDN. If we now train on 70% of InSDN and
    then compare, evaluating "after" against all of InSDN would leak trained
    rows into the test set and inflate the improvement. So THIS script
    re-evaluates the ORIGINAL (baseline) model against the SAME held-out 30%
    split used for the retrained model, giving a true apples-to-apples
    before/after comparison. (Expect the baseline number here to be close to
    but not identical to eval_insdn.py's full-set number -- that's expected
    and fine, it's a different, smaller test set.)
  - proto_encoder is refit from scratch on the union of protocol strings
    across BOTH training sources (ICMP, TCP, UDP) -- this is what actually
    fixes the open proto_encoder/UDP bug, since the original was only ever
    fit on {ICMP, TCP} (flows_labeled.csv contains zero UDP rows).
  - Same RandomForestClassifier hyperparameters as the original notebook
    (max_depth=6, n_estimators=200, min_samples_leaf=3, random_state=42,
    class_weight='balanced') -- kept fixed so "more/better data" is the only
    variable being tested, not a hyperparameter change.
  - Outputs are saved as ids_v2.joblib / proto_encoder_v2.joblib /
    feature_cols_v2.txt -- NOT overwriting the original artifacts, so the
    Week-1 baseline stays reproducible.

Usage:
    python3 retrain_ids_model.py
"""

import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

UPLOADS = '.'
OUT_DIR = '.'

ORIGINAL_MODEL_PATH = f'{UPLOADS}/ids.joblib'
ORIGINAL_ENCODER_PATH = f'{UPLOADS}/proto_encoder.joblib'

INSDN_FILES = ['Normal_data.csv', 'OVS.csv', 'metasploitable-2.csv']
PROTO_REMAP = {0: 'ICMP', 6: 'TCP', 17: 'UDP'}

RF_PARAMS = dict(
    max_depth=6,
    n_estimators=200,
    min_samples_leaf=3,
    random_state=42,
    class_weight='balanced',
)


def load_insdn():
    frames = []
    for fname in INSDN_FILES:
        df = pd.read_csv(f'{UPLOADS}/{fname}', low_memory=False)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full['Label'] = full['Label'].str.strip()

    proto_str = full['Protocol'].map(PROTO_REMAP).fillna('unknown')

    out = pd.DataFrame({
        'proto': proto_str,
        'pps': full['Flow Pkts/s'],
        'bps': full['Flow Byts/s'],
        'duration': full['Flow Duration'] / 1e6,   # microseconds -> seconds
        'label': (full['Label'] != 'Normal').astype(int),
        'multiclass_label': full['Label'],
    })

    # Drop inf/nan rows (near-zero-duration flows -> divide-by-near-zero)
    n_before = len(out)
    finite_mask = np.isfinite(out[['pps', 'bps', 'duration']].to_numpy()).all(axis=1)
    out = out[finite_mask].reset_index(drop=True)
    n_dropped = n_before - len(out)
    print(f'InSDN: loaded {n_before:,} rows, dropped {n_dropped:,} inf/nan rows, '
          f'{len(out):,} remain')
    return out


def load_original():
    df = pd.read_csv(f'{UPLOADS}/flows_labeled.csv')
    out = pd.DataFrame({
        'proto': df['proto'],
        'pps': df['pps'],
        'bps': df['bps'],
        'duration': df['duration'],
        'label': df['label'],
        'multiclass_label': df['label'].map({0: 'Normal', 1: 'hping3_attack'}),
    })
    print(f'Original (flows_labeled.csv): {len(out):,} rows '
          f'({(out.label==0).sum()} normal / {(out.label==1).sum()} attack)')
    return out


def evaluate(model, proto_encoder, X, y_true, multiclass_labels, title):
    y_pred = model.predict(X)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print('=' * 60)
    print(title)
    print('=' * 60)
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1:        {f1:.4f}')
    print(f'Confusion matrix [rows=true, cols=pred], order=[normal, attack]:')
    print(cm)
    print()

    print(f"{'Label':<16}{'N':>10}{'Recall/FPR':>14}")
    for label in sorted(multiclass_labels.unique()):
        idx = (multiclass_labels == label).to_numpy()
        n = idx.sum()
        if label == 'Normal':
            fpr = (y_pred[idx] == 1).mean()
            print(f'{label:<16}{n:>10,}{fpr:>14.4f}   (FP rate)')
        else:
            rec = (y_pred[idx] == 1).mean()
            print(f'{label:<16}{n:>10,}{rec:>14.4f}   (recall)')
    print()
    return dict(precision=precision, recall=recall, f1=f1)


def main():
    print('Loading data sources...\n')
    original = load_original()
    insdn = load_insdn()
    print()

    # Stratified 70/30 split of InSDN ONLY, by multiclass label so each
    # attack type keeps its ratio on both sides.
    insdn_train, insdn_test = train_test_split(
        insdn, test_size=0.30, random_state=42, stratify=insdn['multiclass_label']
    )
    print(f'InSDN split: {len(insdn_train):,} train / {len(insdn_test):,} held-out test\n')

    # Combined training set: ALL of original + 70% of InSDN
    train_df = pd.concat([original, insdn_train], ignore_index=True)
    print(f'Combined training set: {len(train_df):,} rows '
          f'({(train_df.label==0).sum():,} normal / {(train_df.label==1).sum():,} attack)\n')

    # Refit proto_encoder on the union of protocol strings in the TRAINING set
    proto_encoder_v2 = LabelEncoder()
    proto_encoder_v2.fit(train_df['proto'].astype(str))
    print(f'proto_encoder_v2 classes: {list(proto_encoder_v2.classes_)}\n')

    train_df = train_df.copy()
    train_df['proto_enc'] = proto_encoder_v2.transform(train_df['proto'].astype(str))

    feature_cols = ['pps', 'bps', 'duration', 'proto_enc']
    X_train = train_df[feature_cols]
    y_train = train_df['label']

    print('Training retrained model (same hyperparameters as original)...')
    model_v2 = RandomForestClassifier(**RF_PARAMS)
    model_v2.fit(X_train, y_train)
    print('Done.\n')

    # ---- Prepare the held-out InSDN test set for BOTH models ----
    insdn_test = insdn_test.copy()

    # For the retrained model: use proto_encoder_v2 (should have zero fallback hits now)
    insdn_test['proto_enc_v2'] = proto_encoder_v2.transform(insdn_test['proto'].astype(str))
    X_test_v2 = insdn_test[['pps', 'bps', 'duration']].copy()
    X_test_v2['proto_enc'] = insdn_test['proto_enc_v2']

    # For the original (baseline) model: replicate its exact production fallback
    original_encoder = joblib.load(ORIGINAL_ENCODER_PATH)

    def encode_original(p):
        try:
            return int(original_encoder.transform([p])[0])
        except ValueError:
            return 0
    insdn_test['proto_enc_v1'] = insdn_test['proto'].apply(encode_original)
    X_test_v1 = insdn_test[['pps', 'bps', 'duration']].copy()
    X_test_v1['proto_enc'] = insdn_test['proto_enc_v1']

    y_test = insdn_test['label']
    mc_test = insdn_test['multiclass_label']

    original_model = joblib.load(ORIGINAL_MODEL_PATH)

    print('\n' + '#' * 60)
    print('# FAIR BEFORE/AFTER COMPARISON')
    print('# (both evaluated on the SAME held-out 30% InSDN test split,')
    print('#  which the retrained model never saw during training)')
    print('#' * 60 + '\n')

    before = evaluate(
        original_model, original_encoder, X_test_v1, y_test, mc_test,
        'BEFORE (original Week-1 model, hping3-only training)'
    )
    after = evaluate(
        model_v2, proto_encoder_v2, X_test_v2, y_test, mc_test,
        'AFTER (retrained model, hping3 + InSDN-70% training)'
    )

    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f"{'Metric':<12}{'Before':>10}{'After':>10}{'Delta':>10}")
    for k in ['precision', 'recall', 'f1']:
        print(f'{k:<12}{before[k]:>10.4f}{after[k]:>10.4f}{after[k]-before[k]:>+10.4f}')

    # ---- Save new artifacts (do NOT overwrite originals) ----
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(model_v2, f'{OUT_DIR}/ids_v2.joblib')
    joblib.dump(proto_encoder_v2, f'{OUT_DIR}/proto_encoder_v2.joblib')
    with open(f'{OUT_DIR}/feature_cols_v2.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f'\nSaved: {OUT_DIR}/ids_v2.joblib, proto_encoder_v2.joblib, feature_cols_v2.txt')


if __name__ == '__main__':
    main()
