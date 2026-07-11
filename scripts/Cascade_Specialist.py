'''
This script contains the cascade specialist model

Data training and evaluation for the specialist cascade (Stage 1 + Stage 2) on the exoplanet dataset.
The dataset is split into three disjoint sets:
  SET 1 (60%) - trains Stage 1 only.
  SET 2 (20%) - Stage 1 has NOT seen it. We score it with Stage 1, pull the
                ambiguous (0.3-0.7) rows, and TRAIN the specialist on those.
                Honest ambiguity because Stage 1 never trained on these rows.
  SET 3 (20%) - held out from BOTH. The full cascade runs on it end-to-end;
                this is the honest arbiter, compared against the single-model
                baseline on this same set.

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from model_utils import BalancedXGBClassifier
from paths import EXOPLANET_FEATURES_CSV
import warnings
warnings.filterwarnings('ignore')

print('Loading dataset and building all engineered features...')
df = pd.read_csv(EXOPLANET_FEATURES_CSV)

# Engineered features
df['snr_x_depth'] = df['snr'] * df['depth']
df['sec_over_depth'] = df['sec_eclipse_depth'] / (df['depth'] + 1e-9)
df['duration_over_period'] = df['duration'] / df['period']
df['odd_even_x_sec'] = df['odd_even_diff'] * df['sec_eclipse_depth']
df['transit_shape_metric'] = df['depth'] / (df['duration'] + 1e-9)
df['flux_variability'] = df['std_oot'] / (df['archive_snr'] + 1e-9)
df['sec_depth_significance'] = df['sec_eclipse_depth'] / (df['out_of_transit_rms'] + 1e-9)

ignore_cols = ['label', 'kepid', 'disposition']
features = [f for f in df.columns if f not in ignore_cols]
y = df['label']

# --- THREE-WAY SPLIT (disjoint) -------------------------------------------
df_rest, df_test, y_rest, y_test = train_test_split(
    df, y, test_size=0.20, random_state=42, stratify=y)

df_s1, df_s2src, y_s1, y_s2src = train_test_split(
    df_rest, y_rest, test_size=0.25, random_state=42, stratify=y_rest)

print(f'SET 1 (Stage 1 train) : {len(df_s1)}')
print(f'SET 2 (specialist src): {len(df_s2src)}')
print(f'SET 3 (final test)    : {len(df_test)}')

# ------------------------------ Set 1 Training ------------------------------------------
print('\nTraining Stage 1 (XGBoost) on SET 1...')
stage1 = BalancedXGBClassifier(random_state=42, eval_metric='logloss',
                               n_estimators=300, max_depth=5, learning_rate=0.05)
stage1.fit(df_s1[features], y_s1)

# --- Ambiguous Cases Identification ---
s2src_probs = stage1.predict_proba(df_s2src[features])[:, 1]
amb_mask = (s2src_probs >= 0.3) & (s2src_probs <= 0.7)
df_amb_train = df_s2src[amb_mask]
y_amb_train = y_s2src[amb_mask]
print(f'\nAmbiguous cases found in SET 2 to train specialist on: {amb_mask.sum()}'
      f' ({amb_mask.sum()/len(df_s2src)*100:.1f}% of SET 2)')

if amb_mask.sum() < 30:
    print('WARNING: very few ambiguous training cases.')

# -------- Specialist Training --------
print('Training specialist Stage 2 on ambiguous cases only...')
specialist = BalancedXGBClassifier(random_state=42, eval_metric='logloss',
                                    n_estimators=300, max_depth=7, learning_rate=0.05)
specialist.fit(df_amb_train[features], y_amb_train)

# --- Run Full Cascade ---
print('\nRunning full cascade on SET 3 (final held-out test)...')
test_probs_s1 = stage1.predict_proba(df_test[features])[:, 1]

final_preds = []
intercepted = 0
for i in range(len(test_probs_s1)):
    p = test_probs_s1[i]
    if p < 0.3:
        final_preds.append(0)
    elif p > 0.7:
        final_preds.append(1)
    else:
        intercepted += 1
        row = df_test.iloc[[i]][features]
        sp = specialist.predict_proba(row)[:, 1][0]
        final_preds.append(1 if sp > 0.5 else 0)

print(f'Targets intercepted and routed to specialist: {intercepted}')

print('\n' + '=' * 55)
print('TRUE SPECIALIST CASCADE')
print('=' * 55)
print(f'Test set size : {len(y_test)}')
print(f'Accuracy      : {accuracy_score(y_test, final_preds) * 100:.2f}%')
print('\nClassification Report:')
print(classification_report(y_test, final_preds))

# --- Single-Model Baseline ---
print('Training single-model baseline (SET 1) for comparison on SET 3...')
baseline = BalancedXGBClassifier(random_state=42, eval_metric='logloss',
                                 n_estimators=300, max_depth=5, learning_rate=0.05)
baseline.fit(df_s1[features], y_s1)
base_probs = baseline.predict_proba(df_test[features])[:, 1]
base_preds = (base_probs > 0.5).astype(int)

print('\n' + '=' * 55)
print(' SINGLE-MODEL BASELINE (same SET 1 train, same SET 3 test) ')
print('=' * 55)
print(f'Accuracy : {accuracy_score(y_test, base_preds) * 100:.2f}%')
print(f'ROC-AUC  : {roc_auc_score(y_test, base_probs):.4f}')
