'''
This script performs a baseline evaluation of a single combined XGB model trained on all 32 features (including the newly engineered features) to establish a benchmark for comparison with the cascading vetting pipeline.
The script is meant to test the validity of the cascading approach by comparing its performance against a single model trained on the same dataset and features.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from model_utils import BalancedXGBClassifier
from paths import EXOPLANET_FEATURES_CSV
import warnings
warnings.filterwarnings('ignore')

print('Loading dataset and building all 32 features...')
df = pd.read_csv(EXOPLANET_FEATURES_CSV)

# Synthetic features
df['snr_x_depth'] = df['snr'] * df['depth']
df['sec_over_depth'] = df['sec_eclipse_depth'] / (df['depth'] + 1e-9)
df['duration_over_period'] = df['duration'] / df['period']
df['odd_even_x_sec'] = df['odd_even_diff'] * df['sec_eclipse_depth']
df['transit_shape_metric'] = df['depth'] / (df['duration'] + 1e-9)
df['flux_variability'] = df['std_oot'] / (df['archive_snr'] + 1e-9)
df['sec_depth_significance'] = df['sec_eclipse_depth'] / (df['out_of_transit_rms'] + 1e-9)

# Filter out the label and kepid columns for training
ignore_cols = ['label', 'kepid', 'disposition']
all_features = [f for f in df.columns if f not in ignore_cols]
print(f'Single model will train on {len(all_features)} features.')

X = df[all_features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


single_model = BalancedXGBClassifier(
    random_state=42,
    eval_metric='logloss',
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05
)

print('Training single combined model on all 32 features...')
single_model.fit(X_train, y_train)

probs = single_model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)

print('\n' + '=' * 55)
print(' SINGLE COMBINED MODEL — BASELINE (all 32 features) ')
print('=' * 55)
print(f'Test set size      : {len(y_test)}')
print(f'Accuracy           : {accuracy_score(y_test, preds) * 100:.2f}%')
print(f'ROC-AUC            : {roc_auc_score(y_test, probs):.4f}')
print('\nClassification Report:')
print(classification_report(y_test, preds))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, preds))

