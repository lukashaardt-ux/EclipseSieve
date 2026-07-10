'''
This script performs feature engineering on the exoplanet dataset to create additional features that may improve the performance of the Stage 2 Interceptor model. It then trains an XGBoost classifier on the expanded feature set and evaluates its performance on a test set.
The engineered features include combinations and transformations of existing features, such as ratios, products, and significance metrics. 
The script saves the trained Stage 2 Interceptor model along with the list of features used for training.
'''

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from paths import EXOPLANET_FEATURES_CSV, ECLIPSE_SIEVE_STAGE2_PKL
import joblib
import warnings
warnings.filterwarnings('ignore')

# Data loading
print('Loading master dataset...')
df = pd.read_csv(EXOPLANET_FEATURES_CSV)

df['snr_x_depth'] = df['snr'] * df['depth']
df['sec_over_depth'] = df['sec_eclipse_depth'] / (df['depth'] + 1e-9)
df['duration_over_period'] = df['duration'] / df['period']
df['odd_even_x_sec'] = df['odd_even_diff'] * df['sec_eclipse_depth']
df['transit_shape_metric'] = df['depth'] / (df['duration'] + 1e-9)
df['flux_variability'] = df['std_oot'] / (df['archive_snr'] + 1e-9)
df['sec_depth_significance'] = df['sec_eclipse_depth'] / (df['out_of_transit_rms'] + 1e-9)

ignore_cols = ['label', 'kepid', 'disposition']
stage2_features = [f for f in df.columns if f not in ignore_cols]

X = df[stage2_features]
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'\nTraining Stage 2 Intercept Gate on {len(stage2_features)} features...')
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

# We use an XGBoost model specifically tuned to look at the expanded feature set
interceptor_xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_weight,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05
)

interceptor_xgb.fit(X_train, y_train)

preds = interceptor_xgb.predict(X_test)
print('\n' + '='*50)
print(' STAGE 2 INTERCEPTOR BASELINE PERFORMANCE ')
print('='*50)
print(f'Accuracy : {accuracy_score(y_test, preds)*100:.2f}%')
print('\nClassification Report:')
print(classification_report(y_test, preds))

# 8. Export models
joblib.dump({
    'model': interceptor_xgb,
    'features': stage2_features
}, ECLIPSE_SIEVE_STAGE2_PKL)

print(f"\n[SUCCESS] Stage 2 Intercept Gate saved to '{ECLIPSE_SIEVE_STAGE2_PKL}'")