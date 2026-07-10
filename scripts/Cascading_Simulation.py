'''
This script creates a cascading vetting pipeline that first uses the Stage 1 XGB model to classify exoplanet candidates.
Candidates that are classified with high confidence (probability < 0.3 for false positive or probability > 0.7 for exoplanet) are accepted as final predictions.
Candidates that fall into the ambiguous range (0.3 <= probability <= 0.7) are routed to the Stage 2 Interceptor model for further evaluation.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model_utils import BalancedXGBClassifier
from paths import EXOPLANET_FEATURES_CSV, ECLIPSE_SIEVE_PKL, ECLIPSE_SIEVE_STAGE2_PKL
import joblib
import warnings
warnings.filterwarnings('ignore')

print('Loading dataset and creating all features...')
df = pd.read_csv(EXOPLANET_FEATURES_CSV)

df['snr_x_depth'] = df['snr'] * df['depth']
df['sec_over_depth'] = df['sec_eclipse_depth'] / (df['depth'] + 1e-9)
df['duration_over_period'] = df['duration'] / df['period']
df['odd_even_x_sec'] = df['odd_even_diff'] * df['sec_eclipse_depth']
df['transit_shape_metric'] = df['depth'] / (df['duration'] + 1e-9)
df['flux_variability'] = df['std_oot'] / (df['archive_snr'] + 1e-9)
df['sec_depth_significance'] = df['sec_eclipse_depth'] / (df['out_of_transit_rms'] + 1e-9)

y = df['label']

# Load Models
print('Loading Stage 1 (Broad Net) and Stage 2 (Interceptor)...')
stage1_data = joblib.load(ECLIPSE_SIEVE_PKL)
stage1_xgb = stage1_data['XGB']
stage1_features = stage1_data['features']

stage2_data = joblib.load(ECLIPSE_SIEVE_STAGE2_PKL)
stage2_xgb = stage2_data['model']
stage2_features = stage2_data['features']

_, X_test_full, _, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# --- THE CASCADE VETTING PROCESS ---

print('\nExecuting Cascading Pipeline...')

X_test_stage1 = X_test_full[stage1_features]
stage1_probs = stage1_xgb.predict_proba(X_test_stage1)[:, 1]

final_predictions = []
intercept_count = 0

# The Intercept Gate
for i in range(len(stage1_probs)):
    prob = stage1_probs[i]
    
    if prob < 0.3:
        final_predictions.append(0)  
    elif prob > 0.7:
        final_predictions.append(1) 
    
    # Ambiguous cases go to Stage 2 Interceptor
    else:
        intercept_count += 1
        target_stage2_data = X_test_full.iloc[[i]][stage2_features]
        
        stage2_prob = stage2_xgb.predict_proba(target_stage2_data)[:, 1][0]
        final_decision = 1 if stage2_prob > 0.5 else 0
        final_predictions.append(final_decision)

# --- RESULTS ---
print('\n' + '='*50)
print(' FINAL ECLIPSESIEVE CASCADING PIPELINE RESULTS ')
print('='*50)
print(f'Total Targets Evaluated : {len(y_test)}')
print(f'Targets Intercepted   : {intercept_count} (Ambiguous Cases Routed to Stage 2)')
print('-' * 50)
print(f'Final Pipeline Accuracy : {accuracy_score(y_test, final_predictions)*100:.2f}%')
print('\nFinal Classification Report:')
print(classification_report(y_test, final_predictions))