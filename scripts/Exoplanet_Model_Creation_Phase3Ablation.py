'''
This script performs a systematic ablation study on the exoplanet classification model.
The goal is to evaluate the impact of different feature suites on the model's performance by systematically removing them and observing the resulting changes in accuracy, F1-score, and ROC-AUC.
The feature suites are based on domain expertise and prior analyses, and they include:
1. Stellar Context: Features related to the host star's properties.
2. Transit Geometry: Features describing the shape and characteristics of the transit event.
3. Noise and Signal: Features capturing the noise characteristics and signal strength of the light curve.
4. Eclipsing Binaries & Interference: Features that help identify potential false positives due to eclipsing binaries or other interference.
'''

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
import datetime
from paths import EXOPLANET_FEATURES_CSV, results_path
import warnings
warnings.filterwarnings('ignore')

#Load the dataset
print('Loading master dataset...')
df = pd.read_csv(EXOPLANET_FEATURES_CSV)

df['snr_x_depth'] = df['snr'] * df['depth']
df['sec_over_depth'] = df['sec_eclipse_depth'] / (df['depth'] + 1e-9)
df['duration_over_period'] = df['duration'] / df['period']
df['odd_even_x_sec'] = df['odd_even_diff'] * df['sec_eclipse_depth']
df['transit_shape_metric'] = df['depth'] / (df['duration'] + 1e-9)
df['flux_variability'] = df['std_oot'] / (df['archive_snr'] + 1e-9)
df['sec_depth_significance'] = df['sec_eclipse_depth'] / (df['out_of_transit_rms'] + 1e-9)

y = df['label']

# Feature suites based on domain expertise and prior analyses
feature_suites = {
    'Stellar Context': ['planet_radius', 'rp_over_rs', 'stellar_logg', 'stellar_teff'],
    'Transit Geometry': ['transit_shape_metric', 'archive_duration', 'depth', 'depth_ratio', 'duration', 'flux_asymmetry', 'kurtosis', 'skewness', 'symmetry'],
    'Noise and Signal': ['flux_variability','archive_snr', 'bls_power', 'cdpp', 'max_mes', 'max_ses', 'out_of_transit_rms', 'period', 'snr', 'std_oot', 'transit_snr_bls'],
    'Eclipsing Binaries & Interference': ['duration_over_period', 'impact_parameter', 'odd_even_diff', 'odd_even_x_sec', 'sec_eclipse_depth', 'sec_over_depth', 'snr_x_depth', 'sec_depth_significance']
}

all_features = [feature for suite in feature_suites.values() for feature in suite]

def train_tuned_RF(X_train, y_train):
    pipe = Pipeline([('model', RandomForestClassifier(class_weight='balanced', random_state=42))])
    grid = GridSearchCV(pipe, {'model__n_estimators': [200], 'model__max_depth': [20]}, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_tuned_XGB(X_train, y_train):
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipe = Pipeline([('model', XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_weight))])
    grid = GridSearchCV(pipe, {'model__n_estimators': [100, 300], 'model__max_depth': [5]}, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

ablation_records = []

# Flashcards
print('\n--- Running Master Control Group (All 32 Features) ---')
X = df[all_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RF_master = train_tuned_RF(X_train, y_train)
XGB_master = train_tuned_XGB(X_train, y_train)

baselines = {}

for name, model in [('Random Forest', RF_master), ('XGBoost', XGB_master)]:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    base_acc = accuracy_score(y_test, preds)
    base_f1 = f1_score(y_test, preds)
    base_auc = roc_auc_score(y_test, probs)

    baselines[name] = {'acc': base_acc, 'auc': base_auc}

    ablation_records.append({
        'Ablated_Suite': 'None (Control Baseline)',
        'Model': name,
        'Features_Remaining': len(all_features),
        'Accuracy': base_acc,
        'F1-Score': base_f1,
        'ROC-AUC': base_auc,
        'Accuracy_Delta': 0.0,
        'AUC_Delta': 0.0
    })

# Store baselines for Delta (The change in performance relative to the baseline) calculations
baselines = {
    'Random Forest': {'acc': base_acc, 'auc': base_auc},
    'XGBoost': {'acc': base_acc, 'auc': base_auc} 
}
baselines['Random Forest'] = {'acc': ablation_records[0]['Accuracy'], 'auc': ablation_records[0]['ROC-AUC']}
baselines['XGBoost'] = {'acc': ablation_records[1]['Accuracy'], 'auc': ablation_records[1]['ROC-AUC']}


# Ablation Testing
for suite_to_drop, features_to_drop in feature_suites.items():
    print(f'\n--- Amputating Domain: {suite_to_drop} ---')
    
    remaining_features = [f for f in all_features if f not in features_to_drop]
    print(f'Features remaining: {len(remaining_features)} (Dropped {len(features_to_drop)})')
    
    X_ablated = df[remaining_features]
    X_train, X_test, y_train, y_test = train_test_split(X_ablated, y, test_size=0.2, random_state=42)
    
    RF_model = train_tuned_RF(X_train, y_train)
    XGB_model = train_tuned_XGB(X_train, y_train)
    
    for name, model in [('Random Forest', RF_model), ('XGBoost', XGB_model)]:
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        acc_delta = acc - baselines[name]['acc']
        auc_delta = auc - baselines[name]['auc']
        
        ablation_records.append({
            'Ablated_Suite': f'Dropped {suite_to_drop}',
            'Model': name,
            'Features_Remaining': len(remaining_features),
            'Accuracy': acc,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Accuracy_Delta': round(acc_delta, 4),
            'AUC_Delta': round(auc_delta, 4)
        })

df_ablation = pd.DataFrame(ablation_records)
print('\n' + '='*70 + '\n FINAL SYSTEMATIC ABLATION STRESS-TEST MATRIX \n' + '='*70)
print(df_ablation.to_string(index=False))

df_ablation.to_csv(results_path('systematic_ablation_results.csv'), index=False)
print(f"\n[SUCCESS] Ablation testing complete. Data saved to '{results_path('systematic_ablation_results.csv')}'.")