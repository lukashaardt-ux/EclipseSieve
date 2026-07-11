'''
This script performs an analysis to identify ambiguous cases in the exoplanet classification model.
It uses the stage 1 XGB model to predict the probability of being an exoplanet and counts the number of cases that fall into the ambiguous range (0.3-0.7).
The output includes the total number of cases in the held-out test set, the number of ambiguous cases, and the percentage of ambiguous cases.
The script also tags each case as 'ambiguous', 'confident', or 'mid_gap' based on the predicted probabilities, and saves the ambiguous cases for further inspection.
This analysis helps to understand the model's performance on hard-to-classify cases and provides insights into which features contribute to the ambiguity.
This script is highly exploratory.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model_utils import BalancedXGBClassifier
from paths import EXOPLANET_FEATURES_CSV, results_path
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)

print('Loading dataset and building engineered features...')
df = pd.read_csv(EXOPLANET_FEATURES_CSV)
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

df_train, df_test, y_train, y_test = train_test_split(
    df, y, test_size=0.20, random_state=42, stratify=y)

print('Training single model on the 80% train split...')
model = BalancedXGBClassifier(random_state=42, eval_metric='logloss',
                              n_estimators=300, max_depth=5, learning_rate=0.05)
model.fit(df_train[features], y_train)

# Identify ambiguous vs confident on the held-out set.
test_probs = model.predict_proba(df_test[features])[:, 1]
df_test = df_test.copy()
df_test['stage1_prob'] = test_probs
df_test['is_ambiguous'] = (test_probs >= 0.3) & (test_probs <= 0.7)

amb = df_test[df_test['is_ambiguous']]
conf = df_test[~df_test['is_ambiguous']]

print('\n' + '=' * 60)
print(f' AMBIGUOUS: {len(amb)}  |  CONFIDENT: {len(conf)}  (held-out 20%) ')
print('=' * 60)

print(f'  Ambiguous -> planets (label=1): {(amb['label']==1).sum()}, '
      f'false positives (label=0): {(amb['label']==0).sum()}')
print(f'  Confident -> planets (label=1): {(conf['label']==1).sum()}, '
      f'false positives (label=0): {(conf['label']==0).sum()}')

amb_pred = (amb['stage1_prob'] > 0.5).astype(int)
conf_pred = (conf['stage1_prob'] > 0.5).astype(int)
print(f'  On ambiguous cases : {(amb_pred==amb['label']).mean()*100:.1f}%')
print(f'  On confident cases : {(conf_pred==conf['label']).mean()*100:.1f}%')

key_features = ['snr', 'archive_snr', 'depth', 'impact_parameter', 'transit_snr_bls',
                'bls_power', 'sec_eclipse_depth', 'odd_even_diff', 'planet_radius',
                'stellar_teff', 'max_mes', 'max_ses', 'cdpp', 'symmetry',
                'flux_asymmetry', 'duration', 'period']
rows = []
for f in key_features:
    am, cm = amb[f].median(), conf[f].median()
    rel = (am - cm) / (abs(cm) + 1e-9) * 100
    rows.append({'feature': f, 'ambiguous_median': am, 'confident_median': cm,
                 'pct_diff': rel})
gap_df = pd.DataFrame(rows).reindex(
    pd.DataFrame(rows)['pct_diff'].abs().sort_values(ascending=False).index)
print(gap_df.to_string(index=False))

out = amb[['kepid', 'label', 'stage1_prob'] + key_features].copy()
out.to_csv(results_path('ambiguous_cases_detail.csv'), index=False)
