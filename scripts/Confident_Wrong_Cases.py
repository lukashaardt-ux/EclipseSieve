'''
This script identifies cases where the EclipseSieve model is confident in its prediction but is actually wrong.
It uses the stage 1 XGB model to predict the probability of being an exoplanet and counts the number of cases that fall into the confident range (<=0.2 or >=0.8) but are misclassified.
The output includes the total number of confident-wrong cases, the number of cases where the model is confident it's a planet but it's actually a false positive, and the number of cases where the model is confident it's a false positive but it's actually a planet.
The script also saves the confident-wrong cases for further inspection and provides a comparison of median feature values between missed false positives and caught false positives.
This analysis helps to understand the model's performance on cases where it is overly confident and incorrect.
'''

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from model_utils import BalancedXGBClassifier
from paths import EXOPLANET_FEATURES_CSV, ECLIPSE_SIEVE_PKL, results_path

df = pd.read_csv(EXOPLANET_FEATURES_CSV)
df['snr_x_depth'] = df['snr'] * df['depth']
df['sec_over_depth'] = df['sec_eclipse_depth'] / (df['depth'] + 1e-9)
df['duration_over_period'] = df['duration'] / df['period']
df['odd_even_x_sec'] = df['odd_even_diff'] * df['sec_eclipse_depth']
df['transit_shape_metric'] = df['depth'] / (df['duration'] + 1e-9)
df['flux_variability'] = df['std_oot'] / (df['archive_snr'] + 1e-9)
df['sec_depth_significance'] = df['sec_eclipse_depth'] / (df['out_of_transit_rms'] + 1e-9)

stage1_data = joblib.load(ECLIPSE_SIEVE_PKL)
stage1_xgb = stage1_data['XGB']
stage1_features = stage1_data['features']

y = df['label']

_, df_test, _, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

test_probs = stage1_xgb.predict_proba(df_test[stage1_features])[:, 1]
preds = (test_probs > 0.5).astype(int)

confident = (test_probs >= 0.8) | (test_probs <= 0.2)
wrong = (preds != y_test.values)
cw = confident & wrong
print(f'Confident AND wrong(Threshold 0.2 and 0.8): {cw.sum()}')
print(f'Loaded model uses {len(stage1_features)} features')
print('Has engineered 3:', all(f in stage1_features for f in ['transit_shape_metric','flux_variability','sec_depth_significance']))

sure_planet_actually_fp = cw & (preds == 1)   
sure_fp_actually_planet = cw & (preds == 0)   

print('Threshold for confident: 0.8 (planet) and 0.2 (FP)')
print(f'Confident-wrong total: {cw.sum()}')
print(f"  Sure it's a planet, actually FP: {sure_planet_actually_fp.sum()}")
print(f"  Sure it's FP, actually a planet: {sure_fp_actually_planet.sum()}")

# actual cases of confident-wrong predictions
cw_cases = df_test[cw].copy()
cw_cases['pred_prob'] = test_probs[cw]
cw_cases['true_label'] = y_test.values[cw]
cw_cases.to_csv(results_path('confident_wrong_cases.csv'), index=False)

true_negatives = (preds == 0) & (y_test.values == 0) 

contamination_features = ['sec_eclipse_depth', 'odd_even_diff', 'depth_ratio', 
                           'sec_over_depth', 'sec_depth_significance','snr', 
                           'archive_snr', 'depth', 'impact_parameter', 'transit_snr_bls',
                           'bls_power', 'planet_radius',
                           'stellar_teff', 'max_mes', 'max_ses', 'cdpp', 'symmetry',
                           'flux_asymmetry', 'duration', 'period']

comparison = pd.DataFrame({
    'missed_FP_median': df_test[sure_planet_actually_fp][contamination_features].median(),
    'caught_FP_median': df_test[true_negatives][contamination_features].median()
})
print(comparison)

missed_planets = df_test[sure_fp_actually_planet].copy()
missed_planets['pred_prob'] = test_probs[sure_fp_actually_planet]
print(missed_planets[['kepid', 'pred_prob', 'planet_radius'] + contamination_features].to_string())
