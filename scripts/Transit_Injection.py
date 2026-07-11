'''
This script injects a fair eclipsing binary (BEB) transit signal into the exoplanet dataset and evaluates how the stage 1 XGB model responds to these injections.
It measures the model's dependency and sensitivity to the BEB features by observing the predicted probabilities for injected cases across different tiers of BEB contamination.
'''

from model_utils import BalancedXGBClassifier
from paths import EXOPLANET_FEATURES_CSV, ECLIPSE_SIEVE_PKL, ECLIPSE_SIEVE_STAGE2_PKL
import numpy as np
import pandas as pd
import batman
import joblib
import warnings
warnings.filterwarnings('ignore')

print('Loading Models...')
try:
    stage1_data = joblib.load(ECLIPSE_SIEVE_PKL)
    stage1_xgb = stage1_data['XGB']
    stage1_features = stage1_data['features']
    print('Models loaded successfully.')
except Exception as e:
    print(f'ERROR loading models: {e}')

print('Extracting Template...')
try:
    df = pd.read_csv(EXOPLANET_FEATURES_CSV)
    planet_pool = df[df['label'] == 1]
    fp_pool = df[df['label'] == 0]
    planet_template = planet_pool.iloc[[0]].copy()
    print(f'Found {len(planet_pool)} planet examples and {len(fp_pool)} false positive examples.')
except Exception as e:
    print(f'ERROR reading dataset: {e}')

print('Generating BEB transit simulation...')
time = np.linspace(-0.05, 0.05, 1000)

p_beb = batman.TransitParams()
p_beb.t0, p_beb.per, p_beb.rp, p_beb.a, p_beb.inc, p_beb.ecc, p_beb.w = 0., 1., 0.35, 12., 84.8, 0., 90.
p_beb.u, p_beb.limb_dark = [0.1, 0.3], 'quadratic'
flux_beb = (batman.TransitModel(p_beb, time).light_curve(p_beb) * 0.12) + 0.88 + np.random.normal(0, 0.0001, len(time))

def extract_metrics_from_flux(time_array, flux_array):
    calculated_depth = 1.0 - np.min(flux_array)
    in_transit_points = np.sum(flux_array < 0.999)
    calculated_duration_hours = in_transit_points * (time_array[1] - time_array[0]) * 24.0
    calculated_std_oot = np.std(flux_array[flux_array > 0.999])
    return calculated_depth, calculated_duration_hours, calculated_std_oot

depth_b, dur_b, std_b = extract_metrics_from_flux(time, flux_beb)

tiers = [
    ('Easy BEB', 0.0050, 0.0020),
    ('Median BEB', 0.0010, 0.0005),
    ('Hard BEB', 0.0002, 0.0001),
]


def build_fair_beb_vector(planet_template, depth, duration, sec, odd):
    vec = planet_template.copy()  
    vec['depth'] = depth
    vec['duration'] = duration
    realistic_rms = 1.0
    vec['out_of_transit_rms'] = realistic_rms
    vec['std_oot'] = realistic_rms
    vec['snr'] = depth / (realistic_rms + 1e-9)
    vec['sec_eclipse_depth'] = sec
    vec['odd_even_diff'] = odd
    vec['snr_x_depth'] = vec['snr'] * vec['depth']
    vec['sec_over_depth'] = vec['sec_eclipse_depth'] / (vec['depth'] + 1e-9)
    vec['duration_over_period'] = vec['duration'] / vec['period']
    vec['odd_even_x_sec'] = vec['odd_even_diff'] * vec['sec_eclipse_depth']
    vec['transit_shape_metric'] = vec['depth'] / (vec['duration'] + 1e-9)
    vec['flux_variability'] = vec['std_oot'] / (vec['archive_snr'] + 1e-9)
    vec['sec_depth_significance'] = vec['sec_eclipse_depth'] / (vec['out_of_transit_rms'] + 1e-9)
    return vec

print('Running fair injection across EB contamination tiers...')
print(f'{'Scenario':<15} | {'Sec Depth':<10} | {'Odd-Even':<10} | {'Probability':<12} | {'Classification'}')
print('-' * 75)

probs_by_tier = []
for name, sec, odd in tiers:
    vec = build_fair_beb_vector(planet_template, depth_b, dur_b, sec, odd)
    prob = stage1_xgb.predict_proba(vec[stage1_features])[:, 1][0]
    probs_by_tier.append(prob)
    result = 'REJECTED' if prob < 0.5 else 'APPROVED (FP)'
    print(f'{name:<15} | {sec:<10.4f} | {odd:<10.4f} | {prob:<12.4f} | {result}')

spread = max(probs_by_tier) - min(probs_by_tier)
print('-' * 75)
print(f'Probability spread across tiers (Easy -> Hard): {spread:.4f}')
print('Interpretation: a large spread = EB features DO move predictions when catalog is neutral;')
print('                a near-zero spread = EB features are inert even without catalog overshadowing.')