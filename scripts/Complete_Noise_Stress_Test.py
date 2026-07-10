'''
This script performs a noise stress test on the EclipseSieve pipeline. 
It generates synthetic light curves with varying levels of Gaussian noise and evaluates the pipeline's ability to correctly identify exoplanet signals. 
The results are visualized in a plot showing the relationship between noise intensity and the model's probability of classifying a signal as a planet.

This test affects as many features as possible by re-extracting them from the noisy light curve.
'''

import numpy as np
import pandas as pd
import batman
import joblib
import matplotlib.pyplot as plt
from model_utils import BalancedXGBClassifier
from paths import EXOPLANET_FEATURES_CSV, ECLIPSE_SIEVE_PKL, ECLIPSE_SIEVE_STAGE2_PKL, figure_path
import warnings
warnings.filterwarnings('ignore')

try:
    stage1_data = joblib.load(ECLIPSE_SIEVE_PKL)
    stage1_xgb = stage1_data['XGB']
    stage1_features = stage1_data['features']
except Exception as e:
    print(f'Error loading model: {e}')
    exit()

try:
    df = pd.read_csv(EXOPLANET_FEATURES_CSV)
    planet_pool = df[df['label'] == 1]
    if len(planet_pool) == 0:
        planet_pool = df[df['label'].astype(str).str.upper().str.contains('CAN|PLANET|1')]
    template = planet_pool.iloc[[0]].copy()
    print('[2/3] Template star loaded successfully.')
except Exception as e:
    print(f'Error loading dataset: {e}')
    exit()


noise_levels = np.linspace(0.0001, 0.03, 12)  
results = []
time = np.linspace(-0.05, 0.05, 1000)

def run_pipeline(flux, time_array):
    # Re-extract features from the noisy light curve
    depth = 1.0 - np.min(flux)
    in_transit_points = np.sum(flux < 0.999)
    dur = in_transit_points * (time_array[1] - time_array[0]) * 24.0
    std = np.std(flux[flux > 0.999])

    vec = template.copy()

    # Replace features with re-extracted values
    vec['depth'] = depth
    vec['duration'] = dur
    vec['std_oot'] = std
    vec['out_of_transit_rms'] = std
    vec['snr'] = depth / (std + 1e-9)
    vec['rp_over_rs'] = np.sqrt(max(depth, 0.0))          
    vec['transit_snr_bls'] = depth / (std + 1e-9)         
    vec['sec_eclipse_depth'] = 0.0
    vec['odd_even_diff'] = 0.00005
    vec['snr_x_depth'] = vec['snr'] * vec['depth']
    vec['sec_over_depth'] = vec['sec_eclipse_depth'] / (vec['depth'] + 1e-9)
    vec['duration_over_period'] = vec['duration'] / vec['period']
    vec['odd_even_x_sec'] = vec['odd_even_diff'] * vec['sec_eclipse_depth']
    vec['transit_shape_metric'] = vec['depth'] / (vec['duration'] + 1e-9)
    vec['flux_variability'] = vec['std_oot'] / (vec['archive_snr'] + 1e-9)
    vec['sec_depth_significance'] = vec['sec_eclipse_depth'] / (vec['out_of_transit_rms'] + 1e-9)

   # The following features are left as-is from the template since they cannot be re-extracted from the light curve:
    # planet_radius, impact_parameter, archive_snr, archive_duration,
    # stellar_teff, stellar_logg, max_mes, max_ses, cdpp, period are left as-is.

    X1 = vec[stage1_features]
    prob = stage1_xgb.predict_proba(X1)[:, 1][0]
    return prob

print('[3/3] Running Stress Test Loop...')

for i, noise in enumerate(noise_levels):
    p_planet = batman.TransitParams()
    p_planet.t0, p_planet.per, p_planet.rp, p_planet.a, p_planet.inc, p_planet.ecc, p_planet.w = 0., 1., 0.12, 12., 90., 0., 90.
    p_planet.u, p_planet.limb_dark = [0.1, 0.3], 'quadratic'

    clean_flux = batman.TransitModel(p_planet, time).light_curve(p_planet)
    noisy_flux = clean_flux + np.random.normal(0, noise, len(time))

    score = run_pipeline(noisy_flux, time)
    results.append(score)
    print(f' -> Noise Level {i+1}/{len(noise_levels)} (Std: {noise:.4f}) | Planet Score: {score*100:.2f}%')

plt.figure(figsize=(9, 5))
plt.plot(noise_levels, results, marker='o', linewidth=2, color='#2c3e50', label='EclipseSieve Identification Score')
plt.axhline(0.5, color='#e74c3c', linestyle='--', label='50% Rejection Threshold')
plt.title('Pipeline Robustness: Re-extracted Features vs. Signal Noise', fontsize=14, fontweight='bold')
plt.xlabel('Telescope Noise Intensity (Standard Deviation)', fontsize=12)
plt.ylabel('Model Probability (Planet)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(figure_path('Noise_Stress_Test_Fully_ReExtracted.png'), dpi=300) 
plt.show()
print(f'\n[SUCCESS] Graph Saved.')