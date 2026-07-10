'''
This script provides a comprehensive analysis of the Permutation Feature Importance (PFI) across multiple machine learning architectures. 
It evaluates the impact of shuffling each feature on the F1-score of the models, allowing for a comparison of feature importance across different classifiers. 
The results are visualized in a bar chart.
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from model_utils import BalancedXGBClassifier
from paths import EXOPLANET_FEATURES_CSV, ECLIPSE_SIEVE_PKL, figure_path, results_path
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(EXOPLANET_FEATURES_CSV)

df['snr_x_depth'] = df['snr'] * df['depth']
df['sec_over_depth'] = df['sec_eclipse_depth'] / (df['depth'] + 1e-9)
df['duration_over_period'] = df['duration'] / df['period']
df['odd_even_x_sec'] = df['odd_even_diff'] * df['sec_eclipse_depth']
df['transit_shape_metric'] = df['depth'] / (df['duration'] + 1e-9)
df['flux_variability'] = df['std_oot'] / (df['archive_snr'] + 1e-9)
df['sec_depth_significance'] = df['sec_eclipse_depth'] / (df['out_of_transit_rms'] + 1e-9)

features = [f for f in df.columns if f not in ['label', 'kepid', 'disposition']]
X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Loading saved models...')
saved_data = joblib.load(ECLIPSE_SIEVE_PKL)
models = {
    'Logistic Regression': saved_data['LR'],
    'SVM': saved_data['SVM'],
    'Random Forest': saved_data['RF'],
    'XGBoost': saved_data['XGB']
}

pfi_results = {}
fig, axes = plt.subplots(2, 2, figsize=(20, 16)) 
axes = axes.flatten()

colors = ['green', 'orange', 'purple', 'royalblue']

for idx, (name, model) in enumerate(models.items()):
    print(f'Running Permutation Feature Importance (PFI) on {name}...')
    
    result = permutation_importance(model, X_test, y_test, scoring='f1', n_repeats=5, random_state=42, n_jobs=-1)
    
    pfi_df = pd.DataFrame({
        'feature': features,
        'importance': result.importances_mean
    }).sort_values(by='importance', ascending=False)
    
    pfi_results[name] = pfi_df
    
    print(f'Top 10 Features for {name}:\n', pfi_df.head(10), '\n')

    sns.barplot(x='importance', y='feature', data=pfi_df.head(10), ax=axes[idx], color=colors[idx], edgecolor='black')
    
    axes[idx].set_title(f'{name} F1-Score Vulnerability', fontsize=14, pad=10)
    axes[idx].set_xlabel('Drop in F1-Score when Feature is Shuffled', fontsize=12)
    axes[idx].set_ylabel('')
    axes[idx].tick_params(axis='both', labelsize=11)

plt.suptitle('Universal Permutation Feature Importance Across Architectures', fontsize=20, fontweight='bold')

plt.tight_layout(pad=4.0, w_pad=5.0, h_pad=4.0, rect=[0, 0, 1, 0.96]) 
plt.savefig(figure_path('PFI_Comparison.png'), dpi=300, bbox_inches='tight')
plt.show()