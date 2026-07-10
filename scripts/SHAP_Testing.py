'''
This script calculates the SHAP values for the four machine learning architectures (Random Forest, XGBoost, Logistic Regression, and SVM) used in the project.
The SHAP values are computed for the test set are visualized using summary plots (beeswarm plots) to show the impact of each feature on the model's predictions.
'''

import pandas as pd
import numpy as np
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_utils import BalancedXGBClassifier
from paths import EXOPLANET_FEATURES_CSV, ECLIPSE_SIEVE_PKL, figure_path
import matplotlib.pyplot as plt
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

# Load models
saved_data = joblib.load(ECLIPSE_SIEVE_PKL)
rf_pipeline = saved_data['RF']
xgb_pipeline = saved_data['XGB']
lr_pipeline = saved_data['LR']
svm_pipeline = saved_data['SVM']

scaler = lr_pipeline.named_steps['scaler']
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=features)


# RF)
print('Calculating SHAP for Random Forest (TreeExplainer)...')
rf_explainer = shap.TreeExplainer(rf_pipeline.named_steps['model'])
rf_shap = rf_explainer.shap_values(X_test)[..., 1] # Grab probabilities for the positive class

# XGB)
print('Calculating SHAP for XGBoost (TreeExplainer)...')
xgb_explainer = shap.TreeExplainer(xgb_pipeline.named_steps['model'])
xgb_shap = xgb_explainer.shap_values(X_test)

# LR)
print('Calculating SHAP for Logistic Regression (LinearExplainer)...')
lr_explainer = shap.LinearExplainer(lr_pipeline.named_steps['model'], X_train_scaled)
lr_shap = lr_explainer.shap_values(X_test_scaled)

# SVM)
print('Calculating SHAP for SVM (Optimized KernelExplainer)...')
svm_background = shap.kmeans(X_train_scaled, 10)
svm_explainer = shap.KernelExplainer(svm_pipeline.named_steps['model'].predict, svm_background)
X_test_svm_sample = X_test_scaled.head(50)
svm_shap = svm_explainer.shap_values(X_test_svm_sample)

# Beeswarms
plt.figure(figsize=(10, 6))
shap.summary_plot(xgb_shap, X_test, show=False)
plt.title('XGBoost SHAP Summary (Champion Architecture)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(figure_path('XGB_SHAP_Beeswarm.png'), dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(rf_shap, X_test, show=False)
plt.title('Random Forest SHAP Summary', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(figure_path('RF_SHAP_Beeswarm.png'), dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(lr_shap, X_test_scaled, show=False)
plt.title('Logistic Regression SHAP Summary', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(figure_path('LR_SHAP_Beeswarm.png'), dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(svm_shap, X_test_svm_sample, show=False)
plt.title('SVM SHAP Summary (Sampled Core Dynamics)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(figure_path('SVM_SHAP_Beeswarm.png'), dpi=300)
plt.show()
