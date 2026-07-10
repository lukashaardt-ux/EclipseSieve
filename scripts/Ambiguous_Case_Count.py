'''
This is the Ambigous Case Count script for the labled + nonlabeled data set. 
It uses the stage 1 XGB model to predict the probability of being an exoplanet and counts the number of cases that fall into the ambiguous range (0.3-0.7). 
The output includes the total number of cases in the held-out test set, the number of ambiguous cases, and the percentage of ambiguous cases.
'''
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from model_utils import BalancedXGBClassifier
from scipy.stats  import mannwhitneyu
from paths import EXOPLANET_FEATURES_CSV, ECLIPSE_SIEVE_PKL, CANDIDATE_FEATURES_CSV
from statsmodels.stats.multitest import multipletests



def add_engineered(df):
    df['snr_x_depth'] = df['snr'] * df['depth']
    df['sec_over_depth'] = df['sec_eclipse_depth'] / (df['depth'] + 1e-9)
    df['duration_over_period'] = df['duration'] / df['period']
    df['odd_even_x_sec'] = df['odd_even_diff'] * df['sec_eclipse_depth']
    df['transit_shape_metric'] = df['depth'] / (df['duration'] + 1e-9)
    df['flux_variability'] = df['std_oot'] / (df['archive_snr'] + 1e-9)
    df['sec_depth_significance'] = df['sec_eclipse_depth'] / (df['out_of_transit_rms'] + 1e-9)
    return df

stage1_data = joblib.load(ECLIPSE_SIEVE_PKL)
stage1_xgb = stage1_data['XGB']
stage1_features = stage1_data['features']

ex_df = add_engineered(pd.read_csv(EXOPLANET_FEATURES_CSV))
y = ex_df['label']
_, ex_df_test, _, _ = train_test_split(ex_df, y, test_size=0.2, random_state=42)

ex_probs = stage1_xgb.predict_proba(ex_df_test[stage1_features])[:, 1]
ex_ambig = ((ex_probs >= 0.3) & (ex_probs <= 0.7)).sum()
print(f'[Labeled held-out] N={len(ex_df_test)}  ambiguous={ex_ambig}  ({ex_ambig/len(ex_df_test)*100:.1f}%)')

can_df = add_engineered(pd.read_csv(CANDIDATE_FEATURES_CSV))
can_probs = stage1_xgb.predict_proba(can_df[stage1_features])[:, 1]
can_ambig = ((can_probs >= 0.3) & (can_probs <= 0.7)).sum()
print(f'[Candidates] N={len(can_df)}  ambiguous={can_ambig}  ({can_ambig/len(can_df)*100:.1f}%)')

print('Total ambiguous cases (labeled + candidates):', ex_ambig + can_ambig)

# Ambiguous data result
# ambiguous: 0.3-0.7   |   confident: <=0.2 or >=0.8
feature_cols = stage1_features 

def tag_regime(probs):
    regime = np.full(len(probs), 'mid_gap', dtype=object)  # the 0.2-0.3 / 0.7-0.8 dead zones
    regime[(probs >= 0.3) & (probs <= 0.7)] = 'ambiguous'
    regime[(probs <= 0.2) | (probs >= 0.8)] = 'confident'
    return regime

ex_df_test = ex_df_test.copy()
ex_df_test['regime'] = tag_regime(ex_probs)
can_df = can_df.copy()
can_df['regime'] = tag_regime(can_probs)

pooled = pd.concat([
    ex_df_test[feature_cols + ['regime']],
    can_df[feature_cols + ['regime']]
], ignore_index=True)

medians = pooled.groupby('regime')[feature_cols].median()
means   = pooled.groupby('regime')[feature_cols].mean()

print('MEDIANS by regime:\n', medians.T)   
print('\nMEANS by regime:\n', means.T)

# Evidence
results=[]

amb = pooled[pooled['regime']=='ambiguous']
conf = pooled[pooled['regime']=='confident']

for feature in feature_cols:
    a =amb[feature].dropna()
    c =conf[feature].dropna()

    u_stat, p_value = mannwhitneyu(a, c, alternative='two-sided')

    n1, n2 = len(a), len(c)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    results.append({
        'feature': feature,
        'amb_median': a.median(),
        'conf_median': c.median(),
        'p_value': p_value,
        'effect_size': rank_biserial
    })

res_df = pd.DataFrame(results)

res_df['p_adj'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
res_df['sig'] = res_df['p_adj'] < 0.05

res_df = res_df.reindex(res_df['effect_size'].abs().sort_values(ascending=False).index)
print(res_df.to_string(index=False))
