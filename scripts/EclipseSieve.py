'''
Creates the probability distribution of the candidate exoplanets in the KOI list using the trained EclipseSieve (XGBoost) model.
The script loads the candidate features, applies the model to predict probabilities, and assigns a priority tier (all model predictions).
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_utils import BalancedXGBClassifier
from paths import CANDIDATE_FEATURES_CSV, ECLIPSE_SIEVE_PKL, figure_path, results_path
import seaborn as sns
import joblib

bundle=joblib.load(ECLIPSE_SIEVE_PKL)

EclipseSieve=bundle['XGB']
features=bundle['features']
df=pd.read_csv(CANDIDATE_FEATURES_CSV)

#Add added features
df['snr_x_depth']=df['snr']*df['depth']
df['sec_over_depth']=df['sec_eclipse_depth']/(df['depth']+1e-9)
df['duration_over_period']=df['duration']/df['period']
df['odd_even_x_sec']=df['odd_even_diff']*df['sec_eclipse_depth']
df['transit_shape_metric'] = df['depth'] / (df['duration'] + 1e-9)
df['flux_variability'] = df['std_oot'] / (df['archive_snr'] + 1e-9)
df['sec_depth_significance'] = df['sec_eclipse_depth'] / (df['out_of_transit_rms'] + 1e-9)

#Determine Candidates

X_test=df[features]

#Test Model

probs = EclipseSieve.predict_proba(X_test)[:,1]

KICs_probs=[[int(df.iloc[i]['kepid']), probs[i]] for i in range(len(probs))]
sorted_KIC_probs=sorted(KICs_probs,key=lambda x:x[1],reverse=True)

#Finding most likely candidates
for KIC in sorted_KIC_probs[:10]:
    print(f'KIC {KIC[0]} has a {KIC[1]*100:.2f}% chance of being an exoplanet')

#Assigning Priority
def assign_value(KIC_probs):
    for KIC in KIC_probs:
        prob=KIC[1]
        if prob >= 0.75:
            KIC.append('High')
        elif prob >= 0.5:
            KIC.append('Medium')
        elif prob >= 0.25:
            KIC.append('Low')
        else:
            KIC.append('Unlikely')
    return KIC_probs

KIC_prob_tier=assign_value(sorted_KIC_probs)
results_df = pd.DataFrame(KIC_prob_tier, columns=['kepid', 'probability', 'tier'])

#Spread
print('Candidate Tier Spread:')
for tier in results_df['tier'].unique():
    count = results_df[results_df['tier'] == tier].shape[0]
    print(f'  {tier}: {count}')
    print(f'  {tier}: {count/len(results_df)*100:.2f}% of candidates')

#Creation of graphs
sns.histplot(results_df['probability'],bins=25,kde=True,color='purple')
plt.title('Frequency of probabilities in the KOI candidate list')
plt.savefig(figure_path('candidate_probability_frequency.png'))
plt.show()
#Creation of CSV
results_df.to_csv(results_path('candidate_exoplanet_results.csv'),index=False)