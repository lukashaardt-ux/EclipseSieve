'''
This script adds the catalog features to the directed csv file (labeled_exoplanet_features.csv or candidate_features.csv)
'''

import os
import datetime
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from paths import data_path
import warnings
warnings.filterwarnings('ignore')

TESTING_CSV='candidate_features.csv'  # Change to 'labeled_exoplanet_features.csv' to test on the confirmed/false positive set instead
DATA_FILE = data_path(TESTING_CSV) 

df = pd.read_csv(DATA_FILE)
kic_ids = list(df['kepid'].astype(int).unique())

# Robovetter is another exoplanet vetter created by NASA
print(f'Querying Robovetter metrics for {len(kic_ids)} unique candidate stars...')

results = []
batch_size = 500
columns_to_fetch = (
    'kepid, koi_model_snr, koi_prad, '
    'koi_duration, koi_steff, koi_slogg, '
    'koi_max_mult_ev, koi_max_sngle_ev, koi_impact'
)

for i in range(0, len(kic_ids), batch_size):
    batch = kic_ids[i:i+batch_size]
    print(f' Fetching batch {i//batch_size + 1} of {len(kic_ids)//batch_size + 1}...')
    
    koi = NasaExoplanetArchive.query_criteria(
        table='cumulative',
        select=columns_to_fetch,
        where=f'kepid in ({','.join(str(k) for k in batch)})'
    )
    results.append(koi.to_pandas())

catalog_df = pd.concat(results).drop_duplicates(subset='kepid').reset_index(drop=True)

enriched_df = df.merge(catalog_df, on='kepid', how='left')

cols_to_drop = [col for col in enriched_df.columns if col.endswith('_x') or col.endswith('.1')]
enriched_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

rename_map = {
    'koi_model_snr_y': 'archive_snr',
    'koi_prad_y': 'planet_radius',
    'koi_duration_y': 'archive_duration',
    'koi_steff_y': 'stellar_teff',
    'koi_slogg_y': 'stellar_logg',
    'koi_model_snr': 'archive_snr',
    'koi_prad': 'planet_radius',
    'koi_duration': 'archive_duration',
    'koi_steff': 'stellar_teff',
    'koi_slogg': 'stellar_logg',
    'koi_max_mult_ev': 'max_mes',
    'koi_max_sngle_ev': 'max_ses',
    'koi_impact': 'impact_parameter'
}
enriched_df.rename(columns=rename_map, inplace=True)
enriched_df = enriched_df.loc[:, ~enriched_df.columns.duplicated()]

enriched_df.fillna(enriched_df.median(numeric_only=True), inplace=True)

if os.path.exists(DATA_FILE):
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'{DATA_FILE.replace('.csv', '')}_backup_{stamp}.csv'
    df.to_csv(backup_name, index=False)
    print(f'[BACKUP] Original saved to {backup_name}')

enriched_df.to_csv(DATA_FILE, index=False) 
print('\n[SUCCESS] Candidate dataset successfully enriched and saved!')
