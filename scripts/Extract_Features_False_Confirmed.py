'''
Queries confirmed planets + false positives from the NASA Exoplanet Archive,
assigns label=1 (CONFIRMED) / 0 (FALSE POSITIVE), and runs the shared
extraction loop. The feature-extraction engine and download/resume loop live
in features_utils.py — this file only contains the training-set-specific setup.
'''

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

from features_utils import run_extraction
from paths import EXOPLANET_FEATURES_CSV

OUTPUT_FILE = EXOPLANET_FEATURES_CSV

# Queries only CONFIRMED and FALSE POSTIVES
koi = NasaExoplanetArchive.query_criteria(
    table='cumulative',
    select='kepid,koi_disposition,koi_period',
    where="koi_disposition like '%CONFIRMED%' or koi_disposition like '%FALSE POSITIVE%'"
)
df = koi.to_pandas()
df['label'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
df = df.drop_duplicates(subset='kepid').reset_index(drop=True)

# Cap each class so the training set stays roughly balanced
n_each = 3000
confirmed = df[df['label'] == 1].head(n_each)
false_positives = df[df['label'] == 0].head(n_each)
df = pd.concat([confirmed, false_positives]).sample(frac=1, random_state=42).reset_index(drop=True)

# use_labels=True -> each row's label (0/1) is carried into the feature dict.
# run_extraction() is a synthetic function in features_utils.py that handles the extraction loop.
run_extraction(df, OUTPUT_FILE, use_labels=True)