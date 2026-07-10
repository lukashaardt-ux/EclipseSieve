'''
Extract_Features_Candidates.py — builds the UNLABELED candidate set.

Queries candidates from the NASA Exoplanet Archive and runs the shared
extraction loop. The feature-extraction engine and download/resume loop live
in features_utils.py.
'''

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

from features_utils import run_extraction
from paths import CANDIDATE_FEATURES_CSV

OUTPUT_FILE = CANDIDATE_FEATURES_CSV

# Query only CANDIDATE-disposition targets (these are the unlabeled candidates for the model to classify and provide input).
koi = NasaExoplanetArchive.query_criteria(
    table='cumulative',
    select='kepid,koi_disposition,koi_period',
    where="koi_disposition like '%CANDIDATE%'"
)
df = koi.to_pandas()
df = df.drop_duplicates(subset='kepid').reset_index(drop=True)

# run_extraction() is a synthetic function in features_utils.py that handles the extraction loop.
run_extraction(df, OUTPUT_FILE, use_labels=False)