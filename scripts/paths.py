'''
Shared path constants so scripts can be run from any working directory
while data/models/results/figures stay organized in sibling folders.
'''

import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
FIGURES_DIR = os.path.join(ROOT_DIR, 'figures', 'current')

for _dir in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_dir, exist_ok=True)

EXOPLANET_FEATURES_CSV = os.path.join(DATA_DIR, 'labeled_exoplanet_features.csv')
CANDIDATE_FEATURES_CSV = os.path.join(DATA_DIR, 'candidate_features.csv')

ECLIPSE_SIEVE_PKL = os.path.join(MODELS_DIR, 'EclipseSieve.pkl')
ECLIPSE_SIEVE_STAGE2_PKL = os.path.join(MODELS_DIR, 'EclipseSieve_Stage2_Interceptor.pkl')


def data_path(filename):
    return os.path.join(DATA_DIR, filename)


def results_path(filename):
    return os.path.join(RESULTS_DIR, filename)


def figure_path(filename):
    return os.path.join(FIGURES_DIR, filename)
