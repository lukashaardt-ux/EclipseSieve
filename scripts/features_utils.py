'''
features_utils.py — shared feature-extraction engine for EclipseSieve.

Both Extract_Features_False_Confirmed.py (the confirmed/false-positive training
set) and Extract_Features_Candidates.py (the unlabeled candidate set) use the
*identical* photometric feature extraction and the *identical* download +
resume/checkpoint loop. That shared code lives here so there is exactly one
copy to maintain. Any change that occurs here will be reflected in both scripts.

The scripts themselves keep only their own driver logic (which dispositions to
query, whether/how to assign labels, and the output filename).
'''

import warnings
warnings.filterwarnings('ignore')

import lightkurve as lk
import numpy as np
import pandas as pd
from scipy import stats
import os
import concurrent.futures


def extract_features(lc, label=None, known_period=None):
    '''Extract 17 photometric features from a single stitched light curve.

    label is passed straight through into the returned dict so the caller
    controls labelling: the training script passes 0/1, the candidate script
    passes None (unlabeled).
    '''
    lc = lc.remove_nans().normalize()
    flat_lc = lc.flatten(window_length=401)

    period_grid_coarse = np.arange(0.5, 30, 0.5)
    pg_coarse = flat_lc.to_periodogram(method='bls', period=period_grid_coarse, frequency_factor=50)
    rough_period = float(pg_coarse.period_at_max_power.value)

    period_grid_fine = np.linspace(rough_period * 0.8, rough_period * 1.2, 200)
    pg = flat_lc.to_periodogram(method='bls', period=period_grid_fine, frequency_factor=25)

    bls_period = float(pg.period_at_max_power.value)

    if known_period is not None:
        for harmonic in [1, 2, 0.5, 3, 0.33]:
            if abs(bls_period - known_period * harmonic) / known_period < 0.05:
                bls_period = known_period
                break

    period = bls_period
    bls_power = float(pg.max_power.value)
    depth = float(pg.depth_at_max_power.value)
    duration = float(pg.duration_at_max_power.value)
    t0 = float(pg.transit_time_at_max_power.value)

    folded = flat_lc.fold(period=period, epoch_time=t0)
    binned_lc = folded.bin(time_bin_size=0.01)

    flux = binned_lc.flux.value
    time = binned_lc.time.value
    transit_depth = float(1.0 - np.min(flux))
    half_depth = 1.0 - (transit_depth / 2.0)
    in_transit = time[flux < half_depth]
    transit_dur = float(in_transit[-1] - in_transit[0]) if len(in_transit) > 1 else duration
    mid = len(flux) // 2
    symmetry = float(1 - np.mean(np.abs(flux[:mid] - flux[mid:][::-1][:mid])))

    try:
        in_tr_mask = flat_lc.create_transit_mask(period=period, transit_time=t0, duration=duration)
        oot_flux = flat_lc.flux.value[~in_tr_mask]
    except Exception:
        oot_flux = flat_lc.flux.value
    if len(oot_flux) < 10:
        oot_flux = flat_lc.flux.value

    cdpp = float(flat_lc.estimate_cdpp().value)
    std_flux = float(np.std(oot_flux))
    skewness = float(stats.skew(oot_flux))
    kurtosis = float(stats.kurtosis(oot_flux))
    snr = float(transit_depth / std_flux)
    transit_snr_bls = float(bls_power / np.median(pg.power.value))

    folded_section = flat_lc.fold(period=period, epoch_time=t0 + period / 2)
    section_flux = folded_section.bin(time_bin_size=0.01).flux.value
    section_eclipse_depth = float(1.0 - np.min(section_flux))
    depth_ratio = float(transit_depth / (section_eclipse_depth + 1e-9))

    quarter = len(flux) // 4
    ingress = float(np.mean(flux[mid - quarter:mid]))
    egress = float(np.mean(flux[mid:mid + quarter]))
    flux_asymmetry = float(abs(ingress - egress))

    rp_over_rs = np.sqrt(transit_depth)
    out_of_transit_rms = float(np.sqrt(np.mean(oot_flux ** 2)))

    timespan = float(flat_lc.time.value[-1] - flat_lc.time.value[0])
    n_transits = int(timespan / period)
    transit_times = t0 + np.arange(n_transits) * period

    odd_depths = []
    even_depths = []

    for idx, tc in enumerate(transit_times[:30]):
        try:
            seg = flat_lc.truncate(tc - duration, tc + duration)
            if len(seg.flux) < 3:
                continue
            d = float(1.0 - np.min(seg.flux.value))
            if idx % 2 == 0:
                odd_depths.append(d)
            else:
                even_depths.append(d)
        except Exception:
            continue

    if odd_depths and even_depths:
        odd_even_diff = float(abs(np.median(odd_depths) - np.median(even_depths)))
    else:
        odd_even_diff = 0.0

    return {'period': period,
            'bls_power': bls_power,
            'depth': depth,
            'depth_ratio': depth_ratio,
            'duration': duration,
            'rp_over_rs': rp_over_rs,
            'symmetry': symmetry,
            'snr': snr,
            'cdpp': cdpp,
            'std_oot': std_flux,
            'flux_asymmetry': flux_asymmetry,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'sec_eclipse_depth': section_eclipse_depth,
            'transit_snr_bls': transit_snr_bls,
            'odd_even_diff': odd_even_diff,
            'out_of_transit_rms': out_of_transit_rms,
            'label': label}


def download_lc(kepid):
    '''Download and stitch all long-cadence Kepler quarters for one KIC ID.'''
    search = lk.search_lightcurve(f'KIC {kepid}', mission='Kepler', cadence='long')
    if len(search) == 0:
        raise ValueError('No lightcurve found')
    return search.download_all().stitch()


def run_extraction(df, output_file, use_labels):
    '''Shared download + extract + resume/checkpoint loop.

    df          : DataFrame with at least 'kepid', 'koi_period', 'koi_disposition'
                  (and 'label' if use_labels is True).
    output_file : CSV path to write/append results to. Resumes from it if present.
    use_labels  : True  -> read row['label'] and pass it into extract_features
                  False -> pass label=None (unlabeled candidate set)
    '''
    records = []
    done_kepids = set()

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        records = existing_df.to_dict('records')
        done_kepids = set(existing_df['kepid'].astype(int).to_list())
        print(f'Resuming from {len(done_kepids)} already processed entries.')

    for i, row in df.iterrows():
        kepid = row['kepid']
        label = row['label'] if use_labels else None
        known = row['koi_period'] if pd.notna(row.get('koi_period')) else None

        if kepid in done_kepids:
            print(f'Skipping already processed kepid {kepid}')
            continue

        print(f'Processing KIC {kepid} ({i + 1}/{len(df)})')

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(download_lc, kepid)
                lc = future.result(timeout=120)

            features = extract_features(lc, label=label, known_period=known)
            features['kepid'] = kepid
            records.append(features)
            done_kepids.add(int(kepid))
            print(f'Extracted features for kepid {kepid}')

            if len(records) % 10 == 0:
                pd.DataFrame(records).to_csv(output_file, index=False)
                print(f'  Checkpoint: {len(records)} saved')

        except concurrent.futures.TimeoutError:
            print(f'  TIMEOUT — skipping KIC {kepid} after 120s')
            continue
        except Exception as e:
            print(f'Error processing kepid {kepid}: {e}')
            continue

    pd.DataFrame(records).to_csv(output_file, index=False)
    print(f'Saved features for {len(records)} stars to {output_file}')