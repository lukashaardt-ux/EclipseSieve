# Data Documentation and Limitations

> [!IMPORTANT]
> This file documents how the EclipseSieve datasets were produced, what the feature columns mean, and any limitations that should be taken into consideration when relying on them.

## Sources

Both datasets, `labeled_exoplanet_features.csv` and `candidate_features.csv`, were retrieved from the **NASA Exoplanet Archive** `cumulative` table via `astroquery`, then had photometric features extracted from the associated Kepler light curves using the Box Least Squares (BLS) algorithm.

- **Labeled** - contains `CONFIRMED` and `FALSE POSITIVE` data points
- **Candidate** - contains `CANDIDATE` (unlabeled) data points

The NASA Exoplanet Archive changes over time, so many of these pulled results may differ in the future.

## Files

- `labeled_exoplanet_features.csv` - 4,927 rows (confirmed exoplanets + false positives)
- `candidate_features.csv` - 1,845 rows (unconfirmed exoplanet candidates)

## Label Definitions

- `label = 1` - confirmed exoplanet
- `label = 0` - false positive

`candidate_features.csv` is unlabeled, so it does not have the label column.
`labeled_exoplanet_features.csv` has a class imbalance of roughly 60% false positives and 40% confirmed.

## Features

The pool of 32 features is divided into three main groups based on extraction. Each of these groups serves a fundamental purpose in analysis, signal extraction, and experimentation, which is further addressed in the README.

- **17 photometric features** - extracted from a light curve via BLS algorithm:
  - `period` - The orbital period (in days) of the exoplanet candidate.
  - `bls_power` - The maximum power value obtained from the BLS periodogram.
  - `depth` - The fractional drop in flux during the strongest period transit signal.
  - `depth_ratio` - The ratio of the primary transit depth to the secondary eclipse depth.
  - `duration` - The optimal transit duration (in days) derived from the BLS periodogram.
  - `rp_over_rs` - The ratio of a planet's radius to the host star's radius.
  - `symmetry` - A metric measuring how symmetrical the folded transit curve is around its center point.
  - `snr` - A signal-to-noise ratio that represents how well the signal gets through.
  - `cdpp` - Combined differential photometric precision, or the average amount of noise in a star's brightness over a specific time (typically 6 hours)
  - `std_oot` - The standard deviation of the out-of-transit flux.
  - `flux_asymmetry` - The absolute difference between the mean flux of the transit's ingress and egress.
  - `skewness` - The skewness of the out-of-transit flux.
  - `kurtosis` - The statistical kurtosis of the out-of-transit flux.
  - `sec_eclipse_depth` - The fractional flux drop measured exactly half an orbital period away from the first recorded transit.
  - `transit_snr_bls` - The SNR of the transit signal.
  - `odd_even_diff` - The absolute difference between the median depths of odd-numbered transits and even-numbered transits.
  - `out_of_transit_rms` - The root mean square value of out-of-transit flux.
- **8 NASA archive features** - pulled directly from NASA's catalog:
  - `planet_radius` - The radius of the transiting body.
  - `impact_parameter` - The minimum sky-projected distance between the center of the planet’s path and the center of the stellar disk during a transit. 
  - `archive_snr` - The SNR recorded from the NASA archive.
  - `archive_duration` - The duration recorded from the NASA archive.
  - `stellar_teff` - The temperature of the host star's surface.
  - `stellar_logg` - The base-10 logarithm of the surface gravity acceleration of the host star.
  - `max_mes` - The maximum multiple event statistic, representing the total accumulated SNR of all matching transits across observation.
  - `max_ses` - The maximum single event statistic, representing the highest SNR recorded during any transit window.
- **7 engineered features** - synthetic features derived from the above:
  - `snr_x_depth`- The SNR multiplied by the depth. This combination is meant to amplify deep transits.
  - `sec_over_depth` - The ratio of the secondary eclipse depth to the primary depth. This standardizes the strength of the secondary event relative to the main transit signal.
  - `duration_over_period` - The ratio of duration to period. This measures the percent of the period spent in transit.
  - `odd_even_x_sec` - The odd-even-difference multiplied by the secondary eclipse depth. This attempts to alert the model of false positives.
  - `transit_shape_metric` - The depth divided by the total duration of the transit. This hopes to further define the geometry of the transit.
  - `flux_variability` - The ratio of the standard deviation of out-of-transit flux to the archive SNR. This is used to define the reliability of the detection.
  - `sec_depth_significance` - The secondary eclipse depth divided by the out-of-transit RMS. This measures the significance of a secondary drop in brightness.

## Limitations

- **Duration quantization** - `duration` is drawn from `lightkurve`'s default BLS duration grid, which is not freely fit, thus a few certain values are assigned based on what fits best. Since the engineered features `duration_over_period` and `transit_shape_metric` use `duration` they also have this limitation. If you want to make these engineered features more exact without using local light curve derived features you can use `archive_duration` in place of `duration`.

- **Median imputation** - missing catalog values are filled with column medians (see `Improved_DF.py`), so any empty catalog entries are assigned the median value rather than calculated.

- **Planet-radius leakage** - The planet radius from confirmed exoplanets is more refined than that from false positives because further testing and observation is likely performed on a confirmed candidate. This means that planet radius partially encodes the catalog status rather than acting as a strictly physical quantity. This is further elaborated in the README's Limitations section.

