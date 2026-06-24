# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-06-23

### Added

- `spike_time_tiling_coefficient(spikes_a, spikes_b, *, dt, interval) -> float`: spike-time
  tiling coefficient (Cutts and Eglen 2014, J Neurosci), a firing-rate-robust pairwise
  correlation of two spike trains. `dt` is the synchronicity window and `interval` is the
  recording window as an explicit `(start, end)` pair. Returns a value in `[-1, 1]`; identical
  trains give 1.0 and the measure is symmetric.

### Design notes

- Recording interval: passed as an explicit `(start, end)` keyword argument (no default),
  validated by a new `check_interval` helper requiring finite bounds and `start < end`. Every
  spike must lie within the closed interval `[start, end]` or a `ValueError` is raised. The
  tiling fractions TA and TB clip each `[t - dt, t + dt]` window to the interval before taking
  the union, so they measure covered time over total recording time.
- Zero-denominator convention: when `1 - P * T` is zero (for example a saturating `dt` where
  TA or TB reaches 1), that half-term contributes 0, following Cutts and Eglen.
- Empty trains: an STTC involving an empty train is defined as 0.0.

## [0.2.0]

### Added

- `binned_spike_counts(spikes, *, duration, bin_width) -> list[int]`: spike counts in
  `int(duration / bin_width)` consecutive non-overlapping bins tiling `[0, duration)`.
- `time_resolved_rate(spikes, *, duration, bin_width, step) -> tuple[list[float], list[float]]`:
  sliding-window firing rate. Returns `(window_center_times, rates_in_hz)`. Window positions use
  integer indexing of `step` to avoid floating-point drift. The last window included is the
  largest k where `k * step + bin_width <= duration`.
- `psth(trials, *, duration, bin_width) -> tuple[list[float], list[float]]`: peristimulus time
  histogram over multiple trials. Returns `(bin_center_times, mean_rate_per_bin_hz)` averaged
  over all trials. Raises `ValueError` on empty `trials`.
- `time_resolved_fano(trials, *, duration, bin_width) -> tuple[list[float], list[float]]`:
  per-bin Fano factor (population variance / mean of per-trial counts) across trials. Returns
  `(bin_center_times, fano_per_bin)`. Bins where the mean count across all trials is 0 are
  returned as `float('nan')`.

### Design notes

- Boundary policy: all time-resolved functions use `[0, duration)` -- a spike at exactly
  `duration` is excluded, matching the existing `spike_counts` convention.
- Variance choice: `time_resolved_fano` uses population variance (denominator N), matching
  `fano_factor` in metrics.py. With one trial, pvariance of a single sample is 0, so all
  non-empty bins return 0. Callers who need sample variance should use multiple trials.
- Zero-mean bins in `time_resolved_fano`: returned as `float('nan')` so callers can identify
  and exclude them.

## [0.1.0]

### Added
- `firing_rate(spikes, *, duration)`: mean firing rate.
- `inter_spike_intervals(spikes)`: consecutive interval list.
- `cv_isi(spikes)`: coefficient of variation of the inter-spike intervals.
- `cv2(spikes)`: local CV2 averaged over adjacent interval pairs (Holt et al. 1996).
- `lv(spikes)`: local variation (Shinomoto et al. 2003).
- `lvr(spikes, *, refractory)`: revised local variation LvR (Shinomoto et al. 2009).
- `spike_counts(spikes, *, duration, bin_width)`: counts per equal-width bin.
- `fano_factor(spikes, *, duration, bin_width)`: Fano factor of binned counts.
- Pure Python, zero runtime dependencies. Inputs are plain lists of spike times, matching
  spikegen and spikedist.
- Test suite: exact-value tests, known-limit tests (regular train 0, Poisson 1), and
  property-based invariants.
