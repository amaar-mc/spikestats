# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-06-24

### Added

- `cross_correlogram(reference, target, *, bin_width, max_lag) -> list[int]`: binned
  cross-correlogram (CCG). For every spike in `reference` and every spike in `target`, the
  lag `target_time - reference_time` is histogrammed into symmetric bins. Returns
  `2 * n + 1` integer counts with `n = ceil(max_lag / bin_width)`, ordered from the most
  negative lag bin to the most positive. The sign convention is `target - reference`, so
  `cross_correlogram(a, b)` reversed equals `cross_correlogram(b, a)`.
- `autocorrelogram(spikes, *, bin_width, max_lag) -> list[int]`: binned autocorrelogram
  (ACG), the CCG of a train with itself over the same symmetric bin grid. The trivial
  zero-lag self-pairs `i == i` are excluded, so the central bin counts only genuine pairs
  of distinct spikes. The ACG is symmetric.

### Design notes

- Bin convention: `n = ceil(max_lag / bin_width)` bins on each side of zero, for `2 * n + 1`
  bins total. Bin `k` covers the half-open lag interval
  `[(k - n - 0.5) * bin_width, (k - n + 0.5) * bin_width)`, so the central bin (index `n`)
  covers `[-bin_width / 2, +bin_width / 2)` and is centered exactly on lag zero. A lag is
  counted when it falls inside the grid, i.e. `[-(n + 0.5) * bin_width, +(n + 0.5) * bin_width)`,
  which always covers at least `[-max_lag, max_lag]`.
- Validation: `bin_width` and `max_lag` must both be positive and `max_lag` must be at least
  one `bin_width`, otherwise a `ValueError` is raised.
- Empty or single-spike trains: the CCG of any train with an empty train, and the ACG of a
  train with fewer than two spikes, have no pairs, so every bin is zero.

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
