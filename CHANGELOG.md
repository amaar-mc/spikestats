# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Not included in this release

- PyPI publish is queued behind the new-project-creation quota (currently returning 429 for new
  packages). The sdist and wheel are twine-verified in `dist/`; upload will proceed once the
  quota resets.

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
