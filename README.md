# spikestats

<p align="center">
  <img src="assets/logo.png" alt="spikestats logo" width="160">
</p>

Spike-train statistics in pure Python, with zero dependencies.

`spikestats` computes the standard measures of spike-train rate and variability from a plain
list of spike times: firing rate, inter-spike intervals, coefficient of variation, CV2,
local variation (Lv and the refractory-corrected LvR), spike counts, the Fano factor, the
spike-time tiling coefficient (STTC), and the cross- and autocorrelogram (CCG and ACG) for
pairwise coincidence structure. There is nothing to configure
and nothing to install beyond the package itself: the input is a `list[float]` of spike times
and the output is a `float`.

It pairs with [spikegen](https://github.com/amaar-mc/spikegen) for generating trains and
[spikedist](https://github.com/amaar-mc/spikedist) for comparing them. The three share the
same plain-list data model, so they compose without adapters.

## Why

The established tools for these measures, Elephant and spiketools, are excellent but pull in
NumPy, SciPy, and custom data objects (`neo.SpikeTrain` and friends). When you only need a
firing rate or a CV from a list of spike times, that is a heavy dependency tree to carry, and
it is awkward in teaching material, small scripts, and lightweight pipelines. `spikestats`
keeps the math, drops the dependencies, and works on the lists you already have.

## Install

```sh
pip install spikestats
```

## Usage

```python
import spikestats as ss

spikes = [0.012, 0.031, 0.058, 0.090, 0.110, 0.155]  # seconds

ss.firing_rate(spikes, duration=0.2)   # spikes per second
ss.inter_spike_intervals(spikes)       # consecutive differences
ss.cv_isi(spikes)                      # coefficient of variation of the ISIs
ss.cv2(spikes)                         # Holt et al. 1996, robust to rate drift
ss.lv(spikes)                          # Shinomoto et al. 2003 local variation
ss.lvr(spikes, refractory=0.002)       # Shinomoto et al. 2009, refractory-corrected
ss.spike_counts(spikes, duration=0.2, bin_width=0.05)
ss.fano_factor(spikes, duration=0.2, bin_width=0.05)
```

### Pairwise correlation

```python
import spikestats as ss

a = [0.012, 0.058, 0.110, 0.155]  # seconds
b = [0.015, 0.061, 0.300]         # seconds

# Spike-time tiling coefficient (Cutts and Eglen 2014): firing-rate-robust correlation
# of two trains over a recording interval, with a synchronicity window dt.
ss.spike_time_tiling_coefficient(a, b, dt=0.005, interval=(0.0, 0.2))

# Cross-correlogram: histogram of lags (target - reference) in symmetric bins centered on
# zero. Returns 2 * ceil(max_lag / bin_width) + 1 integer counts, most-negative lag first.
ss.cross_correlogram(a, b, bin_width=0.005, max_lag=0.05)

# Autocorrelogram: the train's CCG with itself, excluding the trivial i == i self-pairs.
# Symmetric by construction; peaks appear at multiples of the train's ISI.
ss.autocorrelogram(a, bin_width=0.005, max_lag=0.05)
```

### Time-resolved metrics

```python
import spikestats as ss

spikes = [0.012, 0.031, 0.058, 0.090, 0.110, 0.155, 0.210, 0.245]  # seconds

# Non-overlapping bin counts tiling [0, duration)
counts = ss.binned_spike_counts(spikes, duration=0.3, bin_width=0.1)
# => [3, 3, 2]  (one list[int] per bin)

# Sliding-window firing rate: returns (center_times, rates_hz)
centers, rates = ss.time_resolved_rate(spikes, duration=0.3, bin_width=0.1, step=0.05)

# PSTH averaged over multiple trials: returns (bin_center_times, mean_rate_hz)
trials = [spikes, [0.020, 0.060, 0.100, 0.140, 0.200]]
bin_centers, mean_rates = ss.psth(trials, duration=0.3, bin_width=0.1)

# Per-bin Fano factor across trials: returns (bin_center_times, fano_per_bin)
bin_centers, fano = ss.time_resolved_fano(trials, duration=0.3, bin_width=0.1)
```

## API

All functions take spike times as a sequence of numbers and sort them internally.

### Scalar metrics

- `firing_rate(spikes, *, duration)`: spike count divided by `duration`.
- `inter_spike_intervals(spikes)`: list of consecutive differences (empty for fewer than two spikes).
- `cv_isi(spikes)`: population standard deviation of the ISIs over their mean. Regular train 0, Poisson 1.
- `cv2(spikes)`: mean of `2 * |I(n+1) - I(n)| / (I(n+1) + I(n))` over adjacent intervals.
- `lv(spikes)`: mean of `3 * ((I(n) - I(n+1)) / (I(n) + I(n+1)))^2`. Regular train 0, Poisson 1.
- `lvr(spikes, *, refractory)`: LvR with a refractoriness constant in the spike-time unit.
- `spike_counts(spikes, *, duration, bin_width)`: counts per equal-width bin tiling `[0, n * bin_width)`.
- `fano_factor(spikes, *, duration, bin_width)`: population variance of the bin counts over their mean.

### Pairwise metrics

- `spike_time_tiling_coefficient(spikes_a, spikes_b, *, dt, interval)`: spike-time tiling
  coefficient (Cutts and Eglen 2014). `interval` is a `(start, end)` recording window and `dt`
  is the synchronicity window. Returns a value in `[-1, 1]`; identical trains give 1.0, and the
  measure is symmetric and robust to differing firing rates. Empty trains return 0.0.
- `cross_correlogram(reference, target, *, bin_width, max_lag) -> list[int]`: binned
  cross-correlogram (CCG). Histograms the lag `target_time - reference_time` of every spike
  pair into `2 * n + 1` symmetric bins with `n = ceil(max_lag / bin_width)`. Bin `k` covers
  the half-open lag interval `[(k - n - 0.5) * bin_width, (k - n + 0.5) * bin_width)`, so the
  central bin (index `n`) is centered on lag zero. Counts are ordered most-negative lag first.
  The sign convention is `target - reference`, so `cross_correlogram(a, b)` reversed equals
  `cross_correlogram(b, a)`.
- `autocorrelogram(spikes, *, bin_width, max_lag) -> list[int]`: binned autocorrelogram (ACG),
  the CCG of a train with itself on the same bin grid. The trivial zero-lag self-pairs
  `i == i` are excluded, so the central bin counts only genuine pairs of distinct spikes. The
  ACG is symmetric by construction.

### Time-resolved metrics

All functions use the half-open boundary `[0, duration)`: a spike at exactly `duration` is excluded.

- `binned_spike_counts(spikes, *, duration, bin_width) -> list[int]`: spike counts in
  `int(duration / bin_width)` consecutive non-overlapping bins tiling `[0, duration)`.
- `time_resolved_rate(spikes, *, duration, bin_width, step) -> tuple[list[float], list[float]]`:
  sliding-window firing rate. Window of width `bin_width` steps by `step` across `[0, duration)`.
  Returns `(window_center_times, rates_in_hz)`. Window positions are computed by integer indexing
  of the step size to avoid floating-point drift.
- `psth(trials, *, duration, bin_width) -> tuple[list[float], list[float]]`: peristimulus time
  histogram. `trials` is a sequence of spike-time sequences. Returns `(bin_center_times,
  mean_rate_per_bin_hz)` averaged over trials. Raises `ValueError` on empty `trials`.
- `time_resolved_fano(trials, *, duration, bin_width) -> tuple[list[float], list[float]]`:
  per-bin Fano factor (population variance / mean of per-trial counts). Returns
  `(bin_center_times, fano_per_bin)`. Bins with zero mean across all trials are returned as
  `float('nan')`. Uses population variance (denominator N), so a single trial always gives 0.

Parameters after `*` are keyword-only and have no default values; pass them explicitly.

## Notes

- `cv_isi` uses the population standard deviation, matching the common `numpy.std` default.
- `cv2`, `lv`, and `lvr` need at least three spikes; they raise a clear `ValueError` otherwise.
- `spike_counts` uses `floor(duration / bin_width)` equal-width bins, so every bin has the same
  width; any remainder of the duration and any spikes outside the binned window are ignored.
- `binned_spike_counts` uses `int(duration / bin_width)` bins (same count). All time-resolved
  functions use the half-open interval `[0, duration)`: a spike at `t == duration` is excluded.
- `time_resolved_fano` uses population variance (denominator N). With a single trial, variance
  is 0 for every bin; use multiple trials to get meaningful Fano estimates. Bins where the
  mean count is 0 across all trials are returned as `float('nan')`.
- `spike_time_tiling_coefficient` requires every spike to lie within the closed `interval`
  `[start, end]`; a spike outside raises a `ValueError`. Following the Cutts and Eglen
  convention, when a denominator `1 - P * T` is zero (for example when `dt` is large enough that
  the tiling saturates to 1), that half of the STTC contributes 0. An empty train gives 0.0.
- `cross_correlogram` and `autocorrelogram` use symmetric bins centered on lag zero:
  `n = ceil(max_lag / bin_width)` bins on each side give an odd `2 * n + 1` bins, with the
  central bin covering `[-bin_width / 2, +bin_width / 2)`. A lag is counted when it falls in
  the grid, `[-(n + 0.5) * bin_width, +(n + 0.5) * bin_width)`, which always covers at least
  `[-max_lag, max_lag]`. `bin_width` and `max_lag` must be positive and `max_lag` must be at
  least one `bin_width`, otherwise a `ValueError` is raised. The ACG excludes the `i == i`
  self-pairs and is symmetric.

## License

MIT
