# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
