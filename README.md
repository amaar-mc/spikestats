# spikestats

Spike-train statistics in pure Python, with zero dependencies.

`spikestats` computes the standard measures of spike-train rate and variability from a plain
list of spike times: firing rate, inter-spike intervals, coefficient of variation, CV2,
local variation (Lv and the refractory-corrected LvR), spike counts, and the Fano factor.
There is nothing to configure and nothing to install beyond the package itself: the input
is a `list[float]` of spike times and the output is a `float`.

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

## API

All functions take spike times as a sequence of numbers and sort them internally.

- `firing_rate(spikes, *, duration)`: spike count divided by `duration`.
- `inter_spike_intervals(spikes)`: list of consecutive differences (empty for fewer than two spikes).
- `cv_isi(spikes)`: population standard deviation of the ISIs over their mean. Regular train 0, Poisson 1.
- `cv2(spikes)`: mean of `2 * |I(n+1) - I(n)| / (I(n+1) + I(n))` over adjacent intervals.
- `lv(spikes)`: mean of `3 * ((I(n) - I(n+1)) / (I(n) + I(n+1)))^2`. Regular train 0, Poisson 1.
- `lvr(spikes, *, refractory)`: LvR with a refractoriness constant in the spike-time unit.
- `spike_counts(spikes, *, duration, bin_width)`: counts per equal-width bin tiling `[0, n * bin_width)`.
- `fano_factor(spikes, *, duration, bin_width)`: population variance of the bin counts over their mean.

Parameters after `*` are keyword-only and have no default values; pass them explicitly.

## Notes

- `cv_isi` uses the population standard deviation, matching the common `numpy.std` default.
- `cv2`, `lv`, and `lvr` need at least three spikes; they raise a clear `ValueError` otherwise.
- `spike_counts` uses `floor(duration / bin_width)` equal-width bins, so every bin has the same
  width; any remainder of the duration and any spikes outside the binned window are ignored.

## License

MIT
