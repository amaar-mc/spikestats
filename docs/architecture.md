# Architecture

`spikestats` is a small set of pure functions in `src/spikestats/metrics.py` over the
standard library `statistics` and `math`. A spike train is a sequence of spike times; every
function sorts its input internally and returns a plain `float` (or a list, for intervals
and counts). `_validate.py` holds the shared input checks.

Let `t_1 <= t_2 <= ... <= t_M` be the sorted spike times and let
`I_1, ..., I_N` (with `N = M - 1`) be the inter-spike intervals `I_k = t_{k+1} - t_k`.

## Firing rate

`firing_rate` returns `M / duration`. The duration must be positive.

## Inter-spike intervals

`inter_spike_intervals` returns the consecutive differences `I_k`. With fewer than two
spikes the list is empty.

## Coefficient of variation

`cv_isi` returns `pstdev(I) / mean(I)`, the population standard deviation of the intervals
over their mean. A perfectly regular train gives 0; a homogeneous Poisson process gives 1
in expectation, since exponential intervals have equal mean and standard deviation. The
population standard deviation matches the default of `numpy.std`.

## CV2

`cv2` returns the mean over adjacent interval pairs of
`2 * |I_{k+1} - I_k| / (I_{k+1} + I_k)` (Holt, Softky, Koch, Douglas 1996). Each term
depends only on two neighboring intervals, so a slow drift in rate cancels and CV2 measures
local variability. Each term lies in `[0, 2)`.

## Local variation Lv

`lv` returns `(1 / (N - 1)) * sum over adjacent pairs of 3 * ((I_k - I_{k+1}) / (I_k + I_{k+1}))^2`
(Shinomoto, Shima, Tanji 2003). Like CV2 it is a local measure: a regular train gives 0 and
a Poisson process gives 1 in expectation. Each term lies in `[0, 3)`.

## Revised local variation LvR

`lvr` returns
`(3 / (N - 1)) * sum of (1 - 4 I_k I_{k+1} / (I_k + I_{k+1})^2) * (1 + 4 R / (I_k + I_{k+1}))`
(Shinomoto et al. 2009), where `R` is a refractoriness constant in the same time unit as the
spikes. LvR corrects Lv for the shortening of variability caused by a refractory period.
Setting `R = 0` recovers Lv exactly, since `(I_k - I_{k+1})^2 / (I_k + I_{k+1})^2` equals
`1 - 4 I_k I_{k+1} / (I_k + I_{k+1})^2`; the tests assert this identity.

## Spike counts and Fano factor

`spike_counts` divides `[0, n * bin_width)` into `n = floor(duration / bin_width)` equal-width
bins and counts the spikes in each. Any remainder of the duration and any spikes outside the
binned window are ignored, so every bin has the same width. `fano_factor` returns
`pvariance(counts) / mean(counts)`. For a Poisson process the count variance equals the count
mean, so the Fano factor is 1 in expectation; a regular train binned at an integer number of
spikes per bin gives 0.

## Why pure Python

The statistics are short and standard, and the standard library `statistics` module provides
mean, variance, and standard deviation. Implementing them with no dependencies keeps
installation trivial and lets the functions operate directly on the plain lists produced by
spikegen and consumed by spikedist.

## References

- D. Holt, W. Softky, C. Koch, R. Douglas (1996). Comparison of discharge variability in
  vitro and in vivo in cat visual cortex neurons. Journal of Neurophysiology.
- S. Shinomoto, K. Shima, J. Tanji (2003). Differences in spiking patterns among cortical
  neurons. Neural Computation.
- S. Shinomoto et al. (2009). Relating neuronal firing patterns to functional differentiation
  of cerebral cortex. PLoS Computational Biology.
