"""Spike-train statistics over plain lists of spike times.

Every function takes a sequence of spike times (in seconds, any unit is fine as long as
it is consistent) and returns a plain float. Inputs are sorted internally, so the caller
does not need to pre-sort. The formulas follow the standard definitions cited in the
docstrings; see docs/architecture.md for the references.
"""

from collections.abc import Sequence
from itertools import pairwise
from math import ceil, floor
from statistics import mean, pstdev, pvariance

from ._validate import (
    check_interval,
    check_non_negative,
    check_positive,
    require_spikes,
    sorted_finite,
)


def inter_spike_intervals(spikes: Sequence[float]) -> list[float]:
    """Intervals between consecutive spikes, in input order after sorting.

    Returns an empty list for fewer than two spikes.
    """
    times = sorted_finite(spikes)
    return [b - a for a, b in pairwise(times)]


def firing_rate(spikes: Sequence[float], *, duration: float) -> float:
    """Mean firing rate: spike count divided by the observation duration.

    The rate is expressed per unit of the spike-time unit (spikes per second when times
    are in seconds). `duration` must be positive.
    """
    check_positive("duration", duration)
    times = sorted_finite(spikes)
    return len(times) / duration


def cv_isi(spikes: Sequence[float]) -> float:
    """Coefficient of variation of the inter-spike intervals: pstdev(ISI) / mean(ISI).

    Uses the population standard deviation (matching the common np.std default). A regular
    train gives 0; a Poisson process gives 1 in expectation. Needs at least two spikes.
    """
    isis = inter_spike_intervals(spikes)
    require_spikes(len(isis) + 1, 2, "cv_isi")
    m = mean(isis)
    if m == 0:
        raise ValueError("cv_isi is undefined when the mean interval is zero")
    return float(pstdev(isis) / m)


def cv2(spikes: Sequence[float]) -> float:
    """Mean of the local CV2 over adjacent interval pairs (Holt et al. 1996).

    CV2 = mean over consecutive intervals of 2 * |I(n+1) - I(n)| / (I(n+1) + I(n)). It is
    designed to be robust to slow rate drift. Needs at least three spikes.
    """
    isis = inter_spike_intervals(spikes)
    require_spikes(len(isis) + 1, 3, "cv2")
    total = 0.0
    for a, b in pairwise(isis):
        denominator = a + b
        if denominator == 0:
            raise ValueError("cv2 is undefined when two adjacent intervals are both zero")
        total += 2.0 * abs(b - a) / denominator
    return total / (len(isis) - 1)


def lv(spikes: Sequence[float]) -> float:
    """Local variation of inter-spike intervals (Shinomoto et al. 2003).

    Lv = (1 / (N - 1)) * sum of 3 * ((I(n) - I(n+1)) / (I(n) + I(n+1)))^2 over the N
    intervals. A regular train gives 0, a Poisson process gives 1 in expectation. Needs
    at least three spikes.
    """
    isis = inter_spike_intervals(spikes)
    require_spikes(len(isis) + 1, 3, "lv")
    total = 0.0
    for a, b in pairwise(isis):
        denominator = a + b
        if denominator == 0:
            raise ValueError("lv is undefined when two adjacent intervals are both zero")
        total += 3.0 * ((a - b) / denominator) ** 2
    return total / (len(isis) - 1)


def lvr(spikes: Sequence[float], *, refractory: float) -> float:
    """Revised local variation LvR with a refractoriness constant (Shinomoto et al. 2009).

    LvR = (3 / (N - 1)) * sum of
        (1 - 4 * I(n) * I(n+1) / (I(n) + I(n+1))^2) * (1 + 4 * R / (I(n) + I(n+1)))
    where R is the refractoriness constant in the same time unit as the spikes. LvR
    corrects Lv for the effect of a refractory period. `refractory` must be non-negative
    and a regular train still gives 0. Needs at least three spikes.
    """
    check_non_negative("refractory", refractory)
    isis = inter_spike_intervals(spikes)
    require_spikes(len(isis) + 1, 3, "lvr")
    total = 0.0
    for a, b in pairwise(isis):
        denominator = a + b
        if denominator == 0:
            raise ValueError("lvr is undefined when two adjacent intervals are both zero")
        total += (1.0 - 4.0 * a * b / denominator**2) * (1.0 + 4.0 * refractory / denominator)
    return 3.0 * total / (len(isis) - 1)


def spike_counts(spikes: Sequence[float], *, duration: float, bin_width: float) -> list[int]:
    """Spike counts in consecutive bins of width `bin_width` tiling [0, n * bin_width).

    The number of bins n is floor(duration / bin_width); any remainder of the duration and
    any spikes outside [0, n * bin_width) are ignored, so every bin has equal width. Both
    `duration` and `bin_width` must be positive and the duration must be at least one bin.
    """
    check_positive("duration", duration)
    check_positive("bin_width", bin_width)
    n_bins = int(duration // bin_width)
    if n_bins < 1:
        raise ValueError(
            f"duration {duration!r} must be at least one bin of width {bin_width!r}"
        )
    edge = n_bins * bin_width
    counts = [0] * n_bins
    for t in sorted_finite(spikes):
        if 0.0 <= t < edge:
            index = min(int(t // bin_width), n_bins - 1)
            counts[index] += 1
    return counts


def fano_factor(spikes: Sequence[float], *, duration: float, bin_width: float) -> float:
    """Fano factor of binned spike counts: pvariance(counts) / mean(counts).

    Counts are taken over equal-width bins as in spike_counts. A Poisson process gives 1 in
    expectation; a regular train binned at an integer number of spikes per bin gives 0.
    Raises when no spikes fall inside the binned window.
    """
    counts = spike_counts(spikes, duration=duration, bin_width=bin_width)
    m = mean(counts)
    if m == 0:
        raise ValueError("fano_factor is undefined when no spikes fall in the binned window")
    return float(pvariance(counts) / m)


def _spikes_in_interval(
    spikes: Sequence[float], start: float, end: float, name: str
) -> list[float]:
    """Sort spikes and reject any time outside the closed recording interval [start, end]."""
    times = sorted_finite(spikes)
    for t in times:
        if t < start or t > end:
            raise ValueError(
                f"{name} spike time {t!r} lies outside the recording interval "
                f"[{start!r}, {end!r}]"
            )
    return times


def _tiled_fraction(times: list[float], dt: float, start: float, end: float) -> float:
    """Fraction of [start, end] covered by the union of [t - dt, t + dt] over `times`.

    Each window is clipped to the recording interval before the union is measured, so the
    result is the covered time divided by the total recording time (in [0, 1]).
    """
    if not times:
        return 0.0
    total = end - start
    covered = 0.0
    current_start = max(start, times[0] - dt)
    current_end = min(end, times[0] + dt)
    for t in times[1:]:
        window_start = max(start, t - dt)
        window_end = min(end, t + dt)
        if window_start <= current_end:
            current_end = max(current_end, window_end)
        else:
            covered += current_end - current_start
            current_start = window_start
            current_end = window_end
    covered += current_end - current_start
    return covered / total


def _fraction_near(reference: list[float], target: list[float], dt: float) -> float:
    """Fraction of `reference` spikes within +/- dt of any spike in `target`.

    Both inputs are sorted; uses a two-pointer sweep so the cost is linear in the train
    lengths. Returns 0.0 when `reference` is empty.
    """
    if not reference:
        return 0.0
    if not target:
        return 0.0
    near = 0
    j = 0
    n_target = len(target)
    for t in reference:
        while j < n_target and target[j] < t - dt:
            j += 1
        if j < n_target and target[j] <= t + dt:
            near += 1
    return near / len(reference)


def spike_time_tiling_coefficient(
    spikes_a: Sequence[float],
    spikes_b: Sequence[float],
    *,
    dt: float,
    interval: tuple[float, float],
) -> float:
    """Spike-time tiling coefficient (STTC) of two trains (Cutts and Eglen 2014).

    The STTC is a firing-rate-robust pairwise correlation measure. With a synchronicity
    window `dt` and a recording `interval` given as `(start, end)`:

    - TA is the fraction of the total recording time within +/- dt of any spike in A;
      TB is the same for B.
    - PA is the fraction of spikes in A that lie within +/- dt of any spike in B;
      PB is the fraction of spikes in B that lie within +/- dt of any spike in A.
    - STTC = 0.5 * ((PA - TB) / (1 - PA * TB) + (PB - TA) / (1 - PB * TA)).

    When a denominator `1 - P * T` is zero, that half-term contributes 0 (the Cutts and
    Eglen convention). When either train is empty, the STTC is defined to be 0.0, since the
    correlation between a train and an empty train is undefined and conventionally reported
    as zero. The result lies in `[-1, 1]`; identical trains give 1.0.

    `dt` must be positive. `interval` must be a `(start, end)` pair with `start < end`, and
    every spike must lie within the closed interval `[start, end]`. STTC is symmetric:
    `spike_time_tiling_coefficient(A, B, ...)` equals `spike_time_tiling_coefficient(B, A, ...)`.
    """
    check_positive("dt", dt)
    start, end = check_interval("interval", interval)
    times_a = _spikes_in_interval(spikes_a, start, end, "spikes_a")
    times_b = _spikes_in_interval(spikes_b, start, end, "spikes_b")

    if not times_a or not times_b:
        return 0.0

    tile_a = _tiled_fraction(times_a, dt, start, end)
    tile_b = _tiled_fraction(times_b, dt, start, end)
    prop_a = _fraction_near(times_a, times_b, dt)
    prop_b = _fraction_near(times_b, times_a, dt)

    denominator_a = 1.0 - prop_a * tile_b
    denominator_b = 1.0 - prop_b * tile_a
    half_a = 0.0 if denominator_a == 0.0 else (prop_a - tile_b) / denominator_a
    half_b = 0.0 if denominator_b == 0.0 else (prop_b - tile_a) / denominator_b
    return 0.5 * (half_a + half_b)


def _correlogram_bins(bin_width: float, max_lag: float) -> int:
    """Number of lag bins on each side of zero: ceil(max_lag / bin_width).

    Validates that `bin_width` and `max_lag` are positive and that `max_lag` is at least
    one bin wide, then returns the half-width count `n`. The full correlogram has `2 * n + 1`
    bins (an odd number, with one bin centered exactly on lag zero).
    """
    check_positive("bin_width", bin_width)
    check_positive("max_lag", max_lag)
    if max_lag < bin_width:
        raise ValueError(
            f"max_lag {max_lag!r} must be at least one bin of width {bin_width!r}"
        )
    return ceil(max_lag / bin_width)


def cross_correlogram(
    reference: Sequence[float],
    target: Sequence[float],
    *,
    bin_width: float,
    max_lag: float,
) -> list[int]:
    """Binned cross-correlogram (CCG) of two spike trains.

    For every spike in `reference` and every spike in `target`, the lag is
    `target_time - reference_time`. Each lag is histogrammed into one of `2 * n + 1`
    symmetric bins, where `n = ceil(max_lag / bin_width)`. The bins are centered on lag
    zero: bin index `k` (for `k` in `0 .. 2 * n`) covers the half-open lag interval

        [(k - n - 0.5) * bin_width, (k - n + 0.5) * bin_width),

    so the central bin (index `n`) covers `[-bin_width / 2, +bin_width / 2)` and is centered
    exactly on lag zero. A lag is counted when it falls inside this bin grid, i.e. inside
    `[-(n + 0.5) * bin_width, +(n + 0.5) * bin_width)`; lags outside are ignored. Because
    `n = ceil(max_lag / bin_width)`, the grid always covers at least `[-max_lag, max_lag]`.

    `bin_width` and `max_lag` must both be positive and `max_lag` must be at least one
    `bin_width`. Either train may be empty, in which case there are no pairs and every bin
    is zero.

    Returns a list of `2 * n + 1` non-negative integer counts, ordered from the most
    negative lag bin to the most positive. The sign convention is `target - reference`, so
    `cross_correlogram(a, b)` reversed equals `cross_correlogram(b, a)`.
    """
    n = _correlogram_bins(bin_width, max_lag)
    counts = [0] * (2 * n + 1)
    ref_times = sorted_finite(reference)
    tgt_times = sorted_finite(target)
    for r in ref_times:
        for t in tgt_times:
            # Nearest-bin index via floor(x + 0.5); the +0.5 places half-open bin edges at
            # odd multiples of bin_width / 2, matching the documented bin intervals.
            index = floor((t - r) / bin_width + 0.5) + n
            if 0 <= index <= 2 * n:
                counts[index] += 1
    return counts


def autocorrelogram(
    spikes: Sequence[float],
    *,
    bin_width: float,
    max_lag: float,
) -> list[int]:
    """Binned autocorrelogram (ACG) of a spike train: its CCG with itself.

    The lag of every ordered pair of distinct spikes `(i, j)` with `i != j` is
    `spikes[j] - spikes[i]`, histogrammed using the same symmetric bin grid as
    `cross_correlogram`: `2 * n + 1` bins with `n = ceil(max_lag / bin_width)`, the central
    bin centered on lag zero. The trivial zero-lag self-pairs `i == j` are excluded (the
    standard ACG convention), so the central bin counts only genuine pairs of distinct
    spikes whose lag falls within `[-bin_width / 2, +bin_width / 2)`, not the `N` self
    coincidences.

    Because every ordered pair `(i, j)` with `i != j` has a mirror pair `(j, i)` of the
    opposite lag, the ACG is symmetric: `acg == acg` reversed.

    `bin_width` and `max_lag` must both be positive and `max_lag` must be at least one
    `bin_width`. A train with fewer than two spikes has no distinct pairs, so every bin is
    zero.

    Returns a list of `2 * n + 1` non-negative integer counts, ordered from the most
    negative lag bin to the most positive.
    """
    n = _correlogram_bins(bin_width, max_lag)
    counts = [0] * (2 * n + 1)
    times = sorted_finite(spikes)
    n_spikes = len(times)
    for i in range(n_spikes):
        a = times[i]
        for j in range(n_spikes):
            if i == j:
                continue
            index = floor((times[j] - a) / bin_width + 0.5) + n
            if 0 <= index <= 2 * n:
                counts[index] += 1
    return counts
