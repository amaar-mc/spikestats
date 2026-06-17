"""Spike-train statistics over plain lists of spike times.

Every function takes a sequence of spike times (in seconds, any unit is fine as long as
it is consistent) and returns a plain float. Inputs are sorted internally, so the caller
does not need to pre-sort. The formulas follow the standard definitions cited in the
docstrings; see docs/architecture.md for the references.
"""

from collections.abc import Sequence
from itertools import pairwise
from statistics import mean, pstdev, pvariance

from ._validate import check_non_negative, check_positive, require_spikes, sorted_finite


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
