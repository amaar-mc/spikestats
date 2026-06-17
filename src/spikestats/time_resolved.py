"""Time-resolved (sliding-window) spike-train metrics.

These functions return one value per time bin rather than a single scalar over
the whole train. The bin boundary policy throughout this module is:

- A spike at time t is assigned to the bin whose left edge e satisfies e <= t < e + bin_width.
- A spike exactly at `duration` is excluded (half-open interval [0, duration)).
- Negative spike times are excluded silently by the boundary checks, matching the
  behaviour of `spike_counts` in metrics.py.

All functions validate their parameters explicitly. No default values are provided;
every parameter must be supplied by the caller.
"""

from collections.abc import Sequence
from statistics import mean, pvariance

from ._validate import check_positive, sorted_finite


def binned_spike_counts(
    spikes: Sequence[float],
    *,
    duration: float,
    bin_width: float,
) -> list[int]:
    """Spike counts in consecutive non-overlapping bins tiling [0, duration).

    The number of bins is int(duration / bin_width). Each bin covers
    [k * bin_width, (k + 1) * bin_width) for k = 0, 1, ..., n_bins - 1.
    Spikes outside [0, duration) are excluded. Both `duration` and `bin_width`
    must be positive and `bin_width` must not exceed `duration`.

    Returns a list of length int(duration / bin_width).
    """
    check_positive("duration", duration)
    check_positive("bin_width", bin_width)
    if bin_width > duration:
        raise ValueError(
            f"bin_width {bin_width!r} must not exceed duration {duration!r}"
        )
    n_bins = int(duration / bin_width)
    counts: list[int] = [0] * n_bins
    for t in sorted_finite(spikes):
        if 0.0 <= t < duration:
            index = min(int(t / bin_width), n_bins - 1)
            counts[index] += 1
    return counts


def time_resolved_rate(
    spikes: Sequence[float],
    *,
    duration: float,
    bin_width: float,
    step: float,
) -> tuple[list[float], list[float]]:
    """Sliding-window firing rate across [0, duration).

    A window of width `bin_width` is stepped by `step` from the left. Window
    positions are computed by integer indexing of the step size to avoid
    floating-point drift: window k starts at k * step and ends at k * step +
    bin_width. The last window is the largest k such that k * step + bin_width
    <= duration.

    Rate for each window = (spike count in window) / bin_width, expressed in
    spikes per unit of the input time (Hz when times are in seconds).

    The center of window k is k * step + bin_width / 2.

    `duration`, `bin_width`, and `step` must all be positive. `bin_width` must
    not exceed `duration`.

    Returns (window_center_times, rates_in_hz), each a list of the same length.
    """
    check_positive("duration", duration)
    check_positive("bin_width", bin_width)
    check_positive("step", step)
    if bin_width > duration:
        raise ValueError(
            f"bin_width {bin_width!r} must not exceed duration {duration!r}"
        )
    times = sorted_finite(spikes)

    # Determine window positions using integer arithmetic.
    n_windows = 0
    k = 0
    while True:
        window_end = k * step + bin_width
        if window_end > duration:
            break
        n_windows += 1
        k += 1

    centers: list[float] = []
    rates: list[float] = []
    half = bin_width / 2.0
    for k in range(n_windows):
        w_start = k * step
        w_end = w_start + bin_width
        count = sum(1 for t in times if w_start <= t < w_end)
        centers.append(w_start + half)
        rates.append(count / bin_width)
    return centers, rates


def psth(
    trials: Sequence[Sequence[float]],
    *,
    duration: float,
    bin_width: float,
) -> tuple[list[float], list[float]]:
    """Peristimulus time histogram averaged over multiple trials.

    For each equal-width bin tiling [0, duration), the per-trial spike count is
    divided by bin_width to obtain a rate in Hz, then averaged across trials.
    The number of bins is int(duration / bin_width).

    `trials` must be non-empty. `duration` and `bin_width` must be positive and
    `bin_width` must not exceed `duration`.

    Returns (bin_center_times, mean_rate_per_bin_hz), each a list of length
    int(duration / bin_width).
    """
    check_positive("duration", duration)
    check_positive("bin_width", bin_width)
    if bin_width > duration:
        raise ValueError(
            f"bin_width {bin_width!r} must not exceed duration {duration!r}"
        )
    if len(trials) == 0:
        raise ValueError("psth requires at least one trial; received an empty sequence")
    n_bins = int(duration / bin_width)

    # Sum counts per bin across all trials.
    total_counts: list[int] = [0] * n_bins
    n_trials = len(trials)
    for trial in trials:
        trial_counts = binned_spike_counts(trial, duration=duration, bin_width=bin_width)
        for i in range(n_bins):
            total_counts[i] += trial_counts[i]

    half = bin_width / 2.0
    bin_centers = [k * bin_width + half for k in range(n_bins)]
    # Average count over trials, convert to rate.
    mean_rates = [total_counts[i] / n_trials / bin_width for i in range(n_bins)]
    return bin_centers, mean_rates


def time_resolved_fano(
    trials: Sequence[Sequence[float]],
    *,
    duration: float,
    bin_width: float,
) -> tuple[list[float], list[float]]:
    """Per-bin Fano factor (population variance / mean of per-trial counts) across trials.

    For each equal-width bin k tiling [0, duration), the spike count in that bin
    is collected across all trials, and the Fano factor is variance(counts) /
    mean(counts). This uses the population variance (pvariance, denominator N),
    matching the convention of `fano_factor` in metrics.py.

    With a single trial (N = 1), pvariance is 0 for every bin regardless of
    count; the returned Fano values will all be 0. This is documented behaviour:
    the N-of-1 Fano factor is 0 by the population-variance definition. Callers
    who want sample variance (denominator N - 1) should note that it is
    undefined for N = 1 and use multiple trials.

    A bin whose mean count is 0 across all trials cannot have a Fano factor
    defined; those bins are returned as float('nan') with a consistent policy so
    callers can identify and exclude them.

    `trials` must be non-empty. `duration` and `bin_width` must be positive and
    `bin_width` must not exceed `duration`.

    Returns (bin_center_times, fano_per_bin), each a list of length
    int(duration / bin_width).
    """
    check_positive("duration", duration)
    check_positive("bin_width", bin_width)
    if bin_width > duration:
        raise ValueError(
            f"bin_width {bin_width!r} must not exceed duration {duration!r}"
        )
    if len(trials) == 0:
        raise ValueError(
            "time_resolved_fano requires at least one trial; received an empty sequence"
        )
    n_bins = int(duration / bin_width)

    # Collect per-bin counts across trials: counts_by_bin[k] = list of per-trial counts.
    counts_by_bin: list[list[int]] = [[] for _ in range(n_bins)]
    for trial in trials:
        trial_counts = binned_spike_counts(trial, duration=duration, bin_width=bin_width)
        for k in range(n_bins):
            counts_by_bin[k].append(trial_counts[k])

    half = bin_width / 2.0
    bin_centers = [k * bin_width + half for k in range(n_bins)]
    fano_values: list[float] = []
    for k in range(n_bins):
        bin_counts = counts_by_bin[k]
        m = mean(bin_counts)
        if m == 0.0:
            fano_values.append(float("nan"))
        else:
            fano_values.append(pvariance(bin_counts) / m)
    return bin_centers, fano_values
