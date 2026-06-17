"""Tests for time_resolved.py: exact golden values, boundary cases, and property tests."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from spikestats import (
    binned_spike_counts,
    psth,
    time_resolved_fano,
    time_resolved_rate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regular_train(n_spikes: int, interval: float) -> list[float]:
    """Regular spike train starting at interval/2, with `n_spikes` spikes."""
    return [interval / 2.0 + i * interval for i in range(n_spikes)]


# ---------------------------------------------------------------------------
# binned_spike_counts
# ---------------------------------------------------------------------------


class TestBinnedSpikeCounts:
    def test_basic_binning(self) -> None:
        # Spikes at 0.1, 1.5, 1.6, 2.5; duration=3, bin_width=1 => bins [0,1), [1,2), [2,3)
        result = binned_spike_counts([0.1, 1.5, 1.6, 2.5], duration=3.0, bin_width=1.0)
        assert result == [1, 2, 1]

    def test_length_equals_int_duration_over_bin_width(self) -> None:
        result = binned_spike_counts([], duration=1.0, bin_width=0.25)
        assert len(result) == 4

    def test_spike_at_duration_excluded(self) -> None:
        # Spike exactly at t=1.0 with duration=1.0 should be excluded.
        result = binned_spike_counts([1.0], duration=1.0, bin_width=0.5)
        assert result == [0, 0]

    def test_spike_at_zero_included(self) -> None:
        result = binned_spike_counts([0.0], duration=1.0, bin_width=0.5)
        assert result == [1, 0]

    def test_negative_spike_excluded(self) -> None:
        result = binned_spike_counts([-0.1, 0.5], duration=1.0, bin_width=0.5)
        assert result == [0, 1]

    def test_empty_train_all_zeros(self) -> None:
        result = binned_spike_counts([], duration=2.0, bin_width=0.5)
        assert result == [0, 0, 0, 0]

    def test_sorts_input(self) -> None:
        result = binned_spike_counts([2.5, 0.1, 1.5], duration=3.0, bin_width=1.0)
        assert result == [1, 1, 1]

    def test_sum_equals_in_range_count(self) -> None:
        # Spike at 3.0 is out-of-range (duration=3.0 excludes t>=3.0).
        spikes = [0.1, 0.9, 1.5, 2.7, 3.0, 3.5]
        result = binned_spike_counts(spikes, duration=3.0, bin_width=1.0)
        in_range = sum(1 for t in spikes if 0.0 <= t < 3.0)
        assert sum(result) == in_range

    def test_bin_width_equals_duration(self) -> None:
        # One bin covering [0, 1).
        result = binned_spike_counts([0.1, 0.5, 0.9], duration=1.0, bin_width=1.0)
        assert result == [3]

    def test_bin_width_wider_than_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="bin_width"):
            binned_spike_counts([0.1], duration=0.5, bin_width=1.0)

    def test_non_positive_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            binned_spike_counts([], duration=0.0, bin_width=0.1)

    def test_non_positive_bin_width_raises(self) -> None:
        with pytest.raises(ValueError, match="bin_width"):
            binned_spike_counts([], duration=1.0, bin_width=0.0)


# ---------------------------------------------------------------------------
# time_resolved_rate
# ---------------------------------------------------------------------------


class TestTimeResolvedRate:
    def test_non_overlapping_matches_binned_counts(self) -> None:
        # With step == bin_width the sliding window is identical to non-overlapping bins.
        spikes = [0.1, 1.5, 1.6, 2.5]
        centers, rates = time_resolved_rate(spikes, duration=3.0, bin_width=1.0, step=1.0)
        assert centers == pytest.approx([0.5, 1.5, 2.5])
        assert rates == pytest.approx([1.0, 2.0, 1.0])

    def test_rate_is_count_over_bin_width(self) -> None:
        # One spike in the first 0.1-s window => rate = 1 / 0.1 = 10 Hz.
        spikes = [0.05]
        _, rates = time_resolved_rate(spikes, duration=0.2, bin_width=0.1, step=0.1)
        assert len(rates) == 2
        assert rates[0] == pytest.approx(10.0)
        assert rates[1] == pytest.approx(0.0)

    def test_overlapping_windows(self) -> None:
        # bin_width=0.2, step=0.1, duration=0.4 => windows at [0,.2), [.1,.3), [.2,.4)
        spikes = [0.05, 0.15, 0.25]
        centers, rates = time_resolved_rate(spikes, duration=0.4, bin_width=0.2, step=0.1)
        assert len(centers) == 3
        # Window [0, 0.2): spikes at 0.05, 0.15 => rate = 2 / 0.2 = 10
        assert rates[0] == pytest.approx(10.0)
        # Window [0.1, 0.3): spikes at 0.15, 0.25 => rate = 2 / 0.2 = 10
        assert rates[1] == pytest.approx(10.0)
        # Window [0.2, 0.4): spike at 0.25 => rate = 1 / 0.2 = 5
        assert rates[2] == pytest.approx(5.0)

    def test_center_times_correct(self) -> None:
        centers, _ = time_resolved_rate([], duration=1.0, bin_width=0.4, step=0.2)
        # Windows: [0, 0.4), [0.2, 0.6), [0.4, 0.8), [0.6, 1.0) => 4 windows
        assert centers == pytest.approx([0.2, 0.4, 0.6, 0.8])

    def test_no_window_fits_returns_empty(self) -> None:
        # bin_width == duration; step > 0 but only one window can fit.
        centers, rates = time_resolved_rate([], duration=0.5, bin_width=0.5, step=0.5)
        assert centers == [0.25]
        assert rates == [0.0]

    def test_integral_over_nonoverlapping_equals_spike_count(self) -> None:
        # For step == bin_width (non-overlapping), sum(rate * bin_width) == in-range count.
        spikes = [0.1, 0.5, 0.7, 1.8, 2.2]
        _, rates = time_resolved_rate(spikes, duration=3.0, bin_width=1.0, step=1.0)
        total = sum(r * 1.0 for r in rates)
        in_range = sum(1 for t in spikes if 0.0 <= t < 3.0)
        assert total == pytest.approx(in_range)

    def test_rates_are_non_negative(self) -> None:
        _, rates = time_resolved_rate(
            [0.1, 0.5, 1.2], duration=2.0, bin_width=0.5, step=0.25
        )
        assert all(r >= 0.0 for r in rates)

    def test_non_positive_step_raises(self) -> None:
        with pytest.raises(ValueError, match="step"):
            time_resolved_rate([], duration=1.0, bin_width=0.1, step=0.0)

    def test_bin_width_exceeds_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="bin_width"):
            time_resolved_rate([], duration=0.5, bin_width=1.0, step=0.1)


# ---------------------------------------------------------------------------
# psth
# ---------------------------------------------------------------------------


class TestPsth:
    def test_single_trial_matches_rate(self) -> None:
        # Single trial: PSTH should equal the per-bin rate for that trial.
        spikes = [0.1, 1.5, 1.6, 2.5]
        bin_centers, mean_rates = psth([spikes], duration=3.0, bin_width=1.0)
        assert bin_centers == pytest.approx([0.5, 1.5, 2.5])
        # counts [1, 2, 1], rates [1, 2, 1]
        assert mean_rates == pytest.approx([1.0, 2.0, 1.0])

    def test_two_trials_averaged(self) -> None:
        # Trial 1: spike at 0.1 (bin 0), Trial 2: spike at 0.6 (bin 0).
        # Mean count in bin 0 = (1 + 1) / 2 = 1, rate = 1 / 1.0 = 1.0.
        # All other bins empty.
        _, mean_rates = psth([[0.1], [0.6]], duration=2.0, bin_width=1.0)
        assert mean_rates[0] == pytest.approx(1.0)
        assert mean_rates[1] == pytest.approx(0.0)

    def test_bin_centers_correct(self) -> None:
        bin_centers, _ = psth([[]], duration=1.0, bin_width=0.25)
        assert bin_centers == pytest.approx([0.125, 0.375, 0.625, 0.875])

    def test_empty_trials_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one trial"):
            psth([], duration=1.0, bin_width=0.1)

    def test_all_zero_trials_gives_zero_rates(self) -> None:
        _, mean_rates = psth([[], []], duration=1.0, bin_width=0.5)
        assert mean_rates == [0.0, 0.0]

    def test_sum_of_rates_times_bin_width_equals_mean_spike_count(self) -> None:
        # sum(rate * bin_width) over bins == mean spike count per trial.
        trials = [[0.1, 0.9, 1.5], [0.2, 1.8, 2.1, 2.5]]
        bin_width = 1.0
        duration = 3.0
        _, mean_rates = psth(trials, duration=duration, bin_width=bin_width)
        total_rate_integral = sum(r * bin_width for r in mean_rates)
        mean_in_range_count = sum(
            sum(1 for t in trial if 0.0 <= t < duration) for trial in trials
        ) / len(trials)
        assert total_rate_integral == pytest.approx(mean_in_range_count)

    def test_bin_width_exceeds_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="bin_width"):
            psth([[0.1]], duration=0.5, bin_width=1.0)


# ---------------------------------------------------------------------------
# time_resolved_fano
# ---------------------------------------------------------------------------


class TestTimeResolvedFano:
    def test_single_trial_is_zero(self) -> None:
        # pvariance of a single value is 0, so Fano = 0 for all non-empty bins.
        _, fano = time_resolved_fano([[0.1, 0.5]], duration=1.0, bin_width=0.5)
        # Bin 0: count=1, bin 1: count=1 => pvariance([1])=0, fano=0
        assert fano[0] == pytest.approx(0.0)
        assert fano[1] == pytest.approx(0.0)

    def test_identical_trials_fano_is_zero(self) -> None:
        # Identical counts across all trials => variance 0 => Fano 0.
        spikes = [0.1, 0.5, 1.5]
        trials = [spikes, spikes, spikes]
        _, fano = time_resolved_fano(trials, duration=2.0, bin_width=1.0)
        for f in fano:
            if not math.isnan(f):
                assert f == pytest.approx(0.0)

    def test_zero_mean_bin_returns_nan(self) -> None:
        # Two trials both with no spikes in bin 1 => mean=0, Fano=nan.
        trials = [[0.1], [0.2]]  # all spikes in bin 0 only
        _, fano = time_resolved_fano(trials, duration=2.0, bin_width=1.0)
        assert fano[0] == pytest.approx(0.0)
        assert math.isnan(fano[1])

    def test_known_value(self) -> None:
        # Bin 0 counts across 3 trials: [1, 2, 3].
        # mean=2, pvariance([1,2,3]) = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3.
        # Fano = (2/3) / 2 = 1/3.
        trials: list[list[float]] = [
            [0.1],           # 1 spike in bin 0
            [0.1, 0.2],      # 2 spikes in bin 0
            [0.1, 0.2, 0.3], # 3 spikes in bin 0
        ]
        _, fano = time_resolved_fano(trials, duration=1.0, bin_width=1.0)
        assert fano[0] == pytest.approx(1 / 3)

    def test_returns_correct_length(self) -> None:
        bin_centers, fano = time_resolved_fano([[]], duration=1.0, bin_width=0.25)
        assert len(bin_centers) == 4
        assert len(fano) == 4

    def test_bin_centers_correct(self) -> None:
        bin_centers, _ = time_resolved_fano([[]], duration=1.0, bin_width=0.5)
        assert bin_centers == pytest.approx([0.25, 0.75])

    def test_empty_trials_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one trial"):
            time_resolved_fano([], duration=1.0, bin_width=0.1)

    def test_fano_non_negative_for_non_nan_bins(self) -> None:
        trials = [[0.1, 0.5, 1.5], [0.2, 0.8, 1.9], [0.3, 0.6, 1.2]]
        _, fano = time_resolved_fano(trials, duration=2.0, bin_width=1.0)
        for f in fano:
            if not math.isnan(f):
                assert f >= 0.0

    def test_bin_width_exceeds_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="bin_width"):
            time_resolved_fano([[0.1]], duration=0.5, bin_width=1.0)


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

spikes_strategy = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=50,
)

positive_param = st.floats(
    min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False
)


@given(spikes_strategy, positive_param)
def test_binned_counts_sum_to_in_range(spikes: list[float], bin_width: float) -> None:
    """Sum of binned counts equals number of spikes in [0, duration)."""
    duration = 1.0
    if bin_width > duration:
        return
    result = binned_spike_counts(spikes, duration=duration, bin_width=bin_width)
    in_range = sum(1 for t in spikes if 0.0 <= t < duration)
    assert sum(result) == in_range


@given(spikes_strategy, positive_param, positive_param)
def test_time_resolved_rate_nonnegative(
    spikes: list[float], bin_width: float, step: float
) -> None:
    """All rates returned by time_resolved_rate are non-negative."""
    duration = 1.0
    if bin_width > duration:
        return
    _, rates = time_resolved_rate(spikes, duration=duration, bin_width=bin_width, step=step)
    assert all(r >= 0.0 for r in rates)


@given(spikes_strategy, positive_param)
def test_nonoverlapping_rate_integral_equals_in_range_count(
    spikes: list[float], bin_width: float
) -> None:
    """For step==bin_width, sum(rate * bin_width) equals count within the tiled window.

    Non-overlapping windows tile [0, n_windows * bin_width) where n_windows =
    int(duration / bin_width). Spikes in the remainder [n_windows * bin_width, duration)
    have no window, matching the spike_counts behaviour.
    """
    duration = 1.0
    if bin_width > duration:
        return
    n_windows = int(duration / bin_width)
    tiled_end = n_windows * bin_width
    _, rates = time_resolved_rate(spikes, duration=duration, bin_width=bin_width, step=bin_width)
    total = sum(r * bin_width for r in rates)
    in_tiled = sum(1 for t in spikes if 0.0 <= t < tiled_end)
    assert total == pytest.approx(in_tiled, abs=1e-9)


@given(
    st.lists(spikes_strategy, min_size=1, max_size=5),
    positive_param,
)
def test_psth_rate_integral_equals_mean_in_range_count(
    trials: list[list[float]], bin_width: float
) -> None:
    """sum(mean_rate * bin_width) equals mean number of in-range spikes across trials."""
    duration = 1.0
    if bin_width > duration:
        return
    _, mean_rates = psth(trials, duration=duration, bin_width=bin_width)
    total = sum(r * bin_width for r in mean_rates)
    mean_count = sum(
        sum(1 for t in trial if 0.0 <= t < duration) for trial in trials
    ) / len(trials)
    assert total == pytest.approx(mean_count, abs=1e-9)


@given(
    st.lists(spikes_strategy, min_size=1, max_size=5),
    positive_param,
)
def test_time_resolved_fano_nonneg_or_nan(
    trials: list[list[float]], bin_width: float
) -> None:
    """time_resolved_fano returns non-negative values or nan for zero-mean bins."""
    duration = 1.0
    if bin_width > duration:
        return
    _, fano = time_resolved_fano(trials, duration=duration, bin_width=bin_width)
    for f in fano:
        assert math.isnan(f) or f >= 0.0


@given(spikes_strategy, positive_param)
def test_single_trial_fano_is_zero(spikes: list[float], bin_width: float) -> None:
    """With a single trial, pvariance is 0, so Fano is 0 for all non-empty bins."""
    duration = 1.0
    if bin_width > duration:
        return
    _, fano = time_resolved_fano([spikes], duration=duration, bin_width=bin_width)
    for f in fano:
        # bins with mean 0 are nan; all others must be 0 (pvariance of a single value)
        assert math.isnan(f) or f == pytest.approx(0.0, abs=1e-12)
