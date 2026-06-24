import math
import random

import pytest

from spikestats import (
    autocorrelogram,
    cross_correlogram,
    cv2,
    cv_isi,
    fano_factor,
    firing_rate,
    inter_spike_intervals,
    lv,
    lvr,
    spike_counts,
    spike_time_tiling_coefficient,
)


def poisson_train(rate: float, duration: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    times: list[float] = []
    t = 0.0
    while True:
        t += -math.log(1.0 - rng.random()) / rate
        if t >= duration:
            break
        times.append(t)
    return times


class TestIntervals:
    def test_consecutive_differences(self) -> None:
        assert inter_spike_intervals([0.0, 0.1, 0.3, 0.6]) == pytest.approx([0.1, 0.2, 0.3])

    def test_sorts_input(self) -> None:
        assert inter_spike_intervals([0.3, 0.0, 0.1]) == pytest.approx([0.1, 0.2])

    def test_too_few_spikes_is_empty(self) -> None:
        assert inter_spike_intervals([]) == []
        assert inter_spike_intervals([5.0]) == []

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            inter_spike_intervals([0.0, math.inf])


class TestFiringRate:
    def test_count_over_duration(self) -> None:
        assert firing_rate([0.0, 1.0, 2.0, 3.0], duration=4.0) == 1.0

    def test_empty_train_is_zero(self) -> None:
        assert firing_rate([], duration=2.0) == 0.0

    def test_non_positive_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            firing_rate([0.0, 1.0], duration=0.0)


class TestCvIsi:
    def test_regular_train_is_zero(self) -> None:
        assert cv_isi([0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self) -> None:
        # ISIs [0.1, 0.2, 0.3], mean 0.2, population stdev sqrt(0.02/3).
        expected = math.sqrt(0.02 / 3) / 0.2
        assert cv_isi([0.0, 0.1, 0.3, 0.6]) == pytest.approx(expected)

    def test_needs_two_spikes(self) -> None:
        with pytest.raises(ValueError, match="cv_isi needs at least 2 spikes"):
            cv_isi([0.0])


class TestCv2:
    def test_regular_train_is_zero(self) -> None:
        assert cv2([0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self) -> None:
        # ISIs [1, 3], single pair: 2 * |3 - 1| / (3 + 1) = 1.
        assert cv2([0.0, 1.0, 4.0]) == pytest.approx(1.0)

    def test_needs_three_spikes(self) -> None:
        with pytest.raises(ValueError, match="cv2 needs at least 3 spikes"):
            cv2([0.0, 1.0])


class TestLv:
    def test_regular_train_is_zero(self) -> None:
        assert lv([0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self) -> None:
        # ISIs [1, 3], single pair: 3 * ((1 - 3) / 4) ** 2 = 0.75.
        assert lv([0.0, 1.0, 4.0]) == pytest.approx(0.75)

    def test_needs_three_spikes(self) -> None:
        with pytest.raises(ValueError, match="lv needs at least 3 spikes"):
            lv([0.0, 1.0])


class TestLvr:
    def test_regular_train_is_zero(self) -> None:
        assert lvr([0.0, 1.0, 2.0, 3.0], refractory=0.002) == pytest.approx(0.0, abs=1e-12)

    def test_equals_lv_when_refractory_zero(self) -> None:
        assert lvr([0.0, 1.0, 4.0], refractory=0.0) == pytest.approx(lv([0.0, 1.0, 4.0]))

    def test_negative_refractory_raises(self) -> None:
        with pytest.raises(ValueError, match="refractory"):
            lvr([0.0, 1.0, 4.0], refractory=-1.0)


class TestSpikeCounts:
    def test_bins(self) -> None:
        assert spike_counts([0.5, 1.5, 1.6, 2.5], duration=3.0, bin_width=1.0) == [1, 2, 1]

    def test_remainder_and_outside_ignored(self) -> None:
        # n_bins = 3, edge = 3.0; the spike at 3.2 is outside [0, 3).
        assert spike_counts([0.5, 1.5, 1.6, 2.5, 3.2], duration=3.5, bin_width=1.0) == [1, 2, 1]

    def test_bin_wider_than_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one bin"):
            spike_counts([0.1], duration=0.5, bin_width=1.0)


class TestFanoFactor:
    def test_regular_binning_is_zero(self) -> None:
        spikes = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert fano_factor(spikes, duration=6.0, bin_width=2.0) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self) -> None:
        # counts [1, 2, 1], mean 4/3, population variance 2/9.
        spikes = [0.5, 1.5, 1.6, 2.5]
        assert fano_factor(spikes, duration=3.0, bin_width=1.0) == pytest.approx((2 / 9) / (4 / 3))

    def test_no_spikes_in_window_raises(self) -> None:
        with pytest.raises(ValueError, match="no spikes"):
            fano_factor([], duration=3.0, bin_width=1.0)


class TestSpikeTimeTilingCoefficient:
    def test_identical_trains_is_one(self) -> None:
        spikes = [1.0, 2.0, 3.0, 7.0]
        assert spike_time_tiling_coefficient(
            spikes, spikes, dt=0.05, interval=(0.0, 10.0)
        ) == pytest.approx(1.0)

    def test_worked_example(self) -> None:
        # A = [1.0, 5.0], B = [1.2, 8.0], dt = 0.5, interval (0, 10).
        # TA: union of [0.5,1.5] and [4.5,5.5] = 2.0 / 10 = 0.2.
        # TB: union of [0.7,1.7] and [7.5,8.5] = 2.0 / 10 = 0.2.
        # PA: A spike 1.0 is within 0.5 of B spike 1.2 (yes); 5.0 is not => 1/2 = 0.5.
        # PB: B spike 1.2 is within 0.5 of A spike 1.0 (yes); 8.0 is not => 1/2 = 0.5.
        # STTC = 0.5 * ((0.5 - 0.2)/(1 - 0.5*0.2) + (0.5 - 0.2)/(1 - 0.5*0.2))
        #      = 0.5 * (0.3/0.9 + 0.3/0.9) = 1/3.
        a = [1.0, 5.0]
        b = [1.2, 8.0]
        assert spike_time_tiling_coefficient(
            a, b, dt=0.5, interval=(0.0, 10.0)
        ) == pytest.approx(1.0 / 3.0)

    def test_no_coincidences_is_negative(self) -> None:
        # A = [1.0], B = [9.0], dt = 0.5, interval (0, 10).
        # TA = 1.0/10 = 0.1, TB = 1.0/10 = 0.1, PA = 0, PB = 0.
        # STTC = 0.5 * ((0 - 0.1)/(1 - 0) + (0 - 0.1)/(1 - 0)) = -0.1.
        assert spike_time_tiling_coefficient(
            [1.0], [9.0], dt=0.5, interval=(0.0, 10.0)
        ) == pytest.approx(-0.1)

    def test_symmetry(self) -> None:
        a = [1.0, 3.5, 6.2, 9.1]
        b = [1.3, 3.4, 7.0]
        ab = spike_time_tiling_coefficient(a, b, dt=0.4, interval=(0.0, 10.0))
        ba = spike_time_tiling_coefficient(b, a, dt=0.4, interval=(0.0, 10.0))
        assert ab == pytest.approx(ba)

    def test_saturating_dt_gives_zero(self) -> None:
        # dt large enough that both windows cover the whole interval => TA = TB = 1.
        # Every spike is within dt of every other spike => PA = PB = 1.
        # Each half-term has denominator 1 - 1*1 = 0, so contributes 0 => STTC = 0.
        a = [2.0, 4.0]
        b = [6.0, 8.0]
        assert spike_time_tiling_coefficient(
            a, b, dt=10.0, interval=(0.0, 10.0)
        ) == pytest.approx(0.0)

    def test_empty_train_returns_zero(self) -> None:
        assert spike_time_tiling_coefficient(
            [], [1.0, 2.0], dt=0.5, interval=(0.0, 10.0)
        ) == 0.0
        assert spike_time_tiling_coefficient(
            [], [], dt=0.5, interval=(0.0, 10.0)
        ) == 0.0

    def test_non_positive_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            spike_time_tiling_coefficient([1.0], [2.0], dt=0.0, interval=(0.0, 10.0))

    def test_bad_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval"):
            spike_time_tiling_coefficient([1.0], [2.0], dt=0.5, interval=(10.0, 0.0))

    def test_spike_outside_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="outside the recording interval"):
            spike_time_tiling_coefficient([11.0], [2.0], dt=0.5, interval=(0.0, 10.0))


class TestCrossCorrelogram:
    def test_worked_example(self) -> None:
        # reference = [0.0], target = [0.005, 0.012], bin_width = 0.005, max_lag = 0.02.
        # n = ceil(0.02 / 0.005) = 4, so 2*4+1 = 9 bins, center index 4 (lag 0).
        # Bin k covers [(k - 4 - 0.5)*0.005, (k - 4 + 0.5)*0.005).
        # Lag 0.005 -> floor(0.005/0.005 + 0.5) + 4 = floor(1.5) + 4 = 5;
        #   bin 5 covers [0.0025, 0.0075), holds 0.005.
        # Lag 0.012 -> floor(0.012/0.005 + 0.5) + 4 = floor(2.9) + 4 = 6;
        #   bin 6 covers [0.0075, 0.0125), holds 0.012.
        assert cross_correlogram(
            [0.0], [0.005, 0.012], bin_width=0.005, max_lag=0.02
        ) == [0, 0, 0, 0, 0, 1, 1, 0, 0]

    def test_length_is_odd_two_n_plus_one(self) -> None:
        # n = ceil(0.05 / 0.01) = 5 => 11 bins.
        result = cross_correlogram([0.0], [0.0], bin_width=0.01, max_lag=0.05)
        assert len(result) == 11

    def test_self_pair_lands_in_central_bin(self) -> None:
        # reference == target single spike: the only pair has lag 0, central bin (index n).
        result = cross_correlogram([1.0], [1.0], bin_width=0.01, max_lag=0.05)
        center = len(result) // 2
        assert result[center] == 1
        assert sum(result) == 1

    def test_constant_shift_concentrates_in_one_bin(self) -> None:
        # b is a copy of a shifted by delta = 0.03. With bin_width = 0.01 and max_lag = 0.05
        # (n = 5, 11 bins, center index 5), only the three same-index pairs have lag 0.03;
        # the cross-index lags (0.03 +/- 0.1, ...) fall outside the +/- 0.055 grid.
        # Lag 0.03 -> floor(0.03/0.01 + 0.5) + 5 = floor(3.5) + 5 = 8.
        a = [0.0, 0.1, 0.2]
        b = [0.03, 0.13, 0.23]
        result = cross_correlogram(a, b, bin_width=0.01, max_lag=0.05)
        expected = [0] * 11
        expected[8] = 3
        assert result == expected

    def test_symmetry_reversed(self) -> None:
        # cross_correlogram(a, b) reversed equals cross_correlogram(b, a): the sign
        # convention is target - reference, so swapping the arguments negates every lag.
        a = [0.0, 0.013, 0.041, 0.072]
        b = [0.004, 0.020, 0.055]
        ab = cross_correlogram(a, b, bin_width=0.005, max_lag=0.05)
        ba = cross_correlogram(b, a, bin_width=0.005, max_lag=0.05)
        assert ab == list(reversed(ba))

    def test_empty_train_all_zeros(self) -> None:
        result = cross_correlogram([], [1.0, 2.0], bin_width=0.5, max_lag=2.0)
        assert result == [0] * len(result)
        assert sum(result) == 0

    def test_non_positive_bin_width_raises(self) -> None:
        with pytest.raises(ValueError, match="bin_width"):
            cross_correlogram([0.0], [0.0], bin_width=0.0, max_lag=0.05)

    def test_non_positive_max_lag_raises(self) -> None:
        with pytest.raises(ValueError, match="max_lag"):
            cross_correlogram([0.0], [0.0], bin_width=0.01, max_lag=0.0)

    def test_max_lag_below_bin_width_raises(self) -> None:
        with pytest.raises(ValueError, match="max_lag"):
            cross_correlogram([0.0], [0.0], bin_width=0.05, max_lag=0.01)


class TestAutocorrelogram:
    def test_regular_train_peaks_at_isi_multiples(self) -> None:
        # Regular train at ISI = 0.01, five spikes. bin_width = 0.01, max_lag = 0.03 =>
        # n = 3, 7 bins, center index 3 (lag 0). Ordered distinct pairs have lags that are
        # integer multiples of 0.01, with index = (j - i) + 3:
        #   j - i = +/-1 -> 4 pairs each (bins 4 and 2),
        #   j - i = +/-2 -> 3 pairs each (bins 5 and 1),
        #   j - i = +/-3 -> 2 pairs each (bins 6 and 0),
        #   j - i = +/-4 -> outside the grid, dropped.
        # The central bin (lag 0) is 0 because i == i self-pairs are excluded and no two
        # distinct spikes lie within half a bin of lag 0.
        spikes = [0.0, 0.01, 0.02, 0.03, 0.04]
        assert autocorrelogram(spikes, bin_width=0.01, max_lag=0.03) == [2, 3, 4, 0, 4, 3, 2]

    def test_central_bin_excludes_self_pairs(self) -> None:
        # Two distinct spikes 0.001 apart, smaller than half a 0.01 bin, both land at lag 0;
        # but the self-pairs are excluded, so the central bin counts the two ordered pairs
        # of distinct spikes (0->1 and 1->0), not the two self coincidences.
        result = autocorrelogram([0.0, 0.001], bin_width=0.01, max_lag=0.05)
        center = len(result) // 2
        assert result[center] == 2
        assert sum(result) == 2

    def test_is_symmetric(self) -> None:
        # The ACG is symmetric because every ordered pair (i, j) has a mirror (j, i) of the
        # opposite lag.
        spikes = [0.0, 0.007, 0.018, 0.040, 0.041, 0.090]
        acg = autocorrelogram(spikes, bin_width=0.005, max_lag=0.05)
        assert acg == list(reversed(acg))

    def test_fewer_than_two_spikes_all_zeros(self) -> None:
        assert autocorrelogram([], bin_width=0.01, max_lag=0.05) == [0] * 11
        assert autocorrelogram([1.0], bin_width=0.01, max_lag=0.05) == [0] * 11

    def test_matches_cross_correlogram_minus_self_pairs(self) -> None:
        # The ACG equals the CCG of the train with itself, minus the N self-pairs that all
        # land in the central bin.
        spikes = [0.0, 0.012, 0.033, 0.071]
        acg = autocorrelogram(spikes, bin_width=0.005, max_lag=0.05)
        ccg = cross_correlogram(spikes, spikes, bin_width=0.005, max_lag=0.05)
        center = len(ccg) // 2
        ccg[center] -= len(spikes)
        assert acg == ccg

    def test_non_positive_bin_width_raises(self) -> None:
        with pytest.raises(ValueError, match="bin_width"):
            autocorrelogram([0.0, 1.0], bin_width=0.0, max_lag=0.05)

    def test_max_lag_below_bin_width_raises(self) -> None:
        with pytest.raises(ValueError, match="max_lag"):
            autocorrelogram([0.0, 1.0], bin_width=0.05, max_lag=0.01)


class TestPoissonAsymptotics:
    def test_cv_and_lv_approach_one(self) -> None:
        spikes = poisson_train(rate=50.0, duration=2000.0, seed=7)
        assert len(spikes) > 50_000
        assert cv_isi(spikes) == pytest.approx(1.0, abs=0.05)
        assert lv(spikes) == pytest.approx(1.0, abs=0.05)

    def test_fano_factor_approaches_one(self) -> None:
        spikes = poisson_train(rate=50.0, duration=2000.0, seed=11)
        assert fano_factor(spikes, duration=2000.0, bin_width=1.0) == pytest.approx(1.0, abs=0.1)
