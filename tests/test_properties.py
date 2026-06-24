import pytest
from hypothesis import given
from hypothesis import strategies as st

from spikestats import (
    cv2,
    cv_isi,
    firing_rate,
    inter_spike_intervals,
    lv,
    lvr,
    spike_time_tiling_coefficient,
)

# Strictly increasing spike trains, built by accumulating positive gaps. This guarantees
# positive intervals, so no adjacent pair sums to zero and the variability metrics are
# always defined.
gaps = st.lists(
    st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=200,
)


def train(gap_list: list[float]) -> list[float]:
    times = [0.0]
    t = 0.0
    for g in gap_list:
        t += g
        times.append(t)
    return times


@given(gaps)
def test_intervals_are_positive_and_sum_to_span(gap_list: list[float]) -> None:
    spikes = train(gap_list)
    isis = inter_spike_intervals(spikes)
    assert len(isis) == len(spikes) - 1
    assert all(d > 0 for d in isis)
    assert sum(isis) == pytest.approx(spikes[-1] - spikes[0])


@given(gaps, st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False))
def test_firing_rate_is_count_over_duration(gap_list: list[float], duration: float) -> None:
    spikes = train(gap_list)
    assert firing_rate(spikes, duration=duration) == len(spikes) / duration


@given(gaps)
def test_variability_bounds(gap_list: list[float]) -> None:
    spikes = train(gap_list)
    assert cv_isi(spikes) >= 0.0
    assert 0.0 <= cv2(spikes) < 2.0
    assert 0.0 <= lv(spikes) < 3.0


@given(gaps)
def test_lvr_with_zero_refractory_equals_lv(gap_list: list[float]) -> None:
    spikes = train(gap_list)
    assert lvr(spikes, refractory=0.0) == pytest.approx(lv(spikes))


@given(gaps)
def test_regular_train_has_zero_variability(gap_list: list[float]) -> None:
    n = len(gap_list) + 1
    spikes = [i * 0.01 for i in range(n)]
    assert cv_isi(spikes) == pytest.approx(0.0, abs=1e-9)
    assert cv2(spikes) == pytest.approx(0.0, abs=1e-9)
    assert lv(spikes) == pytest.approx(0.0, abs=1e-9)
    assert lvr(spikes, refractory=0.003) == pytest.approx(0.0, abs=1e-9)


# Spike trains drawn from a fixed [0, 1] recording interval, so every spike lies inside it.
sttc_train = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=40,
)
sttc_dt = st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(sttc_train, sttc_train, sttc_dt)
def test_sttc_in_unit_range(a: list[float], b: list[float], dt: float) -> None:
    value = spike_time_tiling_coefficient(a, b, dt=dt, interval=(0.0, 1.0))
    assert -1.0 <= value <= 1.0


@given(sttc_train, sttc_train, sttc_dt)
def test_sttc_is_symmetric(a: list[float], b: list[float], dt: float) -> None:
    ab = spike_time_tiling_coefficient(a, b, dt=dt, interval=(0.0, 1.0))
    ba = spike_time_tiling_coefficient(b, a, dt=dt, interval=(0.0, 1.0))
    assert ab == pytest.approx(ba)


@given(
    # Spikes confined to [0, 0.6] with dt <= 0.3 means the covered union never reaches
    # past 0.9, so the tiling TA stays below 1. This keeps the test away from the
    # saturated case where TA == PA == 1 makes the denominator zero and, by the Cutts and
    # Eglen convention, drives the STTC to 0 rather than 1.
    st.lists(
        st.floats(min_value=0.0, max_value=0.6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=40,
    ),
    st.floats(min_value=1e-4, max_value=0.3, allow_nan=False, allow_infinity=False),
)
def test_sttc_identical_train_is_one(spikes: list[float], dt: float) -> None:
    assert spike_time_tiling_coefficient(
        spikes, spikes, dt=dt, interval=(0.0, 1.0)
    ) == pytest.approx(1.0)
