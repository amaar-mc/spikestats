from collections.abc import Sequence
from math import isfinite


def check_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number, received {value!r}")


def check_non_negative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number, received {value!r}")


def sorted_finite(spikes: Sequence[float]) -> list[float]:
    """Return the spike times as a sorted list of floats, rejecting non-finite values."""
    times = [float(t) for t in spikes]
    for t in times:
        if not isfinite(t):
            raise ValueError(f"spike times must all be finite, received {t!r}")
    times.sort()
    return times


def require_spikes(n_spikes: int, minimum: int, metric: str) -> None:
    """Raise when there are fewer than `minimum` spikes, naming the metric for the caller."""
    if n_spikes < minimum:
        raise ValueError(
            f"{metric} needs at least {minimum} spikes "
            f"({minimum - 1} intervals), received {n_spikes}"
        )


def check_interval(name: str, interval: tuple[float, float]) -> tuple[float, float]:
    """Validate a recording interval `(start, end)`: both finite and `start < end`.

    Returns the interval as a tuple of floats for downstream use.
    """
    start, end = interval
    start = float(start)
    end = float(end)
    if not isfinite(start) or not isfinite(end):
        raise ValueError(f"{name} bounds must both be finite, received {interval!r}")
    if start >= end:
        raise ValueError(
            f"{name} must be a (start, end) pair with start < end, received {interval!r}"
        )
    return start, end
