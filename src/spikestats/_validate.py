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
