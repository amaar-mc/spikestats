"""Spike-train statistics in pure Python with zero dependencies."""

from .metrics import (
    cv2,
    cv_isi,
    fano_factor,
    firing_rate,
    inter_spike_intervals,
    lv,
    lvr,
    spike_counts,
)
from .time_resolved import (
    binned_spike_counts,
    psth,
    time_resolved_fano,
    time_resolved_rate,
)

__all__ = [
    "binned_spike_counts",
    "cv2",
    "cv_isi",
    "fano_factor",
    "firing_rate",
    "inter_spike_intervals",
    "lv",
    "lvr",
    "psth",
    "spike_counts",
    "time_resolved_fano",
    "time_resolved_rate",
]
__version__ = "0.2.0"
