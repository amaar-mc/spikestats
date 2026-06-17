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

__all__ = [
    "cv2",
    "cv_isi",
    "fano_factor",
    "firing_rate",
    "inter_spike_intervals",
    "lv",
    "lvr",
    "spike_counts",
]
__version__ = "0.1.0"
