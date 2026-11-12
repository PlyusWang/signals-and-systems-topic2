"""Sample-grid time mapping for display and FFT comparison of two WAV files."""
from __future__ import annotations

import numpy as np


def aligned_audio_segment(samples: np.ndarray, start: int, end: int, *,
                          offset_samples: int = 0) -> tuple[np.ndarray, int]:
    """Read on the main file's grid; a positive offset delays the comparison.

    The returned segment always spans [start, end) on the main timeline. Missing
    samples outside the comparison file are zero. The second result counts real
    samples, so callers can distinguish silence from no overlapping audio.
    """
    if start < 0 or end < start:
        raise ValueError("main timeline sample bounds must be nonnegative and ordered")
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    result = np.zeros(end - start, dtype=np.float64)
    source_start = max(0, start - offset_samples)
    source_end = min(x.size, end - offset_samples)
    count = max(0, source_end - source_start)
    if count:
        destination_start = source_start + offset_samples - start
        result[destination_start:destination_start + count] = x[source_start:source_end]
    return result, count
