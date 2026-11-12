"""Audio axes with exact frequency positions and adaptive, readable labels."""
from __future__ import annotations

import math

import pyqtgraph as pg
from PySide6 import QtGui


def format_frequency(hz: float) -> str:
    if abs(hz) < 1e-12:
        return "0"
    return f"{hz / 1000:.9g}k" if abs(hz) >= 1000 else f"{hz:.9g}"


def _nice_ticks(low: float, high: float, count: int) -> tuple[float, list[float]]:
    """Choose a 1/2/5 decade step; return coordinates in the input units."""
    if not math.isfinite(low + high) or high <= low:
        return 1.0, []
    raw = (high - low) / max(1, count)
    power = 10.0 ** math.floor(math.log10(raw))
    step = next((factor * power for factor in (1, 2, 5, 10)
                 if factor * power >= raw * (1 - 1e-12)), 10 * power)
    first = math.ceil(low / step - 1e-10)
    last = math.floor(high / step + 1e-10)
    values = [float(f"{i * step:.12g}") for i in range(first, last + 1)]
    return step, values


class FrequencyAxis(pg.AxisItem):
    def __init__(self, orientation="bottom", *, log_frequency=False):
        self.log_frequency = log_frequency
        self.maximum_frequency = math.inf
        super().__init__(orientation=orientation)
        self.enableAutoSIPrefix(False)
        self.setLabel("频率", units="Hz")

    def set_frequency_scale(self, logarithmic: bool) -> None:
        self.log_frequency = logarithmic
        # Use tickValues at every range/size change instead of a static tick list.
        self.setTicks(None)
        self.picture = None
        self.update()

    def _minimum_distance(self, labels: list[str]) -> float:
        metrics = QtGui.QFontMetricsF(self.style["tickFont"] or QtGui.QFont("Segoe UI", 10))
        if self.orientation in ("left", "right"):
            return max(32.0, metrics.height() + 14)
        return max(52.0, max((metrics.horizontalAdvance(label) for label in labels), default=30) + 22)

    def tickValues(self, minVal, maxVal, size):
        low, high = sorted((float(minVal), float(maxVal)))
        if high <= low or size <= 0 or not math.isfinite(low + high):
            return [(1.0, [])]
        if not self.log_frequency:
            maximum = min(high, self.maximum_frequency)
            minimum = max(0.0, low)
            distance = self._minimum_distance([format_frequency(minimum), format_frequency(maximum)])
            # Clipping to Nyquist must not change the actual pixels per Hz.
            step, ticks = _nice_ticks(low, high, int(size / distance))
            ticks = [value for value in ticks if minimum <= value <= maximum]
            return [(step, ticks)]
        view_low, view_high = low, high
        low = max(-12.0, low)
        high = min(12.0, high, math.log10(self.maximum_frequency)
                   if math.isfinite(self.maximum_frequency) else 12.0)
        if high <= low:
            return [(1.0, [])]
        min_hz, max_hz = 10 ** low, 10 ** high
        if high - low < math.log10(1.8):
            distance = self._minimum_distance([format_frequency(min_hz), format_frequency(max_hz)])
            visible_size = size * (high - low) / (view_high - view_low)
            _, frequencies = _nice_ticks(min_hz, max_hz, int(visible_size / distance))
            candidates = [(0, hz) for hz in frequencies if hz > 0]
        else:
            candidates = [(priority, factor * 10.0 ** decade)
                          for decade in range(math.floor(low), math.ceil(high) + 1)
                          for factor, priority in ((1, 0), (2, 1), (5, 2))
                          if min_hz * (1 - 1e-10) <= factor * 10.0 ** decade <= max_hz * (1 + 1e-10)]
        # Reserve the space occupied by each actual label, in logical pixels.
        # Sorting by priority retains decade labels first on narrow plots.
        accepted = []
        for priority, hz in sorted(candidates):
            position = (math.log10(hz) - view_low) / (view_high - view_low) * size
            width = self._minimum_distance([format_frequency(hz)])
            if all(abs(position - other_pos) >= (width + other_width) / 2
                   for other_pos, other_width, _ in accepted):
                accepted.append((position, width, math.log10(hz)))
        return [(1.0, [value for _, _, value in sorted(accepted)])]

    def tickStrings(self, values, scale, spacing):
        return [format_frequency(10 ** value if self.log_frequency else value) for value in values]


class TimeAxis(pg.AxisItem):
    """Data remains in seconds; labels adapt to local duration and absolute time."""
    def __init__(self, orientation="bottom", *, label="时间"):
        self.time_label = label
        self.display_multiplier = 1.0
        self.display_mode = "seconds"
        self.clock_decimals = 3
        super().__init__(orientation=orientation)
        self.enableAutoSIPrefix(False)
        self.setLabel(label, units="s")

    def setRange(self, mn, mx):
        super().setRange(mn, mx)
        span, position = abs(mx - mn), max(abs(mn), abs(mx))
        mode = "clock" if position >= 60 else (
            "milliseconds" if 0 < span < 1.0 and position < 10 else "seconds")
        decimals = 6 if 0 < span < .01 else (3 if span < 10 else 0)
        if (mode, decimals) != (self.display_mode, self.clock_decimals):
            self.display_mode, self.clock_decimals = mode, decimals
            self.display_multiplier = 1000.0 if mode == "milliseconds" else 1.0
            self.setLabel(self.time_label + (" (分:秒)" if mode == "clock" else ""),
                          units="" if mode == "clock" else ("ms" if mode == "milliseconds" else "s"))

    def tickValues(self, minVal, maxVal, size):
        low, high = sorted((minVal, maxVal))
        metrics = QtGui.QFontMetricsF(self.style["tickFont"] or QtGui.QFont("Segoe UI", 10))
        labels = self.tickStrings([low, high], 1, 0)
        distance = max(68.0, max(metrics.horizontalAdvance(label) for label in labels) + 22)
        return [_nice_ticks(low, high, int(size / distance))]

    def tickStrings(self, values, scale, spacing):
        if self.display_mode == "clock":
            labels = []
            factor = 10 ** self.clock_decimals
            for value in values:
                total = round(abs(value) * factor)
                minutes, remainder = divmod(total, 60 * factor)
                seconds = remainder / factor
                sign = "−" if value < 0 and total else ""
                width = 3 + self.clock_decimals if self.clock_decimals else 2
                labels.append(f"{sign}{minutes:02d}:{seconds:0{width}.{self.clock_decimals}f}")
            return labels
        return [f"{value * self.display_multiplier:.9g}" if abs(value) > 1e-12 else "0"
                for value in values]
