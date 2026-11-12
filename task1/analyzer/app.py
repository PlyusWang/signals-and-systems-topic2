from __future__ import annotations

import os
import sys
import tempfile
import argparse
import threading
from html import escape
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import soundfile as sf
from scipy import signal
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtSvg import QSvgGenerator
from task1.analyzer.axes import FrequencyAxis, TimeAxis
from shared.audio_comparison import aligned_audio_segment

from shared.signal_core import (
    AnalysisCancelled, correlation, cwt_view, estimate_delay, one_sided_spectrum,
    read_audio, stft_parameters, stft_view, write_audio,
)


APP_QSS = """
QWidget { background: #171b1f; color: #d7dce0; font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; font-size: 10pt; }
QMainWindow { background: #171b1f; }
QToolBar { background: #20262b; border: 0; spacing: 7px; padding: 6px 10px; }
QToolButton, QPushButton { background: #293137; color: #e6eaed; border: 1px solid #374149; border-radius: 4px; padding: 6px 10px; }
QToolButton#IconToolButton { padding: 3px; }
QToolButton:hover, QPushButton:hover { background: #354149; border-color: #4b5962; }
QToolButton:pressed, QPushButton:pressed { background: #174c46; }
QToolButton:disabled, QPushButton:disabled { color: #737d84; background: #23292e; }
QLabel#AppTitle { color: #f1f4f5; font-size: 12pt; font-weight: 600; }
QLabel#Muted { color: #8e9aa1; }
QLabel#Readout { color: #66d1b1; font-family: "Consolas"; font-size: 9pt; }
QFrame#ControlStrip { background: #1d2327; border: 1px solid #30383e; border-radius: 3px; }
QFrame#ControlStrip QLabel { color: #9eabb2; }
QFrame#ControlStrip QComboBox { min-width: 78px; }
QFrame#Divider { background: #30383e; max-height: 1px; }
QTabWidget::pane { border: 1px solid #30383e; top: -1px; background: #171b1f; }
QTabBar::tab { background: #20262b; color: #9eabb2; padding: 9px 18px; border: 0; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #e6eaed; border-bottom: 2px solid #49c6a5; background: #1b2125; }
QTabBar::tab:hover:!selected { color: #dce3e6; background: #272e33; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #22292e; color: #dce3e6; border: 1px solid #39434a; border-radius: 3px; padding: 4px 7px; }
QComboBox QAbstractItemView { background: #22292e; selection-background-color: #28695c; }
QStatusBar { background: #20262b; color: #9ba7ad; border-top: 1px solid #30383e; }
QScrollBar:vertical { background: #1c2226; width: 10px; }
QScrollBar::handle:vertical { background: #404a51; min-height: 20px; border-radius: 4px; }
QProgressBar { border: 1px solid #46505a; border-radius: 3px; background: #20262b; text-align: center; font-size: 9pt; }
QProgressBar::chunk { background: #395c6d; }
"""


def _envelope(samples: np.ndarray, maximum_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size <= maximum_points * 2:
        return np.arange(x.size), x
    bucket = int(np.ceil(x.size / maximum_points))
    count = x.size // bucket
    blocks = x[:count * bucket].reshape(count, bucket)
    low, high = blocks.min(axis=1), blocks.max(axis=1)
    if count * bucket < x.size:
        tail = x[count * bucket:]
        low, high = np.r_[low, tail.min()], np.r_[high, tail.max()]
        count += 1
    indices = np.arange(count) * bucket
    xs = np.repeat(indices, 2)
    ys = np.stack((low, high), axis=1).reshape(-1)
    return xs, ys


def _relative_db(values: np.ndarray, *, floor_db: float = -120.0,
                 reference_peak: float | None = None) -> np.ndarray:
    magnitude = np.nan_to_num(np.abs(np.asarray(values, dtype=np.float64)),
                              nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(magnitude, initial=0.0)) if reference_peak is None else reference_peak
    if peak == 0.0:
        return np.full(magnitude.shape, floor_db, dtype=np.float64)
    normalized = magnitude / peak
    return 20.0 * np.log10(np.maximum(normalized, 10 ** (floor_db / 20.0)))


def _finite_levels(values: np.ndarray) -> tuple[float, float]:
    data = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=-120.0,
                         neginf=-120.0, posinf=0.0)
    if not data.size:
        return -120.0, 0.0
    low, high = np.percentile(data, [5, 99.5])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


class ElidedLabel(QtWidgets.QLabel):
    """Keep long filenames readable without cutting a glyph at the edge."""

    def __init__(self, text: str):
        super().__init__()
        self._full_text = text
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full_text = text
        self._refresh_text()

    def _refresh_text(self) -> None:
        super().setText(self.fontMetrics().elidedText(
            self._full_text, QtCore.Qt.TextElideMode.ElideMiddle,
            max(0, self.contentsRect().width())))

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self.fontMetrics().horizontalAdvance(self._full_text),
                            super().sizeHint().height())

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh_text()


class AnalysisSignals(QtCore.QObject):
    ready = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)
    cancelled = QtCore.Signal(int)
    progress = QtCore.Signal(int, int, int, str)


class WaveformViewBox(pg.ViewBox):
    selectionDragged = QtCore.Signal(float, float, bool)

    def __init__(self):
        super().__init__()
        self.select_on_drag = True

    def mouseDragEvent(self, event, axis=None):
        select = self.select_on_drag and not (
            event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if event.button() == QtCore.Qt.MouseButton.LeftButton and select and axis is None:
            event.accept()
            start = self.mapToView(event.buttonDownPos()).x()
            end = self.mapToView(event.pos()).x()
            self.selectionDragged.emit(min(start, end), max(start, end), event.isFinish())
            return
        super().mouseDragEvent(event, axis=axis)


class AnalysisWorker(QtCore.QRunnable):
    def __init__(self, generation: int, samples: np.ndarray, sample_rate: int,
                 selection: tuple[int, int], reference: np.ndarray | None, fft_window: str,
                 full_spectrum: tuple[np.ndarray, np.ndarray] | None,
                 stft_window: int, stft_nfft: int, stft_overlap: float, cwt_voices: int,
                 full_cwt: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                 mode: str = "fft", compute_full_spectrum: bool = False,
                 compute_full_cwt: bool = False, result_key: tuple | None = None,
                 cancel_event: threading.Event | None = None,
                 full_correlation: dict | None = None,
                 comparison_samples: np.ndarray | None = None,
                 comparison_offset_samples: int = 0,
                 full_comparison_spectrum: tuple | None = None):
        super().__init__()
        self.generation = generation
        self.samples = samples
        self.sample_rate = sample_rate
        self.selection = selection
        self.reference = reference
        self.fft_window = fft_window
        self.full_spectrum = full_spectrum
        self.stft_window = stft_window
        self.stft_nfft = stft_nfft
        self.stft_overlap = stft_overlap
        self.cwt_voices = cwt_voices
        self.full_cwt = full_cwt
        self.full_correlation = full_correlation
        self.comparison_samples = comparison_samples
        self.comparison_offset_samples = comparison_offset_samples
        self.full_comparison_spectrum = full_comparison_spectrum
        self.mode = mode
        self.compute_full_spectrum = compute_full_spectrum
        self.compute_full_cwt = compute_full_cwt
        self.result_key = result_key
        self.cancel_event = cancel_event or threading.Event()
        self.signals = AnalysisSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            if self.cancel_event.is_set():
                raise AnalysisCancelled("Analysis superseded")
            start, end = self.selection
            local = self.samples[start:end]
            if not local.size:
                raise ValueError("时间选区为空")
            payload: dict[str, object] = {"mode": self.mode, "result_key": self.result_key}
            if self.mode == "fft":
                nfft = max(256, 1 << (local.size - 1).bit_length())
                spectrum = one_sided_spectrum(local, self.sample_rate, n_fft=nfft,
                                              window=self.fft_window)
                full_spectrum = self.full_spectrum
                if full_spectrum is None and self.compute_full_spectrum:
                    full_nfft = 1 << (self.samples.size - 1).bit_length()
                    full = one_sided_spectrum(self.samples, self.sample_rate, n_fft=full_nfft,
                                              window=self.fft_window)
                    full_spectrum = (full.frequency_hz, full.dbfs)
                payload["full_spectrum_cache_key"] = (
                    (id(self.samples), self.fft_window) if self.compute_full_spectrum else None)
                payload.update({
                    "spectrum": (spectrum.frequency_hz, spectrum.dbfs, spectrum.n_fft,
                                 spectrum.bin_spacing_hz, spectrum.observation_duration_s,
                                 spectrum.sample_count, spectrum.window_main_lobe_width_hz),
                    "full_spectrum": full_spectrum,
                })
                if self.comparison_samples is not None:
                    aligned, count = aligned_audio_segment(
                        self.comparison_samples, start, end,
                        offset_samples=self.comparison_offset_samples)
                    compared = one_sided_spectrum(aligned, self.sample_rate,
                                                   n_fft=nfft, window=self.fft_window)
                    payload["comparison_spectrum"] = (compared.frequency_hz, compared.dbfs, count)
                    full_compared = self.full_comparison_spectrum
                    if full_compared is None:
                        if self.cancel_event.is_set():
                            raise AnalysisCancelled("Analysis superseded")
                        aligned, count = aligned_audio_segment(
                            self.comparison_samples, 0, self.samples.size,
                            offset_samples=self.comparison_offset_samples)
                        full = one_sided_spectrum(aligned, self.sample_rate,
                                                  n_fft=1 << (self.samples.size - 1).bit_length(),
                                                  window=self.fft_window)
                        full_compared = (full.frequency_hz, full.dbfs, count)
                    payload["full_comparison_spectrum"] = full_compared
                    payload["full_comparison_cache_key"] = (
                        id(self.samples), id(self.comparison_samples),
                        self.comparison_offset_samples, self.fft_window)
            elif self.mode == "stft":
                parameters = stft_parameters(
                    local.size, window_length=min(self.stft_window, max(32, local.size // 4)),
                    overlap=self.stft_overlap, n_fft=self.stft_nfft)
                f_stft, t_stft, z_stft = stft_view(
                    local, self.sample_rate,
                    window_length=parameters["window_length"],
                    overlap=self.stft_overlap,
                    n_fft=parameters["fft_length"])
                payload["stft"] = (f_stft, t_stft + start / self.sample_rate, z_stft)
                payload["stft_parameters"] = dict(parameters, requested_window=self.stft_window,
                                                   requested_fft=self.stft_nfft)
            elif self.mode == "cwt":
                f_cwt, t_cwt, cwt_magnitude = cwt_view(
                    local, self.sample_rate, voices_per_octave=self.cwt_voices,
                    maximum_points=12000, cancel_event=self.cancel_event,
                    progress_callback=lambda done, total: self.signals.progress.emit(
                        self.generation, done, total, "选区"))
                full_cwt = self.full_cwt
                if full_cwt is None and self.compute_full_cwt:
                    full_cwt = cwt_view(self.samples, self.sample_rate,
                                        voices_per_octave=self.cwt_voices,
                                        maximum_points=12000,
                                        cancel_event=self.cancel_event,
                                        progress_callback=lambda done, total: self.signals.progress.emit(
                                            self.generation, done, total, "完整信号"))
                payload["full_cwt_cache_key"] = (
                    (id(self.samples), self.cwt_voices) if self.compute_full_cwt else None)
                payload.update({
                    "cwt": (f_cwt, t_cwt + start / self.sample_rate, cwt_magnitude),
                    "full_cwt": full_cwt,
                })
            elif self.mode == "correlation":
                auto = correlation(local, demean=True, normalize=True)
                cross = None
                if self.reference is not None and self.reference.size <= local.size:
                    delay, peak, lags, scores = estimate_delay(local, self.reference)
                    cross = (lags, scores, delay, peak)
                full = self.full_correlation
                if full is None:
                    full_auto = correlation(self.samples, demean=True, normalize=True)
                    full_cross = None
                    if self.reference is not None and self.reference.size <= self.samples.size:
                        delay, peak, lags, scores = estimate_delay(self.samples, self.reference)
                        full_cross = (lags, scores, delay, peak)
                    full = {"autocorrelation": (full_auto.lags, full_auto.values),
                            "crosscorrelation": full_cross}
                payload.update({
                    "autocorrelation": (auto.lags, auto.values),
                    "crosscorrelation": cross,
                    "full_correlation": full,
                    "full_correlation_cache_key": (id(self.samples), id(self.reference)),
                    "selection_start": start,
                })
            else:
                raise ValueError(f"unknown analysis view: {self.mode}")
            if self.cancel_event.is_set():
                raise AnalysisCancelled("Analysis superseded")
            payload["selection_samples"] = local.size
            self.signals.ready.emit(self.generation, payload)
        except AnalysisCancelled:
            self.signals.cancelled.emit(self.generation)
        except Exception as exc:
            self.signals.failed.emit(self.generation, str(exc))


class SignalAnalyzer(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SignalScope | 信号分析器")
        self.resize(1420, 900)
        self.setMinimumSize(900, 640)
        self.setStyleSheet(APP_QSS)
        self.sample_rate = 0
        self.samples = np.empty(0, dtype=np.float64)
        self.comparison_samples: np.ndarray | None = None
        self.comparison_path: Path | None = None
        self.comparison_rate = 0
        self.comparison_offset_s = 0.0
        self.reference: np.ndarray | None = None
        self.reference_rate: int | None = None
        self.audio_path: Path | None = None
        self._closing = False
        self.generation = 0
        self.full_spectrum_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        self.comparison_spectrum_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self.full_cwt_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.full_correlation_cache: dict[tuple[int, int], dict] = {}
        self.selection_cache: OrderedDict[tuple, dict[str, object]] = OrderedDict()
        self.pending_full_spectrum: set[tuple[int, str]] = set()
        self.pending_full_cwt: set[tuple[int, int]] = set()
        self.pending_analysis: dict[int, tuple[tuple[int, str] | None, tuple[int, int] | None]] = {}
        self.worker_cancel_events: dict[int, threading.Event] = {}
        self._updating_view_from_overview = False
        self.analysis_selection: tuple[float, float] | None = None
        self.navigation_analysis_timer = QtCore.QTimer(self)
        self.navigation_analysis_timer.setSingleShot(True)
        self.navigation_analysis_timer.setInterval(140)
        self.navigation_analysis_timer.timeout.connect(self._request_analysis)
        self.player = QMediaPlayer(self)
        self.playback_offset_s = 0.0
        self.playback_end_s = 0.0
        self._tracking_playback = False
        self.player.positionChanged.connect(self._playback_position_changed)
        self.player.mediaStatusChanged.connect(self._playback_media_status_changed)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        self.thread_pool = QtCore.QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self.interactive_thread_pool = QtCore.QThreadPool(self)
        self.interactive_thread_pool.setMaxThreadCount(1)
        self.analysis_debounce = QtCore.QTimer(self)
        self.analysis_debounce.setSingleShot(True)
        self.analysis_debounce.setInterval(120)
        self.analysis_debounce.timeout.connect(self._launch_analysis)
        self._build_ui()
        self._set_empty_state()

    def _build_ui(self) -> None:
        toolbar = QtWidgets.QToolBar("主工具栏", self)
        toolbar.setMovable(False)
        toolbar.setIconSize(QtCore.QSize(17, 17))
        self.addToolBar(toolbar)

        self.open_button = self._tool_button("打开 WAV", self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton), self.open_audio)
        toolbar.addWidget(self.open_button)
        self.play_button = self._tool_button("播放", self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay), self.play_audio)
        toolbar.addWidget(self.play_button)
        self.stop_button = self._tool_button("停止", self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop), self.stop_playback)
        toolbar.addWidget(self.stop_button)
        self.play_selection_button = self._tool_button("播放选区", self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaSeekForward), self.play_selection)
        toolbar.addWidget(self.play_selection_button)
        toolbar.addSeparator()
        self.reference_button = self._tool_button("载入参考 WAV", self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView), self.load_reference)
        toolbar.addWidget(self.reference_button)
        self.export_button = self._tool_button("导出选区", self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton), self.export_selection)
        toolbar.addWidget(self.export_button)
        self.export_plot_button = self._tool_button("导出图像", self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton), self.export_current_plot)
        toolbar.addWidget(self.export_plot_button)
        toolbar.addSeparator()
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.addItems(["汉宁窗", "海明窗", "布莱克曼窗", "矩形窗"])
        self.window_combo.currentTextChanged.connect(self._request_analysis)
        self.frequency_scale = QtWidgets.QComboBox()
        self.frequency_scale.addItems(["对数", "线性"])
        self.frequency_scale.currentTextChanged.connect(self._set_frequency_scale)
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItem("混合声道", "mix")
        self.channel_combo.addItem("左声道", "left")
        self.channel_combo.addItem("右声道", "right")
        self.channel_combo.setToolTip("多声道 WAV 的分析与播放声道")
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        toolbar.addWidget(self.channel_combo)
        self.audio_switch_combo = QtWidgets.QComboBox()
        self.audio_switch_combo.setMinimumWidth(150)
        self.audio_switch_combo.setMaximumWidth(250)
        self.audio_switch_combo.setToolTip("快速切换当前文件所在目录中的 WAV")
        self.audio_switch_combo.currentIndexChanged.connect(self._switch_audio)
        toolbar.addWidget(self.audio_switch_combo)
        self.comparison_button = QtWidgets.QPushButton("选择对比 WAV")
        self.comparison_button.setToolTip("单独选择对比文件，不改变主信号、播放或导出")
        self.comparison_button.clicked.connect(self.open_comparison)
        toolbar.addWidget(self.comparison_button)
        self.overlay_check = QtWidgets.QCheckBox("音频对比")
        self.overlay_check.setToolTip("显示所选对比 WAV；切换主文件后默认关闭")
        self.overlay_check.toggled.connect(self._set_comparison_visible)
        toolbar.addWidget(self.overlay_check)
        toolbar.addSeparator()
        for button in (self.open_button, self.play_button, self.stop_button, self.play_selection_button,
                       self.reference_button, self.export_button, self.export_plot_button):
            button.setObjectName("IconToolButton")
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setFixedSize(34, 30)
            button.setToolTip(button.text())
        self.zoom_out_button = self._tool_button("缩小", self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowDown), lambda: self._zoom_selection(1.7))
        self.zoom_in_button = self._tool_button("放大", self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowUp), lambda: self._zoom_selection(1 / 1.7))
        self.fit_button = self._tool_button("全览", self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarMaxButton), self._fit_file)

        content = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(content)
        outer.setContentsMargins(14, 12, 14, 8)
        outer.setSpacing(10)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("信号分析")
        title.setObjectName("AppTitle")
        header.addWidget(title)
        self.metadata_label = ElidedLabel("打开 WAV 文件开始分析")
        self.metadata_label.setObjectName("Muted")
        self.metadata_label.setMinimumWidth(0)
        self.metadata_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.metadata_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                          QtWidgets.QSizePolicy.Policy.Preferred)
        header.addWidget(self.metadata_label)
        header.addStretch(1)
        header.addSpacing(14)
        header.addWidget(self._muted_label("视图"))
        for button in (self.zoom_out_button, self.zoom_in_button, self.fit_button):
            button.setObjectName("IconToolButton")
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setFixedSize(30, 28)
            header.addWidget(button)
        outer.addLayout(header)

        self.interaction_combo = QtWidgets.QComboBox()
        self.interaction_combo.addItems(["拖动选择", "拖动平移"])
        self.interaction_combo.setToolTip("左键拖动选择；按住 Shift 拖动可临时平移；滚轮缩放")
        self.interaction_combo.currentIndexChanged.connect(self._interaction_changed)
        header.insertWidget(header.count() - 4, self.interaction_combo)
        self.visible_range_button = QtWidgets.QPushButton("清除选区")
        self.visible_range_button.setShortcut(QtGui.QKeySequence("Escape"))
        self.visible_range_button.setToolTip("Esc：清除蓝色分析选区，分析重新跟随浏览窗口")
        self.visible_range_button.clicked.connect(self._use_visible_range)
        header.insertWidget(header.count() - 4, self.visible_range_button)
        self.locate_selection_button = QtWidgets.QPushButton("定位选区")
        self.locate_selection_button.setToolTip("在局部波形中显示已建立的蓝色分析选区")
        self.locate_selection_button.clicked.connect(self._locate_selection)
        header.insertWidget(header.count() - 4, self.locate_selection_button)
        info_row = QtWidgets.QHBoxLayout()
        self.technical_info_label = QtWidgets.QLabel("采样率、时长、声道：--")
        self.technical_info_label.setObjectName("Muted")
        self.technical_info_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                                QtWidgets.QSizePolicy.Policy.Preferred)
        self.technical_info_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        info_row.addWidget(self.technical_info_label)
        info_row.addStretch(1)
        self.overview_toggle = QtWidgets.QCheckBox("全局概览")
        self.overview_toggle.setChecked(True)
        self.overview_toggle.setToolTip("显示或收起完整音频概览；浏览范围和分析选区保持不变")
        self._overview_manual = False
        self.overview_toggle.toggled.connect(self._overview_visibility_changed)
        info_row.addWidget(self.overview_toggle)
        outer.addLayout(info_row)
        waveform_view = WaveformViewBox()
        self.waveform = pg.PlotWidget(title="局部波形", viewBox=waveform_view,
                                      axisItems={"bottom": TimeAxis()})
        self._style_plot(self.waveform)
        self.waveform.setMinimumHeight(135)
        self.waveform.setLabel("left", "幅度")
        self.waveform.getAxis("left").setWidth(55)
        self.waveform.setLabel("bottom", "时间", units="s")
        self.waveform.getAxis("left").enableAutoSIPrefix(False)
        self.waveform.showGrid(x=True, y=True, alpha=0.16)
        self.waveform.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.waveform.getViewBox().setMouseEnabled(x=True, y=False)
        self.waveform.getViewBox().setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.waveform.scene().sigMouseMoved.connect(self._waveform_mouse_moved)
        waveform_view.selectionDragged.connect(self._set_analysis_selection)
        self.waveform.getViewBox().sigXRangeChanged.connect(self._waveform_view_changed)
        self.waveform_curve = self.waveform.plot([], [], pen=pg.mkPen("#48c7a5", width=1.1),
                                                  name="当前文件")
        self.comparison_curve = self.waveform.plot([], [], pen=pg.mkPen("#df95629e", width=1.0),
                                                    name="对比信号")
        self.comparison_curve.setVisible(False)
        self.waveform_curve.setClipToView(True)
        self.waveform_curve.setDownsampling(auto=True, method="peak")
        self.comparison_curve.setClipToView(True)
        self.selection_region = pg.LinearRegionItem(
            [0, 0], brush=pg.mkBrush(114, 154, 246, 38),
            pen=pg.mkPen("#7da6f7", width=1), movable=False)
        self.selection_region.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        for line in self.selection_region.lines:
            line.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.selection_region.setZValue(-5)
        self.waveform.addItem(self.selection_region)
        self.selection_region.hide()
        self.waveform.setToolTip("蓝色：实际分析选区；左键重新拖选，Shift + 拖动平移，Esc 清除选区。暖色竖线：播放位置。")
        comparison_view = WaveformViewBox()
        self.comparison_waveform = pg.PlotWidget(
            title="对比信号", viewBox=comparison_view, axisItems={"bottom": TimeAxis()})
        self._style_plot(self.comparison_waveform)
        self.comparison_waveform.setLabel("left", "幅度")
        self.comparison_waveform.getAxis("left").setWidth(55)
        self.comparison_waveform.getAxis("left").enableAutoSIPrefix(False)
        self.comparison_waveform.setMinimumHeight(110)
        comparison_view.setMouseMode(pg.ViewBox.PanMode)
        comparison_view.setMouseEnabled(x=True, y=False)
        comparison_view.selectionDragged.connect(self._set_analysis_selection)
        comparison_view.sigRangeChangedManually.connect(self._comparison_view_changed)
        self.comparison_waveform.setXLink(self.waveform)
        self.comparison_track_curve = self.comparison_waveform.plot(
            [], [], pen=pg.mkPen("#df9562", width=1.1))
        self.comparison_track_curve.setClipToView(True)
        self.comparison_selection_region = pg.LinearRegionItem(
            [0, 0], brush=pg.mkBrush(114, 154, 246, 30),
            pen=pg.mkPen("#7da6f7", width=1), movable=False)
        self.comparison_selection_region.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        for line in self.comparison_selection_region.lines:
            line.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.comparison_waveform.addItem(self.comparison_selection_region)
        self.comparison_selection_region.hide()
        self.comparison_waveform.hide()
        overview_row = QtWidgets.QHBoxLayout()
        overview_label = QtWidgets.QLabel("全局概览")
        overview_label.setObjectName("Muted")
        overview_label.setFixedWidth(94)
        overview_row.addWidget(overview_label)
        self.overview = self._plot_widget("")
        self.overview.setMinimumHeight(100)
        self.overview.setMaximumHeight(190)
        self.overview.getViewBox().setMouseEnabled(x=False, y=False)
        self.overview.getViewBox().setMenuEnabled(False)
        self.overview.setLabel("bottom", "")
        self.overview.hideAxis("left")
        self.overview_curve = self.overview.plot([], [], pen=pg.mkPen("#4b8174", width=1))
        self.overview_region = pg.LinearRegionItem(
            [0, 1], brush=pg.mkBrush(74, 197, 165, 18),
            pen=pg.mkPen("#74e0c1", width=2), hoverBrush=pg.mkBrush(74, 197, 165, 78),
            hoverPen=pg.mkPen("#a1f1d9", width=3), movable=True)
        self.overview.addItem(self.overview_region)
        self.overview.setToolTip("青色边框：局部波形的浏览窗口；拖动窗口或边界以移动、缩放。")
        self.overview_region.sigRegionChanged.connect(self._overview_changed)
        self.overview_region.sigRegionChangeFinished.connect(self._overview_change_finished)
        self.playback_cursor = self._playback_line(self.waveform)
        self.overview_playback_cursor = self._playback_line(self.overview)
        self.comparison_playback_cursor = self._playback_line(self.comparison_waveform)
        overview_row.addWidget(self.overview, 1)
        overview_container = QtWidgets.QWidget()
        overview_container.setLayout(overview_row)
        self.overview_container = overview_container

        wave_section = QtWidgets.QWidget()
        wave_layout = QtWidgets.QVBoxLayout(wave_section)
        wave_layout.setContentsMargins(0, 0, 0, 0)
        wave_layout.setSpacing(2)
        self.comparison_controls = QtWidgets.QWidget()
        comparison_layout = QtWidgets.QHBoxLayout(self.comparison_controls)
        comparison_layout.setContentsMargins(4, 0, 4, 0)
        self.current_signal_label = QtWidgets.QLabel()
        self.comparison_label = QtWidgets.QLabel()
        for label in (self.current_signal_label, self.comparison_label):
            label.setMinimumWidth(0)
        comparison_layout.addWidget(self.current_signal_label, 1)
        comparison_layout.addWidget(self.comparison_label, 1)
        self.comparison_mode = QtWidgets.QComboBox()
        self.comparison_mode.addItems(["叠加波形", "上下分轨"])
        self.comparison_mode.currentIndexChanged.connect(self._comparison_mode_changed)
        comparison_layout.addWidget(self.comparison_mode)
        comparison_layout.addWidget(self._muted_label("时间偏移"))
        self.comparison_offset = QtWidgets.QDoubleSpinBox()
        self.comparison_offset.setRange(-86400, 86400)
        self.comparison_offset.setDecimals(6)
        self.comparison_offset.setSingleStep(.001)
        self.comparison_offset.setSuffix(" s")
        self.comparison_offset.setToolTip("0：文件起点重合；正值：对比信号向右移动。按主信号采样点取整。")
        self.comparison_offset.valueChanged.connect(self._comparison_offset_changed)
        comparison_layout.addWidget(self.comparison_offset)
        self.comparison_controls.hide()
        wave_layout.addWidget(self.comparison_controls)
        wave_layout.addWidget(self.waveform, 1)
        wave_layout.addWidget(self.comparison_waveform, 1)
        wave_layout.addWidget(overview_container, 0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.fft_plot = pg.PlotWidget(title="FFT 频谱", axisItems={"bottom": FrequencyAxis(log_frequency=True)})
        self.fft_full_plot = pg.PlotWidget(title="完整信号 · FFT", axisItems={"bottom": FrequencyAxis(log_frequency=True)})
        for plot in (self.fft_full_plot, self.fft_plot):
            self._style_plot(plot)
            plot.setLabel("left", "幅度", units="dBFS")
            plot.setLabel("bottom", "频率", units="Hz")
            plot.showGrid(x=True, y=True, alpha=0.13)
            plot.setLogMode(x=False, y=False)
            plot.setYRange(-140, 0, padding=0)
            plot.scene().sigMouseMoved.connect(
                lambda position, source=plot: self._fft_mouse_moved(position, source))
        self.fft_full_plot.setXLink(self.fft_plot)
        self.fft_full_plot.setYLink(self.fft_plot)
        self.fft_selected_curve = self.fft_plot.plot([], [], pen=pg.mkPen("#49c7a5", width=1.5),
                                                       name="当前选区")
        self.fft_full_curve = self.fft_full_plot.plot([], [], pen=pg.mkPen("#49c7a5", width=1.5),
                                                  name="完整文件")
        self.fft_comparison_selected_curve = self.fft_plot.plot(
            [], [], pen=pg.mkPen("#df9562c8", width=1.3))
        self.fft_comparison_full_curve = self.fft_full_plot.plot(
            [], [], pen=pg.mkPen("#df9562c8", width=1.3))
        self.fft_comparison_selected_curve.hide()
        self.fft_comparison_full_curve.hide()
        self._fft_comparison_selected_raw = (np.empty(0), np.empty(0))
        self._fft_comparison_full_raw = (np.empty(0), np.empty(0))
        self._fft_selected_raw = (np.empty(0), np.empty(0))
        self._fft_full_raw = (np.empty(0), np.empty(0))
        self.fft_splitter = self._comparison_splitter(self.fft_full_plot, self.fft_plot)
        self.tabs.addTab(self.fft_splitter, "频谱 FFT")

        self.stft_plot = self._image_plot("短时傅里叶变换", "相对幅度 (dB)")
        self.tabs.addTab(self.stft_plot, "时频 STFT")
        cwt_widget = QtWidgets.QWidget()
        cwt_layout = QtWidgets.QVBoxLayout(cwt_widget)
        cwt_layout.setContentsMargins(0, 0, 0, 0)
        self.cwt_overview_plot = self._image_plot("CWT 全文件概览", "相对幅度 (dB)")
        self.cwt_plot = self._image_plot("CWT 当前选区", "相对幅度 (dB)")
        self.cwt_overview_region = pg.LinearRegionItem(
            [0, 1], brush=pg.mkBrush(114, 154, 246, 25),
            pen=pg.mkPen("#7da6f7", width=1), movable=False)
        self.cwt_overview_plot.addItem(self.cwt_overview_region)
        self.cwt_splitter = self._comparison_splitter(self.cwt_overview_plot, self.cwt_plot)
        self.cwt_overview_plot.setYLink(self.cwt_plot)
        cwt_layout.addWidget(self.cwt_splitter)
        self.tabs.addTab(cwt_widget, "小波 CWT")

        correlation_widget = QtWidgets.QWidget()
        correlation_layout = QtWidgets.QVBoxLayout(correlation_widget)
        correlation_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_full_plot = self._plot_widget("完整信号 · 自相关")
        self.auto_plot = self._plot_widget("当前选区 · 自相关")
        self.cross_full_plot = self._plot_widget("完整信号 · 互相关")
        self.cross_plot = self._plot_widget("当前选区 · 互相关")
        for plot in (self.auto_full_plot, self.auto_plot, self.cross_full_plot, self.cross_plot):
            plot.setLabel("left", "归一化相关值")
            plot.getAxis("bottom").time_label = "滞后"
            plot.setLabel("bottom", "滞后", units="s")
            plot.showGrid(x=True, y=True, alpha=0.18)
        self.auto_curve = self.auto_plot.plot([], [], pen=pg.mkPen("#49c7a5", width=1.4))
        self.auto_full_curve = self.auto_full_plot.plot([], [], pen=pg.mkPen("#49c7a5", width=1.4))
        self.cross_curve = self.cross_plot.plot([], [], pen=pg.mkPen("#49c7a5", width=1.4))
        self.cross_full_curve = self.cross_full_plot.plot([], [], pen=pg.mkPen("#49c7a5", width=1.4))
        for curve in (self.auto_curve, self.auto_full_curve, self.cross_curve, self.cross_full_curve,
                      self.fft_selected_curve, self.fft_full_curve):
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
        self.cross_peak_marker = pg.InfiniteLine(angle=90, movable=False,
                                                  pen=pg.mkPen("#f1bd77", width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.cross_plot.addItem(self.cross_peak_marker)
        self.cross_peak_marker.hide()
        self.cross_full_peak_marker = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#f1bd77", width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.cross_full_plot.addItem(self.cross_full_peak_marker)
        self.cross_full_peak_marker.hide()
        reference_controls = QtWidgets.QHBoxLayout()
        self.selection_reference_button = QtWidgets.QPushButton("将当前选区设为模板")
        self.selection_reference_button.clicked.connect(self.set_selection_reference)
        reference_controls.addWidget(self.selection_reference_button)
        self.reference_status = self._muted_label("模板：未设置")
        reference_controls.addWidget(self.reference_status, 1)
        correlation_layout.addLayout(reference_controls)
        self.correlation_tabs = QtWidgets.QTabWidget()
        self.correlation_tabs.setDocumentMode(True)
        self.auto_splitter = self._comparison_splitter(self.auto_full_plot, self.auto_plot)
        self.cross_splitter = self._comparison_splitter(self.cross_full_plot, self.cross_plot)
        self.correlation_tabs.addTab(self.auto_splitter, "自相关")
        self.correlation_tabs.addTab(self.cross_splitter, "互相关")
        correlation_layout.addWidget(self.correlation_tabs)
        self.tabs.addTab(correlation_widget, "相关分析")
        self.tabs.currentChanged.connect(self._analysis_tab_changed)
        self._plot_by_tab = {self.fft_splitter: self.fft_plot, self.stft_plot: self.stft_plot,
                             cwt_widget: self.cwt_plot, correlation_widget: self.auto_plot}
        self.analysis_controls = QtWidgets.QFrame()
        self.analysis_controls.setObjectName("ControlStrip")
        controls_layout = QtWidgets.QHBoxLayout(self.analysis_controls)
        controls_layout.setContentsMargins(10, 5, 10, 5)
        controls_layout.setSpacing(7)
        self.stft_control_group = QtWidgets.QWidget()
        stft_controls = QtWidgets.QHBoxLayout(self.stft_control_group)
        stft_controls.setContentsMargins(0, 0, 0, 0)
        stft_controls.setSpacing(7)
        stft_controls.addWidget(self._muted_label("窗长"))
        self.stft_window_combo = QtWidgets.QComboBox()
        self.stft_window_combo.addItems(["256", "512", "1024", "2048"])
        self.stft_window_combo.setCurrentText("1024")
        self.stft_window_combo.currentTextChanged.connect(self._request_analysis)
        stft_controls.addWidget(self.stft_window_combo)
        stft_controls.addWidget(self._muted_label("FFT 长度"))
        self.stft_fft_combo = QtWidgets.QComboBox()
        self.stft_fft_combo.addItems(["256", "512", "1024", "2048", "4096"])
        self.stft_fft_combo.setCurrentText("2048")
        self.stft_fft_combo.currentTextChanged.connect(self._request_analysis)
        stft_controls.addWidget(self.stft_fft_combo)
        stft_controls.addWidget(self._muted_label("重叠"))
        self.stft_overlap_combo = QtWidgets.QComboBox()
        self.stft_overlap_combo.addItems(["50%", "75%", "87.5%"])
        self.stft_overlap_combo.setCurrentText("87.5%")
        self.stft_overlap_combo.currentTextChanged.connect(self._request_analysis)
        stft_controls.addWidget(self.stft_overlap_combo)
        self.cwt_control_group = QtWidgets.QWidget()
        cwt_controls = QtWidgets.QHBoxLayout(self.cwt_control_group)
        cwt_controls.setContentsMargins(0, 0, 0, 0)
        cwt_controls.setSpacing(7)
        cwt_controls.addWidget(self._muted_label("每倍频程声部数"))
        self.cwt_voices_combo = QtWidgets.QComboBox()
        self.cwt_voices_combo.addItems(["8", "12", "16"])
        self.cwt_voices_combo.setCurrentText("12")
        self.cwt_voices_combo.currentTextChanged.connect(self._request_analysis)
        cwt_controls.addWidget(self.cwt_voices_combo)
        controls_layout.addWidget(self.stft_control_group)
        controls_layout.addWidget(self.cwt_control_group)
        self.fft_control_group = QtWidgets.QWidget()
        fft_controls = QtWidgets.QGridLayout(self.fft_control_group)
        fft_controls.setContentsMargins(0, 0, 0, 0)
        self.fft_info_label = QtWidgets.QLabel("FFT 频率信息")
        self.fft_info_label.setObjectName("Muted")
        self.fft_info_label.setWordWrap(True)
        fft_controls.addWidget(self._muted_label("FFT 窗"), 0, 0)
        fft_controls.addWidget(self.window_combo, 0, 1)
        fft_controls.addWidget(self._muted_label("频轴"), 0, 2)
        fft_controls.addWidget(self.frequency_scale, 0, 3)
        self.fft_comparison_check = QtWidgets.QCheckBox("对比频谱")
        self.fft_comparison_check.setChecked(True)
        self.fft_comparison_check.setEnabled(False)
        self.fft_comparison_check.toggled.connect(self._fft_comparison_changed)
        self.fft_comparison_check.setToolTip(
            "两份 WAV 使用主时间轴上的同一区间、窗函数及 dBFS 参考；对比信号缺失部分填零")
        fft_controls.addWidget(self.fft_comparison_check, 0, 4)
        self.fft_fit_button = QtWidgets.QPushButton("恢复完整频谱")
        self.fft_fit_button.setToolTip("将左右 FFT 频率轴恢复为当前 WAV 的完整频率范围")
        self.fft_fit_button.clicked.connect(self._fit_fft_range)
        fft_controls.addWidget(self.fft_fit_button, 0, 5)
        fft_controls.addWidget(self.fft_info_label, 1, 0, 1, 6)
        controls_layout.addWidget(self.fft_control_group)
        controls_layout.addStretch(1)

        analysis_section = QtWidgets.QWidget()
        analysis_layout = QtWidgets.QVBoxLayout(analysis_section)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(6)
        analysis_layout.addWidget(self.analysis_controls, 0)
        analysis_layout.addWidget(self.tabs, 1)

        self.analysis_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.analysis_splitter.setChildrenCollapsible(False)
        self.wave_section = wave_section
        self.wave_scroll = QtWidgets.QScrollArea()
        self.wave_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.wave_scroll.setWidgetResizable(True)
        self.wave_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.wave_scroll.setWidget(wave_section)
        self.wave_scroll.setMinimumHeight(250)
        # Keep the analysis canvas usable on small displays; the page scrolls
        # instead of squeezing its frequency/time axes into a few dozen pixels.
        analysis_section.setMinimumHeight(360)
        self.analysis_section = analysis_section
        self.analysis_splitter.addWidget(self.wave_scroll)
        self.analysis_splitter.addWidget(analysis_section)
        self.analysis_splitter.setStretchFactor(0, 1)
        self.analysis_splitter.setStretchFactor(1, 1)
        self.analysis_splitter.setSizes([460, 360])
        outer.addWidget(self.analysis_splitter, 1)

        self.readout = QtWidgets.QLabel("选区：--")
        self.readout.setObjectName("Readout")
        outer.addWidget(self.readout)
        self.playback_readout = QtWidgets.QLabel("播放位置：--")
        self.playback_readout.setObjectName("Muted")
        self.statusBar().addPermanentWidget(self.playback_readout)
        self.analysis_progress = QtWidgets.QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setFixedWidth(245)
        self.analysis_progress.setMaximumHeight(20)
        self.statusBar().addPermanentWidget(self.analysis_progress)
        self.analysis_progress.hide()
        self.content_scroll = QtWidgets.QScrollArea()
        self.content_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content_scroll.setWidget(content)
        self.setCentralWidget(self.content_scroll)
        self._analysis_tab_changed(self.tabs.currentIndex())
        self.statusBar().showMessage("就绪")

    @QtCore.Slot(int)
    def _analysis_tab_changed(self, index: int) -> None:
        show_fft = index == 0
        show_stft = index == 1
        show_cwt = index == 2
        self.fft_control_group.setVisible(show_fft)
        self.stft_control_group.setVisible(show_stft)
        self.cwt_control_group.setVisible(show_cwt)
        self.analysis_controls.setVisible(show_fft or show_stft or show_cwt)
        self._request_analysis()

    def _plot_widget(self, title: str, *, axis_items=None) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=title, axisItems=axis_items or {"bottom": TimeAxis()})
        self._style_plot(plot)
        return plot

    @staticmethod
    def _comparison_splitter(full, selected) -> QtWidgets.QSplitter:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(full)
        splitter.addWidget(selected)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 600])
        return splitter

    @staticmethod
    def _style_plot(plot: pg.PlotWidget) -> None:
        plot.setBackground("#171b1f")
        plot.getPlotItem().getViewBox().setBorder(pg.mkPen("#30383e"))
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen("#59656c"))
            axis.setTextPen(pg.mkPen("#aeb8bd"))
            axis.setTickFont(QtGui.QFont("Segoe UI", 10))
            axis.setStyle(maxTickLevel=0, maxTextLevel=0, tickTextOffset=7)

    def _image_plot(self, title: str, scale_label: str = "相对幅度 (dB)") -> pg.PlotWidget:
        plot = self._plot_widget(title, axis_items={"left": FrequencyAxis("left"), "bottom": TimeAxis()})
        plot.setLabel("left", "频率", units="Hz")
        plot.setLabel("bottom", "时间", units="s")
        plot.getAxis("left").enableAutoSIPrefix(False)
        plot.showGrid(x=True, y=True, alpha=0.08)
        image = pg.ImageItem(axisOrder="row-major")
        image.setColorMap(self._color_map())
        plot.addItem(image)
        plot.image_item = image
        color_bar = pg.ColorBarItem(values=(-100, 0), width=12, colorMap=self._color_map(),
                                    label=scale_label, interactive=False,
                                    pen=pg.mkPen("#59656c"))
        color_bar.setImageItem(image, insert_in=plot.plotItem)
        plot.color_bar = color_bar
        color_bar.axis.setTickFont(QtGui.QFont("Segoe UI", 10))
        color_bar.axis.setStyle(maxTickLevel=0, maxTextLevel=0)
        plot.setToolTip("幅度以当前分析的峰值为 0 dB；CWT 左右图使用共同峰值参考。")
        plot.setAspectLocked(False)
        return plot

    @staticmethod
    def _color_map() -> pg.ColorMap:
        return pg.ColorMap(np.array([0.0, 0.35, 0.7, 1.0]),
                           np.array([[22, 29, 35, 255], [31, 88, 91, 255],
                                     [55, 163, 139, 255], [236, 178, 105, 255]], dtype=np.ubyte))

    def _tool_button(self, text: str, icon: QtGui.QIcon, action) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setText(text)
        button.setIcon(icon)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setToolTip(text)
        button.clicked.connect(action)
        return button

    @staticmethod
    def _muted_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("Muted")
        return label

    def _set_empty_state(self) -> None:
        for button in (self.play_button, self.stop_button, self.play_selection_button,
                       self.reference_button, self.export_button, self.export_plot_button):
            button.setEnabled(False)
        self.selection_reference_button.setEnabled(False)
        self.comparison_button.setEnabled(False)
        self.overlay_check.setEnabled(False)
        self.locate_selection_button.setEnabled(False)
        self.visible_range_button.setEnabled(False)

    def open_audio(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开音频", "", "WAV 音频 (*.wav)")
        if not path:
            return
        try:
            self.load_path(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "无法打开音频", str(exc))

    def load_path(self, path: str, *, remember_previous: bool = True) -> None:
        samples, sample_rate = read_audio(path, channel=self.channel_combo.currentData())
        if not samples.size:
            raise ValueError("WAV 文件不含采样点")
        preserve_view = not remember_previous and self.audio_path == Path(path)
        previous_view = self.overview_region.getRegion() if preserve_view else None
        previous_selection = self.analysis_selection if preserve_view else None
        if remember_previous:
            self.overlay_check.blockSignals(True)
            self.overlay_check.setChecked(False)
            self.overlay_check.blockSignals(False)
        self.samples, self.sample_rate, self.audio_path = samples, sample_rate, Path(path)
        if self.comparison_path is not None:
            self.comparison_samples = self._read_comparison(self.comparison_path)
            self.comparison_rate = sample_rate
        if not preserve_view:
            self.comparison_offset.blockSignals(True)
            self.comparison_offset.setValue(0.0)
            self.comparison_offset.blockSignals(False)
            self.comparison_offset_s = 0.0
        self.comparison_spectrum_cache.clear()
        self._update_comparison_labels()
        self._fit_fft_range()
        self.analysis_selection = None
        self.selection_region.hide()
        self.comparison_selection_region.hide()
        self.full_spectrum_cache.clear()
        self.reference = None
        self.full_cwt_cache.clear()
        self.full_correlation_cache.clear()
        self.selection_cache.clear()
        self.pending_full_spectrum.clear()
        self.pending_full_cwt.clear()
        self.pending_analysis.clear()
        self.stop_playback()
        self.playback_offset_s = self.playback_end_s = 0.0
        self._set_playback_cursor(0.0)
        for plot in (self.fft_plot, self.fft_full_plot, self.stft_plot,
                     self.cwt_plot, self.cwt_overview_plot):
            plot.getAxis("bottom" if plot in (self.fft_plot, self.fft_full_plot) else "left").maximum_frequency = sample_rate / 2
        duration = samples.size / sample_rate
        self._fft_selected_raw = (np.empty(0), np.empty(0))
        self._fft_full_raw = (np.empty(0), np.empty(0))
        self._fft_comparison_selected_raw = (np.empty(0), np.empty(0))
        self._fft_comparison_full_raw = (np.empty(0), np.empty(0))
        for curve in (self.fft_selected_curve, self.fft_full_curve, self.auto_curve,
                      self.auto_full_curve, self.cross_curve, self.cross_full_curve,
                      self.fft_comparison_selected_curve, self.fft_comparison_full_curve):
            curve.setData([], [])
        self.cross_peak_marker.hide()
        self.cross_full_peak_marker.hide()
        for plot, title in ((self.stft_plot, "STFT"),
                            (self.cwt_overview_plot, "CWT 完整信号"),
                            (self.cwt_plot, "CWT 当前选区")):
            plot.image_item.clear()
            plot.setTitle(f"{title} | 等待分析")
            plot.setXRange(0, duration, padding=0)
            plot.setYRange(0, sample_rate / 2, padding=0)
        info = sf.info(path)
        ox, oy = _envelope(samples, 1800)
        self.overview_curve.setData(ox / sample_rate, oy)
        peak = max(float(np.max(np.abs(samples))), 1e-9)
        if self.comparison_samples is not None and self.overlay_check.isChecked():
            peak = max(peak, float(np.max(np.abs(self.comparison_samples))))
        self.waveform.setYRange(-1.1 * peak, 1.1 * peak, padding=0)
        self.overview.setYRange(-1.1 * peak, 1.1 * peak, padding=0)
        self.overview.setXRange(0, duration, padding=0)
        self.overview_region.setBounds((0, duration))
        self.cwt_overview_region.setBounds((0, duration))
        self.waveform.getViewBox().setLimits(
            xMin=0, xMax=duration, minXRange=min(duration, 8 / sample_rate),
            maxXRange=duration)
        self.comparison_waveform.getViewBox().setLimits(
            xMin=0, xMax=duration, minXRange=min(duration, 8 / sample_rate), maxXRange=duration)
        self._set_region(previous_view or (0, min(duration, 1.5)))
        if previous_selection is not None:
            self._set_analysis_selection(*previous_selection)
        self._comparison_mode_changed()
        self._refresh_waveform_curves()
        self.metadata_label.setText(self.audio_path.name)
        self.metadata_label.setToolTip(str(self.audio_path))
        self.technical_info_label.setText(
            f"{sample_rate:,} Hz  |  {duration:.3f} 秒  |  {info.channels} 声道")
        self.audio_switch_combo.blockSignals(True)
        self.audio_switch_combo.clear()
        for audio_path in sorted(self.audio_path.parent.glob("*.wav")):
            self.audio_switch_combo.addItem(audio_path.name, str(audio_path))
        current_index = self.audio_switch_combo.findData(str(self.audio_path))
        self.audio_switch_combo.setCurrentIndex(max(0, current_index))
        self.audio_switch_combo.blockSignals(False)
        for button in (self.play_button, self.stop_button, self.play_selection_button,
                       self.reference_button, self.export_button, self.export_plot_button):
            button.setEnabled(True)
        self.selection_reference_button.setEnabled(True)
        self.comparison_button.setEnabled(True)
        self.overlay_check.setEnabled(self.comparison_samples is not None)
        self.reference = None
        self.reference_rate = None
        self.reference_status.setText("模板：未设置")
        self._request_analysis()
        self.statusBar().showMessage(f"已载入 {self.audio_path.name}", 3500)

    def load_reference(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择互相关参考 WAV", "", "WAV 音频 (*.wav)")
        if not path:
            return
        try:
            reference, sample_rate = read_audio(path, channel=self.channel_combo.currentData())
            if sample_rate != self.sample_rate:
                from scipy.signal import resample_poly
                from math import gcd
                factor = gcd(sample_rate, self.sample_rate)
                reference = resample_poly(reference, self.sample_rate // factor, sample_rate // factor)
            self.reference = reference
            self.reference_rate = self.sample_rate
            self.reference_status.setText(f"模板：{Path(path).name}  |  {self.reference.size:,} 点")
            self.statusBar().showMessage(f"已载入参考信号：{Path(path).name}", 4000)
            self._request_analysis()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "无法载入参考信号", str(exc))

    def set_selection_reference(self) -> None:
        start, end = self._region_samples()
        if end - start < 8:
            self.statusBar().showMessage("请先选择至少 8 个采样点作为相关模板", 4000)
            return
        self.reference = self.samples[start:end].copy()
        self.full_correlation_cache.clear()
        self.reference_rate = self.sample_rate
        self.reference_status.setText(
            f"模板：当前选区  |  {self.reference.size:,} 点  |  {start / self.sample_rate:.4f}–{end / self.sample_rate:.4f} 秒")
        self._request_analysis()

    def _channel_changed(self, _index: int) -> None:
        if self.audio_path is not None:
            self.load_path(str(self.audio_path), remember_previous=False)

    def _switch_audio(self, index: int) -> None:
        path = self.audio_switch_combo.itemData(index)
        if path and Path(path) != self.audio_path:
            try:
                self.load_path(path)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "无法打开音频", str(exc))

    def _set_comparison_visible(self, visible: bool) -> None:
        show = visible and self.comparison_samples is not None
        self.comparison_controls.setVisible(show)
        self.fft_comparison_check.setEnabled(show)
        self._comparison_mode_changed()
        self._fft_comparison_changed()

    def open_comparison(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择对比 WAV", str(self.audio_path.parent) if self.audio_path else "",
            "WAV 音频 (*.wav)")
        if path:
            try:
                self.load_comparison_path(path)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "无法载入对比音频", str(exc))

    def _read_comparison(self, path: Path) -> np.ndarray:
        samples, rate = read_audio(str(path), channel=self.channel_combo.currentData())
        if not samples.size:
            raise ValueError("对比 WAV 文件不含采样点")
        if rate != self.sample_rate:
            divisor = int(np.gcd(rate, self.sample_rate))
            samples = signal.resample_poly(samples, self.sample_rate // divisor, rate // divisor)
        return samples

    def load_comparison_path(self, path: str) -> None:
        if not self.sample_rate:
            raise ValueError("请先打开主信号 WAV")
        samples = self._read_comparison(Path(path))
        self.comparison_path = Path(path)
        self.comparison_samples = samples
        self.comparison_rate = self.sample_rate
        self.comparison_offset.blockSignals(True)
        self.comparison_offset.setValue(0.0)
        self.comparison_offset.blockSignals(False)
        self.comparison_offset_s = 0.0
        self.comparison_spectrum_cache.clear()
        self._fft_comparison_selected_raw = (np.empty(0), np.empty(0))
        self._fft_comparison_full_raw = (np.empty(0), np.empty(0))
        self.overlay_check.setEnabled(True)
        self.overlay_check.setChecked(True)
        self._update_comparison_labels()
        self._set_comparison_visible(True)

    def _comparison_mode_changed(self, *_args) -> None:
        enabled = self.overlay_check.isChecked() and self.comparison_samples is not None
        split = enabled and self.comparison_mode.currentIndex() == 1
        self.comparison_waveform.setVisible(split)
        self.comparison_curve.setVisible(enabled and not split)
        self.waveform.getAxis("bottom").setStyle(showValues=not split)
        self.waveform.getAxis("bottom").showLabel(not split)
        if split:
            self.comparison_waveform.setXRange(*self.waveform.getViewBox().viewRange()[0], padding=0)
        peak = max(float(np.max(np.abs(self.samples), initial=0)), 1e-9)
        if enabled and not split:
            peak = max(peak, float(np.max(np.abs(self.comparison_samples), initial=0)))
        self.waveform.setYRange(-1.1 * peak, 1.1 * peak, padding=0)
        if enabled:
            compare_peak = max(float(np.max(np.abs(self.comparison_samples), initial=0)), 1e-9)
            self.comparison_waveform.setYRange(-1.1 * compare_peak, 1.1 * compare_peak, padding=0)
        self.comparison_waveform.setToolTip("共用主时间轴；对比轨幅度独立适应。拖选蓝色区域仍对主信号分析、播放和导出。")
        self._refresh_waveform_curves()
        self._fit_waveform_layout()

    def _fit_waveform_layout(self) -> None:
        compact = self.height() < 820
        if not self._overview_manual:
            self.overview_toggle.blockSignals(True)
            self.overview_toggle.setChecked(not compact)
            self.overview_toggle.blockSignals(False)
        self.overview_container.setVisible(self.overview_toggle.isChecked())
        self.overview.setVisible(self.overview_toggle.isChecked())
        self.waveform.setMinimumHeight(210)
        self.comparison_waveform.setMinimumHeight(210)
        self.overview.setMinimumHeight(140)
        self.overview.setMaximumHeight(190)
        # PlotWidget's own size hint omits its graphics-layout title/axis space.
        # Preserve real canvas space; short windows can scroll the waveform area.
        layout = self.wave_section.layout()
        minimum = self.waveform.minimumHeight() + 6
        if self.overview_toggle.isChecked():
            minimum += self.overview.minimumHeight() + 26
        if not self.comparison_controls.isHidden():
            minimum += self.comparison_controls.sizeHint().height() + layout.spacing()
        if not self.comparison_waveform.isHidden():
            minimum += self.comparison_waveform.minimumHeight() + layout.spacing()
        self.wave_section.setMinimumHeight(minimum)

    def _overview_visibility_changed(self, _visible: bool) -> None:
        self._overview_manual = True
        self._fit_waveform_layout()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "analysis_splitter"):
            self._fit_waveform_layout()
            if self.comparison_path is not None:
                self._update_comparison_labels()

    def _comparison_view_changed(self, _mask) -> None:
        if self.comparison_waveform.isVisible() and not self._updating_view_from_overview:
            x_range = self.comparison_waveform.getViewBox().viewRange()[0]
            if not np.allclose(x_range, self.waveform.getViewBox().viewRange()[0], atol=1e-9):
                self.waveform.setXRange(*x_range, padding=0)

    def _fft_comparison_enabled(self) -> bool:
        return (self.overlay_check.isChecked() and self.comparison_samples is not None
                and self.fft_comparison_check.isChecked())

    def _fft_comparison_changed(self, *_args) -> None:
        self._fft_comparison_selected_raw = (np.empty(0), np.empty(0))
        self._fft_comparison_full_raw = (np.empty(0), np.empty(0))
        self._refresh_fft_curves()
        if self.tabs.currentIndex() == 0:
            self._request_analysis()

    def _update_comparison_labels(self) -> None:
        for label, path, color, prefix in (
            (self.current_signal_label, self.audio_path, "#48c7a5", "当前"),
            (self.comparison_label, self.comparison_path, "#df9562", "对比"),
        ):
            name = path.name if path is not None else "未载入"
            metrics = label.fontMetrics()
            short = metrics.elidedText(name, QtCore.Qt.TextElideMode.ElideMiddle,
                                      max(80, min(300, self.width() // 4)))
            label.setText(f'<span style="color:{color}">● {prefix}</span>  {escape(short)}')
            label.setToolTip(str(path) if path is not None else "")
        self.comparison_controls.setVisible(
            self.overlay_check.isChecked() and self.comparison_samples is not None)
        self.fft_comparison_check.setEnabled(
            self.overlay_check.isChecked() and self.comparison_samples is not None)

    def _comparison_offset_changed(self, seconds: float) -> None:
        if not self.sample_rate:
            return
        offset_samples = round(seconds * self.sample_rate)
        self.comparison_offset_s = offset_samples / self.sample_rate
        self.comparison_offset.blockSignals(True)
        self.comparison_offset.setValue(self.comparison_offset_s)
        self.comparison_offset.blockSignals(False)
        self.comparison_offset.setToolTip(
            f"实际偏移 {offset_samples:+,} 个主信号采样点（{self.comparison_offset_s:+.9f} 秒）。\n"
            "正值向右移动；零表示文件起点重合。")
        self._refresh_waveform_curves()
        self._fft_comparison_changed()

    def _region_samples(self) -> tuple[int, int]:
        lo, hi = self.analysis_selection or self.overview_region.getRegion()
        start = int(np.clip(round(lo * self.sample_rate), 0, self.samples.size))
        end = int(np.clip(round(hi * self.sample_rate), start, self.samples.size))
        return start, end

    def _interaction_changed(self, index: int) -> None:
        for plot in (self.waveform, self.comparison_waveform):
            view = plot.getViewBox()
            view.select_on_drag = index == 0
            view.setCursor(QtCore.Qt.CursorShape.CrossCursor if index == 0
                           else QtCore.Qt.CursorShape.OpenHandCursor)

    def _set_analysis_selection(self, left: float, right: float, finished: bool = True) -> None:
        if not self.samples.size or not self.sample_rate:
            return
        duration = self.samples.size / self.sample_rate
        left, right = sorted(np.clip((left, right), 0, duration))
        if right - left < 1 / self.sample_rate:
            return
        self.analysis_selection = (float(left), float(right))
        self.selection_region.setRegion(self.analysis_selection)
        self.selection_region.show()
        self.comparison_selection_region.setRegion(self.analysis_selection)
        self.comparison_selection_region.show()
        self.cwt_overview_region.setRegion(self.analysis_selection)
        self._selection_readout()
        if finished:
            self._request_analysis()

    def _use_visible_range(self) -> None:
        self.analysis_selection = None
        self.selection_region.hide()
        self.comparison_selection_region.hide()
        self.cwt_overview_region.setRegion(self.overview_region.getRegion())
        self._selection_readout()
        self._request_analysis()

    def _locate_selection(self) -> None:
        if self.analysis_selection is None or not self.sample_rate:
            return
        lo, hi = self.analysis_selection
        duration = self.samples.size / self.sample_rate
        width = min(duration, max(.02, (hi - lo) * 1.4))
        left = float(np.clip((lo + hi - width) / 2, 0, duration - width))
        self._set_region((left, left + width))

    def _navigation_changed(self) -> None:
        self.cwt_overview_region.setRegion(self.analysis_selection or self.overview_region.getRegion())
        self._selection_readout()
        if self.analysis_selection is None:
            self.navigation_analysis_timer.start()

    def _set_region(self, region: tuple[float, float]) -> None:
        self.overview_region.blockSignals(True)
        self.overview_region.setRegion(region)
        self.cwt_overview_region.setRegion(self.analysis_selection or region)
        self.overview_region.blockSignals(False)
        if self.samples.size:
            self._updating_view_from_overview = True
            self.waveform.setXRange(*region, padding=0)
            self._updating_view_from_overview = False
            self._refresh_waveform_curves(*region)
        self._selection_readout()

    def _waveform_view_changed(self, _view_box, x_range) -> None:
        if not self.samples.size or self._updating_view_from_overview:
            return
        duration = self.samples.size / self.sample_rate
        left, right = map(float, x_range)
        width = min(duration, max(8 / self.sample_rate, right - left))
        left = float(np.clip(left, 0.0, duration - width))
        region = (left, left + width)
        self.overview_region.blockSignals(True)
        self.overview_region.setRegion(region)
        self.overview_region.blockSignals(False)
        self._refresh_waveform_curves(left, left + width)
        self._navigation_changed()

    def _refresh_waveform_curves(self, start_s: float | None = None,
                                 end_s: float | None = None) -> None:
        if not self.samples.size or not self.sample_rate:
            return
        if start_s is None or end_s is None:
            start_s, end_s = self.waveform.getViewBox().viewRange()[0]
        start = int(np.clip(np.floor(start_s * self.sample_rate), 0, self.samples.size))
        end = int(np.clip(np.ceil(end_s * self.sample_rate), start + 1, self.samples.size))
        x, y = _envelope(self.samples[start:end], maximum_points=12000)
        self.waveform_curve.setData((x + start) / self.sample_rate, y)
        if self.comparison_samples is not None and self.overlay_check.isChecked():
            offset = self.comparison_offset_s
            compare_start = max(0, int(np.floor((start_s - offset) * self.sample_rate)))
            compare_end = min(self.comparison_samples.size,
                              int(np.ceil((end_s - offset) * self.sample_rate)))
            if compare_end > compare_start:
                cx, cy = _envelope(self.comparison_samples[compare_start:compare_end],
                                   maximum_points=12000)
                self.comparison_curve.setData((cx + compare_start) / self.sample_rate + offset, cy)
                self.comparison_track_curve.setData((cx + compare_start) / self.sample_rate + offset, cy)
                self.comparison_curve.setVisible(self.comparison_mode.currentIndex() == 0)
            else:
                self.comparison_curve.setData([], [])
                self.comparison_track_curve.setData([], [])
                self.comparison_curve.setVisible(False)
        else:
            self.comparison_curve.setData([], [])
            self.comparison_track_curve.setData([], [])
            self.comparison_curve.setVisible(False)

    def _overview_changed(self) -> None:
        if not self.samples.size:
            return
        region = self.overview_region.getRegion()
        self._updating_view_from_overview = True
        self.waveform.setXRange(*region, padding=0)
        self._updating_view_from_overview = False
        self._refresh_waveform_curves(*region)
        self._navigation_changed()

    def _overview_change_finished(self) -> None:
        if self.analysis_selection is None:
            self.navigation_analysis_timer.start(0)

    def _zoom_selection(self, factor: float) -> None:
        if not self.samples.size:
            return
        lo, hi = self.overview_region.getRegion()
        center = 0.5 * (lo + hi)
        width = min(self.samples.size / self.sample_rate, max(0.01, (hi - lo) * factor))
        duration = self.samples.size / self.sample_rate
        left = min(max(0.0, center - width / 2), duration - width)
        self._set_region((left, left + width))
        self._selection_readout()
        if self.analysis_selection is None:
            self._request_analysis()

    def _fit_file(self) -> None:
        if self.samples.size:
            duration = self.samples.size / self.sample_rate
            self._set_region((0.0, duration))
            self._selection_readout()
            if self.analysis_selection is None:
                self._request_analysis()

    def _selection_readout(self) -> None:
        start, end = self._region_samples()
        duration = (end - start) / self.sample_rate if self.sample_rate else 0
        if not self.sample_rate:
            self.readout.setText("浏览：--    选区：--")
            return
        left, right = self.overview_region.getRegion()
        source = "分析选区" if self.analysis_selection is not None else "分析跟随浏览"
        self.visible_range_button.setEnabled(self.analysis_selection is not None)
        self.locate_selection_button.setEnabled(self.analysis_selection is not None)
        analysis_text = (f"分析 {start / self.sample_rate:.3f}–{end / self.sample_rate:.3f} s"
                         if self.analysis_selection is not None else "分析跟随浏览")
        self.waveform.setTitle(f"浏览 {left:.3f}–{right:.3f} s  |  {analysis_text}", size="9pt")
        if self.comparison_path is not None:
            name = self.fontMetrics().elidedText(self.comparison_path.name,
                                                QtCore.Qt.TextElideMode.ElideMiddle, 350)
            self.comparison_waveform.setTitle(f"对比 · {escape(name)}", size="9pt")
        self.readout.setText(
            f'<span style="color:#74e0c1">浏览</span>  {left:.4f}–{right:.4f} 秒     |     '
            f'<span style="color:#7da6f7">{source}</span>  '
            f"{start / self.sample_rate:.4f}–{end / self.sample_rate:.4f} 秒   "
            f"{end - start:,} 点 · {duration:.4f} 秒")

    def _request_analysis(self, *_args) -> None:
        if self._closing or not self.samples.size:
            return
        start, end = self._region_samples()
        if end - start < 8:
            return
        self.generation += 1
        self.analysis_progress.hide()
        self.statusBar().showMessage("等待分析…")
        self.thread_pool.clear()
        self.interactive_thread_pool.clear()
        for generation, cancel_event in tuple(self.worker_cancel_events.items()):
            if generation >= self.generation:
                continue
            cancel_event.set()
            key_spectrum, key_cwt = self.pending_analysis.pop(generation, (None, None))
            if key_spectrum is not None:
                self.pending_full_spectrum.discard(key_spectrum)
            if key_cwt is not None:
                self.pending_full_cwt.discard(key_cwt)
            self.worker_cancel_events.pop(generation, None)
        self.analysis_debounce.start()

    def _launch_analysis(self) -> None:
        if self._closing or not self.samples.size:
            return
        start, end = self._region_samples()
        if end - start < 8:
            return
        window_name = self._fft_window_name()
        mode = ("fft", "stft", "cwt", "correlation")[self.tabs.currentIndex()]
        key = (id(self.samples), window_name)
        cached_full = self.full_spectrum_cache.get(key)
        comparison_key = self._comparison_spectrum_key()
        cached_comparison = self.comparison_spectrum_cache.get(comparison_key)
        cwt_key = (id(self.samples), int(self.cwt_voices_combo.currentText()))
        cached_cwt = self.full_cwt_cache.get(cwt_key)
        result_key = self._selection_result_key(mode, start, end, window_name)
        cached_selection = self.selection_cache.get(result_key)
        if cached_selection is not None:
            self.selection_cache.move_to_end(result_key)
            payload = dict(cached_selection)
            payload["full_spectrum"] = cached_full
            payload["full_cwt"] = cached_cwt
            payload["full_correlation"] = self.full_correlation_cache.get((id(self.samples), id(self.reference)))
            payload["full_comparison_spectrum"] = cached_comparison
            self._analysis_ready(self.generation, payload)
            return
        compute_full_spectrum = mode == "fft" and cached_full is None and key not in self.pending_full_spectrum
        compute_full_cwt = mode == "cwt" and cached_cwt is None and cwt_key not in self.pending_full_cwt
        if compute_full_spectrum:
            self.pending_full_spectrum.add(key)
        if compute_full_cwt:
            self.pending_full_cwt.add(cwt_key)
        self.pending_analysis[self.generation] = (
            key if compute_full_spectrum else None,
            cwt_key if compute_full_cwt else None,
        )
        worker = AnalysisWorker(self.generation, self.samples, self.sample_rate,
                                (start, end), self.reference, window_name, cached_full,
                                int(self.stft_window_combo.currentText()),
                                int(self.stft_fft_combo.currentText()),
                                float(self.stft_overlap_combo.currentText().rstrip("%")) / 100.0,
                                int(self.cwt_voices_combo.currentText()), cached_cwt,
                                mode, compute_full_spectrum, compute_full_cwt, result_key,
                                cancel_event=threading.Event(),
                                full_correlation=self.full_correlation_cache.get(
                                    (id(self.samples), id(self.reference))),
                                comparison_samples=self.comparison_samples if self._fft_comparison_enabled() else None,
                                comparison_offset_samples=round(self.comparison_offset_s * self.sample_rate),
                                full_comparison_spectrum=cached_comparison)
        worker.signals.ready.connect(self._analysis_ready)
        worker.signals.failed.connect(self._analysis_failed)
        worker.signals.cancelled.connect(self._analysis_cancelled)
        worker.signals.progress.connect(self._analysis_progress)
        self.worker_cancel_events[self.generation] = worker.cancel_event
        pool = self.thread_pool if mode == "cwt" else self.interactive_thread_pool
        pool.start(worker)
        self.statusBar().showMessage("正在分析选区…")

    @QtCore.Slot(int, int, int, str)
    def _analysis_progress(self, generation: int, done: int, total: int, stage: str) -> None:
        if self._closing or generation != self.generation or self.tabs.currentIndex() != 2:
            return
        total = max(1, total)
        self.analysis_progress.setValue(round(100 * done / total))
        self.analysis_progress.setFormat(f"CWT {stage} · {done}/{total} 块 · %p%")
        self.analysis_progress.show()
        self.statusBar().showMessage("正在计算 CWT；切换页面可取消")

    def _fft_window_name(self) -> str:
        return {"汉宁窗": "hann", "海明窗": "hamming", "布莱克曼窗": "blackman",
                "矩形窗": "rectangular"}[self.window_combo.currentText()]

    def _selection_result_key(self, mode: str, start: int, end: int,
                              window_name: str) -> tuple:
        return (id(self.samples), mode, start, end,
                window_name if mode == "fft" else None,
                int(self.stft_window_combo.currentText()) if mode == "stft" else None,
                int(self.stft_fft_combo.currentText()) if mode == "stft" else None,
                self.stft_overlap_combo.currentText() if mode == "stft" else None,
                int(self.cwt_voices_combo.currentText()) if mode == "cwt" else None,
                id(self.reference) if mode == "correlation" and self.reference is not None else None,
                self._comparison_spectrum_key() if mode == "fft" and self._fft_comparison_enabled() else None)

    def _comparison_spectrum_key(self) -> tuple:
        return (id(self.samples), id(self.comparison_samples),
                round(self.comparison_offset_s * self.sample_rate), self._fft_window_name())

    def _set_frequency_scale(self, scale: str) -> None:
        view = self.fft_plot.getViewBox()
        old_log = self.fft_plot.getAxis("bottom").log_frequency
        left, right = view.viewRange()[0]
        if old_log:
            left, right = 10 ** np.clip([left, right], -12, 12)
        nyquist = self.sample_rate / 2 if self.sample_rate else 24000.0
        left = max(20.0 if scale == "对数" else 0.0, left)
        right = min(nyquist, max(right, left * 1.01 + 1))
        if left >= right:
            left, right = (20.0, nyquist) if scale == "对数" else (0.0, nyquist)
        if scale == "线性":
            left = 0.0 if left <= 20.0001 else left
        for plot in (self.fft_full_plot, self.fft_plot):
            axis = plot.getAxis("bottom")
            axis.maximum_frequency = nyquist
            axis.set_frequency_scale(scale == "对数")
        self._refresh_fft_curves()
        self.fft_plot.setXRange(*(np.log10([left, right]) if scale == "对数" else (left, right)), padding=0)

    def _fit_fft_range(self) -> None:
        if not self.sample_rate:
            return
        nyquist = self.sample_rate / 2
        limits = (np.log10(min(20.0, nyquist / 2)), np.log10(nyquist)) \
            if self.frequency_scale.currentText() == "对数" else (0.0, nyquist)
        self.fft_plot.setXRange(*limits, padding=0)

    @staticmethod
    def _log_frequency(values):
        if values is None:
            return np.empty(0)
        values = np.asarray(values)
        if not values.size:
            return values
        return np.log10(np.maximum(values, np.finfo(float).tiny))

    def _refresh_fft_curves(self) -> None:
        x1, y1 = self._fft_selected_raw
        x2, y2 = self._fft_full_raw
        if self.frequency_scale.currentText() == "对数":
            positive1, positive2 = np.asarray(x1) > 0, np.asarray(x2) > 0
            x1, y1 = self._log_frequency(np.asarray(x1)[positive1]), np.asarray(y1)[positive1]
            x2, y2 = self._log_frequency(np.asarray(x2)[positive2]), np.asarray(y2)[positive2]
        self.fft_selected_curve.setData(x1, y1)
        self.fft_full_curve.setData(x2, y2)
        for curve, raw in ((self.fft_comparison_selected_curve, self._fft_comparison_selected_raw),
                           (self.fft_comparison_full_curve, self._fft_comparison_full_raw)):
            cx, cy = raw
            if self.frequency_scale.currentText() == "对数":
                positive = np.asarray(cx) > 0
                cx, cy = self._log_frequency(np.asarray(cx)[positive]), np.asarray(cy)[positive]
            curve.setData(cx, cy)
            curve.setVisible(self._fft_comparison_enabled() and len(cx) > 0)

    def _fft_mouse_moved(self, position, plot) -> None:
        if not self.samples.size:
            return
        # Each panel owns a different scene, even when their local rectangles overlap.
        view = plot.getPlotItem().vb
        if view.sceneBoundingRect().contains(position):
            point = view.mapSceneToView(position)
            frequency = 10 ** point.x() if self.frequency_scale.currentText() == "对数" else point.x()
            self.statusBar().showMessage(f"频率 {frequency:,.2f} Hz    幅度 {point.y():.2f} dBFS")

    def _waveform_mouse_moved(self, position) -> None:
        view = self.waveform.getViewBox()
        if not view.sceneBoundingRect().contains(position) or not self.sample_rate:
            return
        point = view.mapSceneToView(position)
        self.waveform.setToolTip(
            f"时间 {point.x():.6f} 秒 · 幅度 {point.y():.6f}\n"
            "蓝色：分析选区；左键重新拖选，Shift + 拖动平移，Esc 清除选区。")

    @QtCore.Slot(int, object)
    def _analysis_ready(self, generation: int, payload: dict[str, object], *,
                        from_cache: bool = False) -> None:
        if not from_cache:
            key_spectrum, key_cwt = self.pending_analysis.pop(generation, (None, None))
            self.worker_cancel_events.pop(generation, None)
            if key_spectrum is not None:
                self.pending_full_spectrum.discard(key_spectrum)
            if key_cwt is not None:
                self.pending_full_cwt.discard(key_cwt)

        result_key = payload.get("result_key")
        result_is_current_audio = result_key is not None and result_key[0] == id(self.samples)
        full_spectrum = payload.get("full_spectrum")
        spectrum_key = payload.get("full_spectrum_cache_key")
        if full_spectrum is not None and spectrum_key is not None and spectrum_key[0] == id(self.samples):
            self.full_spectrum_cache[spectrum_key] = full_spectrum
        comparison_key = payload.get("full_comparison_cache_key")
        if comparison_key == self._comparison_spectrum_key() and payload.get("full_comparison_spectrum") is not None:
            self.comparison_spectrum_cache[comparison_key] = payload["full_comparison_spectrum"]
            self.comparison_spectrum_cache.move_to_end(comparison_key)
            while len(self.comparison_spectrum_cache) > 3:
                self.comparison_spectrum_cache.popitem(last=False)
        full_cwt = payload.get("full_cwt")
        cwt_key = payload.get("full_cwt_cache_key")
        if full_cwt is not None and cwt_key is not None and cwt_key[0] == id(self.samples):
            self.full_cwt_cache[cwt_key] = full_cwt
        full_correlation = payload.get("full_correlation")
        correlation_key = payload.get("full_correlation_cache_key")
        if (full_correlation is not None and correlation_key == (id(self.samples), id(self.reference))):
            self.full_correlation_cache.clear()
            self.full_correlation_cache[correlation_key] = full_correlation
        cached_payload = dict(payload)
        cached_payload.pop("full_spectrum", None)
        cached_payload.pop("full_cwt", None)
        cached_payload.pop("full_correlation", None)
        cached_payload.pop("full_comparison_spectrum", None)
        if result_is_current_audio:
            self.selection_cache[result_key] = cached_payload
            self.selection_cache.move_to_end(result_key)
            while len(self.selection_cache) > 4:
                self.selection_cache.pop(next(iter(self.selection_cache)))
        if generation != self.generation:
            self._show_cached_current_selection()
            return
        mode = payload["mode"]
        if mode == "cwt":
            current_cwt_key = (id(self.samples), int(self.cwt_voices_combo.currentText()))
            if (current_cwt_key not in self.full_cwt_cache
                    and current_cwt_key not in self.pending_full_cwt):
                self._request_analysis()
        if mode == "fft":
            f, db, n_fft, bin_spacing_hz, duration_s, sample_count, main_lobe_hz = payload["spectrum"]
            full_spectrum = payload.get("full_spectrum") or self.full_spectrum_cache.get(
                (id(self.samples), self._fft_window_name()))
            self._fft_selected_raw = (f, db)
            if full_spectrum is not None:
                ff, fd = full_spectrum
                self._fft_full_raw = (ff, fd)
            selected_comparison = payload.get("comparison_spectrum")
            full_comparison = payload.get("full_comparison_spectrum") or self.comparison_spectrum_cache.get(
                self._comparison_spectrum_key())
            self._fft_comparison_selected_raw = (
                selected_comparison[:2] if selected_comparison is not None and selected_comparison[2]
                else (np.empty(0), np.empty(0)))
            self._fft_comparison_full_raw = (
                full_comparison[:2] if full_comparison is not None and full_comparison[2]
                else (np.empty(0), np.empty(0)))
            self._refresh_fft_curves()
            self.fft_plot.setTitle(f"当前选区 · FFT  |  {duration_s:.4f} 秒 · {sample_count:,} 点")
            self.fft_full_plot.setTitle(
                f"完整信号 · FFT  |  {self.samples.size / self.sample_rate:.4f} 秒 · {self.samples.size:,} 点")
            self.fft_info_label.setText(
                f"频点间隔 {bin_spacing_hz:.3f} Hz  ·  {self.window_combo.currentText()}窗主瓣宽度约 {main_lobe_hz:.2f} Hz")
            self.fft_info_label.setToolTip(
                f"观测时长 {duration_s:.6f} 秒，采样点数 {sample_count:,}，FFT 长度 {n_fft:,}。\n"
                "频点间隔由采样率与 FFT 长度之比决定；补零只使频谱采样更密，不提高分辨相近频率的能力。\n"
                "主瓣宽度按当前窗函数的主瓣零点间距估算，体现有限观测时长下的分辨尺度。\n"
                "对比频谱使用相同主时间范围、窗函数及 dBFS 参考；对比文件超出部分填零。")
            if self._fft_comparison_enabled() and selected_comparison is not None:
                count = selected_comparison[2]
                self.fft_info_label.setText(self.fft_info_label.text() +
                    (f" · 对比有效 {count:,}/{sample_count:,} 点" if count else " · 对比信号未覆盖此选区"))
        elif mode == "stft":
            f_stft, t_stft, z_stft = payload["stft"]
            self._set_spectrogram(self.stft_plot, t_stft, f_stft, _relative_db(np.abs(z_stft), floor_db=-140))
            parameters = payload["stft_parameters"]
            window, nfft, overlap = (parameters[key] for key in
                                     ("window_length", "fft_length", "overlap_samples"))
            adjusted = window != parameters["requested_window"] or nfft != parameters["requested_fft"]
            self.stft_plot.setTitle(
                f"STFT  |  窗长 {window} 点  |  FFT {nfft} 点  |  "
                f"重叠 {overlap} 点" + (" · 自动调整" if adjusted else ""))
            self.stft_plot.setToolTip(
                f"实际窗长 {window} 点（{1000 * window / self.sample_rate:.3f} ms），"
                f"FFT {nfft} 点，重叠 {overlap} 点（{100 * overlap / window:.2f}%），"
                f"帧移 {window - overlap} 点。\n"
                f"请求窗长 {parameters['requested_window']} 点，FFT {parameters['requested_fft']} 点。\n"
                f"频点间隔 {self.sample_rate / nfft:.3f} Hz；色标为相对幅度。")
        elif mode == "cwt":
            full_cwt = payload.get("full_cwt") or self.full_cwt_cache.get(
                (id(self.samples), int(self.cwt_voices_combo.currentText())))
            f_cwt, t_cwt, values = payload["cwt"]
            reference_peak = max(float(np.max(values, initial=0)),
                                 float(np.max(full_cwt[2], initial=0)) if full_cwt is not None else 0)
            if full_cwt is not None:
                full_f_cwt, full_t_cwt, full_values = full_cwt
                self._set_spectrogram(self.cwt_overview_plot, full_t_cwt, full_f_cwt,
                                      _relative_db(full_values, reference_peak=reference_peak))
                self.cwt_overview_region.setBounds((0, self.samples.size / self.sample_rate))
                self.cwt_overview_region.setRegion(self.analysis_selection or self.overview_region.getRegion())
            self._set_spectrogram(self.cwt_plot, t_cwt, f_cwt,
                                  _relative_db(values, reference_peak=reference_peak))
            for plot in (self.cwt_overview_plot, self.cwt_plot):
                plot.image_item.setLevels((-80, 0))
                plot.color_bar.setLevels((-80, 0))
            self.cwt_plot.setTitle(
                f"CWT 选区  |  复 Morlet  |  每倍频程 {self.cwt_voices_combo.currentText()} 个频率")
        elif mode == "correlation":
            lags, values = payload["autocorrelation"]
            self.auto_curve.setData(lags / self.sample_rate, values)
            full = payload.get("full_correlation") or self.full_correlation_cache.get(
                (id(self.samples), id(self.reference)))
            if full is not None:
                full_lags, full_values = full["autocorrelation"]
                self.auto_full_curve.setData(full_lags / self.sample_rate, full_values)
            for plot, curve, marker, cross, label, origin in (
                (self.cross_full_plot, self.cross_full_curve, self.cross_full_peak_marker,
                 full["crosscorrelation"] if full else None, "完整信号", 0),
                (self.cross_plot, self.cross_curve, self.cross_peak_marker,
                 payload["crosscorrelation"], "当前选区", payload["selection_start"]),
            ):
                if cross is not None:
                    lag, values, peak, score = cross
                    curve.setData(lag / self.sample_rate, values)
                    marker.setValue(peak / self.sample_rate)
                    marker.show()
                    plot.setTitle(f"{label} · 互相关 | 时延 {peak / self.sample_rate * 1000:.2f} ms")
                    plot.setToolTip(
                        f"时延相对该输入的起点；原文件位置 {(peak + origin) / self.sample_rate:.6f} 秒，相关系数 {score:.4f}")
                else:
                    curve.setData([], [])
                    marker.hide()
                    plot.setTitle(f"{label} · 互相关 | " +
                                  ("请设置参考模板" if self.reference is None else "输入短于参考模板"))
        self._selection_readout()
        self.analysis_progress.hide()
        self.statusBar().showMessage("分析完成", 1200)

    def _show_cached_current_selection(self) -> None:
        if not self.samples.size:
            return
        start, end = self._region_samples()
        if end - start < 8:
            return
        mode = ("fft", "stft", "cwt", "correlation")[self.tabs.currentIndex()]
        window_name = self._fft_window_name()
        result_key = self._selection_result_key(mode, start, end, window_name)
        payload = self.selection_cache.get(result_key)
        if payload is None:
            return
        self.selection_cache.move_to_end(result_key)
        payload = dict(payload)
        payload["full_spectrum"] = self.full_spectrum_cache.get((id(self.samples), window_name))
        payload["full_cwt"] = self.full_cwt_cache.get(
            (id(self.samples), int(self.cwt_voices_combo.currentText())))
        payload["full_correlation"] = self.full_correlation_cache.get((id(self.samples), id(self.reference)))
        payload["full_comparison_spectrum"] = self.comparison_spectrum_cache.get(self._comparison_spectrum_key())
        self._analysis_ready(self.generation, payload, from_cache=True)

    @staticmethod
    def _spectrogram_rect(time_axis: np.ndarray, frequency: np.ndarray) -> QtCore.QRectF:
        dt = float(np.median(np.diff(time_axis)))
        df = float(np.median(np.diff(frequency)))
        return QtCore.QRectF(float(time_axis[0] - dt / 2),
                             float(frequency[0] - df / 2),
                             float(time_axis[-1] - time_axis[0] + dt),
                             float(frequency[-1] - frequency[0] + df))

    @staticmethod
    def _set_spectrogram(plot: pg.PlotWidget, time_axis: np.ndarray,
                         frequency: np.ndarray, values: np.ndarray) -> None:
        image = plot.image_item
        values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=-120.0,
                               neginf=-120.0, posinf=0.0)
        if values.size == 0 or time_axis.size == 0 or frequency.size == 0:
            return
        image.setImage(values, autoLevels=False)
        if time_axis.size > 1 and frequency.size > 1:
            image.setRect(SignalAnalyzer._spectrogram_rect(time_axis, frequency))
        low, high = _finite_levels(values)
        image.setLevels((float(low), float(high)))
        plot.color_bar.setLevels((float(low), float(high)))
        plot.setXRange(float(time_axis[0]), float(time_axis[-1]), padding=0)
        plot.setYRange(float(frequency[0]), float(frequency[-1]), padding=0)

    @QtCore.Slot(int, str)
    def _analysis_failed(self, generation: int, message: str) -> None:
        self.worker_cancel_events.pop(generation, None)
        key_spectrum, key_cwt = self.pending_analysis.pop(generation, (None, None))
        if key_spectrum is not None:
            self.pending_full_spectrum.discard(key_spectrum)
        if key_cwt is not None:
            self.pending_full_cwt.discard(key_cwt)
        if generation == self.generation:
            self.analysis_progress.hide()
            self.statusBar().showMessage(f"分析错误：{message}", 5000)

    @QtCore.Slot(int)
    def _analysis_cancelled(self, generation: int) -> None:
        if generation == self.generation:
            self.analysis_progress.hide()
            self.statusBar().showMessage("分析已取消", 1500)
        self.worker_cancel_events.pop(generation, None)
        key_spectrum, key_cwt = self.pending_analysis.pop(generation, (None, None))
        if key_spectrum is not None:
            self.pending_full_spectrum.discard(key_spectrum)
        if key_cwt is not None:
            self.pending_full_cwt.discard(key_cwt)

    def play_audio(self) -> None:
        if self.audio_path and self.samples.size:
            self._start_playback(0, self.samples.size, "signalscope_playback.wav")

    def play_selection(self) -> None:
        if not self.samples.size:
            return
        start, end = self._region_samples()
        if end <= start:
            return
        self._start_playback(start, end, "signalscope_selection.wav")

    @staticmethod
    def _playback_line(plot):
        line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#f1bd77", width=1.5))
        line.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        line.setZValue(20)
        plot.addItem(line, ignoreBounds=True)
        line.hide()
        return line

    def _set_playback_cursor(self, seconds: float) -> None:
        duration = self.samples.size / self.sample_rate if self.sample_rate else 0.0
        seconds = float(np.clip(seconds, 0, duration))
        for line in (self.playback_cursor, self.overview_playback_cursor, self.comparison_playback_cursor):
            line.setValue(seconds)
            line.setVisible(bool(self.samples.size))
        self.playback_readout.setText(f"播放位置  {seconds:.3f} / {duration:.3f} 秒")

    def _start_playback(self, start: int, end: int, filename: str) -> None:
        self.stop_playback()
        # Reusing a temporary WAV URL otherwise keeps the previous decoded media.
        self.player.setSource(QtCore.QUrl())
        self.playback_offset_s = start / self.sample_rate
        self.playback_end_s = end / self.sample_rate
        path = Path(tempfile.gettempdir()) / filename
        write_audio(str(path), self.samples[start:end], self.sample_rate)
        self.player.setSource(QtCore.QUrl.fromLocalFile(str(path)))
        self._set_playback_cursor(self.playback_offset_s)
        self._tracking_playback = True
        self.player.play()

    def stop_playback(self) -> None:
        self._tracking_playback = False
        self.player.stop()

    @QtCore.Slot(int)
    def _playback_position_changed(self, position_ms: int) -> None:
        if self._tracking_playback:
            self._set_playback_cursor(min(self.playback_end_s,
                                          self.playback_offset_s + position_ms / 1000))

    @QtCore.Slot(QMediaPlayer.MediaStatus)
    def _playback_media_status_changed(self, status) -> None:
        if self._tracking_playback and status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._tracking_playback = False
            self._set_playback_cursor(self.playback_end_s)

    def export_selection(self) -> None:
        if not self.samples.size:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出选区", "selection.wav", "WAV 音频 (*.wav)")
        if path:
            start, end = self._region_samples()
            write_audio(path, self.samples[start:end], self.sample_rate)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._closing = True
        self.navigation_analysis_timer.stop()
        self.analysis_debounce.stop()
        self.stop_playback()
        for cancel_event in self.worker_cancel_events.values():
            cancel_event.set()
        for pool in (self.thread_pool, self.interactive_thread_pool):
            pool.clear()
            pool.waitForDone()
        super().closeEvent(event)

    def export_current_plot(self) -> None:
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出当前图像", "analysis.png", "PNG 图像 (*.png);;SVG 矢量图 (*.svg)")
        if path:
            current = self.tabs.currentWidget()
            if not isinstance(current, pg.PlotWidget):
                if path.lower().endswith(".svg") or "SVG" in selected_filter:
                    path = str(Path(path).with_suffix(".svg"))
                    generator = QSvgGenerator()
                    generator.setFileName(path)
                    generator.setSize(current.size())
                    generator.setViewBox(current.rect())
                    generator.setTitle("SignalScope 完整信号与当前选区")
                    painter = QtGui.QPainter(generator)
                    try:
                        painter.fillRect(current.rect(), QtGui.QColor("#171b1f"))
                        if self.tabs.currentIndex() == 0:
                            plots = (self.fft_full_plot, self.fft_plot)
                        elif self.tabs.currentIndex() == 2:
                            plots = (self.cwt_overview_plot, self.cwt_plot)
                        elif self.correlation_tabs.currentIndex() == 0:
                            plots = (self.auto_full_plot, self.auto_plot)
                        else:
                            plots = (self.cross_full_plot, self.cross_plot)
                        for plot in plots:
                            position = plot.mapTo(current, QtCore.QPoint())
                            target = QtCore.QRectF(position.x(), position.y(),
                                                   plot.width(), plot.height())
                            # Render each graphics scene directly. QWidget.render
                            # uses the viewport's raster buffer for QGraphicsView.
                            cached_items = [(item, item.cacheMode()) for item in plot.scene().items()]
                            for item, _ in cached_items:
                                item.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.NoCache)
                            painter.save()
                            try:
                                painter.setClipRect(target)
                                plot.scene().render(painter, target, plot.plotItem.sceneBoundingRect(),
                                                    QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
                            finally:
                                painter.restore()
                                for item, cache_mode in cached_items:
                                    item.setCacheMode(cache_mode)
                    finally:
                        painter.end()
                else:
                    current.grab().save(str(Path(path).with_suffix(".png")))
                return
            plot = self._plot_by_tab.get(current)
            if self.tabs.currentIndex() == 3:
                plot = self.auto_plot if self.correlation_tabs.currentIndex() == 0 else self.cross_plot
            if plot is None:
                plot = self.cross_plot
            if path.lower().endswith(".svg") or "SVG" in selected_filter:
                path = str(Path(path).with_suffix(".svg"))
                scene_rect = plot.plotItem.sceneBoundingRect()
                size = QtCore.QSize(max(1, round(scene_rect.width())),
                                    max(1, round(scene_rect.height())))
                generator = QSvgGenerator()
                generator.setFileName(path)
                generator.setSize(size)
                generator.setViewBox(QtCore.QRect(QtCore.QPoint(0, 0), size))
                generator.setTitle("SignalScope 信号分析图")
                painter = QtGui.QPainter(generator)
                try:
                    plot.scene().render(
                        painter,
                        QtCore.QRectF(0, 0, size.width(), size.height()),
                        scene_rect,
                        QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                    )
                finally:
                    painter.end()
            else:
                path = str(Path(path).with_suffix(".png"))
                pg.exporters.ImageExporter(plot.plotItem).export(path)


def main() -> int:
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SignalScope")
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
    window = SignalAnalyzer()
    if "--verify-dir" in sys.argv:
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("audio", nargs="?")
    parser.add_argument("--screenshot")
    parser.add_argument("--tab")
    parser.add_argument("--selection", nargs=2, type=float, metavar=("START", "END"))
    parser.add_argument("--set-reference", action="store_true")
    parser.add_argument("--stft-window")
    parser.add_argument("--stft-fft")
    parser.add_argument("--stft-overlap")
    parser.add_argument("--cwt-voices")
    parser.add_argument("--verify-dir", type=Path, help=argparse.SUPPRESS)
    args, _ = parser.parse_known_args(sys.argv[1:])
    for argument, widget in ((args.stft_window, window.stft_window_combo),
                             (args.stft_fft, window.stft_fft_combo),
                             (args.stft_overlap, window.stft_overlap_combo),
                             (args.cwt_voices, window.cwt_voices_combo)):
        if argument and widget.findText(argument) >= 0:
            widget.setCurrentText(argument)
    if args.tab:
        tabs = {"FFT": 0, "STFT": 1, "CWT": 2, "Correlation": 3,
                "频谱": 0, "时频": 1, "小波": 2, "相关": 3}
        window.tabs.setCurrentIndex(tabs.get(args.tab, 0))
    if args.audio and Path(args.audio).is_file():
        def load_initial_audio() -> None:
            window.load_path(args.audio)
            if args.selection:
                window._set_region((max(0.0, args.selection[0]),
                                    min(window.samples.size / window.sample_rate, args.selection[1])))
                window._set_analysis_selection(*window.overview_region.getRegion())
                window._selection_readout()
                window._request_analysis()
            if args.set_reference:
                window.set_selection_reference()
        QtCore.QTimer.singleShot(0, load_initial_audio)
    if args.screenshot:
        def capture() -> None:
            if window.generation and window.statusBar().currentMessage() == "分析完成":
                window.grab().save(args.screenshot)
                app.quit()
            else:
                QtCore.QTimer.singleShot(300, capture)
        QtCore.QTimer.singleShot(1000, capture)
    if args.verify_dir:
        from task1.analyzer.verification import verify_window
        QtCore.QTimer.singleShot(700, lambda: verify_window(window, args.verify_dir))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
