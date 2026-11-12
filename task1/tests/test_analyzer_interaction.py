import threading

import numpy as np
import pytest
import soundfile as sf
from PySide6 import QtCore

from task1.analyzer.app import AnalysisWorker, SignalAnalyzer, _envelope
from shared.signal_core import AnalysisCancelled, cwt_view


@pytest.mark.parametrize("length,kind", [(10003, "constant"), (10003, "offset"),
                                         (10000, "constant"), (17, "offset")])
def test_envelope_uses_only_real_samples_in_final_bucket(length, kind):
    samples = np.full(length, .4) if kind == "constant" else .4 + .1 * np.sin(np.arange(length))
    indices, values = _envelope(samples, maximum_points=73)
    assert np.min(values) >= np.min(samples)
    assert np.max(values) <= np.max(samples)
    assert indices.size == values.size and np.all(indices < length)
    if length > 146:
        bucket = int(np.ceil(length / 73))
        assert values[-2] == np.min(samples[(length - 1) // bucket * bucket:])
        assert values[-1] == np.max(samples[(length - 1) // bucket * bucket:])


def run_worker(window, mode, selection):
    worker = AnalysisWorker(window.generation, window.samples, window.sample_rate, selection,
                            None, "hann", None, int(window.stft_window_combo.currentText()),
                            int(window.stft_fft_combo.currentText()), .75, 8, None, mode=mode)
    results, errors = [], []
    worker.signals.ready.connect(lambda generation, payload: results.append(payload))
    worker.signals.failed.connect(lambda generation, error: errors.append(error))
    worker.run()
    assert not errors and len(results) == 1
    return results[0]


def test_fft_selection_updates_keep_manual_range_and_fit_resets_it(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, .2 * np.sin(2 * np.pi * 2000 * np.arange(32000) / 16000), 16000)
    window = SignalAnalyzer()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    try:
        window.load_path(str(path))
        window.fft_plot.setXRange(*np.log10((1500, 2500)), padding=0)
        for selection in ((0, 16000), (1000, 12000)):
            window._set_analysis_selection(selection[0] / 16000, selection[1] / 16000)
            payload = run_worker(window, "fft", selection)
            window._analysis_ready(window.generation, payload)
            assert np.allclose(window.fft_plot.getViewBox().viewRange()[0], np.log10((1500, 2500)))
        window.fft_fit_button.click()
        assert np.allclose(window.fft_plot.getViewBox().viewRange()[0], np.log10((20, 8000)))
        sf.write(path, np.ones(44100), 44100)
        window.load_path(str(path))
        assert np.allclose(window.fft_plot.getViewBox().viewRange()[0], np.log10((20, 22050)))
    finally:
        window.close()


@pytest.mark.parametrize("duration_ms,expected_window", [(20, 80), (40, 160), (100, 400), (500, 512)])
def test_stft_reports_actual_execution_parameters_and_frame_coordinates(duration_ms, expected_window):
    window = SignalAnalyzer()
    window.samples = np.sin(2 * np.pi * 2000 * np.arange(16000) / 16000)
    window.sample_rate = 16000
    window.stft_window_combo.setCurrentText("512")
    window.stft_fft_combo.setCurrentText("512")
    try:
        start, end = 1600, 1600 + duration_ms * 16
        payload = run_worker(window, "stft", (start, end))
        parameters = payload["stft_parameters"]
        assert parameters["window_length"] == expected_window
        assert parameters["fft_length"] == 512
        assert parameters["overlap_samples"] == round(expected_window * .75)
        frequency, times, spectrum = payload["stft"]
        hop = expected_window - parameters["overlap_samples"]
        assert np.allclose(np.diff(frequency), 16000 / 512)
        assert np.isclose(times[0], (start + expected_window / 2) / 16000)
        assert np.allclose(np.diff(times), hop / 16000)
        window._analysis_ready(window.generation, payload)
        assert f"窗长 {expected_window} 点" in window.stft_plot.plotItem.titleLabel.text
        assert f"重叠 {parameters['overlap_samples']} 点" in window.stft_plot.toolTip()
    finally:
        window.close()


def test_cwt_progress_counts_real_blocks_and_can_cancel_at_a_boundary():
    samples = np.sin(2 * np.pi * 1000 * np.arange(80000) / 8000)
    progress = []
    cwt_view(samples, 8000, maximum_points=64, voices_per_octave=4,
             frequency_limits=(500, 3000), progress_callback=lambda done, total: progress.append((done, total)))
    assert progress[0] == (0, 2) and progress[-1] == (2, 2)
    assert progress == sorted(progress)
    event = threading.Event()
    def cancel(done, total):
        if done == 1:
            event.set()
    with pytest.raises(AnalysisCancelled):
        cwt_view(samples, 8000, maximum_points=64, voices_per_octave=4,
                 frequency_limits=(500, 3000), cancel_event=event, progress_callback=cancel)


def test_stale_cwt_progress_does_not_reappear_after_tab_switch():
    window = SignalAnalyzer()
    window.samples = np.ones(16000)
    window.sample_rate = 16000
    try:
        window.tabs.setCurrentIndex(2)
        generation = window.generation
        window._analysis_progress(generation, 1, 4, "完整信号")
        assert not window.analysis_progress.isHidden()
        assert window.analysis_progress.value() == 25
        window.tabs.setCurrentIndex(0)
        window._analysis_progress(generation, 3, 4, "完整信号")
        assert window.analysis_progress.isHidden()
        assert "CWT" not in window.statusBar().currentMessage()
    finally:
        window.close()


def test_selection_has_distinct_style_and_clear_control(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, np.zeros(16000), 16000)
    window = SignalAnalyzer()
    try:
        window.load_path(str(path))
        window._set_region((.1, .8))
        window._set_analysis_selection(.2, .4)
        assert window.selection_region.brush.color() != window.overview_region.brush.color()
        assert "分析选区" in window.readout.text()
        assert window.visible_range_button.isEnabled()
        window.visible_range_button.click()
        assert window.analysis_selection is None
        assert not window.selection_region.isVisible()
        assert window._region_samples() == (1600, 12800)
        assert "跟随浏览" in window.readout.text()
    finally:
        window.close()


def test_comparison_resamples_and_uses_explicit_time_offset(tmp_path):
    reference = tmp_path / "reference.wav"
    current = tmp_path / "current.wav"
    samples = np.zeros(8000)
    samples[1600] = .5
    sf.write(reference, samples, 8000)
    sf.write(current, np.zeros(16000), 16000)
    window = SignalAnalyzer()
    try:
        window.load_path(str(reference))
        window.load_path(str(current))
        window.load_comparison_path(str(reference))
        window._set_region((0, 1))
        assert window.comparison_samples.size == 16000
        window.comparison_offset.setValue(.1)
        x, y = window.comparison_curve.getData()
        assert abs(x[np.argmax(y)] - .3) <= 1 / 16000
        assert "reference.wav" in window.comparison_label.toolTip()
        assert "current.wav" in window.current_signal_label.toolTip()
        window.comparison_offset.setValue(-.1)
        x, y = window.comparison_curve.getData()
        assert abs(x[np.argmax(y)] - .1) <= 1 / 16000
        window.overlay_check.setChecked(False)
        assert not window.comparison_curve.isVisible()
    finally:
        window.close()


def test_loading_new_audio_clears_previous_spectrogram_and_fft(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, np.zeros(32000), 16000)
    window = SignalAnalyzer()
    try:
        window.samples = np.ones(96000)
        window.sample_rate = 48000
        window.cwt_plot.image_item.setImage(np.ones((32, 100)))
        window.stft_plot.image_item.setImage(np.ones((32, 100)))
        window.fft_selected_curve.setData([20, 100], [-20, -30])
        window.load_path(str(path), remember_previous=False)
        assert window.cwt_plot.image_item.image is None
        assert window.stft_plot.image_item.image is None
        assert window.fft_selected_curve.getData()[0] is None
        assert "等待分析" in window.cwt_plot.plotItem.titleLabel.text
        assert np.allclose(window.cwt_plot.getViewBox().viewRange()[1], (0, 8000))
    finally:
        window.close()
