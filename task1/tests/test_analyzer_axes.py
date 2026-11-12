import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from task1.analyzer.axes import FrequencyAxis, TimeAxis, format_frequency
from task1.analyzer.app import SignalAnalyzer, _relative_db


def test_frequency_labels_preserve_values_and_use_readable_units():
    assert [format_frequency(v) for v in (0, 20, 500, 1000, 1500, 20000)] == [
        "0", "20", "500", "1k", "1.5k", "20k"]
    assert format_frequency(316.228) == "316.228"


def test_log_ticks_follow_true_frequency_positions_and_adapt_to_width():
    axis = FrequencyAxis(log_frequency=True)
    wide = axis.tickValues(np.log10(20), np.log10(20000), 1000)[0][1]
    narrow = axis.tickValues(np.log10(20), np.log10(20000), 220)[0][1]
    labels = axis.tickStrings(wide, 1, 0)
    assert {"20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"} <= set(labels)
    assert len(narrow) < len(wide)
    assert len(set(labels)) == len(labels)
    assert all(abs(10 ** v - float(label.removesuffix("k")) * (1000 if label.endswith("k") else 1))
               < 1e-5 for v, label in zip(wide, labels))
    zoomed = axis.tickValues(np.log10(1700), np.log10(1900), 700)[0][1]
    assert len(zoomed) >= 3
    assert all(1700 - 1e-7 <= 10 ** v <= 1900 + 1e-7 for v in zoomed)


@pytest.mark.parametrize("sample_rate", [16000, 44100, 48000])
def test_frequency_modes_switch_and_respect_nyquist(sample_rate):
    window = SignalAnalyzer()
    window.samples = np.zeros(sample_rate)
    window.sample_rate = sample_rate
    window._fft_selected_raw = (np.arange(0, sample_rate / 2 + 1, 100), np.zeros(int(sample_rate / 200) + 1))
    # The last length can differ for a noninteger Nyquist/100.
    window._fft_selected_raw = (window._fft_selected_raw[0], np.zeros(window._fft_selected_raw[0].size))
    window._fft_full_raw = window._fft_selected_raw
    try:
        for mode in ("线性", "对数", "线性", "对数"):
            window.frequency_scale.setCurrentText(mode)
            axis = window.fft_plot.getAxis("bottom")
            assert axis._tickLevels is None
            low, high = window.fft_plot.getViewBox().viewRange()[0]
            ticks = axis.tickValues(low, high, 600)[0][1]
            hz = 10 ** np.asarray(ticks) if axis.log_frequency else np.asarray(ticks)
            assert hz.size and np.max(hz) <= sample_rate / 2 + 1e-5
            if mode == "线性":
                assert "0" in axis.tickStrings(ticks, 1, 0)
        axis.set_frequency_scale(False)
        ticks = axis.tickValues(0, 2000, 500)[0][1]
        assert ticks == [0, 500, 1000, 1500, 2000]
    finally:
        window.close()


def test_time_axis_uses_visible_span_and_retains_seconds_coordinates():
    axis = TimeAxis()
    axis.setRange(60.0, 60.2)
    assert axis.labelUnits == ""
    assert axis.tickStrings([60.1], 1, 0.01) == ["01:00.100"]
    ticks = axis.tickValues(60.0, 60.2, 600)[0][1]
    assert all(60 <= value <= 60.2 for value in ticks)
    axis.setRange(0, 180)
    assert axis.tickStrings([60], 1, 10) == ["01:00"]
    axis.setRange(.1, .3)
    assert axis.labelUnits == "ms"
    assert axis.tickStrings([.2], 1, .01) == ["200"]
    axis.setRange(20, 20.2)
    assert axis.labelUnits == "s"
    assert axis.tickStrings([20.1], 1, .01) == ["20.1"]


@pytest.mark.parametrize("logarithmic", [False, True])
def test_zoom_past_nyquist_keeps_tick_labels_spaced(logarithmic):
    axis = FrequencyAxis(log_frequency=logarithmic)
    axis.maximum_frequency = 8000
    low, high = (np.log10(20), np.log10(80000)) if logarithmic else (0, 80000)
    for width in (180, 400, 900):
        ticks = axis.tickValues(low, high, width)[0][1]
        positions = (np.asarray(ticks) - low) / (high - low) * width
        labels = axis.tickStrings(ticks, 1, 0)
        for index in range(len(ticks) - 1):
            required = (axis._minimum_distance([labels[index]])
                        + axis._minimum_distance([labels[index + 1]])) / 2
            assert positions[index + 1] - positions[index] >= required - 1e-7


def test_spectrogram_units_match_peak_relative_amplitudes():
    window = SignalAnalyzer()
    try:
        for plot in (window.stft_plot, window.cwt_plot, window.cwt_overview_plot):
            assert plot.color_bar.getAxis("left").labelText == "相对幅度 (dB)"
            assert isinstance(plot.getAxis("left"), FrequencyAxis)
            assert isinstance(plot.getAxis("bottom"), TimeAxis)
        values = np.array([[1.0, 0.1], [0.01, 0.001]])
        assert np.allclose(_relative_db(values), _relative_db(values / 10))
        assert np.max(_relative_db(values)) == 0
        assert window.fft_plot.getAxis("left").labelUnits == "dBFS"
        assert window.fft_plot.getAxis("bottom").style["tickFont"].pointSizeF() >= 9
    finally:
        window.close()


@pytest.mark.parametrize("sample_rate", [16000, 44100, 48000])
def test_stft_cwt_tone_rows_match_frequency_axis_at_audio_rates(sample_rate):
    from shared.signal_core import cwt_view, stft_view
    time = np.arange(round(.25 * sample_rate)) / sample_rate
    samples = .3 * np.sin(2 * np.pi * 2500 * time)
    frequency, times, z = stft_view(samples, sample_rate, window_length=512,
                                   overlap=.75, n_fft=2048)
    cwt_f, cwt_t, magnitude = cwt_view(samples, sample_rate, maximum_points=512)
    window = SignalAnalyzer()
    try:
        for plot, f, t, values, tolerance in (
            (window.stft_plot, frequency, times, np.abs(z), sample_rate / 2048),
            (window.cwt_plot, cwt_f, cwt_t, magnitude, 200),
        ):
            row = int(np.argmax(values[:, values.shape[1] // 4:3 * values.shape[1] // 4].mean(axis=1)))
            assert abs(f[row] - 2500) < tolerance
            window._set_spectrogram(plot, t, f, _relative_db(values))
            mapped = plot.image_item.mapToParent(QtCore.QPointF(1.5, row + .5))
            assert np.isclose(mapped.y(), f[row], atol=1e-7)
            axis = plot.getAxis("left")
            assert axis.tickStrings([1000, 2000, 5000], 1, 1000) == ["1k", "2k", "5k"]
    finally:
        window.close()


def test_playback_cursor_tracks_selection_stop_and_end_without_reanalysis(tmp_path):
    import soundfile as sf
    from PySide6.QtMultimedia import QMediaPlayer
    window = SignalAnalyzer()
    path = tmp_path / "tone.wav"
    sf.write(path, np.sin(2 * np.pi * 1000 * np.arange(32000) / 16000) * .1, 16000)
    try:
        window.load_path(str(path))
        window._set_region((0.2, 1.5))
        window._set_analysis_selection(0.4, 1.2)
        generation = window.generation
        region = window.overview_region.getRegion()
        window.play_selection()
        window._playback_position_changed(125)
        assert np.isclose(window.playback_cursor.value(), .525)
        assert np.isclose(window.overview_playback_cursor.value(), .525)
        window.stop_playback()
        window._playback_position_changed(0)
        assert np.isclose(window.playback_cursor.value(), .525)
        window._playback_media_status_changed(QMediaPlayer.MediaStatus.EndOfMedia)
        assert np.isclose(window.playback_cursor.value(), .525)
        window.play_selection()
        window._playback_media_status_changed(QMediaPlayer.MediaStatus.EndOfMedia)
        assert np.isclose(window.playback_cursor.value(), 1.2)
        assert window.generation == generation
        assert window.overview_region.getRegion() == region
        assert window._region_samples() == (6400, 19200)
        window.play_audio()
        assert window.playback_cursor.value() == 0
        window.stop_playback()
        window.load_path(str(path))
        assert window.playback_cursor.value() == 0
    finally:
        window.close()


def test_hover_frequency_retains_full_precision():
    window = SignalAnalyzer()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    window.sample_rate = 16000
    window.samples = np.ones(1600)
    window.fft_plot.setXRange(np.log10(20), np.log10(8000), padding=0)
    QtWidgets.QApplication.processEvents()
    try:
        for plot in (window.fft_plot, window.fft_full_plot):
            scene = plot.getViewBox().mapViewToScene(QtCore.QPointF(np.log10(1777.32), -20))
            plot.scene().sigMouseMoved.emit(scene)
            assert "1,777.32 Hz" in window.statusBar().currentMessage()
    finally:
        window.close()


def test_replaying_a_different_selection_reloads_wav_duration(tmp_path):
    import soundfile as sf
    window = SignalAnalyzer()
    path = tmp_path / "changing_selection.wav"
    sf.write(path, .1 * np.sin(2 * np.pi * 700 * np.arange(32000) / 16000), 16000)

    def wait_for_duration(expected):
        loop = QtCore.QEventLoop()
        timeout = QtCore.QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        def changed(duration):
            if duration == expected:
                loop.quit()
        window.player.durationChanged.connect(changed)
        try:
            timeout.start(4000)
            if window.player.duration() != expected:
                loop.exec()
            assert window.player.duration() == expected
        finally:
            window.player.durationChanged.disconnect(changed)
            timeout.stop()
    try:
        window.load_path(str(path))
        window._set_analysis_selection(.2, 1.2)
        window.play_selection()
        wait_for_duration(1000)
        window._set_analysis_selection(.2, .55)
        window.play_selection()
        wait_for_duration(350)
    finally:
        window.close()
