import numpy as np
import pytest
import soundfile as sf
from PySide6 import QtCore, QtTest, QtWidgets

from task1.analyzer.app import SignalAnalyzer
from shared.audio_comparison import aligned_audio_segment


def wait_fft(window):
    app = QtWidgets.QApplication.instance()
    for _ in range(500):
        app.processEvents()
        QtTest.QTest.qWait(5)
        key = window._selection_result_key("fft", *window._region_samples(), window._fft_window_name())
        if key in window.selection_cache and not window.analysis_debounce.isActive():
            return window.selection_cache[key]
    raise AssertionError(window.statusBar().currentMessage())


@pytest.mark.parametrize("offset,expected", [(0, [2, 3, 4, 0, 0]),
                                           (2, [0, 1, 2, 3, 4]),
                                           (-2, [4, 0, 0, 0, 0]),
                                           (20, [0, 0, 0, 0, 0])])
def test_comparison_segment_time_grid_and_missing_samples(offset, expected):
    actual, valid = aligned_audio_segment(np.arange(5), 2, 7, offset_samples=offset)
    assert np.array_equal(actual, expected)
    assert valid == {0: 3, 2: 5, -2: 1, 20: 0}[offset]


def test_opening_main_files_never_automatically_enables_comparison(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"signal{i}.wav"
        sf.write(path, np.full(16000, .1 * i), 16000)
        paths.append(path)
    window = SignalAnalyzer()
    try:
        window.load_path(str(paths[0]))
        window.load_path(str(paths[1]))
        assert window.comparison_samples is None
        assert not window.overlay_check.isChecked()
        window.load_comparison_path(str(paths[0]))
        assert window.comparison_path == paths[0] and window.overlay_check.isChecked()
        window.load_path(str(paths[2]))
        assert not window.overlay_check.isChecked()
        assert window.comparison_path == paths[0]
        assert not window.comparison_curve.isVisible()
        window.load_comparison_path(str(paths[1]))
        assert window.comparison_path == paths[1]
    finally:
        window.close()


@pytest.mark.parametrize("channel,index", [("left", 0), ("right", 1), ("mix", None)])
def test_both_stereo_files_follow_channel_and_resampling(tmp_path, channel, index):
    primary = tmp_path / "primary.wav"
    compare = tmp_path / "compare.wav"
    main = np.column_stack((np.full(16000, .2), np.full(16000, -.4)))
    other = np.column_stack((np.full(8000, .6), np.full(8000, -.2)))
    sf.write(primary, main, 16000, subtype="FLOAT")
    sf.write(compare, other, 8000, subtype="FLOAT")
    window = SignalAnalyzer()
    try:
        window.load_path(str(primary))
        window.load_comparison_path(str(compare))
        window._set_region((.1, .8))
        window._set_analysis_selection(.2, .4)
        window.channel_combo.setCurrentIndex(window.channel_combo.findData(channel))
        main_value = main.mean(axis=1)[0] if index is None else main[0, index]
        compare_value = other.mean(axis=1)[0] if index is None else other[0, index]
        assert np.isclose(window.samples[1000], main_value)
        assert np.isclose(np.mean(window.comparison_samples[100:-100]), compare_value, atol=1e-4)
        assert window.comparison_samples.size == 16000
        assert window.overlay_check.isChecked()
        assert window._region_samples() == (3200, 6400)
        assert np.allclose(window.overview_region.getRegion(), (.1, .8))
    finally:
        window.close()


@pytest.mark.parametrize("offset", [-.1, 0, .1])
def test_fft_comparison_uses_same_interval_window_and_amplitude_reference(tmp_path, offset):
    fs = 16000
    main = np.zeros(fs)
    other = np.zeros(fs // 2)
    main[3200:6400] = .4 * np.sin(2 * np.pi * 2000 * np.arange(3200) / fs)
    other[3200:6400] = main[3200:6400] / 2
    p, c = tmp_path / "main.wav", tmp_path / "other.wav"
    sf.write(p, main, fs, subtype="FLOAT")
    sf.write(c, other, fs, subtype="FLOAT")
    window = SignalAnalyzer()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    try:
        window.load_path(str(p))
        window.load_comparison_path(str(c))
        window._set_analysis_selection(.2, .4)
        window.comparison_offset.setValue(offset)
        payload = wait_fft(window)
        frequency, db, valid = payload["comparison_spectrum"]
        from shared.signal_core import one_sided_spectrum
        aligned, expected_valid = aligned_audio_segment(
            window.comparison_samples, 3200, 6400, offset_samples=round(offset * fs))
        expected = one_sided_spectrum(aligned, fs, n_fft=payload["spectrum"][2], window="hann")
        assert np.array_equal(frequency, payload["spectrum"][0])
        assert np.allclose(db, expected.dbfs) and valid == expected_valid
        if offset == 0:
            k = np.argmin(np.abs(frequency - 2000))
            assert np.isclose(payload["spectrum"][1][k] - db[k], 20 * np.log10(2), atol=1e-6)
        generation = window.generation
        window.comparison_mode.setCurrentText("上下分轨")
        QtWidgets.QApplication.processEvents()
        assert window.generation == generation
        assert window.comparison_waveform.isVisible()
        assert not window.comparison_curve.isVisible()
        assert np.allclose(window.comparison_waveform.getViewBox().viewRange()[0],
                           window.waveform.getViewBox().viewRange()[0])
        window.fft_comparison_check.setChecked(False)
        assert not window.fft_comparison_selected_curve.isVisible()
        window.overlay_check.setChecked(False)
        assert not window.comparison_waveform.isVisible()
        assert window._region_samples() == (3200, 6400)
    finally:
        window.close()


def test_locate_selection_keeps_analysis_bounds(tmp_path):
    path = tmp_path / "main.wav"
    sf.write(path, np.zeros(16000 * 4), 16000)
    window = SignalAnalyzer()
    try:
        window.load_path(str(path))
        window._set_analysis_selection(.2, .4)
        window._set_region((2, 3))
        window.locate_selection_button.click()
        left, right = window.overview_region.getRegion()
        assert left <= .2 < .4 <= right
        assert window._region_samples() == (3200, 6400)
        assert "浏览" in window.waveform.plotItem.titleLabel.text
        assert "分析" in window.waveform.plotItem.titleLabel.text
    finally:
        window.close()


def test_compact_split_tracks_have_real_drawing_area_and_no_overlap(tmp_path):
    path = tmp_path / "主信号_含噪音频_较长的文件名称_16000.wav"
    sf.write(path, np.full(16000, .2), 16000)
    window = SignalAnalyzer()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.resize(900, 640)
    window.show()
    try:
        window.load_path(str(path))
        window.load_comparison_path(str(path))
        window.comparison_mode.setCurrentText("上下分轨")
        QtWidgets.QApplication.processEvents()
        assert window.width() == 900 and window.height() == 640
        assert window.waveform.getViewBox().height() >= 120
        assert window.comparison_waveform.getViewBox().height() >= 120
        assert window.overview.isHidden()
        assert window.waveform.geometry().bottom() < window.comparison_waveform.geometry().top()
        assert window.fft_plot.getViewBox().height() >= 150
        assert window.metadata_label.fontMetrics().horizontalAdvance(window.metadata_label.text()) <= window.metadata_label.contentsRect().width()
        assert window.metadata_label.toolTip() == str(path)
        assert window.technical_info_label.text() == "16,000 Hz  |  1.000 秒  |  1 声道"
        assert window.technical_info_label.width() >= window.technical_info_label.sizeHint().width()
        assert "Hz" not in window.metadata_label.text()
        window.overview_toggle.click()
        QtWidgets.QApplication.processEvents()
        assert window.overview.isVisible()
        assert window.overview.getViewBox().height() >= 70
        assert window.comparison_waveform.geometry().bottom() < window.overview.parentWidget().geometry().top()
        window.content_scroll.ensureWidgetVisible(window.analysis_section)
        QtWidgets.QApplication.processEvents()
        assert window.content_scroll.verticalScrollBar().maximum() > 0
    finally:
        window.close()


def test_split_track_drag_pan_and_zoom_keep_main_analysis_semantics(tmp_path, monkeypatch):
    fs = 16000
    main, other = tmp_path / "main.wav", tmp_path / "other.wav"
    sf.write(main, np.full(fs * 2, .25), fs, subtype="FLOAT")
    sf.write(other, np.full(fs, -.75), fs, subtype="FLOAT")
    window = SignalAnalyzer()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.resize(1440, 1000)
    window.show()
    try:
        window.load_path(str(main))
        window.load_comparison_path(str(other))
        window.comparison_mode.setCurrentText("上下分轨")
        window._set_region((.1, 1.1))
        QtWidgets.QApplication.processEvents()
        plot = window.comparison_waveform
        view = plot.getViewBox()

        def drag(left, right):
            start = plot.mapFromScene(view.mapViewToScene(QtCore.QPointF(left, 0)))
            end = plot.mapFromScene(view.mapViewToScene(QtCore.QPointF(right, 0)))
            QtTest.QTest.mouseMove(plot.viewport(), start, 20)
            QtTest.QTest.mousePress(plot.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=start)
            QtTest.QTest.mouseMove(plot.viewport(), end, 50)
            QtTest.QTest.mouseRelease(plot.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=end)
            QtWidgets.QApplication.processEvents()

        drag(.3, .6)
        bounds = window._region_samples()
        assert np.allclose(bounds, (4800, 9600), atol=20)
        assert np.allclose(window.comparison_selection_region.getRegion(), window.analysis_selection)
        window.interaction_combo.setCurrentIndex(1)
        drag(.7, .8)
        assert not np.allclose(window.overview_region.getRegion(), (.1, 1.1))
        assert np.allclose(view.viewRange()[0], window.waveform.getViewBox().viewRange()[0])
        before = np.diff(view.viewRange()[0])[0]
        view.scaleBy((.8, 1))
        QtWidgets.QApplication.processEvents()
        assert np.isclose(np.diff(window.overview_region.getRegion())[0], .8 * before)
        assert window._region_samples() == bounds

        exported = tmp_path / "export.wav"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a: (str(exported), "WAV"))
        window.export_selection()
        data, rate = sf.read(exported)
        assert rate == fs and len(data) == bounds[1] - bounds[0]
        assert np.allclose(data, .25, atol=4e-5)
        window.audio_output.setVolume(0)
        window.play_selection()
        data, rate = sf.read(window.player.source().toLocalFile())
        assert len(data) == bounds[1] - bounds[0] and np.allclose(data, .25, atol=4e-5)
        window.stop_playback()
        window.play_audio()
        data, rate = sf.read(window.player.source().toLocalFile())
        assert rate == fs and len(data) == fs * 2 and np.allclose(data, .25, atol=4e-5)
        window.stop_playback()
        window.visible_range_button.click()
        assert window.analysis_selection is None
        assert not window.comparison_selection_region.isVisible()
    finally:
        window.close()


def test_comparison_cache_updates_on_file_offset_window_and_selection(tmp_path):
    fs = 16000
    primary, first, second = (tmp_path / name for name in ("primary.wav", "first.wav", "second.wav"))
    t = np.arange(fs) / fs
    sf.write(primary, .4 * np.sin(2 * np.pi * 1500 * t), fs, subtype="FLOAT")
    sf.write(first, .2 * np.sin(2 * np.pi * 1500 * t), fs, subtype="FLOAT")
    sf.write(second, .2 * np.sin(2 * np.pi * 2500 * t), fs, subtype="FLOAT")
    window = SignalAnalyzer()
    try:
        window.load_path(str(primary))
        window.load_comparison_path(str(first))
        window._set_analysis_selection(.2, .4)
        initial = wait_fft(window)["comparison_spectrum"]
        window.load_comparison_path(str(second))
        replaced = wait_fft(window)["comparison_spectrum"]
        assert not np.allclose(initial[1], replaced[1])
        assert replaced[0][np.argmax(replaced[1])] == 2500
        window.comparison_offset.setValue(2)
        missing = wait_fft(window)["comparison_spectrum"]
        assert missing[2] == 0 and not window.fft_comparison_selected_curve.isVisible()
        window.comparison_offset.setValue(0)
        window.window_combo.setCurrentText("海明窗")
        updated = wait_fft(window)["comparison_spectrum"]
        assert updated[2] == 3200 and window.fft_comparison_selected_curve.isVisible()
        window.visible_range_button.click()
        window._set_region((.5, .75))
        window._request_analysis()
        followed = wait_fft(window)["comparison_spectrum"]
        assert followed[2] == 4000
        assert window._region_samples() == (8000, 12000)
    finally:
        window.close()
