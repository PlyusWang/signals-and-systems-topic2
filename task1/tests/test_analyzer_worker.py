import numpy as np
import threading
from PySide6 import QtCore, QtGui, QtTest, QtWidgets
from task1.analyzer.app import AnalysisWorker, SignalAnalyzer, _relative_db
from shared.signal_core import AnalysisCancelled, cwt_view


def _run_worker(mode, samples, sample_rate=8000, selection=(800, 3200), **kwargs):
    worker = AnalysisWorker(
        generation=4,
        samples=samples,
        sample_rate=sample_rate,
        selection=selection,
        reference=None,
        fft_window="hann",
        full_spectrum=None,
        stft_window=256,
        stft_nfft=512,
        stft_overlap=0.75,
        cwt_voices=8,
        full_cwt=None,
        mode=mode,
        **kwargs,
    )
    results = []
    failures = []
    worker.signals.ready.connect(lambda generation, payload: results.append((generation, payload)))
    worker.signals.failed.connect(lambda generation, message: failures.append((generation, message)))
    worker.run()
    assert not failures
    assert len(results) == 1
    return results[0]


def test_worker_computes_only_selected_view_and_preserves_stft_time():
    sample_rate = 8000
    time = np.arange(4096) / sample_rate
    samples = np.sin(2 * np.pi * 1000 * time)
    selection = (800, 3200)
    generation, payload = _run_worker("stft", samples, sample_rate, selection)
    assert generation == 4
    stft_f, stft_t, stft_z = payload["stft"]
    assert stft_f.size == 257
    assert stft_z.shape == (stft_f.size, stft_t.size)
    assert stft_t[0] >= selection[0] / sample_rate
    assert "cwt" not in payload
    assert "full_cwt" not in payload
    assert "spectrum" not in payload
    assert "autocorrelation" not in payload


def test_worker_cwt_mode_computes_local_and_full_overview_only_when_requested():
    sample_rate = 8000
    time = np.arange(4096) / sample_rate
    samples = np.sin(2 * np.pi * 1000 * time)
    generation, payload = _run_worker("cwt", samples, sample_rate, (800, 3200),
                                      compute_full_cwt=True)
    assert generation == 4
    _, local_times, local_magnitude = payload["cwt"]
    assert local_times[0] == 800 / sample_rate
    assert local_magnitude.shape[1] == local_times.size
    _, overview_times, overview_magnitude = payload["full_cwt"]
    assert overview_times[0] == 0
    assert overview_times[-1] <= samples.size / sample_rate
    assert overview_magnitude.shape[1] == overview_times.size
    assert "spectrum" not in payload and "stft" not in payload


def test_worker_reports_cancellation_without_an_analysis_error():
    samples = np.ones(4096)
    cancel_event = threading.Event()
    cancel_event.set()
    worker = AnalysisWorker(
        generation=5, samples=samples, sample_rate=8000, selection=(0, samples.size),
        reference=None, fft_window="hann", full_spectrum=None, stft_window=256,
        stft_nfft=256, stft_overlap=0.75, cwt_voices=4, full_cwt=None,
        mode="cwt", compute_full_cwt=True, cancel_event=cancel_event)
    ready, failed, cancelled = [], [], []
    worker.signals.ready.connect(lambda *args: ready.append(args))
    worker.signals.failed.connect(lambda *args: failed.append(args))
    worker.signals.cancelled.connect(lambda *args: cancelled.append(args))
    worker.run()
    assert not ready and not failed
    assert cancelled == [(5,)]


def test_relative_db_is_finite_for_silence_and_tiny_signals():
    silence = _relative_db(np.zeros((4, 8)))
    assert np.isfinite(silence).all()
    assert np.all(silence == -120.0)
    tiny = _relative_db(np.full((4, 8), 1e-300))
    assert np.isfinite(tiny).all()
    assert np.all(tiny == 0.0)


def test_cwt_and_db_conversion_remain_finite_for_silent_tiny_and_impulse_inputs():
    sample_rate = 8000
    inputs = (np.zeros(2048), np.full(2048, 1e-20),
              np.eye(1, 2048, 1024, dtype=np.float64).reshape(-1))
    for samples in inputs:
        _, _, values = cwt_view(samples, sample_rate, voices_per_octave=4,
                                maximum_points=128, frequency_limits=(200.0, 3000.0))
        db = _relative_db(values)
        assert np.isfinite(values).all()
        assert np.isfinite(db).all()


def test_spectrogram_levels_are_finite_for_constant_values():
    from task1.analyzer.app import _finite_levels

    low, high = _finite_levels(np.zeros((2, 2)))
    assert np.isfinite((low, high)).all()
    assert high > low


def test_spectrogram_grid_places_first_and_last_samples_at_pixel_centers():
    time_axis = np.array([0.01, 0.02, 0.03])
    frequency = np.array([100.0, 200.0, 300.0])
    rect = SignalAnalyzer._spectrogram_rect(time_axis, frequency)
    assert np.isclose(rect.left() + rect.width() / 6, time_axis[0])
    assert np.isclose(rect.right() - rect.width() / 6, time_axis[-1])
    assert np.isclose(rect.top() + rect.height() / 6, frequency[0])
    assert np.isclose(rect.bottom() - rect.height() / 6, frequency[-1])


def test_overview_is_full_file_and_tracks_local_waveform_navigation():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SignalAnalyzer()
    window.samples = np.sin(2 * np.pi * np.arange(10000) / 100)
    window.sample_rate = 1000
    window.overview_region.setBounds((0, 10))
    window.cwt_overview_region.setBounds((0, 10))
    window.overview.setXRange(0, 10, padding=0)
    window._set_region((2, 4))

    local_range = window.waveform.getViewBox().viewRange()[0]
    overview_range = window.overview.getViewBox().viewRange()[0]
    assert np.allclose(local_range, (2, 4))
    assert np.allclose(overview_range, (0, 10))

    window.waveform.setXRange(5, 7, padding=0)
    app.processEvents()
    assert np.allclose(window.overview_region.getRegion(), (5, 7))
    assert np.allclose(window.overview.getViewBox().viewRange()[0], (0, 10))

    window.overview_region.setRegion((1, 3))
    app.processEvents()
    assert np.allclose(window.waveform.getViewBox().viewRange()[0], (1, 3))
    assert np.allclose(window.overview.getViewBox().viewRange()[0], (0, 10))
    assert not any(item is window.overview_region for item in window.waveform.items())
    window.close()


def test_mouse_pan_of_local_waveform_updates_overview_window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SignalAnalyzer()
    window.resize(1200, 850)
    window.samples = np.sin(2 * np.pi * np.arange(10000) / 100)
    window.sample_rate = 1000
    window.overview_region.setBounds((0, 10))
    window.cwt_overview_region.setBounds((0, 10))
    window.overview_curve.setData(np.arange(100), np.sin(np.arange(100) / 10))
    window.overview.setXRange(0, 10, padding=0)
    window._set_region((2, 4))
    window.navigation_analysis_timer.setInterval(10000)
    window.interaction_combo.setCurrentIndex(1)
    window.show()
    app.processEvents()

    view = window.waveform.getViewBox()
    viewport = window.waveform.viewport()
    start = window.waveform.mapFromScene(view.mapViewToScene(QtCore.QPointF(3.0, 0.0)))
    finish = window.waveform.mapFromScene(view.mapViewToScene(QtCore.QPointF(3.5, 0.0)))
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    QtTest.QTest.mouseMove(viewport, finish, 50)
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=finish)
    app.processEvents()

    left, right = window.overview_region.getRegion()
    assert not np.allclose((left, right), (2, 4))
    assert np.isclose(right - left, 2.0, atol=0.05)
    assert np.allclose(window.overview.getViewBox().viewRange()[0], (0, 10))

    overview_view = window.overview.getViewBox()
    overview_viewport = window.overview.viewport()
    center = 0.5 * (left + right)
    drag_start = window.overview.mapFromScene(
        overview_view.mapViewToScene(QtCore.QPointF(center, 0.0)))
    drag_finish = window.overview.mapFromScene(
        overview_view.mapViewToScene(QtCore.QPointF(center + 0.5, 0.0)))
    QtTest.QTest.mouseMove(overview_viewport, drag_start, 10)
    app.processEvents()
    QtTest.QTest.mousePress(overview_viewport, QtCore.Qt.MouseButton.LeftButton,
                            pos=drag_start)
    QtTest.QTest.mouseMove(overview_viewport, drag_finish, 50)
    QtTest.QTest.mouseRelease(overview_viewport, QtCore.Qt.MouseButton.LeftButton,
                              pos=drag_finish)
    app.processEvents()
    moved_region = window.overview_region.getRegion()
    assert not np.allclose(moved_region, (left, right))
    assert np.allclose(window.waveform.getViewBox().viewRange()[0], moved_region)

    resized_start = window.overview.mapFromScene(
        overview_view.mapViewToScene(QtCore.QPointF(moved_region[0], 0.0)))
    resized_finish = window.overview.mapFromScene(
        overview_view.mapViewToScene(QtCore.QPointF(moved_region[0] + 0.3, 0.0)))
    QtTest.QTest.mouseMove(overview_viewport, resized_start, 10)
    app.processEvents()
    QtTest.QTest.mousePress(overview_viewport, QtCore.Qt.MouseButton.LeftButton,
                            pos=resized_start)
    QtTest.QTest.mouseMove(overview_viewport, resized_finish, 50)
    QtTest.QTest.mouseRelease(overview_viewport, QtCore.Qt.MouseButton.LeftButton,
                              pos=resized_finish)
    app.processEvents()
    resized_region = window.overview_region.getRegion()
    assert abs((resized_region[1] - resized_region[0]) -
               (moved_region[1] - moved_region[0])) > 0.1
    assert np.allclose(window.waveform.getViewBox().viewRange()[0], resized_region)
    assert np.allclose(window.overview.getViewBox().viewRange()[0], (0, 10))
    window.navigation_analysis_timer.stop()
    window.close()


def test_local_drag_selects_analysis_without_moving_visible_window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SignalAnalyzer()
    window.samples = np.sin(2 * np.pi * np.arange(10000) / 100)
    window.sample_rate = 1000
    window.overview_region.setBounds((0, 10))
    window.overview.setXRange(0, 10, padding=0)
    window._set_region((2, 4))
    window.show()
    app.processEvents()
    view = window.waveform.getViewBox()
    viewport = window.waveform.viewport()
    start = window.waveform.mapFromScene(view.mapViewToScene(QtCore.QPointF(2.4, 0)))
    finish = window.waveform.mapFromScene(view.mapViewToScene(QtCore.QPointF(3.1, 0)))
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    QtTest.QTest.mouseMove(viewport, finish, 50)
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=finish)
    app.processEvents()
    assert np.allclose(window._region_samples(), (2400, 3100), atol=5)
    assert np.allclose(window.overview_region.getRegion(), (2, 4))
    assert np.allclose(view.viewRange()[0], (2, 4))
    window.overview_region.setRegion((5, 8))
    window._zoom_selection(0.5)
    assert np.allclose(window._region_samples(), (2400, 3100), atol=5)
    assert "浏览" in window.readout.text() and "选区" in window.readout.text()
    window._use_visible_range()
    lo, hi = window.overview_region.getRegion()
    assert np.allclose(window._region_samples(), (round(lo * 1000), round(hi * 1000)))
    window.navigation_analysis_timer.stop()
    window.analysis_debounce.stop()
    window.close()


def test_explicit_selection_is_used_for_audio_export_playback_and_template(tmp_path, monkeypatch):
    import soundfile as sf

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SignalAnalyzer()
    window.samples = np.linspace(-0.5, 0.5, 10000)
    window.sample_rate = 1000
    window.overview_region.setBounds((0, 10))
    window._set_region((2, 4))
    window._set_analysis_selection(2.25, 2.75)
    path = tmp_path / "selected.wav"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a: (str(path), "WAV"))
    window.export_selection()
    samples, rate = sf.read(path)
    assert rate == 1000 and samples.size == 500
    assert np.allclose(samples, window.samples[2250:2750], atol=4e-5)
    window.set_selection_reference()
    assert np.array_equal(window.reference, window.samples[2250:2750])
    window.play_selection()
    window.player.stop()
    played, rate = sf.read(window.player.source().toLocalFile())
    assert rate == 1000 and played.size == 500
    assert np.allclose(played, samples, atol=4e-5)
    window.analysis_debounce.stop()
    window.close()


def test_new_request_cancels_obsolete_jobs_and_releases_pending_full_view_keys():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SignalAnalyzer()
    window.samples = np.ones(16000)
    window.sample_rate = 16000
    window._set_region((0, 1))
    window.generation = 3
    cancelled = threading.Event()
    key = (id(window.samples), 12)
    window.worker_cancel_events[3] = cancelled
    window.pending_analysis[3] = (None, key)
    window.pending_full_cwt.add(key)
    window._request_analysis()
    assert cancelled.is_set()
    assert key not in window.pending_full_cwt
    assert 3 not in window.pending_analysis
    window.analysis_debounce.stop()
    window.close()


def test_switch_to_fft_does_not_wait_for_running_cwt_block(monkeypatch):
    import time
    import task1.analyzer.app as analyzer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    started = threading.Event()
    release = threading.Event()
    def slow_cwt(*args, **kwargs):
        started.set()
        release.wait(5)
        if kwargs["cancel_event"].is_set():
            raise AnalysisCancelled("superseded")
        return np.arange(24), np.arange(32), np.zeros((24, 32))
    monkeypatch.setattr(analyzer, "cwt_view", slow_cwt)
    window = SignalAnalyzer()
    window.samples = np.sin(np.arange(16000))
    window.sample_rate = 16000
    window.overview_region.setBounds((0, 1))
    window._set_region((0, 1))
    try:
        window.tabs.setCurrentIndex(2)
        deadline = time.perf_counter() + 3
        while not started.is_set() and time.perf_counter() < deadline:
            app.processEvents()
            QtTest.QTest.qWait(5)
        assert started.is_set()
        window.tabs.setCurrentIndex(0)
        deadline = time.perf_counter() + 3
        while not any(key[1] == "fft" for key in window.selection_cache) and time.perf_counter() < deadline:
            app.processEvents()
            QtTest.QTest.qWait(5)
        assert not release.is_set()
        assert any(key[1] == "fft" for key in window.selection_cache)
    finally:
        release.set()
        window.thread_pool.waitForDone(5000)
        window.interactive_thread_pool.waitForDone(5000)
        window.close()


def test_comparison_views_put_full_signal_on_left_and_selection_on_right():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SignalAnalyzer()
    window.show()
    app.processEvents()
    for splitter, full, selected in (
        (window.fft_splitter, window.fft_full_plot, window.fft_plot),
        (window.cwt_splitter, window.cwt_overview_plot, window.cwt_plot),
        (window.auto_splitter, window.auto_full_plot, window.auto_plot),
        (window.cross_splitter, window.cross_full_plot, window.cross_plot),
    ):
        assert splitter.orientation() == QtCore.Qt.Orientation.Horizontal
        assert splitter.widget(0) is full
        assert splitter.widget(1) is selected
    assert window.fft_full_curve in window.fft_full_plot.items()
    assert window.fft_full_curve not in window.fft_plot.items()
    window.close()


def test_correlation_worker_separates_whole_file_and_local_template_lags():
    rng = np.random.default_rng(34)
    template = rng.normal(size=80)
    samples = np.zeros(3000)
    samples[1300:1380] = template
    worker = AnalysisWorker(
        1, samples, 1000, (1000, 1800), template, "hann", None,
        256, 256, 0.75, 8, None, mode="correlation")
    results = []
    worker.signals.ready.connect(lambda _, payload: results.append(payload))
    worker.run()
    payload = results[0]
    assert payload["crosscorrelation"][2] == 300
    assert payload["full_correlation"]["crosscorrelation"][2] == 1300
    assert payload["autocorrelation"][0].size == 1599
    assert payload["full_correlation"]["autocorrelation"][0].size == 5999


def test_side_by_side_fft_svg_contains_both_vector_panels(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SignalAnalyzer()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    app.processEvents()
    window.fft_full_curve.setData([100, 200, 300], [-40, -10, -40])
    window.fft_selected_curve.setData([100, 200, 300], [-50, -20, -50])
    path = tmp_path / "comparison.svg"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *args: (str(path), "SVG"))
    try:
        window.export_current_plot()
        root = ET.parse(path).getroot()
        assert root.findall(".//{http://www.w3.org/2000/svg}path")
        assert not root.findall(".//{http://www.w3.org/2000/svg}image")
        assert "完整信号" in path.read_text(encoding="utf-8")
        assert float(root.attrib["viewBox"].split()[2]) >= 1000
    finally:
        window.close()


def test_close_cancels_and_joins_background_analysis(monkeypatch):
    import time
    import task1.analyzer.app as analyzer
    app = QtWidgets.QApplication.instance()
    started = threading.Event()
    def cancellable_cwt(*args, **kwargs):
        started.set()
        assert kwargs["cancel_event"].wait(5)
        raise AnalysisCancelled("window closed")
    monkeypatch.setattr(analyzer, "cwt_view", cancellable_cwt)
    window = SignalAnalyzer()
    window.samples = np.ones(8000)
    window.sample_rate = 8000
    window.overview_region.setBounds((0, 1))
    window._set_region((0, 1))
    window.tabs.setCurrentIndex(2)
    window.analysis_debounce.stop()
    window._launch_analysis()
    deadline = time.perf_counter() + 3
    while not started.is_set() and time.perf_counter() < deadline:
        app.processEvents()
        QtTest.QTest.qWait(5)
    assert started.is_set()
    window.close()
    assert not window.analysis_debounce.isActive()
    assert not window.navigation_analysis_timer.isActive()
    assert window.thread_pool.activeThreadCount() == 0
    assert window.interactive_thread_pool.activeThreadCount() == 0
