"""Opt-in smoke checks executed by the source and frozen applications."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf
from PySide6 import QtCore, QtTest, QtWidgets
from PySide6.QtMultimedia import QMediaPlayer


def verify_window(window, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance()
    result = {"passed": False, "frozen": bool(getattr(__import__("sys"), "frozen", False))}
    original_dialog = QtWidgets.QFileDialog.getSaveFileName

    def wait_for(predicate, timeout=30):
        deadline = time.perf_counter() + timeout
        while not predicate() and time.perf_counter() < deadline:
            app.processEvents()
            QtTest.QTest.qWait(10)
        if not predicate():
            raise AssertionError(window.statusBar().currentMessage())

    try:
        duration = window.samples.size / window.sample_rate
        assert duration > 0.9, "The verification WAV must exceed 0.9 seconds."
        window._set_region((0.2, 0.9))
        app.processEvents()
        view = window.waveform.getViewBox()
        viewport = window.waveform.viewport()
        def drag(left, right):
            start = window.waveform.mapFromScene(view.mapViewToScene(QtCore.QPointF(left, 0)))
            finish = window.waveform.mapFromScene(view.mapViewToScene(QtCore.QPointF(right, 0)))
            QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
            QtTest.QTest.mouseMove(viewport, finish, 50)
            QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=finish)
            app.processEvents()
        drag(0.3, 0.5)
        selected = window._region_samples()
        assert np.allclose(selected, (0.3 * window.sample_rate, 0.5 * window.sample_rate), atol=20)
        assert np.allclose(window.overview_region.getRegion(), (0.2, 0.9))
        window.interaction_combo.setCurrentIndex(1)
        drag(0.55, 0.60)
        assert not np.allclose(window.overview_region.getRegion(), (0.2, 0.9))
        assert window._region_samples() == selected
        before = np.diff(window.overview_region.getRegion())[0]
        window._zoom_selection(0.8)
        assert np.isclose(np.diff(window.overview_region.getRegion())[0], 0.8 * before)
        assert np.allclose(window.overview.getViewBox().viewRange()[0], (0, duration))
        assert window._region_samples() == selected
        window.interaction_combo.setCurrentIndex(0)
        window._set_region((0.2, 0.9))
        window._set_analysis_selection(0.256, 0.288)
        window.set_selection_reference()
        window._set_analysis_selection(0.256, 0.448)
        assert window.selection_region.brush.color() != window.overview_region.brush.color()
        assert "分析选区" in window.readout.text()
        for index, mode in enumerate(("fft", "stft", "cwt", "correlation")):
            window.tabs.setCurrentIndex(index)
            key = window._selection_result_key(mode, *window._region_samples(), window._fft_window_name())
            wait_for(lambda: key in window.selection_cache)
            window.grab().save(str(directory / f"{mode}.png"))
            if mode == "correlation":
                window.correlation_tabs.setCurrentIndex(1)
                app.processEvents()
                window.grab().save(str(directory / "crosscorrelation.png"))
        window.tabs.setCurrentIndex(0)
        app.processEvents()
        window.fft_plot.setXRange(*np.log10((1500, 2500)), padding=0)
        window._set_analysis_selection(.26, .45)
        key = window._selection_result_key("fft", *window._region_samples(), window._fft_window_name())
        wait_for(lambda: key in window.selection_cache)
        assert np.allclose(window.fft_plot.getViewBox().viewRange()[0], np.log10((1500, 2500)))
        window.fft_fit_button.click()
        assert np.allclose(window.fft_plot.getViewBox().viewRange()[0], np.log10((20, window.sample_rate / 2)))
        window.visible_range_button.click()
        assert window.analysis_selection is None and not window.selection_region.isVisible()
        window._set_analysis_selection(.256, .448)
        # Use a different-rate file and an offset while keeping the main WAV intact.
        comparison_path = directory / "comparison_input.wav"
        sf.write(comparison_path, -.5 * window.samples[::2], window.sample_rate // 2,
                 subtype="FLOAT")
        window.load_comparison_path(str(comparison_path))
        window.comparison_offset.setValue(.02)
        window.comparison_mode.setCurrentText("上下分轨")
        app.processEvents()
        assert window.comparison_waveform.isVisible()
        assert np.allclose(window.comparison_waveform.getViewBox().viewRange()[0],
                           window.waveform.getViewBox().viewRange()[0])
        assert window.comparison_waveform.getViewBox().height() >= 120
        assert window.waveform.getViewBox().height() >= 120
        assert window.fft_plot.getViewBox().height() >= 150
        assert window.technical_info_label.width() >= window.technical_info_label.sizeHint().width()
        assert "Hz" not in window.metadata_label.text()
        key = window._selection_result_key("fft", *window._region_samples(), window._fft_window_name())
        wait_for(lambda: key in window.selection_cache)
        assert window.fft_comparison_selected_curve.isVisible()
        window.grab().save(str(directory / "comparison.png"))
        path = directory / "selection.wav"
        QtWidgets.QFileDialog.getSaveFileName = lambda *args: (str(path), "WAV")
        window.export_selection()
        audio, rate = sf.read(path)
        lo, hi = window._region_samples()
        assert len(audio) == hi - lo and rate == window.sample_rate
        assert np.allclose(audio, window.samples[lo:hi], atol=4e-5)
        window.play_selection()
        wait_for(lambda: window.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState, 5)
        generation_before_playback = window.generation
        wait_for(lambda: window.playback_cursor.value() > window.playback_offset_s + .03, 5)
        cursor_position = float(window.playback_cursor.value())
        played, played_rate = sf.read(window.player.source().toLocalFile())
        assert played_rate == window.sample_rate
        assert np.allclose(played, window.samples[lo:hi], atol=4e-5)
        assert window.playback_offset_s < cursor_position <= window.playback_end_s
        assert np.isclose(cursor_position, window.overview_playback_cursor.value())
        window.grab().save(str(directory / "playback.png"))
        window.stop_playback()
        stopped_cursor = float(window.playback_cursor.value())
        app.processEvents()
        assert np.isclose(window.playback_cursor.value(), stopped_cursor)
        assert window.generation == generation_before_playback
        assert window._region_samples() == (lo, hi)
        for suffix, filter_ in (("png", "PNG"), ("svg", "SVG")):
            export_path = directory / f"fft_export.{suffix}"
            QtWidgets.QFileDialog.getSaveFileName = lambda *a, p=export_path, f=filter_: (str(p), f)
            window.export_current_plot()
            assert export_path.stat().st_size > 1000
        root = ET.parse(directory / "fft_export.svg").getroot()
        assert root.findall(".//{http://www.w3.org/2000/svg}path")
        assert not root.findall(".//{http://www.w3.org/2000/svg}image")
        window._set_analysis_selection(.2, .8)
        window.play_selection()
        wait_for(lambda: window.player.duration() == 600, 5)
        window.stop_playback()
        window._set_analysis_selection(lo / window.sample_rate, hi / window.sample_rate)
        result.update(passed=True, local_mouse_selection=True, pan=True, zoom=True,
                      fixed_overview=True, selected_samples=[lo, hi],
                      analyzed_views=["fft", "stft", "cwt", "correlation"],
                      playback_state_playing=True, wav_export=True, png_export=True,
                      live_playback_cursor_seconds=cursor_position,
                      stop_retains_cursor=True, playback_does_not_reanalyze=True,
                      consecutive_selection_reloads_media=True,
                      distinct_selection_navigation_colors=True,
                      fft_manual_zoom_preserved=True, fft_fit_control=True,
                      clear_selection_follows_visible_range=True,
                      explicit_comparison_file=True, comparison_resampling=True,
                      comparison_split_tracks=True, comparison_fft=True,
                      playback_and_export_use_main_signal=True,
                      svg_vector_paths_without_embedded_image=True)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        QtWidgets.QFileDialog.getSaveFileName = original_dialog
        (directory / "verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        app.exit(0 if result["passed"] else 1)
