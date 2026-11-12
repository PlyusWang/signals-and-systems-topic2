"""Native Qt comparison checks; hidden windows leave the user's desktop alone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from PySide6 import QtCore, QtGui, QtTest, QtWidgets
from PySide6.QtMultimedia import QMediaPlayer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from task1.analyzer.app import SignalAnalyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-dpr", type=float, required=True)
    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
    window = SignalAnalyzer()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    window.audio_output.setVolume(0)
    out = ROOT / ".codex" / f"comparison_round8_dpi{round(args.expected_dpr * 100)}"
    out.mkdir(exist_ok=True)
    records = []

    def visible_canvas_height(plot):
        box = plot.mapFromScene(plot.getViewBox().sceneBoundingRect()).boundingRect()
        rect = QtCore.QRect(plot.mapTo(window, box.topLeft()), box.size())
        for scroll in (window.content_scroll, window.wave_scroll):
            if scroll.isAncestorOf(plot):
                viewport = scroll.viewport()
                rect = rect.intersected(QtCore.QRect(viewport.mapTo(window, QtCore.QPoint()), viewport.size()))
        return rect.intersected(window.rect()).height()

    def wait_fft():
        deadline = time.perf_counter() + 15
        while time.perf_counter() < deadline:
            app.processEvents()
            QtTest.QTest.qWait(10)
            key = window._selection_result_key("fft", *window._region_samples(), window._fft_window_name())
            if key in window.selection_cache and not window.analysis_debounce.isActive():
                return window.selection_cache[key]
        raise AssertionError(window.statusBar().currentMessage())

    try:
        assert np.isclose(window.devicePixelRatioF(), args.expected_dpr)
        with tempfile.TemporaryDirectory(prefix="signalscope-compare-") as folder:
            for rate in (16000, 44100, 48000):
                path = Path(folder) / f"主信号_含噪音频_{rate}.wav"
                compare = Path(folder) / f"对比信号_干净音频_{rate}.wav"
                t, ct = np.arange(rate * 2) / rate, np.arange(rate) / (rate // 2)
                sf.write(path, .3 * np.sin(2 * np.pi * 1777.32 * t), rate, subtype="FLOAT")
                sf.write(compare, .15 * np.sin(2 * np.pi * 1777.32 * ct), rate // 2, subtype="FLOAT")
                window.load_path(str(path))
                assert not window.overlay_check.isChecked()
                window.load_comparison_path(str(compare))
                window._set_region((.1, 1.2))
                window._set_analysis_selection(.3, .6)
                for size in ((900, 640), (1366, 768), (1440, 900)):
                    window.resize(*size)
                    for mode in (0, 1):
                        window.comparison_mode.setCurrentIndex(mode)
                        for offset in (-.1, 0, .1):
                            window.comparison_offset.setValue(offset)
                            wait_fft()
                        app.processEvents()
                        QtTest.QTest.qWait(50)
                        window.content_scroll.verticalScrollBar().setValue(0)
                        window.wave_scroll.verticalScrollBar().setValue(0)
                        assert window.waveform.getViewBox().height() >= 120
                        assert window.fft_plot.getViewBox().height() >= 150
                        assert window.technical_info_label.text() == f"{rate:,} Hz  |  2.000 秒  |  1 声道"
                        assert window.technical_info_label.width() >= window.technical_info_label.sizeHint().width()
                        assert "Hz" not in window.metadata_label.text()
                        heights = {"main": round(window.waveform.getViewBox().height(), 1),
                                   "fft": round(window.fft_plot.getViewBox().height(), 1),
                                   "comparison": round(window.comparison_waveform.getViewBox().height(), 1) if mode else None,
                                   "overview": round(window.overview.getViewBox().height(), 1) if window.overview.isVisible() else None}
                        assert (window.width(), window.height()) == size
                        for widget in (window.comparison_mode, window.comparison_offset,
                                       window.locate_selection_button):
                            assert widget.width() >= widget.minimumSizeHint().width(), widget
                        if mode:
                            assert window.comparison_waveform.getViewBox().height() >= 120
                            assert np.allclose(window.comparison_waveform.getViewBox().viewRange()[0],
                                               window.waveform.getViewBox().viewRange()[0])
                        main_visible = visible_canvas_height(window.waveform)
                        assert main_visible >= 120, (size, mode, main_visible)
                        window.grab().save(str(out / f"{rate}_{size[0]}_{mode}_top.png"))
                        if mode:
                            window.wave_scroll.verticalScrollBar().setValue(
                                min(window.wave_scroll.verticalScrollBar().maximum(),
                                    window.comparison_waveform.y()))
                            app.processEvents()
                            assert visible_canvas_height(window.comparison_waveform) >= 120, (size, visible_canvas_height(window.comparison_waveform), window.wave_scroll.verticalScrollBar().value())
                            window.grab().save(str(out / f"{rate}_{size[0]}_{mode}_track.png"))
                        window.content_scroll.verticalScrollBar().setValue(
                            window.content_scroll.verticalScrollBar().maximum())
                        app.processEvents()
                        fft_visible = visible_canvas_height(window.fft_plot)
                        assert fft_visible >= 150, (size, mode, fft_visible)
                        window.grab().save(str(out / f"{rate}_{size[0]}_{mode}_analysis.png"))
                        records.append({"rate": rate, "size": size, "split_tracks": bool(mode),
                                        "canvas_heights": heights, "visible_main_height": main_visible,
                                        "visible_fft_height": fft_visible,
                                        "overview_expanded": window.overview.isVisible(),
                                        "offsets_s": [-.1, 0, .1], "technical_parameters_complete": True})
                window.play_selection()
                deadline = time.perf_counter() + 5
                while window.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                    app.processEvents()
                    QtTest.QTest.qWait(5)
                    assert time.perf_counter() < deadline
                played, played_rate = sf.read(window.player.source().toLocalFile())
                start, end = window._region_samples()
                assert played_rate == rate
                assert np.allclose(played, window.samples[start:end], atol=4e-5)
                window.stop_playback()
    finally:
        window.close()
    result = {"passed": True, "device_pixel_ratio": args.expected_dpr,
              "cases": records, "main_signal_playback_verified": True}
    (ROOT / "task2/experiments/performance" / f"audio_comparison_dpi{round(args.expected_dpr * 100)}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "dpr": args.expected_dpr, "cases": len(records)}))


if __name__ == "__main__":
    main()
