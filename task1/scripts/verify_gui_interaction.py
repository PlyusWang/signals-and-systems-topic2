"""Native Qt checks without taking focus from other desktop applications."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from task1.analyzer.app import SignalAnalyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-dpr", type=float, required=True)
    parser.add_argument("--long-cwt", action="store_true")
    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
    window = SignalAnalyzer()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.resize(1440, 1000)
    window.show()
    dpr = window.devicePixelRatioF()
    assert np.isclose(dpr, args.expected_dpr), (dpr, args.expected_dpr)
    out = ROOT / ".codex" / f"gui_interaction_dpi{round(dpr * 100)}"
    out.mkdir(exist_ok=True)
    record = {"device_pixel_ratio": dpr, "passed": False, "stft": [], "cwt_switch": []}

    def wait(predicate, timeout=60):
        deadline = time.perf_counter() + timeout
        while not predicate() and time.perf_counter() < deadline:
            app.processEvents()
            QtTest.QTest.qWait(10)
        assert predicate(), window.statusBar().currentMessage()

    def wait_result(mode):
        key = window._selection_result_key(mode, *window._region_samples(), window._fft_window_name())
        wait(lambda: key in window.selection_cache and not window.analysis_debounce.isActive()
             and window.generation not in window.pending_analysis)
        return window.selection_cache[key]

    try:
        with tempfile.TemporaryDirectory(prefix="signalscope-interaction-") as directory:
            for rate in (16000, 44100, 48000):
                path = Path(directory) / f"tone_{rate}.wav"
                t = np.arange(rate * 2) / rate
                sf.write(path, .25 * np.sin(2 * np.pi * 1777.32 * t), rate)
                window.tabs.setCurrentIndex(0)
                window.load_path(str(path), remember_previous=False)
                window._set_region((.1, 1.5))
                window._set_analysis_selection(.2, .7)
                wait_result("fft")
                for scale in ("线性", "对数"):
                    window.frequency_scale.setCurrentText(scale)
                    bounds = (1500, 2500) if scale == "线性" else np.log10((1500, 2500))
                    window.fft_plot.setXRange(*bounds, padding=0)
                    window._set_analysis_selection(.3, .65)
                    wait_result("fft")
                    assert np.allclose(window.fft_plot.getViewBox().viewRange()[0], bounds)
                window.fft_fit_button.click()
                for size in ((960, 800), (1440, 1000)):
                    window.resize(*size)
                    app.processEvents()
                    assert window.waveform.getViewBox().width() > 250
                    assert window.fft_plot.getViewBox().height() > 50
                    window.grab().save(str(out / f"fft_{rate}_{size[0]}.png"))
                if rate == 16000:
                    window.stft_window_combo.setCurrentText("512")
                    window.stft_fft_combo.setCurrentText("512")
                    window.tabs.setCurrentIndex(1)
                    for milliseconds in (20, 40, 100, 500):
                        window._set_analysis_selection(.2, .2 + milliseconds / 1000)
                        payload = wait_result("stft")
                        actual = payload["stft_parameters"]
                        assert f"窗长 {actual['window_length']} 点" in window.stft_plot.plotItem.titleLabel.text
                        record["stft"].append(dict(duration_ms=milliseconds, **actual))
                    window._set_analysis_selection(.2, .24)
                    wait_result("stft")
                    window.grab().save(str(out / "stft_short.png"))
                window._set_analysis_selection(.2, .7)
                window.tabs.setCurrentIndex(2)
                wait_result("cwt")
                assert window.analysis_progress.isHidden()
                window.grab().save(str(out / f"cwt_{rate}.png"))
                window.visible_range_button.click()
                assert window.analysis_selection is None
                assert "跟随浏览" in window.readout.text()

            if args.long_cwt:
                for seconds, target in ((60, 0), (180, 1)):
                    path = Path(directory) / f"long_{seconds}.wav"
                    t = np.arange(16000 * seconds) / 16000
                    sf.write(path, .2 * np.sin(2 * np.pi * 1500 * t), 16000)
                    window.load_path(str(path), remember_previous=False)
                    window._set_analysis_selection(.2, .4)
                    window.tabs.setCurrentIndex(2)
                    wait(lambda: not window.analysis_progress.isHidden()
                         and "完整信号" in window.analysis_progress.format()
                         and window.analysis_progress.value() > 0)
                    old_generation = window.generation
                    blocks = window.analysis_progress.format()
                    window.grab().save(str(out / f"progress_{seconds}.png"))
                    started = time.perf_counter()
                    window.tabs.setCurrentIndex(target)
                    assert window.analysis_progress.isHidden()
                    window._analysis_progress(old_generation, 3, 4, "完整信号")
                    assert window.analysis_progress.isHidden()
                    wait_result("fft" if target == 0 else "stft")
                    elapsed = time.perf_counter() - started
                    record["cwt_switch"].append({"seconds": seconds, "progress": blocks,
                                                  "target": "fft" if target == 0 else "stft",
                                                  "result_seconds": elapsed,
                                                  "stale_progress_ignored": True})
                    assert elapsed < 3
            record["passed"] = True
    finally:
        window.close()
        destination = ROOT / "task2/experiments/performance" / f"gui_interaction_dpi{round(dpr * 100)}.json"
        destination.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
