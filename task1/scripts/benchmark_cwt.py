from __future__ import annotations

import json
import argparse
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.signal_core import cwt_view, read_audio, write_audio


def measure(duration_s: int, sample_rate: int = 16000) -> dict[str, float | int]:
    time_axis = np.arange(sample_rate * duration_s, dtype=np.float64) / sample_rate
    samples = (0.45 * np.sin(2 * np.pi * 1500 * time_axis)
               + 0.35 * np.sin(2 * np.pi * 2500 * time_axis))
    del time_axis
    with tempfile.TemporaryDirectory(prefix="signalscope-cwt-") as directory:
        path = str(Path(directory) / "two_carriers.wav")
        write_audio(path, samples, sample_rate)
        samples, sample_rate = read_audio(path)
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak_rss = [baseline]
    stop_monitor = threading.Event()

    def monitor() -> None:
        while not stop_monitor.wait(0.02):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.perf_counter()
    try:
        frequencies, times, values = cwt_view(samples, sample_rate, maximum_points=12000)
    finally:
        elapsed = time.perf_counter() - started
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        stop_monitor.set()
        monitor_thread.join()
    return {
        "duration_s": duration_s,
        "sample_rate_hz": sample_rate,
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mib": round(peak_rss[0] / 2**20, 1),
        "peak_increment_mib": round((peak_rss[0] - baseline) / 2**20, 1),
        "frequency_rows": int(frequencies.size),
        "time_columns": int(times.size),
        "frequency_max_hz": round(float(frequencies[-1]), 1),
        "finite_output": bool(np.isfinite(values).all()),
    }


def measure_gui_switch(duration_s: int, target: str) -> dict[str, float | int | str | bool]:
    from PySide6 import QtCore, QtWidgets
    import shared.signal_core as core
    from task1.analyzer.app import SignalAnalyzer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SignalAnalyzer()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    sample_rate = 16000
    t = np.arange(sample_rate * duration_s) / sample_rate
    window.samples = 0.45 * np.sin(2 * np.pi * 1500 * t) + 0.35 * np.sin(2 * np.pi * 2500 * t)
    window.sample_rate = sample_rate
    window.overview_region.setBounds((0, duration_s))
    window._set_region((0, 1.5))
    window.show()
    entered = threading.Event()
    original_cwt = core.pywt.cwt
    block_times = []

    def observed_cwt(*args, **kwargs):
        if len(args[0]) > 60000:
            entered.set()
        started = time.perf_counter()
        result = original_cwt(*args, **kwargs)
        block_times.append(time.perf_counter() - started)
        return result

    core.pywt.cwt = observed_cwt
    try:
        window.tabs.setCurrentIndex(2)
        deadline = time.perf_counter() + 30
        while not entered.is_set() and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.002)
        if not entered.is_set():
            raise RuntimeError("CWT did not start")
        events = list(window.worker_cancel_events.values())
        started = time.perf_counter()
        window.tabs.setCurrentIndex({"fft": 0, "stft": 1}[target])
        first_event_s = None
        event_loop_ready = threading.Event()
        QtCore.QTimer.singleShot(0, event_loop_ready.set)
        def target_ready():
            return any(key[1] == target for key in window.selection_cache)
        while not target_ready() and time.perf_counter() - started < 30:
            app.processEvents()
            if first_event_s is None and event_loop_ready.is_set():
                first_event_s = time.perf_counter() - started
            time.sleep(0.002)
        if not target_ready():
            raise RuntimeError(f"{target} did not finish after switching")
        return {
            "duration_s": duration_s, "sample_rate_hz": sample_rate, "target": target,
            "switch_to_result_s": round(time.perf_counter() - started, 4),
            "event_loop_response_s": round(first_event_s or 0, 4),
            "cancel_requested": bool(events and all(event.is_set() for event in events)),
            "longest_cwt_block_s": round(max(block_times, default=0), 4),
        }
    finally:
        for event in window.worker_cancel_events.values():
            event.set()
        window.thread_pool.waitForDone(30000)
        window.interactive_thread_pool.waitForDone(30000)
        core.pywt.cwt = original_cwt
        window.navigation_analysis_timer.stop()
        window.analysis_debounce.stop()
        window.close()
        app.processEvents()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("durations", nargs="*", type=int, default=[60, 180])
    parser.add_argument("--gui-only", action="store_true")
    parser.add_argument("--gui-switch", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "task2/experiments/performance/cwt_benchmark.json")
    args = parser.parse_args()
    durations = args.durations or [60, 180]
    results = {"cwt": [], "gui_switch": []}
    if args.gui_only and args.output.exists():
        results = json.loads(args.output.read_text(encoding="utf-8"))
    for duration in durations:
        if not args.gui_only:
            result = measure(duration)
            results["cwt"].append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.gui_switch or args.gui_only:
            for target in ("fft", "stft"):
                result = measure_gui_switch(duration, target)
                results["gui_switch"].append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
