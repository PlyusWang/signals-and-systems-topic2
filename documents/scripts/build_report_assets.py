"""Build report figures from saved results and enlarged native GUI views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def representative_waveforms():
    import numpy as np
    from task2.communication_experiment.modem import ModemConfig, make_frame
    from task2.communication_experiment.noise import NoiseConfig, make_noise_components

    saved = json.loads((ROOT / "task2/experiments/representative/frame.json").read_text(encoding="utf-8"))
    powers = json.loads((ROOT / "task2/experiments/representative/noise_power.json").read_text(encoding="utf-8"))
    config = ModemConfig(sample_rate=saved["sample_rate"], bit_rate=saved["bit_rate"],
                         f0_hz=saved["f0_hz"], f1_hz=saved["f1_hz"])
    clean = np.asarray(make_frame(bytes.fromhex(saved["payload_hex"]), config)["waveform"])
    observation = saved["noise_observation_samples"]
    components = make_noise_components(
        clean, config.sample_rate, saved["seed"],
        NoiseConfig(awgn_snr_db=2, narrowband_inr_db=-3, burst_snr_db=1, nonstationary_snr_db=0),
        output_length=observation + clean.size)
    scale = saved["common_wav_scale"]
    clean = clean * scale
    components = {name: values * scale for name, values in components.items()}
    for name, values in components.items():
        assert np.isclose(np.mean(values ** 2), powers[name]["mean_square"], rtol=1e-12, atol=0)
        assert np.isclose(np.max(np.abs(values)), powers[name]["peak"], rtol=1e-12, atol=0)
    return config, clean, components, observation


def build_noise_waveforms() -> None:
    import numpy as np
    from task2.communication_experiment.runner import _save_plot

    config, clean, components, observation = representative_waveforms()
    count = min(config.sample_rate, clean.size)
    _save_plot(ROOT / "task2/figures/noise_waveforms.png", "waveforms",
               np.arange(count) / config.sample_rate,
               [("clean", clean[:count])] + [
                   (name, components[name][observation:observation + count])
                   for name in ("N1", "N2", "N3", "N4")])


def build_time_frequency_views() -> None:
    from task2.communication_experiment.noise import combine_noise
    from task2.communication_experiment.runner import _save_plot

    config, clean, components, observation = representative_waveforms()
    frame_noise = {name: values[observation:] for name, values in components.items()}
    noisy = combine_noise(clean, frame_noise, ("N1", "N2", "N3"))
    for stem, presentation in (("time_frequency_clean_noisy", False),
                               ("time_frequency_clean_noisy_presentation", True)):
        _save_plot(ROOT / f"task2/figures/{stem}.png", "time_frequency",
                   clean, noisy, config.sample_rate, presentation=presentation,
                   include_cwt=False)


def build_charts() -> None:
    from task2.communication_experiment.runner import _save_plot

    summary = json.loads((ROOT / "task2/experiments/evaluation/seed_sweep_summary.json").read_text(encoding="utf-8"))
    snrs = sorted({row["target_snr_db"] for row in summary})
    methods = ("unprocessed", "wavelet_soft", "bandpass", "psd_wiener",
               "notch_bandpass_auto", "notch_bandpass", "hybrid_notch_wavelet")

    def values(method, field):
        return [next(row[field] for row in summary
                     if row["case"] == "N1_N2" and row["target_snr_db"] == snr
                     and row["method"] == method) for snr in snrs]

    for name, field, ylabel in (("snr_sweep_N1_N2", "output_snr_mean_db", "平均输出 SNR (dB)"),
                                ("ber_sweep_N1_N2", "ber_mean", "有效载荷平均 BER")):
        _save_plot(ROOT / f"task2/figures/{name}.png", "line", snrs,
                   [(method, values(method, field)) for method in methods],
                   "输入 SNR (dB)", ylabel)
    _save_plot(ROOT / "task2/figures/wavelet_sync_comparison.png", "line", snrs, [
        ("未处理：相关同步", values("unprocessed", "ber_mean")),
        ("未处理：真实帧起点", values("unprocessed", "ber_known_start_mean")),
        ("小波：相关同步", values("wavelet_soft", "ber_mean")),
        ("小波：真实帧起点", values("wavelet_soft", "ber_known_start_mean")),
    ], "目标输入 SNR (dB)", "有效载荷平均 BER")


def build_gui_views() -> None:
    import numpy as np
    from PySide6 import QtCore, QtGui, QtTest, QtWidgets
    from task1.analyzer.app import SignalAnalyzer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
    window = SignalAnalyzer()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.resize(1100, 1030)
    window.show()
    out = ROOT / "documents/report/images"
    out.mkdir(exist_ok=True)
    audio_root = ROOT / "task2/audio"

    def wait_result(mode):
        deadline = time.perf_counter() + 45
        key = window._selection_result_key(mode, *window._region_samples(), window._fft_window_name())
        while time.perf_counter() < deadline:
            app.processEvents()
            QtTest.QTest.qWait(10)
            if key in window.selection_cache and not window.analysis_debounce.isActive():
                return
        raise RuntimeError(window.statusBar().currentMessage())

    def capture_pair(plots, name):
        panel = QtWidgets.QWidget()
        panel.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
        layout = QtWidgets.QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        for plot in plots:
            layout.addWidget(plot, 1)
            for side in ("left", "bottom"):
                axis = plot.getAxis(side)
                axis.setTickFont(QtGui.QFont("Segoe UI", 16))
                axis.setLabel(text=axis.labelText, units=axis.labelUnits,
                              **{"font-size": "16pt", "color": "#aeb8bd"})
                axis.setStyle(autoExpandTextSpace=True)
            if hasattr(plot, "color_bar"):
                plot.color_bar.axis.setTickFont(QtGui.QFont("Segoe UI", 14))
                color_axis = plot.color_bar.axis
                color_axis.setLabel(text=color_axis.labelText, units=color_axis.labelUnits,
                                    **{"font-size": "14pt", "color": "#aeb8bd"})
        panel.resize(1100, 390)
        panel.show()
        app.processEvents()
        QtTest.QTest.qWait(150)
        panel.grab().save(str(out / name))
        for plot in plots:
            plot.setParent(window)
        panel.close()

    try:
        window.load_path(str(audio_root / "clean_plus_N1_N2.wav"))
        window._set_region((.20, .90))
        window._set_analysis_selection(.256, .288)
        window.set_selection_reference()
        window._set_analysis_selection(.256, .448)
        window.load_comparison_path(str(audio_root / "clean.wav"))
        window.comparison_mode.setCurrentText("叠加波形")
        window._set_playback_cursor(.32)
        wait_result("fft")
        app.processEvents()
        QtTest.QTest.qWait(100)
        window.grab().save(str(out / "gui_overview.png"))
        window.overlay_check.setChecked(False)
        window.tabs.setCurrentIndex(2)
        wait_result("cwt")
        window.cwt_overview_plot.setTitle("完整信号 · CWT", size="15pt")
        window.cwt_plot.setTitle("选区 256–448 ms · CWT", size="15pt")
        for plot in (window.cwt_overview_plot, window.cwt_plot):
            plot.setYRange(800, 3500, padding=0)
        window.cwt_overview_plot.setXRange(0, window.samples.size / window.sample_rate, padding=0)
        window.cwt_plot.setXRange(.256, .448, padding=0)
        capture_pair((window.cwt_overview_plot, window.cwt_plot), "gui_cwt_core.png")
        window.tabs.setCurrentIndex(3)
        window.correlation_tabs.setCurrentIndex(1)
        wait_result("correlation")
        assert np.isclose(window.cross_full_peak_marker.value(), .256)
        assert np.isclose(window.cross_peak_marker.value(), 0)
        for plot, curve, bounds, title in (
            (window.cross_full_plot, window.cross_full_curve, (.23, .29), "完整信号 · 峰值 256 ms"),
            (window.cross_plot, window.cross_curve, (-.025, .035), "选区 · 相对时延 0 ms"),
        ):
            plot.setTitle(title, size="15pt")
            plot.setXRange(*bounds, padding=0)
            _, y = curve.getData()
            plot.setYRange(min(0, float(np.min(y))), float(np.max(y)) * 1.08, padding=0)
        capture_pair((window.cross_full_plot, window.cross_plot), "gui_correlation_core.png")
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--charts-only", action="store_true")
    parser.add_argument("--noise-waveforms-only", action="store_true")
    parser.add_argument("--time-frequency-only", action="store_true")
    args = parser.parse_args()
    if args.time_frequency_only:
        build_time_frequency_views()
    elif args.noise_waveforms_only:
        build_noise_waveforms()
    else:
        build_noise_waveforms()
        build_charts()
        if not args.charts_only:
            build_gui_views()
