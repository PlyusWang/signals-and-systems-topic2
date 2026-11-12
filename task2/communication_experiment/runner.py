from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 17,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 12,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
})
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy import signal

from task2.communication_experiment.modem import (
    ModemConfig, bit_error_rate, make_frame, random_payload, receive, receive_at_start,
)
from task2.communication_experiment.noise import (
    NoiseConfig, actual_snr_db, assemble_capture, combine_noise, make_noise_components,
    repeated_snr_reference,
)
from shared.signal_core import (
    bandpass, detect_narrowband_interference, measured_snr, notch, stft_view, wavelet_denoise,
    waveform_metrics, wiener_from_noise_segment, write_audio,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "audio"
EXPERIMENTS = ROOT / "experiments"
FIGURES = ROOT / "figures"

for font_path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simsun.ttc")):
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
        break


def _save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=lambda x: float(x)), encoding="utf-8")


def _payload_ber(reference_bits: np.ndarray, payload: bytes, expected_bytes: int) -> float:
    if len(payload) != expected_bytes:
        return 1.0
    decoded_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    return bit_error_rate(reference_bits, decoded_bits)


def _save_plot(path: Path, *args, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = args[0] if args else None
    presentation = bool(kwargs.pop("presentation", False))
    presentation_size = kwargs.pop("presentation_size", None)
    if presentation and presentation_size:
        size = presentation_size
    elif presentation:
        size = {"resolution": (8.15, 5.8), "line": (6.15, 3.7),
                "bars": (6.25, 4.35), "waveforms": (6.5, 2.15),
                "time_frequency": (7.2, 4.85)}.get(renderer, (6.2, 3.6))
    else:
        size = (6.2, 2.45) if renderer == "waveforms" else (7.0, 4.4) if renderer == "bars" else (6.2, 3.4)
    fig, ax = plt.subplots(figsize=size)
    kwargs.pop("title", None)
    if renderer == "waveforms":
        _, time_axis, series = args
        for name, values in series:
            label = {"clean": "干净信号", "N1": "N1 持续底噪", "N2": "N2 窄带干扰",
                     "N3": "N3 突发噪声", "N4": "N4 非平稳噪声"}.get(name, name)
            ax.plot(time_axis, values, label=label, linewidth=0.9)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("幅度")
        if presentation:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, ncol=3, loc="lower center",
                       bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=14)
            fig.subplots_adjust(left=0.19, right=0.98, bottom=0.65, top=0.96)
        else:
            fig.set_size_inches(7.2, 3.0)
            handles, labels = ax.get_legend_handles_labels()
            legend = fig.legend(handles, labels, loc="upper center",
                                bbox_to_anchor=(0.5, 0.99), ncol=3,
                                frameon=False, fontsize=12.5,
                                handlelength=1.7, handletextpad=0.5,
                                columnspacing=1.2, labelspacing=0.55)
            fig.canvas.draw()
            legend_box = legend.get_window_extent().transformed(fig.transFigure.inverted())
            fig.subplots_adjust(left=0.14, right=0.98, bottom=0.30,
                                top=legend_box.y0 - 0.065)
    elif renderer == "spectrograms":
        _, sample_rate, series = args
        for name, values in series:
            f, t, z = stft_view(values, sample_rate, window_length=256, overlap=0.875)
            mesh = ax.pcolormesh(t, f, 20 * np.log10(np.maximum(np.abs(z), 1e-7)),
                                 shading="auto", alpha=0.72, rasterized=True)
            mesh.set_rasterized(True)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("频率 (Hz)")
    elif renderer == "bars":
        _, names, values, ylabel = args
        labels = {"unprocessed": "未处理", "wavelet_soft": "小波软阈值",
                  "wavelet_hard": "小波硬阈值", "bandpass": "带通滤波",
                  "psd_wiener": "PSD-Wiener", "notch_bandpass": "固定陷波+带通",
                  "notch_bandpass_auto": "自动陷波+带通",
                  "hybrid_notch_wavelet": "组合处理", "correlation_detector": "相关检测"}
        ax.barh([labels.get(name, name) for name in names], values, color="#278c78")
        ax.set_xlabel(ylabel)
        ax.tick_params(axis="y", labelsize=16 if presentation else 12)
        fig.subplots_adjust(left=0.34, right=0.98, bottom=0.16, top=0.97)
    elif renderer == "line":
        _, x, series, xlabel, ylabel = args
        for name, y in series:
            label = {"unprocessed": "未处理", "wavelet_soft": "小波软阈值",
                     "bandpass": "带通", "psd_wiener": "PSD-Wiener",
                     "notch_bandpass": "固定陷波 + 带通",
                     "notch_bandpass_auto": "自动陷波 + 带通",
                     "hybrid_notch_wavelet": "组合处理",
                     "independent": "独立 N1 + 随机相位 N2",
                     "shared_tone": "共同窄带干扰", "shared tone": "共同窄带干扰",
                     "notch_bandpass_auto": "自动估频陷波 + 带通",
                     "notch_bandpass": "已知频率陷波 + 带通",
                     "independent plus shared tone": "独立噪声 + 共同窄带干扰"}.get(name, name)
            if presentation:
                label = {"unprocessed": "未处理", "wavelet_soft": "小波",
                         "bandpass": "带通", "psd_wiener": "PSD-Wiener",
                         "notch_bandpass": "固定陷波", "notch_bandpass_auto": "自动陷波",
                         "hybrid_notch_wavelet": "组合处理",
                         "independent": "独立噪声", "shared_tone": "共同干扰",
                         "independent plus shared tone": "独立噪声 + 共同干扰"}.get(name, label)
            ax.plot(x, y, marker="o", label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if presentation:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
                      ncol=min(3, max(1, len(series))), frameon=False, fontsize=14)
            fig.subplots_adjust(left=0.16, right=0.97,
                                bottom=0.36 if len(series) > 2 else 0.30, top=0.96)
        else:
            fig.set_size_inches(7.2, 4.3)
            handles, labels = ax.get_legend_handles_labels()
            legend = fig.legend(handles, labels, loc="upper center",
                                bbox_to_anchor=(0.5, 0.99), ncol=2,
                                frameon=False, fontsize=12.5,
                                handlelength=1.7, handletextpad=0.5,
                                columnspacing=1.4, labelspacing=0.55)
            fig.canvas.draw()
            legend_box = legend.get_window_extent().transformed(fig.transFigure.inverted())
            fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19,
                                top=legend_box.y0 - 0.045)
    elif renderer == "correlation":
        _, lags, values, peak = args
        ax.plot(lags, values, linewidth=0.8)
        ax.axvline(peak, color="#c65c47", linestyle="--", label=f"前导码起点 = {peak} 点")
        ax.set_xlim(max(float(lags[0]), peak - 1600), min(float(lags[-1]), peak + 1600))
        ax.set_xlabel("模板起点 (采样点)")
        ax.set_ylabel("归一化互相关")
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="upper center",
                            bbox_to_anchor=(0.56, 0.99), frameon=False,
                            fontsize=14 if presentation else 12.5)
        fig.canvas.draw()
        legend_box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        fig.subplots_adjust(left=0.19, right=0.96, bottom=0.22,
                            top=legend_box.y0 - 0.065)
        ax.margins(y=0.07)
    elif renderer == "resolution":
        _, sample_rate, samples = args
        from shared.signal_core import cwt_view, one_sided_spectrum
        fig, axes = plt.subplots(2, 3, figsize=size if presentation else (8.8, 5.6))
        if presentation:
            fig.suptitle("双音间隔 80 Hz：观测时长与频率分辨能力",
                         fontsize=19, y=0.99)
        durations = (256, 1024, 4096)
        for axis, count in zip(axes[0], durations):
            spectrum = one_sided_spectrum(samples[:count], sample_rate,
                                          n_fft=4096, window="hann")
            axis.plot(spectrum.frequency_hz, spectrum.dbfs, color="#278c78", linewidth=1.5)
            axis.set_xlim(900, 1180)
            axis.set_ylim(-55, 0)
            axis.set_title(f"FFT：{count / sample_rate * 1000:.0f} ms", fontsize=15 if presentation else None)
            axis.set_xlabel("" if presentation else "频率 (Hz)", fontsize=14 if presentation else None)
            axis.set_ylabel("幅度 (dBFS)" if axis is axes[0, 0] else "",
                            fontsize=14 if presentation else None)
            axis.tick_params(labelsize=14 if presentation else None)
            axis.set_xticks([900, 1080, 1180])
            axis.grid(alpha=0.22)
        for axis, window_length in zip(axes[1, :2], (256, 2048)):
            frequency, times, values = stft_view(samples, sample_rate,
                                                  window_length=window_length, overlap=0.875)
            mesh = axis.pcolormesh(times, frequency,
                                   20 * np.log10(np.maximum(np.abs(values), 1e-7)),
                                   shading="auto", cmap="viridis", vmin=-55, vmax=-8,
                                   rasterized=True)
            mesh.set_rasterized(True)
            axis.set_ylim(850, 1250)
            axis.set_title(f"STFT：{window_length / sample_rate * 1000:.0f} ms 窗",
                           fontsize=15 if presentation else None)
            axis.set_xlabel("时间 (s)", fontsize=14 if presentation else None)
            axis.set_ylabel("频率 (Hz)" if axis is axes[1, 0] else "",
                            fontsize=14 if presentation else None)
            axis.tick_params(labelsize=14 if presentation else None)
            axis.set_yticks([900, 1050, 1200])
            axis.set_xticks([0.0, 0.128, 0.256])
        frequency, times, values = cwt_view(
            samples, sample_rate, maximum_points=samples.size,
            frequency_limits=(850, 1250))
        mesh = axes[1, 2].pcolormesh(times, frequency,
                                    20 * np.log10(np.maximum(values, np.max(values) * 1e-5)),
                                    shading="auto", cmap="viridis", vmin=-35, vmax=0,
                                    rasterized=True)
        mesh.set_rasterized(True)
        axes[1, 2].set_title("CWT：多尺度" if presentation else "CWT：多尺度频率表示",
                             fontsize=15 if presentation else None)
        axes[1, 2].set_xlabel("时间 (s)", fontsize=14 if presentation else None)
        axes[1, 2].set_ylabel("频率 (Hz)", fontsize=14 if presentation else None)
        axes[1, 2].tick_params(labelsize=14 if presentation else None)
        axes[1, 2].set_yticks([900, 1050, 1200])
        axes[1, 2].set_ylabel("")
        axes[1, 2].set_xticks([0.0, 0.128, 0.256])
        if presentation:
            fig.subplots_adjust(left=0.10, right=0.94, bottom=0.16, top=0.87,
                                wspace=0.34, hspace=0.62)
        else:
            fig.subplots_adjust(left=0.10, right=0.95, bottom=0.13, top=0.82,
                                wspace=0.42, hspace=0.74)
        fig.savefig(path, dpi=220 if presentation else None,
                    bbox_inches=None if presentation else "tight")
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        return
    elif renderer == "time_frequency":
        _, clean, noisy, sample_rate = args
        from shared.signal_core import cwt_view
        # A few symbols show the carrier; a long raw trace merges into a solid band.
        count = min(clean.size, noisy.size, int(round(0.006 * sample_rate)) + 1)
        t = np.arange(count) / sample_rate * 1000
        peak = max(float(np.max(np.abs(clean[:count]))),
                   float(np.max(np.abs(noisy[:count]))), np.finfo(float).eps)
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.85), constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.08, h_pad=0.12)
        for axis, values, name in ((axes[0, 0], clean[:count], "Clean"),
                                   (axes[0, 1], noisy[:count], "Noisy")):
            axis.plot(t, values, linewidth=1.1, color="#278c78")
            axis.set_title("干净信号 · 局部" if name == "Clean" else "含噪信号 · 局部")
            axis.set_xlabel("时间 (ms)")
            axis.set_ylabel("幅度")
            axis.set_xlim(float(t[0]), float(t[-1]))
            axis.set_ylim(-peak * 1.12, peak * 1.12)
            axis.set_xticks(np.linspace(t[0], t[-1], 4))
            axis.grid(alpha=0.2)
        mesh = None
        for axis, values, name in ((axes[1, 0], clean, "干净信号 STFT"),
                                   (axes[1, 1], noisy, "含噪信号 STFT")):
            f, st, z = stft_view(values, sample_rate, window_length=256, overlap=0.875)
            mesh = axis.pcolormesh(st, f, 20 * np.log10(np.maximum(np.abs(z), 1e-7)),
                                   shading="auto", cmap="viridis", vmin=-80, vmax=-15,
                                   rasterized=True)
            mesh.set_rasterized(True)
            axis.set_ylim(0, 4200)
            axis.set_title(name)
            axis.set_xlabel("时间 (s)")
            axis.set_ylabel("频率 (Hz)")
        fig.colorbar(mesh, ax=axes[1, :], label="幅度 (dB)")
        fig.savefig(path)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        if not kwargs.pop("include_cwt", True):
            return
        fig, ax = plt.subplots(figsize=(4.55, 2.4), constrained_layout=True)
        frequency, times, values = cwt_view(noisy, sample_rate, maximum_points=6000,
                                            frequency_limits=(250, 4000))
        image = ax.pcolormesh(times, frequency, 20 * np.log10(np.maximum(values, np.max(values) * 1e-5)),
                              shading="auto", cmap="viridis", rasterized=True)
        image.set_rasterized(True)
        ax.set_yscale("log")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("频率 (Hz)")
        fig.colorbar(image, ax=ax, label="相对幅度 (dB)")
        fig.savefig(path.with_name(path.stem + "_cwt.png"))
        fig.savefig(path.with_name(path.stem + "_cwt.pdf"))
        plt.close(fig)
        return
    else:
        raise ValueError(f"unknown plot renderer {renderer}")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=220 if presentation else None,
                bbox_inches=None if presentation else "tight")
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _make_teaching_audio(config: ModemConfig) -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    fs = config.sample_rate
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs
    examples = {
        "teaching_single_tone.wav": 0.5 * np.sin(2 * np.pi * 440 * t),
        "teaching_two_tone.wav": 0.35 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t),
        "teaching_chirp.wav": 0.65 * signal.chirp(t, 200, duration, 3500, method="linear"),
        "teaching_impulse.wav": 0.35 * np.sin(2 * np.pi * 600 * t) + signal.unit_impulse(t.size, 14000) * 0.8,
        "teaching_am.wav": (0.5 + 0.4 * np.cos(2 * np.pi * 5 * t)) * np.cos(2 * np.pi * 1300 * t),
        "teaching_close_tones.wav": (0.45 * np.sin(2 * np.pi * 1000 * t)
                                      + 0.45 * np.sin(2 * np.pi * 1080 * t)),
    }
    for name, values in examples.items():
        write_audio(str(AUDIO / name), values, fs)


def _window_resolution_experiment(config: ModemConfig) -> None:
    fs = config.sample_rate
    duration = 0.256
    time_axis = np.arange(round(fs * duration)) / fs
    samples = 0.45 * np.sin(2 * np.pi * 1000 * time_axis)
    samples += 0.45 * np.sin(2 * np.pi * 1080 * time_axis)
    write_audio(str(AUDIO / "teaching_close_tones.wav"), samples, fs)
    _save_plot(FIGURES / "window_resolution_comparison.png", "resolution", fs, samples)
    _save_plot(FIGURES / "window_resolution_presentation.png", "resolution", fs, samples,
               presentation=True, presentation_size=(8.15, 5.8))
    _save_json(EXPERIMENTS / "window_resolution.json", {
        "sample_rate_hz": fs,
        "tone_frequencies_hz": [1000, 1080],
        "frequency_separation_hz": 80,
        "fft_observation_samples": [256, 1024, 4096],
        "stft_window_samples": [256, 2048],
        "stft_hann_main_lobe_width_hz": [4 * fs / 256, 4 * fs / 2048],
        "fft_nfft_samples": 4096,
        "cwt_frequency_limits_hz": [850, 1250],
    })


def _single_frame_experiment(config: ModemConfig, seed: int = 321) -> list[dict[str, object]]:
    message = "ETU SIGNAL SYSTEMS TOPIC 2: TIME FREQUENCY WAVELET CORRELATION"
    payload = (message.encode("ascii") * 4)[:192]
    frame = make_frame(payload, config)
    clean_frame = np.asarray(frame["waveform"])
    fs = config.sample_rate
    observation_samples = int(0.256 * fs)
    capture_length = observation_samples + clean_frame.size
    _make_teaching_audio(config)
    np.save(AUDIO / "baseband_bits.npy", np.asarray(frame["bits"], dtype=np.uint8))
    exp_dir = EXPERIMENTS / "representative"
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cases = [("N1", ("N1",)), ("N2", ("N2",)), ("N1_N2", ("N1", "N2")),
             ("N3", ("N3",)), ("N1_N3", ("N1", "N3")),
             ("N2_N3", ("N2", "N3")), ("N1_N2_N3", ("N1", "N2", "N3")),
             ("N4", ("N4",)), ("N1_N2_N3_N4", ("N1", "N2", "N3", "N4"))]
    noise = make_noise_components(clean_frame, fs, seed,
        NoiseConfig(awgn_snr_db=2, narrowband_inr_db=-3, burst_snr_db=1, nonstationary_snr_db=0),
        output_length=capture_length)
    mixtures = [("N1",), ("N2",), ("N1", "N2"), ("N3",), ("N1", "N3"),
                ("N2", "N3"), ("N1", "N2", "N3"), ("N4",),
                ("N1", "N2", "N3", "N4")]
    peak = max(float(np.max(np.abs(clean_frame))), *(float(np.max(np.abs(values))) for values in noise.values()))
    for names in mixtures:
        capture = assemble_capture(clean_frame, noise, names, observation_samples)
        peak = max(peak, float(np.max(np.abs(capture))))
    common_scale = min(1.0, 0.95 / max(peak, np.finfo(float).tiny))
    clean_frame = clean_frame * common_scale
    frame["waveform"] = clean_frame
    frame["preamble_waveform"] = np.asarray(frame["preamble_waveform"]) * common_scale
    noise = {name: values * common_scale for name, values in noise.items()}
    clean_capture = np.r_[np.zeros(observation_samples), clean_frame]
    write_audio(str(AUDIO / "clean.wav"), clean_capture, fs)
    _save_json(exp_dir / "frame.json", {
        "seed": seed, "payload_nbytes": len(payload), "payload_hex": payload.hex(),
        "payload_message": payload.decode("ascii"),
        "sample_rate": fs, "bit_rate": config.bit_rate, "f0_hz": config.f0_hz,
        "f1_hz": config.f1_hz, "samples_per_bit": config.samples_per_bit,
        "common_wav_scale": common_scale, "noise_observation_samples": observation_samples,
        "noise_observation_duration_s": observation_samples / fs,
    })
    for n, component in noise.items():
        write_audio(str(AUDIO / f"noise_{n}.wav"), component, fs)
    whole_noise = sum(noise.values())
    write_audio(str(AUDIO / "noise_only_observation.wav"), whole_noise, fs)

    for case, components in cases:
        noisy_capture = assemble_capture(clean_frame, noise, components, observation_samples)
        noisy = noisy_capture[observation_samples:]
        actual_input_snr = actual_snr_db(clean_frame, noisy)
        write_audio(str(AUDIO / f"clean_plus_{case}.wav"), noisy_capture, fs)
        processed: dict[str, np.ndarray] = {}
        runtimes: dict[str, float] = {}
        estimated_tones = detect_narrowband_interference(
            noisy_capture[:observation_samples], fs)
        operations = {
            "wavelet_soft": lambda: wavelet_denoise(noisy, threshold_mode="soft")[0],
            "wavelet_hard": lambda: wavelet_denoise(noisy, threshold_mode="hard")[0],
            "bandpass": lambda: bandpass(noisy, fs, 900, 3200),
            "notch_bandpass": lambda: bandpass(notch(noisy, fs, (1777.0,)), fs, 900, 3200),
            "notch_bandpass_auto": lambda: bandpass(notch(noisy, fs, estimated_tones), fs, 900, 3200),
        }
        operations["psd_wiener"] = lambda: wiener_from_noise_segment(
            noisy, noisy_capture[:observation_samples], fs, window_length=256)
        operations["hybrid_notch_wavelet"] = lambda: wavelet_denoise(
            bandpass(notch(noisy, fs, (1777.0,)), fs, 900, 3200), threshold_mode="soft")[0]
        for method, operation in operations.items():
            t0 = time.perf_counter()
            processed[method] = operation()
            runtimes[method] = time.perf_counter() - t0
        for method, output in processed.items():
            path = AUDIO / f"{method}_{case}.wav"
            write_audio(str(path), np.r_[noisy_capture[:observation_samples], output], fs)
            metrics = waveform_metrics(clean_frame, output)
            output_capture = np.r_[noisy_capture[:observation_samples], output]
            decode_sync = receive(output_capture, len(payload), config)
            decode_known_start = receive_at_start(output_capture, len(payload), observation_samples, config)
            rows.append({"case": case, "noise_components": "+".join(components),
                         "method": method, "input_snr_db": actual_input_snr,
                         "output_snr_db": metrics["snr_db"],
                         "snr_improvement_db": metrics["snr_db"] - actual_input_snr,
                         "mse": metrics["mse"],
                         "ber_correlation_sync": _payload_ber(frame["payload_bits"], decode_sync["payload"], len(payload)),
                         "ber_known_start_diagnostic": _payload_ber(
                             frame["payload_bits"], decode_known_start["payload"], len(payload)),
                         "preamble_errors_correlation_sync": decode_sync["preamble_errors"],
                         "preamble_errors_known_start": decode_known_start["preamble_errors"],
                         "sync_start": decode_sync["start"],
                         "sync_error_samples": decode_sync["start"] - observation_samples,
                         "estimated_tones_hz": ";".join(f"{frequency:.2f}" for frequency in estimated_tones),
                         "runtime_seconds": runtimes[method],
                         "wav": str(path.relative_to(ROOT))})
        decode_started = time.perf_counter()
        receiver = receive(noisy_capture, len(payload), config)
        decode_runtime = time.perf_counter() - decode_started
        regeneration = make_frame(receiver["payload"], config)
        regenerated_path = AUDIO / f"decoded_regenerated_{case}.wav"
        write_audio(str(regenerated_path), regeneration["waveform"], fs)
        detected = receiver["payload"][:len(payload)]
        ber = _payload_ber(frame["payload_bits"], detected, len(payload))
        rows.append({"case": case, "noise_components": "+".join(components),
                     "method": "correlation_detector", "input_snr_db": actual_input_snr,
                     "output_snr_db": "", "snr_improvement_db": "", "mse": "",
                     "ber": ber, "crc_ok": receiver["crc_ok"],
                     "sync_start": int(receiver["start"] - observation_samples),
                     "sync_error_samples": int(receiver["start"] - observation_samples),
                     "sync_peak": receiver["sync_peak"], "runtime_seconds": decode_runtime,
                     "decoded_regenerated_wav": str(regenerated_path.relative_to(ROOT))})

    with (exp_dir / "single_frame_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    _save_json(exp_dir / "noise_power.json", {
        name: {"mean_square": float(np.mean(values ** 2)), "peak": float(np.max(np.abs(values)))}
        for name, values in noise.items()
    })
    _save_plot(FIGURES / "noise_waveforms.png", "waveforms", np.arange(min(fs, clean_frame.size)) / fs,
               [("clean", clean_frame[:fs]),
                ("N1", noise["N1"][observation_samples:observation_samples + fs]),
                ("N2", noise["N2"][observation_samples:observation_samples + fs]),
                ("N3", noise["N3"][observation_samples:observation_samples + fs]),
                ("N4", noise["N4"][observation_samples:observation_samples + fs])])
    _save_plot(FIGURES / "noise_waveforms_presentation.png", "waveforms",
               np.arange(min(fs, clean_frame.size)) / fs,
               [("clean", clean_frame[:fs]),
                ("N1", noise["N1"][observation_samples:observation_samples + fs]),
                ("N2", noise["N2"][observation_samples:observation_samples + fs]),
                ("N3", noise["N3"][observation_samples:observation_samples + fs]),
                ("N4", noise["N4"][observation_samples:observation_samples + fs])],
               presentation=True, presentation_size=(6.5, 2.45))
    display_rows = [row for row in rows if row.get("case") == "N1_N2" and row.get("output_snr_db") != ""]
    _save_plot(FIGURES / "method_snr_N1_N2.png", "bars",
               [str(row["method"]) for row in display_rows],
               [row["output_snr_db"] for row in display_rows], "输出 SNR (dB)")
    _save_plot(FIGURES / "method_snr_N1_N2_presentation.png", "bars",
               [str(row["method"]) for row in display_rows],
               [row["output_snr_db"] for row in display_rows], "输出 SNR (dB)",
               presentation=True, presentation_size=(6.25, 4.35))
    reference = np.asarray(frame["preamble_waveform"])
    delay = int(0.15 * fs)
    sync_noise = make_noise_components(clean_frame, fs, seed + 7001,
        NoiseConfig(awgn_snr_db=20, narrowband_inr_db=30, burst_snr_db=20,
                    nonstationary_snr_db=20), output_length=observation_samples + clean_frame.size)
    sync_frame = clean_frame + sync_noise["N1"][observation_samples:observation_samples + clean_frame.size]
    sync_capture = np.r_[np.zeros(observation_samples), sync_frame]
    sync_observation = np.r_[np.zeros(delay), sync_capture, np.zeros(50)]
    sync_result = receive(sync_observation, 1, config)
    correlation_scores = np.asarray(sync_result["correlation_scores"])
    peak = int(np.argmax(correlation_scores))
    _save_correlation_plots(sync_result["correlation_indices"], correlation_scores,
                            sync_result["start"])
    frame_noise = {name: values[observation_samples:] for name, values in noise.items()}
    _save_plot(FIGURES / "time_frequency_clean_noisy.png", "time_frequency",
               clean_frame, combine_noise(clean_frame, frame_noise, ("N1", "N2", "N3")), fs)
    _save_plot(FIGURES / "time_frequency_clean_noisy_presentation.png", "time_frequency",
               clean_frame, combine_noise(clean_frame, frame_noise, ("N1", "N2", "N3")), fs,
               presentation=True, presentation_size=(7.2, 4.85))
    _save_json(exp_dir / "single_frame_summary.json", {
        "payload_hex": payload.hex(), "frame_samples": int(clean_frame.size),
        "payload_message": payload.decode("ascii"),
        "cases": len(cases), "algorithms_per_waveform_case": len(processed),
        "measured_input_snr_db": {case: next(row["input_snr_db"] for row in rows if row["case"] == case) for case, _ in cases},
    })
    demo_components = make_noise_components(clean_frame, fs, seed + 9000,
        NoiseConfig(awgn_snr_db=12, narrowband_inr_db=-8), output_length=capture_length)
    demo_capture = assemble_capture(clean_frame, demo_components, ("N1", "N2"), observation_samples)
    demo_noisy = demo_capture[observation_samples:]
    demo_result = receive(demo_capture, len(payload), config)
    write_audio(str(AUDIO / "message_demo_received_N1_N2.wav"), demo_capture, fs)
    if len(demo_result["payload"]) == len(payload):
        regenerated = make_frame(demo_result["payload"], config)
        write_audio(str(AUDIO / "message_demo_decoded_regenerated.wav"), regenerated["waveform"], fs)
    _save_json(exp_dir / "message_demo.json", {
        "message": message, "decoded_ascii": demo_result["payload"].decode("ascii", errors="replace"),
        "crc_ok": bool(demo_result["crc_ok"]), "ber": bit_error_rate(
            frame["payload_bits"], demo_result["bits"][len(config.preamble):]),
        "input_snr_db": actual_snr_db(clean_frame, demo_noisy),
        "sync_start": int(demo_result["start"] - observation_samples),
        "noise_observation_samples": observation_samples,
        "noise": "independent N1 + N2",
    })
    return rows


def _save_correlation_plots(indices, scores, peak) -> None:
    _save_plot(FIGURES / "correlation_sync.png", "correlation", indices, scores, peak)
    _save_plot(FIGURES / "correlation_sync_presentation.png", "correlation",
               indices, scores, peak, presentation=True,
               presentation_size=(6.35, 4.6))
    _save_plot(FIGURES / "correlation_sync_compact_presentation.png", "correlation",
               indices, scores, peak, presentation=True,
               presentation_size=(5.7, 3.2))


def _regenerate_correlation_plots(config: ModemConfig) -> None:
    frame_data = json.loads((EXPERIMENTS / "representative/frame.json").read_text(encoding="utf-8"))
    frame = make_frame(bytes.fromhex(frame_data["payload_hex"]), config)
    clean = np.asarray(frame["waveform"])
    observation_samples = int(frame_data["noise_observation_samples"])
    seed = int(frame_data["seed"])
    delay = int(0.15 * config.sample_rate)
    noise = make_noise_components(
        clean, config.sample_rate, seed + 7001,
        NoiseConfig(awgn_snr_db=20, narrowband_inr_db=30, burst_snr_db=20,
                    nonstationary_snr_db=20),
        output_length=observation_samples + clean.size)
    observed = np.r_[np.zeros(delay + observation_samples),
                     clean + noise["N1"][observation_samples:observation_samples + clean.size],
                     np.zeros(50)]
    result = receive(observed, 1, config)
    _save_correlation_plots(result["correlation_indices"],
                            np.asarray(result["correlation_scores"]), result["start"])


def _repeat_average_experiment(config: ModemConfig, seed: int = 873) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    payload = random_payload(seed + 1, 96)
    frame = make_frame(payload, config)
    clean = np.asarray(frame["waveform"])
    fs = config.sample_rate
    results: list[dict[str, object]] = []
    exp_dir = EXPERIMENTS / "repeat_average"
    exp_dir.mkdir(parents=True, exist_ok=True)
    structures = ("independent", "shared_tone", "independent_plus_shared_tone")
    for structure in structures:
        shared_tone = None
        if structure in {"shared_tone", "independent_plus_shared_tone"}:
            t = np.arange(clean.size) / fs
            shared_tone = 0.42 * np.sin(2 * np.pi * 1777 * t + 0.2)
        frames: list[np.ndarray] = []
        delays: list[int] = []
        for _ in range(16):
            delay = int(rng.integers(0, 160))
            delays.append(delay)
            components = make_noise_components(clean, fs, int(rng.integers(0, 2**31 - 1)),
                                               NoiseConfig(awgn_snr_db=0, narrowband_inr_db=-2,
                                                           tone_phase_rad=None),
                                               shared_noise={"N2": shared_tone} if shared_tone is not None else None)
            active_components = ("N2",) if structure == "shared_tone" else ("N1", "N2")
            noisy = combine_noise(clean, components, active_components)
            frames.append(np.r_[np.zeros(delay), noisy, np.zeros(200)])
        common_scale = min(1.0, 0.95 / max(max(float(np.max(np.abs(item))) for item in frames),
                                           float(np.max(np.abs(clean))), np.finfo(float).tiny))
        scaled_clean = clean * common_scale
        scaled_preamble = np.asarray(frame["preamble_waveform"]) * common_scale
        frames = [item * common_scale for item in frames]
        saved_frames = np.zeros((len(frames), clean.size + 360), dtype=np.float64)
        frame_lengths = np.asarray([item.size for item in frames], dtype=np.int32)
        for index, item in enumerate(frames):
            saved_frames[index, :item.size] = item
        np.savez_compressed(exp_dir / f"frames_{structure}.npz", frames=saved_frames,
                            frame_lengths=frame_lengths, delays=np.asarray(delays))
        _save_json(exp_dir / f"frames_{structure}.json", {"seed": seed, "structure": structure,
                    "true_delays_samples": delays, "common_wav_scale": common_scale})
        for count in (1, 2, 4, 8, 16):
            observed = frames[:count]
            single_frame_input_snr, ideal_average_snr, oracle = repeated_snr_reference(
                scaled_clean, observed, delays[:count])
            starts = []
            aligned = []
            unaligned = []
            alignment_started = time.perf_counter()
            for item in observed:
                found, _, _, _ = _find_preamble_start(item, scaled_preamble)
                starts.append(found)
                aligned.append(item[found:found + clean.size])
                unaligned.append(item[:clean.size])
            aligned_stack = np.stack([np.pad(x, (0, clean.size - x.size)) for x in aligned])
            unaligned_stack = np.stack([np.pad(x, (0, clean.size - x.size)) for x in unaligned])
            alignment_runtime = time.perf_counter() - alignment_started
            operation_times: dict[str, float] = {}
            started = time.perf_counter()
            averaged = aligned_stack.mean(axis=0)
            operation_times["correlation_aligned_average"] = alignment_runtime + time.perf_counter() - started
            started = time.perf_counter()
            direct = unaligned_stack.mean(axis=0)
            operation_times["unaligned_direct_average"] = time.perf_counter() - started
            started = time.perf_counter()
            operation_times["oracle_aligned_average"] = time.perf_counter() - started
            started = time.perf_counter()
            average_after = np.stack([wavelet_denoise(item)[0] for item in aligned_stack]).mean(axis=0)
            operation_times["wavelet_then_average"] = alignment_runtime + time.perf_counter() - started
            started = time.perf_counter()
            wavelet_after = wavelet_denoise(averaged)[0]
            operation_times["average_then_wavelet"] = alignment_runtime + time.perf_counter() - started
            outputs = {"unaligned_direct_average": direct,
                       "correlation_aligned_average": averaged, "oracle_aligned_average": oracle,
                       "wavelet_then_average": average_after, "average_then_wavelet": wavelet_after}
            if count == 1:
                outputs["single_frame"] = aligned_stack[0]
            start_errors = np.abs(np.asarray(starts) - np.asarray(delays[:count]))
            for method, result in outputs.items():
                metrics = waveform_metrics(scaled_clean, result)
                decode_started = time.perf_counter()
                decoded = receive(result, len(payload), config)
                decode_runtime = time.perf_counter() - decode_started
                ber = bit_error_rate(frame["payload_bits"], np.unpackbits(np.frombuffer(decoded["payload"][:len(payload)], dtype=np.uint8))) if len(decoded["payload"]) >= len(payload) else 1.0
                known_start = (None if method == "unaligned_direct_average"
                               else receive_at_start(result, len(payload), 0, config))
                known_start_ber = (None if known_start is None else
                                   bit_error_rate(frame["payload_bits"], np.unpackbits(
                                       np.frombuffer(known_start["payload"][:len(payload)], dtype=np.uint8)))
                                   if len(known_start["payload"]) >= len(payload) else 1.0)
                path = AUDIO / "repeat" / structure / f"M{count}_{method}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                write_audio(str(path), result, fs)
                results.append({"noise_structure": structure, "M": count, "method": method,
                                "single_frame_input_snr_db": single_frame_input_snr,
                                "ideal_aligned_average_snr_db": ideal_average_snr,
                                "snr_gain_vs_single_frame_db": metrics["snr_db"] - single_frame_input_snr,
                                "snr_gap_to_ideal_aligned_average_db": metrics["snr_db"] - ideal_average_snr,
                                "output_snr_db": metrics["snr_db"], "mse": metrics["mse"],
                                "ber": ber, "crc_ok": bool(decoded["crc_ok"]),
                                "ber_known_start": known_start_ber,
                                "crc_ok_known_start": (None if known_start is None else
                                                        bool(known_start["crc_ok"])),
                                "correlation_alignment_error_samples_mean": float(np.mean(start_errors)),
                                "runtime_seconds": operation_times.get(method, 0.0) + decode_runtime,
                                "wav": str(path.relative_to(ROOT))})

    with (exp_dir / "repeat_average_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in results for key in row}))
        writer.writeheader()
        writer.writerows(results)
    for structure in structures:
        for method in ("single_frame", "correlation_aligned_average", "oracle_aligned_average"):
            data = [r for r in results if r["noise_structure"] == structure and r["method"] == method]
            _save_plot(FIGURES / f"repeat_{structure}_{method}.png", "line",
                       [r["M"] for r in data], [(method, [r["output_snr_db"] for r in data])],
                        "重复次数 M", "输出 SNR (dB)")
    _save_plot(FIGURES / "repeat_noise_structure.png", "line",
               [1, 2, 4, 8, 16],
               [(structure.replace("_", " "), [next(r["output_snr_db"] for r in results if r["noise_structure"] == structure and r["M"] == m and r["method"] == "correlation_aligned_average") for m in (1, 2, 4, 8, 16)]) for structure in structures],
               "重复次数 M", "输出 SNR (dB)")
    _save_plot(FIGURES / "repeat_noise_structure_presentation.png", "line",
               [1, 2, 4, 8, 16],
               [(structure.replace("_", " "), [next(r["output_snr_db"] for r in results if r["noise_structure"] == structure and r["M"] == m and r["method"] == "correlation_aligned_average") for m in (1, 2, 4, 8, 16)]) for structure in structures],
               "重复次数 M", "输出 SNR (dB)", presentation=True,
               presentation_size=(7.1, 4.55))
    return results


def _evaluation_sweep(config: ModemConfig) -> list[dict[str, object]]:
    """Independent test-seed sweep; all methods receive the same noisy frame."""
    payload_bytes = 192
    snr_values = (-5.0, 0.0, 5.0, 10.0)
    cases = {"N1": ("N1",), "N2": ("N2",), "N1_N2": ("N1", "N2"),
             "N1_N2_N3": ("N1", "N2", "N3"), "N1_N2_N3_N4": ("N1", "N2", "N3", "N4")}
    seeds = (1201, 2309, 3413, 4517, 5623)
    observation_samples = int(0.256 * config.sample_rate)
    rows: list[dict[str, object]] = []
    for case, component_names in cases.items():
        for target_snr in snr_values:
            for seed in seeds:
                payload = random_payload(seed, payload_bytes)
                frame = make_frame(payload, config)
                clean = np.asarray(frame["waveform"])
                components = make_noise_components(clean, config.sample_rate, seed,
                    NoiseConfig(awgn_snr_db=0, narrowband_inr_db=-3, burst_snr_db=0,
                                nonstationary_snr_db=0),
                    output_length=observation_samples + clean.size)
                raw_noise = sum(components[name][observation_samples:] for name in component_names)
                target_noise_power = np.mean(clean**2) * 10 ** (-target_snr / 10)
                scale = np.sqrt(target_noise_power / np.mean(raw_noise**2))
                components = {name: value * scale for name, value in components.items()}
                capture = assemble_capture(clean, components, component_names, observation_samples)
                noisy = capture[observation_samples:]
                measured_input = actual_snr_db(clean, noisy)
                fs = config.sample_rate
                estimated_tones = detect_narrowband_interference(capture[:observation_samples], fs)
                operations = {
                    "unprocessed": lambda: noisy,
                    "wavelet_soft": lambda: wavelet_denoise(noisy, threshold_mode="soft")[0],
                    "bandpass": lambda: bandpass(noisy, fs, 900, 3200),
                    "psd_wiener": lambda: wiener_from_noise_segment(
                        noisy, capture[:observation_samples], fs, window_length=256),
                    "notch_bandpass_auto": lambda: bandpass(
                        notch(noisy, fs, estimated_tones), fs, 900, 3200),
                }
                if "N2" in component_names:
                    operations["notch_bandpass"] = lambda: bandpass(
                        notch(noisy, fs, (1777.0,)), fs, 900, 3200)
                    operations["hybrid_notch_wavelet"] = lambda: wavelet_denoise(
                        bandpass(notch(noisy, fs, (1777.0,)), fs, 900, 3200),
                        threshold_mode="soft")[0]
                else:
                    operations["hybrid_notch_wavelet"] = lambda: wavelet_denoise(
                        bandpass(noisy, fs, 900, 3200), threshold_mode="soft")[0]
                for method, operation in operations.items():
                    t0 = time.perf_counter()
                    output = operation()
                    elapsed = time.perf_counter() - t0
                    metrics = waveform_metrics(clean, output)
                    output_capture = np.r_[capture[:observation_samples], output]
                    decoded = receive(output_capture, payload_bytes, config)
                    known_start = receive_at_start(output_capture, payload_bytes, observation_samples, config)
                    ber = _payload_ber(frame["payload_bits"], decoded["payload"], payload_bytes)
                    known_start_ber = _payload_ber(frame["payload_bits"], known_start["payload"], payload_bytes)
                    rows.append({"case": case, "noise_components": "+".join(component_names),
                                 "target_snr_db": target_snr, "input_snr_db": measured_input,
                                 "seed": seed, "method": method, "output_snr_db": metrics["snr_db"],
                                 "snr_improvement_db": metrics["snr_db"] - measured_input,
                                 "mse": metrics["mse"], "ber": ber,
                                 "ber_known_start_diagnostic": known_start_ber,
                                 "preamble_errors_correlation_sync": decoded["preamble_errors"],
                                 "preamble_errors_known_start": known_start["preamble_errors"],
                                 "crc_ok": bool(decoded["crc_ok"]), "sync_start": int(decoded["start"]),
                                 "sync_error_samples": int(decoded["start"] - observation_samples),
                                 "noise_observation_samples": observation_samples,
                                 "estimated_tones_hz": ";".join(f"{frequency:.2f}" for frequency in estimated_tones),
                                 "runtime_seconds": elapsed})
    exp_dir = EXPERIMENTS / "evaluation"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "seed_sweep.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["case"]), float(row["target_snr_db"]), str(row["method"])), []).append(row)
    summary = []
    for (case, target_snr, method), group in sorted(grouped.items()):
        summary.append({"case": case, "target_snr_db": target_snr, "method": method,
                        "n_seeds": len(group),
                        "input_snr_mean_db": float(np.mean([r["input_snr_db"] for r in group])),
                        "output_snr_mean_db": float(np.mean([r["output_snr_db"] for r in group])),
                        "output_snr_std_db": float(np.std([r["output_snr_db"] for r in group], ddof=1)),
                        "ber_mean": float(np.mean([r["ber"] for r in group])),
                        "ber_std": float(np.std([r["ber"] for r in group], ddof=1)),
                        "ber_known_start_mean": float(np.mean([r["ber_known_start_diagnostic"] for r in group])),
                        "ber_known_start_std": float(np.std([r["ber_known_start_diagnostic"] for r in group], ddof=1)),
                        "crc_success_rate": float(np.mean([r["crc_ok"] for r in group])),
                        "runtime_mean_seconds": float(np.mean([r["runtime_seconds"] for r in group]))})
    _save_json(exp_dir / "seed_sweep_summary.json", summary)
    comparison_methods = ("unprocessed", "wavelet_soft", "bandpass", "psd_wiener",
                          "notch_bandpass_auto", "notch_bandpass", "hybrid_notch_wavelet")
    _save_plot(FIGURES / "snr_sweep_N1_N2.png", "line", list(snr_values),
               [(method, [next(r["output_snr_mean_db"] for r in summary if r["case"] == "N1_N2" and r["target_snr_db"] == snr and r["method"] == method)
                          for snr in snr_values]) for method in comparison_methods],
               "输入 SNR (dB)", "平均输出 SNR (dB)")
    _save_plot(FIGURES / "snr_sweep_N1_N2_presentation.png", "line", list(snr_values),
               [(method, [next(r["output_snr_mean_db"] for r in summary if r["case"] == "N1_N2" and r["target_snr_db"] == snr and r["method"] == method)
                          for snr in snr_values]) for method in comparison_methods],
               "输入 SNR (dB)", "平均输出 SNR (dB)", presentation=True,
               presentation_size=(6.15, 3.7))
    _save_plot(FIGURES / "ber_sweep_N1_N2.png", "line", list(snr_values),
               [(method, [next(r["ber_mean"] for r in summary if r["case"] == "N1_N2" and r["target_snr_db"] == snr and r["method"] == method)
                          for snr in snr_values]) for method in comparison_methods],
               "输入 SNR (dB)", "有效载荷平均 BER")
    _save_plot(FIGURES / "wavelet_sync_comparison.png", "line", list(snr_values), [
        ("未处理：相关同步", [next(r["ber_mean"] for r in summary if r["case"] == "N1_N2" and r["target_snr_db"] == snr and r["method"] == "unprocessed") for snr in snr_values]),
        ("未处理：真实帧起点", [next(r["ber_known_start_mean"] for r in summary if r["case"] == "N1_N2" and r["target_snr_db"] == snr and r["method"] == "unprocessed") for snr in snr_values]),
        ("小波：相关同步", [next(r["ber_mean"] for r in summary if r["case"] == "N1_N2" and r["target_snr_db"] == snr and r["method"] == "wavelet_soft") for snr in snr_values]),
        ("小波：真实帧起点", [next(r["ber_known_start_mean"] for r in summary if r["case"] == "N1_N2" and r["target_snr_db"] == snr and r["method"] == "wavelet_soft") for snr in snr_values]),
    ], "目标输入 SNR (dB)", "有效载荷平均 BER")
    _save_plot(FIGURES / "wavelet_sync_comparison_presentation.png", "line", list(snr_values), [
        ("小波：相关同步", [next(r["ber_mean"] for r in summary if r["case"] == "N1_N2" and r["target_snr_db"] == snr and r["method"] == "wavelet_soft") for snr in snr_values]),
        ("小波：真实帧起点", [next(r["ber_known_start_mean"] for r in summary if r["case"] == "N1_N2" and r["target_snr_db"] == snr and r["method"] == "wavelet_soft") for snr in snr_values]),
    ], "目标输入 SNR (dB)", "有效载荷平均 BER", presentation=True,
        presentation_size=(5.75, 3.7))
    return rows


def _find_preamble_start(samples: np.ndarray, template: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray]:
    from shared.signal_core import estimate_delay
    return estimate_delay(samples, template)


def run_all(*, correlation_figures_only: bool = False) -> None:
    config = ModemConfig()
    if correlation_figures_only:
        _regenerate_correlation_plots(config)
        return
    start = time.perf_counter()
    _window_resolution_experiment(config)
    single = _single_frame_experiment(config)
    repeat = _repeat_average_experiment(config)
    sweep = _evaluation_sweep(config)
    elapsed = time.perf_counter() - start
    summary = {
        "elapsed_seconds": elapsed,
        "single_rows": len(single),
        "repeat_rows": len(repeat),
        "sweep_rows": len(sweep),
        "evaluation_seeds": 5,
        "runtime_environment": "Python; benchmark values are measured by runner",
    }
    _save_json(EXPERIMENTS / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_all()
