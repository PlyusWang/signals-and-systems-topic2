"""Numerically explicit signal-processing primitives for the project."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Callable, Iterable

import numpy as np
import pywt
import soundfile as sf
from scipy.ndimage import median_filter
from scipy import signal


def read_audio(path: str, *, channel: str = "mix") -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if channel not in {"mix", "left", "right"}:
        raise ValueError("channel must be mix, left, or right")
    if channel == "left" or data.shape[1] == 1:
        mono = data[:, 0]
    elif channel == "right":
        mono = data[:, 1]
    else:
        mono = data.mean(axis=1)
    return mono, int(sample_rate)


def write_audio(path: str, samples: np.ndarray, sample_rate: int) -> None:
    data = np.asarray(samples, dtype=np.float64)
    if sample_rate <= 0 or data.ndim not in (1, 2) or not np.all(np.isfinite(data)):
        raise ValueError("audio must be finite mono/multichannel data with positive sample rate")
    sf.write(path, data, sample_rate, subtype="PCM_16")


def _window(name: str, length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("window length must be positive")
    if name == "rectangular":
        return np.ones(length, dtype=np.float64)
    return signal.get_window(name, length, fftbins=True).astype(np.float64)


@dataclass(frozen=True)
class Spectrum:
    frequency_hz: np.ndarray
    amplitude: np.ndarray
    dbfs: np.ndarray
    n_fft: int
    bin_spacing_hz: float
    coherent_gain: float
    observation_duration_s: float
    sample_count: int
    window_main_lobe_width_hz: float


def one_sided_spectrum(
    samples: np.ndarray, sample_rate: int, *, n_fft: int | None = None, window: str = "hann"
) -> Spectrum:
    """Peak-amplitude calibrated one-sided spectrum; DC/Nyquist are not doubled."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size == 0 or sample_rate <= 0:
        raise ValueError("nonempty samples and positive sample rate are required")
    size = int(n_fft or x.size)
    if size < x.size:
        raise ValueError("n_fft must be at least sample count")
    w = _window(window, x.size)
    coherent_sum = float(w.sum())
    if coherent_sum <= 0:
        raise ValueError("window coherent gain must be positive")
    amplitude = np.abs(np.fft.rfft(x * w, n=size)) / coherent_sum
    if size % 2 == 0:
        amplitude[1:-1] *= 2.0
    else:
        amplitude[1:] *= 2.0
    dbfs = 20.0 * np.log10(np.maximum(amplitude, np.finfo(float).tiny))
    main_lobe_bins = {"rectangular": 2.0, "hann": 4.0, "hamming": 4.0,
                      "blackman": 6.0}.get(window, 4.0)
    duration = x.size / sample_rate
    return Spectrum(np.fft.rfftfreq(size, 1.0 / sample_rate), amplitude, dbfs,
                    size, sample_rate / size, coherent_sum / x.size,
                    duration, x.size, main_lobe_bins / duration)


def stft_parameters(sample_count: int, *, window_length: int = 512,
                    overlap: float = .75, n_fft: int | None = None) -> dict[str, int]:
    """Resolve the exact parameters shared by STFT execution and its GUI readout."""
    if sample_count < 1 or window_length < 1 or not 0 <= overlap < 1:
        raise ValueError("positive sample/window lengths and overlap in [0, 1) are required")
    nperseg = min(int(window_length), sample_count)
    noverlap = min(int(round(nperseg * overlap)), nperseg - 1)
    return {"window_length": nperseg, "overlap_samples": noverlap,
            "fft_length": max(nperseg, int(n_fft or nperseg)),
            "hop_samples": nperseg - noverlap}


def stft_view(samples: np.ndarray, sample_rate: int, *, window_length: int = 512,
              overlap: float = 0.75, n_fft: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size == 0 or not 0 <= overlap < 1:
        raise ValueError("nonempty samples and overlap in [0, 1) are required")
    parameters = stft_parameters(x.size, window_length=window_length, overlap=overlap, n_fft=n_fft)
    nperseg, noverlap, nfft = (parameters[key] for key in
                              ("window_length", "overlap_samples", "fft_length"))
    frequency, time, values = signal.stft(x, sample_rate, window="hann", nperseg=nperseg,
                                          noverlap=noverlap, nfft=nfft, boundary=None,
                                          padded=False, scaling="spectrum")
    return frequency, time, values


class AnalysisCancelled(RuntimeError):
    """Raised when a long-running analysis is no longer needed by its caller."""


def cwt_view(samples: np.ndarray, sample_rate: int, *, voices_per_octave: int = 12,
             maximum_points: int = 12000,
             frequency_limits: tuple[float, float] | None = None,
             wavelet: str = "cmor1.5-1.0",
             cancel_event: Event | None = None,
             progress_callback: Callable[[int, int], None] | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CWT magnitude with frequencies in Hz and times in seconds."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size == 0 or sample_rate <= 0 or voices_per_octave < 1 or maximum_points < 32:
        raise ValueError("nonempty samples and valid analysis parameters are required")
    low, high = frequency_limits or (max(10.0, sample_rate / 1024), sample_rate * 0.48)
    high = min(high, sample_rate * 0.48)
    if not 0 < low < high:
        raise ValueError("frequency limits must be positive and ordered below Nyquist")
    count = max(24, int(np.ceil(voices_per_octave * np.log2(high / low))) + 1)
    targets = np.geomspace(high, low, count)
    scales = pywt.central_frequency(wavelet) * sample_rate / targets
    output_indices = np.unique(np.rint(np.linspace(
        0, x.size - 1, min(x.size, maximum_points))).astype(np.int64))
    output = np.empty((count, output_indices.size), dtype=np.float64)
    wavelet_support = pywt.ContinuousWavelet(wavelet)
    halo = int(np.ceil(np.max(scales) * max(abs(wavelet_support.lower_bound),
                                             abs(wavelet_support.upper_bound))))
    core_size = min(65536, max(16384, x.size))
    block_count = (x.size + core_size - 1) // core_size
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled("CWT analysis superseded")
    if progress_callback is not None:
        progress_callback(0, block_count)
    for core_start in range(0, x.size, core_size):
        if cancel_event is not None and cancel_event.is_set():
            raise AnalysisCancelled("CWT analysis superseded")
        core_end = min(core_start + core_size, x.size)
        segment_start = max(0, core_start - halo)
        segment_end = min(x.size, core_end + halo)
        coefficients, frequencies = pywt.cwt(
            x[segment_start:segment_end], scales, wavelet,
            sampling_period=1.0 / sample_rate, method="fft")
        selected_start = int(np.searchsorted(output_indices, core_start, side="left"))
        selected_end = int(np.searchsorted(output_indices, core_end, side="left"))
        if selected_start == selected_end:
            if progress_callback is not None:
                progress_callback(core_start // core_size + 1, block_count)
            continue
        columns = output_indices[selected_start:selected_end] - segment_start
        order = np.argsort(frequencies)
        frequencies = frequencies[order]
        magnitude = np.abs(coefficients[order, :][:, columns])
        linear_frequencies = np.linspace(float(frequencies[0]), float(frequencies[-1]), count)
        upper = np.clip(np.searchsorted(frequencies, linear_frequencies, side="right"), 1,
                        frequencies.size - 1)
        lower = upper - 1
        weight = ((linear_frequencies - frequencies[lower])
                  / (frequencies[upper] - frequencies[lower]))[:, None]
        output[:, selected_start:selected_end] = (
            magnitude[lower] * (1.0 - weight) + magnitude[upper] * weight)
        if progress_callback is not None:
            progress_callback(core_start // core_size + 1, block_count)
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled("CWT analysis superseded")
    times = output_indices.astype(np.float64) / sample_rate
    linear_frequencies = np.linspace(float(frequencies.min()), float(frequencies.max()), count)
    return linear_frequencies, times, output


@dataclass(frozen=True)
class Correlation:
    lags: np.ndarray
    values: np.ndarray
    peak_lag: int
    peak_value: float


def correlation(first: np.ndarray, second: np.ndarray | None = None, *,
                demean: bool = True, normalize: bool = True) -> Correlation:
    """Full correlation; positive lag means `first` occurs later than `second`."""
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = x if second is None else np.asarray(second, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("correlation inputs must be nonempty")
    if demean:
        x, y = x - x.mean(), y - y.mean()
    values = signal.correlate(x, y, mode="full", method="fft")
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    if normalize:
        denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
        values = values / denominator if denominator > 0 else np.zeros_like(values)
    peak_index = int(np.argmax(np.abs(values)))
    return Correlation(lags, values, int(lags[peak_index]), float(values[peak_index]))


def estimate_delay(samples: np.ndarray, template: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Return template start index and normalized valid-correlation peak."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    t = np.asarray(template, dtype=np.float64).reshape(-1)
    if not x.size or not t.size or t.size > x.size:
        raise ValueError("template must be nonempty and no longer than observation")
    values = signal.correlate(x, t, mode="valid", method="fft")
    template_energy = np.sum(t * t)
    cumulative_energy = np.r_[0.0, np.cumsum(x * x, dtype=np.float64)]
    window_energy = cumulative_energy[t.size:] - cumulative_energy[:-t.size]
    scores = np.zeros(values.size, dtype=np.float64)
    minimum_energy = max(float(np.max(window_energy)) * 1e-12, np.finfo(float).tiny)
    valid = (window_energy > minimum_energy) & (template_energy > 0)
    scores[valid] = np.abs(values[valid]) / np.sqrt(template_energy * window_energy[valid])
    index = int(np.argmax(scores))
    return index, float(scores[index]), np.arange(values.size), scores


def wavelet_denoise(samples: np.ndarray, *, wavelet: str = "sym8", level: int | None = None,
                    threshold_mode: str = "soft", sigma: float | None = None
                    ) -> tuple[np.ndarray, dict[str, object]]:
    """DWT thresholding with robust finest-detail MAD noise estimation."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if threshold_mode not in {"soft", "hard"}:
        raise ValueError("threshold_mode must be soft or hard")
    if x.size < 8:
        return x.copy(), {"sigma": 0.0, "threshold": 0.0, "level": 0, "detail_energy": []}
    wav = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(x.size, wav.dec_len)
    chosen = min(max_level, level if level is not None else max(1, int(np.floor(np.log2(x.size)) - 4)))
    if chosen < 1:
        return x.copy(), {"sigma": 0.0, "threshold": 0.0, "level": 0, "detail_energy": []}
    coeffs = pywt.wavedec(x, wav, mode="symmetric", level=chosen)
    finest = coeffs[-1]
    estimate = float(sigma) if sigma is not None else float(
        np.median(np.abs(finest - np.median(finest))) / 0.6744897501960817)
    threshold = estimate * np.sqrt(2.0 * np.log(x.size))
    processed = [coeffs[0]] + [pywt.threshold(c, threshold, mode=threshold_mode) for c in coeffs[1:]]
    reconstructed = pywt.waverec(processed, wav, mode="symmetric")[:x.size]
    energies = [float(np.mean(c * c)) for c in coeffs[1:]]
    return reconstructed, {"sigma": estimate, "threshold": float(threshold), "level": chosen,
                           "wavelet": wavelet, "mode": threshold_mode, "detail_energy": energies}


def wiener_from_noise_segment(samples: np.ndarray, noise_segment: np.ndarray, sample_rate: int, *,
                              window_length: int = 512, overlap: float = 0.75,
                              gain_floor: float = 0.05) -> np.ndarray:
    """STFT Wiener magnitude estimator using a matched one-sided spectrum convention."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    n = np.asarray(noise_segment, dtype=np.float64).reshape(-1)
    if not x.size or not n.size or sample_rate <= 0 or not 0 <= gain_floor <= 1:
        raise ValueError("nonempty signal/noise, positive rate and valid gain floor are required")
    nperseg = min(window_length, x.size)
    noverlap = min(int(round(nperseg * overlap)), nperseg - 1)
    nfft = nperseg
    _, _, z = signal.stft(x, sample_rate, window="hann", nperseg=nperseg,
                          noverlap=noverlap, nfft=nfft, boundary="zeros", padded=True, scaling="spectrum")
    nseg = min(nperseg, n.size)
    n_over = min(noverlap, nseg - 1)
    _, _, noise_stft = signal.stft(
        n, sample_rate, window="hann", nperseg=nseg, noverlap=n_over,
        nfft=nfft, boundary=None, padded=False, scaling="spectrum")
    noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1)
    noise_power = noise_power[:, np.newaxis]
    gain = np.maximum(gain_floor, 1.0 - noise_power / np.maximum(np.abs(z) ** 2,
                                                                  np.finfo(float).tiny))
    _, result = signal.istft(z * gain, sample_rate, window="hann", nperseg=nperseg,
                             noverlap=noverlap, nfft=nfft, boundary=True, scaling="spectrum")
    if result.size < x.size:
        result = np.pad(result, (0, x.size - result.size))
    return result[:x.size]


def detect_narrowband_interference(samples: np.ndarray, sample_rate: int, *,
                                   frequency_range_hz: tuple[float, float] = (900.0, 3200.0),
                                   prominence_db: float = 8.0,
                                   minimum_separation_hz: float = 80.0,
                                   maximum_tones: int = 4) -> np.ndarray:
    """Find prominent narrowband peaks in a noise-only observation segment."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    low_hz, high_hz = frequency_range_hz
    if (x.size < 64 or sample_rate <= 0 or not 0 <= low_hz < high_hz <= sample_rate / 2
            or prominence_db <= 0 or minimum_separation_hz <= 0 or maximum_tones < 1):
        raise ValueError("invalid samples or narrowband detector parameters")
    nperseg = min(2048, x.size)
    frequency, power = signal.welch(
        x, sample_rate, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, nfft=nperseg, detrend="constant", scaling="density")
    bin_spacing = sample_rate / nperseg
    smooth_size = max(9, int(round(240.0 / bin_spacing)) | 1)
    background = median_filter(power, size=smooth_size, mode="nearest")
    relative_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny)
                                / np.maximum(background, np.finfo(float).tiny))
    distance_bins = max(1, int(np.ceil(minimum_separation_hz / bin_spacing)))
    peaks, properties = signal.find_peaks(
        relative_db, height=prominence_db, prominence=prominence_db, distance=distance_bins)
    eligible = (frequency[peaks] >= low_hz) & (frequency[peaks] <= high_hz)
    peaks = peaks[eligible]
    prominence = properties["prominences"][eligible]
    ranked = peaks[np.argsort(prominence)[::-1][:maximum_tones]]
    log_power = 10 * np.log10(np.maximum(power, np.finfo(float).tiny))
    refined = []
    for peak in ranked:
        offset = 0.0
        if 0 < peak < log_power.size - 1:
            left, center, right = log_power[peak - 1:peak + 2]
            denominator = left - 2 * center + right
            if denominator != 0:
                offset = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
        refined.append(frequency[peak] + offset * bin_spacing)
    return np.asarray(sorted(refined), dtype=np.float64)


def bandpass(samples: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if not 0 < low_hz < high_hz < sample_rate / 2:
        raise ValueError("band edges must lie between zero and Nyquist")
    sos = signal.butter(6, (low_hz, high_hz), btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, x) if x.size > 3 * (2 * len(sos) + 1) else signal.sosfilt(sos, x)


def notch(samples: np.ndarray, sample_rate: int, frequencies_hz: Iterable[float],
          quality: float = 25.0) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float64).reshape(-1).copy()
    for frequency in frequencies_hz:
        if 0 < frequency < sample_rate / 2:
            b, a = signal.iirnotch(frequency, quality, fs=sample_rate)
            result = signal.filtfilt(b, a, result) if result.size > 3 * max(len(a), len(b)) else signal.lfilter(b, a, result)
    return result


def waveform_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    x, y = np.asarray(reference, dtype=np.float64).reshape(-1), np.asarray(estimate, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n == 0:
        raise ValueError("metric inputs must be nonempty")
    x, y = x[:n], y[:n]
    error = x - y
    snr = 10 * np.log10(np.mean(x * x) / max(np.mean(error * error), np.finfo(float).tiny))
    return {"snr_db": float(snr), "mse": float(np.mean(error * error)), "rmse": float(np.sqrt(np.mean(error * error)))}


def measured_snr(reference: np.ndarray, observed: np.ndarray) -> float:
    x, y = np.asarray(reference, dtype=np.float64).reshape(-1), np.asarray(observed, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n == 0:
        raise ValueError("SNR inputs must be nonempty")
    return float(10 * np.log10(np.sum(x[:n] ** 2) /
                               max(np.sum((x[:n] - y[:n]) ** 2), np.finfo(float).tiny)))
