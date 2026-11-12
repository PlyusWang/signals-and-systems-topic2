import numpy as np
import soundfile as sf
import threading
from scipy import signal
import pytest

from shared.signal_core import (
    correlation, cwt_view, detect_narrowband_interference, estimate_delay,
    one_sided_spectrum, read_audio, stft_view, wiener_from_noise_segment,
    wavelet_denoise, waveform_metrics,
)


def test_fft_sine_dc_and_nyquist_amplitude_scaling():
    n = 1024
    t = np.arange(n)
    sine = 0.6 * np.sin(2 * np.pi * 64 * t / n)
    spec = one_sided_spectrum(sine, 1024, window="rectangular")
    assert np.isclose(spec.amplitude[64], 0.6, atol=1e-12)
    dc = one_sided_spectrum(np.full(n, 0.25), 1024, window="rectangular")
    assert np.isclose(dc.amplitude[0], 0.25, atol=1e-12)
    nyquist = one_sided_spectrum((-1.0) ** t, 1024, window="rectangular")
    assert np.isclose(nyquist.amplitude[-1], 1.0, atol=1e-12)


def test_fft_parseval_for_raw_fft_and_frequency_grid():
    rng = np.random.default_rng(5)
    x = rng.normal(size=127)
    spec = one_sided_spectrum(x, 8000, window="rectangular")
    raw = np.fft.rfft(x)
    energy_frequency = (raw[0] ** 2 + 2 * np.sum(np.abs(raw[1:]) ** 2)) / x.size
    assert np.isclose(energy_frequency, np.sum(x * x), rtol=1e-12)
    assert np.isclose(spec.frequency_hz[-1], 8000 * 63 / 127)
    assert np.isclose(spec.bin_spacing_hz, 8000 / 127)


def test_fft_zero_padding_reports_bin_spacing_separately_from_resolution():
    fs = 8000
    x = np.sin(2 * np.pi * 440 * np.arange(8000) / fs)
    spec = one_sided_spectrum(x, fs, n_fft=16384, window="hann")
    assert spec.sample_count == 8000
    assert np.isclose(spec.observation_duration_s, 1.0)
    assert np.isclose(spec.bin_spacing_hz, 0.48828125)
    assert np.isclose(spec.window_main_lobe_width_hz, 4.0)
    assert spec.bin_spacing_hz < spec.window_main_lobe_width_hz


def test_stft_reconstructs_with_overlap_add():
    rng = np.random.default_rng(2)
    x = rng.normal(size=4096)
    f, t, z = signal.stft(x, 8000, nperseg=256, noverlap=192, boundary="zeros")
    _, y = signal.istft(z, 8000, nperseg=256, noverlap=192, boundary=True)
    assert np.allclose(y[:x.size], x, atol=1e-10)
    f_view, t_view, z_view = stft_view(x, 8000, window_length=256, overlap=0.75)
    assert z_view.shape == (f_view.size, t_view.size)
    assert f_view[-1] == 4000


def test_cwt_peak_tracks_tone_in_hz():
    fs = 8000
    t = np.arange(4096) / fs
    x = np.sin(2 * np.pi * 1000 * t)
    frequencies, times, magnitude = cwt_view(x, fs, maximum_points=4096,
                                              frequency_limits=(300, 3000))
    middle = magnitude[:, len(times) // 4: 3 * len(times) // 4].mean(axis=1)
    assert abs(frequencies[np.argmax(middle)] - 1000) < 100
    assert times[-1] <= t[-1]
    assert np.allclose(np.diff(frequencies), np.diff(frequencies)[0])


def test_cwt_uniform_hz_rows_track_two_tones_and_chirp():
    fs = 8000
    t = np.arange(8000) / fs
    two_tone = np.sin(2 * np.pi * 700 * t) + 0.8 * np.sin(2 * np.pi * 1800 * t)
    frequencies, _, magnitude = cwt_view(two_tone, fs, maximum_points=8000,
                                          frequency_limits=(300, 3000))
    mean = magnitude[:, 1500:6500].mean(axis=1)
    peaks, _ = signal.find_peaks(mean, distance=12)
    peak_hz = frequencies[peaks]
    assert np.min(np.abs(peak_hz - 700)) < 100
    assert np.min(np.abs(peak_hz - 1800)) < 120

    chirp = signal.chirp(t, f0=500, t1=t[-1], f1=2500)
    frequencies, times, magnitude = cwt_view(chirp, fs, maximum_points=8000,
                                              frequency_limits=(300, 3000))
    for fraction in (0.2, 0.5, 0.8):
        column = int(fraction * (times.size - 1))
        expected = 500 + 2000 * fraction
        measured = frequencies[np.argmax(magnitude[:, column])]
        assert abs(measured - expected) < 180


def test_cwt_long_audio_retains_carrier_band_after_time_column_reduction():
    fs = 16000
    duration = 60
    seconds = np.arange(fs * duration) / fs
    x = np.sin(2 * np.pi * 1500 * seconds) + 0.8 * np.sin(2 * np.pi * 2500 * seconds)
    frequencies, times, magnitude = cwt_view(x, fs, maximum_points=1200)
    middle = magnitude[:, magnitude.shape[1] // 4: 3 * magnitude.shape[1] // 4].mean(axis=1)
    peaks, _ = signal.find_peaks(middle, distance=8)
    peak_hz = frequencies[peaks]
    assert frequencies[-1] > 7000
    assert abs(times[-1] - (duration - 1 / fs)) < 1 / fs
    assert magnitude.shape == (frequencies.size, 1200)
    assert np.min(np.abs(peak_hz - 1500)) < 120
    assert np.min(np.abs(peak_hz - 2500)) < 160


def test_cwt_stops_between_blocks_when_cancelled(monkeypatch):
    import shared.signal_core as core

    cancel_event = threading.Event()
    calls = 0

    def fake_cwt(segment, scales, wavelet, *, sampling_period, method):
        nonlocal calls
        calls += 1
        cancel_event.set()
        frequencies = pywt.central_frequency(wavelet) / (scales * sampling_period)
        return np.zeros((len(scales), len(segment))), frequencies

    import pywt
    monkeypatch.setattr(core.pywt, "cwt", fake_cwt)
    with pytest.raises(core.AnalysisCancelled):
        core.cwt_view(np.zeros(70000), 8000, voices_per_octave=4,
                      maximum_points=1200, frequency_limits=(200, 3000),
                      cancel_event=cancel_event)
    assert calls == 1


def test_read_audio_selects_left_right_and_mixed_channels(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack((np.full(32, 0.25), np.full(32, -0.5))), 8000)
    left, fs = read_audio(str(path), channel="left")
    right, _ = read_audio(str(path), channel="right")
    mixed, _ = read_audio(str(path), channel="mix")
    assert fs == 8000
    assert np.allclose(left, 0.25, atol=1e-4)
    assert np.allclose(right, -0.5, atol=1e-4)
    assert np.allclose(mixed, -0.125, atol=1e-4)


def test_correlation_lag_sign_and_template_offset():
    rng = np.random.default_rng(3)
    template = rng.normal(size=150)
    delayed = np.r_[np.zeros(37), template, np.zeros(20)]
    result = correlation(delayed, template)
    assert result.peak_lag == 37
    assert correlation(template, delayed).peak_lag == -37
    start, peak, _, _ = estimate_delay(delayed, template)
    assert start == 37
    assert peak > 0.999


def test_delay_estimation_ignores_fft_roundoff_in_silent_windows():
    rng = np.random.default_rng(19)
    template = rng.normal(size=256)
    delay = 6000
    observed = np.r_[np.zeros(delay), template, np.zeros(1000)]
    start, score, lags, scores = estimate_delay(observed, template)
    assert start == delay
    assert np.isclose(score, 1.0, atol=1e-10)
    assert np.all(np.isfinite(scores))
    assert np.max(scores) <= 1.0 + 1e-10
    assert np.all(scores[:delay - template.size + 1] == 0)


def test_wavelet_reconstruction_length_and_zero_noise():
    rng = np.random.default_rng(4)
    x = rng.normal(size=2051)
    clean, details = wavelet_denoise(x, sigma=0)
    assert clean.size == x.size
    assert np.allclose(clean, x, atol=1e-10)
    noisy = x + 0.5 * rng.normal(size=x.size)
    soft, _ = wavelet_denoise(noisy, threshold_mode="soft")
    hard, _ = wavelet_denoise(noisy, threshold_mode="hard")
    assert soft.size == hard.size == x.size
    assert details["level"] >= 1
    assert waveform_metrics(x, soft)["mse"] >= 0


def test_wiener_noise_power_uses_the_same_stft_spectrum_scaling(monkeypatch):
    fs = 8000
    rng = np.random.default_rng(91)
    noise = rng.normal(size=2048)
    samples = 0.4 * np.sin(2 * np.pi * 1000 * np.arange(2048) / fs)
    samples += 0.3 * rng.normal(size=samples.size)
    nperseg, noverlap = 256, 192
    _, _, observed = signal.stft(samples, fs, window="hann", nperseg=nperseg,
                                 noverlap=noverlap, nfft=nperseg, boundary="zeros",
                                 padded=True, scaling="spectrum")
    _, _, noise_spectrum = signal.stft(noise, fs, window="hann", nperseg=nperseg,
                                       noverlap=noverlap, nfft=nperseg, boundary=None,
                                       padded=False, scaling="spectrum")
    noise_power = np.mean(np.abs(noise_spectrum) ** 2, axis=1, keepdims=True)
    expected_gain = np.maximum(0.05, 1 - noise_power / np.maximum(
        np.abs(observed) ** 2, np.finfo(float).tiny))
    original_istft = signal.istft
    captured = {}

    def capture_istft(transformed, *args, **kwargs):
        captured["transformed"] = transformed.copy()
        return original_istft(transformed, *args, **kwargs)

    monkeypatch.setattr(signal, "istft", capture_istft)
    result = wiener_from_noise_segment(samples, noise, fs, window_length=nperseg)
    assert result.size == samples.size
    assert np.allclose(captured["transformed"], observed * expected_gain, rtol=1e-11, atol=1e-12)


def test_narrowband_detector_finds_tone_in_noise_observation_without_awgn_false_peak():
    fs = 16000
    time = np.arange(4096) / fs
    detected = detect_narrowband_interference(
        0.2 * np.sin(2 * np.pi * 1777 * time) + 0.08 * np.random.default_rng(27).normal(size=time.size), fs)
    assert detected.size == 1
    assert abs(detected[0] - 1777) < 5
    for seed in (3, 17, 81):
        awgn = np.random.default_rng(seed).normal(size=time.size)
        assert detect_narrowband_interference(awgn, fs).size == 0
