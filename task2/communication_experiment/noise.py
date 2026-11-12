from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class NoiseConfig:
    awgn_snr_db: float = 0.0
    narrowband_inr_db: float = -3.0
    burst_snr_db: float = -2.0
    nonstationary_snr_db: float = -2.0
    tone_hz: float = 1777.0
    burst_count: int = 3
    tone_phase_rad: float | None = None


def _power(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def _scaled(raw: np.ndarray, desired_power: float) -> np.ndarray:
    p = _power(raw)
    return raw * np.sqrt(desired_power / p) if p > 0 else np.zeros_like(raw)


def make_noise_components(signal_part: np.ndarray, sample_rate: int, seed: int,
                          config: NoiseConfig = NoiseConfig(), *,
                          shared_noise: dict[str, np.ndarray] | None = None,
                          output_length: int | None = None) -> dict[str, np.ndarray]:
    """Generate four independently retained noise sequences at specified powers."""
    s = np.asarray(signal_part, dtype=np.float64).reshape(-1)
    if not s.size or sample_rate <= 0:
        raise ValueError("signal and sample rate must be valid")
    rng = np.random.default_rng(seed)
    p_signal = _power(s)
    shared_noise = shared_noise or {}
    n = s.size if output_length is None else int(output_length)
    if n < s.size:
        raise ValueError("output_length cannot be shorter than the reference signal")

    awgn = rng.normal(size=n)
    awgn = _scaled(awgn, p_signal * 10 ** (-config.awgn_snr_db / 10))

    time = np.arange(n) / sample_rate
    phase = rng.uniform(-np.pi, np.pi) if config.tone_phase_rad is None else config.tone_phase_rad
    tone = np.sin(2 * np.pi * config.tone_hz * time + phase)
    tone = _scaled(tone, p_signal * 10 ** (config.narrowband_inr_db / 10))

    burst = np.zeros(n)
    width = max(2, int(round(0.012 * sample_rate)))
    positions = np.linspace(int(0.1 * n), int(0.9 * n), config.burst_count, dtype=int)
    for position in positions:
        lo = max(0, position - width // 2)
        hi = min(n, lo + width)
        burst[lo:hi] += rng.normal(size=hi - lo)
    burst = _scaled(burst, p_signal * 10 ** (-config.burst_snr_db / 10))

    white = rng.normal(size=n)
    b, a = signal.butter(2, 0.08)
    colored = signal.lfilter(b, a, white)
    sweep_frequency = np.linspace(900.0, min(3600.0, sample_rate * 0.4), n)
    phase = 2 * np.pi * np.cumsum(sweep_frequency) / sample_rate
    sweep = np.sin(phase)
    envelope = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / max(1, n)))
    nonstationary = envelope * colored + 0.35 * sweep
    nonstationary = _scaled(nonstationary, p_signal * 10 ** (-config.nonstationary_snr_db / 10))

    generated = {"N1": awgn, "N2": tone, "N3": burst, "N4": nonstationary}
    for name, reference in shared_noise.items():
        if name not in generated:
            raise ValueError(f"unknown shared noise component: {name}")
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        if reference.size != n:
            raise ValueError("shared noise length mismatch")
        generated[name] = reference.copy()
    return generated


def assemble_capture(clean: np.ndarray, components: dict[str, np.ndarray],
                     names: tuple[str, ...], observation_samples: int) -> np.ndarray:
    """Prefix a received noise-only observation to one noisy communication frame."""
    signal_part = np.asarray(clean, dtype=np.float64).reshape(-1)
    if not signal_part.size or observation_samples < 1:
        raise ValueError("clean frame and a positive observation interval are required")
    total = combine_noise(np.zeros(observation_samples + signal_part.size), components, names)
    if total.size != observation_samples + signal_part.size:
        raise ValueError("noise arrays must cover the observation interval and frame")
    total[observation_samples:] += signal_part
    return total


def combine_noise(signal_part: np.ndarray, components: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    result = np.asarray(signal_part, dtype=np.float64).reshape(-1).copy()
    for name in names:
        if name not in components:
            raise KeyError(name)
        result += components[name]
    return result


def actual_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    s, r = np.asarray(clean, dtype=np.float64).reshape(-1), np.asarray(noisy, dtype=np.float64).reshape(-1)
    n = min(s.size, r.size)
    noise_power = _power(r[:n] - s[:n])
    return float(10 * np.log10(_power(s[:n]) / max(noise_power, np.finfo(float).tiny)))


def repeated_snr_reference(clean: np.ndarray, frames: list[np.ndarray],
                           delays: list[int]) -> tuple[float, float, np.ndarray]:
    """Return first-frame SNR and true-delay aligned mean SNR for repeat tests."""
    reference = np.asarray(clean, dtype=np.float64).reshape(-1)
    if not reference.size or not frames or len(frames) != len(delays):
        raise ValueError("clean reference, frames, and matching delays are required")
    aligned = []
    for frame, delay in zip(frames, delays):
        values = np.asarray(frame, dtype=np.float64).reshape(-1)
        if delay < 0 or delay + reference.size > values.size:
            raise ValueError("a delayed frame does not contain the full reference interval")
        aligned.append(values[delay:delay + reference.size])
    first_snr = actual_snr_db(reference, aligned[0])
    ideal_mean = np.mean(np.stack(aligned), axis=0)
    return first_snr, actual_snr_db(reference, ideal_mean), ideal_mean
