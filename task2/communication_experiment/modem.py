from __future__ import annotations

from dataclasses import dataclass
import zlib

import numpy as np

from shared.signal_core import estimate_delay


@dataclass(frozen=True)
class ModemConfig:
    sample_rate: int = 16000
    bit_rate: int = 1000
    f0_hz: float = 1500.0
    f1_hz: float = 2500.0
    amplitude: float = 0.55
    preamble: tuple[int, ...] = (1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
                                  0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0,
                                  1, 0, 0, 1, 1, 1, 0, 0)

    @property
    def samples_per_bit(self) -> int:
        value = self.sample_rate / self.bit_rate
        if not value.is_integer():
            raise ValueError("sample_rate must be an integer multiple of bit_rate")
        return int(value)

    def validate(self) -> None:
        if self.sample_rate <= 0 or self.bit_rate <= 0:
            raise ValueError("sample and bit rates must be positive")
        if max(self.f0_hz, self.f1_hz) >= self.sample_rate / 2 or self.f0_hz == self.f1_hz:
            raise ValueError("FSK carriers must be distinct and below Nyquist")
        if not 0 < self.amplitude <= 1 or not self.preamble:
            raise ValueError("amplitude and preamble are invalid")
        self.samples_per_bit


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def bits_to_bytes(bits: np.ndarray) -> bytes:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if values.size % 8:
        raise ValueError("bit count must be divisible by eight")
    return np.packbits(values, bitorder="big").tobytes()


def append_crc(payload: bytes) -> bytes:
    return payload + (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big")


def check_crc(frame_payload_crc: bytes) -> tuple[bytes, bool]:
    if len(frame_payload_crc) < 4:
        return b"", False
    payload, received = frame_payload_crc[:-4], int.from_bytes(frame_payload_crc[-4:], "big")
    return payload, (zlib.crc32(payload) & 0xFFFFFFFF) == received


def modulate(bits: np.ndarray, config: ModemConfig = ModemConfig(), *,
             initial_phase: float = 0.0) -> np.ndarray:
    config.validate()
    bit_values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any(bit_values > 1) or not bit_values.size:
        raise ValueError("bits must be a nonempty binary sequence")
    spb = config.samples_per_bit
    frequencies = np.repeat(np.where(bit_values == 0, config.f0_hz, config.f1_hz), spb)
    phase = initial_phase + 2.0 * np.pi * np.cumsum(frequencies) / config.sample_rate
    return config.amplitude * np.cos(phase)


def make_frame(payload: bytes, config: ModemConfig = ModemConfig()) -> dict[str, object]:
    config.validate()
    encoded = append_crc(payload)
    payload_crc_bits = bytes_to_bits(encoded)
    preamble = np.asarray(config.preamble, dtype=np.uint8)
    frame_bits = np.concatenate((preamble, payload_crc_bits))
    waveform = modulate(frame_bits, config)
    preamble_waveform = modulate(preamble, config)
    return {
        "bits": frame_bits,
        "payload_bits": bytes_to_bits(payload),
        "payload": payload,
        "crc_bytes": encoded,
        "waveform": waveform,
        "preamble_waveform": preamble_waveform,
        "preamble_samples": preamble.size * config.samples_per_bit,
        "sample_rate": config.sample_rate,
    }


def demodulate_aligned(samples: np.ndarray, bit_count: int, config: ModemConfig = ModemConfig()) -> np.ndarray:
    """Noncoherent two-tone energy receiver, requiring a frame-aligned input."""
    config.validate()
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    spb = config.samples_per_bit
    needed = bit_count * spb
    if bit_count < 1 or x.size < needed:
        raise ValueError("insufficient aligned samples")
    x = x[:needed].reshape(bit_count, spb)
    time = np.arange(spb) / config.sample_rate
    ref0 = np.exp(-2j * np.pi * config.f0_hz * time)
    ref1 = np.exp(-2j * np.pi * config.f1_hz * time)
    energy0 = np.abs(x @ ref0) ** 2
    energy1 = np.abs(x @ ref1) ** 2
    return (energy1 > energy0).astype(np.uint8)


def receive_at_start(samples: np.ndarray, payload_nbytes: int, start: int,
                     config: ModemConfig = ModemConfig()) -> dict[str, object]:
    """Decode from a supplied frame start; useful for diagnostics with known timing."""
    frame = make_frame(bytes(payload_nbytes), config)
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if start < 0:
        raise ValueError("frame start cannot be negative")
    total_bits = len(config.preamble) + (payload_nbytes + 4) * 8
    length = total_bits * config.samples_per_bit
    available = x[start:min(start + length, x.size)]
    if available.size < length:
        available = np.pad(available, (0, length - available.size))
    decisions = demodulate_aligned(available, total_bits, config)
    preamble_bits = np.asarray(config.preamble, dtype=np.uint8)
    payload_crc = bits_to_bytes(decisions[len(preamble_bits):])
    payload, crc_ok = check_crc(payload_crc)
    return {"start": start, "sync_peak": float("nan"), "correlation_indices": np.empty(0, dtype=int),
            "correlation_scores": np.empty(0), "bits": decisions,
            "preamble_errors": int(np.count_nonzero(decisions[:len(preamble_bits)] != preamble_bits)),
            "payload": payload, "crc_ok": crc_ok}


def receive(samples: np.ndarray, payload_nbytes: int, config: ModemConfig = ModemConfig()) -> dict[str, object]:
    """Synchronize on public preamble, then decode the unknown payload and CRC."""
    frame = make_frame(bytes(payload_nbytes), config)
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    start, peak, indices, scores = estimate_delay(x, frame["preamble_waveform"])
    result = receive_at_start(x, payload_nbytes, start, config)
    result.update(sync_peak=peak, correlation_indices=indices, correlation_scores=scores)
    return result


def bit_error_rate(reference: np.ndarray, estimate: np.ndarray) -> float:
    a, b = np.asarray(reference, dtype=np.uint8).reshape(-1), np.asarray(estimate, dtype=np.uint8).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        raise ValueError("BER inputs must be nonempty")
    return float(np.mean(a[:n] != b[:n]))


def random_payload(seed: int, nbytes: int = 64) -> bytes:
    if nbytes <= 0:
        raise ValueError("nbytes must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=nbytes, dtype=np.uint8).tobytes()
