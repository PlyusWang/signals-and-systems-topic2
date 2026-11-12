import numpy as np

from task2.communication_experiment.modem import (
    ModemConfig, bit_error_rate, make_frame, random_payload, receive, receive_at_start,
)
from task2.communication_experiment.noise import (
    NoiseConfig, actual_snr_db, assemble_capture, combine_noise, make_noise_components,
    repeated_snr_reference,
)


def test_zero_noise_fsk_sync_crc_and_payload_decode():
    config = ModemConfig()
    payload = random_payload(44, 32)
    frame = make_frame(payload, config)
    # A leading offset exercises synchronization rather than an assumed frame boundary.
    received = np.r_[np.zeros(23), frame["waveform"], np.zeros(100)]
    result = receive(received, len(payload), config)
    assert result["start"] == 23
    assert result["crc_ok"]
    assert result["payload"] == payload
    assert bit_error_rate(frame["payload_bits"], np.unpackbits(np.frombuffer(result["payload"], dtype=np.uint8))) == 0


def test_known_start_decoder_matches_synchronized_decoder_without_perfect_payload_access():
    config = ModemConfig()
    payload = random_payload(178, 48)
    frame = make_frame(payload, config)
    start = 73
    received = np.r_[np.zeros(start), frame["waveform"], np.zeros(120)]
    synchronized = receive(received, len(payload), config)
    known_start = receive_at_start(received, len(payload), start, config)
    assert known_start["start"] == start
    assert known_start["payload"] == synchronized["payload"] == payload
    assert known_start["crc_ok"] and synchronized["crc_ok"]


def test_noise_components_reproducible_and_combinations_add_exactly():
    clean = np.ones(20000) * 0.2
    first = make_noise_components(clean, 16000, 11)
    second = make_noise_components(clean, 16000, 11)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    combined = combine_noise(clean, first, ("N1", "N2", "N3"))
    assert np.allclose(combined - clean, first["N1"] + first["N2"] + first["N3"])
    assert np.isfinite(actual_snr_db(clean, combined))


def test_all_noise_components_are_retained_and_nonzero():
    clean = np.sin(2 * np.pi * np.arange(16000) / 16)
    components = make_noise_components(clean, 16000, 2)
    assert set(components) == {"N1", "N2", "N3", "N4"}
    assert all(np.mean(x * x) > 0 for x in components.values())


def test_narrowband_phase_varies_by_seed_and_shared_interference_is_exact():
    fs = 16000
    clean = np.sin(2 * np.pi * np.arange(32000) / 16)
    first = make_noise_components(clean, fs, 10, NoiseConfig(tone_phase_rad=None))
    second = make_noise_components(clean, fs, 11, NoiseConfig(tone_phase_rad=None))
    assert not np.allclose(first["N2"], second["N2"])

    shared = np.sin(2 * np.pi * 1777 * np.arange(clean.size) / fs + 0.7)
    a = make_noise_components(clean, fs, 101, shared_noise={"N2": shared})
    b = make_noise_components(clean, fs, 202, shared_noise={"N2": shared})
    assert np.array_equal(a["N2"], b["N2"])
    assert not np.array_equal(a["N1"], b["N1"])


def test_noise_only_observation_prefix_is_present_and_wiener_uses_received_samples():
    fs = 8000
    clean = np.sin(2 * np.pi * 500 * np.arange(4096) / fs)
    observation = 1024
    noises = make_noise_components(clean, fs, 5, output_length=observation + clean.size)
    capture = assemble_capture(clean, noises, ("N1", "N2"), observation)
    assert capture.size == observation + clean.size
    assert np.array_equal(capture[:observation], (noises["N1"] + noises["N2"])[:observation])
    assert np.allclose(capture[observation:], clean + noises["N1"][observation:] + noises["N2"][observation:])
    assert np.all(np.isfinite(actual_snr_db(clean, capture[observation:])))


def test_repeat_snr_reports_single_frame_input_separately_from_true_aligned_mean():
    clean = np.ones(1024)
    first = np.r_[np.zeros(7), clean + np.ones_like(clean)]
    second = np.r_[np.zeros(19), clean - np.ones_like(clean)]
    single_snr, ideal_snr, mean = repeated_snr_reference(clean, [first, second], [7, 19])
    assert np.isclose(single_snr, 0.0)
    assert ideal_snr > 250.0
    assert np.allclose(mean, clean)
