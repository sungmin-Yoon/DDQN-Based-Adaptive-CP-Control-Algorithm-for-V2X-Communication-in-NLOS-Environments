from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .channel import ChannelSnapshot
from .standards import normal_cp_native_samples


@dataclass(frozen=True)
class LinkMetrics:
    expected_prr: float
    post_sinr_db: float
    residual_isi: float
    cp_mean_samples: float
    cp_mean_s: float
    cp_overhead: float
    raw_spectral_efficiency: float
    goodput_bps_hz: float
    useful_rf_ee_bits_per_joule: float


def cp_pattern_samples(action_samples: int | None) -> tuple[int, ...]:
    return normal_cp_native_samples() if action_samples is None else (int(action_samples),) * 14


def cp_overhead_ratio(pattern: tuple[int, ...], fft_size: int) -> float:
    # O_cp = N_cp,total / (15*N_fft). The proposed extension preserves the
    # 1 ms LTE-style timing budget; extra/removed CP samples displace/restore
    # payload-bearing samples in that fixed budget.
    return sum(pattern) / (15 * fft_size)


def residual_tail_energy(h: np.ndarray, cp_samples: int) -> float:
    power = np.abs(h) ** 2
    total = float(power.sum())
    return 0.0 if total <= 0 else float(power[cp_samples + 1 :].sum() / total)


def abstract_link(snapshot: ChannelSnapshot, action_samples: int | None, cfg: dict) -> LinkMetrics:
    radio = cfg["radio"]
    pattern = cp_pattern_samples(action_samples)
    # Average symbol result retains the 80/72 fixed-normal-CP distinction.
    tails = np.asarray([residual_tail_energy(snapshot.impulse_response, n) for n in pattern])
    snr = 10 ** (snapshot.snr_db / 10)
    input_sinr = snr / (1.0 + snr * float(tails.mean()))
    nfft = int(radio["fft_size"])
    occupied = int(radio["occupied_subcarriers"])
    bins = occupied_subcarrier_bins(nfft, occupied)
    response = np.fft.fft(snapshot.impulse_response, nfft)[bins]
    mmse = float(np.mean(1.0 / (1.0 + input_sinr * np.abs(response) ** 2)))
    post = max(1.0 / max(mmse, 1e-15) - 1.0, 0.0)
    # Uncoded QPSK BER is used as the conservative link curve; code rate affects
    # payload efficiency equally for all policies. Waveform calibration is gated.
    ber = 0.5 * math.erfc(math.sqrt(max(post, 0.0)))
    information_bits = int(radio["packet_bits"])
    expected_prr = float(np.clip(math.exp(information_bits * math.log(max(1.0 - ber, 1e-15))), 0, 1))
    overhead = cp_overhead_ratio(pattern, nfft)
    modulation_bits = 2.0
    code_rate = float(radio["code_rate"])
    # Effective spectral efficiency / goodput:
    # G = PRR * log2(M) * R_c * (1 - O_cp), in bit/s/Hz.
    raw_se = modulation_bits * code_rate * (1.0 - overhead)
    goodput = raw_se * expected_prr
    tx_watts = 10 ** ((float(radio["tx_power_dbm"]) - 30.0) / 10.0)
    packet_airtime = information_bits / max(raw_se * float(radio["bandwidth_hz"]), 1e-15)
    rf_energy = tx_watts * packet_airtime
    # RF-only useful energy efficiency: successfully delivered information
    # bits divided by RF transmit energy. Circuit, sensing, processing, and
    # CP-control signaling energy are intentionally outside this metric.
    useful_ee = information_bits * expected_prr / max(rf_energy, 1e-15)
    return LinkMetrics(
        expected_prr, 10 * math.log10(max(post, 1e-20)), float(tails.mean()),
        float(np.mean(pattern)), float(np.mean(pattern)) * snapshot.sample_period_s,
        overhead, raw_se, goodput, useful_ee,
    )


def waveform_link(snapshot: ChannelSnapshot, action_samples: int, cfg: dict, seed: int) -> LinkMetrics:
    """Explicit SC-FDMA validation block; intentionally separate from training."""
    radio, phy = cfg["radio"], cfg["phy"]
    nfft, occupied = int(radio["fft_size"]), int(radio["occupied_subcarriers"])
    if occupied >= nfft or occupied % 2:
        raise ValueError("occupied_subcarriers must be even and smaller than NFFT")
    rng = np.random.default_rng(seed)
    h = snapshot.impulse_response
    snr = 10 ** (snapshot.snr_db / 10)
    noise_var = 1.0 / max(snr, 1e-15)
    bins = occupied_subcarrier_bins(nfft, occupied)
    channel_f = np.fft.fft(h, nfft)
    equalizer = np.conj(channel_f[bins]) / (np.abs(channel_f[bins]) ** 2 + noise_var)
    error_energy = symbol_energy = 0.0
    total_symbols = 0
    previous = np.zeros(nfft + action_samples, dtype=complex)
    for _ in range(int(phy["waveform_blocks"])):
        bits = rng.integers(0, 2, 2 * occupied)
        qpsk = ((1 - 2 * bits[0::2]) + 1j * (1 - 2 * bits[1::2])) / math.sqrt(2)
        spread = np.fft.fft(qpsk) / math.sqrt(occupied)
        mapped = np.zeros(nfft, dtype=complex)
        mapped[bins] = spread
        useful = np.fft.ifft(mapped) * math.sqrt(nfft)
        tx = np.r_[useful[-action_samples:], useful]
        # Previous block tail is retained, creating physical inter-block interference.
        stream = np.r_[previous, tx]
        convolved = np.convolve(stream, h)
        start = len(previous) + action_samples
        rx = convolved[start : start + nfft]
        noise = math.sqrt(noise_var / 2) * (rng.normal(size=nfft) + 1j * rng.normal(size=nfft))
        recovered = np.fft.fft(rx + noise) / math.sqrt(nfft)
        despread = np.fft.ifft(recovered[bins] * equalizer) * math.sqrt(occupied)
        # Hard bit counts are too quantized for packet-level PRR checks at the
        # default validation length. EVM gives a stable BER estimate while still
        # exercising the explicit SC-FDMA waveform path.
        gain = np.vdot(qpsk, despread) / max(float(np.vdot(qpsk, qpsk).real), 1e-15)
        error = despread - gain * qpsk
        error_energy += float(np.vdot(error, error).real)
        symbol_energy += float(abs(gain) ** 2 * np.vdot(qpsk, qpsk).real)
        total_symbols += len(qpsk)
        previous = tx
    evm2 = error_energy / max(symbol_energy, 1e-15)
    effective_sinr = 1.0 / max(evm2, 1e-15)
    ber = 0.5 * math.erfc(math.sqrt(max(effective_sinr, 0.0)))
    prr = float((1.0 - ber) ** int(radio["packet_bits"]))
    base = abstract_link(snapshot, action_samples, cfg)
    return LinkMetrics(
        prr, base.post_sinr_db, base.residual_isi, base.cp_mean_samples,
        base.cp_mean_s, base.cp_overhead, base.raw_spectral_efficiency,
        base.raw_spectral_efficiency * prr,
        base.useful_rf_ee_bits_per_joule * prr / max(base.expected_prr, 1e-15),
    )


def all_action_metrics(snapshot: ChannelSnapshot, cfg: dict) -> list[LinkMetrics]:
    from .environment import cp_sweep_samples

    return [abstract_link(snapshot, int(a), cfg) for a in cp_sweep_samples(cfg)]


def occupied_subcarrier_bins(nfft: int, occupied: int) -> np.ndarray:
    if occupied >= nfft or occupied % 2:
        raise ValueError("occupied_subcarriers must be even and smaller than NFFT")
    return np.r_[np.arange(nfft - occupied // 2, nfft), np.arange(1, occupied // 2 + 1)]
