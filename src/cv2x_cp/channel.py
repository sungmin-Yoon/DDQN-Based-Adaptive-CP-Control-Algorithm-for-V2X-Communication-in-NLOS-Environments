from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .standards import (
    maximum_doppler_hz,
    noise_power_dbm,
    urban_log10_ds_parameters,
    urban_v2v_pathloss_db,
)


@dataclass(frozen=True)
class ChannelSnapshot:
    step: int
    time_s: float
    state: str
    true_ds_s: float
    estimated_ds_s: float
    estimated_max_delay_s: float
    snr_db: float
    rsrp_dbm: float
    doppler_hz: float
    sample_period_s: float
    impulse_response: np.ndarray


def rms_delay_spread(h: np.ndarray, sample_period_s: float) -> float:
    power = np.abs(h) ** 2
    total = float(power.sum())
    if total <= 0:
        return 0.0
    delays = np.arange(len(h), dtype=float) * sample_period_s
    weights = power / total
    mean = float(np.sum(weights * delays))
    return math.sqrt(max(float(np.sum(weights * (delays - mean) ** 2)), 0.0))


class UrbanTraceGenerator:
    """Temporally correlated LOS/urban-canyon/LOS C-V2X traces.

    LOS and building-blocked NLOS pathloss/delay-spread baselines follow the
    local TR 37.885 mapping in :mod:`standards`.  UrbanCanyon is an explicit
    stress state layered on top of the NLOS baseline so the experiment
    can separate CP-limited long-delay outage from pathloss-limited outage.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def generate(self, seed: int) -> list[ChannelSnapshot]:
        rng = np.random.default_rng(seed)
        exp, radio, channel = self.cfg["experiment"], self.cfg["radio"], self.cfg["channel"]
        dt = float(exp["control_interval_ms"]) * 1e-3
        sample_period = 1.0 / float(radio["sample_rate_hz"])
        fc_ghz = float(radio["carrier_hz"]) / 1e9
        noise_dbm = noise_power_dbm(float(radio["bandwidth_hz"]), float(radio["noise_figure_db"]))
        fd = maximum_doppler_hz(float(radio["carrier_hz"]), float(radio["speed_kmh"]))
        log_ds = shadow = interference = estimate_error = None
        previous_state = None
        out: list[ChannelSnapshot] = []
        for step in range(int(exp["episode_steps"])):
            time_s = step * dt
            state = self._scheduled_state(time_s)
            standard_state = self._standard_state(state)
            mean_ds, std_ds = urban_log10_ds_parameters(fc_ghz, standard_state)
            if previous_state != state or log_ds is None:
                log_ds = mean_ds + rng.normal(0.0, std_ds)
                shadow = rng.normal(0.0, self._shadow_std(state))
            else:
                log_ds = self._ar1(log_ds, mean_ds, std_ds, float(channel["ds_correlation"]), rng)
                shadow = self._ar1(shadow, 0.0, self._shadow_std(state), float(channel["shadow_correlation"]), rng)
            interference = self._ar1(
                interference, float(channel["interference_mean_dbm"]),
                float(channel["interference_std_db"]), float(channel["interference_correlation"]), rng,
            )
            estimate_error = self._ar1(
                estimate_error, 0.0, float(channel["ds_estimation_log_std"]),
                float(channel["estimate_correlation"]), rng,
            )
            target_ds = float(np.clip(10.0 ** log_ds * self._ds_multiplier(state), sample_period / 2, 8e-6))
            h = self._impulse_response(target_ds, sample_period, int(channel["max_channel_samples"]), rng)
            if state == "UrbanCanyon":
                h = self._add_late_stress_tap(h, state, int(channel["max_channel_samples"]), rng)
            true_ds = rms_delay_spread(h, sample_period)
            estimated_ds = true_ds * math.exp(float(estimate_error))
            significant = np.flatnonzero(np.abs(h) ** 2 >= 1e-3 * float(np.max(np.abs(h) ** 2)))
            max_delay = float(significant[-1] * sample_period) if len(significant) else 0.0
            pathloss = urban_v2v_pathloss_db(float(radio["distance_m"]), fc_ghz, standard_state)
            received_dbm = float(radio["tx_power_dbm"]) - pathloss - float(shadow)
            received_dbm -= self._extra_loss_db(state)
            combined_noise_mw = 10 ** (noise_dbm / 10) + 10 ** (float(interference) / 10)
            snr_db = received_dbm - 10 * math.log10(combined_noise_mw)
            out.append(ChannelSnapshot(
                step, time_s, state, true_ds, estimated_ds,
                max_delay * math.exp(0.5 * float(estimate_error)), snr_db,
                received_dbm, fd, sample_period, h,
            ))
            previous_state = state
        return out

    def _shadow_std(self, state: str) -> float:
        key = "los_shadow_std_db" if state == "LOS" else "nlos_shadow_std_db"
        return float(self.cfg["channel"][key])

    def _scheduled_state(self, time_s: float) -> str:
        exp = self.cfg["experiment"]
        if float(exp["urban_canyon_start_s"]) <= time_s < float(exp["urban_canyon_end_s"]):
            return "UrbanCanyon"
        return "LOS"

    @staticmethod
    def _standard_state(state: str) -> str:
        return "LOS" if state == "LOS" else "NLOS"

    def _ds_multiplier(self, state: str) -> float:
        c = self.cfg["channel"]
        if state == "UrbanCanyon":
            return float(c["urban_canyon_ds_multiplier"])
        return 1.0

    def _extra_loss_db(self, state: str) -> float:
        c = self.cfg["channel"]
        if state == "UrbanCanyon":
            return float(c["urban_canyon_extra_loss_db"])
        return 0.0

    def _add_late_stress_tap(self, h: np.ndarray, state: str, max_samples: int, rng) -> np.ndarray:
        prefix = "urban_canyon"
        delay = int(self.cfg["channel"][f"{prefix}_late_tap_delay_samples"])
        fraction = float(self.cfg["channel"][f"{prefix}_late_tap_fraction"])
        length = min(max(max(len(h), delay + 1), 2), max_samples)
        out = np.zeros(length, dtype=complex)
        out[: len(h)] = h * math.sqrt(max(1.0 - fraction, 0.0))
        if delay < length:
            phase = rng.uniform(0.0, 2 * math.pi)
            out[delay] += math.sqrt(max(fraction, 0.0)) * complex(math.cos(phase), math.sin(phase))
        norm = np.linalg.norm(out)
        return out / norm if norm > 0 else out

    @staticmethod
    def _ar1(value, mean: float, std: float, rho: float, rng) -> float:
        if value is None:
            return mean + rng.normal(0.0, std)
        return mean + rho * (float(value) - mean) + math.sqrt(1 - rho**2) * rng.normal(0.0, std)

    @staticmethod
    def _impulse_response(ds_s: float, ts: float, max_samples: int, rng) -> np.ndarray:
        # An exponential PDP has RMS delay approximately equal to its time constant.
        length = int(np.clip(math.ceil(8.0 * ds_s / ts) + 1, 2, max_samples))
        delays = np.arange(length) * ts
        powers = np.exp(-delays / max(ds_s, ts / 2))
        powers /= powers.sum()
        phases = rng.uniform(0.0, 2 * math.pi, length)
        h = np.sqrt(powers) * np.exp(1j * phases)
        return h / np.linalg.norm(h)
