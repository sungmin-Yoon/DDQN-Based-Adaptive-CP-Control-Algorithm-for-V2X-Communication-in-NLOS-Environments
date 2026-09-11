from __future__ import annotations

import math

C_LIGHT = 299_792_458.0
THERMAL_NOISE_DBM_HZ = -174.0
LTE_REFERENCE_TS_S = 1.0 / (2048 * 15_000.0)


def normal_cp_native_samples() -> tuple[int, ...]:
    """One 1 ms normal-CP subframe at 15.36 MHz (two 0.5 ms slots)."""
    return (80, 72, 72, 72, 72, 72, 72) * 2


def normal_cp_durations_s() -> tuple[float, ...]:
    return tuple(v / 15.36e6 for v in normal_cp_native_samples())


def normal_cp_overhead_ratio(fft_size: int = 1024) -> float:
    cp = sum(normal_cp_native_samples())
    return cp / (14 * fft_size + cp)


def urban_v2v_pathloss_db(distance_m: float, carrier_ghz: float, state: str) -> float:
    """TR 37.885 urban V2V mapping; random shadowing is added separately."""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    state = state.upper()
    if state == "LOS":
        return 38.77 + 16.7 * math.log10(distance_m) + 18.2 * math.log10(carrier_ghz)
    if state == "NLOS":
        return 36.85 + 30.0 * math.log10(distance_m) + 18.9 * math.log10(carrier_ghz)
    raise ValueError("Only LOS and building-blocked NLOS are in scope")


def urban_log10_ds_parameters(carrier_ghz: float, state: str) -> tuple[float, float]:
    """TR 37.885 urban lgDS=log10(DS/1 s) parameter mapping."""
    x = math.log10(1.0 + carrier_ghz)
    if state.upper() == "LOS":
        return -0.2 * x - 7.5, 0.10
    if state.upper() == "NLOS":
        return -0.3 * x - 7.0, 0.28
    raise ValueError("Only LOS and building-blocked NLOS are in scope")


def noise_power_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    return THERMAL_NOISE_DBM_HZ + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def maximum_doppler_hz(carrier_hz: float, speed_kmh: float) -> float:
    return carrier_hz * (speed_kmh / 3.6) / C_LIGHT

