from cv2x_cp.config import load_config, project_root
from cv2x_cp.standards import normal_cp_durations_s, normal_cp_overhead_ratio


def test_normal_cp_native_mapping():
    durations = normal_cp_durations_s()
    assert len(durations) == 14
    assert abs(durations[0] * 1e6 - 5.2083333333) < 1e-9
    assert abs(durations[1] * 1e6 - 4.6875) < 1e-12
    assert abs(normal_cp_overhead_ratio() - 1 / 15) < 1e-12


def test_default_config_is_pre_registered():
    cfg = load_config(project_root() / "configs" / "default.json")
    assert cfg["radio"]["carrier_hz"] == 5.9e9
    assert cfg["experiment"]["training_seeds"] == [20260715, 20260716, 20260717, 20260718, 20260719]

