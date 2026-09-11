import numpy as np

from cv2x_cp.channel import ChannelSnapshot
from cv2x_cp.phy import cp_overhead_ratio, residual_tail_energy


def test_residual_energy_decreases_with_cp():
    h = np.asarray([1.0, 0.5, 0.25], dtype=complex)
    assert residual_tail_energy(h, 0) > residual_tail_energy(h, 1)
    assert residual_tail_energy(h, 1) > residual_tail_energy(h, 2)


def test_fixed_budget_adaptive_overhead():
    assert cp_overhead_ratio((72,) * 14, 1024) == 1008 / 15360
    assert cp_overhead_ratio((144,) * 14, 1024) == 2016 / 15360

