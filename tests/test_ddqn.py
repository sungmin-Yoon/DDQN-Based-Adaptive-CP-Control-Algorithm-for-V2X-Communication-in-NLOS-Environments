import torch

from cv2x_cp.ddqn import QNetwork, optimize


def test_double_dqn_optimization_is_finite():
    online = QNetwork(8, 3, [16])
    target = QNetwork(8, 3, [16])
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    batch = (torch.zeros(4, 8), torch.zeros(4, dtype=torch.long), torch.ones(4),
             torch.zeros(4, 8), torch.zeros(4))
    assert torch.isfinite(torch.tensor(optimize(online, target, optimizer, batch, .99, 5.0)))

