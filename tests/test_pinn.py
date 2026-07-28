import torch

from pyadm1ode_estimation.estimation.deep_learning.pinn import ADM1PINN, PINNLoss


def test_pinn_forward():
    input_dim = 2
    output_dim = 41
    model = ADM1PINN(input_dim=input_dim, output_dim=output_dim)

    t = torch.randn(10, 1)
    u = torch.randn(10, 1)

    x_hat = model(t, u)
    assert x_hat.shape == (10, 41)


def test_pinn_loss():
    input_dim = 2
    output_dim = 41
    model = ADM1PINN(input_dim=input_dim, output_dim=output_dim)

    # Mock measurement map: just take the first 3 states
    def measurement_map(x):
        return x[:, :3]

    # Mock ODE residual: just return zeros
    def ode_residual(x, u):
        return torch.zeros_like(x)

    loss_fn = PINNLoss(measurement_map, ode_residual)

    t = torch.randn(10, 1, requires_grad=True)
    u = torch.randn(10, 1)
    y_meas = torch.randn(10, 3)

    loss = loss_fn(t, u, y_meas, model)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

    # Check if we can compute gradients
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None
