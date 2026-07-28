from collections.abc import Callable, Sequence

import torch
from torch import nn


class ADM1PINN(nn.Module):
    """Physics-Informed Neural Network (PINN) for the ADM1 (AP 4.3).

    Estimates the state vector x(t) from time t and the time-dependent input
    u(t) (substrate feed).

    The output covers the full extended ADM1da state (41 slots: 12 soluble, 10
    particulate PS/PF fractions, 7 biomass, 8 charge/ion states, 4 gas-phase
    slots). The 4 gas-phase states (p_gas_*, p_total) are not optional: they
    couple back into diff_S_h2/ch4/co2 via the Rho_T_* transfer rates, so the
    ADM_ODE physics residual cannot be closed without them.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 41,
        hidden_layers: Sequence[int] = (64, 64, 64),
        dropout: float = 0.0,
    ):
        """Initialize the PINN.

        Args:
            input_dim: Input dimension (time t + u_dim), where u_dim is the
                number of substrate feed channels, i.e. input_dim = 1 + u_dim.
            output_dim: Dimension of the extended ADM1da state vector (41).
            hidden_layers: Number of neurons per hidden layer.
            dropout: Dropout probability after each hidden layer; ``> 0`` enables
                MC-Dropout (evaluate the net repeatedly in train() mode for an
                uncertainty estimate).
        """
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(
                nn.Tanh()
            )  # Tanh is preferred in PINNs for smooth derivatives
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            t: Time points (shape: [batch, 1]).
            u: Additional inputs / substrate feed (shape: [batch, u_dim]).
        """
        if u is not None:
            x = torch.cat([t, u], dim=-1)
        else:
            x = t
        return self.net(x)


class PINNLoss(nn.Module):
    """Loss function for the PINN, combining data and physics loss."""

    def __init__(
        self,
        measurement_map: Callable,
        ode_residual_func: Callable,
        lambda_phys: float = 0.1,
    ):
        """
        Args:
            measurement_map: Function mapping the state vector x to the
                measurements y (pH, biogas, etc.).
            ode_residual_func: Function f(x, u) returning the ADM1da derivatives
                dx/dt (differentiable, 41-dim, incl. gas-phase coupling). Must be
                autograd-compatible so the physics gradient flows back into the
                net weights.
            lambda_phys: Weighting of the physics loss.
        """
        super().__init__()
        self.measurement_map = measurement_map
        self.ode_residual_func = ode_residual_func
        self.lambda_phys = lambda_phys
        self.mse = nn.MSELoss()

    def forward(
        self, t: torch.Tensor, u: torch.Tensor, y_meas: torch.Tensor, model: ADM1PINN
    ) -> torch.Tensor:
        """Compute the combined loss  L = L_data + lambda_phys * L_phys.

        Args:
            t: Time (requires_grad=True).
            u: Input (substrate).
            y_meas: Measurements (biogas, CH4, pH).
            model: The ADM1PINN model.
        """
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)

        x_hat = model(t, u)

        # 1. Data loss
        y_hat = self.measurement_map(x_hat)
        l_data = self.mse(y_hat, y_meas)

        # 2. Physics loss: derivative of each state variable w.r.t. time t
        dxdt_hat = []
        for i in range(x_hat.shape[1]):
            grad = torch.autograd.grad(
                x_hat[:, i],
                t,
                grad_outputs=torch.ones_like(x_hat[:, i]),
                create_graph=True,
                retain_graph=True,
            )[0]
            dxdt_hat.append(grad)
        dxdt_hat = torch.cat(dxdt_hat, dim=-1)

        # residual: dx/dt - f(x, u)
        f_x_u = self.ode_residual_func(x_hat, u)
        l_phys = self.mse(dxdt_hat, f_x_u)

        return l_data + self.lambda_phys * l_phys
