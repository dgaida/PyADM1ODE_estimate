import torch
import torch.nn as nn
from typing import List, Optional, Callable


class ADM1PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) für den ADM1 (AP 4.3).
    Physics-Informed Neural Network (PINN) for the ADM1 (AP 4.3).

    Das Netzwerk schätzt den Zustandsvektor x(t) basierend auf der Zeit t
    und dem zeitabhängigen Input u(t) (Substratzufuhr).
    The network estimates the state vector x(t) based on time t
    and time-dependent input u(t) (substrate feed).
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 37,
        hidden_layers: List[int] = [64, 64, 64],
    ):
        """
        Initialisiert das PINN.
        Initialize the PINN.

        Args:
            input_dim: Dimension der Eingabe (meist Zeit t + u_dim) / Input dimension (usually time t + u_dim)
            output_dim: Dimension des Zustandsvektors (ADM1: 37) / State vector dimension
            hidden_layers: Liste mit der Anzahl der Neuronen pro Layer / List of neurons per layer
        """
        super(ADM1PINN, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(
                nn.Tanh()
            )  # Tanh is preferred in PINNs for smooth derivatives
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, t: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Vorwärtspass des Netzwerks.
        Forward pass of the network.

        Args:
            t: Zeitpunkte / Time points (shape: [batch, 1])
            u: Zusätzliche Eingangsgrößen (Substratzufuhr) / Additional inputs (substrate feed) (shape: [batch, u_dim])
        """
        if u is not None:
            x = torch.cat([t, u], dim=-1)
        else:
            x = t
        return self.net(x)


class PINNLoss(nn.Module):
    """
    Verlustfunktion für das PINN, kombiniert Daten- und Physik-Loss.
    Loss function for the PINN, combining data and physics loss.
    """

    def __init__(
        self,
        measurement_map: Callable,
        ode_residual_func: Callable,
        lambda_phys: float = 0.1,
    ):
        """
        Args:
            measurement_map: Funktion, die den Zustandsvektor x auf die Messgrößen y abbildet (pH, Biogas, etc.)
                             Function mapping state vector x to measurements y.
            ode_residual_func: Funktion, die das Residuum der ADM1-ODE berechnet.
                               Function calculating the ADM1 ODE residual.
            lambda_phys: Gewichtung des Physik-Loss / Weighting of the physics loss
        """
        super(PINNLoss, self).__init__()
        self.measurement_map = measurement_map
        self.ode_residual_func = ode_residual_func
        self.lambda_phys = lambda_phys
        self.mse = nn.MSELoss()

    def forward(
        self, t: torch.Tensor, u: torch.Tensor, y_meas: torch.Tensor, model: ADM1PINN
    ) -> torch.Tensor:
        """
        Berechnet den kombinierten Loss. / Calculates the combined loss.

        L = L_data + lambda_phys * L_phys

        Args:
            t: Zeit / Time (requires_grad=True)
            u: Input (Substrat) / Input (substrate)
            y_meas: Messwerte (Biogas, CH4, pH) / Measurements
            model: Das ADM1PINN Modell / The ADM1PINN model
        """
        # Sicherstellen, dass t Gradienten trackt für die Ableitung nach der Zeit
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)

        x_hat = model(t, u)

        # 1. Data Loss
        # y_hat = [Biogasproduktion, Methankonzentration, pH]
        y_hat = self.measurement_map(x_hat)
        l_data = self.mse(y_hat, y_meas)

        # 2. Physics Loss (ODE Residuum)
        # d x_hat / dt
        # Wir berechnen die Ableitung jeder Zustandsvariable nach der Zeit t
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

        # Residuum: dx/dt - f(x, u)
        # ode_residual_func sollte f(x, u) in PyTorch implementieren
        f_x_u = self.ode_residual_func(x_hat, u)
        l_phys = self.mse(dxdt_hat, f_x_u)

        return l_data + self.lambda_phys * l_phys
