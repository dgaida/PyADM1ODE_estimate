import numpy as np
from scipy.linalg import cholesky
from scipy.integrate import solve_ivp
from typing import Callable, List, Tuple


class UnscentedKalmanFilter:
    """
    Allgemeine Implementierung des Unscented Kalman Filters (UKF).
    General implementation of the Unscented Kalman Filter (UKF).

    Basierend auf Wan & Van Der Merwe (2000).
    Based on Wan & Van Der Merwe (2000).
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        f: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        """
        Initialisiert den UKF.
        Initialize the UKF.

        Args:
            n_x: Anzahl der Zustände / Number of states
            n_y: Anzahl der Messgrößen / Number of measurements
            f: Prozessmodell-Funktion / Process model function x_k = f(x_{k-1}, u_{k-1}, dt)
            h: Messmodell-Funktion / Measurement model function y_k = h(x_k)
            Q: Prozessrauschen-Kovarianz / Process noise covariance (n_x x n_x)
            R: Messrauschen-Kovarianz / Measurement noise covariance (n_y x n_y)
            alpha: UKF-Parameter (bestimmt die Streuung der Sigma-Punkte) / (determines spread of sigma points)
            beta: UKF-Parameter (berücksichtigt Vorwissen über die Verteilung) / (incorporates prior knowledge of distribution)
            kappa: UKF-Parameter (sekundärer Skalierungsparameter) / (secondary scaling parameter)
        """
        self.n_x = n_x
        self.n_y = n_y
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R

        # Gewichte für UKF berechnen / Calculate weights for UKF
        self.lambd = alpha**2 * (n_x + kappa) - n_x
        self.Wc = np.full(2 * n_x + 1, 1.0 / (2 * (n_x + self.lambd)))
        self.Wm = np.copy(self.Wc)
        self.Wm[0] = self.lambd / (n_x + self.lambd)
        self.Wc[0] = self.lambd / (n_x + self.lambd) + (1 - alpha**2 + beta)

        self.gamma = np.sqrt(n_x + self.lambd)

    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Erzeugt Sigma-Punkte um den Zustand x mit Kovarianz P.
        Generate sigma points around state x with covariance P.
        """
        sigmas = np.zeros((2 * self.n_x + 1, self.n_x))
        sigmas[0] = x

        # Matrix-Quadratwurzel mittels Cholesky-Zerlegung
        # Sicherstellen, dass P positiv definit ist
        try:
            L = cholesky(P, lower=True)
        except np.linalg.LinAlgError:
            # Fallback für numerische Stabilität
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-12)
            P_pos = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = cholesky(P_pos, lower=True)

        LP = self.gamma * L
        for k in range(self.n_x):
            sigmas[k + 1] = x + LP[:, k]
            sigmas[self.n_x + k + 1] = x - LP[:, k]

        return sigmas

    def predict(
        self, x: np.ndarray, P: np.ndarray, u: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Führt den Prädiktionsschritt aus.
        Perform prediction step.

        Returns:
            x_pred: Prädizierter Zustandsmittelwert / Predicted state mean
            P_pred: Prädizierte Zustands-Kovarianz / Predicted state covariance
            sigmas_f: Propagierte Sigma-Punkte / Propagated sigma points
        """
        # Sigma-Punkte generieren
        sigmas = self.generate_sigma_points(x, P)

        # Sigma-Punkte durch das Prozessmodell propagieren
        sigmas_f = np.zeros_like(sigmas)
        for i in range(2 * self.n_x + 1):
            sigmas_f[i] = self.f(sigmas[i], u, dt)

        # Prädizierter Mittelwert
        x_pred = np.sum(self.Wm[:, None] * sigmas_f, axis=0)

        # Prädizierte Kovarianz
        P_pred = np.zeros((self.n_x, self.n_x))
        for i in range(2 * self.n_x + 1):
            diff = sigmas_f[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)
        P_pred += self.Q

        return x_pred, P_pred, sigmas_f

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        sigmas_f: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Führt den Korrekturschritt (Update) mit der Messung y aus.
        Perform update step with measurement y.
        """
        # Sigma-Punkte durch das Messmodell propagieren
        sigmas_h = np.zeros((2 * self.n_x + 1, self.n_y))
        for i in range(2 * self.n_x + 1):
            sigmas_h[i] = self.h(sigmas_f[i])

        # Prädizierter Messmittelwert
        y_pred = np.sum(self.Wm[:, None] * sigmas_h, axis=0)

        # Innovationskovarianz
        S = np.zeros((self.n_y, self.n_y))
        for i in range(2 * self.n_x + 1):
            diff = sigmas_h[i] - y_pred
            S += self.Wc[i] * np.outer(diff, diff)
        S += self.R

        # Kreuzkovarianz
        Pxz = np.zeros((self.n_x, self.n_y))
        for i in range(2 * self.n_x + 1):
            diff_x = sigmas_f[i] - x_pred
            diff_z = sigmas_h[i] - y_pred
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)

        # Kalman-Verstärkung
        K = Pxz @ np.linalg.inv(S)

        # Aktualisierter Mittelwert und Kovarianz
        x_upd = x_pred + K @ (y - y_pred)
        P_upd = P_pred - K @ S @ K.T

        return x_upd, P_upd


class ADM1UKF:
    """
    UKF-Wrapper für das ADM1-Modell.
    UKF wrapper for the ADM1 model.
    """

    def __init__(
        self,
        adm1_model,  # Instanz von pyadm1.core.ADM1
        Q: np.ndarray,
        R: np.ndarray,
        measurement_indices: List[int],
        **ukf_kwargs,
    ):
        """
        Initialisiert den ADM1-UKF.
        Initialize the ADM1-UKF.

        Args:
            adm1_model: Das mechanistische ADM1-Modell / The mechanistic ADM1 model
            Q: Prozessrauschen / Process noise
            R: Messrauschen / Measurement noise
            measurement_indices: Indizes der gemessenen Zustände / Indices of measured states
        """
        self.model = adm1_model
        self.n_x = 37  # ADM1 Zustandsdimension
        self.n_y = len(measurement_indices)
        self.measurement_indices = measurement_indices

        self.ukf = UnscentedKalmanFilter(
            n_x=self.n_x,
            n_y=self.n_y,
            f=self._process_model,
            h=self._measurement_model,
            Q=Q,
            R=R,
            **ukf_kwargs,
        )

    def _process_model(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Wrapper für die ADM1-ODE-Integration mittels Scipy BDF-Solver.
        Wrapper for ADM1 ODE integration using Scipy BDF solver.
        """
        # u ist die Menge der influent-Volumenströme Q
        self.model.create_influent(u.tolist(), 0)

        # Definition der ODE-Funktion für solve_ivp
        def ode_func(t, y):
            return self.model.ADM1_ODE(t, y.tolist())

        # Integration über das Zeitintervall dt mittels BDF (geeignet für steife ODEs)
        sol = solve_ivp(ode_func, (0, dt), x, method="BDF", rtol=1e-6, atol=1e-8)

        if not sol.success:
            # Fallback auf Euler, falls der Solver fehlschlägt (als Notlösung)
            derivatives = self.model.ADM1_ODE(0, x.tolist())
            x_next = x + np.array(derivatives) * dt
        else:
            x_next = sol.y[:, -1]

        # Nicht-Negativität für Konzentrationen sicherstellen
        x_next[:33] = np.maximum(x_next[:33], 1e-12)
        return x_next

    def _measurement_model(self, x: np.ndarray) -> np.ndarray:
        """
        Wählt die gemessenen Zustände aus dem Zustandsvektor aus.
        Selects measured states from the state vector.
        """
        return x[self.measurement_indices]

    def estimate(
        self, x: np.ndarray, P: np.ndarray, u: np.ndarray, y: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Führt einen kompletten UKF-Schritt (Prädiktion und Update) aus.
        Perform one complete UKF step (predict and update).
        """
        x_pred, P_pred, sigmas_f = self.ukf.predict(x, P, u, dt)
        x_upd, P_upd = self.ukf.update(x_pred, P_pred, sigmas_f, y)
        return x_upd, P_upd
