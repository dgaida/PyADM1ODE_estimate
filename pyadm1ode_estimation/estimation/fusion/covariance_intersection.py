import numpy as np
from typing import Tuple, Optional


def covariance_intersection(
    mean1: np.ndarray,
    cov1: np.ndarray,
    mean2: np.ndarray,
    cov2: np.ndarray,
    omega: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fusioniert zwei Zustandsschätzungen mittels Covariance Intersection (AP 4.4).
    Fuses two state estimations using Covariance Intersection (AP 4.4).

    Berechnet eine konservative Schätzung aus Mittelwert und Varianz zweier
    Schätzungen, deren Korrelation unbekannt ist.
    Calculates a conservative estimate from the mean and variance of two
    estimates whose correlation is unknown.

    Args:
        mean1, cov1: Erste Schätzung (z.B. UKF) / First estimate (e.g. UKF)
        mean2, cov2: Zweite Schätzung (z.B. DL-Ensemble) / Second estimate (e.g. DL-Ensemble)
        omega: Gewichtungsfaktor [0, 1]. Falls None, wird omega optimiert (Minimierung der Determinante).
               Weighting factor [0, 1]. If None, omega is optimized (minimizing the determinant).

    Returns:
        mean_fused: Fusionierter Mittelwert / Fused mean
        cov_fused: Fusionierte Kovarianz / Fused covariance
    """
    # Inverse der Kovarianzen berechnen
    inv_cov1 = np.linalg.inv(cov1)
    inv_cov2 = np.linalg.inv(cov2)

    if omega is None:
        # Einfache Suche nach optimalem omega (Minimierung der Determinante von cov_fused)
        best_det = float("inf")
        best_omega = 0.5
        for o in np.linspace(0, 1, 11):
            temp_inv_cov = o * inv_cov1 + (1 - o) * inv_cov2
            # Determinante der invertierten Kovarianz maximieren = Determinante der Kovarianz minimieren
            det_inv = np.linalg.det(temp_inv_cov)
            if det_inv > 0 and 1.0 / det_inv < best_det:
                best_det = 1.0 / det_inv
                best_omega = o
        omega = best_omega

    # Fusionierte Kovarianz
    inv_cov_fused = omega * inv_cov1 + (1 - omega) * inv_cov2
    cov_fused = np.linalg.inv(inv_cov_fused)

    # Fusionierter Mittelwert
    mean_fused = cov_fused @ (omega * inv_cov1 @ mean1 + (1 - omega) * inv_cov2 @ mean2)

    return mean_fused, cov_fused
