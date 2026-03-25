import numpy as np
import pandas as pd
from pyadm1 import BiogasPlant
from pyadm1.substrates import Feedstock
from pyadm1ode_estimation.estimation.ukf import ADM1UKF

def main():
    # 1. Setup ADM1 model
    feedstock = Feedstock(feeding_freq=48)
    plant = BiogasPlant("Example Plant")
    # In a real scenario, we'd add components and initialize properly.
    # For the example, we'll use the underlying ADM1 core directly if possible
    # or a simplified initialization.
    from pyadm1.core import ADM1
    adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)

    # 2. Setup UKF
    # Process noise covariance
    Q = np.eye(37) * 0.01
    # Measurement noise covariance (e.g., pH and Methane production are measured)
    # Note: Measurement indices depend on ADM1 state vector mapping
    # 36: pTOTAL, 8: S_ch4, etc.
    measurement_indices = [8, 36] # S_ch4, pTOTAL
    R = np.eye(2) * 0.05

    ukf_wrapper = ADM1UKF(
        adm1_model=adm1,
        Q=Q,
        R=R,
        measurement_indices=measurement_indices
    )

    # 3. Initial state and covariance
    x0 = np.full(37, 0.1) # Dummy initial state
    P0 = np.eye(37) * 0.1

    # 4. Estimation loop (simulated)
    u = np.array([15.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0]) # Influent flow rates
    dt = 1.0 / 24.0 # 1 hour time step

    print("Starting estimation loop...")
    x = x0
    P = P0

    for t in range(5):
        # Simulated measurement (in reality from sensors)
        # Here we just take the "true" state plus some noise
        true_x = adm1.ADM1_ODE(0, x.tolist()) # This is just a derivative, not next state
        # In a real example, we'd simulate the plant to get next true state
        y = x[measurement_indices] + np.random.normal(0, 0.01, size=2)

        # UKF Step
        x, P = ukf_wrapper.estimate(x, P, u, y, dt)

        print(f"Step {t+1}: Estimated Methane (S_ch4) = {x[8]:.4f}, Pressure (pTOTAL) = {x[36]:.4f}")

if __name__ == "__main__":
    main()
