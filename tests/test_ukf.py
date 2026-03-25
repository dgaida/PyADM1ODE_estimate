import unittest
import numpy as np
from unittest.mock import MagicMock
from pyadm1ode_estimation.estimation.ukf import UnscentedKalmanFilter, ADM1UKF

class TestUKF(unittest.TestCase):
    def test_ukf_initialization(self):
        """Test general UKF initialization."""
        n_x = 2
        n_y = 1
        Q = np.eye(2) * 0.1
        R = np.eye(1) * 0.05

        # Simple identity models
        def f(x, u, dt): return x + u * dt
        def h(x): return x[:1]

        ukf = UnscentedKalmanFilter(n_x, n_y, f, h, Q, R)

        self.assertEqual(ukf.n_x, 2)
        self.assertEqual(ukf.n_y, 1)
        self.assertEqual(len(ukf.Wm), 5) # 2*n_x + 1
        self.assertEqual(len(ukf.Wc), 5)

    def test_ukf_predict_update(self):
        """Test one step of UKF predict and update."""
        n_x = 2
        n_y = 1
        Q = np.eye(2) * 0.01
        R = np.eye(1) * 0.01

        def f(x, u, dt): return x + u * dt
        def h(x): return x[:1]

        ukf = UnscentedKalmanFilter(n_x, n_y, f, h, Q, R)

        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        u = np.array([0.5, 0.0])
        y = np.array([1.55]) # Should be close to x[0] + u[0]*dt = 1.0 + 0.5*1.0 = 1.5
        dt = 1.0

        x_pred, P_pred, sigmas_f = ukf.predict(x, P, u, dt)
        self.assertTrue(np.allclose(x_pred, [1.5, 2.0]))

        x_upd, P_upd = ukf.update(x_pred, P_pred, sigmas_f, y)
        # updated x[0] should move towards y=1.55 from x_pred[0]=1.5
        self.assertGreater(x_upd[0], 1.5)
        self.assertLess(x_upd[0], 1.56)

    def test_adm1ukf_wrapper(self):
        """Test ADM1UKF wrapper initialization and basic call."""
        adm1_mock = MagicMock()
        # Mocking ADM1_ODE to return a list of 37 zeros
        adm1_mock.ADM1_ODE.return_value = [0.0] * 37

        Q = np.eye(37) * 0.01
        R = np.eye(2) * 0.05
        measurement_indices = [8, 36]

        adm1_ukf = ADM1UKF(adm1_mock, Q, R, measurement_indices)

        x = np.full(37, 0.1)
        P = np.eye(37) * 0.1
        u = np.zeros(10)
        y = np.array([0.1, 0.1])
        dt = 1.0

        x_upd, P_upd = adm1_ukf.estimate(x, P, u, y, dt)
        self.assertEqual(len(x_upd), 37)
        self.assertEqual(P_upd.shape, (37, 37))

if __name__ == '__main__':
    unittest.main()
