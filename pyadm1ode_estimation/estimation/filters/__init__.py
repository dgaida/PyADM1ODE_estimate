"""Filter implementations.

The exported ``UnscentedKalmanFilter`` is the Square-Root UKF
(Wan & Van der Merwe 2001) - see :mod:`sr_ukf` for the algorithmic
details. The "SR" prefix is dropped from the public name because it
is the sole filter in the package. Callers never have to think about
the square-root formulation as a separate variant.
"""

from .sr_ukf import UnscentedKalmanFilter

__all__ = ["UnscentedKalmanFilter"]
