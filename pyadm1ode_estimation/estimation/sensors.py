"""Adapters for PyADM1ODE sensor classes to fit the twin-experiment loop.

PyADM1ODE provides ``PhysicalSensor``, ``ChemicalSensor`` and
``GasSensor`` classes that model realistic instrumentation (drift,
response lag, sampling intervals, calibration offset, detection limits,
cross-sensitivity). These are *stateful* — each ``sensor.step(t, dt,
inputs)`` advances the drift accumulator, samples the queue, and
applies the lag filter.

**Why they can't be used directly inside the UKF observation model:**
the UKF's update step calls ``extractor(plant, x)`` once per sigma
point (2n+1 = 89 times for a 44-state filter). If the extractor were
to call ``sensor.step()``, the sensor's drift / sampling clock would
advance 89 times per filter step — completely corrupting the
instrumentation state.

**Where they CAN be used:** on the *truth side* of a twin experiment.
Each sensor is stepped exactly once per simulated measurement, fed
the clean truth value, and produces a noisy measurement that
replaces the simple ``add_measurement_noise`` Gaussian model.

This module provides:

* :class:`SensorAdapter` — wraps one PyADM1ODE sensor with a
  ``read(t, true_value) -> float`` API and tracks ``dt`` internally.
* :func:`measure_truth_with_sensors` — DataFrame-level helper that
  runs the sensors against a clean truth trajectory and returns
  the noisy measurement frame.

The filter-side ``ObservationChannel.extractor`` remains a clean
deterministic function — the filter never touches the sensors. The
``ObservationChannel.noise_std`` should match the sensor's
``measurement_noise`` so the UKF's ``R`` matrix is consistent with
the actual sensor noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyadm1.components.sensors._base import AbstractSensor  # type: ignore[import-not-found]


class SensorAdapter:
    """UKF-friendly wrapper around a PyADM1ODE ``AbstractSensor``.

    The wrapped sensor sees one step per measurement, never per sigma
    point. ``dt`` is tracked internally from the time of the previous
    ``read`` call so the caller only has to provide ``t`` and the
    true signal value.

    Args:
        sensor: An already-constructed PyADM1ODE sensor instance
            (``PhysicalSensor``, ``ChemicalSensor`` or ``GasSensor``).
        signal_key: Key under which the true value will be passed to
            the sensor's ``step()`` inputs dict. Must match one of
            the sensor's ``_candidate_keys``. If the sensor was built
            with an explicit ``signal_key``, use that same value here.
    """

    def __init__(self, sensor: "AbstractSensor", signal_key: str):
        self._sensor = sensor
        self._signal_key = signal_key
        self._t_last: Optional[float] = None
        # Make sure the sensor is in a clean state before its first read.
        self._sensor.initialize()

    @property
    def name(self) -> str:
        return self._sensor.component_id

    def reset(self, t0: float = 0.0) -> None:
        """Re-initialise the sensor before a new run.

        Clears drift, lag-filter state and the last-read time.
        """
        self._sensor.initialize()
        self._t_last = float(t0)

    def read(self, t: float, true_value: float) -> float:
        """Step the sensor by ``dt = t − t_last`` against ``true_value``.

        Returns the measured value (post drift + lag + sampling + noise +
        range clipping). NaN-safe: if the sensor returns a non-finite
        value, we fall back to the clean ``true_value``.
        """
        if self._t_last is None:
            dt = 0.0
        else:
            dt = max(0.0, float(t) - self._t_last)

        if not np.isfinite(true_value):
            # Sensor would propagate NaN through drift/lag math.
            # Skip the step and pretend the sensor saw nothing.
            self._t_last = float(t)
            return float("nan")

        inputs = {self._signal_key: float(true_value)}
        out = self._sensor.step(t=float(t), dt=dt, inputs=inputs)
        self._t_last = float(t)

        # The measured value lives under the sensor's configured
        # output_key. Different sensor types might also surface it
        # under "measurement"; we try both for robustness.
        for key in (self._sensor.output_key, "measurement"):
            if key in out:
                val = out[key]
                if np.isfinite(val):
                    return float(val)
        return float(true_value)  # last-resort fallback


def measure_truth_with_sensors(
    obs_clean: pd.DataFrame,
    sensors: Dict[str, SensorAdapter],
) -> pd.DataFrame:
    """Step each sensor through the clean truth trajectory.

    Replaces :func:`pyadm1ode_estimation.estimation.twin.add_measurement_noise`
    when realistic sensor models (drift, response lag, sampling) are
    required. Columns that have no matching sensor in ``sensors`` are
    copied through unchanged.

    Args:
        obs_clean: DataFrame indexed by simulation time (days), one
            column per observation channel, containing the clean
            truth values ``h(x_truth[k])``.
        sensors: Mapping ``{channel_name: SensorAdapter}``. The
            channel name must match a column in ``obs_clean``.

    Returns:
        A DataFrame with the same shape / index as ``obs_clean`` but
        with each ``sensors[name]``-mapped column replaced by the
        sensor-modulated measurements.
    """
    out = obs_clean.copy()
    times = obs_clean.index.to_numpy(dtype=float)

    for name, adapter in sensors.items():
        if name not in obs_clean.columns:
            continue
        adapter.reset(t0=float(times[0]))
        col_idx = out.columns.get_loc(name)
        for i, t in enumerate(times):
            true_value = float(obs_clean.iloc[i][name])
            measured = adapter.read(t=float(t), true_value=true_value)
            out.iloc[i, col_idx] = measured

    return out


__all__ = ["SensorAdapter", "measure_truth_with_sensors"]
