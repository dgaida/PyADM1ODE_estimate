"""PINN v2 — the collocation trajectory PINN for the ADM1da.

One network per series writes the state trajectory ``x(t) = exp(mu + sigma z(t, u))``
and is trained on the sensor fit plus the ADM1da residual at collocation points;
:mod:`.rolling` runs it causally (daily warm-start retraining on a sliding window).
It shares the differentiable dynamics :func:`pyadm1.core.adm1_torch.adm1da_rhs_torch`
with the UKF's torch backend, and its data with every other estimator through
:mod:`..filter_tuning.datasets`.

* :mod:`.influent` — substrate feed -> ``(q_ad, s_in)`` for a batch of time points.
* :mod:`.physics`  — batched RHS, ion /
  gas closures, residual selection + scaling, finite-difference verification.
* :mod:`.measure`  — differentiable measurement model and sensor noise.
* :mod:`.model`    — :class:`~.model.TrajectoryNet` and its :class:`~.model.ArchConfig`.
* :mod:`.train`    — one trajectory fit (:func:`~.train.train_trajectory`) and metrics.
* :mod:`.rolling`  — causal operation (:func:`~.rolling.run_rolling`).
* :mod:`.data`     — :class:`~.data.BenchmarkSeries` on top of the shared loader.

Use it through the registry: ``build_estimator("pinn", dataset)`` in
:mod:`..estimators`. Equation sheet: ``private_docs/pinn_v2_physics_equations.md``.
"""

from .influent import MODES, N_INFLUENT, InfluentModel
from .physics import (
    ACID_BASE_MODES,
    CLOSURE_ODE_STATES,
    CONSERVED_PAIRS,
    CORE_STATES,
    FULL_ODE_STATES,
    GAS_STATES,
    ION_PAIRS,
    ION_STATES,
    N_CORE,
    ADM1Physics,
    PhysicsConfig,
    acid_base_rates,
    build_params,
    close_state,
    close_state_with_rates,
    ion_equilibrium_residual,
    proton_concentration,
    rhs,
    solve_charge_balance,
    trajectory_residual,
    with_influent,
)

__all__ = [
    "ACID_BASE_MODES",
    "CLOSURE_ODE_STATES",
    "CONSERVED_PAIRS",
    "CORE_STATES",
    "FULL_ODE_STATES",
    "GAS_STATES",
    "ION_PAIRS",
    "ION_STATES",
    "MODES",
    "N_CORE",
    "N_INFLUENT",
    "ADM1Physics",
    "InfluentModel",
    "PhysicsConfig",
    "acid_base_rates",
    "build_params",
    "close_state",
    "close_state_with_rates",
    "ion_equilibrium_residual",
    "proton_concentration",
    "rhs",
    "solve_charge_balance",
    "trajectory_residual",
    "with_influent",
]
