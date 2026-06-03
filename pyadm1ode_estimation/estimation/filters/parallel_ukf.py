"""Process-pool parallelised SR-UKF.

Drop-in subclass of :class:`UnscentedKalmanFilter` that parallelises the
``2n+1`` plant integrations of :meth:`predict` over a worker pool. The
linear-algebra path (QR, cholupdate, weights) stays on the main process
— only the embarrassingly parallel sigma-point propagation moves to
workers.

Worker model
------------
Each worker builds its own ``(process, obs, spec)`` once at pool
startup via a user-provided ``components_builder`` callable. Per task,
the main process ships:

* the current ``process._baseline`` snapshot dict (~1 kB on the simple
  plant; restores the worker's plant to a deterministic baseline);
* the sigma-point state vector ``s_i`` (~n floats);
* the timestep ``dt``.

Workers restore from the snapshot, apply ``s_i``, step ``dt``, and
return ``(propagated_state, h_values)``. The plant object never crosses
the IPC boundary at runtime — only the snapshot dict does.

Constraints
-----------
* ``components_builder`` must be a top-level (importable) function; a
  lambda or closure-bound method will fail to pickle for the worker
  initialiser. The builder is called once per worker and must produce a
  ``(process, obs, spec)`` triple consistent with what the main filter
  uses — same digester, same substrate slots, same sensor set, same
  channel ordering.
* The main filter's ``(process, obs, spec)`` must be produced by the
  same builder so the snapshot dicts are interchangeable across the
  main + worker plants.

Pool lifecycle
--------------
The pool is created in ``__init__`` and held for the lifetime of the
filter. Call :meth:`shutdown` (or use the filter as a context manager)
to release worker processes explicitly. ``__del__`` calls
``shutdown`` defensively, but garbage-collection timing on Windows
makes the explicit form safer.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Optional, Tuple

import numpy as np

from .sr_ukf import UnscentedKalmanFilter

# Worker-local globals, populated by _worker_init in each child process.
# Never accessed from the parent.
_WORKER_PROCESS: Any = None
_WORKER_OBS: Any = None


def _worker_init(components_builder: Callable[[], Tuple[Any, Any, Any]]) -> None:
    """Pool initialiser: build the worker's local ``(process, obs, spec)``.

    Runs once per worker process. ``components_builder`` must be a
    top-level importable callable; ``multiprocessing`` pickles it for
    the worker spawn.
    """
    global _WORKER_PROCESS, _WORKER_OBS
    process, obs, _spec = components_builder()
    process.snapshot()  # baseline so the worker has one before any task arrives
    _WORKER_PROCESS = process
    _WORKER_OBS = obs


def _worker_step(args):
    """Per-task action: restore snapshot, step one sigma, read h-values."""
    snapshot, sigma, dt = args
    process = _WORKER_PROCESS
    obs = _WORKER_OBS
    # Restore worker's plant to the main process's current baseline.
    # process.step internally calls restore() before applying state.
    process._baseline = snapshot
    propagated = process.step(sigma, dt)
    h_values = np.array(
        [float(c.extractor(process.plant, propagated)) for c in obs.channels],
        dtype=float,
    )
    return propagated, h_values


class ParallelUKF(UnscentedKalmanFilter):
    """SR-UKF with sigma-point propagation farmed out to a worker pool.

    Constructor takes everything :class:`UnscentedKalmanFilter` takes,
    plus:

    Args:
        n_workers: Number of worker processes. ``1`` falls through to
            the serial implementation (no pool overhead).
        components_builder: Top-level callable returning
            ``(process, obs, spec)``. Called once per worker on pool
            startup. Must be importable (not a lambda or inner
            function) and must produce a plant identical in topology
            and substrate slot count to the main filter's plant.

    Example::

        # Top-level function so multiprocessing can pickle it.
        def make_components():
            from pyadm1ode_estimation.example_plants import build_simple_plant
            from pyadm1ode_estimation.estimation import (
                InputSpec, build_filter_components,
            )
            plant = build_simple_plant()
            return build_filter_components(
                plant, digester_id="fermenter",
                substrates=[
                    InputSpec("maize_silage", substrate_index=0, initial_flow=10.0),
                    InputSpec("cattle_slurry", substrate_index=1, initial_flow=5.0),
                ],
                sensors=["q_gas", "q_ch4", "ph", "substrate_dose"],
            )

        process, obs, spec = make_components()
        ukf = ParallelUKF(
            process, obs, spec,
            n_workers=4,
            components_builder=make_components,
        )
        # ... use like a normal UKF ...
        ukf.shutdown()
    """

    def __init__(
        self,
        process,
        obs,
        spec,
        *,
        n_workers: int = 1,
        components_builder: Optional[Callable[[], Tuple[Any, Any, Any]]] = None,
        **ukf_kwargs,
    ):
        super().__init__(process, obs, spec, **ukf_kwargs)
        self.n_workers = int(n_workers)
        self.components_builder = components_builder
        self._pool: Optional[Any] = None
        if self.n_workers > 1:
            if components_builder is None:
                raise ValueError(
                    "ParallelUKF requires components_builder when n_workers > 1."
                )
            self._init_pool()

    def _init_pool(self) -> None:
        # Spawn explicitly so behaviour is identical across platforms
        # (Windows is spawn-only; Linux defaults to fork, which would
        # silently share state). Spawn is also required for the plant
        # build to happen cleanly in each child.
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            processes=self.n_workers,
            initializer=_worker_init,
            initargs=(self.components_builder,),
        )

    def shutdown(self) -> None:
        """Release worker processes. Safe to call multiple times."""
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            finally:
                self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Hook override: parallel sigma-point propagation
    # ------------------------------------------------------------------
    def _propagate_sigma_points(
        self, sigma: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._pool is None:
            return super()._propagate_sigma_points(sigma, dt)

        # Snapshot dict is shared across all tasks of this predict — it
        # was just taken on self.process at the start of predict.
        snapshot = self.process._baseline
        tasks = [(snapshot, sigma[i], dt) for i in range(sigma.shape[0])]
        # map preserves task order, so propagated[i] corresponds to
        # sigma[i] exactly as in the serial loop.
        results = self._pool.map(_worker_step, tasks)
        propagated = np.array([r[0] for r in results])
        h_all = np.array([r[1] for r in results])
        return propagated, h_all
