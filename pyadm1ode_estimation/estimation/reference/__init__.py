"""Shipped model parameters of the reference estimators.

* ``cmaes_<variant>_winner.json`` - the CMA-ES-tuned ``theta`` of the UKF per
  variant (``full``, ``adcore``, ``full_fostac7d``), read by
  :func:`..estimators.reference_theta`.
* ``observer.pt`` - the trained GRU observer (PINN B), the default checkpoint of
  ``build_estimator("observer", ...)``.

These are parameters, not results: the tuning and training runs that produced
them write their full output to the git-ignored ``results_*`` folders.
"""
