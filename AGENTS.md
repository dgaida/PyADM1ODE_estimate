# Instructions for AI Agents

Welcome to the **pyadm1ode_estimation** repository. This file contains specific instructions and tips for working with this codebase.

## Technical Context

- This project focuses on state estimation for the **Anaerobic Digestion Model No. 1 (ADM1)**.  
- The core mechanistic model is provided by the [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) package, which implements **ADM1da** (Schlattmann 2011), an agricultural-biogas extension.  
- State estimation approaches include **Unscented Kalman Filter (UKF)**, **Deep Learning Ensembles**, and **Physics-Informed Neural Networks (PINNs)**.  
- The ADM1da state vector $x$ has **41 components** — 12 dissolved, 10 particulate (with sub-fraction disintegration X_PS slow + X_PF fast, each in ch/pr/li, plus X_S_ch/pr/li hydrolysable + X_I), 7 biomass, 8 charge-balance, 4 gas-phase. See [pyadm1/core/adm1.py](https://github.com/dgaida/PyADM1ODE/blob/main/pyadm1/core/adm1.py) for the index map.  

## Development Guidelines

1. **Bilingual Documentation**: All docstrings, comments, and documentation files MUST be written in both **German** and **English** to support the research context.  
2. **PyTorch for Deep Learning**: Use `torch` for all neural network implementations (Ensembles, PINNs).  
3. **ODE Solvers**: When integrating the ADM1 model, prefer the 'BDF' method from `scipy.integrate.solve_ivp` as the system is often stiff.  
4. **State Constraints**: ADM1 state variables (concentrations) must remain non-negative. Ensure your implementations (especially PINN outputs or UKF updates) respect this constraint.  
5. **Testing**: Add tests for new estimation algorithms in the `tests/` directory. Use `pytest` to run them.  

## Key Files

- `pyadm1ode_estimation/estimation/filters/sr_ukf.py`: Production Square-Root UKF (Wan & Van der Merwe 2001) with scaled sigma points and gated observations (pyadm1 BiogasPlant-based). Re-exported as `UnscentedKalmanFilter` from `filters/__init__.py` — the SR formulation is the only filter in the package.  
- `pyadm1ode_estimation/estimation/process_model.py`: ADM1 process model wrapping pyadm1 for sigma-point propagation.  
- `pyadm1ode_estimation/estimation/state_vector.py` / `observation_model.py`: Channel declarations for state and measurements.  
- `pyadm1ode_estimation/estimation/twin.py`: Helpers for synthetic-truth twin experiments.  
- `pyadm1ode_estimation/estimation/deep_learning/pinn.py`: Physics-informed state estimation.  
- `pyadm1ode_estimation/artifacts/calibration_artifact.py`: YAML handoff format from calibration to estimation (calibrated kinetic params, substrate fractions, initial state).  
- `examples/`: End-to-end twin experiment scripts.  
- `docs/de/theory/pinn.md`: Detailed explanation of the PINN approach.  
- `docs/assets/calibration_artifact_example.yaml`: Example calibration artifact (generic).  

## Pre-commit Checks

Before submitting changes, ensure:  
- `ruff` is used for linting.  
- `black` is used for formatting.  
- All tests pass: `python3 -m pytest`.  
