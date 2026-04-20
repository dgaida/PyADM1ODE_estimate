# Instructions for AI Agents

Welcome to the **pyadm1ode_estimation** repository. This file contains specific instructions and tips for working with this codebase.

## Technical Context

- This project focuses on state estimation for the **Anaerobic Digestion Model No. 1 (ADM1)**.  
- The core mechanistic model is provided by the [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) package.  
- State estimation approaches include **Unscented Kalman Filter (UKF)**, **Deep Learning Ensembles**, and **Physics-Informed Neural Networks (PINNs)**.  
- The ADM1 state vector $x$ typically has **37 components** (concentrations, gas phases, pH).  

## Development Guidelines

1. **Bilingual Documentation**: All docstrings, comments, and documentation files MUST be written in both **German** and **English** to support the research context.  
2. **PyTorch for Deep Learning**: Use `torch` for all neural network implementations (Ensembles, PINNs).  
3. **ODE Solvers**: When integrating the ADM1 model, prefer the 'BDF' method from `scipy.integrate.solve_ivp` as the system is often stiff.  
4. **State Constraints**: ADM1 state variables (concentrations) must remain non-negative. Ensure your implementations (especially PINN outputs or UKF updates) respect this constraint.  
5. **Testing**: Add tests for new estimation algorithms in the `tests/` directory. Use `pytest` to run them.  

## Key Files

- `pyadm1ode_estimation/estimation/ukf.py`: Mechanistic state estimation.  
- `pyadm1ode_estimation/estimation/deep_learning/pinn.py`: Physics-informed state estimation.  
- `docs/pinn_description.md`: Detailed explanation of the PINN approach.  

## Pre-commit Checks

Before submitting changes, ensure:  
- `ruff` is used for linting.  
- `black` is used for formatting.  
- All tests pass: `python3 -m pytest`.  
