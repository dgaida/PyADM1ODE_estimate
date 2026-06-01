# Physics-Informed Neural Networks (PINN) for ADM1

This document describes the implementation and functionality of
Physics-Informed Neural Networks (PINNs) for state estimation of the
Anaerobic Digestion Model No. 1 (ADM1).

## Concept

Physics-Informed Neural Networks combine the flexibility of deep learning
with the rigorous structure of mechanistic models. Unlike standard
networks that rely solely on data, PINNs incorporate the underlying
physical laws (expressed as Ordinary Differential Equations, ODEs)
directly into the loss function.

## Loss function

The total loss $L$ consists of two main components:

$$
L = L_{\text{data}} + \lambda_{\text{phys}} \cdot L_{\text{phys}}
$$

### 1. Data loss ($L_{\text{data}}$)

Measures the difference between the network's predictions and the actual
measurements (biogas production, methane concentration, pH value).

### 2. Physics loss ($L_{\text{phys}}$)

Measures how well the network's predictions satisfy the ADM1 ODEs.
Automatic differentiation is used to compute the time derivative of the
state vector and compare it with the state derivatives calculated by the
model equations.

$$
L_{\text{phys}} = \left\| \frac{d\hat{x}}{dt} - f(\hat{x}, u) \right\|^2
$$

## Inputs and outputs

* **Inputs**: Time $t$ and time-dependent substrate feed $u(t)$.  
* **Outputs**: Estimated ADM1 state vector $\hat{x}(t)$ (41 components).  

## Integration in PyADM1ODE_estimation

The PINN is part of the deep-learning ensemble (AP 4.3) and provides a
robust alternative to purely data-driven or purely mechanistic state
estimation (UKF).
