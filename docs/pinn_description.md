# Physics-Informed Neural Networks (PINN) for ADM1

This document describes the implementation and functionality of Physics-Informed Neural Networks (PINNs) for state estimation of the Anaerobic Digestion Model No. 1 (ADM1).

## Concept / Konzept

Physics-Informed Neural Networks combine the flexibility of deep learning with the rigorous structure of mechanistic models. Unlike standard neural networks that rely solely on data, PINNs incorporate the underlying physical laws (expressed as Ordinary Differential Equations, ODEs) directly into the loss function.

Physics-Informed Neural Networks kombinieren die Flexibilität von Deep Learning mit der strengen Struktur mechanistischer Modelle. Im Gegensatz zu Standard-Netzwerken, die rein datenbasiert arbeiten, integrieren PINNs die zugrunde liegenden physikalischen Gesetze (ausgedrückt als gewöhnliche Differenzialgleichungen, ODEs) direkt in die Verlustfunktion.

## Loss Function / Verlustfunktion

The total loss $ consists of two main components:
Die Gesamtverlustfunktion $ besteht aus zwei Hauptkomponenten:

3237L = L_{data} + \lambda_{phys} \cdot L_{phys}3237

### 1. Data Loss ({data}$)
Measures the difference between the network's predictions and the actual measurements (Biogas production, CH4 concentration, pH value).
Misst die Differenz zwischen den Vorhersagen des Netzwerks und den tatsächlichen Messwerten (Biogasproduktion, Methankonzentration, pH-Wert).

### 2. Physics Loss ({phys}$)
Measures how well the network's predictions satisfy the ADM1 ODEs. It uses automatic differentiation to compute the time derivative of the state vector and compares it with the state derivatives calculated by the model equations.
Misst, wie gut die Vorhersagen des Netzwerks die ADM1-ODEs erfüllen. Es nutzt automatische Differenzierung, um die Zeitableitung des Zustandsvektors zu berechnen und vergleicht diese mit den durch die Modellgleichungen berechneten Zustandsableitungen.

3237L_{phys} = \left\| \frac{d\hat{x}}{dt} - f(\hat{x}, u) \right\|^23237

## Inputs and Outputs / Ein- und Ausgänge

- **Inputs**: Time $ and time-dependent substrate feed (t)$.
- **Outputs**: Estimated ADM1 state vector $\hat{x}(t)$ (37 components).

- **Eingänge**: Zeit $ und zeitabhängige Substratzufuhr (t)$.
- **Ausgänge**: Geschätzter ADM1-Zustandsvektor $\hat{x}(t)$ (37 Komponenten).

## Integration in PyADM1ODE_estimation

The PINN is part of the deep learning ensemble (AP 4.3) and provides a robust alternative to purely data-driven or purely mechanistic state estimation (UKF).
Das PINN ist Teil des Deep Learning Ensembles (AP 4.3) und bietet eine robuste Alternative zu rein datengetriebenen oder rein mechanistischen Zustandsschätzungen (UKF).
