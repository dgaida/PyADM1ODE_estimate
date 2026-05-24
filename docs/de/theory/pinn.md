# Physics-Informed Neural Networks (PINN) für ADM1

Dieses Dokument beschreibt die Implementierung und Funktionsweise von
Physics-Informed Neural Networks (PINNs) für die Zustandsschätzung
des Anaerobic Digestion Model No. 1 (ADM1).

## Konzept

Physics-Informed Neural Networks kombinieren die Flexibilität von Deep
Learning mit der strengen Struktur mechanistischer Modelle. Im Gegensatz
zu Standard-Netzwerken, die rein datenbasiert arbeiten, integrieren PINNs
die zugrunde liegenden physikalischen Gesetze (ausgedrückt als gewöhnliche
Differenzialgleichungen, ODEs) direkt in die Verlustfunktion.

## Verlustfunktion

Die Gesamtverlustfunktion $L$ besteht aus zwei Hauptkomponenten:

$$
L = L_{\text{data}} + \lambda_{\text{phys}} \cdot L_{\text{phys}}
$$

### 1. Daten-Loss ($L_{\text{data}}$)

Misst die Differenz zwischen den Vorhersagen des Netzwerks und den
tatsächlichen Messwerten (Biogasproduktion, Methankonzentration, pH-Wert).

### 2. Physik-Loss ($L_{\text{phys}}$)

Misst, wie gut die Vorhersagen des Netzwerks die ADM1-ODEs erfüllen.
Automatische Differenzierung wird genutzt, um die Zeitableitung des
Zustandsvektors zu berechnen und mit den durch die Modellgleichungen
berechneten Zustandsableitungen zu vergleichen.

$$
L_{\text{phys}} = \left\| \frac{d\hat{x}}{dt} - f(\hat{x}, u) \right\|^2
$$

## Ein- und Ausgänge

* **Eingänge**: Zeit $t$ und zeitabhängige Substratzufuhr $u(t)$.
* **Ausgänge**: Geschätzter ADM1-Zustandsvektor $\hat{x}(t)$ (41 Komponenten).

## Integration in PyADM1ODE_estimation

Das PINN ist Teil des Deep-Learning-Ensembles (AP 4.3) und bietet eine
robuste Alternative zu rein datengetriebenen oder rein mechanistischen
Zustandsschätzungen (UKF).
