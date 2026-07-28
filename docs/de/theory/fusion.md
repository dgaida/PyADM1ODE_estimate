# UKF ↔ PINN-Fusion (Kovarianzschnitt)

Der [UKF](../usage/ukf.md) und die PINN-Familie haben **komplementäre** Stärken:
Der UKF ist gut gestellt und gut kalibriert auf den pH-/Ladungsbilanz-Zuständen,
das PINN ([Smoother](pinn.md) oder [amortisierter Observer](observer.md)) ist
stark auf den biogastreibenden Zuständen und prognostiziert natürlich. Die Fusion
kombiniert ihre beiden Zustandstrajektorien, sodass das Ergebnis das Beste aus
beiden behält.

---

## 1. Die Idee: Covariance Intersection

Wir haben zwei Schätzungen desselben Zustands, wissen aber **nicht**, wie
korreliert ihre Fehler sind (sie sehen dieselbe Anlage und dasselbe Rauschen).
Covariance Intersection (CI) fusioniert sie *konservativ* ohne Unabhängigkeit
anzunehmen, sodass die fusionierte Schätzung konsistent bleibt (nie
überkonfident) und mindestens so informativ ist wie jede der beiden Eingaben.

Je Zustand gewichtet CI nach **Information** (inverse Varianz): die konfidentere
Schätzung trägt mehr bei. Ein einzelnes Gewicht $\omega \in [0,1]$ je Zeitschritt
wägt die beiden gegeneinander ab.

---

## 2. Die Mathematik

Nimm zwei Schätzungen mit Diagonalkovarianz, Mittelwerten $m_a, m_b$ und
zustandsweisen Varianzen $v_a = \sigma_a^2,\ v_b = \sigma_b^2$, und schreibe die
Information (inverse Varianz) als $I = 1/v$. CI fusioniert sie als

$$
I_f = \omega\, I_a + (1-\omega)\, I_b, \qquad
m_f = I_f^{-1}\big(\omega\, I_a\, m_a + (1-\omega)\, I_b\, m_b\big),
$$

mit fusionierter Varianz $v_f = 1/I_f$. Das Gewicht $\omega$ ist entweder fest
oder wird je Zeitschritt **optimiert** durch eine Gittersuche, die die
Determinante der fusionierten Kovarianz minimiert. Äquivalent
$\sum_\text{Zustände}\log I_f$ maximiert.

Da $\omega$ je Schritt gewählt wird und die Kovarianzen diagonal sind (aus den
zustandsweisen Standardabweichungen gebaut), ist die Fusion günstig und wirkt auf
eine ganze Trajektorie auf einmal.

---

## 3. Umsetzung

`fuse_ci_diagonal(mean_a, std_a, mean_b, std_b, omega=None)` ist der Kern: ein
CI je Zeitschritt zweier `(T, n)`-Schätzungen, das den fusionierten Mittelwert,
die fusionierte Std und das gewählte $\omega$ je Schritt zurückgibt.

`HybridEstimator` verpackt das im [`BatchEstimator`](pinn.md)-Vertrag. Baue ihn
aus den beiden Teilschätzungen, und `estimate()` gibt ihre CI-Fusion als
`TrajectoryEstimate` zurück:

```python
from pyadm1ode_estimation.estimation.fusion import HybridEstimator

hybrid = HybridEstimator(traj_ukf, traj_pinn)   # zwei TrajectoryEstimates
fused = hybrid.estimate()                        # CI-Fusion auf dem gemeinsamen Gitter
```

Beide Eingaben müssen auf demselben Zeitgitter liegen, `fuse_trajectories_ci`
erzwingt passende Formen.

---

## 4. Vorbehalt: Kalibrierung ist Voraussetzung

CI gewichtet rein nach Kovarianz, ist also **nur so gut wie die Kalibrierung der
Eingaben**. Ein überkonfidenter Schätzer z. B. zu enge MC-Dropout-Bänder wird
zu stark vertraut und zieht die Fusion zu sich. Im Twin-Vergleich *schadet* die
Fusion nie, kann aber die pH-Stärke des UKF noch nicht in die Zustandsschätzung
einmischen, weil die Unsicherheit des PINN noch nicht kalibriert ist. Das
PINN-Band zu kalibrieren (und die pH-Konditionierung zu beheben) ist der offene
Punkt, der die Fusion freischaltet.

---

## Quelldateien

* `pyadm1ode_estimation/estimation/fusion/hybrid.py` — `HybridEstimator`, `fuse_ci_diagonal`, `fuse_trajectories_ci`  

## API-Referenz

::: pyadm1ode_estimation.estimation.fusion.hybrid.HybridEstimator
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: pyadm1ode_estimation.estimation.fusion.hybrid.fuse_ci_diagonal
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
