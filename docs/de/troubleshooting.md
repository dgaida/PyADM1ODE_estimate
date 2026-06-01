# Fehlerbehebung

Typische Probleme und Lösungen.

## Installation

### `ImportError: No module named 'pyadm1'`

Das Basis-Paket muss separat installiert sein:

```bash
pip install git+https://github.com/dgaida/PyADM1ODE.git
```

### `pythonnet` schlägt fehl

`pythonnet` benötigt .NET-Runtime. Auf Windows ist dies meistens vorhanden, auf
Linux:

```bash
sudo apt install mono-complete
```

## Filter-Verhalten

### NIS-Werte chronisch >> n_obs

Der Filter unterschätzt die Unsicherheit. Mögliche Ursachen:

* `process_noise_std` zu klein in einzelnen `StateChannel`s.  
* `noise_std` der `ObservationChannel`s zu klein (Sensor-Rauschen unterschätzt).  
* Modellfehler — z.B. ein nicht modellierter Inhibitor wirkt in der Realität.  

Erste Maßnahme: `process_noise_std` jedes biologischen Channels um Faktor 2–3
erhöhen.

### Filter konvergiert nicht zu sinnvollen Werten

* **Initial-State prüfen**: bei großem initialen Fehler dauert die Konvergenz  
  typischerweise 10–15 Tage (siehe Haugen et al. 2014).  
* **Observability prüfen**: ist der jeweilige Zustand mit der vorhandenen  
  Sensorik überhaupt schätzbar? Siehe
  [Observability-Literaturüberblick](observability/literature_review.md).  
* **Bounds prüfen**: zu enge `lower`/`upper`-Bounds clippen aggressiv und  
  unterdrücken Information.

### `LinAlgError` aus `_cholupdate`-Downdate

Der Square-Root-UKF (`filters/sr_ukf.py`) speichert den Cholesky-Faktor
$S$ direkt. Im Update-Schritt wird ein Rank-1-Downdate $S \to S'$ mit
$S'S'^\top = SS^\top - vv^\top$ gemacht. Wenn dieses Downdate fehlschlägt
($S'$ wäre nicht mehr reell), heißt das: der Filter versucht *mehr*
Information abzuziehen, als die Prior-Kovarianz bereitstellt. Mögliche
Ursachen:

* `Q` (Process Noise) ist deutlich kleiner gesetzt als die tatsächliche  
  Modellunsicherheit → Filter wird zu sicher, Updates kollidieren  
* Sensor-Rauschen `R` zu klein angegeben → Filter „glaubt" der Messung  
  zu sehr  
* Bug im Messmodell (z.B. ein Extractor liefert systematisch falsche  
  Werte und die Innovation passt nicht zur Cov-Struktur)

Anders als beim klassischen UKF ist dieser Fehler *informativ*: er sagt
strukturell, wo das Problem liegt, statt mit Eigenvalue-Flooring zu
maskieren.

## Calibration-Artifact

### `ValueError: schema_version=999`

Das Artefakt hat eine neuere Schema-Version als diese Runtime-Version
unterstützt. Entweder Estimation-Paket aktualisieren oder die Calibration
auf eine kompatible Version zurücksetzen.

### `apply_to_plant`: viele "Übersprungen"-Warnungen

Das Artefakt referenziert Komponenten oder kinetische Parameter, die im
aktuellen Plant-Build nicht existieren. Wenn das beim Hochfahren einer
Produktion passiert, **immer mit `strict=True` aufrufen** — fail-fast statt
silently halb angewendet.

## Doku lokal bauen

### `mkdocs serve` schlägt fehl

```bash
pip install -e ".[docs]"
```

stellt sicher, dass alle Doku-Abhängigkeiten (Material-Theme, i18n-Plugin,
mkdocstrings, mermaid2) vorhanden sind.
