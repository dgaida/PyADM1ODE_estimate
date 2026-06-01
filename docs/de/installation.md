# Installation

## Voraussetzungen

* **Python** ≥ 3.10  
* **PyADM1ODE** (Basis-Paket) — wird transitiv installiert. Wenn nicht  
  automatisch verfügbar, manuell:

  ```bash
  pip install git+https://github.com/dgaida/PyADM1ODE.git
  ```

* **Optional**: PyADM1ODE_calibration für Plant-Builder und historische  
  Datenanbindung.

## Standard-Installation

```bash
git clone https://github.com/dgaida/PyADM1ODE_estimate.git
cd PyADM1ODE_estimate
pip install -r requirements.txt
pip install -e .
```

Damit sind die Kern-Abhängigkeiten installiert: `numpy`, `scipy`, `pandas`,
`pythonnet`, `pyyaml`, `pyadm1`.

## Optionale Komponenten

### Deep-Learning-Komponenten (PINN, Ensemble)

```bash
pip install -e ".[deep_learning]"
```

Installiert zusätzlich `torch` und `scikit-learn`.

### Entwicklungs-Werkzeuge

```bash
pip install -e ".[dev]"
```

Installiert `pytest`, `pytest-cov`, `black`, `ruff`.

### Dokumentation lokal bauen

```bash
pip install -e ".[docs]"
mkdocs serve
```

Die Doku ist dann unter `http://localhost:8000` erreichbar.

## Verifizierung

```bash
python -m pytest tests/test_ukf.py tests/test_observation_model.py \
                 tests/test_state_vector.py tests/test_calibration_artifact.py
```

Erwartet: alle Tests grün.

## Fehlerbehebung

Siehe [Fehlerbehebung](troubleshooting.md) für typische Probleme.
