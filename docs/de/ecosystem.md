# Ecosystem

`PyADM1ODE_estimate` ist eine Erweiterungen des Basis-Pakets **PyADM1ODE** und arbeitet mit der Erweiterung PyADM1ODE_calibration zusammen.

## Die drei Komponenten

### [PyADM1ODE](https://dgaida.github.io/PyADM1ODE/latest/) — Basis

Das mechanistische Prozessmodell. Implementiert **ADM1da** (Schlattmann 2011),
ein 41-State-ODE-System für die anaerobe Vergärung in landwirtschaftlichen
Biogasanlagen. Enthält:

* `pyadm1.core` — ODE und Parameter  
* `pyadm1.components` — modulare Anlagenbausteine (Fermenter, BHKW, Heizung, …)  
* `pyadm1.configurator` — Plant-Builder-API  
* `pyadm1.substrates` — Substrat-Bibliothek  
* `pyadm1.simulation` — Simulator und Parallel-Simulator  

→ [Dokumentation](https://dgaida.github.io/PyADM1ODE/latest/)
→ [GitHub](https://github.com/dgaida/PyADM1ODE)

### [PyADM1ODE_calibration](https://dgaida.github.io/PyADM1ODE_calibration/latest/) — Offline-Kalibrierung

Anpassung der ADM1-Parameter an historische Anlagendaten. Enthält:

* IO-Pipeline für SCADA-Exporte (CSV)  
* Lokale + globale Optimierer  
* Sensitivitäts- und Identifizierbarkeitsanalyse  
* SQLAlchemy-Persistenz für Calibration-Runs  
* Plant-Builder für reale Anlagen  

→ [Dokumentation](https://dgaida.github.io/PyADM1ODE_calibration/latest/)
→ [GitHub](https://github.com/dgaida/PyADM1ODE_calibration)

### PyADM1ODE_estimate — Online-Schätzung

UKF und (geplant) Deep Learning + Fusion zur Echtzeit-Schätzung des
Anlagenzustands. Liest das von Calibration produzierte
[Kalibrierungs-Artefakt](usage/calibration_artifact.md), schätzt aus aktuellen
Sensorwerten den vollen Zustandsvektor und liefert ihn an Regler.

→ [GitHub](https://github.com/dgaida/PyADM1ODE_estimate)

## Datenfluss zwischen den Repos

```text
                       PyADM1ODE
                  (ADM1da-Modell, Plant-API)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   PyADM1ODE_calibration         PyADM1ODE_estimate
            │                           │
            │  schreibt YAML            │  liest YAML +
            ▼                           ▼  Live-Sensoren
   ┌────────────────────┐         ┌───────────────┐
   │ CalibrationArtifact│ ──────► │ UKF / Filter  │
   └────────────────────┘         └───────┬───────┘
                                          ▼
                                  Zustandsschätzung
                                          ▼
                                    Regler / UI
```

## Welches Repo wofür installieren

| Anwendung | Pakete |
|---|---|
| Reines Simulieren der Anlage | nur `pyadm1` |
| Parameter aus historischen Daten fitten | `pyadm1` + `pyadm1ode_calibration` |
| Live-Filter auf produktiver Anlage | `pyadm1` + `pyadm1ode_estimation` |
| End-to-End (Kalibrierung + Live) | alle drei |

Im Live-Betrieb kommt `pyadm1ode_calibration` nur **periodisch** zum Einsatz
(typisch alle paar Wochen bis Monate, wenn der Operator eine Neukalibrierung
anstößt); `pyadm1ode_estimation` läuft kontinuierlich.

## Versionierung über Repos hinweg

Die Doku jedes Repos hat einen `mike`-Versions-Selector (oben rechts).
Empfehlung: bei zusammenhängenden Releases dieselbe Versionsnummer in allen
drei Repos verwenden — dann findet der Nutzer in jeder Versionsleiste denselben
Stand.

## Wer mehr Doku schreibt

Die Repos sind unabhängig veröffentlicht. Beiträge zur Ökosystem-übergreifenden
Konsistenz sind willkommen — siehe
[Ecosystem-Integration](development/ecosystem-integration.md) für die
Checkliste, wie eine neue Erweiterung sich an die anderen anschließt.
