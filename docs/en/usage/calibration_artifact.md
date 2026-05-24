# Calibration artifact

The `CalibrationArtifact` is the **interface between calibration and estimation**.
It is a versioned YAML file that calibration writes and estimation reads
when the live filter starts up.

## Workflow

```text
PyADM1ODE_calibration                     PyADM1ODE_estimate
  │                                            │
  │  optimizes kinetic parameters,             │
  │  fits against historical measurements      │
  │                                            │
  ▼                                            │
[ artifact.yaml ] ───────── handoff ──────────►│
                                               │
                                               │  load_artifact(...)
                                               │  apply_to_plant(...)
                                               │
                                               ▼
                                       calibrated BiogasPlant
                                               │
                                               ▼
                                       UKF starts from here
```

## Format overview

The YAML has five main sections (each except `metadata` is optional):

```yaml
schema_version: 1

metadata:                # Provenance: which plant, which data window
  plant_id: plant_id_xyz
  calibration_run_id: 2026-05-14-3w
  timestamp: 2026-05-14T12:30:00Z
  data_window_start: 2026-04-01
  data_window_end: 2026-04-21
  adm1_version: "ADM1da-0.3.4"
  fitted_against: [Q_gas, P_el]

kinetic:                 # adm1._kinetic overrides per digester
  primary:
    k_dis_PS: 0.045
    k_hyd_ch: 4.2
    k_m_ac: 8.5

substrates:              # Substrate disintegration fractions
  maize_silage:
    f_ch_xc: 0.55

initial_state:           # ADM1da state at the end of the calibration window
  primary:
    _values: [...41 values in canonical ADM1da order...]

residuals:               # Fit quality per channel (informational)
  Q_gas:
    rmse: 145.0
    units: m3/d
```

Full example:
[calibration_artifact_example.yaml](../../assets/calibration_artifact_example.yaml).

## Writing (calibration side)

```python
from pyadm1ode_estimation.artifacts import (
    CalibrationArtifact, CalibrationMetadata, save_artifact,
)

artifact = CalibrationArtifact(
    metadata=CalibrationMetadata(
        plant_id="plant_id_xyz",
        calibration_run_id="2026-05-14-3w",
        timestamp="2026-05-14T12:30:00Z",
        fitted_against=["Q_gas", "P_el"],
    ),
    kinetic={"primary": {"k_dis_PS": 0.045, "k_hyd_ch": 4.2}},
    initial_state={"primary": {"_values": list_of_41_floats}},
)
save_artifact(artifact, "calibrated/plant_2026-05-14.yaml")
```

## Reading (estimation side)

```python
from pyadm1ode_estimation.artifacts import load_artifact, apply_to_plant

artifact = load_artifact("calibrated/plant_2026-05-14.yaml")
plant = build_plant(schema)            # plant-specific plant builder

# strict=True: fail-fast on topology mismatch (recommended for production startup)
# strict=False: warnings only, graceful degradation (good for tests, legacy artifacts)
diag = apply_to_plant(artifact, plant, strict=True)

print(f"Applied: {len(diag['applied'])} parameters")
print(f"Skipped: {len(diag['skipped'])} parameters")
```

## Design decisions

| Decision | Why |
|---|---|
| `schema_version` mandatory | Unknown versions → `ValueError`, no silent mis-read |
| `plant_id` + `run_id` mandatory | Audit trail for multiple parallel calibrations |
| Sections optional | Partial calibrations (kinetic only) allowed |
| `strict=False` default | Graceful degradation on evolved topology |
| `strict=True` for production | Live filter startup should fail-fast on incompatibility |
| Dict structures instead of fixed fields | Plant-agnostic — a new plant doesn't require a schema update |
| `_values: [41 values]` array form | Pragmatic without pyadm1 index-map introspection |
