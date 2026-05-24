"""Calibration artifact — handoff format between calibration and estimation.

Kalibrierungs-Artefakt — Übergabeformat zwischen Kalibrierung und Zustandsschätzung.

Workflow / Ablauf
-----------------
1. ``pyadm1ode_calibration`` optimiert ADM1-Parameter gegen historische
   Messdaten und schreibt das Ergebnis als YAML via :func:`save_artifact`.
2. ``pyadm1ode_estimation`` lädt das YAML beim Hochfahren des Live-Filters
   via :func:`load_artifact` und legt die Parameter via :func:`apply_to_plant`
   auf einen frisch gebauten ``BiogasPlant``.

Design / Designprinzipien
-------------------------
* **Versioniert** — ``schema_version`` macht das Format zukunftssicher.
  Unbekannte Versionen verursachen einen Fehler statt einer stillen
  Fehlinterpretation.
* **Sektioniert und partiell** — jede Sektion (``kinetic``, ``substrates``,
  ``initial_state``, ``residuals``) ist optional. Eine Calibration die nur
  kinetische Parameter fittet schreibt nur diese Sektion.
* **Plant-agnostisch** — Dictionaries statt fester Felder erlauben beliebige
  Anlagen und ADM1-Varianten ohne Schema-Bruch.
* **Tolerant beim Apply** — unbekannte Keys oder fehlende Komponenten
  lösen eine Warnung aus, kein Crash. So bleibt ein älteres Artefakt
  gegen eine weiterentwickelte Topologie noch nutzbar (mit eingeschränkter
  Wirkung).

Example YAML
------------
.. code-block:: yaml

   schema_version: 1

   metadata:
     plant_id: plant_id_xyz
     calibration_run_id: 2026-05-14-3w
     timestamp: 2026-05-14T12:30:00Z
     data_window_start: 2026-04-01
     data_window_end: 2026-04-21
     adm1_version: "1.2"
     calibration_version: "0.2.0"
     fitted_against: [Q_gas, P_el]
     notes: "3-week window"

   kinetic:
     primary:
       k_dis: 0.4
       k_hyd_ch: 10.0

   initial_state:
     primary:
       _values: [0.012, 0.0053, ... ]  # 37 ADM1 entries
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import yaml

if TYPE_CHECKING:
    from pyadm1 import BiogasPlant  # type: ignore[import-not-found]


#: Schema version understood by this runtime. Bump on any breaking change
#: to the YAML structure (renamed sections, semantic shifts, …).
SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CalibrationMetadata:
    """Provenance information for a calibration run.

    Provenance-Information eines Calibration-Runs.

    Identifies which plant was calibrated, against which data window, by
    which software version. Estimation should refuse to apply an artifact
    whose ``plant_id`` does not match the runtime plant.

    Attributes:
        plant_id: Stable identifier of the physical plant.
        calibration_run_id: Identifier of the optimization run that
            produced this artifact. Should be unique per run.
        timestamp: ISO8601 timestamp when the artifact was written.
        data_window_start: Start of the historical data window used in
            calibration (ISO date).
        data_window_end: End of the historical data window (ISO date).
        adm1_version: Version of the ADM1 model these parameters target.
            Useful for detecting incompatible upgrades.
        calibration_version: Version of the calibration software that
            produced the artifact.
        fitted_against: Names of the measurement channels the optimizer
            actually used. Informational only.
        notes: Free-form operator notes (e.g. ``"BHKW1 offline that
            week"``).
    """

    plant_id: str
    calibration_run_id: str
    timestamp: str = ""
    data_window_start: Optional[str] = None
    data_window_end: Optional[str] = None
    adm1_version: str = ""
    calibration_version: str = ""
    fitted_against: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class CalibrationArtifact:
    """Calibrated simulation-model parameters, ready to apply to a plant.

    Kalibrierte Simulationsmodell-Parameter, bereit zum Anwenden auf eine
    Anlage.

    Sections / Sektionen:

    * ``metadata`` — provenance, see :class:`CalibrationMetadata`.
    * ``kinetic`` — per-digester overrides for ``adm1._kinetic``.
      Outer key: digester component id. Inner dict: kinetic parameter
      name → value (units defined by pyadm1).
    * ``substrates`` — per-substrate composition / disintegration
      fractions. Outer key: substrate name as known to the plant builder.
      Inner dict: fraction name → value (e.g. ``f_ch_xc``).
    * ``initial_state`` — per-digester ADM1 state at the end of the
      calibration window. Use as prior mean for the filter's initial
      ``x_hat``. Inner dict accepts either named entries
      (``{"S_su": 0.012, ...}``) or the array shortcut
      ``{"_values": [...]}`` with 37 entries in canonical ADM1 order.
    * ``residuals`` — per-channel fit quality (RMSE, MAE, …).
      Informational only; not consumed by Estimation.
    * ``schema_version`` — must equal :data:`SCHEMA_VERSION` to be
      readable by this runtime.
    """

    metadata: CalibrationMetadata
    kinetic: Dict[str, Dict[str, float]] = field(default_factory=dict)
    substrates: Dict[str, Dict[str, float]] = field(default_factory=dict)
    initial_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    residuals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_artifact(artifact: CalibrationArtifact, path: Union[str, Path]) -> None:
    """Serialize an artifact to a YAML file.

    Schreibt das Artefakt als YAML-Datei.

    Args:
        artifact: The artifact to write.
        path: Destination file path. Parent directories are created if
            missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_yaml_dict(artifact)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            payload,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def load_artifact(path: Union[str, Path]) -> CalibrationArtifact:
    """Load an artifact from a YAML file.

    Lädt ein Artefakt aus einer YAML-Datei.

    Validates that ``schema_version`` matches this runtime. Unknown
    versions raise ``ValueError`` so the operator is forced to upgrade
    rather than silently mis-reading the file.

    Args:
        path: Source file path.

    Returns:
        The deserialized artifact.

    Raises:
        ValueError: If the YAML is not a mapping or carries an unknown
            ``schema_version``.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return _from_yaml_dict(payload)


def _to_yaml_dict(artifact: CalibrationArtifact) -> Dict[str, Any]:
    """Convert an artifact into a YAML-friendly nested dict."""
    return {
        "schema_version": artifact.schema_version,
        "metadata": asdict(artifact.metadata),
        "kinetic": _to_plain(artifact.kinetic),
        "substrates": _to_plain(artifact.substrates),
        "initial_state": _to_plain(artifact.initial_state),
        "residuals": _to_plain(artifact.residuals),
    }


def _from_yaml_dict(payload: Any) -> CalibrationArtifact:
    """Convert a YAML-parsed dict into a :class:`CalibrationArtifact`.

    Raises:
        ValueError: For malformed payloads or unknown schema versions.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            "Calibration artifact YAML must be a mapping at the top "
            f"level; got {type(payload).__name__}."
        )

    schema_version = int(payload.get("schema_version", 0))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported calibration artifact schema_version="
            f"{schema_version}; this runtime expects "
            f"version {SCHEMA_VERSION}."
        )

    meta_payload = payload.get("metadata", {}) or {}
    if not isinstance(meta_payload, dict):
        raise ValueError(
            "'metadata' section must be a mapping; got "
            f"{type(meta_payload).__name__}."
        )

    # Required identification fields. Without these, we can't safely match
    # the artifact to a runtime plant.
    if "plant_id" not in meta_payload:
        raise ValueError("metadata.plant_id is required.")
    if "calibration_run_id" not in meta_payload:
        raise ValueError("metadata.calibration_run_id is required.")

    metadata = CalibrationMetadata(**meta_payload)
    return CalibrationArtifact(
        metadata=metadata,
        kinetic=_dict_section(payload, "kinetic"),
        substrates=_dict_section(payload, "substrates"),
        initial_state=_dict_section(payload, "initial_state"),
        residuals=_dict_section(payload, "residuals"),
        schema_version=schema_version,
    )


def _dict_section(payload: Dict[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    val = payload.get(key)
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise ValueError(
            f"'{key}' section must be a mapping; got {type(val).__name__}."
        )
    return {str(k): dict(v) if isinstance(v, dict) else v for k, v in val.items()}


def _to_plain(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively cast dict values to plain Python for clean YAML output."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[str(k)] = _to_plain(v)
        elif isinstance(v, (list, tuple)):
            out[str(k)] = [float(x) if hasattr(x, "item") else x for x in v]
        else:
            out[str(k)] = float(v) if hasattr(v, "item") else v
    return out


# ---------------------------------------------------------------------------
# Apply to plant
# ---------------------------------------------------------------------------


def apply_to_plant(
    artifact: CalibrationArtifact,
    plant: "BiogasPlant",
    *,
    strict: bool = False,
) -> Dict[str, List[str]]:
    """Overlay the artifact's parameters onto a fresh :class:`BiogasPlant`.

    Trägt die kalibrierten Parameter auf eine bereits gebaute Anlage auf.

    The plant must have its topology built (components present) before
    this is called. ``apply_to_plant`` only writes calibrated values into
    existing slots; it does not create new components or substrate
    definitions.

    Mismatches (unknown digester id, unknown kinetic key, …) trigger a
    warning in non-strict mode so an older artifact can still be applied
    to an evolved topology with graceful degradation. In strict mode the
    same mismatches raise ``KeyError`` — recommended for production
    startup, when you want to fail fast.

    Args:
        artifact: The calibration artifact to apply.
        plant: A fully built ``pyadm1.BiogasPlant``.
        strict: When ``True``, raise on any mismatch; when ``False``
            (default), emit warnings and skip.

    Returns:
        A diagnostics dict with keys ``"applied"`` (parameters actually
        written) and ``"skipped"`` (mismatches encountered).
    """
    applied: List[str] = []
    skipped: List[str] = []

    _apply_kinetic(artifact, plant, applied, skipped, strict)
    _apply_initial_state(artifact, plant, applied, skipped, strict)
    _apply_substrates(artifact, plant, applied, skipped, strict)

    return {"applied": applied, "skipped": skipped}


def _apply_kinetic(
    artifact: CalibrationArtifact,
    plant: "BiogasPlant",
    applied: List[str],
    skipped: List[str],
    strict: bool,
) -> None:
    for digester_id, overrides in artifact.kinetic.items():
        comp = plant.components.get(digester_id)
        if comp is None:
            _miss(skipped, strict, f"kinetic: unknown digester '{digester_id}'")
            continue
        adm1 = getattr(comp, "adm1", None)
        kinetic = getattr(adm1, "_kinetic", None) if adm1 is not None else None
        if kinetic is None:
            _miss(
                skipped,
                strict,
                f"kinetic: digester '{digester_id}' has no adm1._kinetic dict",
            )
            continue
        for key, value in overrides.items():
            if key not in kinetic:
                _miss(
                    skipped,
                    strict,
                    f"kinetic: unknown key '{key}' for digester '{digester_id}'",
                )
                continue
            kinetic[key] = float(value)
            applied.append(f"kinetic.{digester_id}.{key}")


def _apply_initial_state(
    artifact: CalibrationArtifact,
    plant: "BiogasPlant",
    applied: List[str],
    skipped: List[str],
    strict: bool,
) -> None:
    for digester_id, state in artifact.initial_state.items():
        comp = plant.components.get(digester_id)
        if comp is None:
            _miss(
                skipped,
                strict,
                f"initial_state: unknown digester '{digester_id}'",
            )
            continue
        if not isinstance(state, dict):
            _miss(
                skipped,
                strict,
                f"initial_state for '{digester_id}' must be a mapping",
            )
            continue
        current = list(getattr(comp, "adm1_state", []))
        if not current:
            _miss(
                skipped,
                strict,
                f"initial_state: digester '{digester_id}' has no adm1_state",
            )
            continue
        # Array shortcut: ``_values: [...]`` overrides the whole vector.
        if "_values" in state:
            values = list(state["_values"])
            if len(values) != len(current):
                _miss(
                    skipped,
                    strict,
                    f"initial_state: digester '{digester_id}' expects "
                    f"{len(current)} values, got {len(values)}",
                )
                continue
            comp.adm1_state = [float(v) for v in values]
            applied.append(f"initial_state.{digester_id}._values")
        else:
            # Named entries would require an index map from pyadm1.
            # Until that is added here, accept only the array form so
            # operators get a clear error rather than a half-applied
            # state.
            _miss(
                skipped,
                strict,
                f"initial_state for '{digester_id}': named entries are not "
                f"yet supported, use the '_values' array form",
            )


def _apply_substrates(
    artifact: CalibrationArtifact,
    plant: "BiogasPlant",
    applied: List[str],
    skipped: List[str],
    strict: bool,
) -> None:
    # Substrate-fraction override is plant-builder specific. Until a
    # uniform substrate API is settled, surface the data as a single
    # diagnostic line per substrate and leave the actual mutation to the
    # plant builder (which can read ``artifact.substrates`` directly when
    # constructing the BiogasPlant).
    for substrate_name in artifact.substrates:
        skipped.append(
            f"substrates: '{substrate_name}' present in artifact but "
            f"substrate overlay is not implemented in apply_to_plant "
            f"(plant builder reads artifact.substrates directly)"
        )


def _miss(skipped: List[str], strict: bool, msg: str) -> None:
    if strict:
        raise KeyError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    skipped.append(msg)
