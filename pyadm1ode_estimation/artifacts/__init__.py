"""Artefakt-Schnittstellen für den Datenaustausch zwischen Repos.

Artifact interfaces for data exchange between PyADM1ODE repositories.

Aktuell verfügbar / currently provided:

* :class:`CalibrationArtifact` — kalibrierte ADM1-Parameter, geschrieben
  von ``pyadm1ode_calibration`` und gelesen von ``pyadm1ode_estimation``
  beim Hochfahren des Live-Filters.
"""

from .calibration_artifact import (
    SCHEMA_VERSION,
    CalibrationArtifact,
    CalibrationMetadata,
    apply_to_plant,
    load_artifact,
    save_artifact,
)

__all__ = [
    "SCHEMA_VERSION",
    "CalibrationArtifact",
    "CalibrationMetadata",
    "apply_to_plant",
    "load_artifact",
    "save_artifact",
]
