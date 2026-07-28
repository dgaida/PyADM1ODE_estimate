"""Unit tests for the calibration artifact (YAML handoff)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from pyadm1ode_estimation.artifacts import (
    SCHEMA_VERSION,
    CalibrationArtifact,
    CalibrationMetadata,
    apply_to_plant,
    load_artifact,
    save_artifact,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_artifact(**overrides) -> CalibrationArtifact:
    """Minimal valid artifact for use in tests."""
    metadata = CalibrationMetadata(
        plant_id="test_plant",
        calibration_run_id="test_run_001",
        timestamp="2026-05-14T12:30:00Z",
        fitted_against=["Q_gas"],
    )
    base = {
        "metadata": metadata,
        "kinetic": {"primary": {"k_dis": 0.4, "k_hyd_ch": 10.0}},
        "substrates": {"maize": {"f_ch_xc": 0.55}},
        "initial_state": {"primary": {"_values": [0.1, 0.2, 0.3]}},
        "residuals": {"Q_gas": {"rmse": 145.0}},
    }
    base.update(overrides)
    return CalibrationArtifact(**base)


class FakeAdm1:
    """Minimal stand-in for pyadm1's ADM1 with a mutable _kinetic dict."""

    def __init__(self, kinetic):
        self._kinetic = dict(kinetic)


class FakeDigester:
    """Minimal stand-in for a pyadm1 digester component."""

    def __init__(self, kinetic=None, adm1_state=None):
        self.adm1 = FakeAdm1(kinetic or {}) if kinetic is not None else None
        self.adm1_state = list(adm1_state) if adm1_state is not None else []


class FakePlant:
    """Minimal stand-in for pyadm1.BiogasPlant — only ``components``."""

    def __init__(self, components):
        self.components = dict(components)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_save_then_load_preserves_content(self, tmp_path: Path) -> None:
        artifact = make_artifact()
        path = tmp_path / "art.yaml"
        save_artifact(artifact, path)

        loaded = load_artifact(path)
        assert loaded.metadata.plant_id == "test_plant"
        assert loaded.metadata.calibration_run_id == "test_run_001"
        assert loaded.metadata.fitted_against == ["Q_gas"]
        assert loaded.kinetic["primary"]["k_dis"] == pytest.approx(0.4)
        assert loaded.substrates["maize"]["f_ch_xc"] == pytest.approx(0.55)
        assert loaded.initial_state["primary"]["_values"] == [0.1, 0.2, 0.3]
        assert loaded.residuals["Q_gas"]["rmse"] == pytest.approx(145.0)
        assert loaded.schema_version == SCHEMA_VERSION

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        artifact = make_artifact()
        nested = tmp_path / "deep" / "nested" / "art.yaml"
        save_artifact(artifact, nested)
        assert nested.is_file()

    def test_example_yaml_is_loadable(self) -> None:
        # Sanity check: the shipped example must parse.
        example = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "assets"
            / "calibration_artifact_example.yaml"
        )
        if not example.is_file():
            pytest.skip("Example YAML not present")
        artifact = load_artifact(example)
        assert artifact.metadata.plant_id == "plant_id_xyz"
        assert "primary" in artifact.kinetic
        # ADM1da has 41 state components, not 37.
        assert len(artifact.initial_state["primary"]["_values"]) == 41


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_plant_id_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "schema_version: 1\n" "metadata:\n" "  calibration_run_id: foo\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="plant_id is required"):
            load_artifact(path)

    def test_missing_run_id_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "schema_version: 1\n" "metadata:\n" "  plant_id: foo\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="calibration_run_id is required"):
            load_artifact(path)

    def test_unknown_schema_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "schema_version: 999\n"
            "metadata:\n"
            "  plant_id: foo\n"
            "  calibration_run_id: bar\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="schema_version=999"):
            load_artifact(path)

    def test_non_mapping_top_level_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_artifact(path)

    def test_kinetic_non_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "schema_version: 1\n"
            "metadata:\n"
            "  plant_id: foo\n"
            "  calibration_run_id: bar\n"
            "kinetic: 'not a dict'\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="'kinetic' section"):
            load_artifact(path)


# ---------------------------------------------------------------------------
# apply_to_plant
# ---------------------------------------------------------------------------


class TestApplyToPlant:
    def test_applies_kinetic_overrides(self) -> None:
        plant = FakePlant(
            {
                "primary": FakeDigester(
                    kinetic={"k_dis": 0.3, "k_hyd_ch": 8.0},
                    adm1_state=[0.0] * 37,
                ),
            }
        )
        artifact = make_artifact(
            kinetic={"primary": {"k_dis": 0.45, "k_hyd_ch": 10.5}},
            initial_state={},
            substrates={},
        )

        diag = apply_to_plant(artifact, plant)

        assert plant.components["primary"].adm1._kinetic["k_dis"] == 0.45
        assert plant.components["primary"].adm1._kinetic["k_hyd_ch"] == 10.5
        assert "kinetic.primary.k_dis" in diag["applied"]

    def test_applies_initial_state_values(self) -> None:
        plant = FakePlant(
            {
                "primary": FakeDigester(
                    kinetic={"k_dis": 0.3},
                    adm1_state=[0.0] * 5,
                ),
            }
        )
        artifact = make_artifact(
            kinetic={},
            initial_state={"primary": {"_values": [1.0, 2.0, 3.0, 4.0, 5.0]}},
            substrates={},
        )

        diag = apply_to_plant(artifact, plant)

        assert plant.components["primary"].adm1_state == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert "initial_state.primary._values" in diag["applied"]

    def test_unknown_digester_warns_in_non_strict_mode(self) -> None:
        plant = FakePlant({})
        artifact = make_artifact(
            kinetic={"ghost": {"k_dis": 0.4}},
            initial_state={},
            substrates={},
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diag = apply_to_plant(artifact, plant)

        assert any("unknown digester 'ghost'" in str(w.message) for w in caught)
        assert diag["applied"] == []
        assert len(diag["skipped"]) == 1

    def test_unknown_digester_raises_in_strict_mode(self) -> None:
        plant = FakePlant({})
        artifact = make_artifact(
            kinetic={"ghost": {"k_dis": 0.4}},
            initial_state={},
            substrates={},
        )
        with pytest.raises(KeyError, match="ghost"):
            apply_to_plant(artifact, plant, strict=True)

    def test_unknown_kinetic_key_warns(self) -> None:
        plant = FakePlant(
            {
                "primary": FakeDigester(kinetic={"k_dis": 0.3}, adm1_state=[0.0] * 5),
            }
        )
        artifact = make_artifact(
            kinetic={"primary": {"k_dis": 0.4, "k_unknown_xxx": 99.0}},
            initial_state={},
            substrates={},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diag = apply_to_plant(artifact, plant)
        # Known key applied, unknown key skipped.
        assert plant.components["primary"].adm1._kinetic["k_dis"] == 0.4
        assert "k_unknown_xxx" not in plant.components["primary"].adm1._kinetic
        assert any("unknown key 'k_unknown_xxx'" in str(w.message) for w in caught)
        assert any("k_unknown_xxx" in s for s in diag["skipped"])

    def test_initial_state_length_mismatch_skipped(self) -> None:
        plant = FakePlant(
            {
                "primary": FakeDigester(kinetic={}, adm1_state=[0.0] * 37),
            }
        )
        artifact = make_artifact(
            kinetic={},
            initial_state={"primary": {"_values": [1.0, 2.0]}},
            substrates={},
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            diag = apply_to_plant(artifact, plant)
        # Original state untouched on length mismatch.
        assert plant.components["primary"].adm1_state == [0.0] * 37
        assert any("expects 37 values" in s for s in diag["skipped"])
