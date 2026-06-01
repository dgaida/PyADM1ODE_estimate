"""Example sensor definitions for the multi-stage plant.

Demonstrates how to write a realistic :class:`SensorChannelDef`.
The names follow the loose ISA-5.1 convention used in many plants:

* ``FIC`` / ``FIT`` - flow indicator (controller / transmitter)
* ``AIT``           - analytical indicator transmitter (pH, CH₄ %)
* ``WIT``           - weighing indicator transmitter (dosing scale)
* ``TIT``           - temperature indicator transmitter
* ``101`` / ``201`` - loop number (1xx = gas line, 2xx = liquid
  analytical, 3xx / 4xx = substrate dosing)

The mapping connects DB column names to the UKF channels produced by
:func:`pyadm1ode_estimation.estimation.build_ukf`. It meters **every**
substrate input of the example plant - the four solids over the
weighing dosing belt, the slurry over a flow meter.

SCADA layout this targets::

    [Primary Fermenter]
        ├─ Biogas line
        │     ├─ FIC101 - total biogas flow [Nm³/h]
        │     └─ AIT101 - CH₄ %                            (dropped here:
        │                                                   Q_ch4 comes
        │                                                   from FIC102
        │                                                   in this
        │                                                   layout)
        ├─ FIC102 - methane flow [Nm³/h] (derived in
        │           a DB view from FIC101 × AIT101 / 100)
        └─ AIT201 - pH                            [-]

    [Substrate dosing]
        ├─ WIT301 - maize-silage scale            [kg/h]
        ├─ WIT303 - farmyard-manure scale         [kg/h]
        ├─ WIT304 - chicken-litter (HTK) scale    [kg/h]
        ├─ WIT302 - cereal-grain scale            [kg/h]
        └─ FIC401 - cattle-slurry flow            [m³/h]

The example also shows the four data-side concerns a real adapter has
to handle, one per channel kind:

1. **Plain unit conversion** via the built-in table (``Nm3/h → m3/d``).
2. **Custom converter** for mass-to-volume requiring a density
   (``kg/h → m³/d`` for solid silages).
3. **Range validation** (pH ∈ [3, 11], Q_gas ∈ [0, 50 000]).
4. **Vendor-specific quality flags** (``OPC-UA`` ``Good`` / ``Bad`` /
   ``Uncertain`` strings, plus boolean / integer status codes).

The returned list is consumed by ``DataFrameSensorSource`` (or any
other :class:`pyadm1ode_estimation.io.SensorSource` adapter) and feeds
directly into ``ukf.update(y, t=t)``.

Typical usage::

    from pyadm1ode_estimation.example_plants import (
        build_multi_stage_plant,
        build_example_sensor_defs,
    )
    from pyadm1ode_estimation.estimation import InputSpec, build_ukf
    from pyadm1ode_estimation.io import DataFrameSensorSource

    plant = build_multi_stage_plant()
    ukf = build_ukf(
        plant,
        digester_id="primary",
        substrates=[
            InputSpec("maize_silage",   substrate_index=0, initial_flow=4.74),
            InputSpec("solid_manure",   substrate_index=1, initial_flow=13.70),
            InputSpec("chicken_litter", substrate_index=2, initial_flow=1.09),
            InputSpec("slurry",         substrate_index=3, initial_flow=3.68),
            InputSpec("cereal_grain",   substrate_index=4, initial_flow=0.20),
        ],
    )

    defs = build_example_sensor_defs()
    source = DataFrameSensorSource(scada_df, defs)

    prev_t = None
    for t, y in source.stream():
        dt = (t - prev_t) if prev_t is not None else 1.0 / 24.0
        ukf.predict(dt=dt)
        ukf.update(y, t=t)
        prev_t = t
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..io import SensorChannelDef

# ---------------------------------------------------------------------------
# Plant-side physical constants
# ---------------------------------------------------------------------------
#
# Substrate bulk densities, used by the kg/h → m³/d converters on the
# solid-feed dosing scales. Typical values for chopped, pressed
# agricultural silage.

#: Maize-silage bulk density [kg/m³].
_RHO_MAIZE_KG_M3 = 700.0

#: Crushed cereal-grain bulk density [kg/m³].
_RHO_CEREAL_KG_M3 = 600.0

#: Farmyard (solid) manure bulk density [kg/m³].
_RHO_SOLID_MANURE_KG_M3 = 1000.0

#: Dry chicken-litter (HTK) bulk density [kg/m³].
_RHO_CHICKEN_LITTER_KG_M3 = 750.0


# ---------------------------------------------------------------------------
# Valid-range bounds (in model units, i.e. AFTER conversion)
# ---------------------------------------------------------------------------
#
# Generous bounds that drop only impossible readings. For example sensor faults,
# wraparounds, frozen outputs, without clipping legitimate operating
# excursions.

#: pH [-] - below 3 or above 11 means probe failure / coating, not a
#: real fermenter state. Tighten if you trust the probe more.
_PH_RANGE: Tuple[float, float] = (3.0, 11.0)

#: Total biogas flow [m³/d] at full multi-stage scale. Upper bound
#: 50 000 covers a 1 MW-class plant, lower bound 0 catches negative
#: readings from a stalled meter.
_Q_GAS_RANGE: Tuple[float, float] = (0.0, 50_000.0)

#: Methane flow [m³/d] - bounded by Q_gas × 75 % (theoretical max
#: CH₄ content). 30 000 is generous.
_Q_CH4_RANGE: Tuple[float, float] = (0.0, 30_000.0)

#: Substrate-dose volumetric flow [m³/d] at the primary stage.
#: 200 covers transient pulse-feeding peaks well above the ~23 m³/d
#: nominal total dosing of the multi-stage plant.
_Q_SUBSTRATE_RANGE: Tuple[float, float] = (0.0, 200.0)


# ---------------------------------------------------------------------------
# Vendor-specific quality codes
# ---------------------------------------------------------------------------
#
# A realistic SCADA mix: OPC-UA classifier names, PLC boolean
# diagnostic bits, and PLC integer status codes. ``None`` covers the
# common case of a NULL row in the DB export. The set is broader than
# the dataclass default so it actually catches real plant exports.

_BAD_STATUS: Tuple = (
    # OPC-UA "Bad" / "Uncertain" enumeration strings
    "Bad",
    "Uncertain",
    "BAD",
    "UNCERTAIN",
    # Boolean diagnostic bits - False / 0 = sensor fault on most PLCs
    False,
    0,
    # Common integer status codes meaning "no signal" / "out of service"
    -1,
    999,
    9999,
    # NULL rows
    None,
)


# ---------------------------------------------------------------------------
# SCADA → UKF channel mapping
# ---------------------------------------------------------------------------


def build_example_sensor_defs(
    *,
    rho_maize_kg_m3: float = _RHO_MAIZE_KG_M3,
    rho_cereal_kg_m3: float = _RHO_CEREAL_KG_M3,
    rho_solid_manure_kg_m3: float = _RHO_SOLID_MANURE_KG_M3,
    rho_chicken_litter_kg_m3: float = _RHO_CHICKEN_LITTER_KG_M3,
) -> List[SensorChannelDef]:
    """Construct the example sensor-channel mapping.

    The returned list mirrors a realistic SCADA tag set for the
    multi-stage plant and meters **every** substrate input: the four
    solids run over the weighing dosing belt (``WIT3xx``) and the slurry
    over a flow meter (``FIC401``). Substrate UKF-channel names follow
    the convention of :func:`pyadm1ode_estimation.estimation.build_ukf`,
    which prefixes each ``InputSpec.name`` with ``"Q_"`` - so an
    ``InputSpec("maize_silage", ...)`` is observed via the
    ``"Q_maize_silage"`` UKF channel.

    Args:
        rho_maize_kg_m3: Density of maize silage [kg/m³], used by the
            ``kg/h → m³/d`` converter on ``WIT301``. Defaults to
            700 kg/m³ (typical pressed silage).
        rho_cereal_kg_m3: Density of crushed cereal grain [kg/m³], used
            by ``WIT302``. Defaults to 600 kg/m³.
        rho_solid_manure_kg_m3: Density of farmyard manure [kg/m³], used
            by ``WIT303``. Defaults to 1000 kg/m³.
        rho_chicken_litter_kg_m3: Density of dry chicken litter [kg/m³],
            used by ``WIT304``. Defaults to 750 kg/m³.

    Returns:
        List of eight :class:`SensorChannelDef` covering gas, methane, pH
        and all five substrate-dose channels. Use this list with
        :class:`pyadm1ode_estimation.io.DataFrameSensorSource` (or any
        other adapter) to feed ``ukf.update``.
    """
    return [
        # ---- Gas line ------------------------------------------------
        # FIC101: vortex / thermal mass-flow meter on the biogas
        # header. Output Nm³/h, model wants m³/d.
        SensorChannelDef(
            db_column="fic101_q_gas_nm3h",
            ukf_channel="Q_gas",
            unit_in="Nm3/h",
            unit_out="m3/d",
            valid_range=_Q_GAS_RANGE,
            quality_column="fic101_quality",
            bad_status_values=_BAD_STATUS,
        ),
        # FIC102: methane partial-flow channel, typically derived in a
        # DB view from FIC101 * AIT101/100 (NDIR CH4 sensor). Treated
        # here as a first-class measured tag - the DB view is
        # responsible for the multiplication.
        SensorChannelDef(
            db_column="fic102_q_ch4_nm3h",
            ukf_channel="Q_ch4",
            unit_in="Nm3/h",
            unit_out="m3/d",
            valid_range=_Q_CH4_RANGE,
            quality_column="fic102_quality",
            bad_status_values=_BAD_STATUS,
        ),
        # ---- Liquid analytical --------------------------------------
        # AIT201: pH probe in the primary fermenter. Dimensionless;
        # only range + quality matter.
        SensorChannelDef(
            db_column="ait201_ph",
            ukf_channel="pH",
            valid_range=_PH_RANGE,
            quality_column="ait201_quality",
            bad_status_values=_BAD_STATUS,
        ),
        # ---- Substrate dosing ---------------------------------------
        # WIT301: weighing belt for maize silage [kg/h]. Mass-to-volume
        # needs density, which isn't in the built-in unit table -> the
        # converter handles it.
        SensorChannelDef(
            db_column="wit301_maize_kg_h",
            ukf_channel="Q_maize_silage",
            unit_in="kg/h",
            unit_out="m3/d",
            converter=lambda kg_h, rho=rho_maize_kg_m3: kg_h * 24.0 / rho,
            valid_range=_Q_SUBSTRATE_RANGE,
            quality_column="wit301_quality",
            bad_status_values=_BAD_STATUS,
        ),
        # FIC401: magnetic-inductive flow meter on the slurry line
        # [m³/h]. Built-in table handles m³/h -> m³/d directly.
        SensorChannelDef(
            db_column="fic401_slurry_m3h",
            ukf_channel="Q_slurry",
            unit_in="m3/h",
            unit_out="m3/d",
            valid_range=_Q_SUBSTRATE_RANGE,
            quality_column="fic401_quality",
            bad_status_values=_BAD_STATUS,
        ),
        # WIT302: weighing belt for crushed cereal grain [kg/h]. Same
        # pattern as maize, different density.
        SensorChannelDef(
            db_column="wit302_cereal_kg_h",
            ukf_channel="Q_cereal_grain",
            unit_in="kg/h",
            unit_out="m3/d",
            converter=lambda kg_h, rho=rho_cereal_kg_m3: kg_h * 24.0 / rho,
            valid_range=_Q_SUBSTRATE_RANGE,
            quality_column="wit302_quality",
            bad_status_values=_BAD_STATUS,
        ),
        # WIT303: weighing belt for farmyard (solid) manure [kg/h].
        SensorChannelDef(
            db_column="wit303_manure_kg_h",
            ukf_channel="Q_solid_manure",
            unit_in="kg/h",
            unit_out="m3/d",
            converter=lambda kg_h, rho=rho_solid_manure_kg_m3: kg_h * 24.0 / rho,
            valid_range=_Q_SUBSTRATE_RANGE,
            quality_column="wit303_quality",
            bad_status_values=_BAD_STATUS,
        ),
        # WIT304: weighing belt for dry chicken litter / HTK [kg/h].
        SensorChannelDef(
            db_column="wit304_htk_kg_h",
            ukf_channel="Q_chicken_litter",
            unit_in="kg/h",
            unit_out="m3/d",
            converter=lambda kg_h, rho=rho_chicken_litter_kg_m3: kg_h * 24.0 / rho,
            valid_range=_Q_SUBSTRATE_RANGE,
            quality_column="wit304_quality",
            bad_status_values=_BAD_STATUS,
        ),
    ]


def example_scada_columns() -> Dict[str, str]:
    """Return the expected DB-column → description mapping.

    Useful when wiring the adapter to a live DB view - pass the dict
    keys as the ``SELECT`` list and use the values as a documentation
    crib sheet. The order matches :func:`build_example_sensor_defs`.
    """
    return {
        "fic101_q_gas_nm3h": "Total biogas flow on the gas header [Nm³/h]",
        "fic101_quality": "FIC101 OPC-UA quality flag",
        "fic102_q_ch4_nm3h": "Methane flow (DB-derived FIC101 * AIT101 / 100) [Nm³/h]",
        "fic102_quality": "FIC102 derived-channel quality flag",
        "ait201_ph": "pH of the primary fermenter [-]",
        "ait201_quality": "AIT201 OPC-UA quality flag",
        "wit301_maize_kg_h": "Maize-silage dosing belt scale [kg/h]",
        "wit301_quality": "WIT301 PLC diagnostic bit",
        "fic401_slurry_m3h": "Cattle-slurry volumetric flow [m³/h]",
        "fic401_quality": "FIC401 PLC diagnostic bit",
        "wit302_cereal_kg_h": "Cereal-grain dosing belt scale [kg/h]",
        "wit302_quality": "WIT302 PLC diagnostic bit",
        "wit303_manure_kg_h": "Farmyard-manure dosing belt scale [kg/h]",
        "wit303_quality": "WIT303 PLC diagnostic bit",
        "wit304_htk_kg_h": "Chicken-litter (HTK) dosing belt scale [kg/h]",
        "wit304_quality": "WIT304 PLC diagnostic bit",
    }


__all__ = ["build_example_sensor_defs", "example_scada_columns"]
