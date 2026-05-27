"""Multi-stage cascade example plant.

A realistic agricultural anaerobic-digestion topology: three liquid
stages in cascade, two parallel CHP units, and one heating circuit
per stage. The biogas from the primary fermenter feeds both CHPs.
CHP waste heat is distributed across all three heating circuits.

Topology::

    [Primary Fermenter 1200 m³ liquid, 216 m³ gas, 42 °C]
            │
            │   liquid cascade
            ▼
    [Secondary Fermenter 1200 m³ liquid, 216 m³ gas, 42 °C]
            │
            │   liquid cascade
            ▼
    [Digestate Storage 3500 m³ liquid, 476 m³ gas, 35 °C]

    [Primary biogas] ───► [CHP 1 (250 kW)] ──┐
                     ───► [CHP 2 (250 kW)] ──┤
                                             │
                             waste heat ─────┴───► [Heating ×3]

The substrate mix is a four-component co-digestion typical for
mid-sized agricultural plants. Energy-crop silage, two manure
streams (solid + liquid), and cereal whole-plant silage. Nominal
total dosing rate 60 m³/d.

This builder is intended as a realistic but generic example.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from pyadm1 import BiogasPlant  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Topology constants
# ---------------------------------------------------------------------------

#: Three-stage liquid cascade.
#: (component_id, name, V_liq [m³], V_gas [m³], T_ad [K])
_STAGES: List[Tuple[str, str, float, float, float]] = [
    ("primary", "Primary Fermenter", 1200.0, 216.0, 315.15),  # 42 °C
    ("secondary", "Secondary Fermenter", 1200.0, 216.0, 315.15),  # 42 °C
    ("storage", "Digestate Storage", 3500.0, 476.0, 308.15),  # 35 °C
]

#: Two identical CHP units, both fed from the primary's biogas.
#: (chp_id, name)
_CHPS: List[Tuple[str, str]] = [
    ("chp1", "CHP 1 (250 kW)"),
    ("chp2", "CHP 2 (250 kW)"),
]
_CHP_P_EL_NOM = 250.0  # kW
_CHP_ETA_EL = 0.40
_CHP_ETA_TH = 0.45

#: Substrate mix [fraction] — four-component co-digestion. Names must
#: resolve against the PyADM1ODE substrate library. Fractions sum
#: to 1.0; the absolute feed rate is set by _NOMINAL_DAILY_DOSING_M3.
_SUBSTRATE_MIX = {
    "maize_silage_milk_ripeness": 0.67,  # energy-crop silage
    "cattle_manure": 0.32,  # liquid slurry
    "cereal_gps_silage": 0.01,  # cereal whole-plant silage
}

#: Operator-typical nominal daily total dosing [m³/d].
_NOMINAL_DAILY_DOSING_M3 = 60.0

#: Feedstock substrate slots. Unused slots are zero.
_MAX_SUBSTRATE_SLOTS = 10


def build_multi_stage_plant(
    *,
    feeding_freq: int = 24,
    total_simtime: int = 365,
    nominal_daily_dosing_m3: float = _NOMINAL_DAILY_DOSING_M3,
    plant_name: str = "Multi-Stage Cascade Plant",
) -> "BiogasPlant":
    """Construct the three-stage cascade example plant.

    Each fermenter is seeded with a pre-inoculated ADM1 state computed
    from the nominal substrate feed — this avoids the washout-prone
    default ``[0.01] * 41`` initial state and lets the plant start
    from a healthy operating point.

    Args:
        feeding_freq: Number of feeding events per simulated day.
        total_simtime: Total simulation horizon in days.
        nominal_daily_dosing_m3: Total substrate volume per day [m³/d]
            used to seed the inoculated initial state. Defaults to a
            mid-sized agricultural baseline.
        plant_name: Human-readable plant identifier.

    Returns:
        An initialized :class:`pyadm1.BiogasPlant` ready for
        ``plant.simulate()`` or estimator use via
        :class:`pyadm1ode_estimation.estimation.ADM1ProcessModel`.

    Raises:
        ImportError: If the ``pyadm1`` base package is not installed.
    """
    from pyadm1 import BiogasPlant, Feedstock  # type: ignore[import-not-found]
    from pyadm1.components.biological.digester import Digester  # type: ignore[import-not-found]
    from pyadm1.configurator.plant_configurator import (  # type: ignore[import-not-found]
        PlantConfigurator,
    )

    substrate_ids = list(_SUBSTRATE_MIX.keys())
    feedstock = Feedstock(
        substrate_ids,
        feeding_freq=feeding_freq,
        total_simtime=total_simtime,
    )

    # Nominal feed vector for the primary stage, padded to 10 slots.
    q_primary = [
        fraction * nominal_daily_dosing_m3 for fraction in _SUBSTRATE_MIX.values()
    ]
    q_primary = q_primary + [0.0] * (_MAX_SUBSTRATE_SLOTS - len(q_primary))
    q_passthrough = [0.0] * _MAX_SUBSTRATE_SLOTS

    # Build a shared inoculum state from a throwaway primary-sized
    # digester — same pattern as PyADM1ODE's realistic-plant example.
    _, _, vl_primary, vg_primary, t_primary = _STAGES[0]
    _inoc = Digester(
        component_id="_inoc",
        feedstock=feedstock,
        V_liq=vl_primary,
        V_gas=vg_primary,
        T_ad=t_primary,
    )
    inoculum_state = _inoc._build_pre_inoculated_state(q_primary)

    plant = BiogasPlant(plant_name)
    cfg = PlantConfigurator(plant, feedstock)

    # Stages — primary gets the actual feed,
    # secondary + storage are passthrough.
    for stage_id, name, v_liq, v_gas, t_ad in _STAGES:
        cfg.add_digester(
            digester_id=stage_id,
            V_liq=v_liq,
            V_gas=v_gas,
            T_ad=t_ad,
            name=name,
            Q_substrates=q_primary if stage_id == "primary" else q_passthrough,
            adm1_state=list(inoculum_state),
        )

    # CHPs — both 250 kW, identical.
    for chp_id, name in _CHPS:
        cfg.add_chp(
            chp_id,
            P_el_nom=_CHP_P_EL_NOM,
            eta_el=_CHP_ETA_EL,
            eta_th=_CHP_ETA_TH,
            name=name,
        )

    # One heating circuit per stage so the plant tracks heat demand
    # and auxiliary heating per fermenter.
    for stage_id, _name, _vl, _vg, t_ad in _STAGES:
        cfg.add_heating(
            f"heating_{stage_id}",
            target_temperature=t_ad,
            name=f"Heating {stage_id}",
        )

    # Liquid cascade: primary → secondary → storage.
    cfg.connect("primary", "secondary", "liquid")
    cfg.connect("secondary", "storage", "liquid")

    # Gas routing: both CHPs draw from the primary's biogas
    # only (post-fermenter and storage gas headspaces are not piped
    # to the engines in this topology). CHP waste heat is
    # distributed to every heating circuit.
    for chp_id, _ in _CHPS:
        cfg.auto_connect_digester_to_chp("primary", chp_id)
        for stage_id, *_rest in _STAGES:
            cfg.auto_connect_chp_to_heating(chp_id, f"heating_{stage_id}")

    plant.initialize()
    return plant
