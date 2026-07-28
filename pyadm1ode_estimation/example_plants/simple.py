"""Simple single-fermenter example plant.

This is a example of a small mass-balance-closed plant:
one fermenter (with substrate feed and gas headspace), one
digestate-storage tank downstream (no substrate feed, lower
temperature), and one CHP that consumes the biogas from the
fermenter.

Topology::

    [Fermenter 1000 m³ liquid, 150 m³ gas, 38 °C]
            │
            │   liquid digestate (cascade)
            ▼
    [Digestate Storage 1500 m³ liquid, 200 m³ gas, 30 °C]

    [Fermenter biogas] ───► [CHP 100 kW]

The substrate mix is a small two-component co-digestion typical for
agricultural AD: maize silage + cattle slurry. Nominal feed rate
15 m³/d.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyadm1 import BiogasPlant  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Topology constants — small but realistic agricultural AD
# ---------------------------------------------------------------------------

# (component_id, name, V_liq [m³], V_gas [m³], T_ad [K])
_FERMENTER = ("fermenter", "Fermenter", 1000.0, 150.0, 311.15)  # 38 °C
_STORAGE = ("storage", "Digestate Storage", 1500.0, 200.0, 303.15)  # 30 °C

# (chp_id, name, P_el_nom [kW], eta_el, eta_th)
_CHP = ("chp", "CHP 100 kW", 100.0, 0.40, 0.45)

# Substrate mix: 10 m³/d maize silage + 5 m³/d cattle slurry = 15 m³/d total.
# These IDs must resolve against the PyADM1ODE substrate library.
_SUBSTRATES = {
    "maize_silage_milk_ripeness": 10.0,
    "cattle_manure": 5.0,
}

# Feedstock supports up to 10 slots. Unused slots are zero.
_MAX_SUBSTRATE_SLOTS = 10


def build_simple_plant(
    *,
    feeding_freq: int = 24,
    total_simtime: int = 365,
    plant_name: str = "Simple Single-Fermenter Plant",
) -> BiogasPlant:
    """Construct the simple single-fermenter example plant.

    Args:
        feeding_freq: Number of feeding events per simulated day.
            Forwarded to :class:`pyadm1.Feedstock`.
        total_simtime: Total simulation horizon in days. Forwarded to
            :class:`pyadm1.Feedstock`.
        plant_name: Human-readable plant identifier.

    Returns:
        An initialized :class:`pyadm1.BiogasPlant` ready for
        ``plant.simulate()`` or estimator use via
        :class:`pyadm1ode_estimation.estimation.ADM1ProcessModel`.

    Raises:
        ImportError: If the ``pyadm1`` base package is not installed.
    """
    from pyadm1 import BiogasPlant, Feedstock  # type: ignore[import-not-found]
    from pyadm1.configurator.plant_configurator import (  # type: ignore[import-not-found]
        PlantConfigurator,
    )

    substrate_ids = list(_SUBSTRATES.keys())
    q_fermenter = list(_SUBSTRATES.values())
    # Pad to 10 slots — Feedstock always uses fixed-width Q vectors.
    q_fermenter = q_fermenter + [0.0] * (_MAX_SUBSTRATE_SLOTS - len(q_fermenter))
    q_passthrough = [0.0] * _MAX_SUBSTRATE_SLOTS

    feedstock = Feedstock(
        substrate_ids,
        feeding_freq=feeding_freq,
        total_simtime=total_simtime,
    )

    plant = BiogasPlant(plant_name)
    cfg = PlantConfigurator(plant, feedstock)

    # Fermenter — main reactor with substrate feed.
    fid, fname, fv_l, fv_g, ft = _FERMENTER
    cfg.add_digester(
        digester_id=fid,
        V_liq=fv_l,
        V_gas=fv_g,
        T_ad=ft,
        name=fname,
        Q_substrates=q_fermenter,
    )

    # Storage — passthrough digester, no substrate feed.
    sid, sname, sv_l, sv_g, st = _STORAGE
    cfg.add_digester(
        digester_id=sid,
        V_liq=sv_l,
        V_gas=sv_g,
        T_ad=st,
        name=sname,
        Q_substrates=q_passthrough,
    )

    # CHP — consumes biogas from the fermenter.
    cid, cname, p_el, eta_el, eta_th = _CHP
    cfg.add_chp(
        chp_id=cid,
        P_el_nom=p_el,
        eta_el=eta_el,
        eta_th=eta_th,
        name=cname,
    )

    # Connections: liquid cascade fermenter → storage; biogas
    # fermenter → CHP. No heating circuit at this level of complexity.
    cfg.connect(fid, sid, "liquid")
    cfg.auto_connect_digester_to_chp(fid, cid)

    plant.initialize()
    return plant
