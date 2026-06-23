"""Literature-grounded, plant-agnostic realistic test conditions.

This module defines the **measurement noise** and **model error** used to
evaluate the state estimator under conditions close to a real agricultural
biogas plant — deliberately WITHOUT fitting to any specific (and currently
incomplete) plant dataset. The values attach to *sensor types* and to *generic
ADM1 kinetic uncertainty*, so the benchmark stays valid as sensors are added
to the real plant and transfers across different biogas plants (BGAs).

Why not derive these from the current plant data?
-------------------------------------------------
* The sensor suite is not final (sensors are still being added), so fitting
  noise to today's signals would bias the conditions toward missing data.
* The *model error* — the gap between ADM1 and reality — cannot be identified
  from operating data alone (it is confounded with input/measurement error).
  The honest, transferable choice is the **kinetic-parameter uncertainty**
  reported in the ADM1 literature, applied as a multiplicative perturbation.

Literature basis (see ``docs/.../development/realistic_conditions.md``)
----------------------------------------------------------------------
Model error (parameter uncertainty):
  * ADM1 local/global sensitivity analyses perturb the kinetic/stoichiometric
    parameters by ~10 %; the gas-relevant sensitive ones are the
    decay/disintegration/hydrolysis constants, the Monod maximum uptake rate
    ``k_m`` and the half-saturation ``K_S``. (WIT/ScienceDirect SA studies.)
  * UKF-for-AD studies inject a deliberate *plant-model mismatch* of ~28-30 %
    on the degradation-rate constants. (arXiv:2310.15958, Table 2.)
  * Monte-Carlo uncertainty studies treat kinetic parameters as **lognormal**
    (exp of a Gaussian), with hydrolysis constants the most uncertain; biogas
    flow remains comparatively robust to that spread. (IWA WST 92(4):610;
    ScienceDirect S1369703X23000050.)
  => multiplicative lognormal factor ``exp(N(0, sigma))`` on k_dis/k_hyd/k_m/
     k_dec/K_S, with ``sigma`` in 0.2-0.3 (CoV 20-30 %).

Measurement noise (1-sigma, per sensor type):
  * Biogas volume flow: expanded uncertainty ~3 % (drift < 0.15 %/24 h).
  * NDIR CH4/CO2 fraction: ~0.7-1 % absolute vs GC; the *flow* channel
    Q_ch4 = Q_gas x CH4% inherits ~3-4 % combined.
  * pH probe: ~0.05 pH units; FOS/TAC titration ~5-10 %; gravimetric TS ~3 %;
    dosing scale ~3 %.
  (Olythe/Dynament NDIR specs; biogas-volume metrology PMC12693810.)
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

# ---------------------------------------------------------------------------
# Measurement noise (1-sigma) per sensor *type* — plant-agnostic.
# Each entry: (kind, value). kind="relative" -> value * |reading|;
# kind="absolute" -> value in the reading's own units.
# ---------------------------------------------------------------------------
NoiseKind = Literal["relative", "absolute"]

SENSOR_NOISE_MODEL: Dict[str, Tuple[NoiseKind, float]] = {
    "q_gas": ("relative", 0.03),  # biogas flow meter ~3 % expanded uncertainty
    "q_gas_total": ("relative", 0.03),
    "q_ch4": ("relative", 0.04),  # Q_gas x NDIR CH4 (~3 % x ~1 %) combined
    "q_ch4_total": ("relative", 0.04),
    "q_co2": ("relative", 0.04),  # Q_gas x NDIR CO2
    "ph": ("absolute", 0.05),  # pH units
    "vfa": ("relative", 0.08),  # FOS/TAC titration
    "ts": ("relative", 0.03),  # gravimetric TS/VS
    "substrate_dose": ("relative", 0.03),  # dosing scale
}

#: Some UKF-for-AD studies inflate R by ~1.5x for robustness (arXiv:2310.15958);
#: keep 1.0 by default and expose the knob.
R_INFLATION: float = 1.0

# ---------------------------------------------------------------------------
# Model error: multiplicative lognormal perturbation of the biological
# kinetics. This is the *transferable* stand-in for "the model is not 100 %
# reality" — it cannot be measured from plant data, so it is set from the
# ADM1 literature, not from any one plant.
# ---------------------------------------------------------------------------
#: Kinetic-parameter name prefixes perturbed (rates + Monod affinities). The
#: physical equilibrium constants (K_a, K_w, K_H) and stoichiometry (Y_*, f_*)
#: are NOT perturbed.
MODEL_ERROR_PREFIXES: Tuple[str, ...] = ("k_dis", "k_hyd", "k_m_", "k_dec", "K_S")

#: sigma of exp(N(0, sigma)). 0.20 ~ CoV 20 % (sensitivity range);
#: 0.30 ~ plant-model mismatch (upper realistic end). 0.25 is the default.
MODEL_ERROR_KINETIC_SIGMA: float = 0.25


def absolute_sensor_std(catalog_name: str, nominal_value: float) -> float:
    """Absolute 1-sigma noise for a catalog sensor given a representative
    reading magnitude.

    ``relative`` entries are scaled by ``|nominal_value|`` (with a small floor
    so a momentarily-zero reading does not collapse the noise); ``absolute``
    entries ignore it. The result is multiplied by :data:`R_INFLATION`.

    Raises:
        KeyError: if ``catalog_name`` has no entry in
            :data:`SENSOR_NOISE_MODEL`.
    """
    kind, value = SENSOR_NOISE_MODEL[catalog_name]
    if kind == "absolute":
        std = value
    else:
        std = value * max(abs(float(nominal_value)), 1e-9)
    return float(std * R_INFLATION)


def build_sensor_noise(
    nominal_magnitudes: Dict[str, float],
) -> Dict[str, float]:
    """Build a ``build_ukf(sensor_noise=...)`` dict from the realism profile.

    Args:
        nominal_magnitudes: representative reading magnitude per catalog
            sensor name (e.g. ``{"q_gas": 1900, "q_ch4": 900, "q_co2": 850}``).
            ``absolute`` sensors (pH) may be omitted. ``substrate_dose`` is a
            relative factor and is returned as the relative value directly
            (``build_ukf`` already scales it by the per-substrate flow).

    Returns:
        ``{catalog_name: noise_std}`` ready for ``build_ukf``. For
        ``substrate_dose`` the value is the *relative* factor (not absolute).
    """
    out: Dict[str, float] = {}
    for name, (kind, value) in SENSOR_NOISE_MODEL.items():
        if name == "substrate_dose":
            out[name] = float(value * R_INFLATION)  # relative factor
        elif kind == "absolute":
            out[name] = absolute_sensor_std(name, 0.0)
        elif name in nominal_magnitudes:
            out[name] = absolute_sensor_std(name, nominal_magnitudes[name])
    return out


__all__ = [
    "SENSOR_NOISE_MODEL",
    "R_INFLATION",
    "MODEL_ERROR_PREFIXES",
    "MODEL_ERROR_KINETIC_SIGMA",
    "absolute_sensor_std",
    "build_sensor_noise",
]
