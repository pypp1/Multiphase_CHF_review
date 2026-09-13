"""
models/katto1978.py
===================
Katto (1978) generalised correlation for critical heat flux in
vertical uniformly heated round tubes.

Reference
---------
Katto, Y. (1978).
"A generalized correlation of critical heat flux for the forced
convection boiling in vertical uniformly heated round tubes."
Int. J. Heat Mass Transfer, 21, 1527-1542.

Physical model
--------------
Four CHF regimes are identified (L, H, N, HP) based on the
dimensionless groups  rho_v/rho_l,  sigma*rho_l/(G^2l),  and  l/d.

Regime boundaries (eqs. 16-19 of Katto 1978) select the correlation.
Subcooling effect is accounted for via a K factor (eqs. 21-23):
    q_c = q_c0 * (1 + K * dH_i / H_fg)

Units
-----
All inputs and outputs are SI (Pa, kg/m^2/s, m, J/kg, W/m^2).
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class KattoResult:
    """Output of the Katto CHF calculation."""
    regime: str                       # 'L' | 'H' | 'N' | 'HP'
    q_c0:   float                     # W/m^2 -- saturated CHF
    q_c:    float                     # W/m^2 -- CHF with subcooling correction
    K:      float                     # subcooling factor (nan for N-regime)
    x_ex:   float                     # exit quality from heat balance
    warnings: list = field(default_factory=list)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _dim_groups(G: float, d: float, l: float,
                rho_l: float, rho_v: float, sigma: float):
    """Compute the three key dimensionless groups."""
    rho_ratio = rho_v / rho_l
    l_over_d  = l / d
    We_inv    = sigma * rho_l / (G**2 * l)   # inverse Weber-like group sigma*rho_l/(G^2l)
    return rho_ratio, l_over_d, We_inv


def _detect_regime(rho_ratio: float, l_over_d: float, We_inv: float,
                   fluid: str = 'water') -> str:
    """
    Select the CHF regime according to the boundary equations of
    Katto (1978), eqs. (16)-(19).
    """
    C_LH = 0.29 if fluid.lower() == 'freon' else 0.40

    # L/H boundary (eq. 16)
    rhs_LH = C_LH * (rho_ratio**0.133) * (We_inv**-0.29) - 0.0031
    if l_over_d > rhs_LH:
        return 'L'

    # H/HP boundary (eq. 19)
    num_HP = 82.0 * (rho_ratio**0.517) - (We_inv**-0.12)
    den_HP = 107.0 * (We_inv**0.42) - 0.254 * (rho_ratio**0.517)
    rhs_HP = num_HP / den_HP if abs(den_HP) > 1e-15 else np.inf
    if l_over_d < rhs_HP:
        return 'HP'

    # H/N boundary (eq. 18)
    rhs_HN = 0.77 / (We_inv**0.37)
    if l_over_d > rhs_HN:
        return 'N'

    return 'H'


def _qc0(regime: str, G: float, H_fg: float,
         rho_ratio: float, l_over_d: float, We_inv: float,
         fluid: str = 'water') -> float:
    """Saturated CHF q_c0 [W/m^2] for the identified regime."""
    if regime == 'L':
        C = 0.34 if fluid.lower() == 'freon' else 0.25
        return C * G * H_fg * (We_inv**0.043) / l_over_d
    if regime == 'H':
        return (G * H_fg * 0.10 * (rho_ratio**0.133) * (We_inv**1.3)
                / (1.0 + 0.0031 * l_over_d))
    if regime == 'N':
        return (G * H_fg * 0.098 * (rho_ratio**0.133) * (We_inv**0.433)
                * (l_over_d**0.27) / (1.0 + 0.0031 * l_over_d))
    if regime == 'HP':
        return (G * H_fg * 8.20 * (rho_ratio**0.65) * (We_inv**0.453)
                / (1.0 + 107.0 * (We_inv**0.54) * l_over_d))
    raise ValueError(f"Unknown regime '{regime}'.")


def _K_factor(regime: str, l_over_d: float,
              rho_ratio: float, We_inv: float) -> float:
    """
    Subcooling K factor from Katto (1978) eqs. (21)-(23).
    Returns NaN for N-regime (no correlation available).
    """
    if regime == 'L':
        return 1.16

    if regime == 'H':
        if We_inv < 3e-6:
            return 1.8 * (l_over_d / 130.0)**(-5.0 * rho_ratio)
        return (0.075 * (l_over_d / 130.0)**(-5.0 * rho_ratio)
                * (We_inv**-0.25))

    if regime == 'HP':
        if We_inv < 4e-8:
            return 0.664 * rho_ratio**-0.6
        return 3.08 * (We_inv**0.09) * rho_ratio**-0.6

    # N-regime: no correlation for K in Katto 1978 Section 6.4
    return float('nan')


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def katto_chf(
    G:      float,
    d:      float,
    l:      float,
    rho_l:  float,
    rho_v:  float,
    H_fg:   float,
    sigma:  float,
    dH_i:   float = 0.0,
    fluid:  str   = 'water',
) -> KattoResult:
    """
    Compute the CHF according to Katto (1978).

    Parameters
    ----------
    G : float
        Mass flux [kg/(m^2*s)].
    d : float
        Tube inner diameter [m].
    l : float
        Heated length [m].
    rho_l : float
        Saturated liquid density [kg/m^3].
    rho_v : float
        Saturated vapour density [kg/m^3].
    H_fg : float
        Latent heat of vaporisation [J/kg].
    sigma : float
        Surface tension [N/m].
    dH_i : float, optional
        Inlet subcooling enthalpy rise [J/kg].  0 = saturated inlet.
    fluid : str, optional
        'water' (default) or 'freon'.  Affects L/H boundary constant.

    Returns
    -------
    KattoResult
    """
    warnings = []

    rho_ratio, l_over_d, We_inv = _dim_groups(G, d, l, rho_l, rho_v, sigma)
    regime = _detect_regime(rho_ratio, l_over_d, We_inv, fluid)

    # VL-regime flag (very low mass velocity + short tube)
    u_in = G / rho_l
    if u_in < 0.3 and l_over_d < 50:
        warnings.append(
            "Possible VL-regime (u_in < 0.3 m/s, l/d < 50): pool-boiling "
            "may dominate -- Katto 1978 Section 4.1 supplement."
        )

    qc0 = _qc0(regime, G, H_fg, rho_ratio, l_over_d, We_inv, fluid)
    K   = _K_factor(regime, l_over_d, rho_ratio, We_inv)

    if regime == 'N':
        warnings.append(
            "N-regime: subcooling K factor is unavailable (Katto 1978 Section 6.4). "
            "q_c returned equals q_c0 (conservative, no subcooling credit)."
        )
        q_c = qc0
    elif dH_i > 0.0 and not np.isnan(K):
        q_c = qc0 * (1.0 + K * dH_i / H_fg)
    else:
        q_c = qc0

    # Exit quality from heat balance (Katto eq. 24)
    x_ex = 4.0 * q_c * l / (G * H_fg * d) - dH_i / H_fg

    return KattoResult(
        regime=regime, q_c0=qc0, q_c=q_c,
        K=K, x_ex=x_ex, warnings=warnings,
    )
