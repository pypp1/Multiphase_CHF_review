"""
models/biasi1968.py
===================
Biasi et al. (1968) Critical Heat Flux correlation for uniformly
heated round ducts.

Reference
---------
Biasi, L., Clerici, G. C., Tozzi, A., Sala, R. (1968).
"Extension of A.R.S. correlation to burnout prediction with
non-uniform heating."
Journal of Nuclear Energy, 22, 705–716.

Physical model
--------------
CHF is taken as the maximum of two straight lines in the (q, X_0) plane:

  Low-quality branch:
      q_0 = 1.883e3 * y(P) / [D^α * G^(1/6)] * [y(P)/G^(1/6) - X_0]

  High-quality branch:
      q_0 = 3.78e3 * h(P) / [D^α * G^(0.6)] * [1 - X_0]

where
  y(P) = 0.7249 + 0.099·P·exp(-0.032·P)
  h(P) = −1.159 + 0.149·P·exp(-0.019·P) + 8.99·P/(10 + P²)
  α    = 0.4 for D ≥ 1 cm,  0.6 for D < 1 cm
  P    in ata,  D in cm,  G in g/(cm²·s),  q in W/cm²

The intersection with the heat-balance line
  q_t(X) = G·D·H_fg·(X - X_in) / (4·L)
gives the CHF point.

Validity range (uniform heating)
---------------------------------
  0.3 cm ≤ D ≤ 3.75 cm
  20 cm  ≤ L ≤ 600 cm
  1.7 ata ≤ P ≤ 140 ata
  10 g/(cm²·s) ≤ G ≤ 600 g/(cm²·s)
  X_in < 0  (subcooled inlet)

Units
-----
All inputs and outputs are in SI (Pa, kg/m²/s, m, J/kg, W/m²).
Internal conversions to CGS are performed inside the function.
"""

import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BiasiResult:
    """Output of the Biasi CHF calculation."""
    q_chf:    float               # W/m²   — critical heat flux
    x_chf:    float               # —       — quality at CHF point
    branch:   str                 # 'low_x' | 'high_x' | 'both'
    warnings: list = field(default_factory=list)


def _pressure_functions(P_ata: float):
    """Return y(P) and h(P) as defined in Biasi (1968)."""
    y = 0.7249 + 0.099 * P_ata * np.exp(-0.032 * P_ata)
    h = -1.159 + 0.149 * P_ata * np.exp(-0.019 * P_ata) + 8.99 * P_ata / (10.0 + P_ata**2)
    return y, h


def biasi_chf(
    G:     float,
    d:     float,
    L:     float,
    P_Pa:  float,
    H_fg:  float,
    x_in:  float = -0.05,
) -> BiasiResult:
    """
    Compute the CHF according to the Biasi (1968) correlation.

    The CHF is found as the intersection of max(q_low, q_high) with
    the heat-balance line.  Both branches are evaluated and the one
    active at the intersection is recorded.

    Parameters
    ----------
    G : float
        Mass flux [kg/(m²·s)].
    d : float
        Tube inner diameter [m].
    L : float
        Heated length [m].
    P_Pa : float
        System pressure [Pa].
    H_fg : float
        Latent heat of vaporisation [J/kg].
    x_in : float, optional
        Inlet thermodynamic quality (< 0 for subcooled).  Default -0.05.

    Returns
    -------
    BiasiResult
    """
    warnings = []

    # ── Unit conversions: SI → CGS ────────────────────────────────────────────
    P_ata = P_Pa / (9.81e4)          # 1 ata ≈ 98 100 Pa
    G_cgs = G * 1e-4 / 1e-3         # kg/(m²·s) → g/(cm²·s) = 0.1·G_SI
    # Actually: 1 kg/(m²·s) = 1e-4 kg/(cm²·s) = 0.1 g/(cm²·s)
    G_cgs = G * 0.1                  # g/(cm²·s)
    d_cm  = d * 100.0                # m → cm
    L_cm  = L * 100.0                # m → cm
    H_fg_cgs = H_fg * 1e-3          # J/kg → kJ/kg   (Biasi uses kJ/kg… not needed directly)
    H_fg_Jg  = H_fg / 1000.0        # J/g (for CGS heat balance)

    # ── Biasi pressure functions ──────────────────────────────────────────────
    y, h = _pressure_functions(P_ata)

    # ── α coefficient ─────────────────────────────────────────────────────────
    alpha = 0.4 if d_cm >= 1.0 else 0.6

    # ── Validity checks ───────────────────────────────────────────────────────
    if not (0.3 <= d_cm <= 3.75):
        warnings.append(f"D = {d_cm:.2f} cm outside validity range [0.3, 3.75] cm.")
    if not (20 <= L_cm <= 600):
        warnings.append(f"L = {L_cm:.0f} cm outside validity range [20, 600] cm.")
    if not (1.7 <= P_ata <= 140):
        warnings.append(f"P = {P_ata:.1f} ata outside validity range [1.7, 140] ata.")
    if not (10 <= G_cgs <= 600):
        warnings.append(f"G = {G_cgs:.1f} g/(cm²·s) outside validity range [10, 600].")
    if x_in >= 0:
        warnings.append("x_in ≥ 0: correlation is derived for subcooled inlet only.")

    # ── Biasi branch functions [W/cm²] ───────────────────────────────────────
    def q_low(x):
        return 1.883e3 / (d_cm**alpha * G_cgs**(1.0/6.0)) * (y / G_cgs**(1.0/6.0) - x)

    def q_high(x):
        return 3.78e3 * h / (d_cm**alpha * G_cgs**0.6) * (1.0 - x)

    def q_biasi(x):
        """Envelope: maximum of the two branches."""
        return np.maximum(q_low(x), q_high(x))

    # ── Heat-balance line [W/cm²] ─────────────────────────────────────────────
    # q_t(x) = G_cgs * d_cm * H_fg_Jg * (x - x_in) / (4 * L_cm)
    def q_t(x):
        return G_cgs * d_cm * H_fg_Jg * (x - x_in) / (4.0 * L_cm)

    # ── Find intersection ─────────────────────────────────────────────────────
    def residual(x):
        return q_biasi(x) - q_t(x)

    # Search range: from x_in to just below 1.0
    x_lo = x_in
    x_hi = 0.999

    # Check that a root exists in [x_lo, x_hi]
    try:
        f_lo = residual(x_lo)
        f_hi = residual(x_hi)
        if f_lo * f_hi > 0:
            warnings.append("No CHF crossing found in quality range [x_in, 0.999]. "
                            "Returning boundary value.")
            # Fall back: CHF is the minimum of q_biasi on the interval
            x_vals = np.linspace(x_lo, x_hi, 500)
            idx = np.argmin(np.abs(residual(x_vals)))
            x_chf = x_vals[idx]
        else:
            x_chf = brentq(residual, x_lo, x_hi, xtol=1e-6)
    except Exception as exc:
        warnings.append(f"Solver error: {exc}")
        x_chf = float("nan")

    q_chf_Wcm2 = q_t(x_chf)
    q_chf_SI   = q_chf_Wcm2 * 1e4   # W/cm² → W/m²

    # Identify dominant branch at x_chf
    ql = q_low(x_chf)
    qh = q_high(x_chf)
    if abs(ql - qh) < 0.01 * max(abs(ql), abs(qh)):
        branch = "both"
    elif ql > qh:
        branch = "low_x"
    else:
        branch = "high_x"

    return BiasiResult(q_chf=q_chf_SI, x_chf=x_chf,
                       branch=branch, warnings=warnings)
