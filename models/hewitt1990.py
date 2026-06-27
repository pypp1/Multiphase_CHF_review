"""
models/hewitt1990.py
====================
Hewitt & Govan (1990) phenomenological annular-flow model for
Critical Heat Flux (dryout) prediction.

Reference
---------
Hewitt, G. F. and Govan, A. H. (1990).
"Phenomenological modelling of non-equilibrium flows with phase change."
Int. J. Heat Mass Transfer, 33(2), 229–242.

Physical model
--------------
In annular flow the liquid phase travels partly as a wall film and partly
as entrained droplets in the vapour core.  Dryout (CHF) is predicted when
the liquid film flow rate m_LF → 0.

The film flow rate evolves along the tube as:
    dm_LF/dz = (4/d) · (D − E − q/H_fg)           (eq. 7)

with deposition and entrainment rates given by the Hewitt-Govan
correlations (eqs. 3–6):

  Deposition:
    k √(ρG·d/σ) = 0.18               if C/ρG < 0.3
    k √(ρG·d/σ) = 0.083·(C/ρG)^-0.65 if C/ρG ≥ 0.3

  Entrainment onset (critical film flow rate, eq. 6):
    Re_LFC = exp(5.8504 + 0.4249·(ηG/ηL)·√(ρL/ρG))

  Entrainment rate (eq. 5):
    E/m_G = 5.75×10⁻⁵·[(m_LF − m_LFC)²·d·ρL/(σ·ρG²)]^0.316
            if m_LF > m_LFC, else 0

CHF is the heat flux at which m_LF(L) = 0 (found by bisection).

Boundary condition (onset of annular flow)
------------------------------------------
Annular flow is assumed to begin at x = x_onset (default 0.01).
At that point 99 % of liquid is entrained, so:
    m_LF_0 = 0.01 · G · (1 − x_onset)         (conservative)

Units
-----
All inputs and outputs are SI (Pa, kg/m²/s, m, J/kg, W/m²).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from dataclasses import dataclass, field


@dataclass
class HewittResult:
    """Output of the Hewitt & Govan dryout calculation."""
    q_chf:    float               # W/m²   — critical heat flux
    z_dryout: float               # m      — axial dryout location (≈ L at CHF)
    warnings: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _re_lfc(eta_l: float, eta_v: float, rho_l: float, rho_v: float) -> float:
    """Critical film Reynolds number for onset of entrainment (eq. 6)."""
    return np.exp(5.8504 + 0.4249 * (eta_v / eta_l) * np.sqrt(rho_l / rho_v))


def _deposition_rate(C: float, rho_v: float, sigma: float, d: float) -> float:
    """Deposition mass flux D [kg/(m²·s)] from Hewitt-Govan (eqs. 3–4)."""
    C_ratio = C / rho_v
    factor  = np.sqrt(sigma / (rho_v * d))
    if C_ratio < 0.3:
        k = 0.18 * factor
    else:
        k = 0.083 * (C_ratio**-0.65) * factor
    return k * C


def _entrainment_rate(m_LF: float, m_LFC: float, m_G: float,
                      d: float, rho_l: float, rho_v: float, sigma: float) -> float:
    """Entrainment mass flux E [kg/(m²·s)] from Hewitt-Govan (eq. 5)."""
    if m_LF <= m_LFC:
        return 0.0
    term = (m_LF - m_LFC)**2 * d * rho_l / (sigma * rho_v**2)
    return m_G * 5.75e-5 * (term**0.316)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def hewitt_chf(
    G:       float,
    d:       float,
    L:       float,
    rho_l:   float,
    rho_v:   float,
    H_fg:    float,
    sigma:   float,
    eta_l:   float,
    eta_v:   float,
    x_in:    float  = -0.05,
    x_onset: float  = 0.01,
    q_lo:    float  = 1e5,
    q_hi:    float  = 1e7,
) -> HewittResult:
    """
    Compute CHF via the Hewitt-Govan annular-flow dryout model.

    The CHF is the heat flux at which the liquid film reaches zero at
    exactly z = L.  The search is performed by bisection over [q_lo, q_hi].

    Parameters
    ----------
    G : float
        Mass flux [kg/(m²·s)].
    d : float
        Tube inner diameter [m].
    L : float
        Heated length [m].
    rho_l : float
        Saturated liquid density [kg/m³].
    rho_v : float
        Saturated vapour density [kg/m³].
    H_fg : float
        Latent heat of vaporisation [J/kg].
    sigma : float
        Surface tension [N/m].
    eta_l : float
        Dynamic viscosity of liquid [Pa·s].
    eta_v : float
        Dynamic viscosity of vapour [Pa·s].
    x_in : float, optional
        Inlet thermodynamic quality.  Default −0.05 (subcooled).
    x_onset : float, optional
        Quality at which annular flow is assumed to start.  Default 0.01.
    q_lo : float, optional
        Lower bracket for CHF search [W/m²].  Default 100 kW/m².
    q_hi : float, optional
        Upper bracket for CHF search [W/m²].  Default 10 MW/m².

    Returns
    -------
    HewittResult
    """
    warnings = []

    Re_LFC = _re_lfc(eta_l, eta_v, rho_l, rho_v)
    m_LFC  = Re_LFC * eta_l / d       # critical film mass flux [kg/(m²·s)]

    def _film_flow_at_outlet(q_flux: float) -> float:
        """
        Integrate the film ODE from the onset of annular flow to L.
        Returns m_LF at z = L (positive = film survives, ≤ 0 = dryout).
        """
        # Axial position where annular flow starts
        z_onset = (x_onset - x_in) * G * d * H_fg / (4.0 * q_flux)

        if z_onset >= L:
            # The tube never reaches annular flow — no dryout
            return 1.0

        # Boundary condition: 99 % of liquid entrained at x_onset
        m_LF_0 = 0.01 * G * (1.0 - x_onset)

        def dm_dz(z: float, m_LF_arr):
            m_LF = max(0.0, m_LF_arr[0])

            # Local quality
            x_loc = x_in + 4.0 * q_flux * z / (G * d * H_fg)
            if x_loc >= 1.0:
                return [-1e6]   # force rapid dryout

            m_G  = x_loc * G
            m_LE = max(0.0, G * (1.0 - x_loc) - m_LF)  # entrained liquid flux

            # Droplet concentration in gas core
            V_GC = m_G / rho_v + m_LE / rho_l
            C    = m_LE / V_GC if V_GC > 0 else 0.0

            D = _deposition_rate(C, rho_v, sigma, d)
            E = _entrainment_rate(m_LF, m_LFC, m_G, d, rho_l, rho_v, sigma)

            evap = q_flux / H_fg
            return [(4.0 / d) * (D - E - evap)]

        # Stop integration as soon as the film hits zero
        def dryout_event(z, m_LF_arr):
            return m_LF_arr[0]
        dryout_event.terminal  = True
        dryout_event.direction = -1

        sol = solve_ivp(
            dm_dz,
            [z_onset, L],
            [m_LF_0],
            events=dryout_event,
            max_step=L / 200.0,
            method='RK45',
            rtol=1e-4,
            atol=1e-6,
        )

        if sol.status == 1:
            # Dryout event triggered — film reached zero before L
            z_dry = sol.t_events[0][0]
            return z_dry - L   # negative: dryout is upstream of exit
        else:
            return sol.y[0, -1]   # positive: film survives to L

    # ── Bisection to find CHF ─────────────────────────────────────────────────
    f_lo = _film_flow_at_outlet(q_lo)
    f_hi = _film_flow_at_outlet(q_hi)

    if f_lo * f_hi > 0:
        if f_lo > 0:
            warnings.append(
                f"Film survives at q_hi = {q_hi/1e6:.2f} MW/m²: "
                "CHF is above the search bracket. Try increasing q_hi."
            )
        else:
            warnings.append(
                f"Dryout already at q_lo = {q_lo/1e3:.0f} kW/m²: "
                "CHF is below the search bracket. Try decreasing q_lo."
            )
        return HewittResult(q_chf=float('nan'), z_dryout=float('nan'),
                            warnings=warnings)

    q_chf = brentq(_film_flow_at_outlet, q_lo, q_hi, xtol=100.0)

    return HewittResult(q_chf=q_chf, z_dryout=L, warnings=warnings)
