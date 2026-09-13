"""
models/groeneveld2006_table.py
================================
CHF lookup for the 2006 Groeneveld LUT using ONLY the raw tabulated
data from Appendix B -- no regression, no pressure tolerance, no
interpolation between pressure slices.

This model only produces a result when the requested pressure and mass
flux correspond EXACTLY to values printed in Appendix B, and the
converged quality at CHF falls within half a grid step of a tabulated
quality value. Any other operating point returns
`LUTResult(q_chf=None, x_chf=None, ...)` with a warning explaining why,
rather than silently interpolating -- use `groeneveld_regression_chf`
(models/groeneveld2006_regression.py) for non-tabulated conditions.

Reference
---------
Groeneveld, D. C. et al. (2007). "The 2006 CHF look-up table."
Nuclear Engineering and Design, 237, 1909-1922.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from .lut_data import _P_AXIS_RAW, _G_AXIS, _X_AXIS, _CHF_RAW
from .groeneveld2006 import LUTResult


def groeneveld_table_chf(
    G:     float,
    d:     float,
    L:     float,
    P_Pa:  float,
    H_fg:  float,
    x_in:  float,
    q_lo:  float = 1e5,
    q_hi:  float = 2e7,
) -> LUTResult:
    """
    Compute CHF from the raw 2006 Groeneveld CHF look-up table, with no
    regression and no tolerance on pressure or mass flux: both must
    match a tabulated value exactly.

    Parameters
    ----------
    G : float
        Mass flux [kg/(m^2*s)]. Must equal one of the tabulated values
        in `_G_AXIS` exactly.
    d : float
        Tube inner diameter [m].
    L : float
        Heated length [m].
    P_Pa : float
        System pressure [Pa]. Once converted to kPa, must equal one of
        the tabulated pressures in `_P_AXIS_RAW` exactly.
    H_fg : float
        Latent heat of vaporisation [J/kg].
    x_in : float
        Inlet thermodynamic quality.
    q_lo, q_hi : float, optional
        Bisection bracket for the HBM search [W/m^2].

    Returns
    -------
    LUTResult
        q_chf and x_chf are None (with an explanatory warning) unless
        both (P, G) are exactly tabulated AND the converged X_CHF falls
        within half a grid step of a tabulated quality value.
    """
    P_kPa = P_Pa / 1e3

    p_match = np.isin(P_kPa, _P_AXIS_RAW)
    g_match = np.isin(G, _G_AXIS)

    if not (p_match and g_match):
        return LUTResult(
            q_chf=None, x_chf=None,
            warnings=[
                f"P={P_kPa} kPa and/or G={G} kg/m^2s do not correspond "
                "exactly to tabulated values. No result returned. Use "
                "groeneveld_regression_chf for non-tabulated conditions."
            ],
        )

    i_P = int(np.where(_P_AXIS_RAW == P_kPa)[0][0])
    i_G = int(np.where(_G_AXIS == G)[0][0])

    chf_of_x = interp1d(
        _X_AXIS, _CHF_RAW[i_P, i_G, :],
        kind='linear', bounds_error=False, fill_value='extrapolate',
    )

    d_mm = d * 1000.0
    diam_correction = (d_mm / 8.0) ** -0.5   # Groeneveld 2007 eq. in Section 4

    def _residual(q_est: float) -> float:
        X_CHF = x_in + 4.0 * q_est * L / (G * d * H_fg)
        return float(chf_of_x(X_CHF)) * 1e3 * diam_correction - q_est

    try:
        f_lo = _residual(q_lo)
        f_hi = _residual(q_hi)

        if f_lo * f_hi > 0:
            return LUTResult(
                q_chf=None, x_chf=None,
                warnings=[
                    f"HBM residual does not change sign between "
                    f"q_lo={q_lo:.3g} and q_hi={q_hi:.3g} W/m^2; "
                    "bisection cannot proceed."
                ],
            )

        q_chf = brentq(_residual, q_lo, q_hi, xtol=100.0)

    except Exception as exc:
        return LUTResult(q_chf=None, x_chf=None,
                          warnings=[f"HBM solver error: {exc}"])

    X_CHF = x_in + 4.0 * q_chf * L / (G * d * H_fg)

    half_step = np.min(np.diff(_X_AXIS)) / 2.0
    close = np.abs(_X_AXIS - X_CHF) <= half_step
    if not np.any(close):
        return LUTResult(
            q_chf=None, x_chf=None,
            warnings=[
                f"X_CHF = {X_CHF:.4f} does not fall within half a grid "
                "step of any tabulated quality. No result returned."
            ],
        )

    nearest_X = _X_AXIS[np.argmin(np.abs(_X_AXIS - X_CHF))]
    warnings = [
        f"X_CHF = {X_CHF:.4f} accepted within half-grid-step tolerance "
        f"of tabulated value X = {nearest_X:.2f}."
    ]
    return LUTResult(q_chf=q_chf, x_chf=X_CHF, warnings=warnings)
