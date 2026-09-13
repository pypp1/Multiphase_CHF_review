"""
models/groeneveld2006_regression.py
=====================================
CHF lookup for the 2006 Groeneveld LUT using ONLY the polynomial
pressure regression (models/lut_regression.py) -- no raw tabulated CHF
data (`_CHF_RAW`) is read anywhere in this file.

Mass flux and quality are treated as continuous: for a given pressure,
the four regression curves surrounding the requested (G, X) point are
bilinearly interpolated (see `evaluate_regression_at_point`).

Reference
---------
Groeneveld, D. C. et al. (2007). "The 2006 CHF look-up table."
Nuclear Engineering and Design, 237, 1909-1922.
"""

from scipy.optimize import brentq

from .lut_data import _G_AXIS, _X_AXIS, _P_AXIS_RAW
from .lut_regression import build_regression_curves, evaluate_regression_at_point
from .groeneveld2006 import LUTResult


# -----------------------------------------------------------------------------
# Build regression curves once at module load time
# -----------------------------------------------------------------------------

_CURVES = build_regression_curves()


def groeneveld_regression_chf(
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
    Compute CHF from the pressure-regression bridge over the 2006
    Groeneveld CHF look-up table. Mass flux and quality are continuous.

    Parameters
    ----------
    G : float
        Mass flux [kg/(m^2*s)].
    d : float
        Tube inner diameter [m].
    L : float
        Heated length [m].
    P_Pa : float
        System pressure [Pa].
    H_fg : float
        Latent heat of vaporisation [J/kg].
    x_in : float
        Inlet thermodynamic quality.
    q_lo, q_hi : float, optional
        Bisection bracket for the HBM search [W/m^2].

    Returns
    -------
    LUTResult
    """
    warnings = []
    P_kPa = P_Pa / 1e3

    if P_kPa < _P_AXIS_RAW.min() or P_kPa > _P_AXIS_RAW.max():
        warnings.append(
            f"P = {P_kPa:.0f} kPa is outside the tabulated range "
            f"[{_P_AXIS_RAW.min():.0f}, {_P_AXIS_RAW.max():.0f}] kPa; "
            "extrapolation is unreliable."
        )

    d_mm = d * 1000.0
    diam_correction = (d_mm / 8.0) ** -0.5   # Groeneveld 2007 eq. in Section 4

    def _residual(q_est: float) -> float:
        X_CHF = x_in + 4.0 * q_est * L / (G * d * H_fg)
        q_reg, _ = evaluate_regression_at_point(
            P_kPa, G, X_CHF, _CURVES, _G_AXIS, _X_AXIS
        )
        return q_reg * 1e3 * diam_correction - q_est

    try:
        f_lo = _residual(q_lo)
        f_hi = _residual(q_hi)

        if f_lo * f_hi > 0:
            warnings.append(
                f"HBM residual does not change sign between "
                f"q_lo={q_lo:.3g} and q_hi={q_hi:.3g} W/m^2; "
                "bisection cannot proceed."
            )
            return LUTResult(q_chf=None, x_chf=None, warnings=warnings)

        q_chf = brentq(_residual, q_lo, q_hi, xtol=100.0)

    except Exception as exc:
        warnings.append(f"HBM solver error: {exc}")
        return LUTResult(q_chf=None, x_chf=None, warnings=warnings)

    X_CHF = x_in + 4.0 * q_chf * L / (G * d * H_fg)
    _, final_warnings = evaluate_regression_at_point(
        P_kPa, G, X_CHF, _CURVES, _G_AXIS, _X_AXIS
    )
    warnings.extend(final_warnings)

    return LUTResult(q_chf=q_chf, x_chf=X_CHF, warnings=warnings)
