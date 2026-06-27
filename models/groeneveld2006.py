"""
models/groeneveld2006.py
========================
2006 CHF Look-Up Table (LUT) by Groeneveld et al.

Reference
---------
Groeneveld, D. C. et al. (2007).
"The 2006 CHF look-up table."
Nuclear Engineering and Design, 237, 1909–1922.

Physical model
--------------
The LUT gives the normalised CHF for a vertical 8 mm ID tube cooled
by water as a function of pressure (P), mass flux (G) and local
thermodynamic quality (X).

To apply to a different tube diameter the diameter correction is:
    CHF(D) = CHF(8 mm) × (D / 8)^(−1/2)         (Groeneveld 2007, §4)

The Heat Balance Method (HBM) is used to find the self-consistent
(q_CHF, X) pair along the tube:
    X_out = X_in + 4·q·L / (G·d·H_fg)
    q_CHF = LUT(P, G, X_out) × diameter_correction

Iteration converges to the q for which the LUT CHF equals the
assumed heat flux.

LUT data included
-----------------
This module embeds a representative subset of the 2006 LUT at
P = 7 000 kPa (Table in Appendix B of Groeneveld 2007, rows for
P = 7 000 kPa).  The full table can be extended by populating
`_LUT_DATA` below.

For pressures other than those in the embedded table, nearest-pressure
interpolation is used with a warning.

Units
-----
All inputs/outputs in SI (Pa, kg/m²/s, m, J/kg, W/m²).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from dataclasses import dataclass, field


@dataclass
class LUTResult:
    """Output of the Groeneveld 2006 LUT CHF calculation."""
    q_chf:    float               # W/m²   — critical heat flux
    x_chf:    float               # —       — quality at CHF
    warnings: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 2006 LUT data — P = 7 000 kPa
# From Groeneveld et al. (2007), Appendix B, P = 7 000 kPa rows.
# Quality axis: 23 values from -0.50 to 1.00
# Mass flux axis (kg/m²/s): rows as listed in the paper
# ─────────────────────────────────────────────────────────────────────────────

_X_AXIS = np.array([
    -0.50, -0.40, -0.30, -0.20, -0.15, -0.10, -0.05,
     0.00,  0.05,  0.10,  0.15,  0.20,  0.25,  0.30,
     0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.80,
     0.90,  1.00
])

# Mass flux grid (kg/m²/s)
_G_AXIS = np.array([0, 50, 100, 300, 500, 750, 1000, 1500,
                    2000, 2500, 3000, 3500, 4000, 4500, 5000,
                    5500, 6000, 6500, 7000, 7500, 8000])

# CHF matrix [kW/m²] at P = 7 000 kPa — shape (21, 23)
# Source: Table in Appendix B of Groeneveld et al. (2007)
_CHF_7000 = np.array([
    # G=0
    [5069, 4676, 4093, 3540, 3282, 3007, 2760, 2534, 2354, 2161, 1999, 1843, 1709, 1587, 1476, 1376, 1288, 1205, 1061, 939,  821,  710,  0],
    # G=50
    [5919, 5536, 5191, 4863, 4462, 4520, 4306, 3998, 3399, 2986, 2264, 2042, 1712, 1588, 1477, 1366, 1151, 1010,  473,  325,  200,  129,  0],
    # G=100
    [6912, 6301, 5871, 5481, 5462, 5073, 4600, 4000, 3609, 3069, 2549, 2380, 2216, 2087, 1949, 1798, 1700, 1652, 1541, 1280, 1078,  708,  501],
    # G=300
    [7445, 6709, 6259, 6020, 5914, 5761, 5662, 5495, 5182, 4752, 4464, 4070, 3764, 3611, 3417, 3250, 3028, 2738, 2286, 1994, 1408,  869,   0],
    # G=500
    [7842, 6895, 6435, 6188, 5996, 5931, 5818, 5672, 5408, 4922, 4521, 4196, 3812, 3989, 3602, 3459, 3221, 2905, 2482, 1985, 1547,  869,   0],
    # G=750
    [9129, 7841, 6867, 6263, 6154, 5998, 5895, 5776, 5430, 4987, 4196, 3918, 3812, 3464, 3327, 3118, 2770, 2312, 1904, 1400,  742,  341,   0],
    # G=1000
    [10186, 8774, 7390, 6532, 6313, 6276, 6162, 5864, 4920, 5366, 4399, 3935, 3723, 3447, 3112, 2884, 2713, 2432, 2085,  767,  506,  341,   0],
    # G=1500
    [11920, 10072, 8460, 7262, 6915, 6647, 6308, 5729, 4561, 5059, 3612, 4039, 2991, 3279, 2264, 2490, 2698, 1591,  599,  318,  372,  191,   0],
    # G=2000
    [13294, 11209, 9219, 7557, 7118, 7342, 6388, 6095, 4800, 4390, 3568, 4524, 3801, 3926, 1919, 3929, 2926, 1903,  897,  512,  288,  163,   0],
    # G=2500
    [14680, 12245, 9774, 7920, 7382, 6765, 5895, 4497, 4178, 3639, 3207, 2867, 2552, 2211, 1941, 1487,  813,  521,  342,  177,  103,   58,   0],
    # G=3000
    [15851, 13569, 10463, 8551, 8024, 8038, 7642, 6596, 5771, 5187, 4839, 4355, 3698, 3152, 2289, 1188, 1188,  878,  578,  348,  179,   63,   0],
    # G=3500
    [16889, 14072, 11223, 8783, 7744, 6972, 5738, 4518, 3739, 3127, 2816, 2482, 2188, 1798, 1357,  851,  631,  531,  388,  224,   96,   44,   0],
    # G=4000
    [17783, 14824, 11868, 9619, 8281, 7208, 6209, 5381, 4156, 3342, 2472, 2067, 1647, 1279, 1106,  867,  779,  487,  266,  244,   99,   44,   0],
    # G=4500
    [18619, 15498, 12439, 9619, 9281, 7208, 6725, 5594, 4329, 3504, 2981, 2677, 2057, 1647, 1239, 1006,  867,  779,  653,  440,  119,   47,   0],
    # G=5000
    [19434, 16132, 12870, 10084, 9272, 7844, 7415, 5486, 4350, 4505, 2251, 2246, 1960, 1179, 1279, 1006,  854,  850,  854,  276,  106,   47,   0],
    # G=5500
    [20138, 16733, 13579, 10563, 9272, 7844, 6153, 4649, 3405, 2688, 2325, 2076, 1662, 1397, 1476, 1339, 1184, 1184,  637,  298,  131,   52,   0],
    # G=6000
    [20703, 17309, 14047, 11951, 10355, 9156, 6697, 4756, 3313, 2547, 2442, 2325, 2076, 1662, 1938, 1515, 1303, 1182,  702,  360,  149,   69,   0],
    # G=6500
    [21284, 17855, 14610, 11951, 10355, 9156, 7135, 4950, 3437, 2733, 2542, 2442, 1938, 1938, 1688, 1515, 1303, 1103,  702,  360,  149,   69,   0],
    # G=7000
    [21888, 18357, 15013, 12260, 10817, 9456, 7309, 4949, 3504, 2648, 2348, 2199, 2009, 2199, 1652, 1339, 1184, 1182,  637,  433,  182,   86,   0],
    # G=7500
    [22505, 18841, 15385, 12539, 11219, 9779, 7455, 5004, 3629, 3017, 2792, 2596, 2263, 2348, 2263, 1776, 1504, 1264,  838,  433,  182,   86,   0],
    # G=8000
    [23064, 19369, 15794, 12917, 11519, 10059, 7792, 5163, 3777, 3222, 3122, 3301, 2917, 2547, 2272, 2021, 1794, 1573, 1136,  668,  312,  150,   0],
])


# ─────────────────────────────────────────────────────────────────────────────
# Build interpolator
# ─────────────────────────────────────────────────────────────────────────────

_lut_7000 = RegularGridInterpolator(
    (_G_AXIS, _X_AXIS),
    _CHF_7000,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)

# Registry: pressure [kPa] → interpolator
_AVAILABLE_PRESSURES = {7000: _lut_7000}


def _nearest_interpolator(P_kPa: float):
    """Return the interpolator for the closest available pressure."""
    avail = sorted(_AVAILABLE_PRESSURES.keys())
    nearest = min(avail, key=lambda p: abs(p - P_kPa))
    return _AVAILABLE_PRESSURES[nearest], nearest


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def groeneveld_chf(
    G:     float,
    d:     float,
    L:     float,
    P_Pa:  float,
    H_fg:  float,
    x_in:  float = -0.05,
    q_lo:  float = 1e5,
    q_hi:  float = 2e7,
) -> LUTResult:
    """
    Compute CHF from the 2006 Groeneveld look-up table using the
    Heat Balance Method (HBM).

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
        Inlet thermodynamic quality.  Default −0.05.
    q_lo : float, optional
        Lower bracket for HBM search [W/m²].  Default 100 kW/m².
    q_hi : float, optional
        Upper bracket for HBM search [W/m²].  Default 20 MW/m².

    Returns
    -------
    LUTResult
    """
    warnings = []

    P_kPa = P_Pa / 1e3
    lut, p_used = _nearest_interpolator(P_kPa)

    if abs(p_used - P_kPa) > 100:
        warnings.append(
            f"Requested P = {P_kPa:.0f} kPa; nearest LUT pressure used: "
            f"{p_used:.0f} kPa.  Results may be inaccurate."
        )

    d_mm = d * 1000.0
    diam_correction = (d_mm / 8.0)**-0.5   # Groeneveld 2007 eq. in §4

    def _heat_balance_error(q_est: float) -> float:
        x_out = x_in + 4.0 * q_est * L / (G * d * H_fg)
        x_out = np.clip(x_out, _X_AXIS[0], _X_AXIS[-1])
        chf_8mm = lut((G, x_out)) * 1e3   # kW/m² → W/m²
        chf_d   = chf_8mm * diam_correction
        return chf_d - q_est

    try:
        f_lo = _heat_balance_error(q_lo)
        f_hi = _heat_balance_error(q_hi)

        if f_lo * f_hi > 0:
            if f_lo > 0:
                warnings.append(
                    f"LUT CHF exceeds q_hi = {q_hi/1e6:.1f} MW/m²: "
                    "CHF is above the search bracket."
                )
            else:
                warnings.append(
                    f"LUT CHF falls below q_lo = {q_lo/1e3:.0f} kW/m²: "
                    "CHF is below the search bracket."
                )
            return LUTResult(q_chf=float('nan'), x_chf=float('nan'),
                             warnings=warnings)

        q_chf = brentq(_heat_balance_error, q_lo, q_hi, xtol=100.0)

    except Exception as exc:
        warnings.append(f"HBM solver error: {exc}")
        return LUTResult(q_chf=float('nan'), x_chf=float('nan'),
                         warnings=warnings)

    x_chf = x_in + 4.0 * q_chf * L / (G * d * H_fg)
    return LUTResult(q_chf=q_chf, x_chf=x_chf, warnings=warnings)
