"""
water_properties.py
===================
Saturated water/steam thermophysical properties by piecewise linear
interpolation on IAPWS-IF97 table data.

Valid range: 4 - 10 MPa.

Units
-----
P       Pa
T_sat   K
rho_l   kg/m^3
rho_v   kg/m^3
H_fg    J/kg
sigma   N/m
cp_l    J/(kg*K)
eta_l   Pa*s
eta_v   Pa*s

Reference anchor points (IAPWS-IF97):
  P[MPa]  T_sat[degC]  rho_l   rho_v   H_fg[kJ/kg]  sigma[mN/m]  cp_l[kJ/kgK]  eta_l[uPa*s]  eta_v[uPa*s]
  4.0      250.4      799.2   20.09   1714.0        26.00        4.864          109.0          15.6
  5.0      263.9      777.0   25.77   1639.7        22.91        4.987          101.0          16.3
  6.0      275.6      756.1   31.89   1570.5        20.00        5.147           95.0          17.0
  7.0      285.8      736.0   38.55   1505.2        17.24        5.350           89.0          17.7
  7.5      290.5      725.4   42.23   1472.2        15.93        5.472           87.0          18.0
  8.0      295.0      714.5   46.19   1438.3        14.64        5.616           85.0          18.3
  8.5      299.3      703.2   50.46   1403.6        13.38        5.789           83.0          18.7
  9.0      303.3      691.5   55.10   1367.6        12.14        5.999           80.0          19.0
 10.0      311.0      667.6   65.44   1293.0         9.80        6.518           75.0          19.8
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SatProps:
    """Saturated water/steam properties at a given pressure."""
    P:      float   # Pa
    T_sat:  float   # K
    rho_l:  float   # kg/m^3
    rho_v:  float   # kg/m^3
    H_fg:   float   # J/kg
    sigma:  float   # N/m
    cp_l:   float   # J/(kg*K)
    eta_l:  float   # Pa*s
    eta_v:  float   # Pa*s


# Table: [P_MPa, T_sat_C, rho_l, rho_v, H_fg_Jkg, sigma_Nm, cp_l_JkgK, eta_l_Pas, eta_v_Pas]
_TABLE = np.array([
    [4.0,  250.4, 799.2,  20.09, 1714.0e3, 26.00e-3, 4864.0, 109.0e-6, 15.6e-6],
    [5.0,  263.9, 777.0,  25.77, 1639.7e3, 22.91e-3, 4987.0, 101.0e-6, 16.3e-6],
    [6.0,  275.6, 756.1,  31.89, 1570.5e3, 20.00e-3, 5147.0,  95.0e-6, 17.0e-6],
    [7.0,  285.8, 736.0,  38.55, 1505.2e3, 17.24e-3, 5350.0,  89.0e-6, 17.7e-6],
    [7.5,  290.5, 725.4,  42.23, 1472.2e3, 15.93e-3, 5472.0,  87.0e-6, 18.0e-6],
    [8.0,  295.0, 714.5,  46.19, 1438.3e3, 14.64e-3, 5616.0,  85.0e-6, 18.3e-6],
    [8.5,  299.3, 703.2,  50.46, 1403.6e3, 13.38e-3, 5789.0,  83.0e-6, 18.7e-6],
    [9.0,  303.3, 691.5,  55.10, 1367.6e3, 12.14e-3, 5999.0,  80.0e-6, 19.0e-6],
    [10.0, 311.0, 667.6,  65.44, 1293.0e3,  9.80e-3, 6518.0,  75.0e-6, 19.8e-6],
])


def sat_props(P_Pa: float) -> SatProps:
    """
    Return saturated water/steam properties at pressure P_Pa [Pa].

    Interpolation is piecewise linear on the IAPWS-IF97 anchor table.
    Valid range: 4 MPa <= P <= 10 MPa.  Values outside this range are
    extrapolated (numpy.interp flat-extrapolation at the boundary).

    Parameters
    ----------
    P_Pa : float
        Pressure in Pascal.

    Returns
    -------
    SatProps
        Dataclass containing all saturated thermophysical properties.
    """
    P_MPa = P_Pa / 1e6
    P_tab = _TABLE[:, 0]

    def _interp(col: int) -> float:
        return float(np.interp(P_MPa, P_tab, _TABLE[:, col]))

    return SatProps(
        P     = P_Pa,
        T_sat = 273.15 + _interp(1),
        rho_l = _interp(2),
        rho_v = _interp(3),
        H_fg  = _interp(4),
        sigma = _interp(5),
        cp_l  = _interp(6),
        eta_l = _interp(7),
        eta_v = _interp(8),
    )
