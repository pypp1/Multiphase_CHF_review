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

Reference anchor points generated with the `iapws` Python package
(IAPWS-IF97 for T_sat, densities, enthalpies and cp; IAPWS releases for
surface tension and viscosity). Max linear-interpolation error vs. the
direct IAPWS evaluation over 4-10 MPa: < 0.3 % for all properties.
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
    [ 4.0, 250.36,  798.4,  20.09,  1713.5e3, 25.96e-3,   4869.0, 106.1e-6, 17.4e-6],
    [ 5.0, 263.94,  777.4,  25.35,  1639.7e3, 22.76e-3,   5032.0, 100.1e-6, 18.0e-6],
    [ 6.0, 275.59,  758.0,  30.82,  1570.8e3, 20.03e-3,   5208.0,  95.3e-6, 18.4e-6],
    [ 7.0, 285.83,  739.7,  36.52,  1505.1e3, 17.63e-3,   5400.0,  91.3e-6, 18.9e-6],
    [ 7.5, 290.54,  730.9,  39.48,  1473.1e3, 16.54e-3,   5504.0,  89.5e-6, 19.1e-6],
    [ 8.0, 295.01,  722.2,  42.50,  1441.5e3, 15.51e-3,   5614.0,  87.7e-6, 19.3e-6],
    [ 8.5, 299.27,  713.6,  45.61,  1410.3e3, 14.53e-3,   5730.0,  86.1e-6, 19.5e-6],
    [ 9.0, 303.35,  705.2,  48.80,  1379.2e3, 13.60e-3,   5854.0,  84.6e-6, 19.8e-6],
    [10.0, 311.00,  688.4,  55.45,  1317.6e3, 11.86e-3,   6127.0,  81.7e-6, 20.2e-6],
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
