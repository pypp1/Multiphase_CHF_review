"""
validate_lut_regression.py
===========================
Standalone diagnostic for the pressure-regression bridge used by the
2006 Groeneveld CHF look-up table (see models/lut_regression.py).

This is NOT part of the analysis pipeline and is not run by main.py.
Run it directly:

    python validate_lut_regression.py

It fits the polynomial regression curves, prints a fit-error summary
table, and saves three check figures to results/:

    lut_regression_check.png            G-X error map + per-pressure box plot
    lut_regression_operating_zone.png   same map with the cases' envelope
    lut_regression_slice_P7000_G1500.png  CHF and error along x at fixed P, G
"""

from pathlib import Path

import numpy as np

from main import _load_cases, _is_lofa, _is_pressurisation
from models import groeneveld_regression_chf
from models.lut_data import _G_AXIS, _X_AXIS
from models.lut_regression import (
    build_regression_curves,
    compute_regression_errors,
    plot_regression_error_map,
    plot_operating_zone_map,
    plot_regression_quality_slice,
)
from water_properties import sat_props

RESULTS_DIR = Path(__file__).parent / "results"

SLICE_P_KPA = 7000
SLICE_G = 1500


def _case_operating_points():
    """
    Return an array of (G, P_Pa, X_CHF) rows: every operating point
    evaluated by the active cases, with the critical quality predicted
    by the LUT regression model at that point.
    """
    rows = []
    for case in _load_cases():
        if not getattr(case, 'ACTIVE', False):
            continue
        if _is_lofa(case):
            G_arr = np.arange(case.G_START, case.G_END + case.G_STEP, case.G_STEP)
            points = [(G, case.P_PA) for G in G_arr]
        elif _is_pressurisation(case):
            P_arr = np.linspace(case.P_NOM_PA, case.P_PEAK_PA, case.N_STEPS)
            points = [(case.G_KGM2S, P) for P in P_arr]
        else:
            points = [(case.G_KGM2S, case.P_PA)]

        for G, P_Pa in points:
            r = groeneveld_regression_chf(G, case.D_M, case.L_M, P_Pa,
                                          sat_props(P_Pa).H_fg, case.X_IN)
            if r.x_chf is not None:
                rows.append((G, P_Pa, r.x_chf))
    return np.array(rows)


def main():
    print("Fitting degree-2 polynomial regression curves over the pressure axis...")
    curves = build_regression_curves()

    print("\nRegression fit error vs. tabulated Appendix B data:\n")
    errors = compute_regression_errors(curves, _G_AXIS, _X_AXIS)

    ops = _case_operating_points()
    print(f"\nCase operating envelope: G = {ops[:, 0].min():.0f}-{ops[:, 0].max():.0f} "
          f"kg/(m^2*s), X_CHF = {ops[:, 2].min():.3f}-{ops[:, 2].max():.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)

    path = RESULTS_DIR / "lut_regression_check.png"
    plot_regression_error_map(curves, errors, _G_AXIS, _X_AXIS, save_path=str(path))
    print(f"\nSaved {path}")

    path = RESULTS_DIR / "lut_regression_operating_zone.png"
    plot_operating_zone_map(curves, errors, _G_AXIS, _X_AXIS,
                            points=ops[:, [0, 2]], save_path=str(path))
    print(f"Saved {path}")

    at_slice_G = ops[ops[:, 0] == SLICE_G]
    x_range = ((at_slice_G[:, 2].min(), at_slice_G[:, 2].max())
               if len(at_slice_G) else None)
    path = RESULTS_DIR / f"lut_regression_slice_P{SLICE_P_KPA}_G{SLICE_G}.png"
    plot_regression_quality_slice(curves, _G_AXIS, _X_AXIS, SLICE_P_KPA, SLICE_G,
                                  x_range=x_range, save_path=str(path))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
