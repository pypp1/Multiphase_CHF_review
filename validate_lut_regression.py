"""
validate_lut_regression.py
===========================
Standalone diagnostic for the pressure-regression bridge used by the
2006 Groeneveld CHF look-up table (see models/lut_regression.py).

This is NOT part of the analysis pipeline and is not run by main.py.
Run it directly:

    python validate_lut_regression.py

It fits the polynomial regression curves, prints a fit-error summary
table, and saves a check figure to results/lut_regression_check.png.
"""

from pathlib import Path

from models.lut_data import _G_AXIS, _X_AXIS
from models.lut_regression import (
    build_regression_curves,
    compute_regression_errors,
    plot_regression_curves,
)

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    print("Fitting degree-2 polynomial regression curves over the pressure axis...")
    curves = build_regression_curves()

    print("\nRegression fit error vs. tabulated Appendix B data:\n")
    compute_regression_errors(curves, _G_AXIS, _X_AXIS)

    RESULTS_DIR.mkdir(exist_ok=True)
    save_path = RESULTS_DIR / "lut_regression_check.png"
    plot_regression_curves(curves, _G_AXIS, _X_AXIS, save_path=str(save_path))
    print(f"\nSaved check figure to {save_path}")


if __name__ == "__main__":
    main()
