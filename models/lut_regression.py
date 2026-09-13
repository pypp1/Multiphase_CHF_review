"""
models/lut_regression.py
=========================
Polynomial-regression bridge for the 2006 Groeneveld CHF look-up table.

The raw table (see `models/lut_data.py`) only has 15 discrete pressure
slices. To evaluate CHF at an arbitrary pressure that falls between two
tabulated slices, this module fits one degree-2 polynomial per (mass
flux, quality) grid point across the pressure axis, and provides tools
to evaluate, validate and visualise those curves.

Reference
---------
Groeneveld, D. C. et al. (2007). "The 2006 CHF look-up table."
Nuclear Engineering and Design, 237, 1909-1922.
"""

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from .lut_data import _P_AXIS_RAW, _G_AXIS, _X_AXIS, _CHF_RAW


# -----------------------------------------------------------------------------
# SECTION A -- Regression
# -----------------------------------------------------------------------------

def build_regression_curves():
    """
    Fit one degree-2 polynomial per (mass flux, quality) grid point,
    across the pressure axis, to the tabulated CHF values.

    Zero and negative CHF values are excluded from each fit -- they are
    unphysical placeholders (e.g. the x = 1.0 column, which is always 0).

    Returns
    -------
    np.ndarray of shape (len(_G_AXIS), len(_X_AXIS)), dtype=object
        Each entry is a numpy.polynomial.polynomial.Polynomial fitted to
        the valid (pressure, CHF) points at that (G, X) grid point, or
        None if fewer than 3 valid points were available to fit.
    """
    n_G = len(_G_AXIS)
    n_X = len(_X_AXIS)
    curves = np.empty((n_G, n_X), dtype=object)

    for i_G in range(n_G):
        for i_X in range(n_X):
            chf_vals = _CHF_RAW[:, i_G, i_X]
            valid = chf_vals > 0
            if np.count_nonzero(valid) < 3:
                curves[i_G, i_X] = None
                continue
            curves[i_G, i_X] = Polynomial.fit(
                _P_AXIS_RAW[valid], chf_vals[valid], deg=2
            )

    return curves


def evaluate_lut_regression(P_kPa, curves, G_axis, X_axis):
    """
    Evaluate the fitted regression curves at a given pressure.

    Parameters
    ----------
    P_kPa : float
        Pressure at which to evaluate the curves [kPa].
    curves : np.ndarray
        Output of `build_regression_curves()`.
    G_axis, X_axis : array-like
        Mass flux and quality axes matching the shape of `curves`.

    Returns
    -------
    np.ndarray of shape (len(G_axis), len(X_axis))
        Predicted CHF [kW/m^2], with negative predictions clipped to zero.
    """
    n_G = len(G_axis)
    n_X = len(X_axis)
    chf = np.zeros((n_G, n_X))

    for i_G in range(n_G):
        for i_X in range(n_X):
            curve = curves[i_G, i_X]
            chf[i_G, i_X] = 0.0 if curve is None else curve(P_kPa)

    return np.clip(chf, 0.0, None)


def _bracket(value, axis, warnings, name, unit=""):
    """
    Return (i_lo, clamped_value) such that axis[i_lo] <= clamped_value
    <= axis[i_lo+1]. A value exactly on a grid point is treated as the
    lower bound of its bracket. A value outside [axis[0], axis[-1]] is
    clamped to the nearest boundary and a warning is appended to
    `warnings`.
    """
    n = len(axis)
    unit_sfx = f" {unit}" if unit else ""
    if value <= axis[0]:
        if value < axis[0]:
            warnings.append(
                f"{name} = {value:.4g}{unit_sfx} is below the tabulated "
                f"range; clamped to {axis[0]:.4g}{unit_sfx}."
            )
        return 0, axis[0]
    if value >= axis[-1]:
        if value > axis[-1]:
            warnings.append(
                f"{name} = {value:.4g}{unit_sfx} is above the tabulated "
                f"range; clamped to {axis[-1]:.4g}{unit_sfx}."
            )
        return n - 2, axis[-1]
    i = int(np.searchsorted(axis, value, side='right') - 1)
    i = min(max(i, 0), n - 2)
    return i, value


def evaluate_regression_at_point(P_kPa, G, X, curves, G_axis, X_axis):
    """
    Evaluate CHF at a continuous (G, X) point, at a given pressure, by
    bilinearly interpolating the four regression curves surrounding
    (G, X) in the tabulated grid.

    Parameters
    ----------
    P_kPa : float
        Pressure at which to evaluate the regression curves [kPa].
    G : float
        Continuous mass flux [kg/(m^2*s)].
    X : float
        Continuous thermodynamic quality.
    curves : np.ndarray
        Output of `build_regression_curves()`.
    G_axis, X_axis : array-like
        Mass flux and quality axes matching the shape of `curves`.

    Returns
    -------
    (q_kWm2, warnings) : (float, list of str)
        Bilinearly-interpolated CHF [kW/m^2] (clipped to >= 0), and any
        warnings raised while bracketing G or X (e.g. clamping to the
        tabulated range).
    """
    warnings = []
    G_axis = np.asarray(G_axis)
    X_axis = np.asarray(X_axis)

    i_G, G_c = _bracket(G, G_axis, warnings, "G", "kg/(m^2*s)")
    i_X, X_c = _bracket(X, X_axis, warnings, "X")

    def _eval(curve):
        return 0.0 if curve is None else curve(P_kPa)

    q00 = _eval(curves[i_G,     i_X])
    q10 = _eval(curves[i_G + 1, i_X])
    q01 = _eval(curves[i_G,     i_X + 1])
    q11 = _eval(curves[i_G + 1, i_X + 1])

    t_G = (G_c - G_axis[i_G]) / (G_axis[i_G + 1] - G_axis[i_G])
    t_X = (X_c - X_axis[i_X]) / (X_axis[i_X + 1] - X_axis[i_X])

    q = ((1 - t_G) * (1 - t_X) * q00 + t_G * (1 - t_X) * q10
         + (1 - t_G) * t_X       * q01 + t_G * t_X       * q11)

    return float(max(q, 0.0)), warnings


# -----------------------------------------------------------------------------
# SECTION B -- Error evaluation
# -----------------------------------------------------------------------------

def compute_regression_errors(curves, G_axis, X_axis):
    """
    Compare the regression curves against the tabulated data at every
    tabulated pressure and summarise the fit error.

    Returns
    -------
    dict with keys:
        'abs_errors' : np.ndarray (n_P, n_G, n_X) -- CHF_regression - CHF_table [kW/m^2]
        'rel_errors' : np.ndarray (n_P, n_G, n_X) -- relative error [%]
        'mae'   : float -- mean absolute error [kW/m^2]
        'rmse'  : float -- root mean square error [kW/m^2]
        'mre'   : float -- mean relative error [%]
        'rmsre' : float -- root mean square relative error [%]
        'bias'  : float -- signed mean relative error [%];
                  positive = regression overpredicts CHF (non-conservative),
                  negative = regression underpredicts CHF (conservative)
        'mard'  : float -- mean absolute relative deviation [%]
                  (the standard accuracy metric used in CHF publications,
                  including Groeneveld et al. (2007) themselves)
        'p95'   : float -- 95th percentile of absolute relative
                  deviation [%]; captures worst-case behaviour
    Errors are computed only where the tabulated CHF > 0; other entries
    in 'abs_errors'/'rel_errors' are NaN.
    """
    n_P, n_G, n_X = _CHF_RAW.shape
    abs_errors = np.full((n_P, n_G, n_X), np.nan)
    rel_errors = np.full((n_P, n_G, n_X), np.nan)

    print(f"{'P [kPa]':>10} | {'MAE [kW/m2]':>12} | {'RMSE [kW/m2]':>13} | "
          f"{'Bias [%]':>8} | {'MARD [%]':>8} | {'RMSRE [%]':>10} | {'P95 [%]':>8}")
    print("-" * 84)

    all_abs = []
    all_rel = []

    for i_P, P_kPa in enumerate(_P_AXIS_RAW):
        chf_reg = evaluate_lut_regression(P_kPa, curves, G_axis, X_axis)
        chf_table = _CHF_RAW[i_P]
        valid = chf_table > 0

        abs_valid = chf_reg[valid] - chf_table[valid]
        rel_valid = abs_valid / chf_table[valid] * 100.0

        abs_errors[i_P][valid] = abs_valid
        rel_errors[i_P][valid] = rel_valid
        all_abs.append(abs_valid)
        all_rel.append(rel_valid)

        mae_p = np.mean(np.abs(abs_valid))
        rmse_p = np.sqrt(np.mean(abs_valid**2))
        bias_p = np.mean(rel_valid)
        mard_p = np.mean(np.abs(rel_valid))
        rmsre_p = np.sqrt(np.mean(rel_valid**2))
        p95_p = np.percentile(np.abs(rel_valid), 95)

        print(f"{P_kPa:>10} | {mae_p:>12.1f} | {rmse_p:>13.1f} | "
              f"{bias_p:>8.2f} | {mard_p:>8.2f} | {rmsre_p:>10.2f} | {p95_p:>8.2f}")

    all_abs = np.concatenate(all_abs)
    all_rel = np.concatenate(all_rel)

    mae = np.mean(np.abs(all_abs))
    rmse = np.sqrt(np.mean(all_abs**2))
    mre = np.mean(all_rel)
    rmsre = np.sqrt(np.mean(all_rel**2))
    bias = np.mean(all_rel)
    mard = np.mean(np.abs(all_rel))
    p95 = np.percentile(np.abs(all_rel), 95)

    print("-" * 84)
    print(f"{'overall':>10} | {mae:>12.1f} | {rmse:>13.1f} | "
          f"{bias:>8.2f} | {mard:>8.2f} | {rmsre:>10.2f} | {p95:>8.2f}")

    return {
        'abs_errors': abs_errors,
        'rel_errors': rel_errors,
        'mae': mae,
        'rmse': rmse,
        'mre': mre,
        'rmsre': rmsre,
        'bias': bias,
        'mard': mard,
        'p95': p95,
    }


# -----------------------------------------------------------------------------
# SECTION C -- Visualisation
# -----------------------------------------------------------------------------

_SAMPLE_GX_PAIRS = [
    (500, 0.0),
    (1500, 0.0),
    (1500, 0.3),
    (1500, 0.5),
    (3000, 0.3),
]
_MARKERS = ['o', 's', '^', 'D', 'v']


def plot_regression_curves(curves, G_axis, X_axis, save_path=None):
    """
    Produce a compact two-panel figure:
      1. Regression curves vs. raw data for a representative subset of
         (G, X) pairs.
      2. Box plot of the relative-error distribution at each tabulated
         pressure.

    If `save_path` is given the figure is saved there (PNG, 150 DPI);
    otherwise it is shown interactively.
    """
    import matplotlib.pyplot as plt

    G_axis = np.asarray(G_axis)
    X_axis = np.asarray(X_axis)

    fig, (ax_curves, ax_box) = plt.subplots(
        2, 1, figsize=(3.15, 5.2), dpi=150
    )

    # --- Panel 1: regression curves with raw data -------------------------
    P_dense = np.linspace(_P_AXIS_RAW.min(), _P_AXIS_RAW.max(), 200)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(_SAMPLE_GX_PAIRS)))

    for (G_val, X_val), marker, color in zip(_SAMPLE_GX_PAIRS, _MARKERS, colors):
        i_G = int(np.argmin(np.abs(G_axis - G_val)))
        i_X = int(np.argmin(np.abs(X_axis - X_val)))

        raw = _CHF_RAW[:, i_G, i_X]
        valid = raw > 0
        label = f"G={G_axis[i_G]:.0f}, x={X_axis[i_X]:.2f}"

        ax_curves.plot(
            _P_AXIS_RAW[valid], raw[valid],
            marker, color=color, markersize=3.5, linestyle='none', label=label,
        )

        curve = curves[i_G, i_X]
        if curve is not None:
            chf_dense = np.clip(curve(P_dense), 0.0, None)
            ax_curves.plot(P_dense, chf_dense, '-', color=color, linewidth=1.0)

    ax_curves.set_xlabel("Pressure [kPa]", fontsize=8)
    ax_curves.set_ylabel("CHF [kW/m^2]", fontsize=8)
    ax_curves.tick_params(labelsize=7)
    ax_curves.legend(fontsize=5.5, frameon=False, loc='best')
    ax_curves.spines['top'].set_visible(False)
    ax_curves.spines['right'].set_visible(False)

    # --- Panel 2: relative-error box plot per pressure slice ---------------
    box_data = []
    for i_P in range(len(_P_AXIS_RAW)):
        chf_reg = evaluate_lut_regression(_P_AXIS_RAW[i_P], curves, G_axis, X_axis)
        chf_table = _CHF_RAW[i_P]
        valid = chf_table > 0
        rel_err = (chf_reg[valid] - chf_table[valid]) / chf_table[valid] * 100.0
        box_data.append(rel_err)

    positions = np.arange(len(_P_AXIS_RAW))
    ax_box.boxplot(
        box_data, positions=positions, widths=0.6,
        showfliers=False, patch_artist=True,
        boxprops=dict(facecolor='#8da0cb', edgecolor='#333333', linewidth=0.6),
        medianprops=dict(color='#333333', linewidth=1.0),
        whiskerprops=dict(color='#333333', linewidth=0.6),
        capprops=dict(color='#333333', linewidth=0.6),
    )
    ax_box.axhline(0.0, color='black', linewidth=0.7, linestyle='--')
    ax_box.axhline(5.0, color='gray', linewidth=0.6, linestyle='--')
    ax_box.axhline(-5.0, color='gray', linewidth=0.6, linestyle='--')

    ax_box.set_xticks(positions)
    ax_box.set_xticklabels([f"{p:.0f}" for p in _P_AXIS_RAW], rotation=90, fontsize=6)
    ax_box.set_xlabel("Pressure [kPa]", fontsize=8)
    ax_box.set_ylabel("Relative error [%]", fontsize=8)
    ax_box.tick_params(labelsize=7)
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
