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


# Discrete MARD classes for the G-X error map [%]; values above the last
# bound fall into the "over" class (> 50 %).
_MARD_BOUNDS = [0.0, 5.0, 10.0, 20.0, 50.0]
_MARD_COLORS = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20']
_MARD_OVER = '#99000d'
_NOFIT_COLOR = '#d9d9d9'
_NOFIT_HATCH = '#8c8c8c'


def _mard_map(rel_errors):
    """Mean absolute relative error over the pressure axis, per (G, X) cell [%]."""
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.abs(rel_errors), axis=0)


def _draw_error_map(ax, curves, rel_errors, G_axis, X_axis):
    """
    Draw the per-cell MARD map on `ax` with equispaced (index-based)
    cells, discrete colour classes and hatched no-fit cells. Returns the
    QuadMesh so the caller can attach a colorbar.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Rectangle

    n_G, n_X = len(G_axis), len(X_axis)
    mard = _mard_map(rel_errors)
    no_fit = np.array([[curves[i, j] is None for j in range(n_X)]
                       for i in range(n_G)])
    mard[no_fit] = np.nan

    cmap = ListedColormap(_MARD_COLORS)
    cmap.set_over(_MARD_OVER)
    norm = BoundaryNorm(_MARD_BOUNDS, cmap.N)

    x_edges = np.arange(n_X + 1) - 0.5
    g_edges = np.arange(n_G + 1) - 0.5

    mesh = ax.pcolormesh(
        x_edges, g_edges, np.ma.masked_invalid(mard),
        cmap=cmap, norm=norm, edgecolor='white', linewidth=0.3,
    )

    # No-fit cells: grey and hatched, so they are not read as zero error
    for i_G, i_X in zip(*np.nonzero(no_fit)):
        ax.add_patch(Rectangle(
            (i_X - 0.5, i_G - 0.5), 1, 1, facecolor=_NOFIT_COLOR,
            edgecolor=_NOFIT_HATCH, hatch='////', linewidth=0,
        ))

    ax.set_xticks(np.arange(n_X))
    ax.set_xticklabels([f"{x:g}" for x in X_axis], rotation=90, fontsize=5)
    ax.set_yticks(np.arange(n_G))
    ax.set_yticklabels([f"{g:g}" for g in G_axis], fontsize=5)
    ax.set_xlabel("Quality x [-]", fontsize=8)
    ax.set_ylabel("G [kg/(m^2 s)]", fontsize=8)
    ax.tick_params(length=1.5, pad=1)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(g_edges[0], g_edges[-1])
    ax.set_aspect('equal')

    ax.legend(
        handles=[Patch(facecolor=_NOFIT_COLOR, hatch='////',
                       edgecolor=_NOFIT_HATCH, linewidth=0, label='no fit')],
        fontsize=5.5, frameon=False, loc='lower left',
        bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2, handlelength=1.2,
    )
    return mesh


def _add_mard_colorbar(fig, ax, mesh):
    cbar = fig.colorbar(mesh, ax=ax, extend='max', spacing='uniform',
                        fraction=0.046, pad=0.03)
    cbar.set_ticks(_MARD_BOUNDS)
    cbar.ax.tick_params(labelsize=6, length=1.5)
    cbar.set_label("MARD over P [%]", fontsize=7)
    return cbar


def _save_or_show(fig, save_path):
    import matplotlib.pyplot as plt
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_regression_error_map(curves, errors, G_axis, X_axis, save_path=None):
    """
    Two-panel diagnostic figure:
      (a) G-X map of the mean absolute relative deviation of each
          regression curve from the table, averaged over all tabulated
          pressures (one cell = one fitted polynomial);
      (b) box plot of the relative-error distribution at each tabulated
          pressure.

    `errors` is the dict returned by `compute_regression_errors()`.
    If `save_path` is given the figure is saved there (PNG, 150 DPI);
    otherwise it is shown interactively.
    """
    import matplotlib.pyplot as plt

    G_axis = np.asarray(G_axis)
    X_axis = np.asarray(X_axis)
    rel_errors = errors['rel_errors']

    fig, (ax_map, ax_box) = plt.subplots(
        2, 1, figsize=(3.4, 5.6), dpi=150,
        gridspec_kw=dict(height_ratios=[1.2, 1.0]),
    )

    # --- Panel (a): per-cell MARD map --------------------------------------
    mesh = _draw_error_map(ax_map, curves, rel_errors, G_axis, X_axis)
    _add_mard_colorbar(fig, ax_map, mesh)
    ax_map.set_title("(a)", fontsize=8, loc='right')

    # --- Panel (b): relative-error box plot per pressure slice -------------
    box_data = [r[~np.isnan(r)] for r in rel_errors.reshape(len(_P_AXIS_RAW), -1)]
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
    ax_box.set_title("(b)", fontsize=8, loc='right')

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_operating_zone_map(curves, errors, G_axis, X_axis, points, save_path=None):
    """
    Same G-X MARD map as panel (a) of `plot_regression_error_map()`, with
    the operating envelope of the analysed cases overlaid.

    `points` is an iterable of (G, X_CHF) pairs actually reached by the
    cases. The rectangle encloses every tabulated cell whose regression
    curve enters the bilinear interpolation at those points; the points
    themselves are drawn as small markers.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    G_axis = np.asarray(G_axis)
    X_axis = np.asarray(X_axis)
    pts = np.asarray(list(points), dtype=float)

    fig, ax = plt.subplots(figsize=(3.4, 3.6), dpi=150)
    mesh = _draw_error_map(ax, curves, errors['rel_errors'], G_axis, X_axis)
    _add_mard_colorbar(fig, ax, mesh)

    # Continuous (G, X) -> fractional cell index on the equispaced grid
    g_idx = np.interp(pts[:, 0], G_axis, np.arange(len(G_axis)))
    x_idx = np.interp(pts[:, 1], X_axis, np.arange(len(X_axis)))

    g_lo, g_hi = np.floor(g_idx.min()), np.ceil(g_idx.max())
    x_lo, x_hi = np.floor(x_idx.min()), np.ceil(x_idx.max())
    ax.add_patch(Rectangle(
        (x_lo - 0.5, g_lo - 0.5), x_hi - x_lo + 1, g_hi - g_lo + 1,
        fill=False, edgecolor='black', linewidth=1.2, zorder=3,
    ))
    ax.plot(x_idx, g_idx, 'o', markerfacecolor='white', markeredgecolor='black',
            markeredgewidth=0.4, markersize=2.2, linestyle='none', zorder=4,
            label='case points')

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_regression_quality_slice(curves, G_axis, X_axis, P_kPa, G,
                                  x_range=None, save_path=None):
    """
    CHF and signed relative error along the quality axis at a fixed
    tabulated pressure and mass flux:
      (a) tabulated CHF vs. regression CHF;
      (b) signed relative error (regression - table) / table.

    `x_range` = (x_min, x_max), if given, is shaded as the quality range
    reached by the analysed cases at this (P, G).
    """
    import matplotlib.pyplot as plt

    G_axis = np.asarray(G_axis)
    X_axis = np.asarray(X_axis)
    i_P = int(np.argmin(np.abs(_P_AXIS_RAW - P_kPa)))
    i_G = int(np.argmin(np.abs(G_axis - G)))
    if _P_AXIS_RAW[i_P] != P_kPa or G_axis[i_G] != G:
        raise ValueError(f"P = {P_kPa} kPa and G = {G} must be tabulated values.")

    chf_table = _CHF_RAW[i_P, i_G, :].astype(float)
    chf_reg = evaluate_lut_regression(P_kPa, curves, G_axis, X_axis)[i_G, :]
    valid = chf_table > 0
    rel_err = np.full(len(X_axis), np.nan)
    rel_err[valid] = (chf_reg[valid] - chf_table[valid]) / chf_table[valid] * 100.0

    fig, (ax_chf, ax_err) = plt.subplots(
        2, 1, figsize=(3.15, 4.4), dpi=150, sharex=True,
        gridspec_kw=dict(height_ratios=[1.3, 1.0]),
    )

    for ax in (ax_chf, ax_err):
        if x_range is not None:
            ax.axvspan(*x_range, color='#bdbdbd', alpha=0.45, linewidth=0,
                       label='cases' if ax is ax_chf else None)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)

    # --- Panel (a): CHF along x --------------------------------------------
    ax_chf.plot(X_axis[valid], chf_table[valid], 'o', color='#333333',
                markersize=3.5, linestyle='none', label='table')
    ax_chf.plot(X_axis, chf_reg, '-s', color='#2c7fb8', markersize=2.5,
                linewidth=1.0, label='regression')
    ax_chf.set_ylabel("CHF [kW/m^2]", fontsize=8)
    ax_chf.legend(fontsize=6, frameon=False, loc='best')
    ax_chf.set_title(f"P = {P_kPa:.0f} kPa, G = {G:.0f} kg/(m^2 s)",
                     fontsize=7)

    # --- Panel (b): signed relative error ----------------------------------
    widths = np.diff(X_axis).min() * 0.7
    colors = np.where(rel_err >= 0, '#d7301f', '#2c7fb8')
    ax_err.bar(X_axis[valid], rel_err[valid], width=widths,
               color=colors[valid], edgecolor='none')
    ax_err.axhline(0.0, color='black', linewidth=0.7)
    ax_err.axhline(5.0, color='gray', linewidth=0.6, linestyle='--')
    ax_err.axhline(-5.0, color='gray', linewidth=0.6, linestyle='--')
    ax_err.set_xlabel("Quality x [-]", fontsize=8)
    ax_err.set_ylabel("Relative error [%]", fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, save_path)
