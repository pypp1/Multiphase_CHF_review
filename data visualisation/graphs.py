"""
data visualisation/graphs.py
=============================
Generates comparison plots for the CHF toolkit's active cases and saves
them into this same folder.

Usage
-----
    python "data visualisation/graphs.py"

Outputs
-------
Steady-state cases (e.g. case_1a, case_2a) -- two alternative formats:
  - steady_margin_table.csv / .tex   q_CHF, X_CHF and margin, model x case
  - steady_margin_dotplot.png        CHF margin dot plot, model x case

Per LOFA case (e.g. case_3a, case_3b):
  - <case>_margin_vs_G.png           CHF margin vs mass flux, per model;
                                     dryout mass flux given in the legend

LOFA cases combined:
  - lofa_ratio_to_lut.png            q_CHF(model) / q_CHF(LUT regression)
                                     vs mass flux, one panel per case

Per pressurisation case (e.g. case_4a):
  - <case>_margin_vs_P.png           CHF margin vs pressure at constant heat
                                     flux, per model
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from water_properties import sat_props
from models import (biasi_chf, katto_chf, hewitt_chf,
                    groeneveld_table_chf, groeneveld_regression_chf)
from main import _load_cases, _is_lofa, _is_pressurisation

OUT_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Models: case flag, display label and plot style
# -----------------------------------------------------------------------------

MODELS = ["Katto", "Biasi", "Hewitt", "LUT-table", "LUT-regr"]
REFERENCE = "LUT-regr"

FLAGS = {
    "Katto":     "RUN_KATTO",
    "Biasi":     "RUN_BIASI",
    "Hewitt":    "RUN_HEWITT",
    "LUT-table": "RUN_GROENEVELD_TABLE",
    "LUT-regr":  "RUN_GROENEVELD_REGRESSION",
}

LABELS = {
    "Katto":     "Katto (1978)",
    "Biasi":     "Biasi (1968)",
    "Hewitt":    "Hewitt-Govan (1990)",
    "LUT-table": "LUT 2006 (table)",
    "LUT-regr":  "LUT 2006 (regression)",
}

# Okabe-Ito colours; distinct markers so the plots survive grey-scale print.
# The table LUT only returns isolated points, so it is drawn without a line.
STYLE = {
    "Katto":     dict(color="#0072B2", marker="o", linestyle="-"),
    "Biasi":     dict(color="#E69F00", marker="s", linestyle="-"),
    "Hewitt":    dict(color="#CC79A7", marker="^", linestyle="-"),
    "LUT-table": dict(color="#009E73", marker="D", linestyle="none",
                      markerfacecolor="none", markersize=4.5),
    "LUT-regr":  dict(color="#009E73", marker="v", linestyle="-"),
}

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.0,
    "lines.markersize": 2.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIG_W = 3.4  # single-column width [in]


def _active_models(case):
    return [m for m in MODELS if getattr(case, FLAGS[m], False)]


def _eval_model(name, G, D_M, L_M, P_PA, X_IN, props):
    """Return the predicted CHF [W/m^2], or NaN if the model gives no result."""
    dH_i = max(0.0, -X_IN * props.H_fg)
    if name == "Katto":
        q = katto_chf(G, D_M, L_M, props.rho_l, props.rho_v,
                      props.H_fg, props.sigma, dH_i=dH_i).q_c
    elif name == "Biasi":
        q = biasi_chf(G, D_M, L_M, P_PA, props.H_fg, X_IN).q_chf
    elif name == "Hewitt":
        q = hewitt_chf(G, D_M, L_M, props.rho_l, props.rho_v,
                       props.H_fg, props.sigma, props.eta_l, props.eta_v,
                       x_in=X_IN).q_chf
    elif name == "LUT-table":
        q = groeneveld_table_chf(G, D_M, L_M, P_PA, props.H_fg, x_in=X_IN).q_chf
    elif name == "LUT-regr":
        q = groeneveld_regression_chf(G, D_M, L_M, P_PA, props.H_fg, x_in=X_IN).q_chf
    else:
        raise ValueError(name)
    return np.nan if q is None else float(q)


def _eval_safe(name, *args):
    try:
        return _eval_model(name, *args)
    except Exception:
        return np.nan


def _crossing(x, y, level=1.0):
    """
    First abscissa at which `y` falls below `level`, linearly
    interpolated between samples. Returns x[0] if y starts below
    `level`, None if it never crosses.
    """
    for i in range(len(y)):
        if np.isnan(y[i]) or y[i] >= level:
            continue
        if i == 0 or np.isnan(y[i - 1]):
            return x[i]
        t = (y[i - 1] - level) / (y[i - 1] - y[i])
        return x[i - 1] + t * (x[i] - x[i - 1])
    return None


def _dryout_label(x_d, x0, symbol, fmt):
    """Legend suffix for the dryout point returned by `_crossing()`."""
    if x_d is None:
        return ", no dryout"
    if x_d == x0:
        return ", dryout at start"
    return f", ${symbol}$ = {x_d:{fmt}}"


def _case_stem(case):
    path = getattr(case, "__file__", None)
    if path:
        return Path(path).stem
    return case.NAME.split("--")[0].strip().lower().replace(" ", "_")


def _case_label(case):
    """Short label, e.g. 'Case 1a (d = 8 mm)'."""
    return f"{case.NAME.split('--')[0].strip()} (d = {case.D_M * 1e3:.0f} mm)"


# -----------------------------------------------------------------------------
# Steady-state cases: table and dot plot
# -----------------------------------------------------------------------------

def _steady_results(steady_cases):
    """{case_stem: {model: (q_chf [W/m^2], x_chf, margin)}}"""
    results = {}
    for case in steady_cases:
        props = sat_props(case.P_PA)
        q_nom = case.Q_NOM_MW * 1e6
        active = _active_models(case)
        row = {}
        for m in MODELS:
            q = (_eval_safe(m, case.G_KGM2S, case.D_M, case.L_M,
                            case.P_PA, case.X_IN, props)
                 if m in active else np.nan)
            x = case.X_IN + 4.0 * q * case.L_M / (case.G_KGM2S * case.D_M * props.H_fg)
            row[m] = (q, x, q / q_nom)
        results[_case_stem(case)] = row
    return results


def write_steady_table(steady_cases, results):
    """Save the steady-state comparison as CSV and as a LaTeX booktabs table."""
    stems = [_case_stem(c) for c in steady_cases]

    with open(OUT_DIR / "steady_margin_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [f"{s}_{k}" for s in stems
                                for k in ("qchf_MWm2", "x_chf", "margin")])
        for m in MODELS:
            row = [LABELS[m]]
            for s in stems:
                q, x, mg = results[s][m]
                row += (["", "", ""] if np.isnan(q)
                        else [f"{q / 1e6:.3f}", f"{x:.3f}", f"{mg:.3f}"])
            w.writerow(row)

    col_spec = "l" + "ccc" * len(steady_cases)
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join([""] + [rf"\multicolumn{{3}}{{c}}{{{_case_label(c)}}}"
                           for c in steady_cases]) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                for i in range(len(steady_cases))),
        " & ".join(["Model"] + [r"$q''_\mathrm{CHF}$ [MW/m$^2$]",
                                r"$X_\mathrm{CHF}$", "Margin"]
                   * len(steady_cases)) + r" \\",
        r"\midrule",
    ]
    for m in MODELS:
        cells = [LABELS[m]]
        for s in stems:
            q, x, mg = results[s][m]
            cells += (["--", "--", "--"] if np.isnan(q)
                      else [f"{q / 1e6:.3f}", f"{x:.3f}", f"{mg:.2f}"])
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "steady_margin_table.tex").write_text("\n".join(lines) + "\n")

    # Console version
    print("\nSteady-state CHF margin (q_CHF [MW/m^2] / X_CHF / margin):")
    print(f"  {'':<24}" + "".join(f"{_case_label(c):>30}" for c in steady_cases))
    for m in MODELS:
        cells = []
        for s in stems:
            q, x, mg = results[s][m]
            cells.append("--" if np.isnan(q)
                         else f"{q / 1e6:.3f} / {x:.3f} / {mg:.2f}")
        print(f"  {LABELS[m]:<24}" + "".join(f"{c:>30}" for c in cells))


def plot_steady_dotplot(steady_cases, results):
    """Cleveland dot plot: one row per model, one marker per case."""
    case_markers = ["o", "s", "D", "^"]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.2))
    y = np.arange(len(MODELS))[::-1]

    for k, case in enumerate(steady_cases):
        s = _case_stem(case)
        margins = np.array([results[s][m][2] for m in MODELS])
        colors = [STYLE[m]["color"] for m in MODELS]
        filled = k == 0
        ax.scatter(margins, y, marker=case_markers[k % len(case_markers)], s=22,
                   facecolors=colors if filled else "none", edgecolors=colors,
                   linewidths=1.0, zorder=3)
        # Legend proxy in neutral colour
        ax.scatter([], [], marker=case_markers[k % len(case_markers)], s=22,
                   facecolors="0.3" if filled else "none", edgecolors="0.3",
                   label=_case_label(case))

    for yi in y:
        ax.axhline(yi, color="0.9", linewidth=0.6, zorder=0)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in MODELS])
    ax.set_xlabel("CHF margin $q''_\\mathrm{CHF} / q''_\\mathrm{nom}$ [-]")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT_DIR / "steady_margin_dotplot.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# LOFA cases
# -----------------------------------------------------------------------------

def _lofa_results(case):
    G_arr = np.arange(case.G_START, case.G_END + case.G_STEP, case.G_STEP)
    props = sat_props(case.P_PA)
    q_chf = {m: np.array([_eval_safe(m, G, case.D_M, case.L_M,
                                     case.P_PA, case.X_IN, props)
                          for G in G_arr])
             for m in _active_models(case)}
    return G_arr, q_chf


def plot_lofa_margin(case, G_arr, q_chf):
    q_nom = case.Q_NOM_MW * 1e6
    fig, ax = plt.subplots(figsize=(FIG_W, 2.6))

    for m, q in q_chf.items():
        margin = q / q_nom
        label = LABELS[m]
        if m != "LUT-table":
            G_d = _crossing(G_arr, margin)
            label += _dryout_label(G_d, G_arr[0], "G_d", ".0f")
            if G_d is not None:
                ax.plot(G_d, 1.0, marker="x", color=STYLE[m]["color"],
                        markersize=5, markeredgewidth=1.2, zorder=4)
        ax.plot(G_arr, margin, label=label, **STYLE[m])

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mass flux G [kg m$^{-2}$ s$^{-1}$]")
    ax.set_ylabel("CHF margin [-]")
    ax.set_title(f"{_case_label(case)}, P = {case.P_PA / 1e6:.0f} MPa, "
                 f"$q''_\\mathrm{{nom}}$ = {case.Q_NOM_MW} MW/m$^2$")
    ax.invert_xaxis()
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(OUT_DIR / f"{_case_stem(case)}_margin_vs_G.png")
    plt.close(fig)


def plot_lofa_ratio(lofa):
    """
    q_CHF(model) / q_CHF(LUT regression) vs mass flux, one panel per
    LOFA case. `lofa` is a list of (case, G_arr, q_chf) tuples.
    """
    lofa = [item for item in lofa if REFERENCE in item[2]]
    if not lofa:
        return

    fig, axes = plt.subplots(1, len(lofa), figsize=(FIG_W * 1.9, 2.6),
                             sharey=True, squeeze=False)
    for ax, (case, G_arr, q_chf) in zip(axes[0], lofa):
        q_ref = q_chf[REFERENCE]
        for m, q in q_chf.items():
            if m == REFERENCE:
                continue
            ax.plot(G_arr, q / q_ref, label=LABELS[m], **STYLE[m])
        ax.axhline(1.0, color=STYLE[REFERENCE]["color"], linestyle="--",
                   linewidth=0.8, label="LUT 2006 (regression), reference")
        ax.set_xlabel("Mass flux G [kg m$^{-2}$ s$^{-1}$]")
        ax.set_title(_case_label(case))
        ax.invert_xaxis()

    axes[0][0].set_ylabel(r"$q''_\mathrm{CHF} / q''_\mathrm{CHF,\,LUT\,regr}$ [-]")
    axes[0][-1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lofa_ratio_to_lut.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Pressurisation cases
# -----------------------------------------------------------------------------

def plot_pressurisation(case):
    """CHF margin vs pressure at constant heat flux, per model."""
    P_arr = np.linspace(case.P_NOM_PA, case.P_PEAK_PA, case.N_STEPS)
    P_MPa = P_arr / 1e6
    q_nom = case.Q_NOM_MW * 1e6

    fig, ax = plt.subplots(figsize=(FIG_W, 2.6))
    for m in _active_models(case):
        q = np.array([_eval_safe(m, case.G_KGM2S, case.D_M, case.L_M,
                                 P, case.X_IN, sat_props(P))
                      for P in P_arr])
        margin = q / q_nom
        label = LABELS[m]
        if m != "LUT-table":
            P_d = _crossing(P_MPa, margin)
            label += _dryout_label(P_d, P_MPa[0], "P_d", ".2f")
            if P_d is not None:
                ax.plot(P_d, 1.0, marker="x", color=STYLE[m]["color"],
                        markersize=5, markeredgewidth=1.2, zorder=4)
        ax.plot(P_MPa, margin, label=label, **STYLE[m])

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Pressure P [MPa]")
    ax.set_ylabel("CHF margin [-]")
    ax.set_title(f"{_case_label(case)}, G = {case.G_KGM2S:.0f} kg m$^{{-2}}$ s$^{{-1}}$, "
                 f"$q''$ = {case.Q_NOM_MW} MW/m$^2$")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(OUT_DIR / f"{_case_stem(case)}_margin_vs_P.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    cases = _load_cases()
    active = [c for c in cases if getattr(c, 'ACTIVE', False)]

    steady_cases = []
    lofa = []

    for case in active:
        try:
            if _is_lofa(case):
                G_arr, q_chf = _lofa_results(case)
                plot_lofa_margin(case, G_arr, q_chf)
                lofa.append((case, G_arr, q_chf))
            elif _is_pressurisation(case):
                plot_pressurisation(case)
            else:
                steady_cases.append(case)
        except Exception as exc:
            print(f"[ERROR] plotting case '{getattr(case, 'NAME', '?')}' failed: {exc}")

    if steady_cases:
        results = _steady_results(steady_cases)
        write_steady_table(steady_cases, results)
        plot_steady_dotplot(steady_cases, results)
    plot_lofa_ratio(lofa)

    print(f"\nPlots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
