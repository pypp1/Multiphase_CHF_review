"""
data visualisation/graphs.py
=============================
Generates comparison plots for the CHF toolkit's active cases and saves
them as PNG files into this same folder.

Usage
-----
    python "data visualisation/graphs.py"

Plots produced
--------------
Per pressurisation case (e.g. case_4a):
  - <case>_margin_vs_t.png       CHF margin vs transient step, per model
  - <case>_qchf_vs_t.png         Predicted CHF & applied heat flux vs t
  - <case>_qchf_vs_pressure.png  Predicted CHF vs pressure, per model

Per LOFA case (e.g. case_3a, case_3b):
  - <case>_margin_vs_G.png       CHF margin vs mass flux, per model

Steady-state cases (e.g. case_1a, case_2a):
  - steady_margin_comparison.png Bar chart of CHF margin, model x case

All cases:
  - parity_<A>_vs_<B>.png         Matched-condition scatter of model A vs B
                                   predicted CHF (same G, P, x_in), colored
                                   by case, with a 45deg agreement line.
                                   One file per model pair with >=2 shared points.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from water_properties import sat_props
from models import biasi_chf, katto_chf, hewitt_chf, groeneveld_chf
from main import _load_cases, _is_lofa, _is_pressurisation

OUT_DIR = Path(__file__).resolve().parent

# Consistent colours per model
COLORS = {
    "Katto":      "#1f77b4",
    "Biasi":      "#ff7f0e",
    "LUT":        "#2ca02c",
    "Hewitt":     "#d62728",
}


def _active_models(case):
    models = []
    if getattr(case, 'RUN_KATTO',      False): models.append("Katto")
    if getattr(case, 'RUN_BIASI',      False): models.append("Biasi")
    if getattr(case, 'RUN_GROENEVELD', False): models.append("LUT")
    if getattr(case, 'RUN_HEWITT',     False): models.append("Hewitt")
    return models


def _eval_model(name, G, D_M, L_M, P_PA, X_IN, props):
    dH_i = max(0.0, -X_IN * props.H_fg)
    if name == "Katto":
        r = katto_chf(G, D_M, L_M, props.rho_l, props.rho_v,
                       props.H_fg, props.sigma, dH_i=dH_i)
        return r.q_c
    elif name == "Biasi":
        r = biasi_chf(G, D_M, L_M, P_PA, props.H_fg, X_IN)
        return r.q_chf
    elif name == "LUT":
        r = groeneveld_chf(G, D_M, L_M, P_PA, props.H_fg, x_in=X_IN)
        return r.q_chf
    elif name == "Hewitt":
        r = hewitt_chf(G, D_M, L_M, props.rho_l, props.rho_v,
                        props.H_fg, props.sigma, props.eta_l, props.eta_v,
                        x_in=X_IN)
        return r.q_chf
    raise ValueError(name)


# -----------------------------------------------------------------------------
# Pressurisation case plots
# -----------------------------------------------------------------------------

def plot_pressurisation(case, parity_store):
    models = _active_models(case)
    t_arr = np.linspace(0.0, 1.0, case.N_STEPS)

    P_arr = case.P_NOM_PA + t_arr * (case.P_PEAK_PA - case.P_NOM_PA)
    q_arr = (case.Q_NOM_MW + t_arr * (case.Q_PEAK_MW - case.Q_NOM_MW)) * 1e6

    q_chf = {m: np.full_like(t_arr, np.nan) for m in models}
    margin = {m: np.full_like(t_arr, np.nan) for m in models}

    stem = _case_stem(case)

    for i, t in enumerate(t_arr):
        props_t = sat_props(P_arr[i])
        point = {"case": stem}
        for m in models:
            try:
                q = _eval_model(m, case.G_KGM2S, case.D_M, case.L_M,
                                 P_arr[i], case.X_IN, props_t)
            except Exception:
                q = np.nan
            q_chf[m][i] = q
            if not np.isnan(q):
                margin[m][i] = q / q_arr[i]
                point[m] = q
        parity_store.append(point)

    # margin vs t
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        ax.plot(t_arr, margin[m], label=m, color=COLORS.get(m), marker="o", ms=3)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="margin = 1 (dryout)")
    ax.set_xlabel("Transient parameter t [-]")
    ax.set_ylabel("CHF margin [-]")
    ax.set_title(f"{case.NAME}\nCHF margin vs transient progress")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}_margin_vs_t.png", dpi=150)
    plt.close(fig)

    # q_CHF (predicted) & applied heat flux vs t
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        ax.plot(t_arr, q_chf[m] / 1e6, label=f"q_CHF ({m})", color=COLORS.get(m))
    ax.plot(t_arr, q_arr / 1e6, label="applied q''", color="black", linestyle="--")
    ax.set_xlabel("Transient parameter t [-]")
    ax.set_ylabel("Heat flux [MW/m^2]")
    ax.set_title(f"{case.NAME}\nPredicted CHF vs applied heat flux")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}_qchf_vs_t.png", dpi=150)
    plt.close(fig)

    # q_CHF vs pressure
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        ax.plot(P_arr / 1e6, q_chf[m] / 1e6, label=m, color=COLORS.get(m), marker="o", ms=3)
    ax.set_xlabel("Pressure [MPa]")
    ax.set_ylabel("Predicted CHF [MW/m^2]")
    ax.set_title(f"{case.NAME}\nPredicted CHF vs pressure")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}_qchf_vs_pressure.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# LOFA case plots
# -----------------------------------------------------------------------------

def plot_lofa(case, parity_store):
    models = _active_models(case)
    G_arr = np.arange(case.G_START, case.G_END + case.G_STEP, case.G_STEP)
    q_nom = case.Q_NOM_MW * 1e6
    props = sat_props(case.P_PA)

    margin = {m: np.full_like(G_arr, np.nan, dtype=float) for m in models}
    stem = _case_stem(case)

    for i, G in enumerate(G_arr):
        point = {"case": stem}
        for m in models:
            try:
                q = _eval_model(m, G, case.D_M, case.L_M, case.P_PA, case.X_IN, props)
            except Exception:
                q = np.nan
            if not np.isnan(q):
                margin[m][i] = q / q_nom
                point[m] = q
        parity_store.append(point)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        ax.plot(G_arr, margin[m], label=m, color=COLORS.get(m), marker="o", ms=3)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="margin = 1 (dryout)")
    ax.set_xlabel("Mass flux G [kg/(m^2*s)]")
    ax.set_ylabel("CHF margin [-]")
    ax.set_title(f"{case.NAME}\nCHF margin vs mass flux (LOFA)")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}_margin_vs_G.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Steady-state cases: bar chart comparison
# -----------------------------------------------------------------------------

def plot_steady_comparison(steady_cases, parity_store):
    if not steady_cases:
        return

    all_models = ["Katto", "Biasi", "LUT", "Hewitt"]
    case_labels = [c.NAME for c in steady_cases]
    margins = {m: [] for m in all_models}

    for case in steady_cases:
        props = sat_props(case.P_PA)
        q_nom = case.Q_NOM_MW * 1e6
        active = _active_models(case)
        point = {"case": _case_stem(case)}
        for m in all_models:
            if m not in active:
                margins[m].append(np.nan)
                continue
            try:
                q = _eval_model(m, case.G_KGM2S, case.D_M, case.L_M,
                                 case.P_PA, case.X_IN, props)
            except Exception:
                q = np.nan
            margins[m].append(q / q_nom if not np.isnan(q) else np.nan)
            if not np.isnan(q):
                point[m] = q
        parity_store.append(point)

    x = np.arange(len(case_labels))
    width = 0.8 / len(all_models)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(all_models):
        vals = margins[m]
        ax.bar(x + i * width, vals, width, label=m, color=COLORS.get(m))
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(x + width * (len(all_models) - 1) / 2)
    ax.set_xticklabels(case_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("CHF margin [-]")
    ax.set_title("Steady-state CHF margin comparison across models and cases")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "steady_margin_comparison.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Cross-model parity plots (matched conditions, all cases combined)
# -----------------------------------------------------------------------------

def plot_model_parity(parity_store):
    if not parity_store:
        return

    all_models = ["Katto", "Biasi", "LUT", "Hewitt"]
    case_names = sorted({pt["case"] for pt in parity_store})
    cmap = plt.get_cmap("tab10")
    case_colors = {c: cmap(i % 10) for i, c in enumerate(case_names)}

    for model_a, model_b in combinations(all_models, 2):
        xs, ys, colors, labels_seen = [], [], [], set()
        for pt in parity_store:
            if model_a in pt and model_b in pt:
                xs.append(pt[model_a] / 1e6)
                ys.append(pt[model_b] / 1e6)
                colors.append(case_colors[pt["case"]])
                labels_seen.add(pt["case"])

        if len(xs) < 2:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 6))
        for case in sorted(labels_seen):
            mask = [c == case_colors[case] for c in colors]
            cx = [x for x, m in zip(xs, mask) if m]
            cy = [y for y, m in zip(ys, mask) if m]
            ax.scatter(cx, cy, label=case, color=case_colors[case], alpha=0.8, s=25)

        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, color="k", linestyle="--", linewidth=1, label="perfect agreement")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{model_a} predicted CHF [MW/m^2]")
        ax.set_ylabel(f"{model_b} predicted CHF [MW/m^2]")
        ax.set_title(f"Matched-condition parity: {model_a} vs {model_b}\n"
                      f"(same G, P, x_in -- points above the line: {model_b} more conservative)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"parity_{model_a.lower()}_vs_{model_b.lower()}.png", dpi=150)
        plt.close(fig)


def _case_stem(case):
    path = getattr(case, "__file__", None)
    if path:
        return Path(path).stem
    return case.NAME.split("--")[0].strip().lower().replace(" ", "_")


def main():
    cases = _load_cases()
    active = [c for c in cases if getattr(c, 'ACTIVE', False)]

    parity_store = []
    steady_cases = []

    for case in active:
        try:
            if _is_lofa(case):
                plot_lofa(case, parity_store)
            elif _is_pressurisation(case):
                plot_pressurisation(case, parity_store)
            else:
                steady_cases.append(case)
        except Exception as exc:
            print(f"[ERROR] plotting case '{getattr(case, 'NAME', '?')}' failed: {exc}")

    plot_steady_comparison(steady_cases, parity_store)
    plot_model_parity(parity_store)

    print(f"Plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
