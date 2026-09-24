"""
main.py
=======
CHF Toolkit -- main entry point.

Discovers all case files in the `cases/` directory, loads those with
ACTIVE = True, and evaluates the requested CHF models.

Usage
-----
    python main.py              # run all active cases
    python main.py --list       # list all cases and their ACTIVE status
"""

import sys
import importlib
import importlib.util
from pathlib import Path
import numpy as np

from water_properties import sat_props
from models import (biasi_chf, katto_chf, hewitt_chf,
                     groeneveld_table_chf, groeneveld_regression_chf)


# -----------------------------------------------------------------------------
# Case discovery
# -----------------------------------------------------------------------------

CASES_DIR = Path(__file__).parent / "cases"


def _load_cases():
    """Return a list of case modules found in the cases/ directory."""
    case_files = sorted(CASES_DIR.glob("case_*.py"))
    modules = []
    for path in case_files:
        spec   = importlib.util.spec_from_file_location(path.stem, path)
        mod    = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        modules.append(mod)
    return modules


# -----------------------------------------------------------------------------
# Pretty printing helpers
# -----------------------------------------------------------------------------

def _header(title: str):
    w = 68
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")


def _result_row(model: str, q_chf_W: float, margin: float, extra: str = ""):
    status = "NO DRYOUT" if margin >= 1.0 else "DRYOUT"
    print(f"  {model:<22}  q_CHF = {q_chf_W/1e6:7.3f} MW/m^2  "
          f"margin = {margin:5.3f}  {status}  {extra}")


def _warn(warnings: list):
    for w in warnings:
        print(f"    ! {w}")


# -----------------------------------------------------------------------------
# Single steady-state evaluation
# -----------------------------------------------------------------------------

def _run_steady(case, props):
    _header(case.NAME)
    print(f"  {case.DESCRIPTION}")
    print(f"\n  P = {case.P_PA/1e6:.2f} MPa  |  G = {case.G_KGM2S:.0f} kg/(m^2*s)  "
          f"|  d = {case.D_M*1000:.0f} mm  |  L = {case.L_M:.2f} m  "
          f"|  x_in = {case.X_IN:.3f}")
    print(f"  q_nom = {case.Q_NOM_MW:.2f} MW/m^2\n")

    q_nom = case.Q_NOM_MW * 1e6
    # Inlet subcooling enthalpy (from inlet quality)
    dH_i  = max(0.0, -case.X_IN * props.H_fg)   # J/kg, only if subcooled

    if getattr(case, 'RUN_BIASI', False):
        r = biasi_chf(case.G_KGM2S, case.D_M, case.L_M,
                      case.P_PA, props.H_fg, case.X_IN)
        _result_row("Biasi (1968)", r.q_chf, r.q_chf / q_nom,
                    f"[branch={r.branch}]")
        _warn(r.warnings)

    if getattr(case, 'RUN_KATTO', False):
        r = katto_chf(case.G_KGM2S, case.D_M, case.L_M,
                      props.rho_l, props.rho_v, props.H_fg, props.sigma,
                      dH_i=dH_i)
        _result_row("Katto (1978)", r.q_c, r.q_c / q_nom,
                    f"[regime={r.regime}]")
        _warn(r.warnings)

    if getattr(case, 'RUN_HEWITT', False):
        r = hewitt_chf(case.G_KGM2S, case.D_M, case.L_M,
                       props.rho_l, props.rho_v, props.H_fg, props.sigma,
                       props.eta_l, props.eta_v, x_in=case.X_IN)
        if not np.isnan(r.q_chf):
            _result_row("Hewitt-Govan (1990)", r.q_chf, r.q_chf / q_nom)
        else:
            print("  Hewitt-Govan (1990)    CHF solver did not converge.")
        _warn(r.warnings)

    if getattr(case, 'RUN_GROENEVELD_TABLE', False):
        r = groeneveld_table_chf(case.G_KGM2S, case.D_M, case.L_M,
                                  case.P_PA, props.H_fg, x_in=case.X_IN)
        if r.q_chf is None:
            print(f"  {'LUT (table)':<22}  no result -- {r.warnings[0]}")
        else:
            _result_row("LUT (table)", r.q_chf, r.q_chf / q_nom)
        _warn(r.warnings if r.q_chf is not None else [])

    if getattr(case, 'RUN_GROENEVELD_REGRESSION', False):
        r = groeneveld_regression_chf(case.G_KGM2S, case.D_M, case.L_M,
                                       case.P_PA, props.H_fg, x_in=case.X_IN)
        if r.q_chf is None:
            print(f"  {'LUT (regression)':<22}  no result -- {r.warnings[0]}")
        else:
            _result_row("LUT (regression)", r.q_chf, r.q_chf / q_nom)
        _warn(r.warnings if r.q_chf is not None else [])


# -----------------------------------------------------------------------------
# LOFA quasi-static sweep
# -----------------------------------------------------------------------------

def _run_lofa(case, props):
    _header(case.NAME)
    print(f"  {case.DESCRIPTION}")
    print(f"\n  P = {case.P_PA/1e6:.2f} MPa  |  q = {case.Q_NOM_MW} MW/m^2  "
          f"|  d = {case.D_M*1000:.0f} mm  |  L = {case.L_M:.2f} m")
    print(f"  G sweep: {case.G_START:.0f} -> {case.G_END:.0f} step {case.G_STEP:.0f} kg/(m^2*s)\n")

    G_arr = np.arange(case.G_START, case.G_END + case.G_STEP, case.G_STEP)
    q_nom = case.Q_NOM_MW * 1e6

    # Table header
    models = []
    if getattr(case, 'RUN_KATTO',                False): models.append(("Katto",      "K"))
    if getattr(case, 'RUN_BIASI',                False): models.append(("Biasi",      "B"))
    if getattr(case, 'RUN_GROENEVELD_TABLE',      False): models.append(("LUT-table", "LT"))
    if getattr(case, 'RUN_GROENEVELD_REGRESSION', False): models.append(("LUT-regr",  "LR"))
    if getattr(case, 'RUN_HEWITT',                False): models.append(("Hewitt",     "H"))

    hdr_models = "  ".join(f"{m[0]:>12}" for m in models)
    print(f"  {'G [kg/m^2s]':>12}  {hdr_models}")
    print(f"  {'-'*60}")

    dryout_G = {m[0]: None for m in models}

    for G in G_arr:
        dH_i = max(0.0, -case.X_IN * props.H_fg)
        row = f"  {G:>12.0f}"
        for name, _ in models:
            try:
                if name == "Katto":
                    r = katto_chf(G, case.D_M, case.L_M,
                                  props.rho_l, props.rho_v, props.H_fg, props.sigma,
                                  dH_i=dH_i)
                    q = r.q_c
                elif name == "Biasi":
                    r = biasi_chf(G, case.D_M, case.L_M,
                                  case.P_PA, props.H_fg, case.X_IN)
                    q = r.q_chf
                elif name == "LUT-table":
                    r = groeneveld_table_chf(G, case.D_M, case.L_M,
                                              case.P_PA, props.H_fg, x_in=case.X_IN)
                    q = r.q_chf if r.q_chf is not None else float('nan')
                elif name == "LUT-regr":
                    r = groeneveld_regression_chf(G, case.D_M, case.L_M,
                                                   case.P_PA, props.H_fg, x_in=case.X_IN)
                    q = r.q_chf if r.q_chf is not None else float('nan')
                elif name == "Hewitt":
                    r = hewitt_chf(G, case.D_M, case.L_M,
                                   props.rho_l, props.rho_v, props.H_fg, props.sigma,
                                   props.eta_l, props.eta_v, x_in=case.X_IN)
                    q = r.q_chf

                margin = q / q_nom if not np.isnan(q) else float('nan')
                flag = " DRYOUT" if (not np.isnan(margin) and margin < 1.0) else "         "
                row += f"  {margin:>10.3f}{flag}" if not np.isnan(margin) else f"  {'N/A':>19}"
                if not np.isnan(margin) and margin < 1.0 and dryout_G[name] is None:
                    dryout_G[name] = G
            except Exception:
                row += f"  {'ERR':>19}"
        print(row)

    print()
    for name, _ in models:
        if dryout_G[name] is not None:
            print(f"  - {name}: dryout onset at G = {dryout_G[name]:.0f} kg/(m^2*s)")
        else:
            print(f"  - {name}: no dryout predicted in sweep range.")


# -----------------------------------------------------------------------------
# Pressurisation quasi-static sweep
# -----------------------------------------------------------------------------

def _run_pressurisation(case):
    _header(case.NAME)
    print(f"  {case.DESCRIPTION}")
    print(f"\n  G = {case.G_KGM2S:.0f} kg/(m^2*s)  |  d = {case.D_M*1000:.0f} mm  "
          f"|  L = {case.L_M:.2f} m")
    print(f"  P:  {case.P_NOM_PA/1e6:.2f} -> {case.P_PEAK_PA/1e6:.2f} MPa")
    print(f"  q'' = {case.Q_NOM_MW:.2f} MW/m^2 (constant)\n")

    step_arr = np.linspace(0.0, 1.0, case.N_STEPS)
    dryout_step = {}

    models = []
    if getattr(case, 'RUN_KATTO',                False): models.append("Katto")
    if getattr(case, 'RUN_BIASI',                False): models.append("Biasi")
    if getattr(case, 'RUN_GROENEVELD_TABLE',      False): models.append("LUT-table")
    if getattr(case, 'RUN_GROENEVELD_REGRESSION', False): models.append("LUT-regr")
    if getattr(case, 'RUN_HEWITT',                False): models.append("Hewitt")
    for m in models:
        dryout_step[m] = None

    hdr_m = "  ".join(f"{m:>12}" for m in models)
    print(f"  {'step':>6}  {'P [MPa]':>9}  {hdr_m}")
    print(f"  {'-'*70}")

    for step in step_arr:
        P_step = case.P_NOM_PA  + step * (case.P_PEAK_PA  - case.P_NOM_PA)
        q_nom  = case.Q_NOM_MW * 1e6
        props_step = sat_props(P_step)
        dH_i    = max(0.0, -case.X_IN * props_step.H_fg)
        row = f"  {step:6.3f}  {P_step/1e6:9.3f}"

        for name in models:
            try:
                if name == "Katto":
                    r = katto_chf(case.G_KGM2S, case.D_M, case.L_M,
                                  props_step.rho_l, props_step.rho_v,
                                  props_step.H_fg, props_step.sigma, dH_i=dH_i)
                    q = r.q_c
                elif name == "Biasi":
                    r = biasi_chf(case.G_KGM2S, case.D_M, case.L_M,
                                  P_step, props_step.H_fg, case.X_IN)
                    q = r.q_chf
                elif name == "LUT-table":
                    r = groeneveld_table_chf(case.G_KGM2S, case.D_M, case.L_M,
                                              P_step, props_step.H_fg, x_in=case.X_IN)
                    q = r.q_chf if r.q_chf is not None else float('nan')
                elif name == "LUT-regr":
                    r = groeneveld_regression_chf(case.G_KGM2S, case.D_M, case.L_M,
                                                   P_step, props_step.H_fg, x_in=case.X_IN)
                    q = r.q_chf if r.q_chf is not None else float('nan')
                elif name == "Hewitt":
                    r = hewitt_chf(case.G_KGM2S, case.D_M, case.L_M,
                                   props_step.rho_l, props_step.rho_v,
                                   props_step.H_fg, props_step.sigma,
                                   props_step.eta_l, props_step.eta_v, x_in=case.X_IN)
                    q = r.q_chf

                margin = q / q_nom if not np.isnan(q) else float('nan')
                flag = " DRYOUT" if (not np.isnan(margin) and margin < 1.0) else "         "
                row += f"  {margin:>10.3f}{flag}" if not np.isnan(margin) else f"  {'N/A':>19}"
                if not np.isnan(margin) and margin < 1.0 and dryout_step[name] is None:
                    dryout_step[name] = step
            except Exception:
                row += f"  {'ERR':>19}"
        print(row)

    print()
    for name in models:
        if dryout_step[name] is not None:
            print(f"  - {name}: dryout onset at step = {dryout_step[name]:.3f}")
        else:
            print(f"  - {name}: no dryout predicted across full transient.")


# -----------------------------------------------------------------------------
# Case router
# -----------------------------------------------------------------------------

def _is_lofa(case) -> bool:
    return hasattr(case, 'G_START') and hasattr(case, 'G_END')


def _is_pressurisation(case) -> bool:
    return hasattr(case, 'P_NOM_PA') and hasattr(case, 'P_PEAK_PA')


def run_case(case):
    if _is_lofa(case):
        props = sat_props(case.P_PA)
        _run_lofa(case, props)
    elif _is_pressurisation(case):
        _run_pressurisation(case)
    else:
        props = sat_props(case.P_PA)
        _run_steady(case, props)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    cases = _load_cases()

    if "--list" in sys.argv:
        print("\nAvailable cases:")
        for c in cases:
            status = "ACTIVE" if getattr(c, 'ACTIVE', False) else "skipped"
            print(f"  [{status:>7}]  {getattr(c, 'NAME', c.__name__)}")
        return

    active = [c for c in cases if getattr(c, 'ACTIVE', False)]
    skipped = len(cases) - len(active)

    print(f"\nCHF Toolkit -- {len(active)} case(s) active, {skipped} skipped.")

    for case in active:
        try:
            run_case(case)
        except Exception as exc:
            print(f"\n  [ERROR] Case '{getattr(case, 'NAME', '?')}' failed: {exc}")

    print(f"\n{'='*68}")
    print("  Done.")


if __name__ == "__main__":
    main()
