"""
cases/case_3a_lofa_d8mm.py
============================
Case 3a -- Loss-of-Flow Accident (LOFA): d = 8 mm
Quasi-static G sweep: 1500 -> 300 kg/(m^2*s) at constant heat flux.

Flag
----
ACTIVE = True   ->  case is executed by main.py
ACTIVE = False  ->  case is skipped
"""

# -- Execution flag ------------------------------------------------------------
ACTIVE = True

# -- Case metadata -------------------------------------------------------------
NAME        = "Case 3a -- LOFA quasi-static sweep (d = 8 mm)"
DESCRIPTION = (
    "Loss-of-flow transient modelled as a quasi-static sequence of steady states.  "
    "Mass flux decreases from nominal to 300 kg/(m^2*s); heat flux is held constant.  "
    "Dryout onset is the first step where CHF margin < 1."
)

# -- Geometry ------------------------------------------------------------------
D_M = 0.008      # m
L_M = 3.8        # m

# -- Operating conditions (constant during transient) -------------------------
P_PA     = 7.0e6 # Pa
X_IN     = -0.041
Q_NOM_MW = 1.1   # MW/m^2  (held constant)

# -- LOFA sweep parameters -----------------------------------------------------
G_START  = 1500.0  # kg/(m^2*s)  initial mass flux
G_END    = 300.0   # kg/(m^2*s)  final mass flux
G_STEP   = -50.0   # kg/(m^2*s)  step size (negative = decreasing)

# -- Models to run -------------------------------------------------------------
RUN_BIASI      = True
RUN_KATTO      = True
RUN_HEWITT     = False   # computationally expensive; disable for sweep
RUN_GROENEVELD_TABLE      = True
RUN_GROENEVELD_REGRESSION = True
