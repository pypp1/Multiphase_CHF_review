"""
cases/case_3b_lofa_d11mm.py
=============================
Case 3b -- Loss-of-Flow Accident (LOFA): d = 11 mm
Quasi-static G sweep: 1500 -> 300 kg/(m^2*s) at constant heat flux.

Flag
----
ACTIVE = True   ->  case is executed by main.py
ACTIVE = False  ->  case is skipped
"""

# -- Execution flag ------------------------------------------------------------
ACTIVE = True

# -- Case metadata -------------------------------------------------------------
NAME        = "Case 3b -- LOFA quasi-static sweep (d = 11 mm)"
DESCRIPTION = (
    "Identical to Case 3a but with the larger BWR tube diameter.  "
    "Allows direct comparison of the diameter effect during a loss-of-flow event."
)

# -- Geometry ------------------------------------------------------------------
D_M = 0.011      # m
L_M = 3.8        # m

# -- Operating conditions -----------------------------------------------------
P_PA     = 7.0e6
X_IN     = -0.041
Q_NOM_MW = 0.7   # MW/m^2

# -- LOFA sweep parameters ----------------------------------------------------
G_START  = 1500.0
G_END    = 300.0
G_STEP   = -50.0

# -- Models to run ------------------------------------------------------------
RUN_BIASI      = True
RUN_KATTO      = True
RUN_HEWITT     = True
RUN_GROENEVELD_TABLE      = True
RUN_GROENEVELD_REGRESSION = True
