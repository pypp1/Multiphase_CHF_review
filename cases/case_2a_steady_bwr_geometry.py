"""
cases/case_2a_steady_bwr_geometry.py
=====================================
Case 2a -- Steady-state: BWR geometry
P = 7 MPa, G = 1500 kg/(m^2*s), d = 11 mm

Flag
----
ACTIVE = True   ->  case is executed by main.py
ACTIVE = False  ->  case is skipped
"""

# -- Execution flag ------------------------------------------------------------
ACTIVE = True

# -- Case metadata -------------------------------------------------------------
NAME        = "Case 2a -- Steady-state, BWR geometry (d = 11 mm)"
DESCRIPTION = (
    "Same operating conditions as Case 1a but with the larger BWR rod diameter.  "
    "Used to assess the diameter effect on CHF margin."
)

# -- Geometry ------------------------------------------------------------------
D_M = 0.011      # m  --  inner tube diameter
L_M = 3.8        # m  --  heated length

# -- Operating conditions ------------------------------------------------------
P_PA    = 7.0e6  # Pa
G_KGM2S = 1500.0 # kg/(m^2*s)
X_IN    = -0.041 # inlet quality
Q_NOM_MW = 0.7   # MW/m^2  nominal heat flux

# -- Models to run -------------------------------------------------------------
RUN_BIASI      = True
RUN_KATTO      = True
RUN_HEWITT     = True
RUN_GROENEVELD_TABLE      = True
RUN_GROENEVELD_REGRESSION = True
