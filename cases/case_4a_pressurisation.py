"""
cases/case_4a_pressurisation.py
================================
Case 4a — Pressurisation transient: d = 8 mm
P rises 7 → 8.5 MPa; heat flux rises 100 → 150 % of nominal.

Flag
----
ACTIVE = True   →  case is executed by main.py
ACTIVE = False  →  case is skipped
"""

# ── Execution flag ────────────────────────────────────────────────────────────
ACTIVE = True

# ── Case metadata ─────────────────────────────────────────────────────────────
NAME        = "Case 4a — Pressurisation transient (d = 8 mm)"
DESCRIPTION = (
    "Quasi-static pressurisation event.  A dimensionless transient parameter "
    "t ∈ [0, 1] drives P from P_NOM to P_PEAK and heat flux from Q_NOM to Q_PEAK.  "
    "Dryout onset is the first step where CHF margin < 1."
)

# ── Geometry ──────────────────────────────────────────────────────────────────
D_M = 0.008      # m
L_M = 3.8        # m

# ── Operating conditions ─────────────────────────────────────────────────────
G_KGM2S  = 1500.0  # kg/(m²·s)  (constant during transient)
X_IN     = -0.041

# ── Pressurisation sweep parameters ──────────────────────────────────────────
P_NOM_PA   = 7.0e6  # Pa  initial pressure
P_PEAK_PA  = 8.5e6  # Pa  peak pressure
Q_NOM_MW   = 1.1    # MW/m²  nominal heat flux
Q_PEAK_MW  = 1.65   # MW/m²  peak heat flux (150 % nominal)
N_STEPS    = 30     # number of quasi-static steps

# ── Models to run ─────────────────────────────────────────────────────────────
RUN_BIASI      = False   #quality
RUN_KATTO      = True
RUN_HEWITT     = True
RUN_GROENEVELD = True
