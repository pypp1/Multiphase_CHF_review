"""
cases/case_1a_steady_lut_reference.py
======================================
Case 1a — Steady-state reference: LUT geometry
P = 7 MPa, G = 1500 kg/(m²·s), d = 8 mm

Flag
----
ACTIVE = True   →  case is executed by main.py
ACTIVE = False  →  case is skipped
"""

# ── Execution flag ────────────────────────────────────────────────────────────
ACTIVE = True

# ── Case metadata ─────────────────────────────────────────────────────────────
NAME        = "Case 1a — Steady-state, LUT reference (d = 8 mm)"
DESCRIPTION = (
    "Steady-state CHF prediction at nominal BWR/LUT conditions.  "
    "Tube diameter matches the 8 mm reference tube of the Groeneveld LUT."
)

# ── Geometry ──────────────────────────────────────────────────────────────────
D_M = 0.008      # m  —  inner tube diameter
L_M = 3.8        # m  —  heated length

# ── Operating conditions ──────────────────────────────────────────────────────
P_PA   = 7.0e6   # Pa  —  system pressure
G_KGM2S = 1500.0 # kg/(m²·s)  —  mass flux
X_IN   = -0.041  # —          —  inlet thermodynamic quality (subcooled)
Q_NOM_MW = 1.1   # MW/m²      —  nominal heat flux (for margin calculation)

# ── Models to run  (set each to True/False independently) ────────────────────
RUN_BIASI      = True
RUN_KATTO      = True
RUN_HEWITT     = True
RUN_GROENEVELD = True
