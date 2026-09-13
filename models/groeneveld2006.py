"""
models/groeneveld2006.py
========================
Shared result type for the 2006 Groeneveld CHF look-up table (LUT).

This module previously implemented a single `groeneveld_chf()` function
that silently chose between the raw tabulated data and a pressure
regression depending on whether the requested pressure happened to be
tabulated. That implicit behaviour has been retired in favour of two
explicit, independent models that make the distinction visible to the
caller:

  * models/groeneveld2006_table.py      -- groeneveld_table_chf()
    Raw Appendix B data only; returns no result unless (P, G) are
    exactly tabulated and the converged quality falls on the grid.

  * models/groeneveld2006_regression.py -- groeneveld_regression_chf()
    Polynomial pressure regression only (models/lut_regression.py);
    G and X are continuous.

Both share the `LUTResult` type defined here.

Reference
---------
Groeneveld, D. C. et al. (2007).
"The 2006 CHF look-up table."
Nuclear Engineering and Design, 237, 1909-1922.
"""

from dataclasses import dataclass, field


@dataclass
class LUTResult:
    """
    Output of a Groeneveld 2006 LUT CHF calculation.

    q_chf and x_chf are None when no result could be produced (e.g. the
    requested operating point does not correspond to a tabulated value
    for the table-only model); see `warnings` for the reason.
    """
    q_chf:    float               # W/m^2 -- critical heat flux, or None
    x_chf:    float               # --      quality at CHF, or None
    warnings: list = field(default_factory=list)
