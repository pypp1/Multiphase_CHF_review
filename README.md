# CHF Toolkit

Critical Heat Flux (CHF) calculation toolkit for water-cooled vertical tubes.
Implements independent prediction methods covering different physical mechanisms.

## Models implemented

| File | Reference | Method |
|---|---|---|
| `models/biasi1968.py` | Biasi et al. (1968), *J. Nucl. Energy* 22, 705 | Empirical correlation (two-line envelope) |
| `models/katto1978.py` | Katto (1978), *IJHMT* 21, 1527 | Dimensionless generalised correlation (4 regimes) |
| `models/hewitt1990.py` | Hewitt & Govan (1990), *IJHMT* 33, 229 | Phenomenological annular-flow dryout model |
| `models/groeneveld2006_table.py` | Groeneveld et al. (2007), *NED* 237, 1909 | 2006 CHF look-up table (raw Appendix B data only) + heat balance method |
| `models/groeneveld2006_regression.py` | Groeneveld et al. (2007), *NED* 237, 1909 | Same look-up table, bridged across pressures by per-(G, X) polynomial regression |

The Groeneveld LUT is split into two independent, explicit models instead of
one function that silently interpolates:

- **`groeneveld_table_chf`** only returns a result when the requested
  pressure and mass flux match a value tabulated in Appendix B exactly (and
  the converged quality lands on the grid); otherwise it reports why no
  result is available. Use it when you need a value traceable to the
  published table with no interpolation involved.
- **`groeneveld_regression_chf`** treats pressure, mass flux and quality as
  continuous: `models/lut_regression.py` fits one degree-2 polynomial per
  (mass flux, quality) grid point across the 15 tabulated pressures, and
  bilinearly interpolates between the surrounding curves for any (G, X).
  Use it for arbitrary operating points (e.g. transient sweeps).

Both share the `LUTResult` type (`models/groeneveld2006.py`) and read the
same raw table data (`models/lut_data.py`, transcribed from Appendix B of
Groeneveld et al. (2007) for all 15 tabulated pressures: 100, 300, 500,
1000, 2000, 3000, 5000, 7000, 10000, 12000, 14000, 16000, 18000, 20000 and
21000 kPa).

## Repository layout

```
chf_toolkit/
|
|-- main.py                      entry point: discovers and runs active cases
|-- validate_lut_regression.py   standalone diagnostic for the LUT regression fit
|-- water_properties.py          saturated water/steam properties (IAPWS-IF97, 4-10 MPa)
|-- requirements.txt
|
|-- models/
|   |-- __init__.py
|   |-- biasi1968.py
|   |-- katto1978.py
|   |-- hewitt1990.py
|   |-- groeneveld2006.py              shared LUTResult type
|   |-- groeneveld2006_table.py        raw-table-only LUT model
|   |-- groeneveld2006_regression.py   regression-bridged LUT model
|   |-- lut_data.py                    raw Appendix B data (all 15 pressures)
|   `-- lut_regression.py              pressure-regression fitting/evaluation/plotting
|
|-- cases/
|   |-- __init__.py
|   |-- case_1a_steady_lut_reference.py
|   |-- case_2a_steady_bwr_geometry.py
|   |-- case_3a_lofa_d8mm.py
|   |-- case_3b_lofa_d11mm.py
|   `-- case_4a_pressurisation.py
|
`-- data visualisation/
    |-- graphs.py                comparison plots for the active cases
    `-- *.png                    generated figures
```

## Usage

```bash
pip install -r requirements.txt

# Run all active cases
python main.py

# List all cases and their active status
python main.py --list
```

## Enabling / disabling cases

Each file in `cases/` has a flag at the top:

```python
ACTIVE = True    # execute this case
ACTIVE = False   # skip this case
```

Set `ACTIVE = False` to exclude a case without deleting it.
Each case also has per-model flags to control which correlations are
evaluated: `RUN_BIASI`, `RUN_KATTO`, `RUN_HEWITT`,
`RUN_GROENEVELD_TABLE`, `RUN_GROENEVELD_REGRESSION`.

## Adding a new case

Create a new file `cases/case_<name>.py` following the template of an
existing case.  At minimum it needs:

```python
ACTIVE = True
NAME   = "My new case"
D_M    = 0.010    # tube diameter [m]
L_M    = 4.0      # heated length [m]
P_PA   = 7.0e6    # pressure [Pa]
G_KGM2S = 1500.0  # mass flux [kg/(m^2*s)]
X_IN   = -0.05    # inlet quality
Q_NOM_MW = 1.0    # nominal heat flux [MW/m^2]
RUN_BIASI = True; RUN_KATTO = True; RUN_HEWITT = True
RUN_GROENEVELD_TABLE = True; RUN_GROENEVELD_REGRESSION = True
```

For transient sweeps, see `case_3a_lofa_d8mm.py` (LOFA) or
`case_4a_pressurisation.py` (pressurisation).

## Water properties validity

`water_properties.py` covers **4-10 MPa** by piecewise linear interpolation
on IAPWS-IF97 anchor data.  Extrapolation outside this range will produce
results but accuracy degrades rapidly.

## Validating the LUT regression fit

`validate_lut_regression.py` is a standalone diagnostic, independent of the
main analysis pipeline:

```bash
python validate_lut_regression.py
```

It fits the pressure-regression curves, prints a per-pressure fit-error
table (MAE, RMSE, Bias, MARD, RMSRE, P95) comparing the regression against
the tabulated Appendix B data, and saves three check figures to `results/`:

- `lut_regression_check.png` -- (a) G-X map of the MARD of each regression
  curve over all tabulated pressures (discrete classes, no-fit cells
  hatched); (b) box plot of the relative error at each pressure.
- `lut_regression_operating_zone.png` -- the same map with the operating
  points (G, X_CHF) of the active cases and the cells they interpolate from.
- `lut_regression_slice_P7000_G1500.png` -- tabulated vs. regression CHF
  and signed relative error along x at P = 7000 kPa, G = 1500 kg/(m^2*s).

## Known limitations

- Biasi (1968): uses CGS units internally; requires subcooled inlet (X_in < 0).
- Katto (1978): subcooling factor K is unavailable in the N-regime (Section 6.4 of the paper).
- Hewitt-Govan (1990): property set is hardcoded for steam-water; requires `eta_l`, `eta_v`.
- Groeneveld LUT, table model (`groeneveld_table_chf`): only produces a
  result at the 15 pressures and mass fluxes actually tabulated in
  Appendix B, and only when the converged quality falls within half a
  grid step of a tabulated value; otherwise it reports no result rather
  than interpolating.
- Groeneveld LUT, regression model (`groeneveld_regression_chf`): the
  degree-2 polynomial fit across the full 100-21000 kPa pressure range is
  only a rough approximation of the true (non-monotonic) CHF-vs-pressure
  behaviour; see `validate_lut_regression.py`'s output for the actual
  fit error at each pressure before trusting it far from a tabulated value.
