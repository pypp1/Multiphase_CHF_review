# CHF Toolkit

Critical Heat Flux (CHF) calculation toolkit for water-cooled vertical tubes.
Implements four independent prediction methods covering different physical mechanisms.

## Models implemented

| File | Reference | Method |
|---|---|---|
| `models/biasi1968.py` | Biasi et al. (1968), *J. Nucl. Energy* 22, 705 | Empirical correlation (two-line envelope) |
| `models/katto1978.py` | Katto (1978), *IJHMT* 21, 1527 | Dimensionless generalised correlation (4 regimes) |
| `models/hewitt1990.py` | Hewitt & Govan (1990), *IJHMT* 33, 229 | Phenomenological annular-flow dryout model |
| `models/groeneveld2006.py` | Groeneveld et al. (2007), *NED* 237, 1909 | 2006 CHF look-up table + heat balance method |

## Repository layout

```
chf_toolkit/
|
|-- main.py                   entry point: discovers and runs active cases
|-- water_properties.py       saturated water/steam properties (IAPWS-IF97, 4-10 MPa)
|-- requirements.txt
|
|-- models/
|   |-- __init__.py
|   |-- biasi1968.py
|   |-- katto1978.py
|   |-- hewitt1990.py
|   `-- groeneveld2006.py
|
`-- cases/
    |-- __init__.py
    |-- case_1a_steady_lut_reference.py
    |-- case_2a_steady_bwr_geometry.py
    |-- case_3a_lofa_d8mm.py
    |-- case_3b_lofa_d11mm.py
    `-- case_4a_pressurisation.py
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
Each case also has per-model flags (`RUN_BIASI`, `RUN_KATTO`, etc.) to
control which correlations are evaluated.

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
RUN_BIASI = True; RUN_KATTO = True; RUN_HEWITT = True; RUN_GROENEVELD = True
```

For transient sweeps, see `case_3a_lofa_d8mm.py` (LOFA) or
`case_4a_pressurisation.py` (pressurisation).

## Water properties validity

`water_properties.py` covers **4-10 MPa** by piecewise linear interpolation
on IAPWS-IF97 anchor data.  Extrapolation outside this range will produce
results but accuracy degrades rapidly.

## Known limitations

- Biasi (1968): uses CGS units internally; requires subcooled inlet (X_in < 0).
- Katto (1978): subcooling factor K is unavailable in the N-regime (Section 6.4 of the paper).
- Hewitt-Govan (1990): property set is hardcoded for steam-water; requires `eta_l`, `eta_v`.
- Groeneveld LUT (2006): only the P = 7 000 kPa slice is embedded.
  Other pressures fall back to nearest-neighbour with a warning.
