# GVWD — Glide-Vehicle Wedge-Derived

Engineering-realistic hypersonic-glide-vehicle generator and panel-method
aero/heating analyser. Five geometry modes, single shared
`GVWDRunConfig`, deterministic provenance via SHA-256 hashing of every
run, integrated PyQt5 tab in the parent waverider GUI, 146-test
self-validation suite (`pytest gvwd/tests/`).

```
gvwd/
├── thermo/        # oblique-shock, tangent-wedge, Newtonian, Oswatitsch
├── geometry/      # 5 modes + fins, mesh + volume primitives
├── aero/          # panel method, viscous strip theory, sweep driver
├── heating/       # Fay-Riddell, Tauber-Sutton 1991, Eckert distributed
├── export/        # STL (pure-Python), STEP/IGES (cadquery / OCP)
├── io/            # YAML config, run-artifact writer, SHA-256 provenance
├── viz/           # matplotlib plotting helpers (sweep heatmaps, etc.)
├── tests/         # 146 tests, all green
├── examples/      # 6 numbered Python examples + 5 YAML configs
└── docs/
    ├── methodology.md   ← physics, conventions, scope limits
    ├── validation.md    ← what is checked, against what reference
    └── glossary.md      ← symbols, terms, acronyms
```

## Install

GVWD lives as a top-level package in the parent
`GUI_Claude_AI/Github` repository. There is no separate install
step — `pip install -r ../requirements.txt` from the parent's root
covers it.

Optional extras (silently degrade when absent):

* `cadquery` — STEP / IGES export
* `h5py` — HDF5-format sweep output
* `matplotlib` — sweep heatmap plots
* `PyQt5` — GUI tab

## Quick start

### As a library

```python
import math
from gvwd.geometry import EngineeringFlat
from gvwd.aero.coefficients import aero_coefficients_full

geom = EngineeringFlat(
    M_design=15.0, theta_fore=math.radians(8.0),
    Lambda=math.radians(75.0),
    L_fore=2.5, L_center=1.5, b_base=0.5, h_base=0.4,
)
out = aero_coefficients_full(
    geom.mesh, M_inf=15.0, alpha_rad=0.0,
    altitude_km=30.0, T_w=1500.0, Re_x_tr=1.0e6,
)
print(f"CL={out['CL']:.4f}  CD={out['CD_total']:.4f}  L/D={out['LD']:.3f}")
```

### From a YAML config

```python
from gvwd.examples._runner import run_from_config

result = run_from_config("gvwd/examples/configs/engineering_flat_htv2_class.yaml")
print(f"Wrote artifacts to: {result['artifact'].base_dir}")
```

### From the GUI

Run the parent app:

```
python waverider_gui.py
```

Select the **GVWD Waverider** tab (12th tab). The combo box at the
top selects between the 5 modes; the parameter pane changes
accordingly.

## Five geometry modes

| Mode                   | Layer        | Description                                |
|------------------------|--------------|--------------------------------------------|
| `engineering_flat`     | engineering  | HTV-2-archetype flat-bottom                |
| `engineering_shallow_v`| engineering  | Same with a dihedral lower surface         |
| `caret`                | reference    | Nonweiler caret (analytic-truth)           |
| `flat_delta`           | reference    | Flat-bottomed delta (analytic-truth)       |
| `multi_wedge`          | reference    | Oswatitsch `n`-ramp (analytic-truth)       |

The reference modes are deliberately small parameter-space objects
that have closed-form V and known L/D properties; they exist to
anchor the panel method against analytic truth. The engineering
modes are what the user actually designs around.

See `docs/methodology.md` for the physics rationale and `docs/glossary.md`
for symbol definitions.

## Examples

`gvwd/examples/` ships six standalone scripts:

```
01_htv2_class_flat_bottom.py
02_fattah2_class_with_fins.py
03_caret_reference_M6.py
04_flat_delta_reference_M5.py
05_two_ramp_oswatitsch_M5.py
06_mach_alpha_sweep_engineering.py
```

Each writes a complete run-artifact directory under `results/`.

## Testing

```
pytest gvwd/tests/
```

→ 146 tests pass in ~8 s. Coverage breakdown is in
`docs/validation.md`. The test suite is the validation gate — a
clean checkout plus this command constitutes a full revalidation.

## GUI integration

`gvwd_waverider_tab.py` (at the repository root, not inside `gvwd/`)
is the PyQt5 wrapper. It exposes:

* mode selector (5 options),
* mode-specific parameter pages (5 pages stacked in a `QStackedWidget`),
* common groups: fins (engineering modes only), atmosphere, sweep
  grid,
* threaded sweep worker (`QThread` `_SweepWorker`),
* STL / STEP export,
* on-design + sweep readouts that match the standalone library
  to machine precision (verified per Phase 7 DoD).

## Provenance

Every run produces a `config_sha256.txt` deterministic hash of the
input config. Re-running the same YAML produces the same hash and
bit-identical geometry.stl output. See `docs/methodology.md` §7.

## Limits of validity

GVWD is an **engineering** tool, not a CFD solver. The panel method
is tangent-wedge with Newtonian fall-back, ideal-gas (γ=1.4),
no shock interactions, no real-gas chemistry, base-flow `Cp = 0`.
Use the sweep grid to bracket trends; do **not** treat absolute
drag predictions as <5%-accurate. See `docs/methodology.md` §4
for the full scope statement.

## Documentation index

* [`docs/methodology.md`](docs/methodology.md) — physics, sign
  conventions, panel-method derivation, scope limits.
* [`docs/validation.md`](docs/validation.md) — what is checked
  by `pytest gvwd/tests/`, against what reference, with what
  tolerance.
* [`docs/glossary.md`](docs/glossary.md) — symbols, terms,
  acronyms.

## Related projects

* `pswr/` — Plasma-Sheath-Shaped Variable-Wedge Waverider, the
  sibling project that shares the same parent GUI shell.
* `waverider_generator/`, `shadow_waverider.py`,
  `planar_waverider.py` — the three pre-existing waverider
  generation methods in the parent GUI. GVWD does not depend on
  these; the parent GUI hosts all four.
