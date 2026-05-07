# WADOS — Waverider Aerodynamic Design & Optimization System

[![tests](https://github.com/aklothakis/wados/actions/workflows/tests.yml/badge.svg)](https://github.com/aklothakis/wados/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyQt5 desktop application for parametric design, aerodynamic analysis, and
multi-objective optimization of hypersonic waveriders. WADOS is a hub: each
waverider design method plugs in as its own tab, sharing a common 3-D
visualization, mesh export, and aero-evaluation backbone.

---

## Quick start

```bash
git clone https://github.com/aklothakis/wados.git
cd wados
pip install -r requirements.txt
python waverider_gui.py
```

Verify the install:

```bash
python check_setup.py
pytest gvwd/tests/ pysagas/tests/
```

Optional dependencies (each silently degrades when absent — the GUI
checks at import time):

| Package      | Enables                                                       |
|--------------|---------------------------------------------------------------|
| `gmsh`       | High-quality surface meshing for the aero solver              |
| `torch`      | PyTorch surrogate models in the surrogate / off-design tabs   |
| `anthropic`  | Claude AI assistant tab                                       |
| `python-docx`| Report export                                                 |

---

## Methods

WADOS ships with the following waverider design methods. Each method has
its own tab in the GUI; switch between them from the top tab bar.

### Mature

| Tab                    | Method                                                                                  |
|------------------------|-----------------------------------------------------------------------------------------|
| **OC Waverider**       | Osculating-cone, Taylor–Maccoll conical-flow design point.                              |
| **Cone-derived (SHADOW)** | Polynomial leading edge projected onto a conical shock; gradient-based optimizer.   |
| **Planar Waverider**   | 9-parameter analytical model (Jessen et al. 2026) with closed-form aero evaluation.     |
| **VMOF Waverider**     | Variable-Mach Osculating Flowfield (Liu et al. 2019).                                   |
| **MFOF Waverider**     | Multi-Flowfield Osculating Framework — refactored production version of the VMOF method. |
| **GVWD Waverider**     | Glide-Vehicle Wedge-Derived (HTV-2 / Fattah-2 / Avangard archetype) — five geometry modes. |
| **PSWR-1 Waverider**   | Plasma-Sheath-Shaped Variable-Wedge waverider — variable-wedge β(y) for plasma-sheath shaping. |

### Experimental / research

These tabs are wired into the GUI but their APIs are not stable. Use with
the understanding that defaults and outputs may change between revisions.

| Tab                          | Method                                                                |
|------------------------------|-----------------------------------------------------------------------|
| **VMN Waverider**            | Variable Mach Number Waverider (Li et al. 2018).                      |
| **VMPLO Waverider**          | Variable-Mach Power-Law Osculating Waverider.                         |
| **Hybrid Waverider**         | Hybrid OC / cone-derived blends.                                      |
| **Liu 2019 Waverider**       | Paper-faithful reference implementation of the Liu et al. 2019 method (used for MFOF equivalence checks). |
| **Multi-Mach Hunter**        | Surrogate-driven multi-Mach design exploration.                       |
| **Off-design Surrogate**     | Pre-trained NN surrogate for off-design coefficient prediction.       |

---

## Pipeline overview

```
 GUI tab  →  geometry generator  →  Mesh (STL)  →  PySAGAS aero  →  CL, CD, L/D, Cm
                       ↓
                   STEP / STL export (CadQuery)
                       ↓
                   NSGA-II optimization (pymoo)  /  surrogate ensemble
```

- **Geometry generators** live in dedicated subpackages (`gvwd/`, `pswr/`,
  `liu2019/`, `mfof/`, `waverider_generator/`) or as standalone modules
  (`planar_waverider.py`, `shadow_waverider.py`).
- **Aero solver**: vendored [`pysagas/`](pysagas/), an oblique-shock + Prandtl–Meyer
  panel method.
- **Optimization**: `optimization_engine.py` (NSGA-II via pymoo) and
  `surrogate_tab.py` / `ai_surrogate.py` (sklearn / PyTorch ensembles).
- **Long-running work** runs in `QThread` workers — the GUI never blocks.

---

## Repository layout

```
wados/
├── waverider_gui.py            # Main GUI entry point
├── *_tab.py                    # Per-method tab widgets
├── gvwd/                       # Glide-Vehicle Wedge-Derived library
├── pswr/                       # Plasma-Sheath Variable-Wedge library
├── liu2019/  mfof/             # Variable-Mach osculating-flowfield libraries
├── waverider_generator/        # Osculating-cone generator
├── pysagas/                    # Vendored aero solver (panel method)
├── surrogate_model/            # Pre-trained ensemble surrogates (.pkl)
├── examples/                   # Example scripts and YAML configs
├── docs/                       # Additional documentation
├── Optimization_Guide.md
├── Surrogate_Optimization_Tutorial.md
├── requirements.txt
└── LICENSE
```

---

## Coordinate system

A common convention across all methods:

- `x` = streamwise (nose → base)
- `y` = spanwise (or transverse, depending on method — check the
  method's own docstring)
- `z` = vertical
- Origin at the waverider tip
- STEP export scale factor: `1000` (m → mm)

---

## Documentation

- [`Optimization_Guide.md`](Optimization_Guide.md) — NSGA-II setup, pymoo
  configuration, multi-objective strategies.
- [`Surrogate_Optimization_Tutorial.md`](Surrogate_Optimization_Tutorial.md) — building
  and using surrogate models for fast design exploration.
- [`gvwd/docs/`](gvwd/docs/) — GVWD library: validation suite, methodology,
  glossary.
- [`docs/`](docs/) — additional notes and design references.

---

## Testing

```bash
# GVWD library — 146 tests, ~10 s
pytest gvwd/tests/

# Vendored PySAGAS solver
pytest pysagas/tests/
```

CI runs both suites on every push and pull-request via
[GitHub Actions](.github/workflows/tests.yml).

---

## Acknowledgments

WADOS bundles a vendored copy of [PySAGAS](https://github.com/kieran-mackle/hypysagas)
(distributed on PyPI as `hypysagas`) for its panel-method aero solver.
The vendored copy guarantees the GUI works against the exact internals
it was developed against; install from PyPI if you only need the solver.

Method references are listed in each method's tab module header. Key
papers:

- Liu et al., *Acta Astronautica* **162** (2019) — variable-Mach
  osculating flowfield method (VMOF / MFOF).
- Li et al. (2018) — variable-Mach-number wide-speed-range design (VMN).
- Sobieczky (1990) / Rodi — osculating-cone framework.
- Nonweiler — caret waverider reference geometry.
- Tauber & Sutton (1991) — convective heating correlation.

---

## License

MIT — see [LICENSE](LICENSE).
