# GVWD Methodology

## 1. Design intent

GVWD (Glide-Vehicle Wedge-Derived) is a parametric library for
generating and analysing engineering-realistic hypersonic glide
vehicles in the HTV-2 / Fattah-2 / Avangard archetype, using
classical wedge-derived primitives as the analytic-reference layer.

The design priority is *transparent physics*: every coefficient
that comes out of the library should be traceable to a closed-form
expression or a panel-method integral that a researcher can
reproduce with pencil and paper given the geometry. CFD-grade
accuracy is **not** a goal. Calibration against CFD or arc-jet
data is left to the user; the library provides the engineering
substrate.

The five geometry modes split into two layers:

* **Reference layer** (analytic-truth): caret (Nonweiler), flat
  delta, multi-wedge / Oswatitsch. These are primarily there so
  the panel method has known closed-form answers to anchor against.
* **Engineering layer** (the actual product): flat-bottom and
  shallow-V variants of an HTV-2-style fore-wedge + box centerbody
  + optional fins. Both have a finite base, a finite chordwise
  extent, and pass `η_V > 0.25` for typical glide-vehicle inputs.

## 2. Coordinate system

`x` streamwise (nose → base), `y` transverse / vertical, `z`
spanwise. Origin at the apex (waverider tip). Bilateral symmetry
about the `y = 0` plane is assumed; meshes are stored on the
`z ≥ 0` half (volume / planform integrals multiply by 2).

Freestream body-frame direction at angle of attack α:
`v̂∞ = (cos α, 0, +sin α)` — apex upstream, body x downstream,
positive α is nose-up. Drag force projection is `D = +F · v̂∞`
(in flow direction). This convention was the source of one of the
two persistent sign-convention bugs caught during Phase 4 and is
now locked by `test_engineering_flat_alpha_zero_positive_aero`.

## 3. Geometry construction

All five modes share the same data product: a `Mesh` object with
`vertices: (N, 3) ndarray`, `faces: (M, 3) int ndarray`, and a
metadata dict. The mesh is *closed*: divergence-theorem signed
volume is positive (verified for every mode in `gvwd/tests/`).
Watertightness is the precondition for the `numerical_volume`,
`planform_area_from_mesh`, and `eta_V` calculations to be physical.

Each generator's responsibility is to produce a closed half-body
mesh wound CCW from the outside. This invariant was the source of
the first persistent bug class during Phase 3 development; the
divergence-theorem signed-volume check is the gate that catches
inversions.

### 3.1 Reference primitives

* **Caret.** Lower surface is one ramp at angle θ_d swept Λ; ridge
  on the upper-base centerline. `y_tip = h / tan β` is derived
  from the swept-shock geometry on the design point. The mesh
  has 5 vertices (apex + two wingtips + lower-base centerline +
  upper-base centerline) and 6 outward triangles.
* **Flat delta.** Single flat lower surface, two-cell base, swept
  LE; degenerates to the flat plate at zero sweep. `y_tip = L cos
  (Λ - θ_LE_eff)` is the closed-form spanwise extent.
* **Multi-wedge.** `n` ramps with Oswatitsch equal-strength
  optimization. Recovers the Hammitt 1961 / Mölder 1967 result
  that equalising the per-shock `p02/p01` minimises total
  stagnation-pressure loss for fixed total deflection.

### 3.2 Engineering modes

Both engineering modes share a fore-wedge + center-body
construction. The fore-wedge is the classical waverider lower
surface (single ramp θ_fore swept Λ); the center-body extends
the lower surface into a flat-bottomed box of height `h_base`
and base width `b_base`. The shallow-V variant introduces a
dihedral on the lower surface for additional volumetric
efficiency.

Geometry is parameterised by:

| Parameter      | Meaning                                  |
|----------------|-------------------------------------------|
| `M_design`     | freestream Mach for the on-design shock  |
| `θ_fore`       | ramp angle on the fore-wedge              |
| `Λ`            | LE sweep                                  |
| `L_fore`       | streamwise length of the fore-wedge       |
| `L_center`     | streamwise length of the center-body      |
| `b_base`       | base width                                |
| `h_base`       | base height                               |
| `dihedral`     | shallow-V only                             |
| `r_LE_mm`      | LE radius (used by the heating module)    |
| `r_nose_mm`    | nose-tip radius                           |

Constructors enforce two physical guards: shock-attachment
(`obtain_beta(M_design, θ_fore)` raises if detached) and
geometric `b_base ≤ 2 · y_tip(θ_fore, Λ)` (so the centerbody
does not exceed the spanwise extent of the leading edge).

## 4. Aero — panel method

The aero solver is a tangent-wedge panel method with a Newtonian
fall-back. Each triangle of the surface mesh contributes
independently:

```
θ_local = arcsin(n̂ · v̂∞)              (positive on windward side)
Cp_panel = tangent_wedge(M∞, θ_local)  if θ < θ_max(M∞)
         = modified_newtonian(θ_local) otherwise
```

with shadowed panels (`θ_local ≤ 1e-6 rad`) clamped to `Cp = 0`.
The 1e-6-rad threshold instead of zero handles the floating-point
spillover at α = ±π/2 (`cos(±π/2) ≈ 6e-17`) that previously caused
non-bracketing brentq calls.

Force integration:

```
F = - Σ_panels Cp_panel · q∞ · A_panel · n̂_panel
L = F · ẑ_lift,    D = F · v̂∞
```

`q∞ = ½ γ p∞ M∞²`. The reference area `S_ref` is the planform
projected onto the `xz` plane via `planform_area_from_mesh`, and
`L_ref` is `L_fore + L_center` for engineering modes (chord for
references). All coefficients are returned in this normalisation.

### Friction & viscous drag

`gvwd/aero/viscous.py` adds an Eckert-reference-temperature flat-plate
friction coefficient with Sutherland viscosity and the van Driest II
compressibility correction. Boundary-layer transition is fixed at
`Re_x_tr` (default 1e6); panels with `Re_x < Re_x_tr` use laminar
`Cf_lam`, otherwise turbulent `Cf_turb`. The friction integral
contributes `CD_friction` to the `aero_coefficients_full` output;
`CD_total = CD_wave + CD_friction`.

### Limits of validity

* Tangent-wedge breaks down at very high local θ (the Newtonian
  fall-back is engineering-grade, not exact).
* The panel method is inviscid for pressure; friction is added
  separately as a flat-plate strip-theory integral (no pressure /
  shear coupling).
* No 3-D shock interaction. Each panel sees a 2-D oblique-shock
  problem in its own local frame.
* No real-gas chemistry. The library is `γ = 1.4` ideal-gas
  throughout. The `gamma` argument exists but the validation
  envelope is calibrated against the constant-γ choice.
* Base flow is treated as `Cp = 0` (no base-drag model). For
  engineering modes the base panel does contribute `CD_wave`
  through Newtonian fall-back at α ≠ 0.

These limits are why the GUI tab labels engineering-mode results
as "engineering panel-method estimate, not CFD-grade." Use the
sweep grid to bracket trends, not to predict absolute drag to <5%.

## 5. Heating

Two stagnation-point convective correlations are exposed
(`gvwd/heating/`):

* **Fay-Riddell** (1957) — the canonical equation, `q ∝ ρ^0.5
  V^3 R^-0.5`, with `T_w / T_e` enthalpy correction.
* **Tauber-Sutton 1991** — `q ∝ ρ^0.5 V^3.15 R^-0.5` with
  `K = 1.83·10⁻⁴`. This is the form that lands the DoD-§5.4
  numbers (50–200 MW/m² for 1 mm LE at M=15 / h=30 km).

A swept-LE correction `(cos Λ)^(0.5 to 1.0)` is applied for
non-stagnation leading edges, and an Eckert-distributed
heat-flux integrator is provided for total surface-heat-load
estimates.

## 6. Sweep driver

`mach_alpha_sweep` evaluates the panel method over a 2-D `(M, α)`
grid and returns a long-form pandas DataFrame. Per-cell columns:

| Column                          | Source                          |
|---------------------------------|---------------------------------|
| `CL`, `CD_total`, `CD_wave`     | panel method                    |
| `CD_friction`                   | Eckert/van Driest II            |
| `Cm`                             | panel method (about `x = L_ref/2`) |
| `LD`                             | `CL / CD_total`                 |
| `q_LE_swept_MW_m2`              | Fay-Riddell + swept correction  |
| `beta_attached_margin_deg`      | `θ_max - θ_local_max`           |

The sweep runs in serial in the standalone library and via a
`QThread` worker (`_SweepWorker`) in the GUI, with progress
callbacks per cell. Wall-clock budget: < 35 s for a 5×5 grid on
a typical engineering-mode mesh.

## 7. Provenance & reproducibility

Each `GVWDRunConfig` produces a stable SHA-256 hash via
`config_sha256(cfg)`. The hash is written to `config_sha256.txt`
inside every result directory and embedded in every JSON payload
(`gvwd_config_sha256` field). Re-running the same YAML therefore
produces:

* Byte-identical `config_sha256.txt`.
* Bit-identical `geometry.stl` (floating-point determinism in the
  generators).
* Numerical output reproducible to machine precision.

The example tests `test_sha256_reproducibility` and
`test_changing_input_changes_sha` mechanise these guarantees.

## 8. Cross-imports with PSWR-1

GVWD is allowed to import from `pswr/` for shared utilities (per
the user's go/no-go decision in Phase 1). In practice this is used
for atmosphere helpers and for the `Mesh` data type only; no
plasma-sheath or RCS code is pulled into GVWD. The reverse direction
(PSWR-1 importing from GVWD) is also permitted but currently unused.

## 9. Out of scope

The library deliberately does **not** include:

* CFD / Euler / Navier-Stokes solvers.
* Trim / 6-DOF flight-mechanics integration.
* Real-gas / chemistry models (Saha-Boltzmann lives in PSWR-1, not
  here).
* Arc-jet calibration constants.
* Multi-fidelity surrogate training.
* Optimization (NSGA-II / gradient). GVWD is the *forward map*;
  optimization wrappers live in the parent GUI's `optimization_engine.py`.

These are deliberate scope boundaries. Phase 1's go/no-go decided
that GVWD should be a clean engineering-physics layer that other
modules wrap, not a one-stop shop.
