# GVWD Glossary

Terms used throughout the codebase, the test names, and the docs.

## Symbols

| Symbol         | Meaning |
|----------------|---------|
| `M`, `M∞`     | Freestream Mach number |
| `α`            | Angle of attack (rad in code, deg in YAML/GUI) |
| `β`            | Oblique-shock wave angle |
| `θ`            | Flow-deflection angle (wedge angle) |
| `θ_d`          | Design ramp angle (caret / flat-delta) |
| `θ_fore`       | Fore-wedge ramp angle (engineering modes) |
| `θ_local`      | Per-panel local deflection angle in tangent-wedge |
| `θ_max`        | Detachment-limit wedge angle for given M |
| `Λ`            | LE sweep angle |
| `γ`            | Specific-heat ratio (1.4 throughout) |
| `δ`            | Per-ramp deflection in multi-wedge / Oswatitsch |
| `δ_total`      | Total deflection across all `n` ramps |
| `Cp`           | Pressure coefficient |
| `Cp_max`       | Modified-Newtonian post-shock max `Cp` |
| `CL`, `CD`     | Lift / drag coefficients (planform-area normalised) |
| `CD_wave`      | Wave drag (panel-method pressure integral) |
| `CD_friction`  | Skin-friction drag (Eckert / van Driest II) |
| `CD_total`     | `CD_wave + CD_friction` |
| `Cm`           | Pitching-moment coefficient about `x = L_ref/2` |
| `L/D`          | Aerodynamic-efficiency ratio |
| `q∞`           | Freestream dynamic pressure `½ γ p∞ M∞²` |
| `q_LE`         | Leading-edge convective heat flux |
| `R_LE`         | Leading-edge radius |
| `R_N`          | Nose radius |
| `T_w`          | Wall temperature (surface boundary condition) |
| `Re_x`         | Local chord-Reynolds number |
| `Re_x_tr`      | Transition Reynolds number (laminar→turbulent) |
| `S_ref`        | Reference area (mesh planform projection) |
| `L_ref`        | Reference length (`L_fore + L_center` engineering, chord otherwise) |
| `V`            | Body volume (numerical, divergence-theorem) |
| `η_V`          | Volumetric efficiency `V^(2/3) / S_planform` |
| `π_OS`         | Oswatitsch total-pressure-recovery ratio across `n` ramps |
| `π_OS_with_normal` | Same, including a normal-shock at the design point |
| `M_n*`         | Equal-strength normal-Mach across all `n` ramps |
| `n̂`            | Outward unit normal to a mesh triangle |
| `v̂∞`           | Freestream unit vector in body frame `(cos α, 0, sin α)` |

## Geometric terms

* **Apex / tip.** The nose-most vertex of the waverider.
  Coincides with the coordinate origin.
* **Lower surface.** The compression / windward surface, traced
  along the design-shock surface for the reference modes.
* **Upper surface.** Freestream surface (≈ Cp = 0 inviscidly).
* **Base.** Aft-most face of the body, at `x = L_fore + L_center`
  for engineering modes.
* **Centerbody.** Flat-bottomed box section aft of the fore-wedge
  in the engineering modes.
* **Fore-wedge.** The waverider-derived ramp section ahead of the
  centerbody.
* **Half-body.** The mesh as stored, with `z ≥ 0` only. Volume
  and planform integrators multiply by 2 to recover the full body.
* **y_tip.** Spanwise (z) extent of the leading edge at the base.
* **Diamond fin.** The cross-section profile used by
  `gvwd/geometry/fins.py`: two wedges meeting at a maximum-thickness
  station, parameterised by `t/c` and `max_thickness_loc`.

## Method terms

* **Tangent-wedge.** Local-flow approximation in which each panel
  is treated as a 2-D oblique shock in its own plane.
* **Modified Newtonian.** Newtonian impact theory rescaled so that
  `Cp(θ=π/2) = Cp_max` at the given M (instead of 2).
* **Oswatitsch equal-strength.** Multi-shock optimization
  property: equal `p02/p01` across all ramps minimises total
  stagnation-pressure loss for given total deflection.
* **Eckert reference temperature.** Compressible-flow flat-plate
  friction-coefficient correction that maps high-temperature
  boundary-layer effects onto an equivalent low-temperature
  reference state.
* **van Driest II.** Compressibility correction to incompressible
  turbulent-flat-plate `Cf` formulas.
* **Fay-Riddell.** Stagnation-point convective heating correlation
  for laminar boundary-layer flow at a sphere or cylinder LE.
* **Tauber-Sutton 1991.** A slightly more accurate
  stagnation-point convective correlation with `q ∝ V^3.15`
  velocity scaling and an additive radiative term.
* **Sutton-Graves.** A different Apollo-era convective
  correlation with `V^3` scaling. **Not** used in GVWD because
  it over-predicts heating at glide-vehicle Mach numbers; PSWR-1
  uses it but with a relaxed q_LE gate.
* **Divergence-theorem volume.** `V = (1/6) Σ_triangles
  v0 · (v1 × v2)`. Positive iff the mesh is closed and outward
  CCW.

## Modes

* **engineering_flat.** Flat-bottomed engineering mode (HTV-2
  archetype). Default for the GUI tab.
* **engineering_shallow_v.** Same plus a dihedral on the lower
  surface for higher η_V. Default for the Fattah-2 archetype.
* **caret.** Nonweiler caret reference mode, single ramp +
  upper-base-centerline ridge.
* **flat_delta.** Flat-bottomed delta reference mode, single
  ramp + flat upper.
* **multi_wedge.** Oswatitsch-derived `n`-ramp reference mode.
  Available with rectangular or delta extrusion.

## Files & artifacts

* **`config.yaml`.** Per-run configuration written verbatim into
  the result directory.
* **`config_sha256.txt`.** Provenance hash of the run config.
* **`geometry.stl`.** Body mesh in IEEE-754 binary STL.
* **`geometry.step`.** Optional CAD-grade body in STEP via
  cadquery.
* **`coefficients_on_design.json`.** On-design `(M, α=0)` aero
  output.
* **`sweep_results.h5` / `sweep_results.json`.** Off-design
  `(M, α)` grid output (long-form).
* **`volumetric.json`.** `V`, `S_planform`, `η_V`.
* **`heating.json`.** Spec-defined heating summary (when
  populated).
* **`plots/`.** Figures generated by `gvwd/viz/plotting.py`.
* **`log.txt`.** One-line per-run record.

## Acronyms

* **DoD** — Definition-of-Done acceptance criterion.
* **GVWD** — Glide-Vehicle Wedge-Derived (this library).
* **HTV-2** — Hypersonic Technology Vehicle 2 (DARPA). The
  flat-bottom engineering archetype.
* **LE / TE** — Leading edge / trailing edge.
* **OPM** — Oblique-shock + Prandtl-Meyer panel method (PySAGAS,
  the parent project's vendored solver — not used inside GVWD,
  but the GUI may pass GVWD-generated meshes to it).
* **PSWR-1** — Plasma-Sheath-Shaped Variable-Wedge Waverider, the
  sibling project at the same parent-GUI level.
* **TS 1991** — Tauber-Sutton 1991 convective heating
  correlation.
* **CCW** — Counter-clockwise (mesh-winding convention from
  outside the surface).
