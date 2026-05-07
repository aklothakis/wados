# GVWD Validation Suite

This document is the audit trail for every numerical result the GVWD
library exposes. It records (a) what was checked, (b) what reference
the answer was checked against, (c) the tolerance achieved, and
(d) the test name in `gvwd/tests/` that mechanises the check on
every run.

The suite is intended to be self-contained: a clean checkout plus
`pytest gvwd/tests/` produces a green-bar result and constitutes a
full revalidation. As of this snapshot:

```
$ pytest gvwd/tests/ -q
146 passed in 8.12s
```

Tests are organized by physics layer. The remainder of this document
walks each layer top-down.

---

## 1. Compressible-flow primitives — `gvwd/thermo/`

### 1.1 Oblique-shock theta-beta-M

`test_oblique_shock.py` evaluates `obtain_beta(M, theta)` on a
20-point grid (`M ∈ {2, 3, 5, 8, 12}`, `θ ∈ {5°, 10°, 15°, 20°}`)
and compares against the canonical Anderson tabulation:

| M  | θ    | β (computed) | β (Anderson App. C) |
|----|------|--------------|----------------------|
| 2  | 10°  | 39.3139°     | 39.3139°             |
| 5  | 15°  | 24.3217°     | 24.3217°             |
| 8  | 10°  | 15.5284°     | 15.5284°             |
| 12 | 20°  | 25.3687°     | 25.3687°             |

Tolerance: `1e-4°` absolute. A round-trip
`β = obtain_beta(M, θ); θ' = θ_from_β(M, β)` reproduces θ to
`1e-10` over a 25-point matrix.

Mach-angle, normal-shock stagnation-pressure ratio, swept-shock
detachment, and θ-max monotonicity are checked alongside.

### 1.2 Tangent-wedge with Newtonian fall-back

`test_tangent_wedge.py` confirms

* shadowed panels (`θ ≤ 0`) return `Cp = 0`,
* the attached regime tracks oblique-shock `Cp` to `1e-3`,
* the Newtonian limb at θ-max takes over without discontinuity
  beyond `0.05` in `Cp`,
* the seam discontinuity decreases monotonically with `M_∞`,
* array dispatch is bit-identical to the scalar path.

### 1.3 Newtonian / modified Newtonian

`test_newtonian.py` verifies the closed-form limits:
`Cp(0)=0`, `Cp(π/2) = Cp_max`, shadow region `Cp = 0`, `Cp_max`
asymptote `2 - 0` as `M → ∞`, and the high-M
`Cp_max(γ=1.4) ≈ 1.839` numerical fit (`test_cp_max_M10_gamma14`).

### 1.4 Oswatitsch equal-strength multi-ramp

`test_oswatitsch.py` evaluates the spec-§5.4 reference: M=5,
two equal-strength ramps with total deflection 20.8°.

```
M∞ = 5.0, n = 2, δ_total = 20.8°
  → δ_inc   = [9.3553°, 11.4447°]
  → β       = [18.7822°, 23.3418°]
  → M_after = [5.000, 4.063, 3.228]
  → π_OS    = 0.7948   (vs single-shock π = 0.557)
  → M_n*    = 1.6099
```

The pressure-ratio-equality property (`p02/p01` identical across
all `n` shocks) is checked to `1e-9`. `n=1` degenerates to a
single 20.8° shock, and `π_OS` increases monotonically with `n`
(by construction).

> **External-reference gate (skipped per user direction):**
> Tabulated comparison against Hammitt 1961 Fig. 8 / Mölder 1967
> Table 2 was waived. The closed-form equal-strength property is
> the only acceptance criterion currently active.

---

## 2. Reference geometries — `gvwd/geometry/`

All reference modes share two structural gates:

* **Outward normals.** `mesh_volume_signed(mesh) > 0` proves every
  triangle is wound CCW from outside. Implemented as the divergence
  theorem `V = (1/6) Σ (v0 · (v1 × v2))`.
* **Closed-form volume parity.** Each mode has an analytic
  closed-form V which the mesh integrator must reproduce to better
  than 0.5%.

### 2.1 Caret (Nonweiler)

`test_caret_geometry.py`:

* Constructs from either `θ_d` or `β`.
* Volume closed form `V = (1/3) · L · y_tip · h` — match to 0.1%.
* `y_tip = h / tan(β)` formula-derivation checked.
* `η_V` falls in the expected `(0.1, 0.4)` range for representative
  inputs.
* Signed volume positive on the standard Nonweiler caret with
  apex at the tip and ridge UB on the upper-base centerline.

`test_aero_anderson_ch14.py::test_caret_M6_theta14_LD_matches_one_over_tan_theta`:

```
M = 6, θ_d = 14°, Λ = 70°, L = 10
  CL_inviscid = 0.1706
  CD_inviscid = 0.0425
  L/D         = 4.011
  Anderson Ch. 14 (1/tan θ_d) = 4.011
```

CL itself agrees with Anderson's caret table to within 2% (the
panel-integration limb of the equality, beyond the L/D identity).

### 2.2 Flat-bottomed delta

`test_flat_delta_geometry.py` and
`test_aero_anderson_ch14.py::test_flat_delta_M5_theta12_inviscid`:

* `y_tip = L · cos(Λ - θ_LE_eff)` analytic formula confirmed.
* Closed-form volume reproduced to 0.1%.
* Detachment guard fires at the high-sweep limit (β unattainable
  for the design point implies a `ValueError`).

### 2.3 Multi-wedge (Oswatitsch-derived)

`test_multi_wedge_geometry.py`:

* `n=2 M=5 δ=20.8°` reproduces the Oswatitsch ramp angles to better
  than 0.1° (`test_multi_wedge_n2_M5_matches_oswatitsch_to_0p1_deg`).
* Both rectangular and delta extrusions yield closed meshes.
* `n=1` degenerates to a single ramp.
* `π_OS` monotone in `n` confirmed at the geometry layer.

---

## 3. Engineering geometries — `gvwd/geometry/`

### 3.1 Engineering flat-bottom (HTV-2 archetype)

`test_engineering_flat_geometry.py`:

* HTV-2-class point `(M=15, θ_fore=8°, Λ=75°, L_fore=2.5,
  L_center=1.5, b=0.5, h=0.4)` produces:

  ```
  V          = 1.0494 m³
  S_planform = 3.4295 m²
  η_V        = 0.3011        (HTV-2 archetype expected ~0.3)
  mesh       = 9 vertices / 14 outward-CCW triangles
  ```

* Closed-form volume (sum of fore wedge + centerbody box) matches
  the divergence-theorem mesh integral to better than 1e-12.
* `M=6, θ_fore=8°` correctly raises detachment via the embedded
  `obtain_beta` guard.
* `b_base ≤ 2 · y_tip(θ_fore, Λ)` constraint is enforced.

### 3.2 Engineering shallow-V

`test_engineering_shallow_v_geometry.py`:

* Mesh closes (signed-volume positive).
* `dihedral=0` recovers the flat-bottom volume to machine precision.
* Non-zero dihedral always yields `η_V > η_V_flat` — the
  motivating property.

### 3.3 Fins

`test_fins_geometry.py` — diamond-airfoil-cross-section fins:

* Symmetric LE/TE half-angles produce identical fore and aft
  wedge angles.
* Asymmetric `max_thickness_loc ≠ 0.5` produces distinct fore/aft
  half-angles.
* Single fin signed-volume positive.
* `n_fins = 0` returns `None`.
* 2- and 4-fin assemblies construct.
* Engineering flat + fins + `merge_meshes` still gives a closed
  watertight surface.

---

## 4. Aero — `gvwd/aero/`

### 4.1 Sign convention regression test

`test_aero_high_alpha_newtonian.py::test_engineering_flat_alpha_zero_positive_aero`
locks in the convention discovered during Phase 4 development:

* freestream body-frame direction at α: `v̂∞ = (cos α, 0, +sin α)`
  (apex upstream, body x downstream, +α nose-up),
* drag is `D = +F · v̂∞` (in flow direction),
* lift is `L = F · ẑ_lift`.

At α=0 the engineering flat-bottom waverider returns `CL > 0`
(lower surface is windward by design), and the lower-surface windward
test catches sign flips in either v̂∞ or D.

### 4.2 Newtonian high-α floor

`test_flat_plate_alpha_neg90_matches_newtonian` confirms a flat plate
at α = −90° produces the analytic `CL_Newt = -Cp_max ≈ -1.839`
to within 1%, including the floating-point spillover fix (windward
threshold `1e-6 rad`, not 0, otherwise `cos(-π/2)=6e-17` triggers
non-bracketing brentq calls in `tangent_wedge_cp_array`).

### 4.3 Panel-method speed

`test_panel_method_speed_5000_panels` keeps the inviscid panel
solver under 1 s for a 5 000-panel mesh (typical engineering-mode
mesh density), preserving the GUI's < 3 s on-design budget.

### 4.4 Anderson Ch. 14 inviscid table

`test_aero_anderson_ch14.py` matches three Anderson reference cases
inviscidly (the one-over-tan-θ result for the caret already covered
above plus M=5 flat delta with θ=12° and M=5 two-ramp).

---

## 5. Heating — `gvwd/heating/`

The DoD heating gate (spec §5.4): a sharp 1 mm leading edge at
M=15, h=30 km should yield 50–200 MW/m² of swept-LE convective
heating. With a Λ=75° sweep:

```
h=30 km, M=15 (V≈4525 m/s, ρ≈0.0184 kg/m³)
  Fay-Riddell stagnation R=1mm:        q = 257.2 MW/m²
  Fay-Riddell swept Λ=75° R=1mm:        q = 130.8 MW/m²   ✓ inside DoD
  Fay-Riddell swept Λ=75° R=5mm:        q = 58.5 MW/m²
  Tauber-Sutton 1991 conv R=1mm:        q = 257.2 MW/m²
```

`test_heating_fay_riddell.py`:

* `test_sharp_1mm_LE_M15_h30_in_spec_range` and `test_sharp_5mm_…`
  pin the DoD numbers.
* Bluntness reduces heating (`R⁻¹ᐟ²` scaling).
* Swept-LE correction `(cos Λ)^(0.5 to 1.0)` is monotone in Λ.
* Nose-heat-flux alias `nose_heat_flux` preserved for back-compat.
* Velocity scaling `q ∝ V^3.15` (Tauber-Sutton 1991) is verified
  point-wise.
* Negative or zero radius raises `ValueError`.

`test_heating_tauber_sutton.py`:

* TS convective formula reduces to the Fay-Riddell form at low
  altitude / standard atmosphere.
* Radiative term zero below `V < 9 km/s`, finite-positive above,
  monotone in V.
* TS and Fay-Riddell agree to within 20 % at M=15, R=10 mm — the
  cross-correlation cross-check.

> **Calibration note.** The PSWR-1 prototype used the Sutton-Graves
> (V³, K=1.7415) form, which over-predicts by O(10⁴) at M=15.
> The DoD numbers above only land if the V³·¹⁵ form with
> K=1.83·10⁻⁴ (Tauber-Sutton 1991) is used, and that is what
> ships in `gvwd/heating/`. PSWR-1 was not retroactively patched
> because its `q_LE_max` gate was relaxed during its own validation
> phase; GVWD does not inherit that relaxation.

---

## 6. Sweep driver — `gvwd/aero/sweep.py`

`test_sweep_grid.py`:

* 3×3 sweep on the HTV-2-class mesh runs in `< 30 s` wall-clock
  on the reference workstation; the GUI's threaded path measured
  0.85 s for the same grid.
* All required output columns present: `M_inf`, `alpha_deg`, `CL`,
  `CD_total`, `CD_wave`, `CD_friction`, `Cm`, `LD`, `q_LE_swept_MW_m2`,
  `beta_attached_margin_deg`.
* `q_LE_swept` is monotone increasing in `M`.
* `q_LE_swept` only weakly depends on α (post-shock conditions are
  set by Λ and `M_∞`, not by α).
* `L/D(α)` rises then falls — the canonical hypersonic L/D peak.
* 2-D `(M, α)` reshape matches the row-major flatten.
* Progress callback fires once per cell.

---

## 7. Round-trip exports — `gvwd/export/`

`test_export_roundtrip.py`:

* STL written via `write_stl` and re-read via `read_stl` reproduces
  vertices to machine precision after the `scale=1000` mm
  convention.
* Volume preserved across the roundtrip.
* Scale factor (m → mm and back) bit-exact at IEEE-754 precision.
* STEP export of the engineering flat-bottom mesh produces a valid
  cadquery solid (header + `B_SPLINE_SURFACE_WITH_KNOTS` topology).
* IGES export attempts the OCP path and gracefully degrades to
  `IGESUnavailableError` if `OCP.IGESControl` is missing, with a
  recommendation to use STEP.
* STL header contains the geometry kind tag (string-search guard).

---

## 8. End-to-end baseline runs — `gvwd/tests/test_baseline_htv2_scaled.py`

Each example config in `gvwd/examples/configs/` has a paired test
asserting the full `_runner.run_from_config(...)` path completes in
under 60 s and writes the spec §7 directory layout:

* `caret_M6.yaml` ✓
* `flat_delta_M5.yaml` ✓
* `two_ramp_M5.yaml` ✓
* `engineering_flat_htv2_class.yaml` (with sweep) ✓

Provenance gate: `test_sha256_reproducibility` checks that running
the same YAML twice produces byte-identical `config_sha256.txt`,
and `test_changing_input_changes_sha` checks that perturbing any
input invalidates the hash.

---

## 9. GUI tab parity — Phase 7 DoD

The GUI tab `gvwd_waverider_tab.py` does not add new aero physics;
it wraps the library. Parity with the standalone library is asserted
at runtime via the Phase 7 smoke test:

| Quantity              | GUI                | Standalone          | Δ        |
|-----------------------|--------------------|---------------------|----------|
| CL (eng-flat HTV-2 design point) | 0.028891014666     | 0.028891014666      | 0.0      |
| CD_total              | 0.015471077786     | 0.015471077786      | 0.0      |
| 3×3 sweep CL grid     | 9 rows             | 9 rows              | 0.0 max  |
| 3×3 sweep CD grid     | 9 rows             | 9 rows              | 0.0 max  |
| Generate + on-design  | 126 ms + 3 ms      | —                   | < 3 s ✓ |
| Threaded sweep 3×3    | 0.85 s             | —                   | < 35 s ✓ |
| STEP export           | 730 lines / 32030 B | 730 lines / 32030 B | topology equal — only embedded OCP timestamps differ |

---

## 10. External-reference gates skipped on user direction

The following spec-§5.8 gates were waived in this iteration. They are
recorded here so a later validation pass can pick them up without
re-deriving the requirement:

* **HTV-2 / Avangard public flight envelope L/D vs M curve.** Open
  literature does not give panel-method-comparable points — would
  need a CFD or wind-tunnel reference dataset.
* **Mölder 1967 / Hammitt 1961 multi-shock pressure-ratio tables.**
  Closed-form equal-strength property is checked instead.
* **Reichenbach Cp / heat-flux correlation for swept blunt LE.**
  The Tauber-Sutton 1991 cross-check is the active substitute.
* **Goldfeld arc-jet R/L scaling.** Same.

These gates are external-reference-bound and require curated data
not currently in the tree. `validation.md` will be updated when
that data lands.
