"""Full 3D VMPLO waverider geometry + integration with the existing CAD pipeline.

Spec reference: VMPLO_implementation_prompt.md ``vmplo/geometry.py``.

Two surfaces per instance:

* **Upper** — flat freestream-aligned surface at ``y = ICC(z)`` from the
  leading edge ``(x_LE, ICC(z), z)`` to the base plane top
  ``(L, ICC(z), z)``.
* **Lower** — compression surface traced in each osculating plane by
  :func:`powerlaw.solve_osculating_plane`, converted from cone-local
  ``(x, r)`` to global ``(X, Y, Z)`` via ``y_lower = ICC(z) + r_LE -
  r_stream(x)``.

Wingtip is *not* forced degenerate — if ``ICC(W)`` places the LE above
the streamline at ``x = L`` the body has finite thickness at the
wingtip, which is physically correct for a VMPLO design.
``cad_export._build_wingtip_face`` handles both pinch and non-pinch
tips.

### Integration contract (used by the tab + CAD pipeline)

Attributes expected by downstream code:

* ``upper_surface_streams``, ``lower_surface_streams`` —
  ``list[ndarray(n_x, 3)]`` of length ``n_z`` (= ``n_planes + 2`` when
  the tab sets ``n_planes``).  Every stream has the same number of
  points ``n_x``: this is a hard requirement of
  ``cad_export._make_bspline_face``.
* ``leading_edge`` — ``ndarray(n_z, 3)``, shared with
  ``stream[i, 0]``.
* ``length``, ``height``, ``width``, ``beta_deg`` — plain floats.
* ``n_streamwise`` — ``int``, matches ``streams[0].shape[0]``.
* ``_mach_per_station``, ``_n_per_station``, ``cone_angles_deg``,
  ``_z_all`` — ``ndarray(n_z)`` used by the Distribution / Cone-Angle
  preview canvases in the tab.
* ``to_CAD(sides, export, filename, **kwargs)`` — same signature as
  ``shadow_waverider_tab._export_step_nurbs`` uses with
  ``build_waverider_solid``.
* ``compute_volume()``, ``export_stl(filename)``, ``get_body_geometry()``
  — signatures the tab already calls.
"""

from __future__ import annotations

import struct
import numpy as np

from waverider_generator.vmplo.bspline import BSpline1D
from waverider_generator.vmplo.osculating import OsculatingAssembly


class VMPLOWaverider:
    """3D VMPLO waverider assembled from an :class:`OsculatingAssembly`.

    Parameters
    ----------
    assembly : OsculatingAssembly
        Configured per-plane flow model (Ma, n, ICC, beta, L, W, H, x_LE).
    n_planes : int
        Number of *interior* spanwise planes.  Stream count is
        ``n_planes + 2`` (adds symmetry-plane and wingtip endpoints).
    n_streamwise : int
        Points per streamline (LE to base plane).

    Notes
    -----
    All streams are produced with the **same length** ``n_streamwise``.
    The tip stream is a real geometric stream (not forced to collapse);
    ``cad_export._build_wingtip_face`` handles non-pinch tips via a
    loft, and pinch tips via the ``us_span < 1e-8`` early return.
    """

    def __init__(self,
                 assembly: OsculatingAssembly,
                 n_planes: int = 25,
                 n_streamwise: int = 40):
        if n_planes < 3:
            raise ValueError("n_planes must be >= 3.")
        if n_streamwise < 5:
            raise ValueError("n_streamwise must be >= 5.")

        self.assembly = assembly
        self.n_planes = int(n_planes)
        self.n_streamwise = int(n_streamwise)

        # Scalar aliases for compat (tab reads these directly)
        self.length = assembly.L
        self.height = assembly.H
        self.width = assembly.W
        self.x_LE = assembly.x_LE
        self.beta_deg = assembly.beta_design
        self.gamma = assembly.gamma

        # Build everything
        self._build()

    # ================================================================== #
    #  Build                                                              #
    # ================================================================== #

    def _build(self) -> None:
        """Trace every span station, assemble 3D streams (classical OC layout).

        Layout — classical osculating-cone waverider:

        * LE is **swept in x**: ``x_LE(z)`` runs linearly from
          ``x_LE_centerline`` (≈0, near the nose) at z=0 to ``x = L``
          at z=W (base-plane wingtip corner).  LE y-coordinate
          follows the body surface plus an ICC shaping offset.
        * Streams go from the LE at ``j=0`` to the base plane at
          ``j=-1`` — matches the shadow-waverider / OC convention
          that ``cad_export.build_waverider_solid`` expects.
        * Upper surface: freestream-aligned (flat) at ``y = y_LE(z)``.
        * Lower surface: per-plane compression streamline traced by
          T-M or MOC, converted from cone-local ``r`` to global
          ``y = y_LE(z) - (r(x) - r_LE)``.
        * At the wingtip (z=W), ``x_LE = L`` so all stream points
          collapse onto the base-plane wingtip corner — pinch tip.
        * The body **does not extend forward of the LE** — there is
          no nose wedge in the streams.  The LE curve itself is the
          forward boundary of the body, and at the centerline
          (``x_LE ≈ 0``) the LE is effectively at the nose.

        Stream structure (n_x points per stream):

            j=0     : LE point at this z, ``(x_LE(z), y_LE(z), z)``
            j=1..-2 : interior points, ``x ∈ (x_LE(z), L)``
            j=-1    : base plane, ``(L, y_base, z)``
        """
        n_z = self.n_planes + 2
        n_x = self.n_streamwise
        W = self.width
        L = self.length

        # Spanwise stations (endpoints included)
        z_stations = np.linspace(0.0, W, n_z)

        # Per-station distributions (for viz and CAD)
        self._z_all = z_stations.copy()
        self._mach_per_station = np.array(
            [self.assembly.Ma_at_z(z) for z in z_stations])
        self._n_per_station = np.array(
            [self.assembly.n_at_z(z) for z in z_stations])
        self.cone_angles_deg = np.array(
            [self.assembly.cone_angle_at_z(z) for z in z_stations])
        self._icc_per_station = np.array(
            [self.assembly.ICC_at_z(z) for z in z_stations])

        # LE curve — swept: x_LE(z) linear 0→L, y_LE(z) from
        # body-surface + ICC shaping.
        x_LE_arr = np.array([self.assembly.x_LE_at_z(z) for z in z_stations])
        y_LE_arr = np.array([self.assembly.y_LE_at_z(z) for z in z_stations])
        self.leading_edge = np.column_stack([x_LE_arr, y_LE_arr, z_stations])

        # ``j_LE`` is retained for backward-compat — in the
        # no-nose-wedge layout it is always 0 (the LE is at
        # ``stream[0]``, matching the shadow/OC convention).
        self._j_LE = 0

        # --- Per-station streams -------------------------------------
        upper_streams: list[np.ndarray] = []
        lower_streams: list[np.ndarray] = []

        for i, z_i in enumerate(z_stations):
            le_i = self.leading_edge[i]
            x_LE_i = float(x_LE_arr[i])
            y_LE_i = float(y_LE_arr[i])

            # x-grid from x_LE(z) to L, n_x points.  At the wingtip
            # (z=W) this collapses to n_x copies of (L, ...).
            if x_LE_i >= L - 1e-9:
                x_line = np.full(n_x, L)
                # Degenerate tip stream: n_x copies of the wingtip LE
                upper = np.tile(le_i, (n_x, 1))
                lower = np.tile(le_i, (n_x, 1))
                upper_streams.append(upper)
                lower_streams.append(lower)
                continue

            x_line = np.linspace(x_LE_i, L, n_x)

            # ---- Upper surface: flat freestream-aligned at y_LE(z)---
            upper = np.column_stack([
                x_line,
                np.full(n_x, y_LE_i),
                np.full(n_x, z_i),
            ])
            upper_streams.append(upper)

            # ---- Lower surface: compression streamline --------------
            try:
                xs, rs = self.assembly.build_strip(z_i, n_x=n_x)
            except Exception:
                xs = x_line.copy()
                rs = np.zeros(n_x)

            if xs.size != n_x or not np.allclose(xs, x_line):
                rs = np.interp(x_line, xs, rs)
                xs = x_line.copy()

            # Cone-local r → global y:  compression surface below the
            # LE by (r(x) - r_LE).  At x=x_LE gives y=y_LE (on the LE).
            r_LE = float(rs[0])
            y_lower = y_LE_i - (rs - r_LE)
            lower = np.column_stack([
                xs,
                y_lower,
                np.full(n_x, z_i),
            ])
            lower_streams.append(lower)

        self.upper_surface_streams = upper_streams
        self.lower_surface_streams = lower_streams

        # Pin the LE point to exactly leading_edge[i] (absorb numeric
        # drift from the streamline integrator at x=x_LE).
        for i in range(n_z):
            self.upper_surface_streams[i][0] = self.leading_edge[i]
            self.lower_surface_streams[i][0] = self.leading_edge[i]

    # ================================================================== #
    #  Spec-style API                                                     #
    # ================================================================== #

    def lower_surface(self, n_z: int | None = None, n_x: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X, Y, Z)`` meshgrid of the lower surface.

        Defaults to the arrays already built; pass different ``n_z`` /
        ``n_x`` to resample by linear interpolation from the stored
        streams.
        """
        if n_z is None and n_x is None:
            streams = self.lower_surface_streams
            X = np.array([s[:, 0] for s in streams])
            Y = np.array([s[:, 1] for s in streams])
            Z = np.array([s[:, 2] for s in streams])
            return X, Y, Z
        # Resample
        nz = n_z if n_z is not None else len(self.lower_surface_streams)
        nx = n_x if n_x is not None else self.n_streamwise
        X, Y, Z = self.lower_surface()
        return (_resample_grid(X, nz, nx),
                _resample_grid(Y, nz, nx),
                _resample_grid(Z, nz, nx))

    def upper_surface(self, n_z: int | None = None, n_x: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_z is None and n_x is None:
            streams = self.upper_surface_streams
            X = np.array([s[:, 0] for s in streams])
            Y = np.array([s[:, 1] for s in streams])
            Z = np.array([s[:, 2] for s in streams])
            return X, Y, Z
        nz = n_z if n_z is not None else len(self.upper_surface_streams)
        nx = n_x if n_x is not None else self.n_streamwise
        X, Y, Z = self.upper_surface()
        return (_resample_grid(X, nz, nx),
                _resample_grid(Y, nz, nx),
                _resample_grid(Z, nz, nx))

    def exit_plane_contour(self) -> np.ndarray:
        """Closed (y, z) contour in the exit plane ``x = L``."""
        upper_te = np.array([s[-1, 1:] for s in self.upper_surface_streams])
        lower_te = np.array([s[-1, 1:] for s in self.lower_surface_streams])
        # Wingtip first, then centreline the other way — forms a closed loop
        contour = np.concatenate([
            upper_te,
            lower_te[::-1],
        ])
        return contour

    # ================================================================== #
    #  Metrics                                                            #
    # ================================================================== #

    def compute_volume(self) -> float:
        """Estimate body volume by cross-sectional area integration.

        Positive Y-thickness only (upper above lower).  Integrated in
        (x, z) via the trapezoidal rule.  Returns volume in m^3 for the
        full (mirrored) vehicle.
        """
        us = self.upper_surface_streams
        ls = self.lower_surface_streams
        n_span = len(us)
        n_stream = us[0].shape[0]

        # For each streamwise j, compute cross-section area A(x_j)
        areas = []
        x_stations = []
        for j in range(n_stream):
            zs = []
            dys = []
            for i in range(n_span):
                zu, yu = us[i][j, 2], us[i][j, 1]
                _, yl = ls[i][j, 2], ls[i][j, 1]
                zs.append(zu)
                dys.append(max(yu - yl, 0.0))
            zs = np.asarray(zs)
            dys = np.asarray(dys)
            order = np.argsort(zs)
            areas.append(float(np.trapz(dys[order], zs[order])))
            x_stations.append(us[0][j, 0])

        if len(areas) < 2:
            return 0.0
        # Half-vehicle integral, times 2 for full mirror
        vol_half = float(np.trapz(areas, x_stations))
        return 2.0 * vol_half

    def wetted_area(self) -> float:
        """Sum of upper + lower surface triangle areas (half span)."""
        def _area_grid(X, Y, Z):
            a = 0.0
            for i in range(X.shape[0] - 1):
                for j in range(X.shape[1] - 1):
                    p00 = np.array([X[i, j],     Y[i, j],     Z[i, j]])
                    p10 = np.array([X[i + 1, j], Y[i + 1, j], Z[i + 1, j]])
                    p01 = np.array([X[i, j + 1], Y[i, j + 1], Z[i, j + 1]])
                    p11 = np.array([X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]])
                    a += 0.5 * np.linalg.norm(np.cross(p10 - p00, p01 - p00))
                    a += 0.5 * np.linalg.norm(np.cross(p10 - p11, p01 - p11))
            return a
        Xu, Yu, Zu = self.upper_surface()
        Xl, Yl, Zl = self.lower_surface()
        return float(_area_grid(Xu, Yu, Zu) + _area_grid(Xl, Yl, Zl)) * 2.0

    def planform_area(self) -> float:
        """Projected area onto the x-z plane (full vehicle, both halves).

        With stream[0] = LE (swept-LE classical layout), the chord at
        each spanwise station is ``stream[-1, 0] - stream[0, 0]`` =
        ``L - x_LE(z)``.  Integrated over z ∈ [0, W] and doubled for
        the mirror half.
        """
        us = self.upper_surface_streams
        le = self.leading_edge
        chords = []
        zs = []
        for i, s in enumerate(us):
            if s.shape[0] < 2:
                continue
            chords.append(float(s[-1, 0] - s[0, 0]))
            zs.append(float(le[i, 2]))
        if len(chords) < 2:
            return 0.0
        order = np.argsort(zs)
        zs_sorted = np.asarray(zs)[order]
        chords_sorted = np.asarray(chords)[order]
        area_half = float(np.trapz(chords_sorted, zs_sorted))
        return 2.0 * area_half

    def volumetric_efficiency(self, definition: str = "corda") -> float:
        V = self.compute_volume()
        if definition == "corda":
            S = self.planform_area()
            return float(V ** (2.0 / 3.0) / S) if S > 1e-12 else 0.0
        if definition == "box":
            return float(V / (self.length * self.width * self.height))
        raise ValueError(f"Unknown definition: {definition}")

    # ================================================================== #
    #  CAD pipeline (compat: routes through build_waverider_solid)        #
    # ================================================================== #

    def to_CAD(self, sides: str = "both", export: bool = False,
               filename: str = "vmplo_waverider.step", **kwargs):
        """Export via :func:`cad_export.build_waverider_solid`.

        Matches the shadow-waverider pipeline — the robust 4-face
        NURBS-solid builder with wingtip TE closure and fuse/compound
        fallback.  Kept identical to the approved pattern we built for
        the old VMPLO in this session so it's a known-good route.
        """
        import cadquery as cq
        from waverider_generator.cad_export import build_waverider_solid

        scale = float(kwargs.get("scale", 1000.0))

        upper_streams = self.upper_surface_streams
        lower_streams = self.lower_surface_streams
        le_curve = self.leading_edge
        centerline_upper = upper_streams[0]
        centerline_lower = lower_streams[0]
        te_upper = np.vstack([s[-1] for s in upper_streams])
        te_lower = np.vstack([s[-1] for s in lower_streams])

        print(f"[VMPLO to_CAD] sides={sides} scale={scale} "
              f"streams={len(upper_streams)}x{upper_streams[0].shape[0]}")
        print(f"[VMPLO to_CAD]   stream[0] LE (centerline): {upper_streams[0][0]}")
        print(f"[VMPLO to_CAD]   stream[-1] LE (wingtip):   {upper_streams[-1][0]}")
        print(f"[VMPLO to_CAD]   te_upper[0] / te_upper[-1]: "
              f"{te_upper[0]} / {te_upper[-1]}")
        print(f"[VMPLO to_CAD]   te_lower[0] / te_lower[-1]: "
              f"{te_lower[0]} / {te_lower[-1]}")

        try:
            right_side = _build_vmplo_solid_iso(
                upper_streams, lower_streams,
                centerline_upper, centerline_lower,
                te_upper, te_lower,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            # Fall back to shadow's build_waverider_solid
            print(f"[VMPLO to_CAD] iso-curve builder failed ({exc}); "
                  f"falling back to build_waverider_solid.")
            right_side = build_waverider_solid(
                upper_streams, lower_streams, le_curve,
                centerline_upper, centerline_lower,
                te_upper, te_lower,
            )
        right_side = right_side.scale(scale)

        if sides == "right":
            result = cq.Workplane("XY").newObject([right_side])
        elif sides == "left":
            result = cq.Workplane("XY").newObject(
                [right_side.mirror(mirrorPlane="XY")])
        else:   # "both"
            # Compound of two halves — exact match of shadow_waverider's
            # export wrapper (shadow_waverider_tab.py:2775-2784).
            # Earlier attempts to use BRepAlgoAPI_Fuse returned a
            # non-null-but-empty shape (inputs share only a face, no
            # volume overlap) so a compound is the reliable path.
            left_side = right_side.mirror(mirrorPlane="XY")
            from OCP.TopoDS import TopoDS_Compound
            from OCP.BRep import BRep_Builder
            builder = BRep_Builder()
            comp = TopoDS_Compound()
            builder.MakeCompound(comp)
            builder.Add(comp, right_side.wrapped)
            builder.Add(comp, left_side.wrapped)
            result = cq.Workplane("XY").newObject([cq.Shape(comp)])

        if export:
            cq.exporters.export(result, filename)
            import os
            size = os.path.getsize(filename)
            print(f"[VMPLO STEP] Wrote {size} bytes to {filename}")
        return result

    # ================================================================== #
    #  STL export (binary)                                                #
    # ================================================================== #

    def export_stl(self, filename: str, mirror: bool = True) -> None:
        """Write a binary STL of the (optionally mirrored) waverider."""
        vertices, triangles = self._build_mesh(mirror=mirror)
        with open(filename, "wb") as f:
            f.write(b"\x00" * 80)
            f.write(struct.pack("<I", len(triangles)))
            for tri in triangles:
                v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                n = np.cross(v1 - v0, v2 - v0)
                nrm = np.linalg.norm(n)
                n = n / nrm if nrm > 1e-14 else np.array([0.0, 0.0, 1.0])
                f.write(struct.pack("<3f", *n))
                f.write(struct.pack("<3f", *v0))
                f.write(struct.pack("<3f", *v1))
                f.write(struct.pack("<3f", *v2))
                f.write(struct.pack("<H", 0))
        print(f"[VMPLO] STL exported to {filename}")

    def _build_mesh(self, mirror: bool = True
                    ) -> tuple[np.ndarray, np.ndarray]:
        us = self.upper_surface_streams
        ls = self.lower_surface_streams
        n_span = len(us)
        n_stream = us[0].shape[0]

        def _grid_to_verts(streams):
            return np.vstack(streams)

        verts_u = _grid_to_verts(us)
        verts_l = _grid_to_verts(ls)
        if mirror:
            verts_u_L = verts_u.copy(); verts_u_L[:, 2] *= -1.0
            verts_l_L = verts_l.copy(); verts_l_L[:, 2] *= -1.0
            vertices = np.vstack([verts_u, verts_l, verts_u_L, verts_l_L])
            lo_u, lo_l = 0, n_span * n_stream
            loL_u = 2 * n_span * n_stream
            loL_l = 3 * n_span * n_stream
            offsets = [(lo_u, lo_l), (loL_u, loL_l)]
        else:
            vertices = np.vstack([verts_u, verts_l])
            offsets = [(0, n_span * n_stream)]

        tris: list[list[int]] = []
        for off_u, off_l in offsets:
            # Upper quads (outward normals depend on winding; we output
            # two winding orders across the mesh which is OK for
            # watertight-check tooling that tolerates orientation)
            for i in range(n_span - 1):
                for j in range(n_stream - 1):
                    a = off_u + i * n_stream + j
                    b = a + 1
                    c = off_u + (i + 1) * n_stream + j
                    d = c + 1
                    tris.append([a, c, b]); tris.append([b, c, d])
            for i in range(n_span - 1):
                for j in range(n_stream - 1):
                    a = off_l + i * n_stream + j
                    b = a + 1
                    c = off_l + (i + 1) * n_stream + j
                    d = c + 1
                    tris.append([a, b, c]); tris.append([b, d, c])
            # LE cap
            for i in range(n_span - 1):
                u0 = off_u + i * n_stream
                u1 = off_u + (i + 1) * n_stream
                l0 = off_l + i * n_stream
                l1 = off_l + (i + 1) * n_stream
                tris.append([u0, l0, u1]); tris.append([u1, l0, l1])
            # TE cap
            for i in range(n_span - 1):
                u0 = off_u + i * n_stream + n_stream - 1
                u1 = off_u + (i + 1) * n_stream + n_stream - 1
                l0 = off_l + i * n_stream + n_stream - 1
                l1 = off_l + (i + 1) * n_stream + n_stream - 1
                tris.append([u0, u1, l0]); tris.append([u1, l1, l0])
            # Symmetry cap (i=0) — only in non-mirrored mode does this
            # actually cap; in mirror mode the right and left halves
            # would share it, and we output both sides so the interior
            # is double-covered.  Downstream STL consumers handle this.
            if not mirror:
                for j in range(n_stream - 1):
                    tris.append([off_u + j, off_u + j + 1, off_l + j])
                    tris.append([off_u + j + 1, off_l + j + 1, off_l + j])
            # Tip cap (i = n_span-1)
            it = n_span - 1
            for j in range(n_stream - 1):
                u0 = off_u + it * n_stream + j
                u1 = u0 + 1
                l0 = off_l + it * n_stream + j
                l1 = l0 + 1
                tris.append([u0, l0, u1]); tris.append([u1, l0, l1])

        return vertices, np.asarray(tris, dtype=int)

    # ================================================================== #
    #  Summary (tab info panel)                                           #
    # ================================================================== #

    def get_body_geometry(self) -> dict:
        """Summary dict the tab reads into the info panel."""
        info = {
            "M_inf": float(np.mean(self._mach_per_station)),
            "beta_deg": self.beta_deg,
            "length": self.length,
            "height": self.height,
            "width": self.width,
            "x_LE": self.x_LE,
            "mach_distribution": self._mach_per_station.copy(),
            "n_distribution": self._n_per_station.copy(),
            "cone_angles_deg": self.cone_angles_deg.copy(),
            "z_stations": self._z_all.copy(),
            "icc_distribution": self._icc_per_station.copy(),
        }
        try:
            info["volume"] = self.compute_volume()
            info["planform_area"] = self.planform_area()
            info["vol_efficiency_corda"] = self.volumetric_efficiency("corda")
            info["vol_efficiency_box"] = self.volumetric_efficiency("box")
        except Exception:
            pass
        return info

    def summary(self) -> dict:
        return self.get_body_geometry()

    # ================================================================== #
    #  Plot helpers (spec API; optional, GUI has its own canvases)        #
    # ================================================================== #

    def plot_3d(self, ax=None, show_shock: bool = False):  # pragma: no cover
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        for s in self.upper_surface_streams:
            ax.plot(s[:, 2], s[:, 0], s[:, 1], color="steelblue", lw=0.6)
        for s in self.lower_surface_streams:
            ax.plot(s[:, 2], s[:, 0], s[:, 1], color="indianred", lw=0.6)
        le = self.leading_edge
        ax.plot(le[:, 2], le[:, 0], le[:, 1], "k-", lw=1.5)
        return ax

    def __repr__(self):
        return (f"VMPLOWaverider(beta={self.beta_deg:.2f}°, "
                f"L={self.length}, W={self.width}, H={self.height}, "
                f"n_planes={self.n_planes}, n_streamwise={self.n_streamwise})")


# ---------------------------------------------------------------------- #
#  Small helper                                                           #
# ---------------------------------------------------------------------- #

def _build_vmplo_solid_iso(upper_streams, lower_streams,
                           centerline_upper, centerline_lower,
                           te_upper, te_lower):
    """Build a watertight waverider half-solid using surface iso-curves.

    This is a variant of :func:`cad_export.build_waverider_solid` that
    closes the back, symmetry, and wingtip faces along the **actual
    iso-curves of the upper/lower B-spline surfaces**, rather than
    rebuilding separate spline edges through the ``te_upper`` /
    ``centerline_upper`` / etc. point arrays.  The separate-spline
    approach creates curves that agree with the surface only at the
    corner points and drift by several centimetres in between —
    visible as an open back face in CAD viewers and flagged as an
    invalid solid by ``BRepCheck_Analyzer``.

    The four boundary edges of every structured B-spline face built by
    :func:`cad_export._make_bspline_face` are, in the order
    :func:`face.Edges` returns them:

        u=0 (centerline), v=0 (LE), u=1 (tip), v=1 (TE / base plane)

    We extract these for both upper and lower faces, then assemble
    back + symmetry + optional wingtip faces from the extracted edges
    plus straight line segments where needed.  Sewing with a
    generous tolerance then produces a closed shell.
    """
    import cadquery as cq
    from waverider_generator.cad_export import (
        _make_bspline_face, _sew_faces_to_solid, _build_wingtip_face,
    )
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    # 1. Build the main upper/lower NURBS surfaces
    upper_faces = _make_bspline_face(upper_streams)
    lower_faces = _make_bspline_face(lower_streams)
    upper_face = upper_faces[0]
    lower_face = lower_faces[0]

    # 2. Extract the four boundary edges of each face by classifying
    #    each edge based on where its midpoint lies geometrically:
    #
    #      - LE edge:        all z-values vary, x min  → small x
    #      - TE edge:        all z-values vary, x max  → x ≈ L
    #      - centerline edge: all y-values vary, z ≈ 0
    #      - wingtip  edge:  all y-values vary, z ≈ W (may be degenerate)
    #
    #    This is more robust than relying on a fixed Edges() ordering.
    def classify_edges(face, te_points, le_points, cl_points, tip_points):
        """Return dict {name: cq.Edge} identifying the 4 boundary edges.

        Classification uses only edge endpoints, not midpoints — safer
        for degenerate (zero-length) edges at poles.
        """
        x_TE = float(np.mean(te_points[:, 0]))
        x_LE = float(np.mean(le_points[:, 0]))
        z_CL = float(cl_points[0, 2])
        z_TIP = float(tip_points[0, 2])

        # Collect all edges with endpoint info
        edge_info = []
        exp = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
        while exp.More():
            e = cq.Edge(TopoDS.Edge_s(exp.Current()))
            p0, p1 = e.startPoint(), e.endPoint()
            midx = 0.5 * (p0.x + p1.x)
            midy = 0.5 * (p0.y + p1.y)
            midz = 0.5 * (p0.z + p1.z)
            dx = abs(p1.x - p0.x)
            dz = abs(p1.z - p0.z)
            edge_info.append((e, midx, midy, midz, dx, dz))
            exp.Next()

        edges = {}
        # Classify by which edges have near-constant x (spanwise-running
        # edges: LE and TE) vs near-constant z (streamwise-running
        # edges: centerline and wingtip).
        span_total = max(z_TIP - z_CL, 1e-9)
        chord_total = max(x_TE - x_LE, 1e-9)

        for (e, mx, my, mz, dx, dz) in edge_info:
            # spanwise edge: dz dominates, dx is small
            is_spanwise = (dz > 0.5 * span_total) or (dx < 0.1 * chord_total)
            if is_spanwise:
                # LE vs TE: based on mean x
                if abs(mx - x_LE) < abs(mx - x_TE):
                    key = "LE"
                else:
                    key = "TE"
            else:
                # streamwise edge (LE→TE).  Centerline vs wingtip by z.
                if abs(mz - z_CL) < abs(mz - z_TIP):
                    key = "centerline"
                else:
                    key = "wingtip"

            # Avoid overwriting if we've seen this key already — pick
            # the better-matching candidate
            if key in edges:
                # Keep whichever has midpoint closer to the canonical
                # position for that class
                prev = edges[key]
                pp0 = prev.startPoint(); pp1 = prev.endPoint()
                prev_mx = 0.5 * (pp0.x + pp1.x)
                if key == "LE":
                    if abs(mx - x_LE) < abs(prev_mx - x_LE):
                        edges[key] = e
                elif key == "TE":
                    if abs(mx - x_TE) < abs(prev_mx - x_TE):
                        edges[key] = e
                # centerline / wingtip: keep the first one
            else:
                edges[key] = e
        return edges

    u_edges = classify_edges(upper_face, te_upper,
                              np.array([upper_streams[i][0] for i in range(len(upper_streams))]),
                              centerline_upper,
                              upper_streams[-1])
    l_edges = classify_edges(lower_face, te_lower,
                              np.array([lower_streams[i][0] for i in range(len(lower_streams))]),
                              centerline_lower,
                              lower_streams[-1])

    # If edge classification missed something, fall back: raise so
    # the caller can use the old builder.
    for label in ("TE", "centerline"):
        if label not in u_edges:
            raise RuntimeError(
                f"Could not classify upper face's {label} edge — "
                "iso-curve builder requires clean boundary edges.")
        if label not in l_edges:
            raise RuntimeError(
                f"Could not classify lower face's {label} edge.")

    # 3. Back face: bounded by upper TE iso-curve, lower TE iso-curve,
    #    + sym line at centerline + optional wingtip closure.
    sym_start_upper = cq.Vector(*centerline_upper[-1])
    sym_start_lower = cq.Vector(*centerline_lower[-1])
    sym_edge_back = cq.Edge.makeLine(sym_start_upper, sym_start_lower)

    back_edges = [u_edges["TE"], sym_edge_back, l_edges["TE"]]
    # Optional wingtip closure if TE endpoints at the tip side don't coincide
    wt_upper = tuple(float(c) for c in te_upper[-1])
    wt_lower = tuple(float(c) for c in te_lower[-1])
    wt_gap = np.linalg.norm(np.array(wt_upper) - np.array(wt_lower))
    if wt_gap > 1e-8:
        back_edges.append(cq.Edge.makeLine(cq.Vector(*wt_upper),
                                           cq.Vector(*wt_lower)))

    try:
        back_face = cq.Face.makeFromWires(cq.Wire.assembleEdges(back_edges))
    except Exception:
        # Fallback: ruled loft between the two TE iso-curves
        from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
        from OCP.TopAbs import TopAbs_FACE
        loft = BRepOffsetAPI_ThruSections(False, True)
        loft.AddWire(cq.Wire.assembleEdges([u_edges["TE"]]).wrapped)
        loft.AddWire(cq.Wire.assembleEdges([l_edges["TE"]]).wrapped)
        loft.Build()
        f_exp = TopExp_Explorer(loft.Shape(), TopAbs_FACE)
        if not f_exp.More():
            raise RuntimeError("Back-face ruled loft produced no faces.")
        back_face = cq.Face(TopoDS.Face_s(f_exp.Current()))

    # 4. Symmetry face: bounded by centerline iso-curves + LE/TE line segments.
    nose_upper = cq.Vector(*centerline_upper[0])
    nose_lower = cq.Vector(*centerline_lower[0])
    nose_line = None
    nose_diff = np.linalg.norm(np.asarray(centerline_upper[0]) -
                               np.asarray(centerline_lower[0]))
    if nose_diff > 1e-8:
        nose_line = cq.Edge.makeLine(nose_lower, nose_upper)

    sym_edges = [u_edges["centerline"], sym_edge_back, l_edges["centerline"]]
    if nose_line is not None:
        sym_edges.append(nose_line)

    try:
        sym_face = cq.Face.makeFromWires(cq.Wire.assembleEdges(sym_edges))
    except Exception as exc:
        raise RuntimeError(f"Symmetry face assembly failed: {exc}") from exc

    # 5. Wingtip face: the classical pinch case returns None from
    #    _build_wingtip_face; non-pinch uses a ThruSections loft between
    #    the two wingtip iso-curves.
    wingtip_face = None
    if "wingtip" in u_edges and "wingtip" in l_edges:
        u_wt = u_edges["wingtip"]
        l_wt = l_edges["wingtip"]
        # Endpoint-based degeneracy check (avoids positionAt on poles)
        u_ends = (u_wt.startPoint() - u_wt.endPoint()).Length
        l_ends = (l_wt.startPoint() - l_wt.endPoint()).Length
        # Endpoint separation between upper and lower wingtip edges
        us_sep = (u_wt.startPoint() - l_wt.startPoint()).Length
        ue_sep = (u_wt.endPoint() - l_wt.endPoint()).Length
        if u_ends > 1e-6 and l_ends > 1e-6 and (us_sep > 1e-6 or ue_sep > 1e-6):
            try:
                from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
                from OCP.TopAbs import TopAbs_FACE
                loft = BRepOffsetAPI_ThruSections(False, True)
                loft.AddWire(cq.Wire.assembleEdges([u_wt]).wrapped)
                loft.AddWire(cq.Wire.assembleEdges([l_wt]).wrapped)
                loft.Build()
                f_exp = TopExp_Explorer(loft.Shape(), TopAbs_FACE)
                if f_exp.More():
                    wingtip_face = cq.Face(TopoDS.Face_s(f_exp.Current()))
            except Exception as exc:
                print(f"[VMPLO iso] Wingtip loft failed ({exc}); "
                      "solid will still be closed via sewing.")

    # 6. Assemble + sew
    all_faces = upper_faces + lower_faces + [back_face, sym_face]
    if wingtip_face is not None:
        all_faces.append(wingtip_face)
        print(f"[VMPLO iso] Sewing {len(all_faces)} faces (with wingtip)")
    else:
        print(f"[VMPLO iso] Sewing {len(all_faces)} faces (no wingtip — pinch)")

    # Use a moderately generous sewing tolerance; the iso-curve edges
    # should be geometrically identical at the topology level, but
    # OpenCASCADE may require some tolerance to dedupe them.
    return _sew_faces_to_solid(all_faces, tolerance=1e-3)


def _resample_grid(M: np.ndarray, nz_new: int, nx_new: int) -> np.ndarray:
    nz_old, nx_old = M.shape
    zu = np.linspace(0, 1, nz_old)
    xu = np.linspace(0, 1, nx_old)
    zn = np.linspace(0, 1, nz_new)
    xn = np.linspace(0, 1, nx_new)
    # Interpolate row-wise then column-wise
    row = np.array([np.interp(xn, xu, M[i]) for i in range(nz_old)])
    out = np.array([np.interp(zn, zu, row[:, j]) for j in range(nx_new)]).T
    return out
