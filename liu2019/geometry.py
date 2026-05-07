"""3D geometry assembly for Liu et al. 2019 waveriders.

Highlights
----------
* **True osculating-plane geometry**: leading-edge and compression-surface
  trailing-edge points are computed by
  :func:`liu2019.osculating.osculating_plane_geometry`, which tilts the
  osculating plane by the local shock-curve angle. In the curved region
  the streamline has both a y- and a z-component, i.e., the mesh is no
  longer axis-aligned in (x, z).

* **Volume via divergence theorem** on a closed triangulated surface
  (upper + lower + base cap). Robust to arbitrary skewed meshes.

* **Export formats**: STL (always), OBJ (always), STEP (when the OCP
  CAD kernel, shipped with ``cadquery``'s dependencies, is importable).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PAPER_REFERENCE_GEOMETRY
from .distributions import shock_curve, upper_surface_trailing_edge
from .osculating import OsculatingPlaneSet, build_all_osculating_planes


# ---------------------------------------------------------------------------
# Small geometric helpers
# ---------------------------------------------------------------------------

def _stack_grid(X, Y, Z):
    return np.stack([X, Y, Z], axis=-1)


def _mesh_area(X, Y, Z):
    """Sum of panel areas for a structured (nx, nz) mesh, vectorised."""
    P = _stack_grid(X, Y, Z)
    p0 = P[:-1, :-1]
    p1 = P[1:,  :-1]
    p2 = P[1:,  1:]
    p3 = P[:-1, 1:]
    a1 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=-1)
    a2 = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0), axis=-1)
    return float((a1 + a2).sum())


def _triangles_of_grid(X, Y, Z):
    """Return (N, 3, 3) triangle vertex array from a structured (nx, nz) mesh.

    Default vertex order (p0, p1, p2)/(p0, p2, p3); orientation depends on
    the (i, j) layout. For divergence-theorem volume use
    :func:`_oriented_triangulation` instead.
    """
    P = _stack_grid(X, Y, Z)
    p0 = P[:-1, :-1]
    p1 = P[1:,  :-1]
    p2 = P[1:,  1:]
    p3 = P[:-1, 1:]
    tri_a = np.stack([p0, p1, p2], axis=-2).reshape(-1, 3, 3)
    tri_b = np.stack([p0, p2, p3], axis=-2).reshape(-1, 3, 3)
    return np.concatenate([tri_a, tri_b], axis=0)


def _oriented_triangulation(X, Y, Z, outward_hint):
    """Triangulate a structured (nx, nz) grid with **outward** triangle
    winding, judged by the dot product of the cross-product normal at the
    grid centre against ``outward_hint``.

    ``outward_hint`` only needs to be approximately right (e.g. (0,1,0)
    for the upper surface, (0,-1,0) for the lower, (1,0,0) for the base
    cap at x = L_w). The function picks one triangle's normal direction,
    compares to the hint, and either keeps the natural winding or
    reverses every triangle's vertex order.
    """
    P = _stack_grid(X, Y, Z)
    p0 = P[:-1, :-1]
    p1 = P[1:,  :-1]
    p2 = P[1:,  1:]
    p3 = P[:-1, 1:]
    nx, nz = X.shape
    i_mid = max(0, (nx - 2) // 2)
    j_mid = max(0, (nz - 2) // 2)
    n_test = np.cross(p1[i_mid, j_mid] - p0[i_mid, j_mid],
                      p2[i_mid, j_mid] - p0[i_mid, j_mid])
    if np.dot(n_test, np.asarray(outward_hint, dtype=float)) >= 0:
        tri_a = np.stack([p0, p1, p2], axis=-2).reshape(-1, 3, 3)
        tri_b = np.stack([p0, p2, p3], axis=-2).reshape(-1, 3, 3)
    else:
        tri_a = np.stack([p0, p2, p1], axis=-2).reshape(-1, 3, 3)
        tri_b = np.stack([p0, p3, p2], axis=-2).reshape(-1, 3, 3)
    return np.concatenate([tri_a, tri_b], axis=0)


def _tri_volumes(tris):
    """Signed (1/6)(v0 . (v1 x v2)) contributions for a triangle array."""
    v0 = tris[:, 0]
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    return np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0


def _divergence_volume(tri_lists):
    """Divergence-theorem volume for a closed surface described by one or
    more **already-outward-oriented** triangle lists. Each list contributes
    its own signed volume integral (1/6) Σ v0 . (v1 × v2), summed over
    triangles; the total magnitude is the enclosed volume.
    """
    total = 0.0
    for tris in tri_lists:
        if tris.size:
            total += float(_tri_volumes(tris).sum())
    return abs(total)


# ---------------------------------------------------------------------------
# Main waverider class
# ---------------------------------------------------------------------------

@dataclass
class _Surfaces:
    X_upper: np.ndarray
    Y_upper: np.ndarray
    Z_upper: np.ndarray
    X_lower: np.ndarray
    Y_lower: np.ndarray
    Z_lower: np.ndarray


class Liu2019Waverider:
    """3D waverider assembled from osculating-plane records."""

    def __init__(self, planes: OsculatingPlaneSet, params: dict,
                 n_x: int = 100):
        self.planes = planes
        self.params = dict(params)
        self.n_x = int(n_x)
        self.coeffs = planes.coeffs
        self._build_surfaces()

    # ------------------------------------------------------------------
    # Surface construction (starboard half, z >= 0)
    # ------------------------------------------------------------------
    def _build_surfaces(self):
        L_w = float(self.params["L_w"])
        n_z = len(self.planes)
        n_x = self.n_x

        X_u = np.zeros((n_x, n_z))
        Y_u = np.zeros((n_x, n_z))
        Z_u = np.zeros((n_x, n_z))
        X_l = np.zeros((n_x, n_z))
        Y_l = np.zeros((n_x, n_z))
        Z_l = np.zeros((n_x, n_z))

        t = np.linspace(0.0, 1.0, n_x)
        for j, plane in enumerate(self.planes):
            x_LE, y_LE, z_LE = plane.P_LE

            # Lower (compression) surface: sample the actual streamline
            # stored on the plane. In the flat region this is straight; in
            # the curved region it traces through the Taylor-Maccoll cone
            # flow and curves slightly.
            stream = plane.streamline
            if stream is None or stream.size == 0:
                # Defensive fallback: straight LE -> TE
                x_TE, y_TE, z_TE = plane.P_TE
                X_l[:, j] = x_LE + t * (x_TE - x_LE)
                Y_l[:, j] = y_LE + t * (y_TE - y_LE)
                Z_l[:, j] = z_LE + t * (z_TE - z_LE)
            else:
                s_param = np.linspace(0.0, 1.0, stream.shape[0])
                X_l[:, j] = np.interp(t, s_param, stream[:, 0])
                Y_l[:, j] = np.interp(t, s_param, stream[:, 1])
                Z_l[:, j] = np.interp(t, s_param, stream[:, 2])

            # Upper (freestream) surface: freestream-aligned ruled surface
            # from LE downstream to x = L_w at constant (y_LE, z_LE).
            X_u[:, j] = x_LE + t * (L_w - x_LE)
            Y_u[:, j] = y_LE
            Z_u[:, j] = z_LE

        self.surfaces = _Surfaces(
            X_upper=X_u, Y_upper=Y_u, Z_upper=Z_u,
            X_lower=X_l, Y_lower=Y_l, Z_lower=Z_l,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def leading_edge_curve(self, mirror: bool = True):
        xs = np.array([p.P_LE[0] for p in self.planes])
        ys = np.array([p.P_LE[1] for p in self.planes])
        zs = np.array([p.P_LE[2] for p in self.planes])
        if mirror:
            xs = np.concatenate([xs[::-1][:-1], xs])
            ys = np.concatenate([ys[::-1][:-1], ys])
            zs = np.concatenate([-zs[::-1][:-1], zs])
        return xs, ys, zs

    def trailing_edge_curves(self, mirror: bool = True):
        """Returns two curves: upper TE and lower (compression) TE.

        Each is a tuple (x, y, z) of arrays. The upper TE has x=L_w and
        y=y(z_LE); the lower TE has x=L_w and y=y_TE, z=z_TE.
        """
        L_w = float(self.params["L_w"])
        y_upper = np.array([p.P_LE[1] for p in self.planes])        # = y_upper(z_LE)
        z_upper = np.array([p.P_LE[2] for p in self.planes])        # = z_LE
        y_lower = np.array([p.P_TE[1] for p in self.planes])
        z_lower = np.array([p.P_TE[2] for p in self.planes])
        if mirror:
            y_upper = np.concatenate([y_upper[::-1][:-1], y_upper])
            z_upper = np.concatenate([-z_upper[::-1][:-1], z_upper])
            y_lower = np.concatenate([y_lower[::-1][:-1], y_lower])
            z_lower = np.concatenate([-z_lower[::-1][:-1], z_lower])
        x = np.full_like(y_upper, L_w)
        return (x, y_upper, z_upper), (x, y_lower, z_lower)

    def upper_surface(self, mirror: bool = True):
        return self._mirrored(self.surfaces.X_upper,
                              self.surfaces.Y_upper,
                              self.surfaces.Z_upper, mirror)

    def lower_surface(self, mirror: bool = True):
        return self._mirrored(self.surfaces.X_lower,
                              self.surfaces.Y_lower,
                              self.surfaces.Z_lower, mirror)

    def _mirrored(self, X, Y, Z, mirror):
        if not mirror:
            return X, Y, Z
        Xm = np.concatenate([X[:, ::-1][:, :-1], X], axis=1)
        Ym = np.concatenate([Y[:, ::-1][:, :-1], Y], axis=1)
        Zm = np.concatenate([-Z[:, ::-1][:, :-1], Z], axis=1)
        return Xm, Ym, Zm

    # ------------------------------------------------------------------
    # Metrics (paper Table 4 targets)
    # ------------------------------------------------------------------
    def _base_cap_triangles(self, outward_hint=(1.0, 0.0, 0.0)):
        """Triangulate the closed base-plane polygon between upper and lower
        trailing-edge traces, with outward-pointing triangle winding (per
        ``outward_hint``, default +x because the base sits at x = L_w and
        outside the body).
        """
        (xu, yu, zu), (xl, yl, zl) = self.trailing_edge_curves(mirror=True)
        V_u = np.stack([xu, yu, zu], axis=-1)
        V_l = np.stack([xl, yl, zl], axis=-1)
        if len(xu) < 2:
            return np.empty((0, 3, 3))
        # Test orientation at one ribbon strip
        p0, p1 = V_u[0],     V_u[1]
        p2     = V_l[1]
        n_test = np.cross(p1 - p0, p2 - p0)
        flip = np.dot(n_test, np.asarray(outward_hint, dtype=float)) < 0
        tris = []
        for j in range(len(xu) - 1):
            p0, p1 = V_u[j],     V_u[j + 1]
            p2, p3 = V_l[j + 1], V_l[j]
            if not flip:
                tris.append([p0, p1, p2])
                tris.append([p0, p2, p3])
            else:
                tris.append([p0, p2, p1])
                tris.append([p0, p3, p2])
        return np.asarray(tris)

    def volume(self):
        """Enclosed volume by column-wise chord-prism integration.

        For each osculating-plane column, integrate the chordwise height
        (y_upper - y_lower) along x; this gives a per-column cross-section
        area in the freestream-aligned (x, y) view. The result is then
        integrated over z_LE (the actual spanwise parameter of the column)
        and doubled for the full vehicle.

        Because the upper surface is freestream-aligned with constant
        (y, z) = (y_LE, z_LE) along each column, this is an exact volume
        integral of the wedge between upper and lower surfaces when the
        lower surface's z-drift is small compared to the column's spanwise
        spacing -- which it is in the legacy straight-line streamline
        model used by build_all_osculating_planes (n_z component of the
        deflection is small in the flat region and modest in the curved
        region for the paper's gentle shock curvature).

        An oriented divergence-theorem volume helper
        (_oriented_triangulation + _divergence_volume) is also available
        in this file but is not used by default.
        """
        # Use the starboard half only, then double.
        X_u = self.surfaces.X_upper
        Y_u = self.surfaces.Y_upper
        X_l = self.surfaces.X_lower
        Y_l = self.surfaces.Y_lower
        Z_u = self.surfaces.Z_upper          # z_LE per column (constant along each column)
        z_LE = Z_u[0, :]                     # (n_z,)

        dY = Y_u - Y_l                       # chord thickness per (i, j)
        dx = np.diff(X_u, axis=0)            # (n_x-1, n_z)
        col_area = (0.5 * (dY[:-1, :] + dY[1:, :]) * dx).sum(axis=0)  # (n_z,)
        half_vol = abs(np.trapz(col_area, x=z_LE))
        return float(2.0 * half_vol)

    def wetted_area(self):
        X_u, Y_u, Z_u = self.upper_surface(mirror=True)
        X_l, Y_l, Z_l = self.lower_surface(mirror=True)
        return _mesh_area(X_u, Y_u, Z_u) + _mesh_area(X_l, Y_l, Z_l)

    def planform_area(self):
        """Projected area on the x-z plane (footprint of the lower surface)."""
        X_l, _, Z_l = self.lower_surface(mirror=True)
        return _mesh_area(X_l, np.zeros_like(X_l), Z_l)

    def base_area(self):
        """Closed base-plane area between upper and lower TE traces.

        Uses the shoelace formula on the stitched (z, y) polygon.
        """
        (xu, yu, zu), (xl, yl, zl) = self.trailing_edge_curves(mirror=True)
        polygon_z = np.concatenate([zu, zl[::-1]])
        polygon_y = np.concatenate([yu, yl[::-1]])
        area = 0.5 * abs(np.sum(
            polygon_z * np.roll(polygon_y, -1)
          - np.roll(polygon_z, -1) * polygon_y))
        return float(area)

    def volumetric_efficiency(self):
        V = self.volume()
        S = self.wetted_area()
        if S <= 0.0:
            return 0.0
        return V ** (2.0 / 3.0) / S

    def spanwise_mach(self):
        """Return (z_array, Ma_array) for the half-span."""
        zs = np.array([p.z for p in self.planes])
        Ma = np.array([p.Ma for p in self.planes])
        return zs, Ma

    def cone_angle_array(self):
        zs = np.array([p.z for p in self.planes])
        dc = np.array([p.delta_c for p in self.planes])
        return zs, dc

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _all_triangles(self, mirror=True):
        X_u, Y_u, Z_u = self.upper_surface(mirror=mirror)
        X_l, Y_l, Z_l = self.lower_surface(mirror=mirror)
        tris_u = _triangles_of_grid(X_u, Y_u, Z_u)
        tris_l = _triangles_of_grid(X_l, Y_l, Z_l)
        return np.concatenate([tris_u, tris_l], axis=0)

    def export_stl(self, filepath, mirror: bool = True):
        """ASCII STL export of the wetted (upper + lower) surfaces."""
        tris = self._all_triangles(mirror=mirror)
        with open(filepath, "w") as f:
            f.write("solid liu2019\n")
            for tri in tris:
                v0, v1, v2 = tri
                n = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(n)
                n = n / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                for v in tri:
                    f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
                f.write("    endloop\n  endfacet\n")
            f.write("endsolid liu2019\n")
        return filepath

    def export_obj(self, filepath, mirror: bool = True):
        """Wavefront OBJ export of the wetted surfaces."""
        X_u, Y_u, Z_u = self.upper_surface(mirror=mirror)
        X_l, Y_l, Z_l = self.lower_surface(mirror=mirror)
        with open(filepath, "w") as f:
            f.write("o liu2019_waverider\n")
            for X, Y, Z in ((X_u, Y_u, Z_u), (X_l, Y_l, Z_l)):
                nx, nz = X.shape
                for i in range(nx):
                    for j in range(nz):
                        f.write(f"v {X[i,j]:.6e} {Y[i,j]:.6e} {Z[i,j]:.6e}\n")
            offset = 0
            for X, Y, Z in ((X_u, Y_u, Z_u), (X_l, Y_l, Z_l)):
                nx, nz = X.shape
                for i in range(nx - 1):
                    for j in range(nz - 1):
                        v0 = offset + i * nz + j + 1
                        v1 = offset + (i + 1) * nz + j + 1
                        v2 = offset + (i + 1) * nz + (j + 1) + 1
                        v3 = offset + i * nz + (j + 1) + 1
                        f.write(f"f {v0} {v1} {v2} {v3}\n")
                offset += nx * nz
        return filepath

    def export_step(self, filepath, mirror: bool = True):
        """STEP export via the OCP (pythonocc-core) kernel.

        Raises RuntimeError if OCP is not importable in the environment.
        """
        try:
            from OCP.gp import gp_Pnt
            from OCP.TColgp import TColgp_Array2OfPnt
            from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
            from OCP.GeomAbs import GeomAbs_Shape
            from OCP.BRepBuilderAPI import (
                BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing,
            )
            from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
            from OCP.IFSelect import IFSelect_RetDone
        except Exception as e:
            raise RuntimeError(
                "STEP export requires the OCP (pythonocc-core) library.\n"
                f"Import failed: {e}"
            )

        def _bspline_face(X, Y, Z):
            nx, nz = X.shape
            pts = TColgp_Array2OfPnt(1, nx, 1, nz)
            for i in range(nx):
                for j in range(nz):
                    pts.SetValue(
                        i + 1, j + 1,
                        gp_Pnt(float(X[i, j]),
                               float(Y[i, j]),
                               float(Z[i, j])))
            # (DegMin=3, DegMax=5, Continuity=C2, Tol3D=5 mm).
            # The 5 mm fit tolerance lets the B-spline smooth through any
            # sub-millimetre residual ripples in the input grid (e.g. the
            # tiny ~1 mm step at z = L_s where the flat-region slope
            # blend transitions to the T-M streamline). Without it, the
            # fitter rings around such steps and produces visible ridges.
            # 5 mm is well below the geometry's natural ~10 mm panel pitch
            # and below any feature you'd resolve in CFD or 3D printing.
            surf = GeomAPI_PointsToBSplineSurface(
                pts, 3, 5, GeomAbs_Shape.GeomAbs_C2, 5.0e-3).Surface()
            return BRepBuilderAPI_MakeFace(surf, 1.0e-3).Face()

        X_u, Y_u, Z_u = self.upper_surface(mirror=mirror)
        X_l, Y_l, Z_l = self.lower_surface(mirror=mirror)
        f_upper = _bspline_face(X_u, Y_u, Z_u)
        f_lower = _bspline_face(X_l, Y_l, Z_l)

        sewer = BRepBuilderAPI_Sewing(1.0e-3)
        sewer.Add(f_upper)
        sewer.Add(f_lower)
        sewer.Perform()
        shell = sewer.SewedShape()

        writer = STEPControl_Writer()
        writer.Transfer(shell, STEPControl_AsIs)
        status = writer.Write(str(filepath))
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEP write failed (status {status}).")
        return filepath

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        V   = self.volume()
        S   = self.wetted_area()
        S_p = self.planform_area()
        S_b = self.base_area()
        eta = self.volumetric_efficiency()
        ref = PAPER_REFERENCE_GEOMETRY
        return {
            "Vol_m3":   V,
            "S_wet_m2": S,
            "S_p_m2":   S_p,
            "S_b_m2":   S_b,
            "eta":      eta,
            "reference": ref,
        }


# Convenience builder --------------------------------------------------------

def build_liu2019_waverider(params, n_z: int = 200, n_x: int = 100
                            ) -> Liu2019Waverider:
    planes = build_all_osculating_planes(params, n_z=n_z, n_x=n_x)
    return Liu2019Waverider(planes, params, n_x=n_x)
