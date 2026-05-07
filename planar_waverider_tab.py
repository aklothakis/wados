"""
Planar Waverider Design Tab

GUI tab implementing the 9-parameter planar waverider design method from:
  Jessen, Larsson, Brehm (2026) — "Comparative optimization of hypersonic
  waveriders using analytical and computational methods"
  Aerospace Science and Technology 172, 111703.
"""

import os
import numpy as np
import traceback
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox, QSplitter,
    QScrollArea, QComboBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from planar_waverider import PlanarWaverider
from planar_waverider_aero import PlanarWaveriderAero, atmosphere
from planar_waverider_optimizer import (
    PlanarWaveriderEvaluator, PlanarWaveriderOptimizer,
)

# Optional CAD export
try:
    from waverider_generator.cad_export import to_CAD
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
#  Background worker for geometry generation + aero analysis
# ──────────────────────────────────────────────────────────────────────

class StepExportWorker(QThread):
    """Exports STEP file in a background thread.

    Always exports the SHARP geometry (no blunting).  Blunting is applied
    only in the Python preview; CAD-level blunting will be done via
    external CAD API (e.g. Onshape) in the future.

    Uses approximation B-spline surfaces (GeomAPI_PointsToBSplineSurface.Init)
    for smooth, artifact-free STEP geometry.
    """

    progress = pyqtSignal(str)
    finished = pyqtSignal(str)   # filename
    error = pyqtSignal(str)

    def __init__(self, waverider, filename, parent=None):
        super().__init__(parent)
        self.waverider = waverider
        self.filename = filename

    # ------------------------------------------------------------------
    #  Build sharp-LE streams from x_le to L
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sharp_streams(wr, y_positions, n_pts=65):
        """Build sharp-LE streams from x_le to L, uniform spacing.

        Always produces the unblunted (sharp) geometry regardless of
        whatever blunting method the waverider preview uses.

        Parameters
        ----------
        wr : PlanarWaverider
        y_positions : ndarray, shape (n_span,)
        n_pts : int
            Points per stream.

        Returns
        -------
        upper_streams, lower_streams : list of ndarray (n_pts, 3)
            Coordinates in meters.
        """
        theta_base = np.radians(wr.wedge_angle_deg)
        L = wr.length

        upper_streams = []
        lower_streams = []

        for y_j in y_positions:
            x_le_j = float(wr._leading_edge_x(y_j))
            z_le_j = float(wr._leading_edge_z(x_le_j))
            T_star_j = float(wr._angle_perturbation(
                np.array([abs(y_j)]))[0])
            theta_j = T_star_j * theta_base
            chord = L - x_le_j

            t = np.linspace(0.0, 1.0, n_pts)
            x_arr = (x_le_j + t * chord if chord > 1e-9
                     else np.full(n_pts, x_le_j))
            z_upper = np.full(n_pts, z_le_j)
            z_lower = z_le_j - np.tan(theta_j) * (x_arr - x_le_j)
            y_arr = np.full(n_pts, y_j)

            upper_streams.append(
                np.column_stack([x_arr, y_arr, z_upper]))
            lower_streams.append(
                np.column_stack([x_arr, y_arr, z_lower]))

        return upper_streams, lower_streams

    # ------------------------------------------------------------------
    #  Approximation B-spline surface (smooths curvature breaks)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_approx_bspline_face(streams, tol3d=1e-4, label="surface"):
        """Build an approximation B-spline surface from a structured grid.

        Uses GeomAPI_PointsToBSplineSurface.Init() in approximation mode
        (least-squares fit within tolerance) instead of .Interpolate()
        (exact fit).  This smooths out the G2 curvature break at the
        arc-to-flat junction that causes Gibbs oscillation with exact
        interpolation.

        Parameters
        ----------
        streams : list of ndarray (n_pts, 3)
            Streamlines ordered centerline (i=0) to wingtip (i=-1).
        tol3d : float
            Max deviation from data points [meters].  Default 1e-4 m
            = 0.1 mm — visually imperceptible.
        label : str
            Name for diagnostic messages.

        Returns
        -------
        list of cq.Face
        """
        import cadquery as cq
        from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
        from OCP.GeomAbs import GeomAbs_C2
        from OCP.TColgp import TColgp_Array2OfPnt
        from OCP.gp import gp_Pnt
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

        n_u = len(streams)
        n_v = streams[0].shape[0]

        grid = TColgp_Array2OfPnt(1, n_u, 1, n_v)
        for i in range(n_u):
            pts = streams[i]
            for j in range(n_v):
                grid.SetValue(i + 1, j + 1,
                              gp_Pnt(float(pts[j][0]),
                                     float(pts[j][1]),
                                     float(pts[j][2])))

        # Approximation: degree 3–8, C2 continuity, within tol3d
        approx = GeomAPI_PointsToBSplineSurface()
        approx.Init(grid, 3, 8, GeomAbs_C2, tol3d)

        if not approx.IsDone():
            # Fallback: relax tolerance 10x
            print(f"[STEP {label}] Approx failed at tol={tol3d:.1e}, "
                  f"retrying at {tol3d*10:.1e}")
            approx2 = GeomAPI_PointsToBSplineSurface()
            approx2.Init(grid, 3, 8, GeomAbs_C2, tol3d * 10)
            if not approx2.IsDone():
                raise RuntimeError(
                    f"{label}: approx B-spline failed on "
                    f"{n_u}x{n_v} grid (tol={tol3d*10:.1e})")
            bspline_surface = approx2.Surface()
        else:
            bspline_surface = approx.Surface()

        face_builder = BRepBuilderAPI_MakeFace(bspline_surface, 1e-6)
        face_builder.Build()
        if not face_builder.IsDone():
            raise RuntimeError(
                f"{label}: MakeFace failed "
                f"(error={face_builder.Error()})")

        print(f"[STEP {label}] Approx B-spline OK "
              f"({n_u}x{n_v}, tol={tol3d:.1e})")
        return [cq.Face(face_builder.Face())]

    # ------------------------------------------------------------------
    #  Main export pipeline
    # ------------------------------------------------------------------

    def run(self):
        try:
            import cadquery as cq
            from cadquery import exporters
            from waverider_generator.cad_export import _sew_faces_to_solid

            wr = self.waverider
            L = wr.length
            half_w = wr.width / 2.0

            # ── Right-half spanwise stations (y >= 0) ────────────────
            # Include exact wingtip (y = half_w) so the B-spline
            # surface converges to a sharp point there.
            n_main = 25
            y_core = np.linspace(0, half_w * 0.96, n_main)
            tip_fracs = np.array([0.002, 0.005, 0.01, 0.02, 0.04, 0.08])
            y_tips = half_w * (1.0 - tip_fracs)
            y_all = np.unique(np.concatenate(
                [y_core, y_tips, [half_w]]))
            mask = np.concatenate([[True], np.diff(y_all) > 1e-10])
            y_positions = y_all[mask]
            n_stations = len(y_positions)

            # ── Build sharp streams in METERS ─────────────────────
            self.progress.emit(
                f"Building sharp streams ({n_stations} stations)...")
            upper_streams, lower_streams = \
                self._build_sharp_streams(wr, y_positions, n_pts=65)
            n_pts = upper_streams[0].shape[0]
            print(f"[STEP] {n_stations} stations, "
                  f"{n_pts} pts/stream (sharp)")

            # ── Approximation B-spline surfaces ──────────────────────
            self.progress.emit("Fitting approx B-spline surfaces...")
            tol = 1e-4  # 0.1 mm
            upper_faces = self._make_approx_bspline_face(
                upper_streams, tol3d=tol, label="upper")
            lower_faces = self._make_approx_bspline_face(
                lower_streams, tol3d=tol, label="lower")

            # ── Closure faces ────────────────────────────────────────
            self.progress.emit("Building closure faces...")

            te_upper = np.array([s[-1] for s in upper_streams])
            te_lower = np.array([s[-1] for s in lower_streams])

            # Back face (trailing edge, x = L)
            e_te_upper = cq.Edge.makeSpline(
                [cq.Vector(*map(float, p)) for p in te_upper])
            e_te_lower = cq.Edge.makeSpline(
                [cq.Vector(*map(float, p)) for p in te_lower])
            e_center_te = cq.Edge.makeLine(
                cq.Vector(*map(float, te_upper[0])),
                cq.Vector(*map(float, te_lower[0])))
            back_edges = [e_te_upper, e_te_lower, e_center_te]
            tip_te_dist = float(np.linalg.norm(
                te_upper[-1] - te_lower[-1]))
            if tip_te_dist > 1e-8:
                e_tip_te = cq.Edge.makeLine(
                    cq.Vector(*map(float, te_upper[-1])),
                    cq.Vector(*map(float, te_lower[-1])))
                back_edges.append(e_tip_te)
            back_face = cq.Face.makeFromWires(
                cq.Wire.assembleEdges(back_edges))

            # Symmetry face (y = 0 plane)
            e_sym_upper = cq.Edge.makeSpline(
                [cq.Vector(*map(float, p))
                 for p in upper_streams[0]])
            e_sym_lower = cq.Edge.makeSpline(
                [cq.Vector(*map(float, p))
                 for p in lower_streams[0]])
            sym_edges = [e_center_te, e_sym_upper, e_sym_lower]
            # Upper and lower streams share the same nose point
            # when blunted, so check if a closing edge is needed
            nose_dist = float(np.linalg.norm(
                upper_streams[0][0] - lower_streams[0][0]))
            if nose_dist > 1e-8:
                e_nose = cq.Edge.makeLine(
                    cq.Vector(*map(float, lower_streams[0][0])),
                    cq.Vector(*map(float, upper_streams[0][0])))
                sym_edges.append(e_nose)
            sym_face = cq.Face.makeFromWires(
                cq.Wire.assembleEdges(sym_edges))

            # Wingtip face
            all_faces = (upper_faces + lower_faces
                         + [back_face, sym_face])
            tip_chord = float(np.linalg.norm(
                upper_streams[-1][-1] - upper_streams[-1][0]))
            if tip_te_dist > 5e-4 and tip_chord > 1e-3:
                try:
                    e_tip_upper = cq.Edge.makeSpline(
                        [cq.Vector(*map(float, p))
                         for p in upper_streams[-1]])
                    e_tip_lower = cq.Edge.makeSpline(
                        [cq.Vector(*map(float, p))
                         for p in lower_streams[-1]])
                    tip_edges = [e_tip_upper, e_tip_lower]
                    if tip_te_dist > 1e-8:
                        tip_edges.append(e_tip_te)
                    tip_nose_dist = float(np.linalg.norm(
                        upper_streams[-1][0]
                        - lower_streams[-1][0]))
                    if tip_nose_dist > 1e-8:
                        e_tip_nose = cq.Edge.makeLine(
                            cq.Vector(*map(float,
                                           lower_streams[-1][0])),
                            cq.Vector(*map(float,
                                           upper_streams[-1][0])))
                        tip_edges.append(e_tip_nose)
                    tip_face = cq.Face.makeFromWires(
                        cq.Wire.assembleEdges(tip_edges))
                    all_faces.append(tip_face)
                except Exception:
                    print("[STEP] Wingtip face skipped "
                          "(degenerate)")

            # ── Sew → scale → export ─────────────────────────────────
            self.progress.emit("Sewing right-half solid...")
            right_solid = _sew_faces_to_solid(all_faces)

            self.progress.emit("Scaling to mm...")
            right_solid = right_solid.scale(1000.0)
            result = cq.Workplane("XY").newObject([right_solid])

            self.progress.emit("Writing STEP file...")
            exporters.export(result, self.filename)

            # Export blunting metadata CSV alongside STEP
            csv_path = self._export_blunting_csv(wr, self.filename)
            if csv_path:
                self.progress.emit(
                    f"Blunting CSV: {os.path.basename(csv_path)}")

            self.finished.emit(self.filename)

        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    @staticmethod
    def _export_blunting_csv(wr, step_path):
        """Write per-station blunting parameters to CSV for CAD filleting.

        Always exported alongside the STEP file, even when blunting
        method is 'none'.  The CSV provides the per-station geometry
        (y, LE x/z, chord, wedge angle, R_eff) so the user can apply
        fillets in SolidWorks or other CAD tools.

        Coordinates are in millimeters (same scale as STEP).
        """
        base, _ = os.path.splitext(step_path)
        csv_path = base + '_blunting.csv'
        scale = 1000.0  # m -> mm

        meta = wr.blunting_metadata()
        method = meta.get('method', 'none') if meta else 'none'
        R_base = meta.get('R_base', 0.0) if meta else 0.0
        stations = meta.get('stations', []) if meta else []

        # Build station data from the waverider directly so the CSV
        # is always useful for CAD filleting, even with method='none'.
        theta_base = np.radians(wr.beta_deg)
        L = wr.length
        half_w = wr.width / 2.0

        # Use the same spanwise stations as the STEP export
        n_main = 25
        y_core = np.linspace(0, half_w * 0.96, n_main)
        tip_fracs = np.array([0.002, 0.005, 0.01, 0.02, 0.04, 0.08])
        y_tips = half_w * (1.0 - tip_fracs)
        y_all = np.unique(np.concatenate(
            [y_core, y_tips, [half_w]]))
        mask = np.concatenate([[True], np.diff(y_all) > 1e-10])
        y_positions = y_all[mask]

        # Build a lookup from metadata stations (if any)
        meta_by_y = {}
        for st in stations:
            y_key = round(st.get('y', -999) * 1e6)  # µm key
            meta_by_y[y_key] = st

        with open(csv_path, 'w') as f:
            f.write(f"# Blunting metadata for CAD filleting\n")
            f.write(f"# Method: {method},"
                    f" R_base: {R_base*scale:.4f} mm\n")
            f.write(f"# Scale: millimeters (same as STEP)\n")
            f.write("y_mm,x_le_mm,z_le_mm,chord_mm,"
                    "wedge_angle_deg,R_eff_mm\n")

            for y_j in y_positions:
                x_le_j = float(wr._leading_edge_x(y_j))
                z_le_j = float(wr._leading_edge_z(x_le_j))
                T_star_j = float(
                    wr._angle_perturbation(np.array([abs(y_j)]))[0])
                theta_j = T_star_j * theta_base
                chord = L - x_le_j

                # Compute R_eff: hard clamp min(R, 0.5 * TE thickness)
                R = wr.R
                R_eff = 0.0
                if (R > 0 and theta_j >= np.radians(0.5)
                        and chord > 1e-8):
                    te_thickness = chord * np.tan(theta_j)
                    R_max = (0.5 * te_thickness
                             if te_thickness > 1e-9 else 0.0)
                    R_eff = min(R, R_max)

                f.write(
                    f"{y_j*scale:.4f},"
                    f"{x_le_j*scale:.4f},"
                    f"{z_le_j*scale:.4f},"
                    f"{chord*scale:.4f},"
                    f"{np.degrees(theta_j):.4f},"
                    f"{R_eff*scale:.4f}\n")

        return csv_path


class PlanarWaveriderWorker(QThread):
    """Generates waverider geometry and runs aero analysis in a thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object, dict)  # (waverider, aero_results)
    error = pyqtSignal(str)

    def __init__(self, params, aero_params, parent=None):
        super().__init__(parent)
        self.params = params          # dict for PlanarWaverider constructor
        self.aero_params = aero_params  # dict with M_inf, alpha_deg, altitude_km, etc.

    def run(self):
        try:
            self.progress.emit("Generating geometry...")
            wr = PlanarWaverider(**self.params)
            nx = self.aero_params.get('nx', 60)
            ny = self.aero_params.get('ny', 40)
            wr.generate(nx=nx, ny=ny)

            self.progress.emit("Computing aerodynamic forces...")
            aero = PlanarWaveriderAero(gamma=self.params.get('gamma', 1.4))
            results = aero.compute_forces(
                wr,
                M_inf=self.aero_params['M_inf'],
                alpha_deg=self.aero_params['alpha_deg'],
                altitude_km=self.aero_params['altitude_km'],
                T_wall=self.aero_params.get('T_wall', None),
            )

            self.finished.emit(wr, results)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class PlanarOptimizationWorker(QThread):
    """Runs planar waverider optimization in a background thread."""

    progress_update = pyqtSignal(int, int, float, float)  # iter, max, best_LD, best_eta
    optimization_complete = pyqtSignal(dict)  # best result dict
    error_occurred = pyqtSignal(str)

    def __init__(self, fixed_params, bounds, opt_settings, parent=None):
        """
        Parameters
        ----------
        fixed_params : dict
            length, R, M_inf, alpha_deg, altitude_km, gamma, T_wall, nx, ny
        bounds : list of (lo, hi)
            7 tuples for [width, n, beta_deg, epsilon, p1, p2, p3]
        opt_settings : dict
            popsize, maxiter, tol, eta_vol_min, seed
        """
        super().__init__(parent)
        self.fixed_params = fixed_params
        self.bounds = bounds
        self.opt_settings = opt_settings
        self._optimizer = None

    def run(self):
        try:
            fp = self.fixed_params
            evaluator = PlanarWaveriderEvaluator(
                length=fp['length'],
                R=fp['R'],
                M_inf=fp['M_inf'],
                alpha_deg=fp['alpha_deg'],
                altitude_km=fp['altitude_km'],
                gamma=fp.get('gamma', 1.4),
                T_wall=fp.get('T_wall'),
                nx=fp.get('nx', 60),
                ny=fp.get('ny', 40),
            )

            s = self.opt_settings
            self._optimizer = PlanarWaveriderOptimizer(
                evaluator=evaluator,
                bounds=self.bounds,
                eta_vol_min=s.get('eta_vol_min', 0.0),
                popsize=s.get('popsize', 15),
                maxiter=s.get('maxiter', 100),
                tol=s.get('tol', 1e-6),
                callback=self._on_generation,
                seed=s.get('seed'),
            )

            best_x, best_metrics, history = self._optimizer.optimize_ld()

            if best_x is not None and best_metrics is not None:
                result = dict(best_metrics)
                result['best_x'] = best_x.tolist()
                result['n_evals'] = evaluator.cache_stats()[0]
                result['n_cache_hits'] = evaluator.cache_stats()[1]
                result['history'] = history
                self.optimization_complete.emit(result)
            else:
                self.error_occurred.emit(
                    "Optimization found no feasible design.")

        except Exception as e:
            self.error_occurred.emit(f"{e}\n{traceback.format_exc()}")

    def _on_generation(self, iteration, maxiter, best_x, best_LD,
                       best_metrics, history):
        """Callback from optimizer after each DE generation."""
        eta = best_metrics.get('eta_vol', 0.0) if best_metrics else 0.0
        self.progress_update.emit(iteration, maxiter, best_LD, eta)

    def stop(self):
        """Signal optimizer to stop after current generation."""
        if self._optimizer is not None:
            self._optimizer.stop()


# ──────────────────────────────────────────────────────────────────────
#  3D Canvas
# ──────────────────────────────────────────────────────────────────────

class PlanarWaveriderCanvas(FigureCanvas):
    """Matplotlib 3D canvas for planar waverider visualization."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8), facecolor='#0A0A0A')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1A1A1A')
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_waverider(self, wr, show_upper=True, show_lower=True,
                       show_le=True, show_info=True, aero_results=None):
        """Plot the planar waverider surfaces."""
        self.ax.clear()
        self.ax.set_facecolor('#1A1A1A')

        if wr is None or wr.upper_surface_x is None:
            self.ax.text(0, 0, 0, "No geometry", color='white',
                         fontsize=14, ha='center')
            self.draw()
            return

        # Plot upper surface
        if show_upper:
            surf_u = self.ax.plot_surface(
                wr.upper_surface_x, wr.upper_surface_y, wr.upper_surface_z,
                color='#4488CC', alpha=0.5, edgecolor='#335577',
                linewidth=0.2, shade=True,
            )
            surf_u._facecolors2d = surf_u._facecolor3d
            surf_u._edgecolors2d = surf_u._edgecolor3d

        # Plot lower surface
        if show_lower:
            surf_l = self.ax.plot_surface(
                wr.lower_surface_x, wr.lower_surface_y, wr.lower_surface_z,
                color='#CC6644', alpha=0.6, edgecolor='#884422',
                linewidth=0.2, shade=True,
            )
            surf_l._facecolors2d = surf_l._facecolor3d
            surf_l._edgecolors2d = surf_l._edgecolor3d

        # Plot leading edge
        if show_le and wr.leading_edge is not None:
            le = wr.leading_edge
            self.ax.plot(le[:, 0], le[:, 1], le[:, 2],
                         color='#FFAA00', linewidth=2.5, label='Leading Edge')

        # Axis labels and styling
        self.ax.set_xlabel('X (streamwise)', color='#888888', fontsize=8)
        self.ax.set_ylabel('Y (spanwise)', color='#888888', fontsize=8)
        self.ax.set_zlabel('Z (vertical)', color='#888888', fontsize=8)
        self.ax.tick_params(colors='#666666', labelsize=7)

        # Legend (proxy artists for surfaces)
        import matplotlib.patches as mpatches
        handles = []
        if show_upper:
            handles.append(mpatches.Patch(color='#4488CC', label='Upper Surface'))
        if show_lower:
            handles.append(mpatches.Patch(color='#CC6644', label='Lower Surface'))
        if show_le:
            handles.append(self.ax.plot([], [], color='#FFAA00',
                                        linewidth=2, label='Leading Edge')[0])
        if handles:
            self.ax.legend(handles=handles, loc='upper right', fontsize=7,
                           facecolor='#2A2A2A', edgecolor='#555555',
                           labelcolor='#CCCCCC')

        # Equal aspect ratio
        self._set_axes_equal()

        # Info panel overlay
        if show_info and aero_results:
            self._draw_info(wr, aero_results)

        self.draw()

    def _set_axes_equal(self):
        """Make axes of 3D plot have equal scale."""
        ax = self.ax
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        centers = limits.mean(axis=1)
        max_range = (limits[:, 1] - limits[:, 0]).max() / 2.0
        ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
        ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
        ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])

    def _draw_info(self, wr, res):
        """Draw text info overlay on the figure (matches cone-derived style)."""
        # Remove old text annotations
        for txt in self.fig.texts:
            txt.remove()

        bw, bh = wr.base_dimensions()
        vol = wr.volume()
        S_ref = res.get('S_ref', 0)
        # Volumetric efficiency: V / S_ref^(3/2)
        vol_eff = vol / S_ref**1.5 if S_ref > 1e-8 else 0.0

        info = (
            "WAVERIDER INFO\n"
            f"  Method          Planar (Jessen 2026)\n"
            f"  Mach            {res.get('M_inf', 0):.2f}\n"
            f"  Shock \u03b2         {wr.beta_deg:.2f}\u00b0\n"
            f"  Wedge \u03b8         {wr.wedge_angle_deg:.4f}\u00b0\n"
            f"  Power-law n     {wr.n:.2f}\n"
            f"  Epsilon         {wr.epsilon:.3f}\n"
            f"  p1, p2, p3      {wr.p1:.2f}, {wr.p2:.2f}, {wr.p3:.2f}\n"
            f"  LE Radius       {wr.R:.4f} m\n"
            f"  Length           {wr.length:.4f} m\n"
            f"  Width            {wr.width:.4f} m\n"
            f"  Planform Area    {S_ref:.4f} m\u00b2\n"
            f"  Volume           {vol:.6f} m\u00b3\n"
            f"  Vol Efficiency   {vol_eff:.6f}\n"
            f"  Base             {bw:.4f} x {bh:.4f} m\n"
            f"  L/D              {res.get('L_over_D', 0):.4f}\n"
            f"  CL               {res.get('CL', 0):.6f}\n"
            f"  CD               {res.get('CD', 0):.6f}"
        )

        self.fig.text(
            0.02, 0.97, info, transform=self.fig.transFigure,
            fontsize=8, color='#CCCCCC', family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1A1A1A',
                      edgecolor='#FF8800', alpha=0.85),
        )


# ──────────────────────────────────────────────────────────────────────
#  Chebyshev preview canvas (small 2D plot for perturbation curve)
# ──────────────────────────────────────────────────────────────────────

class ChebyshevPreviewCanvas(FigureCanvas):
    """Small 2D plot showing the Chebyshev perturbation T*(y)."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(3.5, 1.8), facecolor='#1A1A1A')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#222222')
        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.20)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMaximumHeight(140)

    def update_plot(self, p1, p2, p3, width):
        """Recompute and plot the Chebyshev perturbation curve."""
        self.ax.clear()
        self.ax.set_facecolor('#222222')

        try:
            tmp = PlanarWaverider(width=width, p1=p1, p2=p2, p3=p3)
            tmp._compute_chebyshev_coefficients()
            y = np.linspace(0, width / 2.0, 200)
            T_star = tmp._angle_perturbation(y)

            self.ax.plot(y, T_star, color='#66CCFF', linewidth=1.5)
            self.ax.axhline(1.0, color='#555555', linewidth=0.5, linestyle='--')

            # Mark control points
            y_p1 = width / 3.0
            y_p2 = width / 6.0
            y_p3 = 0.0
            self.ax.plot(y_p1, p1, 'o', color='#FF6644', markersize=5, label='p1')
            self.ax.plot(y_p2, p2, 's', color='#44FF66', markersize=5, label='p2')
            self.ax.plot(y_p3, p3, '^', color='#FFAA00', markersize=5, label='p3')

            self.ax.set_xlabel('y [m]', color='#888888', fontsize=7)
            self.ax.set_ylabel('T*(y)', color='#888888', fontsize=7)
            self.ax.set_title('Chebyshev Perturbation', color='#AAAAAA',
                              fontsize=8)
            self.ax.tick_params(colors='#666666', labelsize=6)
            self.ax.legend(fontsize=6, loc='upper right',
                           facecolor='#333333', edgecolor='#555555',
                           labelcolor='#CCCCCC')
        except Exception:
            self.ax.text(0.5, 0.5, 'Error', color='red', ha='center',
                         va='center', transform=self.ax.transAxes)

        self.draw()


# ──────────────────────────────────────────────────────────────────────
#  Main Tab Widget
# ──────────────────────────────────────────────────────────────────────

class PlanarWaveriderTab(QWidget):
    """GUI tab for designing planar waveriders (Jessen et al. 2026)."""

    waverider_generated = pyqtSignal(object)  # emits PlanarWaverider

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.waverider = None
        self.aero_results = None
        self.worker = None
        self.opt_worker = None
        self._best_design = None  # best optimizer result
        self.init_ui()

    # ── UI Construction ─────────────────────────────────────────────

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel (scrollable controls)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        left_layout.addWidget(self._create_flow_group())
        left_layout.addWidget(self._create_geometry_group())
        left_layout.addWidget(self._create_perturbation_group())
        left_layout.addWidget(self._create_blunting_group())
        left_layout.addWidget(self._create_mesh_group())
        left_layout.addWidget(self._create_viscous_group())
        left_layout.addWidget(self._create_generate_group())
        left_layout.addWidget(self._create_results_group())
        left_layout.addWidget(self._create_export_group())
        left_layout.addWidget(self._create_optimization_group())
        left_layout.addStretch()

        # Connect width to chebyshev preview (both widgets exist now)
        self.width_spin.valueChanged.connect(self._update_chebyshev_preview)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(420)

        # Right panel (3D view + results text)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 3D view options
        opts_layout = QHBoxLayout()
        self.show_upper_check = QCheckBox("Upper")
        self.show_upper_check.setChecked(True)
        self.show_lower_check = QCheckBox("Lower")
        self.show_lower_check.setChecked(True)
        self.show_le_check = QCheckBox("LE")
        self.show_le_check.setChecked(True)
        self.show_info_check = QCheckBox("Info")
        self.show_info_check.setChecked(True)
        for cb in [self.show_upper_check, self.show_lower_check,
                   self.show_le_check, self.show_info_check]:
            cb.stateChanged.connect(self._update_3d_view)
            opts_layout.addWidget(cb)
        opts_layout.addStretch()
        right_layout.addLayout(opts_layout)

        # 3D canvas + toolbar
        self.canvas_3d = PlanarWaveriderCanvas()
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, right_widget)
        right_layout.addWidget(self.toolbar_3d)
        right_layout.addWidget(self.canvas_3d, stretch=1)

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 9))
        self.results_text.setMaximumHeight(200)
        right_layout.addWidget(self.results_text)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

    # ── Control Groups ──────────────────────────────────────────────

    def _create_flow_group(self):
        group = QGroupBox("Flow Conditions")
        layout = QGridLayout()

        layout.addWidget(QLabel("Mach:"), 0, 0)
        self.mach_spin = QDoubleSpinBox()
        self.mach_spin.setRange(1.5, 25.0)
        self.mach_spin.setValue(6.85)
        self.mach_spin.setSingleStep(0.5)
        self.mach_spin.setDecimals(2)
        layout.addWidget(self.mach_spin, 0, 1)

        layout.addWidget(QLabel("Altitude [km]:"), 1, 0)
        self.alt_spin = QDoubleSpinBox()
        self.alt_spin.setRange(0, 80)
        self.alt_spin.setValue(25.0)
        self.alt_spin.setSingleStep(1.0)
        self.alt_spin.setDecimals(1)
        layout.addWidget(self.alt_spin, 1, 1)

        layout.addWidget(QLabel("Alpha [deg]:"), 2, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(-5.0, 15.0)
        self.alpha_spin.setValue(0.0)
        self.alpha_spin.setSingleStep(0.5)
        self.alpha_spin.setDecimals(2)
        layout.addWidget(self.alpha_spin, 2, 1)

        group.setLayout(layout)
        return group

    def _create_geometry_group(self):
        group = QGroupBox("Geometry Parameters")
        layout = QGridLayout()

        layout.addWidget(QLabel("Length [m]:"), 0, 0)
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.1, 50.0)
        self.length_spin.setValue(1.0)
        self.length_spin.setSingleStep(0.1)
        self.length_spin.setDecimals(3)
        layout.addWidget(self.length_spin, 0, 1)

        layout.addWidget(QLabel("Width [m]:"), 1, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.05, 50.0)
        self.width_spin.setValue(12.0)
        self.width_spin.setSingleStep(0.5)
        self.width_spin.setDecimals(3)
        layout.addWidget(self.width_spin, 1, 1)

        layout.addWidget(QLabel("Power-law n:"), 2, 0)
        self.n_spin = QDoubleSpinBox()
        self.n_spin.setRange(0.1, 20.0)
        self.n_spin.setValue(0.5)
        self.n_spin.setSingleStep(0.1)
        self.n_spin.setDecimals(2)
        self.n_spin.setToolTip("LE power-law exponent (0.5 = parabolic)")
        layout.addWidget(self.n_spin, 2, 1)

        layout.addWidget(QLabel("Shock angle [deg]:"), 3, 0)
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(5.0, 85.0)
        self.beta_spin.setValue(9.0)
        self.beta_spin.setSingleStep(0.5)
        self.beta_spin.setDecimals(2)
        self.beta_spin.setToolTip("Planar shock angle beta")
        layout.addWidget(self.beta_spin, 3, 1)

        layout.addWidget(QLabel("LE perturbation:"), 4, 0)
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(-1.0, 1.0)
        self.epsilon_spin.setValue(0.0)
        self.epsilon_spin.setSingleStep(0.05)
        self.epsilon_spin.setDecimals(3)
        self.epsilon_spin.setToolTip("Parabolic LE perturbation epsilon")
        layout.addWidget(self.epsilon_spin, 4, 1)

        group.setLayout(layout)
        return group

    def _create_perturbation_group(self):
        group = QGroupBox("Chebyshev Perturbations")
        layout = QVBoxLayout()

        grid = QGridLayout()
        grid.addWidget(QLabel("p1 (y=w/3):"), 0, 0)
        self.p1_spin = QDoubleSpinBox()
        self.p1_spin.setRange(0.1, 3.0)
        self.p1_spin.setValue(1.0)
        self.p1_spin.setSingleStep(0.05)
        self.p1_spin.setDecimals(3)
        self.p1_spin.setToolTip("Angle multiplier at y = w/3 (LE tip)")
        grid.addWidget(self.p1_spin, 0, 1)

        grid.addWidget(QLabel("p2 (y=w/6):"), 1, 0)
        self.p2_spin = QDoubleSpinBox()
        self.p2_spin.setRange(0.1, 3.0)
        self.p2_spin.setValue(1.0)
        self.p2_spin.setSingleStep(0.05)
        self.p2_spin.setDecimals(3)
        self.p2_spin.setToolTip("Angle multiplier at y = w/6")
        grid.addWidget(self.p2_spin, 1, 1)

        grid.addWidget(QLabel("p3 (y=0):"), 2, 0)
        self.p3_spin = QDoubleSpinBox()
        self.p3_spin.setRange(0.1, 3.0)
        self.p3_spin.setValue(1.0)
        self.p3_spin.setSingleStep(0.05)
        self.p3_spin.setDecimals(3)
        self.p3_spin.setToolTip("Angle multiplier at centerline y = 0")
        grid.addWidget(self.p3_spin, 2, 1)

        layout.addLayout(grid)

        # Chebyshev preview plot
        self.cheb_preview = ChebyshevPreviewCanvas()
        layout.addWidget(self.cheb_preview)

        # Connect spinboxes to live preview
        for spin in [self.p1_spin, self.p2_spin, self.p3_spin]:
            spin.valueChanged.connect(self._update_chebyshev_preview)

        group.setLayout(layout)
        return group

    def _create_blunting_group(self):
        group = QGroupBox("Leading Edge Blunting")
        layout = QGridLayout()

        # Method dropdown
        layout.addWidget(QLabel("Method:"), 0, 0)
        self.blunting_combo = QComboBox()
        self.blunting_combo.addItems([
            "None (Sharp)",
            "Inscribed Circle",
            u"B\u00e9zier G2 (Fu 2020)",
            "T&B Exterior",
        ])
        self.blunting_combo.setCurrentIndex(0)
        self.blunting_combo.setToolTip(
            "Blunting method for 3D preview.\n"
            "STEP always exports sharp geometry.")
        self.blunting_combo.currentIndexChanged.connect(
            self._on_blunting_method_changed)
        layout.addWidget(self.blunting_combo, 0, 1)

        # LE radius
        layout.addWidget(QLabel("LE Radius [m]:"), 1, 0)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.0, 0.5)
        self.radius_spin.setValue(0.0)
        self.radius_spin.setSingleStep(0.001)
        self.radius_spin.setDecimals(4)
        self.radius_spin.setToolTip("Leading edge nose radius (0 = sharp)")
        self.radius_spin.setEnabled(False)  # disabled when "None"
        layout.addWidget(self.radius_spin, 1, 1)

        # R/L percentage label
        self.r_over_l_label = QLabel("R/L = 0.00%")
        layout.addWidget(self.r_over_l_label, 2, 0, 1, 2)
        self.radius_spin.valueChanged.connect(self._update_r_over_l)
        self.length_spin.valueChanged.connect(self._update_r_over_l)

        # Note about STEP export
        note = QLabel("STEP exports sharp geometry.\n"
                       "Blunting CSV exported alongside for CAD filleting.")
        note.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(note, 3, 0, 1, 2)

        group.setLayout(layout)
        return group

    def _create_mesh_group(self):
        group = QGroupBox("Mesh Resolution")
        layout = QGridLayout()

        layout.addWidget(QLabel("Streamwise:"), 0, 0)
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(10, 300)
        self.nx_spin.setValue(60)
        self.nx_spin.setSingleStep(10)
        layout.addWidget(self.nx_spin, 0, 1)

        layout.addWidget(QLabel("Spanwise:"), 1, 0)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(10, 200)
        self.ny_spin.setValue(40)
        self.ny_spin.setSingleStep(10)
        layout.addWidget(self.ny_spin, 1, 1)

        group.setLayout(layout)
        return group

    def _create_viscous_group(self):
        group = QGroupBox("Viscous Model")
        layout = QGridLayout()

        layout.addWidget(QLabel("Wall temp:"), 0, 0)
        self.twall_combo = QComboBox()
        self.twall_combo.addItems(["Adiabatic", "Custom"])
        self.twall_combo.currentIndexChanged.connect(self._on_twall_mode_changed)
        layout.addWidget(self.twall_combo, 0, 1)

        layout.addWidget(QLabel("T_wall [K]:"), 1, 0)
        self.twall_spin = QDoubleSpinBox()
        self.twall_spin.setRange(200, 3000)
        self.twall_spin.setValue(300.0)
        self.twall_spin.setSingleStep(50)
        self.twall_spin.setDecimals(0)
        self.twall_spin.setEnabled(False)
        layout.addWidget(self.twall_spin, 1, 1)

        group.setLayout(layout)
        return group

    def _create_generate_group(self):
        group = QGroupBox("Generate")
        layout = QVBoxLayout()

        self.generate_btn = QPushButton("Generate Waverider")
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #2B5B2B; color: white; "
            "padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background-color: #3B7B3B; }"
        )
        self.generate_btn.clicked.connect(self.generate_waverider)
        layout.addWidget(self.generate_btn)

        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        # Preset buttons
        preset_layout = QHBoxLayout()
        btn_initial = QPushButton("Paper Initial")
        btn_initial.setToolTip("Initial guess from Jessen et al. Table 1")
        btn_initial.clicked.connect(self._load_preset_initial)
        preset_layout.addWidget(btn_initial)

        btn_opt_a = QPushButton("Paper Opt-A")
        btn_opt_a.setToolTip("Analytical optimized from Table 1")
        btn_opt_a.clicked.connect(self._load_preset_opt_analytical)
        preset_layout.addWidget(btn_opt_a)

        btn_opt_c = QPushButton("Paper Opt-C")
        btn_opt_c.setToolTip("CFD optimized from Table 1")
        btn_opt_c.clicked.connect(self._load_preset_opt_cfd)
        preset_layout.addWidget(btn_opt_c)

        layout.addLayout(preset_layout)

        group.setLayout(layout)
        return group

    def _create_results_group(self):
        group = QGroupBox("Results")
        layout = QGridLayout()

        self.result_labels = {}
        row = 0
        for key, label_text in [
            ('wedge_angle', 'Wedge angle:'),
            ('L_over_D', 'L/D:'),
            ('CL', 'CL:'),
            ('CD', 'CD:'),
            ('L', 'Lift [N]:'),
            ('D', 'Drag [N]:'),
            ('D_inv', 'D_inviscid [N]:'),
            ('D_visc', 'D_viscous [N]:'),
            ('D_base', 'D_base [N]:'),
            ('D_le', 'D_LE [N]:'),
            ('S_ref', 'S_ref [m2]:'),
            ('volume', 'Volume [m3]:'),
            ('base_w', 'Base width [m]:'),
            ('base_h', 'Base height [m]:'),
        ]:
            layout.addWidget(QLabel(label_text), row, 0)
            lbl = QLabel("—")
            lbl.setFont(QFont("Courier", 9))
            self.result_labels[key] = lbl
            layout.addWidget(lbl, row, 1)
            row += 1

        group.setLayout(layout)
        return group

    def _create_export_group(self):
        group = QGroupBox("Export")
        layout = QGridLayout()

        stl_btn = QPushButton("STL")
        stl_btn.clicked.connect(self.export_stl)
        layout.addWidget(stl_btn, 0, 0)

        step_btn = QPushButton("STEP")
        step_btn.clicked.connect(self.export_step)
        step_btn.setEnabled(CADQUERY_AVAILABLE)
        if not CADQUERY_AVAILABLE:
            step_btn.setToolTip("CadQuery not available")
        layout.addWidget(step_btn, 0, 1)

        send_btn = QPushButton("Send to Aero Tab")
        send_btn.setToolTip("Send mesh to main Aero Analysis tab")
        send_btn.clicked.connect(self._send_to_aero_tab)
        layout.addWidget(send_btn, 1, 0, 1, 2)

        group.setLayout(layout)
        return group

    # ── Optimization Group ────────────────────────────────────────

    def _create_optimization_group(self):
        """Build the optimization controls group box."""
        group = QGroupBox("Optimization (Jessen et al.)")
        layout = QVBoxLayout()

        # ── Bounds grid: 7 variables x (lo, hi) ──
        bounds_grid = QGridLayout()
        bounds_grid.setSpacing(3)

        self._opt_bound_spins = {}
        defaults = PlanarWaveriderOptimizer.DEFAULT_BOUNDS

        row = 0
        for name, label, lo_default, hi_default, dec, step in [
            ('width', 'Width [m]', 0.1, 60.0, 2, 1.0),
            ('n', 'n (power)', 0.1, 0.9, 2, 0.05),
            ('beta_deg', 'Beta [deg]', 1.0, 20.0, 2, 0.5),
            ('epsilon', 'Epsilon', -1.0, 1.0, 2, 0.1),
            ('p1', 'p1', 0.5, 3.0, 2, 0.1),
            ('p2', 'p2', 0.5, 3.0, 2, 0.1),
            ('p3', 'p3', 0.5, 3.0, 2, 0.1),
        ]:
            bounds_grid.addWidget(QLabel(label), row, 0)

            lo_spin = QDoubleSpinBox()
            lo_spin.setDecimals(dec)
            lo_spin.setRange(-100.0, 1000.0)
            lo_spin.setSingleStep(step)
            lo_spin.setValue(lo_default)
            bounds_grid.addWidget(lo_spin, row, 1)

            hi_spin = QDoubleSpinBox()
            hi_spin.setDecimals(dec)
            hi_spin.setRange(-100.0, 1000.0)
            hi_spin.setSingleStep(step)
            hi_spin.setValue(hi_default)
            bounds_grid.addWidget(hi_spin, row, 2)

            self._opt_bound_spins[name] = (lo_spin, hi_spin)
            row += 1

        # Column headers
        bounds_header = QHBoxLayout()
        bounds_header.addWidget(QLabel("Variable"))
        bounds_header.addWidget(QLabel("Lower"))
        bounds_header.addWidget(QLabel("Upper"))
        layout.addLayout(bounds_header)
        layout.addLayout(bounds_grid)

        # ── Settings row ──
        settings_grid = QGridLayout()
        settings_grid.setSpacing(3)

        settings_grid.addWidget(QLabel("Population:"), 0, 0)
        self.opt_popsize_spin = QSpinBox()
        self.opt_popsize_spin.setRange(5, 100)
        self.opt_popsize_spin.setValue(15)
        self.opt_popsize_spin.setToolTip("DE population size multiplier (total = pop x 7)")
        settings_grid.addWidget(self.opt_popsize_spin, 0, 1)

        settings_grid.addWidget(QLabel("Max iter:"), 0, 2)
        self.opt_maxiter_spin = QSpinBox()
        self.opt_maxiter_spin.setRange(5, 10000)
        self.opt_maxiter_spin.setValue(100)
        self.opt_maxiter_spin.setToolTip("Maximum DE generations")
        settings_grid.addWidget(self.opt_maxiter_spin, 0, 3)

        settings_grid.addWidget(QLabel("eta_vol min:"), 1, 0)
        self.opt_eta_spin = QDoubleSpinBox()
        self.opt_eta_spin.setDecimals(4)
        self.opt_eta_spin.setRange(0.0, 1.0)
        self.opt_eta_spin.setSingleStep(0.001)
        self.opt_eta_spin.setValue(0.0)
        self.opt_eta_spin.setToolTip("Minimum volumetric efficiency constraint (0 = off)")
        settings_grid.addWidget(self.opt_eta_spin, 1, 1)

        settings_grid.addWidget(QLabel("Seed:"), 1, 2)
        self.opt_seed_spin = QSpinBox()
        self.opt_seed_spin.setRange(0, 99999)
        self.opt_seed_spin.setValue(42)
        self.opt_seed_spin.setSpecialValueText("Random")
        self.opt_seed_spin.setToolTip("Random seed (0 = random)")
        settings_grid.addWidget(self.opt_seed_spin, 1, 3)

        layout.addLayout(settings_grid)

        # ── Buttons ──
        btn_layout = QHBoxLayout()

        self.opt_run_btn = QPushButton("Run Optimization")
        self.opt_run_btn.setStyleSheet(
            "QPushButton { background-color: #8B4513; color: white; "
            "padding: 6px; font-weight: bold; }"
            "QPushButton:hover { background-color: #A0522D; }"
        )
        self.opt_run_btn.clicked.connect(self._run_optimization)
        btn_layout.addWidget(self.opt_run_btn)

        self.opt_stop_btn = QPushButton("Stop")
        self.opt_stop_btn.setEnabled(False)
        self.opt_stop_btn.clicked.connect(self._stop_optimization)
        btn_layout.addWidget(self.opt_stop_btn)

        self.opt_load_btn = QPushButton("Load Best")
        self.opt_load_btn.setEnabled(False)
        self.opt_load_btn.setToolTip("Load best design into geometry spinboxes")
        self.opt_load_btn.clicked.connect(self._load_best_design)
        btn_layout.addWidget(self.opt_load_btn)

        layout.addLayout(btn_layout)

        # ── Progress ──
        self.opt_progress_bar = QProgressBar()
        self.opt_progress_bar.setRange(0, 100)
        self.opt_progress_bar.setValue(0)
        layout.addWidget(self.opt_progress_bar)

        self.opt_status_label = QLabel("")
        self.opt_status_label.setWordWrap(True)
        layout.addWidget(self.opt_status_label)

        group.setLayout(layout)
        return group

    # ── Optimization handlers ────────────────────────────────────

    def _run_optimization(self):
        """Collect parameters and launch the optimization worker."""
        if self.opt_worker is not None and self.opt_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Optimization already running.")
            return

        # Fixed parameters from current GUI values
        T_wall = None
        if self.twall_combo.currentIndex() == 1:
            T_wall = self.twall_spin.value()

        fixed_params = {
            'length': self.length_spin.value(),
            'R': self.radius_spin.value(),
            'M_inf': self.mach_spin.value(),
            'alpha_deg': self.alpha_spin.value(),
            'altitude_km': self.alt_spin.value(),
            'gamma': 1.4,
            'T_wall': T_wall,
            'nx': self.nx_spin.value(),
            'ny': self.ny_spin.value(),
        }

        # Bounds from spinboxes
        bounds = []
        for name in PlanarWaveriderEvaluator.VAR_NAMES:
            lo_spin, hi_spin = self._opt_bound_spins[name]
            lo = lo_spin.value()
            hi = hi_spin.value()
            if lo >= hi:
                QMessageBox.warning(
                    self, "Invalid Bounds",
                    f"Lower bound for '{name}' must be less than upper bound.")
                return
            bounds.append((lo, hi))

        # Optimizer settings
        seed_val = self.opt_seed_spin.value()
        opt_settings = {
            'popsize': self.opt_popsize_spin.value(),
            'maxiter': self.opt_maxiter_spin.value(),
            'tol': 1e-6,
            'eta_vol_min': self.opt_eta_spin.value(),
            'seed': seed_val if seed_val > 0 else None,
        }

        # Update UI state
        self.opt_run_btn.setEnabled(False)
        self.opt_stop_btn.setEnabled(True)
        self.opt_load_btn.setEnabled(False)
        self.opt_progress_bar.setValue(0)
        self.opt_progress_bar.setRange(0, opt_settings['maxiter'])
        self.opt_status_label.setText("Starting optimization...")

        # Launch worker
        self.opt_worker = PlanarOptimizationWorker(
            fixed_params, bounds, opt_settings)
        self.opt_worker.progress_update.connect(self._on_opt_progress)
        self.opt_worker.optimization_complete.connect(self._on_opt_complete)
        self.opt_worker.error_occurred.connect(self._on_opt_error)
        self.opt_worker.start()

    def _stop_optimization(self):
        """Signal the optimization to stop."""
        if self.opt_worker is not None:
            self.opt_worker.stop()
            self.opt_status_label.setText("Stopping after current generation...")

    def _on_opt_progress(self, iteration, maxiter, best_LD, best_eta):
        """Update progress bar and status label."""
        self.opt_progress_bar.setValue(iteration)
        self.opt_status_label.setText(
            f"Gen {iteration}/{maxiter}  |  "
            f"Best L/D: {best_LD:.4f}  |  "
            f"eta_vol: {best_eta:.5f}")

    def _on_opt_complete(self, result):
        """Handle completed optimization."""
        self.opt_run_btn.setEnabled(True)
        self.opt_stop_btn.setEnabled(False)
        self._best_design = result
        self.opt_load_btn.setEnabled(True)

        LD = result.get('L_over_D', 0)
        eta = result.get('eta_vol', 0)
        n_evals = result.get('n_evals', 0)
        n_hits = result.get('n_cache_hits', 0)
        best_x = result.get('best_x', [])

        var_names = PlanarWaveriderEvaluator.VAR_NAMES
        var_str = ', '.join(
            f"{var_names[i]}={best_x[i]:.4f}" for i in range(len(best_x)))

        self.opt_progress_bar.setValue(self.opt_progress_bar.maximum())
        self.opt_status_label.setText(
            f"Done! Best L/D: {LD:.4f} | eta_vol: {eta:.5f}\n"
            f"Evals: {n_evals} (cache hits: {n_hits})\n"
            f"{var_str}")

        # Save history CSV alongside working directory
        history = result.get('history', [])
        if history:
            import os
            csv_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'optimization_history.csv')
            PlanarWaveriderOptimizer.save_history_csv(history, csv_path)

    def _on_opt_error(self, msg):
        """Handle optimization error."""
        self.opt_run_btn.setEnabled(True)
        self.opt_stop_btn.setEnabled(False)
        self.opt_status_label.setText("Optimization failed!")
        QMessageBox.critical(self, "Optimization Error", msg)

    def _load_best_design(self):
        """Load the best optimized design into the geometry spinboxes."""
        if self._best_design is None:
            return
        best_x = self._best_design.get('best_x', [])
        if len(best_x) < 7:
            return

        # Map best_x -> spinboxes
        # x = [width, n, beta_deg, epsilon, p1, p2, p3]
        self.width_spin.setValue(best_x[0])
        self.n_spin.setValue(best_x[1])
        self.beta_spin.setValue(best_x[2])
        self.epsilon_spin.setValue(best_x[3])
        self.p1_spin.setValue(best_x[4])
        self.p2_spin.setValue(best_x[5])
        self.p3_spin.setValue(best_x[6])

        # Trigger geometry regeneration
        self.generate_waverider()

    # ── Slot Handlers ───────────────────────────────────────────────

    def _on_twall_mode_changed(self, idx):
        self.twall_spin.setEnabled(idx == 1)

    def _update_r_over_l(self):
        R = self.radius_spin.value()
        L = self.length_spin.value()
        pct = R / L * 100 if L > 0 else 0.0
        self.r_over_l_label.setText(f"R/L = {pct:.2f}%")

    def _on_blunting_method_changed(self, idx):
        """Enable/disable R spinbox based on blunting method selection."""
        is_blunted = (idx != 0)  # 0 = "None (Sharp)"
        self.radius_spin.setEnabled(is_blunted)
        if not is_blunted:
            self.radius_spin.setValue(0.0)

    def _update_chebyshev_preview(self):
        self.cheb_preview.update_plot(
            self.p1_spin.value(), self.p2_spin.value(),
            self.p3_spin.value(), self.width_spin.value(),
        )

    def _update_3d_view(self):
        """Refresh 3D canvas with current checkbox states."""
        self.canvas_3d.plot_waverider(
            self.waverider,
            show_upper=self.show_upper_check.isChecked(),
            show_lower=self.show_lower_check.isChecked(),
            show_le=self.show_le_check.isChecked(),
            show_info=self.show_info_check.isChecked(),
            aero_results=self.aero_results,
        )

    # ── Generation ──────────────────────────────────────────────────

    def generate_waverider(self):
        """Launch background worker to generate geometry + aero."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Generation already running.")
            return

        # Map blunting combo index to method string
        _method_map = {
            0: 'none', 1: 'inscribed', 2: 'bezier_g2', 3: 'tb_exterior'
        }
        blunting_method = _method_map.get(
            self.blunting_combo.currentIndex(), 'none')

        params = {
            'length': self.length_spin.value(),
            'width': self.width_spin.value(),
            'n': self.n_spin.value(),
            'beta_deg': self.beta_spin.value(),
            'epsilon': self.epsilon_spin.value(),
            'p1': self.p1_spin.value(),
            'p2': self.p2_spin.value(),
            'p3': self.p3_spin.value(),
            'R': self.radius_spin.value(),
            'blunting_method': blunting_method,
            'M_inf': self.mach_spin.value(),
            'gamma': 1.4,
        }

        T_wall = None
        if self.twall_combo.currentIndex() == 1:
            T_wall = self.twall_spin.value()

        aero_params = {
            'M_inf': self.mach_spin.value(),
            'alpha_deg': self.alpha_spin.value(),
            'altitude_km': self.alt_spin.value(),
            'T_wall': T_wall,
            'nx': self.nx_spin.value(),
            'ny': self.ny_spin.value(),
        }

        self.generate_btn.setEnabled(False)
        self.progress_label.setText("Generating...")

        self.worker = PlanarWaveriderWorker(params, aero_params)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg):
        self.progress_label.setText(msg)

    def _on_finished(self, wr, results):
        self.waverider = wr
        self.aero_results = results
        self.generate_btn.setEnabled(True)
        self.progress_label.setText("Done")

        # Update 3D view
        self._update_3d_view()

        # Update result labels
        self._update_results(wr, results)

        # Update results text area
        self._update_results_text(wr, results)

        # Emit signal for inter-tab communication
        self.waverider_generated.emit(wr)

        # Update Chebyshev preview
        self._update_chebyshev_preview()

    def _on_error(self, msg):
        self.generate_btn.setEnabled(True)
        self.progress_label.setText("Error!")
        QMessageBox.critical(self, "Error", msg)

    def _update_results(self, wr, res):
        """Fill the results labels."""
        fmt = {
            'wedge_angle': f"{wr.wedge_angle_deg:.4f} deg",
            'L_over_D': f"{res['L_over_D']:.4f}",
            'CL': f"{res['CL']:.6f}",
            'CD': f"{res['CD']:.6f}",
            'L': f"{res['L']:.1f}",
            'D': f"{res['D']:.1f}",
            'D_inv': f"{res['D_inviscid']:.1f}",
            'D_visc': f"{res['D_viscous']:.1f}",
            'D_base': f"{res['D_base']:.1f}",
            'D_le': f"{res.get('D_le', 0):.1f}",
            'S_ref': f"{res['S_ref']:.4f}",
            'volume': f"{wr.volume():.6f}",
        }
        bw, bh = wr.base_dimensions()
        fmt['base_w'] = f"{bw:.4f}"
        fmt['base_h'] = f"{bh:.4f}"

        for key, val in fmt.items():
            if key in self.result_labels:
                self.result_labels[key].setText(val)

    def _update_results_text(self, wr, res):
        """Write detailed results to the text area."""
        T_inf, P_inf, rho_inf, a_inf = atmosphere(res['altitude_km'])
        V_inf = res['M_inf'] * a_inf

        lines = [
            "=" * 55,
            "  PLANAR WAVERIDER (Jessen et al. 2026)",
            "=" * 55,
            "",
            "  Flow Conditions",
            f"    Mach         = {res['M_inf']:.2f}",
            f"    Altitude     = {res['altitude_km']:.1f} km",
            f"    Alpha        = {res['alpha_deg']:.2f} deg",
            f"    T_inf        = {T_inf:.2f} K",
            f"    P_inf        = {P_inf:.1f} Pa",
            f"    V_inf        = {V_inf:.1f} m/s",
            f"    q_inf        = {res['q_inf']:.1f} Pa",
            "",
            "  Geometry",
            f"    Length       = {wr.length:.3f} m",
            f"    Width        = {wr.width:.3f} m",
            f"    n (power)    = {wr.n:.3f}",
            f"    beta (shock) = {wr.beta_deg:.2f} deg",
            f"    theta (wedge)= {wr.wedge_angle_deg:.4f} deg",
            f"    epsilon      = {wr.epsilon:.3f}",
            f"    p1, p2, p3   = {wr.p1:.3f}, {wr.p2:.3f}, {wr.p3:.3f}",
            f"    R (LE radius)= {wr.R:.4f} m",
            f"    S_ref        = {res['S_ref']:.4f} m2",
            f"    Volume       = {wr.volume():.6f} m3",
            "",
            "  Aerodynamic Performance",
            f"    L/D          = {res['L_over_D']:.4f}",
            f"    CL           = {res['CL']:.6f}",
            f"    CD           = {res['CD']:.6f}",
            f"    Lift         = {res['L']:.1f} N",
            f"    Drag (total) = {res['D']:.1f} N",
            f"      Inviscid   = {res['D_inviscid']:.1f} N",
            f"      Viscous    = {res['D_viscous']:.1f} N",
            f"      Base       = {res['D_base']:.1f} N",
            f"      LE         = {res.get('D_le', 0):.1f} N",
            "",
            "=" * 55,
        ]
        self.results_text.setText('\n'.join(lines))

    # ── Presets (Table 1 from the paper) ────────────────────────────

    def _load_preset_initial(self):
        """Initial guess: Table 1 row 1 (actual paper values)."""
        self.mach_spin.setValue(6.85)
        self.alt_spin.setValue(25.0)
        self.alpha_spin.setValue(0.0)
        self.length_spin.setValue(40.0)
        self.width_spin.setValue(12.0)
        self.n_spin.setValue(0.5)
        self.beta_spin.setValue(9.0)
        self.epsilon_spin.setValue(0.0)
        self.p1_spin.setValue(1.0)
        self.p2_spin.setValue(1.0)
        self.p3_spin.setValue(1.0)
        self.radius_spin.setValue(0.1)  # R/L = 0.25%

    def _load_preset_opt_analytical(self):
        """Analytical optimized: Table 1 row 2 (actual paper values)."""
        self.mach_spin.setValue(6.85)
        self.alt_spin.setValue(25.0)
        self.alpha_spin.setValue(0.0)
        self.length_spin.setValue(40.0)
        self.width_spin.setValue(12.81)
        self.n_spin.setValue(0.90)
        self.beta_spin.setValue(5.42)
        self.epsilon_spin.setValue(-0.35)
        self.p1_spin.setValue(1.47)
        self.p2_spin.setValue(1.54)
        self.p3_spin.setValue(1.57)
        self.radius_spin.setValue(0.1)

    def _load_preset_opt_cfd(self):
        """CFD optimized: Table 1 row 3 (actual paper values)."""
        self.mach_spin.setValue(6.85)
        self.alt_spin.setValue(25.0)
        self.alpha_spin.setValue(0.0)
        self.length_spin.setValue(40.0)
        self.width_spin.setValue(19.06)
        self.n_spin.setValue(0.90)
        self.beta_spin.setValue(9.00)
        self.epsilon_spin.setValue(-0.56)
        self.p1_spin.setValue(0.98)
        self.p2_spin.setValue(1.02)
        self.p3_spin.setValue(0.97)
        self.radius_spin.setValue(0.1)

    # ── Parameter serialisation (JSON save / load) ─────────────────

    def get_params_dict(self):
        """Return all design parameters as a JSON-serialisable dict."""
        return {
            'mach': self.mach_spin.value(),
            'altitude': self.alt_spin.value(),
            'alpha': self.alpha_spin.value(),
            'length': self.length_spin.value(),
            'width': self.width_spin.value(),
            'n': self.n_spin.value(),
            'beta': self.beta_spin.value(),
            'epsilon': self.epsilon_spin.value(),
            'p1': self.p1_spin.value(),
            'p2': self.p2_spin.value(),
            'p3': self.p3_spin.value(),
            'le_radius': self.radius_spin.value(),
            'nx': self.nx_spin.value(),
            'ny': self.ny_spin.value(),
            'twall_mode': self.twall_combo.currentText(),
            'twall': self.twall_spin.value(),
        }

    def set_params_dict(self, d):
        """Restore parameters from a dict (e.g. loaded from JSON)."""
        from PyQt5.QtWidgets import QDoubleSpinBox, QSpinBox, QComboBox

        def _s(widget, value):
            if value is None:
                return
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                widget.setCurrentIndex(idx if idx >= 0 else 0)

        _s(self.mach_spin, d.get('mach'))
        _s(self.alt_spin, d.get('altitude'))
        _s(self.alpha_spin, d.get('alpha'))
        _s(self.length_spin, d.get('length'))
        _s(self.width_spin, d.get('width'))
        _s(self.n_spin, d.get('n'))
        _s(self.beta_spin, d.get('beta'))
        _s(self.epsilon_spin, d.get('epsilon'))
        _s(self.p1_spin, d.get('p1'))
        _s(self.p2_spin, d.get('p2'))
        _s(self.p3_spin, d.get('p3'))
        _s(self.radius_spin, d.get('le_radius'))
        _s(self.nx_spin, d.get('nx'))
        _s(self.ny_spin, d.get('ny'))
        _s(self.twall_combo, d.get('twall_mode'))
        _s(self.twall_spin, d.get('twall'))

    # ── Export ──────────────────────────────────────────────────────

    def export_stl(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate waverider first!")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save STL", "planar_waverider.stl", "STL (*.stl)")
        if not fn:
            return
        try:
            verts, faces = self.waverider.get_mesh()
            self._write_stl(fn, verts, faces)
            QMessageBox.information(self, "Success", f"Saved: {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _write_stl(self, filename, vertices, faces):
        """Write binary STL file."""
        import struct
        with open(filename, 'wb') as f:
            f.write(b'\0' * 80)  # header
            f.write(struct.pack('<I', len(faces)))
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal /= norm
                f.write(struct.pack('<3f', *normal))
                f.write(struct.pack('<3f', *v0))
                f.write(struct.pack('<3f', *v1))
                f.write(struct.pack('<3f', *v2))
                f.write(struct.pack('<H', 0))

    def export_step(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate waverider first!")
            return
        if not CADQUERY_AVAILABLE:
            QMessageBox.warning(self, "Warning", "CadQuery not available!")
            return
        if hasattr(self, '_step_worker') and self._step_worker is not None \
                and self._step_worker.isRunning():
            QMessageBox.warning(self, "Busy", "STEP export already running.")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save STEP", "planar_waverider.step", "STEP (*.step)")
        if not fn:
            return

        self.progress_label.setText("Exporting STEP...")
        self._step_worker = StepExportWorker(self.waverider, fn)
        self._step_worker.progress.connect(self._on_progress)
        self._step_worker.finished.connect(self._on_step_done)
        self._step_worker.error.connect(self._on_step_error)
        self._step_worker.start()

    def _on_step_done(self, filename):
        self.progress_label.setText("STEP export done")
        QMessageBox.information(self, "Success", f"Saved: {filename}")

    def _on_step_error(self, msg):
        self.progress_label.setText("STEP export failed")
        QMessageBox.critical(self, "Error", msg)

    def _send_to_aero_tab(self):
        """Send mesh data to the main aero analysis tab."""
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate waverider first!")
            return
        if self.parent_gui and hasattr(self.parent_gui, 'imported_geometry'):
            verts, faces = self.waverider.get_mesh()
            self.parent_gui.imported_geometry = {
                'vertices': verts,
                'faces': faces,
                'source': 'planar_waverider',
                'params': self.waverider.to_dict(),
            }
            QMessageBox.information(
                self, "Sent",
                "Planar waverider mesh sent to Aero Analysis tab.")
        else:
            QMessageBox.warning(
                self, "Warning",
                "Parent GUI or imported_geometry not available.")
