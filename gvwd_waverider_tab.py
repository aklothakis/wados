"""GVWD GUI tab — Glide-Vehicle Wedge-Derived geometry hub (Phase 7).

Plugs into the existing PyQt5 hypersonic-waverider hub
(``waverider_gui.py``) as a sibling tab to the PSWR-1, OC, CD, and VMPLO
tabs. Five geometry modes selectable by combo-box, defaulting to the
engineering flat-bottom HTV-2 / Fattah-2 / Avangard archetype.

UX skeleton mirrors :mod:`pswr_waverider_tab`: left scroll panel with
parameter QGroupBoxes + action buttons, right side 3D matplotlib canvas
with a bottom tab-strip of analysis plots. Long-running operations
(Mach-alpha sweep) run in a QThread.
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QGridLayout,
                             QDoubleSpinBox, QSpinBox, QCheckBox,
                             QMessageBox, QSplitter, QApplication,
                             QScrollArea, QTabWidget, QStackedWidget,
                             QComboBox, QProgressBar, QFileDialog,
                             QDialog, QTextEdit, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# GVWD library
from gvwd.io.config import (
    GVWDRunConfig, FinsConfig, SweepRunConfig,
    EngineeringFlatConfig, EngineeringShallowVConfig,
    CaretConfig, FlatDeltaConfig, MultiWedgeConfig,
    config_sha256, build_geometry,
)
from gvwd.geometry import (
    Mesh, FinParams, generate_fins, merge_meshes,
    numerical_volume, planform_area_from_mesh, eta_V,
)
from gvwd.aero.coefficients import aero_coefficients_full
from gvwd.aero.sweep import SweepConfig, mach_alpha_sweep
from gvwd.aero.panel_method import freestream_direction
from gvwd.export.stl import write_stl
from gvwd.export.step import write_step, CadqueryUnavailableError
from gvwd.thermo.oblique_shock import (
    mach_angle, theta_max as _theta_max, ShockDetachedError,
)


_MODE_LABELS = [
    ("engineering_flat",      "Engineering flat-bottom (HTV-2 archetype)"),
    ("engineering_shallow_v", "Engineering shallow-V"),
    ("caret",                 "Reference: Nonweiler caret"),
    ("flat_delta",            "Reference: Flat-bottomed delta"),
    ("multi_wedge",           "Reference: Multi-wedge / Oswatitsch"),
]


# ======================================================================
#  Per-fin builder (independent position + rotation per fin)
# ======================================================================

def _rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _generate_single_fin(
    *,
    root_chord: float,
    tip_chord: float,
    span: float,
    sweep_LE_rad: float,
    t_c: float,
    max_thickness_loc: float,
    LE_style: str,
    LE_radius_m: float,
    position: tuple,                 # (x, y, z) in body frame
    rotation_deg: tuple,             # (roll_x, pitch_y, yaw_z) extrinsic XYZ
    fin_idx: int = 1,
) -> Optional[Mesh]:
    """Build a single fin mesh placed at ``position`` with extrinsic
    XYZ rotation ``rotation_deg`` applied in the body frame.

    The local fin frame is identical to ``gvwd.geometry.fins.FinParams``:
    chord along +x_local, span along +y_local, thickness along +-z_local.
    The local mesh is rotated as ``R = R_z(yaw) · R_y(pitch) · R_x(roll)``
    and then translated to ``position``.
    """
    from gvwd.geometry.fins import FinParams, _fin_local_mesh

    # Validate via FinParams (n_fins=2 chosen so post-init passes).
    params = FinParams(
        n_fins=2,
        root_chord=root_chord,
        tip_chord=tip_chord,
        span=span,
        sweep_LE=sweep_LE_rad,
        dihedral=0.0,
        t_c=t_c,
        max_thickness_loc=max_thickness_loc,
        LE_style=LE_style,
        LE_radius=LE_radius_m,
    )
    v_loc, f_loc, l_loc = _fin_local_mesh(params)

    rx, ry, rz = (math.radians(a) for a in rotation_deg)
    R = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)
    v = (R @ v_loc.T).T + np.asarray(position, dtype=float)

    # Tag labels with fin index so visual classification stays per-fin.
    labels = np.array(
        [f"{lbl}_fin{fin_idx}" for lbl in l_loc],
        dtype=object,
    )
    return Mesh(
        vertices=v,
        faces=f_loc.copy(),
        labels=labels,
        metadata={
            "kind": "fin",
            "fin_idx": int(fin_idx),
            "root_chord": float(root_chord),
            "tip_chord": float(tip_chord),
            "span": float(span),
            "sweep_LE_deg": math.degrees(sweep_LE_rad),
            "t_c": float(t_c),
            "max_thickness_loc": float(max_thickness_loc),
            "LE_style": LE_style,
            "LE_radius_m": float(LE_radius_m),
            "position_xyz": tuple(float(x) for x in position),
            "rotation_xyz_deg": tuple(float(a) for a in rotation_deg),
        },
    )


# ======================================================================
#  Canvas classes
# ======================================================================

def _classify_label(label: str) -> str:
    """Bin a face-label string into one of {'lower','upper','base',
    'fin','LE','other'}.

    Engineering modes use compound labels like ``forebody_lower``,
    ``centerbody_side_left``, ``forebody_upper``; reference modes
    use plain ``lower_left``, ``upper_right``, ``base_left``, etc.
    Match wherever the role appears in the string."""
    s = str(label).lower()
    if "leading_edge" in s or "le_strip" in s:
        return "LE"
    if "fin" in s:
        return "fin"
    if "lower" in s or "ramp" in s:
        return "lower"
    if "upper" in s:
        return "upper"
    if "side" in s:
        return "base"          # render side panels in the base color
    if "base" in s:
        return "base"
    return "other"


class _Mesh3DCanvas(FigureCanvas):
    """3-D rendering styled to match the PSWR-1 tab.

    Lower surface (windward) is shaded indianred, upper surface
    steelblue, base panels in a muted gray, fins in burnt-orange,
    and the leading-edge strip is rendered as a thick black line.
    A monospaced info panel with all vehicle characteristics is
    drawn in the top-left of the figure (orange-bordered dark box).
    """

    SURFACE_COLORS = {
        "lower": "#4682b4",   # steelblue
        "upper": "#cd5c5c",   # indianred
        "base":  "#888888",   # gray
        "fin":   "#cb6e1b",   # burnt orange
        "LE":    "#000000",   # black (rendered as line)
        "other": "#7e8a93",   # slate
    }

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 7))
        # Transparent figure so the dark Qt background shows through
        self.fig.patch.set_alpha(0.0)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("none")
        super().__init__(self.fig)
        self.setParent(parent)
        self._info_text = None
        self._draw_default()

    def _draw_default(self):
        self.ax.clear()
        self.ax.set_title("Click 'Generate Geometry'", color="white")
        self.ax.tick_params(colors="#888888")
        self.fig.tight_layout()
        self.draw()

    # ------------------------------------------------------------------
    #  Public entry-point
    # ------------------------------------------------------------------

    def plot_mesh(self, mesh: Optional[Mesh],
                   *,
                   info: Optional[dict] = None,
                   title_prefix: str = "GVWD",
                   title_extra: str = "") -> None:
        """Render ``mesh`` and (optionally) overlay an info panel.

        Style mirrors :class:`pswr_waverider_tab.PSWRCanvas3D`:
        translucent filled polygons (so each surface reads as a
        coloured patch even on coarse meshes) **plus** bright edge
        lines drawn on top — lower=indianred, upper=steelblue,
        base=gray, fins=burnt-orange. The leading edge is a single
        thick black polyline extracted from the lower/upper boundary
        chain.
        """
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.ax.clear()
        if self._info_text is not None:
            try:
                self._info_text.remove()
            except Exception:
                pass
            self._info_text = None
        if mesh is None:
            self._draw_default(); return

        v = mesh.vertices

        # ----- Group faces by category --------------------------------
        cat_faces = {k: [] for k in self.SURFACE_COLORS}
        for face, lbl in zip(mesh.faces, mesh.labels):
            cat = _classify_label(lbl)
            cat_faces[cat].append(face)

        # Plot-axis frame mapping. GVWD body frame is
        # (x stream, y span, z vertical). The plot puts:
        #   plot.x = mesh.y (span)         — left/right axis
        #   plot.y = mesh.x (streamwise)   — depth-into-screen
        #   plot.z = mesh.z (vertical)     — up axis

        def _tri_xyz(face_list):
            return [
                [(v[i][1], v[i][0], v[i][2]) for i in tri]
                for tri in face_list
            ]

        def _draw_surface(face_list, color, *, fill_alpha, edge_alpha,
                          edge_lw):
            """Render a surface category as a translucent filled
            polygon set with a brighter edge wireframe drawn on
            top. Combining both gives the PSWR-1-style appearance
            on the much coarser GVWD meshes."""
            if not face_list:
                return
            polys = Poly3DCollection(
                _tri_xyz(face_list),
                facecolor=color, edgecolor="none",
                alpha=fill_alpha,
            )
            self.ax.add_collection3d(polys)
            for face in face_list:
                tri = v[face]
                xs = [tri[0, 1], tri[1, 1], tri[2, 1], tri[0, 1]]
                ys = [tri[0, 0], tri[1, 0], tri[2, 0], tri[0, 0]]
                zs = [tri[0, 2], tri[1, 2], tri[2, 2], tri[0, 2]]
                self.ax.plot(xs, ys, zs, color=color,
                             alpha=edge_alpha, linewidth=edge_lw)

        legend_elements = []

        # Lower surface — windward (steelblue), labeled "Upper" in legend
        if cat_faces["lower"]:
            _draw_surface(cat_faces["lower"],
                          self.SURFACE_COLORS["lower"],
                          fill_alpha=0.55, edge_alpha=0.95, edge_lw=1.4)
            legend_elements.append(
                Patch(facecolor=self.SURFACE_COLORS["lower"],
                       alpha=0.55, label="Upper Surface"))
        # Upper surface — leeward (indianred), labeled "Lower" in legend
        if cat_faces["upper"]:
            _draw_surface(cat_faces["upper"],
                          self.SURFACE_COLORS["upper"],
                          fill_alpha=0.55, edge_alpha=0.95, edge_lw=1.4)
            legend_elements.append(
                Patch(facecolor=self.SURFACE_COLORS["upper"],
                       alpha=0.55, label="Lower Surface"))
        # Base / side panels — muted gray
        if cat_faces["base"]:
            _draw_surface(cat_faces["base"],
                          self.SURFACE_COLORS["base"],
                          fill_alpha=0.30, edge_alpha=0.7, edge_lw=0.8)
            legend_elements.append(
                Patch(facecolor=self.SURFACE_COLORS["base"],
                       alpha=0.4, label="Base / Side"))
        # Fins — saturated burnt orange
        if cat_faces["fin"]:
            _draw_surface(cat_faces["fin"],
                          self.SURFACE_COLORS["fin"],
                          fill_alpha=0.6, edge_alpha=0.95, edge_lw=1.4)
            legend_elements.append(
                Patch(facecolor=self.SURFACE_COLORS["fin"],
                       alpha=0.6, label="Fins"))
        if cat_faces["other"]:
            _draw_surface(cat_faces["other"],
                          self.SURFACE_COLORS["other"],
                          fill_alpha=0.30, edge_alpha=0.6, edge_lw=0.5)

        # ----- Leading edge: thick black polyline ------------------
        le_xyz = self._extract_LE_polyline(mesh, cat_faces)
        if le_xyz is not None and len(le_xyz) >= 2:
            # plot frame: x=span(mesh.y), y=streamwise(mesh.x),
            # z=vertical(mesh.z). NaN rows separate disjoint wing
            # chains in a single plot call.
            self.ax.plot(le_xyz[:, 1], le_xyz[:, 0], le_xyz[:, 2],
                          color="black", linewidth=3.0,
                          solid_capstyle="round", zorder=10)
            legend_elements.append(
                Line2D([0], [0], color="black", linewidth=3.0,
                        label="Leading Edge"))

        # ----- Style ----------------------------------------------
        self.ax.set_xlabel("Z (Span)", color="#FFFFFF")
        self.ax.set_ylabel("X (Streamwise)", color="#FFFFFF")
        self.ax.set_zlabel("Y (Vertical)", color="#FFFFFF")
        kind = mesh.metadata.get("kind", "?")
        # If the title_prefix already names the mode, do not repeat it
        if kind.lower() in title_prefix.lower():
            title = title_prefix
        else:
            title = f"{title_prefix}: {kind}"
        if title_extra:
            title = f"{title} {title_extra}"
        self.ax.set_title(title, color="#FFFFFF")
        self.ax.tick_params(colors="#888888")

        if legend_elements:
            leg = self.ax.legend(handles=legend_elements,
                                    loc="upper left", fontsize=9)
            # Default Matplotlib gives dark text on a white legend
            # frame; that's already correct against our white legend
            # background, so we leave the text color alone.

        self._set_axes_equal_from_mesh(v)
        # Sensible default view: looking at the body from above-front,
        # apex on the left, wings receding into screen.
        try:
            self.ax.view_init(elev=22, azim=-60)
        except Exception:
            pass

        # Info panel
        if info is not None:
            self._draw_info_panel(info)

        self.fig.tight_layout()
        self.draw()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _extract_LE_polyline(
        self, mesh: Mesh,
        cat_faces: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """Trace the leading-edge polyline.

        Strategy:
        1. If lower & upper triangles share edges directly (caret,
           flat-delta), use those.
        2. Otherwise (engineering modes), take the *boundary* of the
           lower-surface mesh — edges that appear in exactly one
           lower triangle — and pick the chain whose vertices lie on
           the spanwise extremity (max |y| for each x). That is the
           swept LE between the lower surface and the side panels.

        Returns ``None`` if no LE can be reconstructed."""
        try:
            if cat_faces is None:
                cat_faces = {"lower": [], "upper": []}
                for face, lbl in zip(mesh.faces, mesh.labels):
                    cat = _classify_label(lbl)
                    cat_faces.setdefault(cat, []).append(face)

            def _edge_counts(face_list):
                """Map each undirected edge -> count of triangles
                that include it."""
                ec: dict = {}
                for face in face_list:
                    a, b, c = int(face[0]), int(face[1]), int(face[2])
                    for u, v_ in ((a, b), (b, c), (c, a)):
                        e = tuple(sorted((u, v_)))
                        ec[e] = ec.get(e, 0) + 1
                return ec

            lower_ec = _edge_counts(cat_faces.get("lower", []))
            upper_ec = _edge_counts(cat_faces.get("upper", []))

            # Strategy 1: lower&upper share edges directly
            lu_shared = sorted(set(lower_ec) & set(upper_ec))

            # Strategy 2: boundary of lower (edges in exactly one
            # lower triangle), restricted to vertices on the +y
            # spanwise side (right wing, half-body convention). The
            # mirror-image LE is implied by symmetry.
            if lu_shared:
                edges_for_chain = lu_shared
            else:
                lower_boundary = [e for e, c in lower_ec.items() if c == 1]
                ys = mesh.vertices[:, 1]
                # Drop edges along the body centerline (y ≈ 0 on both
                # endpoints) — those are not LE, just symmetry seam
                edges_for_chain = [
                    e for e in lower_boundary
                    if not (abs(ys[e[0]]) < 1e-6 and abs(ys[e[1]]) < 1e-6)
                ]

            if not edges_for_chain:
                return None

            # Build adjacency for the candidate edges
            adj: dict = {}
            for a, b in edges_for_chain:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)

            visited_edges = set()
            visited_verts = set()
            chains = []  # list of np.ndarray polylines

            def _walk_chain(start):
                chain = [start]
                cur = start
                prev = None
                while True:
                    nbrs = [n for n in adj[cur]
                            if tuple(sorted((cur, n))) not in visited_edges]
                    if not nbrs:
                        break
                    nbrs.sort(key=lambda n: mesh.vertices[n, 0], reverse=True)
                    if prev is not None and prev in nbrs and len(nbrs) > 1:
                        nbrs = [n for n in nbrs if n != prev]
                    nxt = nbrs[0]
                    visited_edges.add(tuple(sorted((cur, nxt))))
                    chain.append(nxt)
                    prev, cur = cur, nxt
                    if cur == start:
                        break
                return chain

            # Walk every connected component starting from a vertex
            # that has not yet been visited. For each component start
            # at its lowest-x vertex (likely the apex / wing root).
            remaining = set(adj.keys())
            while remaining:
                start = min(remaining,
                            key=lambda i: mesh.vertices[i, 0])
                chain = _walk_chain(start)
                if len(chain) >= 2:
                    chains.append(mesh.vertices[chain])
                remaining -= set(chain)
                # Avoid infinite loops on disconnected isolated verts
                if len(chain) == 1:
                    break

            if not chains:
                return None
            # Concatenate with NaN separators so a single ax.plot()
            # call breaks between disjoint LE chains (left & right
            # wing).
            sep = np.full((1, 3), np.nan)
            joined = []
            for i, c in enumerate(chains):
                if i > 0:
                    joined.append(sep)
                joined.append(c)
            return np.vstack(joined)
        except Exception:
            return None

    def _set_axes_equal_from_mesh(self, v: np.ndarray):
        try:
            mins = v.min(axis=0); maxs = v.max(axis=0)
            center = 0.5 * (mins + maxs)
            radius = 0.5 * float(np.max(maxs - mins))
            if radius <= 0:
                radius = 1.0
            # Plot frame: ax_x=mesh.y(span), ax_y=mesh.x(stream),
            # ax_z=mesh.z(vertical).
            self.ax.set_xlim3d(center[1] - radius, center[1] + radius)
            self.ax.set_ylim3d(center[0] - radius, center[0] + radius)
            self.ax.set_zlim3d(center[2] - radius, center[2] + radius)
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  Info panel (top-left orange-bordered dark box)
    # ------------------------------------------------------------------

    def _draw_info_panel(self, info: dict, title: str = "PSWR-1 Variable Wedge"):
        # Render header from the info dict if provided, else default
        title_line = info.pop("__title__", title) if isinstance(info, dict) else title
        # Format: left-padded keys, right-padded values, monospace
        max_key = max((len(str(k)) for k in info.keys()), default=12)
        max_key = max(max_key, 12)
        lines = [title_line]
        for k, val in info.items():
            lines.append(f"  {str(k):<{max_key}}  {val}")
        text_block = "\n".join(lines)

        self._info_text = self.fig.text(
            0.02, 0.98, text_block,
            transform=self.fig.transFigure,
            fontsize=8, fontfamily="monospace",
            verticalalignment="top", color="white",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#1A1A1A",
                      edgecolor="#D97706", alpha=0.85),
        )


class _SimpleCanvas(FigureCanvas):
    """Generic 2-D plot canvas with a placeholder."""

    def __init__(self, title: str, parent=None):
        self.fig = Figure(figsize=(9, 4))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._title = title
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Run a sweep to populate.",
                     ha="center", va="center", color="gray",
                     transform=self.ax.transAxes)
        self.ax.set_title(self._title)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.tight_layout()
        self.draw()


# ======================================================================
#  Sweep worker (QThread)
# ======================================================================

class _SweepWorker(QThread):
    progress = pyqtSignal(int, int)   # cells_done, cells_total
    finished_ok = pyqtSignal(object)   # the DataFrame
    finished_err = pyqtSignal(str)

    def __init__(self, mesh: Mesh, sw_cfg: SweepConfig, parent=None):
        super().__init__(parent)
        self.mesh = mesh
        self.sw_cfg = sw_cfg
        self._n_total = sw_cfg.M_grid[2] * sw_cfg.alpha_grid_deg[2]

    def run(self):
        try:
            done = [0]
            def on_cell(i, j, M, a, row):
                done[0] += 1
                self.progress.emit(done[0], self._n_total)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                df = mach_alpha_sweep(self.mesh, self.sw_cfg,
                                       on_cell=on_cell)
            self.finished_ok.emit(df)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.finished_err.emit(f"{type(e).__name__}: {e}")


# ======================================================================
#  Mode-specific parameter widgets
# ======================================================================

def _spin(lo, hi, default, step, decimals=3, suffix="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(default)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    if suffix:
        s.setSuffix(" " + suffix)
    return s


class _EngFlatPage(QWidget):
    """Engineering flat-bottom mode parameter page."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        gb = QGroupBox("Engineering flat-bottom geometry")
        g = QGridLayout(); row = 0
        self.M_design = _spin(2, 25, 15.0, 0.5, 2)
        self.M_design.setToolTip("Design Mach number for forebody shock attachment.\n"
                                   "Typical HTV-2 class: 15-20.")
        g.addWidget(QLabel("M_design:"), row, 0); g.addWidget(self.M_design, row, 1); row += 1
        self.theta_fore = _spin(0.5, 30, 8.0, 0.5, 2, "deg")
        self.theta_fore.setToolTip("Forebody lower-surface inclination (wedge angle).\n"
                                     "Sets the design oblique-shock angle via theta-beta-M.")
        g.addWidget(QLabel("theta_fore:"), row, 0); g.addWidget(self.theta_fore, row, 1); row += 1
        self.Lambda = _spin(40, 85, 75.0, 1.0, 2, "deg")
        self.Lambda.setToolTip("LE sweep from spanwise axis. 55-80 deg typical.\n"
                                  "Higher = narrower planform, sweeps LE heating.")
        g.addWidget(QLabel("Lambda:"), row, 0); g.addWidget(self.Lambda, row, 1); row += 1
        self.L_fore = _spin(0.5, 10.0, 2.5, 0.1, 3, "m")
        self.L_fore.setToolTip("Forebody (compression-surface) streamwise length.")
        g.addWidget(QLabel("L_fore:"), row, 0); g.addWidget(self.L_fore, row, 1); row += 1
        self.L_center = _spin(0.0, 5.0, 1.5, 0.1, 3, "m")
        self.L_center.setToolTip("Centerbody (prismatic frustum) streamwise length.")
        g.addWidget(QLabel("L_center:"), row, 0); g.addWidget(self.L_center, row, 1); row += 1
        self.b_base = _spin(0.05, 5.0, 0.5, 0.05, 3, "m")
        self.b_base.setToolTip("Half-width of the base. Must be <= b_LE_fore.")
        g.addWidget(QLabel("b_base:"), row, 0); g.addWidget(self.b_base, row, 1); row += 1
        self.h_base = _spin(0.05, 2.0, 0.4, 0.05, 3, "m")
        self.h_base.setToolTip("Base height (vertical extent of the base rectangle).")
        g.addWidget(QLabel("h_base:"), row, 0); g.addWidget(self.h_base, row, 1); row += 1
        self.r_LE = _spin(0.5, 50, 5.0, 0.5, 1, "mm")
        self.r_LE.setToolTip("Leading-edge radius. Sharp 1 mm = aggressive heating.")
        g.addWidget(QLabel("r_LE:"), row, 0); g.addWidget(self.r_LE, row, 1); row += 1
        self.r_nose = _spin(0.5, 100, 10.0, 0.5, 1, "mm")
        self.r_nose.setToolTip("Nose-tip spherical radius.")
        g.addWidget(QLabel("r_nose:"), row, 0); g.addWidget(self.r_nose, row, 1); row += 1
        self.theta_upper = _spin(0.0, 10.0, 0.0, 0.5, 2, "deg")
        self.theta_upper.setToolTip("Upper-surface inclination (0 = horizontal).")
        g.addWidget(QLabel("theta_upper:"), row, 0); g.addWidget(self.theta_upper, row, 1); row += 1
        gb.setLayout(g); layout.addWidget(gb)
        layout.addStretch()

    def get_config(self) -> EngineeringFlatConfig:
        return EngineeringFlatConfig(
            M_design=self.M_design.value(),
            theta_fore_deg=self.theta_fore.value(),
            Lambda_deg=self.Lambda.value(),
            L_fore=self.L_fore.value(),
            L_center=self.L_center.value(),
            b_base=self.b_base.value(),
            h_base=self.h_base.value(),
            r_LE_mm=self.r_LE.value(),
            r_nose_mm=self.r_nose.value(),
            theta_upper_deg=self.theta_upper.value(),
        )


class _EngShallowVPage(_EngFlatPage):
    """Shallow-V variant: same as flat-bottom plus dihedral_lower."""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Add dihedral spinbox to existing first GroupBox layout
        gb = self.findChild(QGroupBox)
        gl = gb.layout()
        row = gl.rowCount()
        self.dihedral_lower = _spin(0.0, 20.0, 5.0, 0.5, 2, "deg")
        self.dihedral_lower.setToolTip(
            "V-trough dihedral angle below the planform.\n"
            "0 deg = flat bottom; ~5 deg = mild lateral stability boost.")
        gl.addWidget(QLabel("dihedral_lower:"), row, 0)
        gl.addWidget(self.dihedral_lower, row, 1)

    def get_config(self) -> EngineeringShallowVConfig:
        f = super().get_config()
        return EngineeringShallowVConfig(
            M_design=f.M_design,
            theta_fore_deg=f.theta_fore_deg,
            Lambda_deg=f.Lambda_deg,
            L_fore=f.L_fore,
            L_center=f.L_center,
            b_base=f.b_base,
            h_base=f.h_base,
            r_LE_mm=f.r_LE_mm,
            r_nose_mm=f.r_nose_mm,
            theta_upper_deg=f.theta_upper_deg,
            dihedral_lower_deg=self.dihedral_lower.value(),
        )


class _CaretPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        gb = QGroupBox("Nonweiler caret"); g = QGridLayout(); row = 0
        self.M_design = _spin(2, 25, 6.0, 0.5, 2)
        self.M_design.setToolTip(
            "Design Mach number. The caret rides the planar oblique\n"
            "shock generated at this M with deflection theta_d.")
        g.addWidget(QLabel("M_design:"), row, 0); g.addWidget(self.M_design, row, 1); row += 1
        self.theta_d = _spin(2, 30, 14.0, 0.5, 2, "deg")
        self.theta_d.setToolTip(
            "Lower-surface ramp angle (flow-deflection angle).\n"
            "Anderson Ch. 14: inviscid L/D = 1/tan(theta_d).")
        g.addWidget(QLabel("theta_d:"), row, 0); g.addWidget(self.theta_d, row, 1); row += 1
        self.Lambda = _spin(40, 85, 70.0, 1.0, 2, "deg")
        self.Lambda.setToolTip(
            "Leading-edge sweep from the spanwise axis.\n"
            "Higher sweep narrows the planform and reduces LE heating.")
        g.addWidget(QLabel("Lambda:"), row, 0); g.addWidget(self.Lambda, row, 1); row += 1
        self.L = _spin(0.5, 50, 10.0, 0.5, 3, "m")
        self.L.setToolTip("Streamwise body length (apex to base).")
        g.addWidget(QLabel("L:"), row, 0); g.addWidget(self.L, row, 1); row += 1
        gb.setLayout(g); layout.addWidget(gb); layout.addStretch()

    def get_config(self) -> CaretConfig:
        return CaretConfig(
            M_design=self.M_design.value(),
            theta_d_deg=self.theta_d.value(),
            Lambda_deg=self.Lambda.value(),
            L=self.L.value(),
        )


class _FlatDeltaPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        gb = QGroupBox("Flat-bottomed delta"); g = QGridLayout(); row = 0
        self.M_design = _spin(2, 25, 5.0, 0.5, 2)
        self.M_design.setToolTip(
            "Design Mach number for the planar oblique-shock attached\n"
            "to the lower flat surface at theta_d.")
        g.addWidget(QLabel("M_design:"), row, 0); g.addWidget(self.M_design, row, 1); row += 1
        self.theta_d = _spin(2, 30, 12.0, 0.5, 2, "deg")
        self.theta_d.setToolTip(
            "Lower-surface flow-deflection angle (single ramp).\n"
            "Used in the theta-beta-M relation to set the design shock.")
        g.addWidget(QLabel("theta_d:"), row, 0); g.addWidget(self.theta_d, row, 1); row += 1
        self.Lambda = _spin(40, 85, 75.0, 1.0, 2, "deg")
        self.Lambda.setToolTip(
            "Leading-edge sweep. Lambda=0 reduces to a flat plate;\n"
            "high Lambda detaches the swept shock at high theta_d.")
        g.addWidget(QLabel("Lambda:"), row, 0); g.addWidget(self.Lambda, row, 1); row += 1
        self.L = _spin(0.5, 50, 8.0, 0.5, 3, "m")
        self.L.setToolTip("Streamwise body length (apex to base).")
        g.addWidget(QLabel("L:"), row, 0); g.addWidget(self.L, row, 1); row += 1
        gb.setLayout(g); layout.addWidget(gb); layout.addStretch()

    def get_config(self) -> FlatDeltaConfig:
        return FlatDeltaConfig(
            M_design=self.M_design.value(),
            theta_d_deg=self.theta_d.value(),
            Lambda_deg=self.Lambda.value(),
            L=self.L.value(),
        )


class _MultiWedgePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        gb = QGroupBox("Multi-wedge / Oswatitsch"); g = QGridLayout(); row = 0
        self.M_design = _spin(2, 25, 5.0, 0.5, 2)
        self.M_design.setToolTip(
            "Freestream design Mach number entering the first ramp.")
        g.addWidget(QLabel("M_design:"), row, 0); g.addWidget(self.M_design, row, 1); row += 1
        self.n = QSpinBox(); self.n.setRange(1, 6); self.n.setValue(2)
        self.n.setToolTip(
            "Number of equal-strength oblique-shock ramps.\n"
            "n=1 collapses to a single shock; pi_OS rises with n.")
        g.addWidget(QLabel("n ramps:"), row, 0); g.addWidget(self.n, row, 1); row += 1
        self.delta_total = _spin(2, 50, 20.8, 0.5, 2, "deg")
        self.delta_total.setToolTip(
            "Total flow deflection across all n ramps. The Oswatitsch\n"
            "solver splits this into equal-strength increments.")
        g.addWidget(QLabel("delta_total:"), row, 0); g.addWidget(self.delta_total, row, 1); row += 1
        self.L = _spin(0.5, 50, 8.0, 0.5, 3, "m")
        self.L.setToolTip("Total streamwise length of the ramp stack.")
        g.addWidget(QLabel("L:"), row, 0); g.addWidget(self.L, row, 1); row += 1
        self.half_span = _spin(0.05, 5.0, 1.0, 0.05, 3, "m")
        self.half_span.setToolTip("Spanwise half-width (z-extent of half-body).")
        g.addWidget(QLabel("half_span:"), row, 0); g.addWidget(self.half_span, row, 1); row += 1
        self.height = _spin(0.05, 5.0, 0.6, 0.05, 3, "m")
        self.height.setToolTip("Vertical body height (used by both extrusion modes).")
        g.addWidget(QLabel("height:"), row, 0); g.addWidget(self.height, row, 1); row += 1
        self.extrusion = QComboBox(); self.extrusion.addItems(["rectangular", "delta"])
        self.extrusion.setToolTip(
            "Spanwise extrusion mode:\n"
            "  rectangular = constant cross-section (inlet style)\n"
            "  delta       = triangular planform (waverider style).")
        g.addWidget(QLabel("extrusion:"), row, 0); g.addWidget(self.extrusion, row, 1); row += 1
        gb.setLayout(g); layout.addWidget(gb); layout.addStretch()

    def get_config(self) -> MultiWedgeConfig:
        return MultiWedgeConfig(
            M_design=self.M_design.value(),
            n=self.n.value(),
            delta_total_deg=self.delta_total.value(),
            L=self.L.value(),
            half_span=self.half_span.value(),
            extrusion=self.extrusion.currentText(),
            height=self.height.value(),
        )


# ======================================================================
#  Per-fin parameter panel
# ======================================================================

def _default_fin_specs(n: int, body_attach_x: float = 3.25,
                        body_attach_z: float = -0.05) -> list:
    """Default per-fin layouts: distribute n fins evenly around the
    body's +x axis, attachment at ``body_attach_x`` (just below the
    centerbody centreline at z = body_attach_z). Roll = 0 places the
    fin span along +y; angles increase CCW about +x.

    n=2 → vertical pair (rolls 90, 270)
    n=4 → X-tail (rolls 45, 135, 225, 315)
    n=3 → tri-tail (rolls 90, 210, 330)
    other n → uniform spacing starting at 45°
    """
    if n <= 0:
        return []
    if n == 2:
        rolls = [90.0, 270.0]
    elif n == 4:
        rolls = [45.0, 135.0, 225.0, 315.0]
    else:
        # Uniform spacing
        rolls = [(i * 360.0 / n + 45.0) % 360.0 for i in range(n)]
    out = []
    for r in rolls:
        out.append({
            "root_chord": 0.30, "tip_chord": 0.10, "span": 0.40,
            "sweep_LE_deg": 45.0, "t_c": 0.05,
            "max_thickness_loc": 0.50,
            "LE_style": "blunt_cylinder", "LE_radius_mm": 1.0,
            "pos_x": body_attach_x, "pos_y": 0.0, "pos_z": body_attach_z,
            "roll_deg": r, "pitch_deg": 0.0, "yaw_deg": 0.0,
        })
    return out


class _FinPanel(QGroupBox):
    """Collapsible per-fin parameter group: geometry + position +
    rotation. Each fin in the assembly gets its own ``_FinPanel``."""

    def __init__(self, fin_idx: int, parent=None):
        super().__init__(f"Fin {fin_idx}", parent)
        self.fin_idx = fin_idx
        self.setCheckable(False)
        layout = QGridLayout(); row = 0

        # ---- Geometry ----
        self.root_chord = _spin(0.05, 1.0, 0.30, 0.05, 3, "m")
        self.root_chord.setToolTip("Streamwise chord at the fin root.")
        layout.addWidget(QLabel("root chord:"), row, 0)
        layout.addWidget(self.root_chord, row, 1); row += 1

        self.tip_chord = _spin(0.02, 1.0, 0.10, 0.05, 3, "m")
        self.tip_chord.setToolTip("Streamwise chord at the fin tip.")
        layout.addWidget(QLabel("tip chord:"), row, 0)
        layout.addWidget(self.tip_chord, row, 1); row += 1

        self.span = _spin(0.05, 2.0, 0.40, 0.05, 3, "m")
        self.span.setToolTip("Fin span (root-to-tip distance along the local +y axis).")
        layout.addWidget(QLabel("span:"), row, 0)
        layout.addWidget(self.span, row, 1); row += 1

        self.sweep_LE = _spin(0.0, 70.0, 45.0, 1.0, 1, "deg")
        self.sweep_LE.setToolTip("Leading-edge sweep of the fin planform.")
        layout.addWidget(QLabel("LE sweep:"), row, 0)
        layout.addWidget(self.sweep_LE, row, 1); row += 1

        self.t_c = _spin(0.02, 0.10, 0.05, 0.005, 3)
        self.t_c.setToolTip("Thickness-to-chord ratio of the diamond section.")
        layout.addWidget(QLabel("t/c:"), row, 0)
        layout.addWidget(self.t_c, row, 1); row += 1

        self.xt_c = _spin(0.30, 0.70, 0.50, 0.05, 2)
        self.xt_c.setToolTip(
            "Chordwise location of max thickness (0.5 = symmetric).")
        layout.addWidget(QLabel("x_t/c:"), row, 0)
        layout.addWidget(self.xt_c, row, 1); row += 1

        self.LE_style = QComboBox()
        self.LE_style.addItems(["sharp", "blunt_cylinder"])
        self.LE_style.setCurrentText("blunt_cylinder")
        self.LE_style.setToolTip(
            "  sharp           - zero LE radius\n"
            "  blunt_cylinder  - cylindrical LE for heating realism")
        layout.addWidget(QLabel("LE style:"), row, 0)
        layout.addWidget(self.LE_style, row, 1); row += 1

        self.LE_radius = _spin(0.5, 5.0, 1.0, 0.1, 1, "mm")
        self.LE_radius.setToolTip(
            "Cylindrical-LE radius (used when LE style = blunt_cylinder).")
        layout.addWidget(QLabel("LE radius:"), row, 0)
        layout.addWidget(self.LE_radius, row, 1); row += 1

        # ---- Position (body frame) ----
        layout.addWidget(QLabel("--- Position (body frame) ---"),
                          row, 0, 1, 2); row += 1
        self.pos_x = _spin(-2.0, 20.0, 3.25, 0.05, 3, "m")
        self.pos_x.setToolTip(
            "Streamwise position of the fin's local-frame origin (root LE).")
        layout.addWidget(QLabel("position x:"), row, 0)
        layout.addWidget(self.pos_x, row, 1); row += 1

        self.pos_y = _spin(-3.0, 3.0, 0.0, 0.05, 3, "m")
        self.pos_y.setToolTip(
            "Spanwise position of the fin's local-frame origin.")
        layout.addWidget(QLabel("position y:"), row, 0)
        layout.addWidget(self.pos_y, row, 1); row += 1

        self.pos_z = _spin(-3.0, 3.0, -0.05, 0.05, 3, "m")
        self.pos_z.setToolTip(
            "Vertical position of the fin's local-frame origin.")
        layout.addWidget(QLabel("position z:"), row, 0)
        layout.addWidget(self.pos_z, row, 1); row += 1

        # ---- Rotation (extrinsic XYZ) ----
        layout.addWidget(QLabel("--- Rotation (extrinsic XYZ) ---"),
                          row, 0, 1, 2); row += 1
        self.roll = _spin(-180.0, 360.0, 90.0, 1.0, 2, "deg")
        self.roll.setToolTip(
            "Roll angle (rotation about the body x-axis).\n"
            "0   = fin span along +y\n"
            "90  = fin span along +z (vertical)\n"
            "45  = X-tail upper-right.")
        layout.addWidget(QLabel("roll  (about X):"), row, 0)
        layout.addWidget(self.roll, row, 1); row += 1

        self.pitch = _spin(-90.0, 90.0, 0.0, 1.0, 2, "deg")
        self.pitch.setToolTip(
            "Pitch angle (rotation about body y-axis after roll).")
        layout.addWidget(QLabel("pitch (about Y):"), row, 0)
        layout.addWidget(self.pitch, row, 1); row += 1

        self.yaw = _spin(-90.0, 90.0, 0.0, 1.0, 2, "deg")
        self.yaw.setToolTip(
            "Yaw angle (rotation about body z-axis, applied last).")
        layout.addWidget(QLabel("yaw   (about Z):"), row, 0)
        layout.addWidget(self.yaw, row, 1); row += 1

        self.setLayout(layout)

    # --------------------------------------------------------------
    def to_spec(self) -> dict:
        return {
            "root_chord": self.root_chord.value(),
            "tip_chord": self.tip_chord.value(),
            "span": self.span.value(),
            "sweep_LE_deg": self.sweep_LE.value(),
            "t_c": self.t_c.value(),
            "max_thickness_loc": self.xt_c.value(),
            "LE_style": self.LE_style.currentText(),
            "LE_radius_mm": self.LE_radius.value(),
            "pos_x": self.pos_x.value(),
            "pos_y": self.pos_y.value(),
            "pos_z": self.pos_z.value(),
            "roll_deg": self.roll.value(),
            "pitch_deg": self.pitch.value(),
            "yaw_deg": self.yaw.value(),
        }

    def set_from_spec(self, spec: dict) -> None:
        self.root_chord.setValue(float(spec.get("root_chord", 0.3)))
        self.tip_chord.setValue(float(spec.get("tip_chord", 0.1)))
        self.span.setValue(float(spec.get("span", 0.4)))
        self.sweep_LE.setValue(float(spec.get("sweep_LE_deg", 45.0)))
        self.t_c.setValue(float(spec.get("t_c", 0.05)))
        self.xt_c.setValue(float(spec.get("max_thickness_loc", 0.5)))
        self.LE_style.setCurrentText(str(spec.get("LE_style", "blunt_cylinder")))
        self.LE_radius.setValue(float(spec.get("LE_radius_mm", 1.0)))
        self.pos_x.setValue(float(spec.get("pos_x", 3.25)))
        self.pos_y.setValue(float(spec.get("pos_y", 0.0)))
        self.pos_z.setValue(float(spec.get("pos_z", -0.05)))
        self.roll.setValue(float(spec.get("roll_deg", 0.0)))
        self.pitch.setValue(float(spec.get("pitch_deg", 0.0)))
        self.yaw.setValue(float(spec.get("yaw_deg", 0.0)))


# ======================================================================
#  Main GVWD tab
# ======================================================================

class GVWDWaveriderTab(QWidget):
    """GVWD geometry hub tab with five geometry modes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.geom = None              # current geometry object (Caret, FlatDelta, ...)
        self.body_mesh: Optional[Mesh] = None
        self.fins_mesh: Optional[Mesh] = None
        self.full_mesh: Optional[Mesh] = None
        self.on_design: Optional[dict] = None
        self.sweep_df = None
        self._sweep_worker: Optional[_SweepWorker] = None
        self._init_ui()

    # ------------------------------------------------------------------
    #  UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        main = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # ---- Left ----
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(330)
        left_w = QWidget(); left_l = QVBoxLayout(left_w)

        # Mode selector
        mode_group = QGroupBox("Mode")
        mg = QGridLayout()
        self.mode_combo = QComboBox()
        for _, label in _MODE_LABELS:
            self.mode_combo.addItem(label)
        self.mode_combo.setToolTip(
            "Geometry generator selector:\n"
            "  Engineering modes: HTV-2 / Fattah-2 archetypes (use\n"
            "    these for production design).\n"
            "  Reference modes: caret / flat-delta / multi-wedge —\n"
            "    closed-form analytic-truth shapes for cross-checks.")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mg.addWidget(QLabel("Geometry mode:"), 0, 0)
        mg.addWidget(self.mode_combo, 0, 1)
        mode_group.setLayout(mg); left_l.addWidget(mode_group)

        # Stacked parameter pages (one per mode)
        self.param_stack = QStackedWidget()
        self.page_eflat = _EngFlatPage()
        self.page_eshallowv = _EngShallowVPage()
        self.page_caret = _CaretPage()
        self.page_fdelta = _FlatDeltaPage()
        self.page_mwedge = _MultiWedgePage()
        for p in (self.page_eflat, self.page_eshallowv, self.page_caret,
                   self.page_fdelta, self.page_mwedge):
            self.param_stack.addWidget(p)
        left_l.addWidget(self.param_stack)

        # Fins (engineering modes only — visibility toggled in _on_mode_changed)
        self.fins_group = self._create_fins_group()
        left_l.addWidget(self.fins_group)

        # Atmosphere / Sweep grid (always visible)
        left_l.addWidget(self._create_atm_group())
        left_l.addWidget(self._create_sweep_group())

        # Action buttons
        left_l.addWidget(self._create_actions_group())

        # Outputs panel
        self.output_label = QLabel("Run 'Generate Geometry' to start.")
        self.output_label.setStyleSheet(
            "color: #aaaaaa; font-size: 10px; font-family: monospace; "
            "background-color: #1A1A1A; padding: 6px; border-radius: 3px;")
        self.output_label.setWordWrap(True)
        left_l.addWidget(self.output_label)
        left_l.addStretch()
        left_scroll.setWidget(left_w)
        splitter.addWidget(left_scroll)

        # ---- Right ----
        right_w = QWidget(); right_l = QVBoxLayout(right_w)
        # 3D canvas + toolbar
        self.canvas_3d = _Mesh3DCanvas()
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)
        right_l.addWidget(self.toolbar_3d)
        right_l.addWidget(self.canvas_3d, 5)

        # Bottom analysis tabs
        self.bottom_tabs = QTabWidget()
        self.canvas_LD_heat = _SimpleCanvas("L/D vs Mach and AoA")
        self.canvas_qLE_heat = _SimpleCanvas("LE heat flux vs Mach and AoA")
        self.canvas_polar = _SimpleCanvas("Drag polar")
        self.canvas_LDvsM = _SimpleCanvas("L/D vs Mach (per AoA)")
        self.canvas_shockdet = _SimpleCanvas("Shock-detachment margin")
        self.bottom_tabs.addTab(self.canvas_LD_heat, "L/D heatmap")
        self.bottom_tabs.addTab(self.canvas_qLE_heat, "q_LE heatmap")
        self.bottom_tabs.addTab(self.canvas_polar, "Polar")
        self.bottom_tabs.addTab(self.canvas_LDvsM, "L/D vs M")
        self.bottom_tabs.addTab(self.canvas_shockdet, "Shock detach")
        right_l.addWidget(self.bottom_tabs, 4)
        splitter.addWidget(right_w)

        splitter.setSizes([400, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main.addWidget(splitter)

        self._on_mode_changed(0)   # init visibility

    # ------------------------------------------------------------------
    #  Subgroup builders
    # ------------------------------------------------------------------

    def _create_fins_group(self) -> QGroupBox:
        g = QGroupBox("Aft control fins (per-fin)")
        g.setToolTip(
            "Diamond-airfoil control fins. Each fin has independent\n"
            "geometry, position (x, y, z), and rotation (roll/pitch/yaw).\n"
            "Engineering modes (flat / shallow-V) only.")
        outer = QVBoxLayout()

        # Header with enable + count
        header = QGridLayout(); hr = 0
        self.fins_enable = QCheckBox("Enable fins")
        self.fins_enable.setChecked(False)
        self.fins_enable.setToolTip(
            "Toggle to add fins to the merged mesh.")
        header.addWidget(self.fins_enable, hr, 0, 1, 2); hr += 1
        self.fins_n = QSpinBox(); self.fins_n.setRange(0, 6)
        self.fins_n.setValue(4)
        self.fins_n.setToolTip(
            "Number of fins (0-6). Each fin gets its own parameter\n"
            "panel below; defaults are populated for common layouts:\n"
            "  2 → vertical pair  /  4 → X-tail  /  3 → tri-tail.")
        self.fins_n.valueChanged.connect(self._rebuild_fin_panels)
        header.addWidget(QLabel("n_fins:"), hr, 0)
        header.addWidget(self.fins_n, hr, 1); hr += 1
        self.fins_reset_btn = QPushButton("Reset positions to default layout")
        self.fins_reset_btn.setToolTip(
            "Repopulate every fin panel with the default symmetric\n"
            "layout for the current n_fins.")
        self.fins_reset_btn.clicked.connect(
            lambda: self._rebuild_fin_panels(force_reset=True))
        header.addWidget(self.fins_reset_btn, hr, 0, 1, 2); hr += 1

        outer.addLayout(header)

        # Container for per-fin panels (a vertical stack inside a QWidget
        # so we can clear & rebuild it as n_fins changes).
        self._fin_panels_host = QWidget()
        self._fin_panels_layout = QVBoxLayout(self._fin_panels_host)
        self._fin_panels_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._fin_panels_host)
        self.fin_panels: list = []

        g.setLayout(outer)
        # Build the initial fin panels matching the default count
        self._rebuild_fin_panels()
        return g

    def _rebuild_fin_panels(self, *, force_reset: bool = False) -> None:
        """(Re)create per-fin parameter panels based on ``self.fins_n``.

        - If the count grew, new panels are appended with the default
          spec for that index (preserving existing panels' values).
        - If the count shrank, excess panels are deleted.
        - If ``force_reset`` is True, every panel is reset to the
          default layout for the current count.
        """
        if not hasattr(self, "fins_n"):
            return
        n_target = int(self.fins_n.value())
        defaults = _default_fin_specs(n_target)

        # Drop excess panels
        while len(self.fin_panels) > n_target:
            old = self.fin_panels.pop()
            old.setParent(None)
            old.deleteLater()

        # Add missing panels
        while len(self.fin_panels) < n_target:
            idx = len(self.fin_panels) + 1
            panel = _FinPanel(idx, parent=self._fin_panels_host)
            self._fin_panels_layout.addWidget(panel)
            spec = defaults[idx - 1] if idx - 1 < len(defaults) else {}
            panel.set_from_spec(spec)
            self.fin_panels.append(panel)

        if force_reset:
            for i, panel in enumerate(self.fin_panels):
                panel.set_from_spec(defaults[i])

    def _create_atm_group(self) -> QGroupBox:
        g = QGroupBox("Atmosphere & wall")
        layout = QGridLayout(); row = 0
        self.alt_km = _spin(0.0, 80.0, 30.0, 1.0, 1, "km")
        self.alt_km.setToolTip("Flight altitude (US Std 1976).")
        layout.addWidget(QLabel("altitude:"), row, 0); layout.addWidget(self.alt_km, row, 1); row += 1
        self.T_w = _spin(300.0, 3000.0, 1500.0, 50.0, 0, "K")
        self.T_w.setToolTip("Wall temperature for Eckert reference state.")
        layout.addWidget(QLabel("T_w:"), row, 0); layout.addWidget(self.T_w, row, 1); row += 1
        self.Re_x_tr = _spin(1e4, 1e8, 1.0e6, 1e5, 0)
        self.Re_x_tr.setToolTip("Re_x at laminar->turbulent BL transition.")
        layout.addWidget(QLabel("Re_x,tr:"), row, 0); layout.addWidget(self.Re_x_tr, row, 1); row += 1
        g.setLayout(layout)
        return g

    def _create_sweep_group(self) -> QGroupBox:
        g = QGroupBox("Mach-alpha sweep grid")
        g.setToolTip(
            "2-D grid swept by the panel-method solver in a background\n"
            "QThread. Runs after 'Run Mach-alpha Sweep' is clicked.")
        layout = QGridLayout(); row = 0
        self.M_lo = _spin(2.0, 25.0, 5.0, 0.5, 1)
        self.M_lo.setToolTip("Lowest freestream Mach number in the sweep.")
        layout.addWidget(QLabel("M_lo:"), row, 0); layout.addWidget(self.M_lo, row, 1)
        self.M_hi = _spin(2.0, 25.0, 20.0, 0.5, 1)
        self.M_hi.setToolTip("Highest freestream Mach number in the sweep.")
        layout.addWidget(QLabel("M_hi:"), row, 2); layout.addWidget(self.M_hi, row, 3); row += 1
        self.n_M = QSpinBox(); self.n_M.setRange(2, 32); self.n_M.setValue(8)
        self.n_M.setToolTip("Number of Mach grid points (linear from M_lo to M_hi).")
        layout.addWidget(QLabel("n_M:"), row, 0); layout.addWidget(self.n_M, row, 1)
        self.n_alpha = QSpinBox(); self.n_alpha.setRange(2, 32); self.n_alpha.setValue(6)
        self.n_alpha.setToolTip("Number of alpha grid points.")
        layout.addWidget(QLabel("n_alpha:"), row, 2); layout.addWidget(self.n_alpha, row, 3); row += 1
        self.alpha_lo = _spin(-15.0, 30.0, 0.0, 1.0, 1, "deg")
        self.alpha_lo.setToolTip(
            "Lowest angle of attack in the sweep (negative = nose-down).")
        layout.addWidget(QLabel("alpha_lo:"), row, 0); layout.addWidget(self.alpha_lo, row, 1)
        self.alpha_hi = _spin(-15.0, 30.0, 15.0, 1.0, 1, "deg")
        self.alpha_hi.setToolTip("Highest angle of attack in the sweep.")
        layout.addWidget(QLabel("alpha_hi:"), row, 2); layout.addWidget(self.alpha_hi, row, 3); row += 1
        g.setLayout(layout)
        return g

    def _create_actions_group(self) -> QGroupBox:
        g = QGroupBox("Actions")
        layout = QGridLayout(); row = 0

        self.btn_generate = QPushButton("Generate Geometry")
        self.btn_generate.setToolTip(
            "Build the 3-D mesh from the current parameters and render\n"
            "it in the canvas. Fast (typically <300 ms).")
        self.btn_generate.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 6px; }")
        self.btn_generate.clicked.connect(self.generate_geometry)
        layout.addWidget(self.btn_generate, row, 0, 1, 2); row += 1

        self.btn_on_design = QPushButton("Run On-Design Aero")
        self.btn_on_design.setToolTip(
            "Panel-method aero at (M=M_design, alpha=0).\n"
            "Updates CL, CD, L/D in the info panel and output box.")
        self.btn_on_design.clicked.connect(self.run_on_design)
        layout.addWidget(self.btn_on_design, row, 0, 1, 2); row += 1

        self.btn_sweep = QPushButton("Run Mach-alpha Sweep")
        self.btn_sweep.setToolTip(
            "Full (M, alpha) grid sweep. Runs in a background QThread\n"
            "with a per-cell progress bar; populates the bottom-tab\n"
            "heatmaps and polar plots.")
        self.btn_sweep.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white; "
            "font-weight: bold; padding: 6px; }")
        self.btn_sweep.clicked.connect(self.run_sweep)
        layout.addWidget(self.btn_sweep, row, 0, 1, 2); row += 1

        self.sweep_progress = QProgressBar(); self.sweep_progress.setRange(0, 100)
        self.sweep_progress.setToolTip("Sweep progress (cells completed / total).")
        layout.addWidget(self.sweep_progress, row, 0, 1, 2); row += 1

        self.btn_export_stl = QPushButton("Export STL...")
        self.btn_export_stl.setToolTip(
            "Write the current mesh as binary STL (mm units).")
        self.btn_export_stl.clicked.connect(self.export_stl_dialog)
        layout.addWidget(self.btn_export_stl, row, 0)
        self.btn_export_step = QPushButton("Export STEP...")
        self.btn_export_step.setToolTip(
            "Write the current mesh as a CAD-grade STEP file via\n"
            "cadquery (BREP solid, mm units).")
        self.btn_export_step.clicked.connect(self.export_step_dialog)
        layout.addWidget(self.btn_export_step, row, 1); row += 1

        help_btn = QPushButton("?")
        help_btn.setFixedWidth(32)
        help_btn.setToolTip(
            "Open the in-app help dialog (modes, actions, physics limits).")
        help_btn.clicked.connect(self.show_help_dialog)
        layout.addWidget(help_btn, row, 0)

        g.setLayout(layout)
        return g

    # ------------------------------------------------------------------
    #  Mode switching
    # ------------------------------------------------------------------

    def _on_mode_changed(self, idx: int):
        self.param_stack.setCurrentIndex(idx)
        # Fins only meaningful for engineering modes (idx 0, 1)
        is_eng = idx in (0, 1)
        self.fins_group.setVisible(is_eng)
        self.fins_group.setEnabled(is_eng)

    @property
    def current_mode(self) -> str:
        return _MODE_LABELS[self.mode_combo.currentIndex()][0]

    def _current_geom_config(self):
        idx = self.mode_combo.currentIndex()
        page = self.param_stack.currentWidget()
        return page.get_config()

    def _build_run_config(self) -> GVWDRunConfig:
        sweep = SweepRunConfig(
            enabled=False,
            M_grid=(self.M_lo.value(), self.M_hi.value(), self.n_M.value()),
            alpha_grid_deg=(self.alpha_lo.value(), self.alpha_hi.value(),
                              self.n_alpha.value()),
            altitude_km=self.alt_km.value(),
            T_w=self.T_w.value(),
            Re_x_tr=self.Re_x_tr.value(),
        )
        # FinsConfig (library) carries a single set of fin params for
        # legacy paths (provenance hash, save/load via YAML). Pull
        # representative values from the first fin panel; per-fin
        # placement overrides are stored separately in the GUI and
        # applied during generate_geometry().
        n_fins_eff = (self.fins_n.value() if self.fins_enable.isChecked()
                      and self.fins_group.isEnabled() else 0)
        if self.fin_panels:
            fp0 = self.fin_panels[0].to_spec()
        else:
            fp0 = {"root_chord": 0.30, "tip_chord": 0.10, "span": 0.40,
                    "sweep_LE_deg": 45.0, "t_c": 0.05,
                    "max_thickness_loc": 0.50,
                    "LE_style": "blunt_cylinder", "LE_radius_mm": 1.0}
        fins = FinsConfig(
            n_fins=n_fins_eff,
            root_chord=fp0["root_chord"],
            tip_chord=fp0["tip_chord"],
            span=fp0["span"],
            sweep_LE_deg=fp0["sweep_LE_deg"],
            dihedral_deg=45.0,             # legacy field; ignored by GUI
            t_c=fp0["t_c"],
            max_thickness_loc=fp0["max_thickness_loc"],
            LE_style=fp0["LE_style"],
            LE_radius_mm=fp0["LE_radius_mm"],
            attach_x_frac=0.5,             # legacy field; ignored by GUI
        )
        return GVWDRunConfig(
            geometry=self._current_geom_config(),
            fins=fins, sweep=sweep,
        )

    # ------------------------------------------------------------------
    #  Generate geometry
    # ------------------------------------------------------------------

    def generate_geometry(self):
        try:
            cfg = self._build_run_config()
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                geom = build_geometry(cfg)
            self.geom = geom
            self.body_mesh = geom.mesh
            self.fins_mesh = None
            # Build per-fin meshes if enabled and engineering mode.
            # Each fin panel carries its own geometry + position +
            # rotation; we merge them all into a single fins mesh.
            if (self.fins_enable.isChecked()
                    and self.fins_group.isEnabled()
                    and self.fin_panels
                    and isinstance(cfg.geometry,
                                     (EngineeringFlatConfig,
                                      EngineeringShallowVConfig))):
                per_fin_meshes = []
                for i, panel in enumerate(self.fin_panels):
                    s = panel.to_spec()
                    fin_mesh = _generate_single_fin(
                        root_chord=s["root_chord"],
                        tip_chord=s["tip_chord"],
                        span=s["span"],
                        sweep_LE_rad=math.radians(s["sweep_LE_deg"]),
                        t_c=s["t_c"],
                        max_thickness_loc=s["max_thickness_loc"],
                        LE_style=s["LE_style"],
                        LE_radius_m=s["LE_radius_mm"] * 1e-3,
                        position=(s["pos_x"], s["pos_y"], s["pos_z"]),
                        rotation_deg=(s["roll_deg"], s["pitch_deg"], s["yaw_deg"]),
                        fin_idx=i + 1,
                    )
                    if fin_mesh is not None:
                        per_fin_meshes.append(fin_mesh)
                if per_fin_meshes:
                    self.fins_mesh = merge_meshes(per_fin_meshes)
            if self.fins_mesh is not None:
                self.full_mesh = merge_meshes([self.body_mesh, self.fins_mesh])
            else:
                self.full_mesh = self.body_mesh

            self._refresh_canvas3d()
            dt = time.perf_counter() - t0
            self._update_output_label(geom_dt=dt)
        except (ShockDetachedError, ValueError) as e:
            self.output_label.setText(f"Geometry build failed: {e}")
            self.output_label.setStyleSheet(
                "color: #ef4444; font-size: 10px; font-family: monospace; "
                "background-color: #1A1A1A; padding: 6px; border-radius: 3px;")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.output_label.setText(f"Generate error: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    #  On-design aero
    # ------------------------------------------------------------------

    def run_on_design(self):
        if self.full_mesh is None:
            QMessageBox.warning(self, "No geometry",
                                  "Click 'Generate Geometry' first.")
            return
        cfg = self._build_run_config()
        if cfg.geometry.mode == "multi_wedge":
            QMessageBox.information(self, "Mode",
                                      "On-design aero not defined for the multi-wedge mode "
                                      "(no natural alpha=0 reference). Use the sweep instead.")
            return
        M_design = cfg.geometry.M_design
        try:
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.on_design = aero_coefficients_full(
                    self.full_mesh,
                    M_inf=float(M_design), alpha_rad=0.0,
                    altitude_km=self.alt_km.value(),
                    T_w=self.T_w.value(),
                    Re_x_tr=self.Re_x_tr.value(),
                )
            dt = time.perf_counter() - t0
            self._update_output_label(on_design_dt=dt)
            # Refresh the canvas info panel so the new CL/CD/L/D show up
            self._refresh_canvas3d()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.output_label.setText(f"On-design error: {e}")

    # ------------------------------------------------------------------
    #  Sweep (threaded)
    # ------------------------------------------------------------------

    def run_sweep(self):
        if self.full_mesh is None:
            QMessageBox.warning(self, "No geometry",
                                  "Click 'Generate Geometry' first.")
            return
        if self._sweep_worker is not None and self._sweep_worker.isRunning():
            return
        sw_cfg = SweepConfig(
            M_grid=(self.M_lo.value(), self.M_hi.value(), self.n_M.value()),
            alpha_grid_deg=(self.alpha_lo.value(), self.alpha_hi.value(),
                              self.n_alpha.value()),
            altitude_km=self.alt_km.value(),
            T_w=self.T_w.value(),
            Re_x_tr=self.Re_x_tr.value(),
        )
        self.btn_sweep.setEnabled(False)
        self.sweep_progress.setRange(0, sw_cfg.M_grid[2] * sw_cfg.alpha_grid_deg[2])
        self.sweep_progress.setValue(0)
        self.output_label.setText(
            f"Sweep running: {sw_cfg.M_grid[2]}x{sw_cfg.alpha_grid_deg[2]}={sw_cfg.M_grid[2]*sw_cfg.alpha_grid_deg[2]} cells...")

        self._sweep_worker = _SweepWorker(self.full_mesh, sw_cfg, self)
        self._sweep_worker.progress.connect(self._on_sweep_progress)
        self._sweep_worker.finished_ok.connect(self._on_sweep_done)
        self._sweep_worker.finished_err.connect(self._on_sweep_err)
        self._sweep_worker.start()

    def _on_sweep_progress(self, done: int, total: int):
        self.sweep_progress.setValue(done)

    def _on_sweep_done(self, df):
        self.sweep_df = df
        self.btn_sweep.setEnabled(True)
        self._sweep_worker = None

        # Render plots into the existing _SimpleCanvas instances. Lazy
        # import to keep startup snappy.
        from gvwd.viz.plotting import (
            plot_LD_heatmap, plot_q_LE_heatmap, plot_polar_CL_vs_CD,
            plot_LD_vs_M_at_alpha, plot_shock_detachment_diagnostic,
            apply_style,
        )
        apply_style("paper")
        for canvas, plot_fn in [
            (self.canvas_LD_heat, plot_LD_heatmap),
            (self.canvas_qLE_heat, plot_q_LE_heatmap),
            (self.canvas_polar, plot_polar_CL_vs_CD),
            (self.canvas_LDvsM, plot_LD_vs_M_at_alpha),
            (self.canvas_shockdet, plot_shock_detachment_diagnostic),
        ]:
            canvas.fig.clear()
            new_ax = canvas.fig.add_subplot(111)
            try:
                plot_fn(df, ax=new_ax)
            except Exception as e:
                new_ax.text(0.5, 0.5, f"plot error: {e}",
                              ha="center", transform=new_ax.transAxes)
            canvas.draw()
        self._update_output_label(sweep_df=df)
        # Refresh the 3-D canvas info panel with the sweep summary
        self._refresh_canvas3d()

    def _on_sweep_err(self, msg: str):
        self.btn_sweep.setEnabled(True)
        self._sweep_worker = None
        self.output_label.setText(f"Sweep failed: {msg}")
        self.output_label.setStyleSheet(
            "color: #ef4444; font-size: 10px; font-family: monospace; "
            "background-color: #1A1A1A; padding: 6px; border-radius: 3px;")

    # ------------------------------------------------------------------
    #  3-D canvas refresh + info panel
    # ------------------------------------------------------------------

    def _refresh_canvas3d(self) -> None:
        """Re-render the 3-D canvas with the current mesh, info panel,
        and a parameter-rich title. Called after Generate, On-Design,
        and Sweep completion so the info panel always reflects the
        latest computed state."""
        if self.full_mesh is None:
            return
        info = self._build_info_dict()
        title_extra = self._build_title_suffix()
        title_prefix = f"GVWD: {self.current_mode}"
        self.canvas_3d.plot_mesh(
            self.full_mesh,
            info=info,
            title_prefix=title_prefix,
            title_extra=title_extra,
        )

    def _build_info_dict(self) -> dict:
        """Compose the top-left vehicle-characteristics dict shown on
        the 3-D canvas. Keys are rendered as left-padded labels."""
        info: dict = {}
        # Mode header
        info["__title__"] = f"GVWD ({self.current_mode})"

        cfg = self._build_run_config()
        g = cfg.geometry
        # Geometry params
        if hasattr(g, "M_design"):
            info["M_design"] = f"{g.M_design:.2f}"
        if hasattr(g, "theta_fore_deg"):
            info["theta_fore"] = f"{g.theta_fore_deg:.2f} deg"
        if hasattr(g, "theta_d_deg"):
            info["theta_d"] = f"{g.theta_d_deg:.2f} deg"
        if hasattr(g, "Lambda_deg"):
            info["Lambda"] = f"{g.Lambda_deg:.2f} deg"
        if hasattr(g, "L"):
            info["Length"] = f"{g.L:.4f} m"
        if hasattr(g, "L_fore"):
            info["L_fore"] = f"{g.L_fore:.4f} m"
        if hasattr(g, "L_center"):
            info["L_center"] = f"{g.L_center:.4f} m"
        if hasattr(g, "b_base"):
            info["b_base"] = f"{g.b_base:.4f} m"
        if hasattr(g, "h_base"):
            info["h_base"] = f"{g.h_base:.4f} m"
        if hasattr(g, "dihedral_lower_deg"):
            info["dihedral"] = f"{g.dihedral_lower_deg:.2f} deg"
        if hasattr(g, "n") and g.__class__.__name__ == "MultiWedgeConfig":
            info["n_ramps"] = f"{g.n}"
            info["delta_total"] = f"{g.delta_total_deg:.2f} deg"
        # Mesh-derived volumetrics
        try:
            V = numerical_volume(self.full_mesh)
            S = planform_area_from_mesh(self.full_mesh)
            ev = eta_V(V, S)
            info["Volume"] = f"{V:.4f} m^3"
            info["Planform"] = f"{S:.4f} m^2"
            info["eta_V"] = f"{ev:.4f}"
        except Exception:
            pass
        # On-design aero (if available)
        if self.on_design is not None:
            d = self.on_design
            info["CL"] = f"{d['CL']:+.4f}"
            info["CD_total"] = f"{d['CD_total']:+.4f}"
            info["CD_wave"] = f"{d['CD_wave']:+.4f}"
            info["CD_friction"] = f"{d['CD_friction']:+.4f}"
            info["L/D"] = f"{d['LD']:.3f}"
            info["Cm"] = f"{d['Cm']:+.4f}"
        # Sweep summary (if available)
        if self.sweep_df is not None and len(self.sweep_df) > 0:
            df = self.sweep_df
            info["LD_max"] = f"{df['LD'].max():.3f}"
            info["q_LE_max"] = f"{df['q_LE_swept_MW_m2'].max():.1f} MW/m^2"
        return info

    def _build_title_suffix(self) -> str:
        """Compact one-liner to follow the mode kind in the canvas
        title — mirrors the PSWR-1 'M=6.00, beta=12-16, Lambda=70'
        format."""
        cfg = self._build_run_config()
        g = cfg.geometry
        bits = []
        if hasattr(g, "M_design"):
            bits.append(f"M={g.M_design:.2f}")
        if hasattr(g, "theta_fore_deg"):
            bits.append(f"theta_fore={g.theta_fore_deg:.1f} deg")
        if hasattr(g, "theta_d_deg"):
            bits.append(f"theta_d={g.theta_d_deg:.1f} deg")
        if hasattr(g, "Lambda_deg"):
            bits.append(f"Lambda={g.Lambda_deg:.0f} deg")
        if self.full_mesh is not None:
            bits.append(f"{self.full_mesh.n_faces} faces")
        return f"({', '.join(bits)})" if bits else ""

    # ------------------------------------------------------------------
    #  Output label
    # ------------------------------------------------------------------

    def _update_output_label(self, *, geom_dt: float = None,
                                on_design_dt: float = None,
                                sweep_df=None) -> None:
        lines = []
        if self.full_mesh is not None:
            try:
                V = numerical_volume(self.full_mesh)
                S = planform_area_from_mesh(self.full_mesh)
                ev = eta_V(V, S)
                lines.append(f"GEOMETRY  ({self.full_mesh.n_faces} faces)")
                lines.append(f"  V       = {V:.4f} m^3")
                lines.append(f"  S_plan  = {S:.4f} m^2")
                lines.append(f"  eta_V   = {ev:.4f}")
            except Exception:
                pass
            if geom_dt is not None:
                lines.append(f"  build dt = {geom_dt*1000:.1f} ms")

        if self.on_design is not None:
            d = self.on_design
            lines.append("")
            lines.append(f"ON-DESIGN (M={d['M_inf']}, alpha=0)")
            lines.append(f"  CL          = {d['CL']:+.5f}")
            lines.append(f"  CD_total    = {d['CD_total']:+.5f}")
            lines.append(f"  CD_wave     = {d['CD_wave']:+.5f}")
            lines.append(f"  CD_friction = {d['CD_friction']:+.5f}")
            lines.append(f"  Cm          = {d['Cm']:+.5f}")
            lines.append(f"  L/D         = {d['LD']:.3f}")
            if on_design_dt is not None:
                lines.append(f"  eval dt     = {on_design_dt*1000:.1f} ms")

        if sweep_df is not None and len(sweep_df) > 0:
            lines.append("")
            lines.append(f"SWEEP ({len(sweep_df)} cells)")
            lines.append(f"  L/D range:   {sweep_df['LD'].min():.3f} .. "
                          f"{sweep_df['LD'].max():.3f}")
            lines.append(f"  CL range:    {sweep_df['CL'].min():+.4f} .. "
                          f"{sweep_df['CL'].max():+.4f}")
            lines.append(f"  q_LE max:    {sweep_df['q_LE_swept_MW_m2'].max():.1f} MW/m^2")
            lines.append(f"  margin range: "
                          f"{sweep_df['beta_attached_margin_deg'].min():+.2f} .. "
                          f"{sweep_df['beta_attached_margin_deg'].max():+.2f} deg")

        self.output_label.setText("\n".join(lines) or "No data yet.")
        self.output_label.setStyleSheet(
            "color: #aaaaaa; font-size: 10px; font-family: monospace; "
            "background-color: #1A1A1A; padding: 6px; border-radius: 3px;")

    # ------------------------------------------------------------------
    #  Export
    # ------------------------------------------------------------------

    def export_stl_dialog(self):
        if self.full_mesh is None:
            QMessageBox.warning(self, "No geometry", "Generate first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export STL", "gvwd_geometry.stl", "STL Files (*.stl)")
        if not path: return
        try:
            write_stl(self.full_mesh, path,
                        header=f"gvwd-export {self.current_mode}")
            QMessageBox.information(self, "Saved",
                                      f"STL written:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"{type(e).__name__}: {e}")

    def export_step_dialog(self):
        if self.full_mesh is None:
            QMessageBox.warning(self, "No geometry", "Generate first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export STEP", "gvwd_geometry.step",
            "STEP Files (*.step *.stp)")
        if not path: return
        try:
            write_step(self.full_mesh, path)
            QMessageBox.information(self, "Saved",
                                      f"STEP written:\n{path}")
        except CadqueryUnavailableError as e:
            QMessageBox.critical(self, "STEP export unavailable",
                                  f"cadquery is required for STEP export.\n{e}")
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Export failed", f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    #  Help dialog
    # ------------------------------------------------------------------

    _HELP_TEXT = (
        "GVWD: Glide-Vehicle Wedge-Derived geometry hub\n"
        "===============================================\n\n"
        "MODES\n"
        "  Engineering flat-bottom : HTV-2 / Avangard archetype.\n"
        "      Triangular flat-bottom forebody + frustum centerbody +\n"
        "      rectangular base + optional aft fins. Headline mode.\n"
        "  Engineering shallow-V   : same as flat-bottom plus a small\n"
        "      lower-surface dihedral angle (V-trough) for lateral\n"
        "      stability. dihedral_lower=0 reduces to the flat-bottom.\n"
        "  Caret                    : Nonweiler textbook reference.\n"
        "  Flat-bottomed delta      : single-inclined-plane reference.\n"
        "  Multi-wedge / Oswatitsch : n equal-strength oblique-shock\n"
        "      ramps. Inlet-design utility.\n\n"
        "ACTIONS\n"
        "  Generate Geometry  : build the 3-D mesh, render in the canvas\n"
        "  Run On-Design Aero : panel-method CL, CD, L/D at M=M_design,\n"
        "      alpha=0. Quick — runs in the GUI thread.\n"
        "  Run Mach-alpha Sweep : full grid sweep over (M, alpha). Runs\n"
        "      in a background QThread; progress bar updates per cell.\n"
        "  Export STL/STEP : write the current geometry mesh to disk.\n\n"
        "READOUTS\n"
        "  V, S_planform, eta_V : numerical volume + planform + ratio.\n"
        "  CL, CD_wave, CD_friction, CD_total, L/D : aero breakdown.\n"
        "  Cm                 : pitching moment about mid-chord.\n"
        "  q_LE max           : peak LE stagnation heat flux\n"
        "                       (Tauber-Sutton 1991, swept).\n"
        "  beta_attached_margin : theta_max - max(theta_local).\n"
        "      Negative = at least one panel exceeded attached-shock\n"
        "      regime and the panel method fell back to Newtonian.\n\n"
        "PHYSICS LIMITS\n"
        "  - Perfect gas, gamma=1.4. Real-gas equilibrium air would\n"
        "    lower T_post via dissociation; not modelled here.\n"
        "  - Tauber-Sutton 1991 heating (V^3.15 form). For sharp 1 mm LE\n"
        "    at M=15, h=30km, swept Lambda=75 deg: ~130 MW/m^2.\n"
        "  - Tangent-wedge with Newtonian fallback at theta>theta_max.\n"
        "    Discontinuity at the boundary is intrinsic to the model.\n"
    )

    def show_help_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("GVWD — Help")
        dlg.resize(700, 580)
        layout = QVBoxLayout(dlg)
        text = QTextEdit(); text.setReadOnly(True)
        text.setFont(QFont("Consolas", 9))
        text.setPlainText(self._HELP_TEXT)
        layout.addWidget(text)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        layout.addWidget(bb)
        dlg.exec_()


# ======================================================================
#  Standalone smoke test
# ======================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tab = GVWDWaveriderTab()
    tab.setWindowTitle("GVWD Waverider Tab (standalone)")
    tab.resize(1280, 820)
    tab.show()
    sys.exit(app.exec_())
