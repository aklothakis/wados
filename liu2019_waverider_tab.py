"""GUI tab for the Liu et al. 2019 variable-Mach osculating flowfield waverider.

Reproduces the paper's test case (Ma_center = 6, Ma_tip = 13, beta = 13 deg,
L_w = 6 m, W = 3 m) and validates against Table 4 and Figure 12 targets.
"""

from __future__ import annotations

import os
import traceback
from typing import Optional

import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAction, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMenu,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
    QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from liu2019.config import (
    PAPER_PARAMS, PAPER_REFERENCE_AERO, PAPER_REFERENCE_GEOMETRY, TOLERANCES,
)
from liu2019.distributions import (
    Ma_distribution, shock_curve, upper_surface_trailing_edge,
)
from liu2019.geometry import Liu2019Waverider, build_liu2019_waverider
from liu2019.osculating import build_all_osculating_planes
from liu2019.shock import beta_detachment, mach_angle
from liu2019.aero import Liu2019AeroEvaluator


# --- Theme constants (consistent with VMPLO / OC tabs) -----------------------
BG_DARK   = "#1A1A1A"
ACCENT    = "#D97706"
BTN_GREEN = "#2B5B2B"
BTN_HOVER = "#3B7B3B"
TEXT_MUTED = "#aaaaaa"


# ============================================================================
#  Workers
# ============================================================================

class _GeometryWorker(QThread):
    finished_ok = pyqtSignal(object)   # Liu2019Waverider
    failed      = pyqtSignal(str)

    def __init__(self, params: dict, n_z: int, n_x: int):
        super().__init__()
        self.params, self.n_z, self.n_x = params, n_z, n_x

    def run(self):
        try:
            wr = build_liu2019_waverider(self.params, n_z=self.n_z, n_x=self.n_x)
            self.finished_ok.emit(wr)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class _AeroWorker(QThread):
    finished_ok = pyqtSignal(object)   # list of dicts
    failed      = pyqtSignal(str)
    progress    = pyqtSignal(int)

    def __init__(self, waverider: Liu2019Waverider):
        super().__init__()
        self.waverider = waverider

    def run(self):
        try:
            evaluator = Liu2019AeroEvaluator(self.waverider)
            rows = evaluator.evaluate_paper_trajectory(
                progress_callback=self.progress.emit)
            self.finished_ok.emit(rows)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# ============================================================================
#  Canvas classes
# ============================================================================

def _style_dark_axes(ax, three_d=False):
    ax.set_facecolor(BG_DARK)
    for spine in ax.spines.values():
        spine.set_color("#666666")
    ax.tick_params(colors="#CCCCCC")
    ax.xaxis.label.set_color("#CCCCCC")
    ax.yaxis.label.set_color("#CCCCCC")
    if three_d and hasattr(ax, "zaxis"):
        ax.zaxis.label.set_color("#CCCCCC")
    if hasattr(ax, "title"):
        ax.title.set_color("#FFFFFF")


class _DarkFigureCanvas(FigureCanvas):
    def __init__(self, three_d=False, figsize=(8, 5)):
        self.fig = Figure(figsize=figsize)
        self.fig.patch.set_facecolor(BG_DARK)
        if three_d:
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig.add_subplot(111)
        _style_dark_axes(self.ax, three_d=three_d)
        super().__init__(self.fig)


class Canvas3D(_DarkFigureCanvas):
    def __init__(self):
        super().__init__(three_d=True, figsize=(9, 7))
        self._info_text = None
        # Title prefix used in plot(); subclasses / sibling tabs (MFOF)
        # can override this to relabel the visualization without
        # touching the rest of the plot code.
        self.title_prefix = "Liu 2019 Waverider"
        # Header line of the info panel (top-left orange-bordered box).
        # Override in subclasses to relabel the panel header.
        self.info_panel_header = "LIU 2019 WAVERIDER"

    def plot(self, wr: Optional[Liu2019Waverider],
             show_upper=True, show_lower=True, show_le=True, show_info=True):
        ax = self.ax
        ax.clear()
        _style_dark_axes(ax, three_d=True)
        if self._info_text is not None:
            try:
                self._info_text.remove()
            except Exception:
                pass
            self._info_text = None

        if wr is None:
            ax.set_title("No geometry generated", color="white")
            self.draw()
            return

        X_u, Y_u, Z_u = wr.upper_surface(mirror=True)
        X_l, Y_l, Z_l = wr.lower_surface(mirror=True)
        le_x, le_y, le_z = wr.leading_edge_curve(mirror=True)

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = []

        stride = max(1, X_u.shape[1] // 40)
        if show_upper:
            ax.plot_surface(Z_u, X_u, Y_u, color="steelblue", alpha=0.35,
                            linewidth=0, rstride=stride, cstride=stride)
            legend_elements.append(Patch(facecolor="steelblue", alpha=0.4,
                                         label="Upper surface"))
        if show_lower:
            ax.plot_surface(Z_l, X_l, Y_l, color="indianred", alpha=0.75,
                            linewidth=0, rstride=stride, cstride=stride)
            legend_elements.append(Patch(facecolor="indianred", alpha=0.7,
                                         label="Lower surface"))
        if show_le:
            ax.plot(le_z, le_x, le_y, color="white", linewidth=2.0)
            legend_elements.append(Line2D([0], [0], color="white",
                                          linewidth=2.0, label="Leading edge"))

        ax.set_xlabel("Z  (span)")
        ax.set_ylabel("X  (streamwise)")
        ax.set_zlabel("Y  (vertical)")
        ax.set_title(
            f"{self.title_prefix}  (Ma {wr.params['Ma_center']:.0f}-"
            f"{wr.params['Ma_tip']:.0f},  β {wr.params['beta_deg']:.1f}°)",
            color="white")
        self._set_axes_equal(ax, Z_u, X_u, Y_u)

        if legend_elements:
            leg = ax.legend(handles=legend_elements, loc="upper right",
                            facecolor=BG_DARK, edgecolor="#555555",
                            labelcolor="white", fontsize=9)

        if show_info:
            self._draw_info_panel(wr)

        self.fig.tight_layout()
        self.draw()

    # ------------------------------------------------------------------
    def _draw_info_panel(self, wr: Liu2019Waverider):
        """Orange-bordered info box in the top-left of the 3D view,
        following the SHADOW tab's style."""
        p = wr.params
        Ma_range = (
            f"{min(p['Ma_center'], p['Ma_tip']):.1f}"
            f"-{max(p['Ma_center'], p['Ma_tip']):.1f}"
        )
        zs, dc = wr.cone_angle_array()
        V  = wr.volume()
        Sw = wr.wetted_area()
        Sp = wr.planform_area()
        Sb = wr.base_area()
        eta = wr.volumetric_efficiency()

        lines = [
            self.info_panel_header,
            f"  Method          Var-Mach OC (Rodi)",
            f"  Ma range        {Ma_range}",
            f"  Shock β         {p['beta_deg']:.2f}°",
            f"  Cone θc         {dc.min():.2f}-{dc.max():.2f}°",
            f"  Length L_w      {p['L_w']:.3f} m",
            f"  Span W          {p['W']:.3f} m",
            f"  Flat L_s        {p['L_s']:.3f} m",
            f"  Volume          {V:.4f} m³",
            f"  S_wet           {Sw:.4f} m²",
            f"  Planform S_p    {Sp:.4f} m²",
            f"  Base S_b        {Sb:.4f} m²",
            f"  Vol Efficiency  {eta:.4f}",
        ]
        info = "\n".join(lines)
        self._info_text = self.fig.text(
            0.02, 0.98, info, transform=self.fig.transFigure,
            fontsize=8, fontfamily="monospace", verticalalignment="top",
            color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1A1A1A",
                      edgecolor=ACCENT, alpha=0.85),
        )

    @staticmethod
    def _set_axes_equal(ax, X, Y, Z):
        lims = np.array([[X.min(), X.max()],
                         [Y.min(), Y.max()],
                         [Z.min(), Z.max()]])
        centre = lims.mean(axis=1)
        radius = 0.5 * np.max(lims[:, 1] - lims[:, 0])
        ax.set_xlim(centre[0] - radius, centre[0] + radius)
        ax.set_ylim(centre[1] - radius, centre[1] + radius)
        ax.set_zlim(centre[2] - radius, centre[2] + radius)


class BasePlaneCanvas(_DarkFigureCanvas):
    def plot(self, wr: Optional[Liu2019Waverider], params: dict):
        ax = self.ax
        ax.clear()
        _style_dark_axes(ax)
        if wr is None:
            ax.set_title("Base-plane view — generate first")
            self.draw()
            return
        coeffs = wr.coeffs
        W = float(params["W"])
        L_s = float(params["L_s"])
        z = np.linspace(-W/2.0, W/2.0, 400)
        # Liu's cubic y(z) = a*z^3 + b*z^2 + c*z + d describes the
        # freestream-surface trailing edge ONLY for z >= 0 (paper Eqs. 5-8
        # are derived from the half-span); the negative-z half is by
        # mirror symmetry. Evaluating the cubic at negative z directly
        # gives spurious values (the a*z^3 term flips sign), which
        # historically produced an asymmetric tail in this plot. The
        # actual 3D geometry is unaffected: mfof.geometry / liu2019.geometry
        # build only the half-span and mirror it in _mirrored().
        y_upper = upper_surface_trailing_edge(
            np.abs(z), coeffs["a"], coeffs["b"], coeffs["c"], coeffs["d"])
        y_shock = shock_curve(z, coeffs["A"], L_s)

        ax.plot(z, y_upper, color="white",
                linewidth=1.8, label="Freestream TE (y(z))")
        ax.plot(z, y_shock, color=ACCENT, linestyle="--", linewidth=1.6,
                label="Shock curve y_s(z)")

        zs_half = np.array([p.z for p in wr.planes])
        y_TE_l_half = np.array([p.P_TE[1] for p in wr.planes])
        zs_full = np.concatenate([-zs_half[::-1][:-1], zs_half])
        y_TE_l = np.concatenate([y_TE_l_half[::-1][:-1], y_TE_l_half])
        ax.plot(zs_full, y_TE_l, color="indianred", linewidth=1.8,
                label="Compression TE")

        ax.set_xlabel("z  (m)")
        ax.set_ylabel("y  (m)")
        ax.set_title("Base plane (x = L_w)", color="white")
        ax.grid(True, alpha=0.2, color="#555555")
        leg = ax.legend(facecolor=BG_DARK, edgecolor="#555555", labelcolor="white")
        ax.set_aspect("equal", adjustable="datalim")
        self.fig.tight_layout()
        self.draw()


class MachZCanvas(_DarkFigureCanvas):
    def plot(self, params: dict):
        ax = self.ax
        ax.clear()
        _style_dark_axes(ax)
        W = float(params["W"])
        Ma_c = float(params["Ma_center"])
        Ma_t = float(params["Ma_tip"])
        z = np.linspace(-W/2.0, W/2.0, 400)
        Ma = Ma_distribution(z, W, Ma_c, Ma_t)
        ax.plot(z, Ma, color=ACCENT, linewidth=2.0)
        ax.fill_between(z, Ma, alpha=0.15, color=ACCENT)
        for m in (6, 8, 10, 13):
            ax.axhline(m, color="#666666", linewidth=0.6, linestyle="--")
            ax.text(W/2.0, m, f" Ma={m}", va="center", color=TEXT_MUTED,
                    fontsize=8)
        ax.set_xlabel("z  (m)")
        ax.set_ylabel("design Ma")
        ax.set_title("Spanwise Mach distribution (paper Eq. 1)", color="white")
        ax.grid(True, alpha=0.2, color="#555555")
        self.fig.tight_layout()
        self.draw()


class ConeAngleCanvas(_DarkFigureCanvas):
    def plot(self, wr: Optional[Liu2019Waverider]):
        ax = self.ax
        ax.clear()
        _style_dark_axes(ax)
        if wr is None:
            ax.set_title("Cone angle δ_c(z) — generate first")
            self.draw()
            return
        zs = np.array([p.z for p in wr.planes])
        dc = np.array([p.delta_c for p in wr.planes])
        ax.plot(zs, dc, color="#7EC8E3", linewidth=1.8)
        ax.fill_between(zs, dc, alpha=0.15, color="#7EC8E3")
        ax.set_xlabel("z  (m, half-span)")
        ax.set_ylabel("δ_c  (deg)")
        ax.set_title("Cone half-angle per osculating plane", color="white")
        ax.grid(True, alpha=0.2, color="#555555")
        self.fig.tight_layout()
        self.draw()


class AeroCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(9, 6))
        self.fig.patch.set_facecolor(BG_DARK)
        self.axes = {
            "CL":  self.fig.add_subplot(2, 3, 1),
            "CD":  self.fig.add_subplot(2, 3, 2),
            "L_D": self.fig.add_subplot(2, 3, 3),
            "Cmz": self.fig.add_subplot(2, 3, 4),
            "Xcp": self.fig.add_subplot(2, 3, 5),
        }
        for ax in self.axes.values():
            _style_dark_axes(ax)
        super().__init__(self.fig)
        self.plot_reference_only()

    def _reset(self):
        for ax in self.axes.values():
            ax.clear()
            _style_dark_axes(ax)

    def plot_reference_only(self):
        self._reset()
        self._draw_reference()
        self.fig.tight_layout()
        self.draw()

    def _draw_reference(self):
        ma_list = sorted(PAPER_REFERENCE_AERO.keys())
        for key, ax in self.axes.items():
            vals = [PAPER_REFERENCE_AERO[m][key] for m in ma_list]
            ax.plot(ma_list, vals, marker="o", linestyle="--",
                    color="#888888", markerfacecolor="none",
                    label="Paper Fig. 12")
            ax.set_title(key, color="white")
            ax.set_xlabel("Ma", color="#AAAAAA")
            ax.grid(True, alpha=0.2, color="#555555")

    def plot(self, aero_rows):
        self._reset()
        self._draw_reference()
        if not aero_rows:
            self.fig.tight_layout()
            self.draw()
            return
        ma = [r["Ma"] for r in aero_rows]
        for key, ax in self.axes.items():
            comp = [r[key] for r in aero_rows]
            ax.plot(ma, comp, marker="s", color=ACCENT, linewidth=1.8,
                    label="Computed")
            leg = ax.legend(facecolor=BG_DARK, edgecolor="#555555",
                            labelcolor="white", fontsize=8)
        self.fig.tight_layout()
        self.draw()


# ============================================================================
#  Metric card
# ============================================================================

class MetricCard(QFrame):
    def __init__(self, name: str, unit: str = ""):
        super().__init__()
        self.name = name
        self.unit = unit
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ background-color: #0D0D0D; border: 1px solid #333; "
            f"border-radius: 4px; }}"
        )
        self.setFixedHeight(74)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(0)

        self.name_label = QLabel(f"{name}{(' [' + unit + ']') if unit else ''}")
        self.name_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px;")

        self.value_label = QLabel("—")
        self.value_label.setStyleSheet(
            "color: white; font-size: 16px; font-weight: bold;")

        self.ref_label = QLabel("paper: —")
        self.ref_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px;")

        layout.addWidget(self.name_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.ref_label)

    def set(self, computed, reference, tolerance):
        if computed is None or reference is None:
            self.value_label.setText("—")
            self.ref_label.setText("paper: —")
            return
        self.value_label.setText(f"{computed:.4f}")
        dev = (computed - reference) / reference if reference else 0.0
        colour = "#6CBB6C" if abs(dev) <= tolerance else "#E06C6C"
        self.ref_label.setText(
            f"paper: {reference:.4f}  Δ {dev*100:+.2f}%")
        self.ref_label.setStyleSheet(f"color: {colour}; font-size: 10px;")


# ============================================================================
#  Main tab
# ============================================================================

class Liu2019WaveriderTab(QWidget):
    """Liu et al. 2019 variable-Mach osculating flowfield waverider GUI tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.waverider: Optional[Liu2019Waverider] = None
        self.aero_rows: list = []
        self._geom_worker: Optional[_GeometryWorker] = None
        self._aero_worker: Optional[_AeroWorker] = None
        self._init_ui()
        self._check_feasibility()
        self._update_Ma_plot()
        self._update_beta_hint()

    # --------------------------------------------------------------
    def _init_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([320, 1000])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

    # --- Left panel -----------------------------------------------
    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._group_geometry())
        layout.addWidget(self._group_mach())
        layout.addWidget(self._group_mesh())

        # Feasibility status
        self.feasibility_label = QLabel("Geometry valid")
        self.feasibility_label.setWordWrap(True)
        self.feasibility_label.setStyleSheet(
            "QLabel { background-color: #1f3f1f; color: #B0E0B0; "
            "border: 1px solid #2B5B2B; border-radius: 3px; "
            "padding: 6px; font-size: 10px; }"
        )
        layout.addWidget(self.feasibility_label)

        # Actions
        btn_row = QHBoxLayout()
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(self.reset_btn)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BTN_GREEN}; color: white; "
            f"font-weight: bold; padding: 8px; }}"
            f"QPushButton:hover {{ background-color: {BTN_HOVER}; }}"
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        self.generate_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self.generate_btn)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _group_geometry(self) -> QGroupBox:
        g = QGroupBox("Geometry (Table 1)")
        grid = QGridLayout(g)

        def spin(val, rng, step, dec=3):
            s = QDoubleSpinBox()
            s.setRange(*rng)
            s.setSingleStep(step)
            s.setDecimals(dec)
            s.setValue(val)
            s.valueChanged.connect(self._on_param_changed)
            return s

        row = 0
        self.beta_spin = spin(13.0, (5.0, 30.0), 0.1, 2)
        self.beta_spin.setToolTip(
            "<b>β — design shock angle [deg]</b><br><br>"
            "Single prescribed wave angle of the oblique / conical shock.<br><br>"
            "<b>Formulas:</b><br>"
            "&nbsp;&nbsp;Apex height (Eq. 5):<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>d = L<sub>w</sub>·tan β</i><br>"
            "&nbsp;&nbsp;θ–β–M (2D wedge deflection):<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>tan θ = 2 cot β · "
            "(M²sin²β − 1) / (M²(γ + cos 2β) + 2)</i><br>"
            "&nbsp;&nbsp;Cone half-angle: δ<sub>c</sub>(Ma, β) from the "
            "Taylor–Maccoll ODE (event Vθ = 0).<br><br>"
            "<b>Attachment window:</b> &nbsp; arcsin(1/M) < β < β<sub>det</sub>(M) "
            "for every M on the span. The hint below reports the live "
            "feasible range.<br><br>"
            "<b>Effect:</b> larger β → taller d → deeper compression wedge "
            "→ ↑ Vol, ↑ S<sub>wet</sub>, ↑ C<sub>L</sub>, ↑ C<sub>D</sub>."
        )
        grid.addWidget(QLabel("β  [deg]"), row, 0)
        grid.addWidget(self.beta_spin, row, 1); row += 1

        # Live hint below β — SHADOW-style attached Mach-angle / detachment info
        self.beta_hint_label = QLabel("")
        self.beta_hint_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 9px;")
        self.beta_hint_label.setWordWrap(True)
        grid.addWidget(self.beta_hint_label, row, 0, 1, 2); row += 1

        self.Lw_spin = spin(6.000, (1.0, 50.0), 0.1, 3)
        self.Lw_spin.setToolTip(
            "<b>L<sub>w</sub> — vehicle length [m]</b><br><br>"
            "Streamwise distance from nose (x = 0) to base plane "
            "(x = L<sub>w</sub>). Aerodynamic moment reference length.<br><br>"
            "<b>Formulas it appears in:</b><br>"
            "&nbsp;&nbsp;(Eq. 5): &nbsp;<i>d = L<sub>w</sub>·tan β</i><br>"
            "&nbsp;&nbsp;(LE intercept): &nbsp;<i>x<sub>LE</sub> "
            "= L<sub>w</sub> − (y(z) − y<sub>s</sub>(z)) / tan β</i><br>"
            "&nbsp;&nbsp;(TE projection): &nbsp;<i>y<sub>TE</sub> "
            "= y<sub>LE</sub> − (L<sub>w</sub> − x<sub>LE</sub>)·tan δ<sub>c</sub></i><br><br>"
            "<b>Effect:</b> linearly scales every streamwise dimension. "
            "Vol ∝ L<sub>w</sub>, S<sub>wet</sub> ∝ L<sub>w</sub>."
        )
        grid.addWidget(QLabel("L_w [m]"), row, 0)
        grid.addWidget(self.Lw_spin, row, 1); row += 1

        self.W_spin = spin(3.000, (0.5, 20.0), 0.1, 3)
        self.W_spin.setToolTip(
            "<b>W — full span [m]</b><br><br>"
            "Lateral extent of the vehicle, from −W/2 to +W/2. Fixes the "
            "tip location z<sub>5</sub> = W/2.<br><br>"
            "<b>Mach polynomial (Eq. 1):</b><br>"
            "&nbsp;&nbsp;<i>Ma(z) = m·z² + n</i><br>"
            "&nbsp;&nbsp;with <i>m = (Ma<sub>tip</sub> − Ma<sub>center</sub>) / (W/2)²</i>,<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i>n = Ma<sub>center</sub></i><br><br>"
            "<b>Effect:</b> wider W → flatter Ma(z) gradient, "
            "laterally stretched shock curve, ↑ planform & base area."
        )
        grid.addWidget(QLabel("W   [m]"), row, 0)
        grid.addWidget(self.W_spin, row, 1); row += 1

        self.Ls_spin = spin(0.300, (0.0, 10.0), 0.05, 3)
        self.Ls_spin.setToolTip(
            "<b>L<sub>s</sub> — flat-shock half-width [m]</b><br><br>"
            "Half-length of the central flat segment of the shock curve "
            "in the base plane.<br><br>"
            "<b>Shock curve (Eq. 3):</b><br>"
            "&nbsp;&nbsp;<i>y<sub>s</sub>(z) = 0</i> &nbsp; for |z| ≤ L<sub>s</sub><br>"
            "&nbsp;&nbsp;<i>y<sub>s</sub>(z) = A·(|z| − L<sub>s</sub>)<sup>4</sup></i> &nbsp; "
            "for |z| > L<sub>s</sub><br><br>"
            "<b>Osculating radius (Eq. 9):</b><br>"
            "&nbsp;&nbsp;<i>R(z) = (1 + y′<sub>s</sub>²)<sup>3/2</sup> / |y″<sub>s</sub>|</i><br>"
            "&nbsp;&nbsp;so <b>R → ∞</b> inside |z| ≤ L<sub>s</sub> (2D wedge flow) "
            "and <b>R finite</b> outside (true osculating cone).<br><br>"
            "<b>Constraint:</b> &nbsp; L<sub>s</sub> < W/2."
        )
        grid.addWidget(QLabel("L_s [m]"), row, 0)
        grid.addWidget(self.Ls_spin, row, 1); row += 1

        self.y5_spin = spin(0.1608, (0.01, 5.0), 0.01, 4)
        self.y5_spin.setToolTip(
            "<b>y<sub>5</sub> — tip trailing-edge height [m]</b><br><br>"
            "y-coordinate of the freestream-surface trailing edge at the "
            "tip z<sub>5</sub> = W/2.<br><br>"
            "<b>Enters the coefficient equations:</b><br>"
            "&nbsp;&nbsp;(Eq. 4): &nbsp;<i>A = y<sub>5</sub> / (z<sub>5</sub> − L<sub>s</sub>)<sup>4</sup></i><br>"
            "&nbsp;&nbsp;(Eq. 7, δ<sub>5</sub>=δ<sub>6</sub>=0):<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>b = 3·(y<sub>5</sub> − d) / z<sub>5</sub>²</i><br>"
            "&nbsp;&nbsp;(Eq. 8, δ<sub>5</sub>=δ<sub>6</sub>=0):<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<i>a = (y<sub>5</sub> − d − b·z<sub>5</sub>²) / z<sub>5</sub>³</i><br><br>"
            "<b>Effect:</b> smaller y<sub>5</sub> flattens the tip, lowers "
            "the shock-curve rise near the tips, and shrinks base area.<br><br>"
            "<b>Constraint:</b> &nbsp; y<sub>5</sub> < y<sub>6</sub>."
        )
        grid.addWidget(QLabel("y₅  [m]"), row, 0)
        grid.addWidget(self.y5_spin, row, 1); row += 1

        self.y6_spin = spin(1.608, (0.1, 10.0), 0.01, 4)
        self.y6_spin.setToolTip(
            "<b>y<sub>6</sub> — centreline apex / nose height [m]</b><br><br>"
            "Reference height of the upper surface at the centreline "
            "z = 0.<br><br>"
            "<b>Why it's a reference, not a free parameter:</b><br>"
            "At z = 0 the cubic y(z) = a z³ + b z² + c z + d evaluates to "
            "<i>y(0) = d = L<sub>w</sub>·tan β</i> regardless of y<sub>6</sub>. "
            "So y<sub>6</sub> does not enter Eqs. 5–8; it only bounds "
            "y<sub>5</sub> and is retained for paper cross-checking.<br><br>"
            "<b>Constraint:</b> &nbsp; y<sub>5</sub> < y<sub>6</sub>."
        )
        grid.addWidget(QLabel("y₆  [m]"), row, 0)
        grid.addWidget(self.y6_spin, row, 1); row += 1

        return g

    def _group_mach(self) -> QGroupBox:
        g = QGroupBox("Mach distribution (Eq. 1)")
        grid = QGridLayout(g)

        def spin(val):
            s = QDoubleSpinBox()
            s.setRange(3.0, 20.0)
            s.setSingleStep(0.5)
            s.setDecimals(2)
            s.setValue(val)
            s.valueChanged.connect(self._on_param_changed)
            return s

        self.Ma_c_spin = spin(6.0)
        self.Ma_c_spin.setToolTip(
            "<b>Ma<sub>center</sub> — centreline design Mach number</b><br><br>"
            "<b>Formula:</b><br>"
            "&nbsp;&nbsp;(Eq. 1 at z = 0): &nbsp;"
            "<i>Ma(0) = n = Ma<sub>center</sub></i><br>"
            "&nbsp;&nbsp;(Paper Fig. 1): &nbsp;"
            "<i>Ma(z) = m·z² + Ma<sub>center</sub></i><br><br>"
            "<b>Role:</b> the local cone half-angle is<br>"
            "&nbsp;&nbsp;<i>δ<sub>c</sub>(z) = Taylor-Maccoll(Ma(z), β)</i><br>"
            "so Ma<sub>center</sub> directly sets δ<sub>c</sub>(0), the "
            "compression-surface angle at the centreline.<br><br>"
            "<b>Effect:</b> ↑Ma<sub>center</sub> ⇒ ↑δ<sub>c</sub>(0) "
            "⇒ deeper central compression surface ⇒ ↑ Vol locally."
        )
        grid.addWidget(QLabel("Ma_center"), 0, 0)
        grid.addWidget(self.Ma_c_spin, 0, 1)

        self.Ma_t_spin = spin(13.0)
        self.Ma_t_spin.setToolTip(
            "<b>Ma<sub>tip</sub> — tip design Mach number</b><br><br>"
            "<b>Formula:</b><br>"
            "&nbsp;&nbsp;(Eq. 1 at z = ±W/2): &nbsp;"
            "<i>Ma(±W/2) = Ma<sub>tip</sub></i><br>"
            "&nbsp;&nbsp;fixes <i>m = (Ma<sub>tip</sub> − Ma<sub>center</sub>) / (W/2)²</i>.<br><br>"
            "<b>Role:</b> outboard (|z| → W/2) cone half-angle<br>"
            "&nbsp;&nbsp;<i>δ<sub>c</sub>(±W/2) = Taylor-Maccoll(Ma<sub>tip</sub>, β)</i>.<br><br>"
            "<b>Effect:</b> ↑Ma<sub>tip</sub> ⇒ ↑δ<sub>c</sub> at tips "
            "⇒ deeper outboard compression ⇒ ↑ Vol, ↑ S<sub>wet</sub>, "
            "but L/D grows only modestly because planform increases too."
        )
        grid.addWidget(QLabel("Ma_tip"), 1, 0)
        grid.addWidget(self.Ma_t_spin, 1, 1)
        return g

    def _group_mesh(self) -> QGroupBox:
        g = QGroupBox("Mesh resolution")
        grid = QGridLayout(g)

        self.n_z_spin = QSpinBox()
        self.n_z_spin.setRange(40, 500)
        self.n_z_spin.setValue(200)
        self.n_z_spin.setSingleStep(10)
        self.n_z_spin.setToolTip(
            "<b>n<sub>z</sub> — number of osculating planes</b><br><br>"
            "Sample count of the starboard half-span "
            "<i>z ∈ [0, W/2]</i>, uniformly spaced.<br><br>"
            "<b>Convergence:</b><br>"
            "&nbsp;&nbsp;Vol, S<sub>wet</sub> ~ <i>O(1/n<sub>z</sub>²)</i> "
            "(trapezoidal integral of the column cross-section)<br><br>"
            "Typical range 100–300; 200 is plenty for the paper case."
        )
        grid.addWidget(QLabel("n_z planes"), 0, 0)
        grid.addWidget(self.n_z_spin, 0, 1)

        self.n_x_spin = QSpinBox()
        self.n_x_spin.setRange(40, 400)
        self.n_x_spin.setValue(100)
        self.n_x_spin.setSingleStep(10)
        self.n_x_spin.setToolTip(
            "<b>n<sub>x</sub> — streamwise samples per osculating plane</b><br><br>"
            "Number of points along each cone-surface streamline from "
            "<i>x<sub>LE</sub>(z)</i> to the base plane <i>x = L<sub>w</sub></i>.<br><br>"
            "Sets the chordwise mesh resolution for surface area and "
            "STL/OBJ/STEP export. Also controls B-spline fit quality "
            "when exporting to STEP."
        )
        grid.addWidget(QLabel("n_x strips"), 1, 0)
        grid.addWidget(self.n_x_spin, 1, 1)
        return g

    # --- Right panel ----------------------------------------------
    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.sub_tabs = QTabWidget()

        # 3D view with view-toggle checkboxes above the canvas
        self.canvas_3d = Canvas3D()
        v3d = QWidget(); l3 = QVBoxLayout(v3d)
        l3.setContentsMargins(0, 0, 0, 0)

        check_bar = QHBoxLayout()
        self.chk_upper = QCheckBox("Upper"); self.chk_upper.setChecked(True)
        self.chk_lower = QCheckBox("Lower"); self.chk_lower.setChecked(True)
        self.chk_le    = QCheckBox("Leading Edge"); self.chk_le.setChecked(True)
        self.chk_info  = QCheckBox("Info"); self.chk_info.setChecked(True)
        for cb in (self.chk_upper, self.chk_lower, self.chk_le, self.chk_info):
            cb.stateChanged.connect(self._redraw_3d)
            check_bar.addWidget(cb)
        check_bar.addStretch()
        l3.addLayout(check_bar)

        l3.addWidget(NavigationToolbar(self.canvas_3d, self))
        l3.addWidget(self.canvas_3d)
        self.sub_tabs.addTab(v3d, "3D view")

        # Base plane
        self.canvas_base = BasePlaneCanvas(figsize=(9, 4))
        self.sub_tabs.addTab(self.canvas_base, "Base plane")

        # Ma(z)
        self.canvas_Ma = MachZCanvas(figsize=(9, 4))
        self.sub_tabs.addTab(self.canvas_Ma, "Ma(z)")

        # delta_c(z)
        self.canvas_dc = ConeAngleCanvas(figsize=(9, 4))
        self.sub_tabs.addTab(self.canvas_dc, "δ_c(z)")

        # Aerodynamics
        self.canvas_aero = AeroCanvas()
        self.sub_tabs.addTab(self.canvas_aero, "Aerodynamics")

        # Validation table
        self.validation_table = QTableWidget(0, 5)
        self.validation_table.setHorizontalHeaderLabels(
            ["Metric", "Computed", "Paper ref.", "Δ %", "Status"])
        self.validation_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.validation_table.verticalHeader().setVisible(False)
        self.sub_tabs.addTab(self.validation_table, "Validation")

        layout.addWidget(self.sub_tabs)

        # Action row (metric cards removed — the info panel inside the 3D
        # view now shows all vehicle statistics, matching the SHADOW tab).
        actions = QHBoxLayout()

        # Export menu button (STL / OBJ / STEP)
        self.export_btn = QToolButton()
        self.export_btn.setText("Export ▾")
        self.export_btn.setPopupMode(QToolButton.InstantPopup)
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip(
            "Export the wetted surface in one of:\n"
            "  • STL  — ASCII triangulated mesh (universal, for CFD/3D print)\n"
            "  • OBJ  — Wavefront quadrilateral mesh (viewers, Blender)\n"
            "  • STEP — CAD B-spline (for CAD editing / meshers), via OCP"
        )
        menu = QMenu(self.export_btn)
        act_stl  = QAction("Export STL…",  menu)
        act_obj  = QAction("Export OBJ…",  menu)
        act_step = QAction("Export STEP…", menu)
        act_stl.triggered.connect(lambda: self._on_export("stl"))
        act_obj.triggered.connect(lambda: self._on_export("obj"))
        act_step.triggered.connect(lambda: self._on_export("step"))
        menu.addAction(act_stl)
        menu.addAction(act_obj)
        menu.addAction(act_step)
        self.export_btn.setMenu(menu)
        actions.addWidget(self.export_btn)

        self.run_aero_btn = QPushButton("Run aero (Ma 6, 8, 10, 13)")
        self.run_aero_btn.clicked.connect(self._on_run_aero)
        self.run_aero_btn.setEnabled(False)
        self.run_aero_btn.setToolTip(
            "Evaluate C_L, C_D, L/D, C_mz, X_cp at the paper's four-point "
            "constant-q trajectory (Ma = 6, 8, 10, 13) via modified-Newtonian "
            "impact theory. Populates the Aerodynamics and Validation sub-tabs."
        )
        actions.addWidget(self.run_aero_btn)

        actions.addStretch()

        self.validation_btn = QPushButton("Validation: —")
        self.validation_btn.clicked.connect(
            lambda: self.sub_tabs.setCurrentWidget(self.validation_table))
        actions.addWidget(self.validation_btn)

        layout.addLayout(actions)
        return w

    # --------------------------------------------------------------
    #  Parameter handling
    # --------------------------------------------------------------
    def _read_params(self) -> dict:
        return {
            "beta_deg":  float(self.beta_spin.value()),
            "L_w":       float(self.Lw_spin.value()),
            "W":         float(self.W_spin.value()),
            "L_s":       float(self.Ls_spin.value()),
            "y5":        float(self.y5_spin.value()),
            "z5":        float(self.W_spin.value()) / 2.0,
            "y6":        float(self.y6_spin.value()),
            "z6":        0.0,
            "delta5":    0.0,
            "delta6":    0.0,
            "Ma_center": float(self.Ma_c_spin.value()),
            "Ma_tip":    float(self.Ma_t_spin.value()),
            "gamma":     1.4,
        }

    def _on_param_changed(self, *_):
        self._check_feasibility()
        self._update_Ma_plot()
        self._update_beta_hint()

    def _update_beta_hint(self):
        """Recompute the live 'β feasible range' hint under the β spinbox."""
        p = self._read_params()
        Ma_min = min(p["Ma_center"], p["Ma_tip"])
        Ma_max = max(p["Ma_center"], p["Ma_tip"])
        try:
            mu_lower_bound = mach_angle(Ma_min)       # largest μ sets tight lower
            beta_det_upper = beta_detachment(Ma_min)  # smallest β_det sets tight upper
            # A practical recommendation: ~1.4× the Mach angle at the lowest Ma,
            # clamped to [10°, 20°]. The paper uses 13° for Ma = 6–13 — this
            # heuristic returns 13.4° there.
            recommended = min(max(1.4 * mu_lower_bound, 10.0), 20.0)
            self.beta_hint_label.setText(
                f"β feasible: "
                f"{mu_lower_bound:.2f}° (μ @ Ma={Ma_min:.1f}) "
                f"< β < "
                f"{beta_det_upper:.2f}° (β_det @ Ma={Ma_min:.1f})  |  "
                f"Recommended ≈ {recommended:.1f}°  "
                f"(paper uses 13.0°)"
            )
        except Exception as e:
            self.beta_hint_label.setText(f"β range unavailable: {e}")

    def _check_feasibility(self) -> bool:
        p = self._read_params()
        msg, ok = "Geometry valid", True
        try:
            if p["L_s"] >= p["W"] / 2.0:
                ok, msg = False, "L_s must be < W/2"
            elif p["y5"] >= p["y6"]:
                ok, msg = False, "tip height y5 must be < nose height y6"
            else:
                for Ma in (p["Ma_center"], p["Ma_tip"]):
                    mu = mach_angle(Ma)
                    b_det = beta_detachment(Ma)
                    if p["beta_deg"] <= mu:
                        ok, msg = False, (
                            f"β = {p['beta_deg']:.1f}° is below Mach angle "
                            f"μ = {mu:.2f}° at Ma = {Ma:.1f}")
                        break
                    if p["beta_deg"] >= b_det:
                        ok, msg = False, (
                            f"β = {p['beta_deg']:.1f}° exceeds detachment "
                            f"β_det = {b_det:.2f}° at Ma = {Ma:.1f}")
                        break
                if ok:
                    msg = (f"β = {p['beta_deg']:.1f}° attached across "
                           f"Ma {p['Ma_center']:.0f}–{p['Ma_tip']:.0f}")
        except Exception as e:
            ok, msg = False, f"parameter error: {e}"

        if ok:
            self.feasibility_label.setStyleSheet(
                "QLabel { background-color: #1f3f1f; color: #B0E0B0; "
                "border: 1px solid #2B5B2B; border-radius: 3px; "
                "padding: 6px; font-size: 10px; }")
        else:
            self.feasibility_label.setStyleSheet(
                "QLabel { background-color: #3f1f1f; color: #E0B0B0; "
                "border: 1px solid #5B2B2B; border-radius: 3px; "
                "padding: 6px; font-size: 10px; }")
        self.feasibility_label.setText(msg)
        self.generate_btn.setEnabled(ok)
        return ok

    def _reset_defaults(self):
        self.beta_spin.setValue(PAPER_PARAMS["beta_deg"])
        self.Lw_spin.setValue(PAPER_PARAMS["L_w"])
        self.W_spin.setValue(PAPER_PARAMS["W"])
        self.Ls_spin.setValue(PAPER_PARAMS["L_s"])
        self.y5_spin.setValue(PAPER_PARAMS["y5"])
        self.y6_spin.setValue(PAPER_PARAMS["y6"])
        self.Ma_c_spin.setValue(PAPER_PARAMS["Ma_center"])
        self.Ma_t_spin.setValue(PAPER_PARAMS["Ma_tip"])
        self.n_z_spin.setValue(200)
        self.n_x_spin.setValue(100)

    def _update_Ma_plot(self):
        self.canvas_Ma.plot(self._read_params())

    # --------------------------------------------------------------
    #  Generate
    # --------------------------------------------------------------
    def _on_generate(self):
        if not self._check_feasibility():
            return
        params = self._read_params()
        n_z = self.n_z_spin.value()
        n_x = self.n_x_spin.value()

        self.generate_btn.setEnabled(False)
        self.run_aero_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)

        self._geom_worker = _GeometryWorker(params, n_z, n_x)
        self._geom_worker.finished_ok.connect(self._on_geometry_ready)
        self._geom_worker.failed.connect(self._on_worker_error)
        self._geom_worker.start()

    def _on_geometry_ready(self, wr: Liu2019Waverider):
        self.waverider = wr
        self.aero_rows = []
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.run_aero_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

        self._redraw_3d()
        self.canvas_base.plot(wr, self._read_params())
        self.canvas_dc.plot(wr)
        self.canvas_aero.plot_reference_only()
        self._update_validation_table(wr, aero_rows=[])

    def _redraw_3d(self):
        if self.waverider is None:
            return
        self.canvas_3d.plot(
            self.waverider,
            show_upper=self.chk_upper.isChecked(),
            show_lower=self.chk_lower.isChecked(),
            show_le=self.chk_le.isChecked(),
            show_info=self.chk_info.isChecked(),
        )

    def _on_worker_error(self, message: str):
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Liu 2019 waverider",
                             f"Worker failed:\n\n{message}")

    # --------------------------------------------------------------
    #  Aero
    # --------------------------------------------------------------
    def _on_run_aero(self):
        if self.waverider is None:
            return
        self.run_aero_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self._aero_worker = _AeroWorker(self.waverider)
        self._aero_worker.finished_ok.connect(self._on_aero_ready)
        self._aero_worker.failed.connect(self._on_worker_error)
        self._aero_worker.start()

    def _on_aero_ready(self, rows: list):
        self.aero_rows = rows
        self.progress_bar.setVisible(False)
        self.run_aero_btn.setEnabled(True)
        self.canvas_aero.plot(rows)
        if self.waverider is not None:
            self._update_validation_table(self.waverider, rows)
        self.sub_tabs.setCurrentWidget(self.canvas_aero)

    # --------------------------------------------------------------
    #  Export
    # --------------------------------------------------------------
    _EXPORT_META = {
        "stl":  ("STL",  "*.stl",  "liu2019_waverider.stl"),
        "obj":  ("OBJ",  "*.obj",  "liu2019_waverider.obj"),
        "step": ("STEP", "*.step", "liu2019_waverider.step"),
    }

    def _on_export(self, fmt: str):
        if self.waverider is None:
            return
        label, pattern, default_name = self._EXPORT_META[fmt]
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export Liu 2019 waverider as {label}",
            os.path.join(os.getcwd(), default_name),
            f"{label} files ({pattern})")
        if not path:
            return
        try:
            if fmt == "stl":
                self.waverider.export_stl(path)
            elif fmt == "obj":
                self.waverider.export_obj(path)
            elif fmt == "step":
                self.waverider.export_step(path)
            QMessageBox.information(
                self, f"Export {label}", f"{label} written:\n{path}")
        except RuntimeError as e:
            QMessageBox.warning(
                self, f"Export {label}",
                f"{label} export not available in this environment:\n\n{e}")
        except Exception as e:
            QMessageBox.critical(self, f"Export {label}", f"Failed:\n{e}")

    # --------------------------------------------------------------
    #  Validation
    # --------------------------------------------------------------
    def _update_validation_table(self, wr: Liu2019Waverider, aero_rows):
        rows = []
        geom = {
            "Vol_m3":   wr.volume(),
            "S_wet_m2": wr.wetted_area(),
            "S_p_m2":   wr.planform_area(),
            "S_b_m2":   wr.base_area(),
            "eta":      wr.volumetric_efficiency(),
        }
        for k, v in geom.items():
            ref = PAPER_REFERENCE_GEOMETRY[k]
            tol = TOLERANCES[k]
            dev = (v - ref) / ref if ref else 0.0
            rows.append((k, v, ref, dev, abs(dev) <= tol))

        for r in aero_rows:
            ref = PAPER_REFERENCE_AERO.get(int(r["Ma"]), {})
            for key in ("CL", "CD", "L_D", "Cmz", "Xcp"):
                ref_v = ref.get(key)
                tol = TOLERANCES[key]
                v = r[key]
                dev = ((v - ref_v) / ref_v) if ref_v else 0.0
                ok = abs(dev) <= tol if ref_v is not None else True
                rows.append((f"{key} @ Ma={int(r['Ma'])}", v, ref_v, dev, ok))

        t = self.validation_table
        t.setRowCount(len(rows))
        passed = 0
        for i, (name, v, ref, dev, ok) in enumerate(rows):
            t.setItem(i, 0, QTableWidgetItem(name))
            t.setItem(i, 1, QTableWidgetItem(f"{v:.4f}"))
            t.setItem(i, 2, QTableWidgetItem("—" if ref is None else f"{ref:.4f}"))
            t.setItem(i, 3, QTableWidgetItem("—" if ref is None else f"{dev*100:+.2f}%"))
            status_item = QTableWidgetItem("PASS" if ok else "FAIL")
            status_item.setForeground(QColor("#6CBB6C" if ok else "#E06C6C"))
            t.setItem(i, 4, status_item)
            passed += int(ok)

        total = len(rows)
        self.validation_btn.setText(f"Validation: {passed}/{total} pass")
        colour = "#2B5B2B" if passed == total else "#5B2B2B"
        self.validation_btn.setStyleSheet(
            f"QPushButton {{ background-color: {colour}; color: white; "
            "padding: 6px 10px; }}")
