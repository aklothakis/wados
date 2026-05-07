"""Variable-Mach Power-Law Osculating (VMPLO) Waverider Tab.

GUI tab for designing waveriders with spanwise-varying Mach number
and power-law generating body exponent.
"""

import math
import sys
import numpy as np

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QGridLayout,
                             QDoubleSpinBox, QSpinBox, QCheckBox, QFileDialog,
                             QMessageBox, QSplitter, QApplication, QScrollArea,
                             QTabWidget, QStackedWidget, QComboBox)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from waverider_generator.vmplo.bspline import BSpline1D
from waverider_generator.vmplo.osculating import OsculatingAssembly
from waverider_generator.vmplo.geometry import VMPLOWaverider
# (Legacy SpanwiseDistribution import removed — VMPLO now uses
# BSpline1D from waverider_generator.vmplo.bspline.)

try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False


# ======================================================================
#  Canvas classes
# ======================================================================

class VMPLOCanvas3D(FigureCanvas):
    """3D matplotlib canvas for stream-based VMPLO waverider visualisation."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        self._info_text = None

    def plot_waverider(self, wr, half_only=False, show_upper=True,
                       show_lower=True, show_le=True, show_info=True):
        self.ax.clear()
        if hasattr(self, '_info_text') and self._info_text is not None:
            try:
                self._info_text.remove()
            except Exception:
                pass
            self._info_text = None

        if wr is None:
            self.ax.set_title('No waverider generated')
            self.draw()
            return

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = []

        if show_upper and hasattr(wr, 'upper_surface_streams'):
            for s in wr.upper_surface_streams:
                if s.shape[0] < 2:
                    continue
                self.ax.plot(s[:, 2], s[:, 0], s[:, 1],
                             color='steelblue', alpha=0.6, linewidth=0.8)
                if not half_only:
                    self.ax.plot(-s[:, 2], s[:, 0], s[:, 1],
                                 color='steelblue', alpha=0.6, linewidth=0.8)
            legend_elements.append(Patch(facecolor='steelblue', alpha=0.4,
                                         label='Upper Surface'))

        if show_lower and hasattr(wr, 'lower_surface_streams'):
            for s in wr.lower_surface_streams:
                if s.shape[0] < 2:
                    continue
                self.ax.plot(s[:, 2], s[:, 0], s[:, 1],
                             color='indianred', alpha=0.6, linewidth=0.8)
                if not half_only:
                    self.ax.plot(-s[:, 2], s[:, 0], s[:, 1],
                                 color='indianred', alpha=0.6, linewidth=0.8)
            legend_elements.append(Patch(facecolor='indianred', alpha=0.4,
                                         label='Lower Surface'))

        if show_le and hasattr(wr, 'leading_edge'):
            le = wr.leading_edge
            self.ax.plot(le[:, 2], le[:, 0], le[:, 1], 'k-', linewidth=2.5)
            if not half_only:
                self.ax.plot(-le[:, 2], le[:, 0], le[:, 1], 'k-', linewidth=2.5)
            legend_elements.append(Line2D([0], [0], color='black',
                                           linewidth=2.5, label='Leading Edge'))

        beta = getattr(wr, 'beta_deg', 0)
        self.ax.set_xlabel('Z (Span)', color='#FFFFFF')
        self.ax.set_ylabel('X (Streamwise)', color='#FFFFFF')
        self.ax.set_zlabel('Y (Vertical)', color='#FFFFFF')
        self.ax.set_title(f'VMPLO Waverider (beta={beta:.1f} deg)', color='#FFFFFF')
        self.ax.tick_params(colors='#888888')
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper left')
        self._set_axes_equal()

        if show_info:
            self._draw_info_panel(wr)
        self.fig.tight_layout()
        self.draw()

    def _draw_info_panel(self, wr):
        lines = ['VMPLO WAVERIDER']
        beta = getattr(wr, 'beta_deg', 0)
        length = getattr(wr, 'length', 0)
        lines.append(f"  Beta           {beta:.1f} deg")
        lines.append(f"  Length          {length:.4f} m")

        # Mach range
        if hasattr(wr, '_mach_per_station'):
            ma = wr._mach_per_station
            lines.append(f"  Ma range       {ma.min():.1f} - {ma.max():.1f}")

        # Exponent range
        if hasattr(wr, '_n_per_station'):
            n = wr._n_per_station
            lines.append(f"  n range        {n.min():.2f} - {n.max():.2f}")

        # Cone angles
        if hasattr(wr, 'cone_angles_deg'):
            ca = wr.cone_angles_deg
            lines.append(f"  Cone angles    {ca.min():.1f} - {ca.max():.1f} deg")

        # Volume
        vol = 0.0
        try:
            vol = wr.compute_volume()
            lines.append(f"  Volume          {vol:.6f} m3")
        except Exception:
            pass

        # Planform area (full vehicle = 2 * half-span integration)
        a_plan = 0.0
        try:
            us = wr.upper_surface_streams
            chords = []
            z_pos = []
            for s in us:
                if s.shape[0] < 2:
                    continue
                chords.append(s[-1, 0] - s[0, 0])
                z_pos.append(s[0, 2])
            if len(chords) > 2:
                a_half = float(np.trapz(chords, z_pos))
                a_plan = 2.0 * a_half  # full vehicle (both halves)
                lines.append(f"  Planform Area   {a_plan:.4f} m2")
        except Exception:
            pass

        if a_plan > 1e-12 and vol > 0:
            eta_v = vol ** (2.0 / 3.0) / a_plan
            lines.append(f"  Vol. Efficiency {eta_v:.4f}")

        info = "\n".join(lines)
        self._info_text = self.fig.text(
            0.02, 0.98, info, transform=self.fig.transFigure,
            fontsize=8, fontfamily='monospace', verticalalignment='top',
            color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A1A1A',
                      edgecolor='#D97706', alpha=0.85))

    def _set_axes_equal(self):
        try:
            limits = np.array([self.ax.get_xlim3d(), self.ax.get_ylim3d(),
                               self.ax.get_zlim3d()])
            center = np.mean(limits, axis=1)
            radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
            self.ax.set_xlim3d([center[0] - radius, center[0] + radius])
            self.ax.set_ylim3d([center[1] - radius, center[1] + radius])
            self.ax.set_zlim3d([center[2] - radius, center[2] + radius])
        except Exception:
            pass


class DistributionProfileCanvas(FigureCanvas):
    """2D dual-axis plot showing Ma(z) and n(z) distributions."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 3))
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()
        super().__init__(self.fig)
        self.setParent(parent)
        self._plot_default()

    def _plot_default(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax2 = self.ax1.twinx()
        z = np.linspace(0, 1.0, 30)
        self._draw(z, np.linspace(6, 10, 30), np.ones(30))

    def update_profile(self, wr):
        self.ax1.clear()
        self.ax2.clear()
        self.ax2 = self.ax1.twinx()
        if wr is None:
            self._plot_default()
            return
        ma = getattr(wr, '_mach_per_station', None)
        n_exp = getattr(wr, '_n_per_station', None)
        if ma is None:
            self._plot_default()
            return
        z = np.linspace(0, 1.0, len(ma))
        n_vals = n_exp if n_exp is not None else np.ones(len(ma))
        self._draw(z, ma, n_vals)

    def _draw(self, z, ma, n_exp):
        self.ax1.plot(z, ma, 'b-', linewidth=2, label='Mach')
        self.ax1.fill_between(z, ma, alpha=0.1, color='blue')
        self.ax1.set_xlabel('Spanwise position (normalised)')
        self.ax1.set_ylabel('Mach number', color='blue')
        self.ax1.tick_params(axis='y', labelcolor='blue')

        self.ax2.plot(z, n_exp, 'r--', linewidth=2, label='Exponent n')
        self.ax2.fill_between(z, n_exp, alpha=0.1, color='red')
        self.ax2.set_ylabel('Power-law exponent n', color='red')
        self.ax2.tick_params(axis='y', labelcolor='red')

        self.ax1.set_title('Spanwise Distributions')
        self.ax1.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        self.fig.tight_layout()
        self.draw()


class ConeAngleProfileCanvas(FigureCanvas):
    """2D plot showing spanwise cone angle variation."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 3))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._plot_default()

    def _plot_default(self):
        self.ax.clear()
        z = np.linspace(0, 1.0, 30)
        delta = np.linspace(5, 10, 30)
        self._draw(z, delta)

    def update_profile(self, wr):
        self.ax.clear()
        if wr is None or not hasattr(wr, 'cone_angles_deg'):
            self._plot_default()
            return
        delta = wr.cone_angles_deg
        z = np.linspace(0, 1.0, len(delta))
        self._draw(z, delta)

    def _draw(self, z, delta):
        self.ax.plot(z, delta, 'r-', linewidth=2)
        self.ax.fill_between(z, delta, alpha=0.15, color='red')
        self.ax.set_xlabel('Spanwise position (normalised)')
        self.ax.set_ylabel('Cone half-angle (deg)')
        self.ax.set_title('Spanwise Cone Angle Profile')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


# ======================================================================
#  Main tab widget
# ======================================================================

class VMPLOWaveriderTab(QWidget):
    """GUI tab for Variable-Mach Power-Law Osculating waverider design."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.waverider = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Horizontal)

        # -- Left panel (parameters) --
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(300)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(self._create_mach_group())
        left_layout.addWidget(self._create_exponent_group())
        left_layout.addWidget(self._create_shock_group())
        left_layout.addWidget(self._create_icc_group())
        left_layout.addWidget(self._create_geometry_group())
        left_layout.addWidget(self._create_resolution_group())

        # Generate button
        self.generate_btn = QPushButton("Generate VMPLO Waverider")
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #2B5B2B; color: white; "
            "font-weight: bold; padding: 8px; font-size: 14px; }"
            "QPushButton:hover { background-color: #3B7B3B; }")
        self.generate_btn.clicked.connect(self.generate_waverider)
        left_layout.addWidget(self.generate_btn)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label)

        self.derived_label = QLabel(
            "Ma range: --\nn range: --\nCone angles: --")
        self.derived_label.setStyleSheet(
            "color: #aaaaaa; font-size: 10px; font-family: monospace; "
            "background-color: #1A1A1A; padding: 6px; border-radius: 3px;")
        self.derived_label.setWordWrap(True)
        left_layout.addWidget(self.derived_label)

        left_layout.addWidget(self._create_export_group())
        left_layout.addStretch()
        left_scroll.setWidget(left_widget)
        main_splitter.addWidget(left_scroll)

        # -- Right panel (visualization) --
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # View checkboxes
        check_layout = QHBoxLayout()
        self.show_upper_check = QCheckBox("Upper")
        self.show_upper_check.setChecked(True)
        self.show_lower_check = QCheckBox("Lower")
        self.show_lower_check.setChecked(True)
        self.show_le_check = QCheckBox("Leading Edge")
        self.show_le_check.setChecked(True)
        self.show_info_check = QCheckBox("Info")
        self.show_info_check.setChecked(True)
        for cb in [self.show_upper_check, self.show_lower_check,
                   self.show_le_check, self.show_info_check]:
            cb.stateChanged.connect(self._update_3d_plot)
            check_layout.addWidget(cb)
        check_layout.addStretch()
        right_layout.addLayout(check_layout)

        # 3D canvas with placeholder
        self.canvas_3d = VMPLOCanvas3D()
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)

        self.placeholder_label = QLabel(
            "Click 'Generate VMPLO Waverider'\nto create geometry")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet(
            "color: grey; font-style: italic; font-size: 12px;")

        self.canvas_stack = QStackedWidget()
        self.canvas_stack.addWidget(self.placeholder_label)
        canvas_widget = QWidget()
        cl = QVBoxLayout(canvas_widget)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.toolbar_3d)
        cl.addWidget(self.canvas_3d)
        self.canvas_stack.addWidget(canvas_widget)
        self.canvas_stack.setCurrentIndex(0)

        # Bottom tabs (distribution profiles)
        self.bottom_tabs = QTabWidget()
        self.canvas_dist = DistributionProfileCanvas()
        self.bottom_tabs.addTab(self.canvas_dist, "Distributions")
        self.canvas_cone = ConeAngleProfileCanvas()
        self.bottom_tabs.addTab(self.canvas_cone, "Cone Angle Profile")

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self.canvas_stack)
        right_splitter.addWidget(self.bottom_tabs)
        right_splitter.setSizes([500, 200])
        right_layout.addWidget(right_splitter)

        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([400, 800])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(main_splitter)

        self._update_beta_hint()

    # ==============================================================
    #  Parameter groups
    # ==============================================================

    def _create_mach_group(self):
        group = QGroupBox("Mach Distribution")
        layout = QGridLayout()

        layout.addWidget(QLabel("Ma centre:"), 0, 0)
        self.ma_center_spin = QDoubleSpinBox()
        self.ma_center_spin.setRange(1.5, 25.0)
        self.ma_center_spin.setValue(6.0)
        self.ma_center_spin.setSingleStep(0.5)
        self.ma_center_spin.setDecimals(1)
        self.ma_center_spin.setToolTip(
            "Design Mach number at the symmetry plane (z=0).")
        layout.addWidget(self.ma_center_spin, 0, 1)

        layout.addWidget(QLabel("Ma tip:"), 1, 0)
        self.ma_tip_spin = QDoubleSpinBox()
        self.ma_tip_spin.setRange(1.5, 25.0)
        self.ma_tip_spin.setValue(10.0)
        self.ma_tip_spin.setSingleStep(0.5)
        self.ma_tip_spin.setDecimals(1)
        self.ma_tip_spin.setToolTip(
            "Design Mach number at the wingtip (z=half_span).")
        layout.addWidget(self.ma_tip_spin, 1, 1)

        layout.addWidget(QLabel("Distribution:"), 2, 0)
        self.ma_dist_combo = QComboBox()
        self.ma_dist_combo.addItems([
            "Constant (use centre value)",
            "Linear (centre to tip)",
            "Quadratic (Liu et al.)"])
        self.ma_dist_combo.setCurrentIndex(2)
        self.ma_dist_combo.setToolTip(
            "Constant: uniform Ma across span.\n"
            "Linear: linearly varies from centre to tip.\n"
            "Quadratic: Ma(z) = m*z^2 + Ma_centre (Liu et al. 2019).")
        layout.addWidget(self.ma_dist_combo, 2, 1)

        layout.addWidget(QLabel("Control points:"), 3, 0)
        self.ma_ncp_spin = QSpinBox()
        self.ma_ncp_spin.setRange(2, 10)
        self.ma_ncp_spin.setValue(6)
        self.ma_ncp_spin.setToolTip("Number of B-spline control points for the distribution.")
        layout.addWidget(self.ma_ncp_spin, 3, 1)

        group.setLayout(layout)
        self.ma_center_spin.valueChanged.connect(self._update_beta_hint)
        return group

    def _create_exponent_group(self):
        group = QGroupBox("Power-Law Exponent")
        layout = QGridLayout()

        layout.addWidget(QLabel("n centre:"), 0, 0)
        self.n_center_spin = QDoubleSpinBox()
        self.n_center_spin.setRange(0.55, 1.00)
        self.n_center_spin.setValue(1.00)
        self.n_center_spin.setSingleStep(0.05)
        self.n_center_spin.setDecimals(2)
        self.n_center_spin.setToolTip(
            "Power-law exponent at symmetry plane.\n"
            "1.0 = cone (standard OC), <1.0 = blunter body (more volume).")
        layout.addWidget(self.n_center_spin, 0, 1)

        layout.addWidget(QLabel("n tip:"), 1, 0)
        self.n_tip_spin = QDoubleSpinBox()
        self.n_tip_spin.setRange(0.55, 1.00)
        self.n_tip_spin.setValue(1.00)
        self.n_tip_spin.setSingleStep(0.05)
        self.n_tip_spin.setDecimals(2)
        self.n_tip_spin.setToolTip(
            "Power-law exponent at wingtip.\n"
            "1.0 = cone, <1.0 = blunter body.")
        layout.addWidget(self.n_tip_spin, 1, 1)

        layout.addWidget(QLabel("Distribution:"), 2, 0)
        self.n_dist_combo = QComboBox()
        self.n_dist_combo.addItems([
            "Constant (use centre value)",
            "Linear (centre to tip)"])
        self.n_dist_combo.setCurrentIndex(0)
        self.n_dist_combo.setToolTip(
            "Constant: uniform n across span.\n"
            "Linear: linearly varies from centre to tip.")
        layout.addWidget(self.n_dist_combo, 2, 1)

        group.setLayout(layout)
        return group

    def _create_shock_group(self):
        group = QGroupBox("Shock Parameters")
        layout = QGridLayout()

        layout.addWidget(QLabel("Shock Angle (deg):"), 0, 0)
        beta_row = QHBoxLayout()
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(5.0, 45.0)
        self.beta_spin.setValue(13.0)
        self.beta_spin.setSingleStep(0.5)
        self.beta_spin.setDecimals(1)
        self.beta_spin.setToolTip(
            "Conical shock half-angle in degrees.\n"
            "Must be above Mach angle for shock attachment.")
        beta_row.addWidget(self.beta_spin)
        self.auto_beta_btn = QPushButton("Auto")
        self.auto_beta_btn.setFixedWidth(50)
        self.auto_beta_btn.setToolTip("Set shock angle to 1.2x Mach angle of Ma centre")
        self.auto_beta_btn.clicked.connect(self._auto_calculate_beta)
        beta_row.addWidget(self.auto_beta_btn)
        bw = QWidget()
        bw.setLayout(beta_row)
        layout.addWidget(bw, 0, 1)

        self.beta_hint_label = QLabel("")
        self.beta_hint_label.setStyleSheet("color: #888888; font-size: 10px;")
        self.beta_hint_label.setWordWrap(True)
        layout.addWidget(self.beta_hint_label, 1, 0, 1, 2)

        group.setLayout(layout)
        self.beta_spin.valueChanged.connect(self._update_beta_hint)
        return group

    def _create_geometry_group(self):
        group = QGroupBox("Geometry")
        layout = QGridLayout()

        layout.addWidget(QLabel("Length L (m):"), 0, 0)
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.5, 20.0)
        self.length_spin.setValue(3.0)
        self.length_spin.setSingleStep(0.1)
        self.length_spin.setDecimals(3)
        self.length_spin.setToolTip(
            "Vehicle length (m).  Streamwise extent from x=0 (nose) to "
            "x=L (base plane).")
        layout.addWidget(self.length_spin, 0, 1)

        layout.addWidget(QLabel("Height H (m):"), 1, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.01, 20.0)
        self.height_spin.setValue(round(3.0 * np.tan(13.0 * np.pi / 180), 3))
        self.height_spin.setSingleStep(0.01)
        self.height_spin.setDecimals(4)
        self.height_spin.setToolTip("Max vehicle height (reference for ICC).")
        layout.addWidget(self.height_spin, 1, 1)

        layout.addWidget(QLabel("Half-Width W (m):"), 2, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 10.0)
        self.width_spin.setValue(2.0)
        self.width_spin.setSingleStep(0.1)
        self.width_spin.setDecimals(4)
        self.width_spin.setToolTip("Vehicle half-span.")
        layout.addWidget(self.width_spin, 2, 1)

        layout.addWidget(QLabel("LE x at centreline (m):"), 3, 0)
        self.x_le_spin = QDoubleSpinBox()
        self.x_le_spin.setRange(0.001, 1.0)
        self.x_le_spin.setValue(0.05)
        self.x_le_spin.setSingleStep(0.01)
        self.x_le_spin.setDecimals(4)
        self.x_le_spin.setToolTip(
            "LE x-position at the symmetry plane (z=0).  The LE sweeps "
            "linearly from here to x=L at the wingtip.  Small positive "
            "(must be > 0 to avoid the n<1 power-law singularity).")
        layout.addWidget(self.x_le_spin, 3, 1)

        group.setLayout(layout)
        return group

    def _create_icc_group(self):
        """LE shaping (ICC depth-below-nose offset).

        In the swept-LE VMPLO formulation, the LE y-coordinate is
        derived from the power-law body surface plus an ICC-driven
        shaping offset.  ICC(z) acts as an additional *depth below
        nose level* at the base plane (0 = LE follows the body
        surface exactly; positive = LE dips further below).
        Fractions of the vehicle height H.
        """
        group = QGroupBox("LE Shaping (ICC depth-below-nose)")
        layout = QGridLayout()

        layout.addWidget(QLabel("ICC centre (/H):"), 0, 0)
        self.icc_center_spin = QDoubleSpinBox()
        self.icc_center_spin.setRange(0.0, 0.99)
        self.icc_center_spin.setValue(0.00)
        self.icc_center_spin.setSingleStep(0.05)
        self.icc_center_spin.setDecimals(3)
        self.icc_center_spin.setToolTip(
            "LE depth-below-nose offset at the symmetry plane, as a "
            "fraction of H.  0 = LE follows the body surface only "
            "(classical waverider).  Positive = LE dips further.")
        layout.addWidget(self.icc_center_spin, 0, 1)

        layout.addWidget(QLabel("ICC tip (/H):"), 1, 0)
        self.icc_tip_spin = QDoubleSpinBox()
        self.icc_tip_spin.setRange(0.0, 0.99)
        self.icc_tip_spin.setValue(0.30)
        self.icc_tip_spin.setSingleStep(0.05)
        self.icc_tip_spin.setDecimals(3)
        self.icc_tip_spin.setToolTip(
            "LE depth-below-nose offset at the wingtip, as a fraction "
            "of H.")
        layout.addWidget(self.icc_tip_spin, 1, 1)

        layout.addWidget(QLabel("Distribution:"), 2, 0)
        self.icc_dist_combo = QComboBox()
        self.icc_dist_combo.addItems([
            "Constant (use centre value)",
            "Linear (centre to tip)",
            "Quadratic (Liu-style)",
        ])
        self.icc_dist_combo.setCurrentIndex(1)
        self.icc_dist_combo.setToolTip(
            "Constant: flat ICC across span.\n"
            "Linear: straight ramp from centre to tip.\n"
            "Quadratic: y(z) = m*z^2 + y_center (dF/dz=0 at z=0).")
        layout.addWidget(self.icc_dist_combo, 2, 1)

        layout.addWidget(QLabel("Control points:"), 3, 0)
        self.icc_ncp_spin = QSpinBox()
        self.icc_ncp_spin.setRange(2, 10)
        self.icc_ncp_spin.setValue(6)
        self.icc_ncp_spin.setToolTip(
            "Number of B-spline control points for the ICC distribution.")
        layout.addWidget(self.icc_ncp_spin, 3, 1)

        group.setLayout(layout)
        return group

    def _create_resolution_group(self):
        group = QGroupBox("Resolution")
        layout = QGridLayout()

        layout.addWidget(QLabel("Osculating Planes:"), 0, 0)
        self.n_planes_spin = QSpinBox()
        self.n_planes_spin.setRange(10, 100)
        self.n_planes_spin.setValue(20)
        self.n_planes_spin.setToolTip("Number of spanwise stations.")
        layout.addWidget(self.n_planes_spin, 0, 1)

        layout.addWidget(QLabel("Streamwise Points:"), 1, 0)
        self.n_stream_spin = QSpinBox()
        self.n_stream_spin.setRange(10, 100)
        self.n_stream_spin.setValue(30)
        self.n_stream_spin.setToolTip("Number of streamwise points per streamline.")
        layout.addWidget(self.n_stream_spin, 1, 1)

        group.setLayout(layout)
        return group

    def _create_export_group(self):
        group = QGroupBox("Export")
        layout = QGridLayout()
        stl_btn = QPushButton("STL")
        stl_btn.clicked.connect(self.export_stl)
        step_btn = QPushButton("STEP")
        step_btn.clicked.connect(self.export_step)
        step_btn.setEnabled(CADQUERY_AVAILABLE)
        iges_btn = QPushButton("IGES")
        iges_btn.clicked.connect(self.export_iges)
        tdm_btn = QPushButton("3DM")
        tdm_btn.clicked.connect(self.export_3dm)
        self.half_vehicle_check = QCheckBox("Half vehicle (right side only)")
        layout.addWidget(stl_btn, 0, 0)
        layout.addWidget(step_btn, 0, 1)
        layout.addWidget(iges_btn, 0, 2)
        layout.addWidget(tdm_btn, 0, 3)
        layout.addWidget(self.half_vehicle_check, 1, 0, 1, 4)
        group.setLayout(layout)
        return group

    # ==============================================================
    #  Generation
    # ==============================================================

    def generate_waverider(self):
        try:
            self.status_label.setText("Generating...")
            self.status_label.setStyleSheet("color: black")
            self.generate_btn.setEnabled(False)
            QApplication.processEvents()

            # ---- Gather scalar params -----------------------------------
            L = self.length_spin.value()
            H = self.height_spin.value()
            W = self.width_spin.value()
            x_LE = self.x_le_spin.value()
            beta_design = self.beta_spin.value()
            ma_center = self.ma_center_spin.value()
            ma_tip = self.ma_tip_spin.value()
            n_center = self.n_center_spin.value()
            n_tip = self.n_tip_spin.value()
            icc_center_frac = self.icc_center_spin.value()
            icc_tip_frac = self.icc_tip_spin.value()

            # ---- Build B-spline distributions ---------------------------
            ma_dist_idx = self.ma_dist_combo.currentIndex()
            n_cp = self.ma_ncp_spin.value()
            # BSpline1D needs n_internal_knots = n_cp - 4 (coeff size =
            # n_internal + 4).  Clamp to a minimum of 1.
            k_ma = max(n_cp - 4, 1)
            if ma_dist_idx == 0:
                Ma_sp = BSpline1D.constant(ma_center, 0.0, W,
                                            n_internal_knots=k_ma)
            elif ma_dist_idx == 1:
                Ma_sp = BSpline1D.linear(ma_center, ma_tip, 0.0, W,
                                          n_internal_knots=k_ma)
            else:
                Ma_sp = BSpline1D.quadratic_liu(ma_center, ma_tip, 0.0, W,
                                                 n_internal_knots=k_ma)

            n_dist_idx = self.n_dist_combo.currentIndex()
            if n_dist_idx == 0:
                n_sp = BSpline1D.constant(n_center, 0.0, W,
                                           n_internal_knots=4)
            else:
                n_sp = BSpline1D.linear(n_center, n_tip, 0.0, W,
                                         n_internal_knots=4)

            icc_dist_idx = self.icc_dist_combo.currentIndex()
            icc_ncp = self.icc_ncp_spin.value()
            k_icc = max(icc_ncp - 4, 1)
            y_icc_center = icc_center_frac * H
            y_icc_tip = icc_tip_frac * H
            if icc_dist_idx == 0:
                icc_sp = BSpline1D.constant(y_icc_center, 0.0, W,
                                             n_internal_knots=k_icc)
            elif icc_dist_idx == 1:
                icc_sp = BSpline1D.linear(y_icc_center, y_icc_tip, 0.0, W,
                                           n_internal_knots=k_icc)
            else:
                icc_sp = BSpline1D.quadratic_liu(
                    y_icc_center, y_icc_tip, 0.0, W,
                    n_internal_knots=k_icc)

            # ---- Assemble + build waverider ----------------------------
            assembly = OsculatingAssembly(
                Ma_spline=Ma_sp,
                n_spline=n_sp,
                ICC_spline=icc_sp,
                US_spline=None,
                beta_design=beta_design,
                L=L, W=W, H=H,
                x_LE=x_LE,
            )
            self.waverider = VMPLOWaverider(
                assembly,
                n_planes=self.n_planes_spin.value(),
                n_streamwise=self.n_stream_spin.value(),
            )

            # Update visualizations
            self.canvas_stack.setCurrentIndex(1)
            self._update_3d_plot()
            self.canvas_dist.update_profile(self.waverider)
            self.canvas_cone.update_profile(self.waverider)

            # Derived quantities
            info_lines = []
            if hasattr(self.waverider, '_mach_per_station'):
                ma = self.waverider._mach_per_station
                info_lines.append(f"Ma range: {ma.min():.1f} - {ma.max():.1f}")
            if hasattr(self.waverider, '_n_per_station'):
                n = self.waverider._n_per_station
                info_lines.append(f"n range: {n.min():.2f} - {n.max():.2f}")
            if hasattr(self.waverider, 'cone_angles_deg'):
                ca = self.waverider.cone_angles_deg
                info_lines.append(f"Cone angles: {ca.min():.1f} - {ca.max():.1f} deg")
            info_lines.append(f"Length: {self.waverider.length:.4f} m")
            info_lines.append(
                f"Streams: {len(self.waverider.upper_surface_streams)} US + "
                f"{len(self.waverider.lower_surface_streams)} LS")
            try:
                vol = self.waverider.compute_volume()
                info_lines.append(f"Volume: {vol:.6f} m3")
            except Exception:
                pass
            self.derived_label.setText("\n".join(info_lines))

            self.status_label.setText(
                f"Generated: L={self.waverider.length:.3f}m")
            self.status_label.setStyleSheet("color: green")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red")
            import traceback
            traceback.print_exc()
        finally:
            self.generate_btn.setEnabled(True)

    # ==============================================================
    #  Visualisation
    # ==============================================================

    def _update_3d_plot(self):
        if self.waverider is None:
            return
        half_only = self.half_vehicle_check.isChecked()
        self.canvas_3d.plot_waverider(
            self.waverider,
            half_only=half_only,
            show_upper=self.show_upper_check.isChecked(),
            show_lower=self.show_lower_check.isChecked(),
            show_le=self.show_le_check.isChecked(),
            show_info=self.show_info_check.isChecked())

    # ==============================================================
    #  Validation
    # ==============================================================

    def _auto_calculate_beta(self):
        ma = self.ma_center_spin.value()
        mu = math.degrees(math.asin(1.0 / max(ma, 1.01)))
        self.beta_spin.setValue(round(mu * 1.2, 1))

    def _update_beta_hint(self):
        ma = self.ma_center_spin.value()
        mu = math.degrees(math.asin(1.0 / max(ma, 1.01)))
        beta = self.beta_spin.value()

        if beta < mu + 0.1:
            self.beta_hint_label.setText(
                f"WARNING: beta={beta:.1f} < Mach angle {mu:.1f}! Shock detached!")
            self.beta_hint_label.setStyleSheet(
                "color: #EF4444; font-size: 10px; font-weight: bold;")
        else:
            self.beta_hint_label.setText(
                f"Mach angle {mu:.1f} | "
                f"Rec: {mu*1.2:.1f} ({mu+0.5:.1f}-{mu*1.5:.1f})")
            self.beta_hint_label.setStyleSheet("color: #4ADE80; font-size: 10px;")

    # ==============================================================
    #  Export
    # ==============================================================

    def export_stl(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save STL", "vmplo_waverider.stl", "STL Files (*.stl)")
        if filename:
            try:
                self.waverider.export_stl(filename)
                self.status_label.setText(f"Exported to {filename}")
                QMessageBox.information(self, "Success", f"STL exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed:\n{str(e)}")

    def export_iges(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save IGES", "vmplo_waverider.igs",
            "IGES Files (*.igs *.iges)")
        if not filename:
            return
        try:
            n_resample = 50
            length = self.waverider.length
            scale = 1000.0

            us_list = self._resample_to_common_x(
                self.waverider.upper_surface_streams[:-1],
                length, n_resample)
            ls_list = self._resample_to_common_x(
                self.waverider.lower_surface_streams[:-1],
                length, n_resample)

            half_only = self.half_vehicle_check.isChecked()
            self._export_iges_from_streams(
                us_list, ls_list, filename,
                scale=scale, half_only=half_only)

            import os
            sz = os.path.getsize(filename) // 1024
            self.status_label.setText(f"IGES exported: {sz} KB")
            QMessageBox.information(self, "Success",
                                    f"IGES exported to:\n{filename}\n({sz} KB)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Export Failed",
                f"Failed to export IGES:\n{str(e)}")

    @staticmethod
    def _export_iges_from_streams(upper_streams, lower_streams, filename,
                                  scale=1000.0, half_only=False):
        """Export upper and lower surfaces as IGES B-spline surfaces.

        Uses OCP (pythonOCC) directly — no CadQuery needed.
        """
        from OCP.TColgp import TColgp_Array2OfPnt
        from OCP.gp import gp_Pnt
        from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
        from OCP.GeomAbs import GeomAbs_C2
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.IGESControl import IGESControl_Writer
        from OCP.IFSelect import IFSelect_RetDone

        def make_bspline_face(streams, sc):
            n_u = len(streams)
            n_v = streams[0].shape[0]
            grid = TColgp_Array2OfPnt(1, n_u, 1, n_v)
            for i, stream in enumerate(streams):
                for j in range(n_v):
                    pt = stream[j] * sc
                    grid.SetValue(i + 1, j + 1,
                                  gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))
            approx = GeomAPI_PointsToBSplineSurface()
            approx.Init(grid, 3, 8, GeomAbs_C2, 1e-3)
            if not approx.IsDone():
                approx2 = GeomAPI_PointsToBSplineSurface()
                approx2.Init(grid, 3, 8, GeomAbs_C2, 1e-2)
                if not approx2.IsDone():
                    raise RuntimeError("BSpline surface fit failed")
                surf = approx2.Surface()
            else:
                surf = approx.Surface()
            face_builder = BRepBuilderAPI_MakeFace(surf, 1e-3)
            face_builder.Build()
            if not face_builder.IsDone():
                raise RuntimeError("MakeFace failed")
            return face_builder.Face()

        def mirror_streams(streams):
            mirrored = []
            for s in streams:
                m = s.copy()
                m[:, 2] = -m[:, 2]
                mirrored.append(m)
            return list(reversed(mirrored))

        if half_only:
            upper_full = upper_streams
            lower_full = lower_streams
        else:
            # Mirror and join, skipping duplicate centerline (Z=0)
            upper_full = mirror_streams(upper_streams)[:-1] + upper_streams
            lower_full = mirror_streams(lower_streams)[:-1] + lower_streams

        upper_face = make_bspline_face(upper_full, scale)
        lower_face = make_bspline_face(lower_full, scale)

        writer = IGESControl_Writer()
        writer.AddShape(upper_face)
        writer.AddShape(lower_face)
        writer.ComputeModel()
        status = writer.Write(filename)
        if status != IFSelect_RetDone:
            raise RuntimeError("IGES write failed")

    def export_3dm(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save 3DM", "vmplo_waverider.3dm",
            "Rhino Files (*.3dm)")
        if not filename:
            return
        try:
            n_resample = 50
            length = self.waverider.length
            scale = 1000.0

            us_list = self._resample_to_common_x(
                self.waverider.upper_surface_streams[:-1],
                length, n_resample)
            ls_list = self._resample_to_common_x(
                self.waverider.lower_surface_streams[:-1],
                length, n_resample)

            half_only = self.half_vehicle_check.isChecked()
            self._export_3dm_surfaces(
                us_list, ls_list, filename,
                scale=scale, half_only=half_only)

            import os
            sz = os.path.getsize(filename) // 1024
            self.status_label.setText(f"3DM exported: {sz} KB")
            QMessageBox.information(self, "Success",
                                    f"3DM exported to:\n{filename}\n({sz} KB)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Export Failed",
                f"Failed to export 3DM:\n{str(e)}")

    @staticmethod
    def _export_3dm_surfaces(upper_streams, lower_streams, filename,
                             scale=1000.0, half_only=False):
        """Export upper and lower surfaces as NURBS in Rhino 3DM format."""
        import rhino3dm as r3d

        model = r3d.File3dm()

        def make_nurbs_surface(streams, sc):
            n_u = len(streams)
            n_v = streams[0].shape[0]
            deg_u = min(3, n_u - 1)
            deg_v = min(3, n_v - 1)

            surf = r3d.NurbsSurface.Create(
                3, False,
                deg_u + 1, deg_v + 1,
                n_u, n_v)

            for i, stream in enumerate(streams):
                for j in range(n_v):
                    pt = stream[j] * sc
                    surf.Points[i, j] = r3d.Point4d(
                        float(pt[0]), float(pt[1]),
                        float(pt[2]), 1.0)

            n_knots_u = n_u + deg_u - 1
            for k in range(n_knots_u):
                surf.KnotsU[k] = float(k)
            n_knots_v = n_v + deg_v - 1
            for k in range(n_knots_v):
                surf.KnotsV[k] = float(k)

            return surf

        def mirror_streams(streams):
            mirrored = []
            for s in reversed(streams):
                m = s.copy()
                m[:, 2] = -m[:, 2]
                mirrored.append(m)
            return mirrored

        if half_only:
            upper_full = list(upper_streams)
            lower_full = list(lower_streams)
        else:
            upper_full = mirror_streams(upper_streams)[:-1] + list(upper_streams)
            lower_full = mirror_streams(lower_streams)[:-1] + list(lower_streams)

        # --- NURBS surfaces (editable, for reference) ---
        upper_surf = make_nurbs_surface(upper_full, scale)
        lower_surf = make_nurbs_surface(lower_full, scale)
        model.Objects.AddSurface(upper_surf)
        model.Objects.AddSurface(lower_surf)

        # --- Watertight closed mesh ---
        # Uses the same proven topology as export_stl: two full grids
        # with 6 closure face strips (upper, lower, left, right, LE, TE).
        n_span = len(upper_full)
        n_stream = upper_full[0].shape[0]

        mesh = r3d.Mesh()

        # Upper grid vertices
        for i in range(n_span):
            for j in range(n_stream):
                pt = upper_full[i][j] * scale
                mesh.Vertices.Add(float(pt[0]), float(pt[1]), float(pt[2]))
        lo = n_span * n_stream
        # Lower grid vertices
        for i in range(n_span):
            for j in range(n_stream):
                pt = lower_full[i][j] * scale
                mesh.Vertices.Add(float(pt[0]), float(pt[1]), float(pt[2]))

        # Upper surface quads
        for i in range(n_span - 1):
            for j in range(n_stream - 1):
                a = i * n_stream + j
                b = a + 1
                c = (i + 1) * n_stream + j
                d = c + 1
                mesh.Faces.AddFace(a, c, d, b)

        # Lower surface quads (reversed winding)
        for i in range(n_span - 1):
            for j in range(n_stream - 1):
                a = lo + i * n_stream + j
                b = a + 1
                c = lo + (i + 1) * n_stream + j
                d = c + 1
                mesh.Faces.AddFace(a, b, d, c)

        # Left side (i=0)
        for j in range(n_stream - 1):
            mesh.Faces.AddFace(j, lo + j, lo + j + 1, j + 1)

        # Right side (i=n_span-1)
        for j in range(n_stream - 1):
            u0 = (n_span - 1) * n_stream + j
            l0 = lo + u0
            mesh.Faces.AddFace(u0, u0 + 1, l0 + 1, l0)

        # Base / TE (j=n_stream-1)
        for i in range(n_span - 1):
            u0 = i * n_stream + (n_stream - 1)
            u1 = (i + 1) * n_stream + (n_stream - 1)
            l0 = lo + i * n_stream + (n_stream - 1)
            l1 = lo + (i + 1) * n_stream + (n_stream - 1)
            mesh.Faces.AddFace(u0, u1, l1, l0)

        # LE (j=0)
        for i in range(n_span - 1):
            u0 = i * n_stream
            u1 = (i + 1) * n_stream
            l0 = lo + i * n_stream
            l1 = lo + (i + 1) * n_stream
            mesh.Faces.AddFace(u0, l0, l1, u1)

        mesh.Normals.ComputeNormals()
        mesh.Compact()

        is_closed = mesh.IsClosed
        model.Objects.AddMesh(mesh)

        n_faces = mesh.Faces.Count
        print(f"[3DM] 2 NURBS + 1 mesh ({n_faces} faces, "
              f"closed={is_closed}) -> {filename}")
        model.Write(filename, 7)

    @staticmethod
    def _resample_to_common_x(streams, length, n_pts=50):
        """Resample all streams onto a common X grid [0, length].

        Streams that don't reach x=0 (outer streams whose LE is further
        aft) are extended backward with their LE point held constant.
        This ensures interpPlate() sees points at the same X stations
        across all spanwise positions, producing a smooth surface.
        """
        x_common = np.linspace(0, length, n_pts)
        resampled = []

        for s in streams:
            if len(s) < 2:
                resampled.append(np.tile(s[0], (n_pts, 1)))
                continue

            xs = s[:, 0]
            x_le = xs[0]
            x_te = xs[-1]

            new_pts = np.zeros((n_pts, 3))
            for k, x in enumerate(x_common):
                if x <= x_le:
                    new_pts[k] = s[0]
                elif x >= x_te:
                    new_pts[k] = s[-1]
                else:
                    for dim in range(3):
                        new_pts[k, dim] = float(
                            np.interp(x, xs, s[:, dim]))
            resampled.append(new_pts)

        return resampled

    def export_step(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save STEP", "vmplo_waverider.step", "STEP Files (*.step)")
        if not filename:
            return
        try:
            # Delegate to the generator's to_CAD, which routes through
            # build_waverider_solid — the same watertight pipeline that
            # shadow_waverider uses (4-face NURBS + wingtip TE closure +
            # sewn solid). The earlier hand-rolled path here dropped the
            # tip stream via [:-1], which left a missing wedge at the
            # wingtip and produced the LE-truncation / back-face-gap
            # defects.
            sides = 'right' if self.half_vehicle_check.isChecked() else 'both'
            self.waverider.to_CAD(
                sides=sides, export=True, filename=filename, scale=1000.0)
            self.status_label.setText(f"Exported to {filename}")
            QMessageBox.information(self, "Success", f"STEP exported to:\n{filename}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Export Failed",
                f"Failed to export STEP:\n{str(e)}\n\n"
                f"Try STL export instead.")

    # ==============================================================
    #  Settings save/load
    # ==============================================================

    def get_params_dict(self):
        return {
            'ma_center': self.ma_center_spin.value(),
            'ma_tip': self.ma_tip_spin.value(),
            'ma_dist': self.ma_dist_combo.currentIndex(),
            'ma_ncp': self.ma_ncp_spin.value(),
            'n_center': self.n_center_spin.value(),
            'n_tip': self.n_tip_spin.value(),
            'n_dist': self.n_dist_combo.currentIndex(),
            'beta': self.beta_spin.value(),
            'length': self.length_spin.value(),
            'height': self.height_spin.value(),
            'width': self.width_spin.value(),
            'x_LE': self.x_le_spin.value(),
            'icc_center': self.icc_center_spin.value(),
            'icc_tip': self.icc_tip_spin.value(),
            'icc_dist': self.icc_dist_combo.currentIndex(),
            'icc_ncp': self.icc_ncp_spin.value(),
            'n_planes': self.n_planes_spin.value(),
            'n_streamwise': self.n_stream_spin.value(),
            'half_vehicle': self.half_vehicle_check.isChecked(),
        }

    def set_params_dict(self, d):
        def _s(widget, value):
            if value is None:
                return
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(int(value))

        _s(self.ma_center_spin, d.get('ma_center'))
        _s(self.ma_tip_spin, d.get('ma_tip'))
        _s(self.ma_dist_combo, d.get('ma_dist'))
        _s(self.ma_ncp_spin, d.get('ma_ncp'))
        _s(self.n_center_spin, d.get('n_center'))
        _s(self.n_tip_spin, d.get('n_tip'))
        _s(self.n_dist_combo, d.get('n_dist'))
        _s(self.beta_spin, d.get('beta'))
        _s(self.length_spin, d.get('length'))
        _s(self.height_spin, d.get('height'))
        _s(self.width_spin, d.get('width'))
        _s(self.x_le_spin, d.get('x_LE'))
        _s(self.icc_center_spin, d.get('icc_center'))
        _s(self.icc_tip_spin, d.get('icc_tip'))
        _s(self.icc_dist_combo, d.get('icc_dist'))
        _s(self.icc_ncp_spin, d.get('icc_ncp'))
        _s(self.n_planes_spin, d.get('n_planes'))
        _s(self.n_stream_spin, d.get('n_streamwise'))
        _s(self.half_vehicle_check, d.get('half_vehicle'))
        # Legacy keys (x1-x4) ignored — schema changed with the
        # VMPLO rewrite.


if __name__ == '__main__':
    app = QApplication(sys.argv)
    tab = VMPLOWaveriderTab()
    tab.setWindowTitle("VMPLO Waverider Tab (Standalone Test)")
    tab.resize(1200, 800)
    tab.show()
    sys.exit(app.exec_())
