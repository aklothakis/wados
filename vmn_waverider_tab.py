"""Variable Mach Number Waverider Tab — Li et al. (2018) wide-speed range design."""

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

from waverider_generator.vmn_generator import VMNWaverider

try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False


# ======================================================================
#  Canvas classes
# ======================================================================

class VMNCanvas3D(FigureCanvas):
    """3D matplotlib canvas for stream-based waverider visualisation."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        self._info_text = None

    def plot_waverider(self, wr, half_only=False, show_upper=True,
                       show_lower=True, show_le=True, show_info=True,
                       title_prefix='VMN Waverider'):
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

        ma_min = getattr(wr, 'Ma_min', 0)
        ma_max = getattr(wr, 'Ma_max', 0)
        beta = getattr(wr, 'beta_deg', 0)
        self.ax.set_xlabel('Z (Span)', color='#FFFFFF')
        self.ax.set_ylabel('X (Streamwise)', color='#FFFFFF')
        self.ax.set_zlabel('Y (Vertical)', color='#FFFFFF')
        self.ax.set_title(f'{title_prefix} (Ma={ma_min:.0f}-{ma_max:.0f}, '
                          f'beta={beta:.1f} deg)', color='#FFFFFF')
        self.ax.tick_params(colors='#888888')
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper left')
        self._set_axes_equal()

        if show_info:
            self._draw_info_panel(wr, title_prefix)
        self.fig.tight_layout()
        self.draw()

    def _draw_info_panel(self, wr, title='VMN WAVERIDER'):
        ma_min = getattr(wr, 'Ma_min', 0)
        ma_max = getattr(wr, 'Ma_max', 0)
        beta = getattr(wr, 'beta_deg', 0)
        length = getattr(wr, 'length', 0)
        direction = getattr(wr, 'direction', '')
        lines = [title]
        lines.append(f"  Ma range       {ma_min:.0f} - {ma_max:.0f}")
        lines.append(f"  beta           {beta:.1f} deg")
        lines.append(f"  Direction      {direction}")
        lines.append(f"  Length          {length:.4f} m")
        if hasattr(wr, 'cone_angles_deg'):
            ca = wr.cone_angles_deg
            lines.append(f"  Cone angles    {ca.min():.1f} - {ca.max():.1f} deg")

        vol = 0.0
        a_plan = 0.0
        try:
            from waverider_gui import calculate_waverider_volume
            vol = calculate_waverider_volume(wr)
            lines.append(f"  Volume          {vol:.6f} m3")
        except Exception:
            pass

        try:
            # Compute planform area from streams (X-Z projection)
            # Integrate chord (x_te - x_le) across span (z)
            us = wr.upper_surface_streams
            chords = []
            z_pos = []
            for s in us:
                if s.shape[0] < 2:
                    continue
                chords.append(s[-1, 0] - s[0, 0])
                z_pos.append(s[0, 2])
            if len(chords) > 2:
                a_plan = float(np.trapz(chords, z_pos))
                lines.append(f"  Planform Area   {a_plan:.6f} m2")
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


class MachProfileCanvas(FigureCanvas):
    """2D plot showing spanwise Mach distribution."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 3))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._plot_default()

    def _plot_default(self):
        self.ax.clear()
        z = np.linspace(0, 1.0, 30)
        ma = np.linspace(10, 6, 30)
        self._draw(z, ma)

    def update_profile(self, wr):
        self.ax.clear()
        if wr is None or not hasattr(wr, '_mach_per_station'):
            self._plot_default()
            return
        ma = wr._mach_per_station
        z = np.linspace(0, 1.0, len(ma))
        self._draw(z, ma)

    def _draw(self, z, ma):
        self.ax.plot(z, ma, 'b-', linewidth=2)
        self.ax.fill_between(z, ma, alpha=0.15, color='blue')
        self.ax.set_xlabel('Spanwise position (normalised)')
        self.ax.set_ylabel('Design Mach number')
        self.ax.set_title('Spanwise Mach Distribution')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


class ConeAngleCanvas(FigureCanvas):
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

class VMNWaveriderTab(QWidget):
    """GUI tab for Variable Mach Number waverider design."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.waverider = None
        self.view_mode = 'vmn'
        self._updating = False
        self._ma_min_ref = None
        self._ma_max_ref = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Horizontal)

        # -- Left panel --
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(300)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(self._create_mach_group())
        left_layout.addWidget(self._create_shock_group())
        left_layout.addWidget(self._create_geometry_group())
        left_layout.addWidget(self._create_resolution_group())
        left_layout.addWidget(self._create_blunting_group())

        # Generate button
        self.generate_btn = QPushButton("Generate VMN Waverider")
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 8px; font-size: 14px; }")
        self.generate_btn.clicked.connect(self.generate_waverider)
        left_layout.addWidget(self.generate_btn)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label)

        self.derived_label = QLabel(
            "Cone angles: --\nVehicle width: --\nVehicle height: --")
        self.derived_label.setStyleSheet(
            "color: #aaaaaa; font-size: 10px; font-family: monospace; "
            "background-color: #1A1A1A; padding: 6px; border-radius: 3px;")
        self.derived_label.setWordWrap(True)
        left_layout.addWidget(self.derived_label)

        left_layout.addWidget(self._create_export_group())
        left_layout.addStretch()
        left_scroll.setWidget(left_widget)
        main_splitter.addWidget(left_scroll)

        # -- Right panel --
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # View mode toggle
        toggle_layout = QHBoxLayout()
        self.btn_ma_min = QPushButton("Ma_min")
        self.btn_vmn = QPushButton("VMN")
        self.btn_ma_max = QPushButton("Ma_max")
        for btn in [self.btn_ma_min, self.btn_vmn, self.btn_ma_max]:
            btn.setCheckable(True)
            btn.setEnabled(False)
            btn.setStyleSheet(
                "QPushButton { padding: 4px 12px; } "
                "QPushButton:checked { background-color: #D97706; color: white; "
                "font-weight: bold; }")
        self.btn_vmn.setChecked(True)
        self.btn_ma_min.clicked.connect(lambda: self._set_view_mode('ma_min'))
        self.btn_vmn.clicked.connect(lambda: self._set_view_mode('vmn'))
        self.btn_ma_max.clicked.connect(lambda: self._set_view_mode('ma_max'))
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.btn_ma_min)
        toggle_layout.addWidget(self.btn_vmn)
        toggle_layout.addWidget(self.btn_ma_max)
        toggle_layout.addStretch()
        right_layout.addLayout(toggle_layout)

        # 3D canvas with placeholder
        self.canvas_3d = VMNCanvas3D()
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)

        self.placeholder_label = QLabel(
            "Click 'Generate VMN Waverider'\nto create geometry")
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

        # Bottom tabs
        self.bottom_tabs = QTabWidget()
        self.canvas_mach = MachProfileCanvas()
        self.bottom_tabs.addTab(self.canvas_mach, "Mach Distribution")
        self.canvas_cone = ConeAngleCanvas()
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
        group = QGroupBox("Mach Range")
        layout = QGridLayout()

        layout.addWidget(QLabel("Ma min:"), 0, 0)
        self.ma_min_spin = QDoubleSpinBox()
        self.ma_min_spin.setRange(1.5, 20.0)
        self.ma_min_spin.setValue(6.0)
        self.ma_min_spin.setSingleStep(0.5)
        self.ma_min_spin.setDecimals(1)
        self.ma_min_spin.setToolTip(
            "Minimum design Mach number.\n"
            "Assigned to wingtip (decreasing) or centreline (increasing).")
        layout.addWidget(self.ma_min_spin, 0, 1)

        layout.addWidget(QLabel("Ma max:"), 1, 0)
        self.ma_max_spin = QDoubleSpinBox()
        self.ma_max_spin.setRange(1.5, 20.0)
        self.ma_max_spin.setValue(10.0)
        self.ma_max_spin.setSingleStep(0.5)
        self.ma_max_spin.setDecimals(1)
        self.ma_max_spin.setToolTip(
            "Maximum design Mach number.\n"
            "Assigned to centreline (decreasing) or wingtip (increasing).")
        layout.addWidget(self.ma_max_spin, 1, 1)

        layout.addWidget(QLabel("Direction:"), 2, 0)
        self.direction_combo = QComboBox()
        self.direction_combo.addItems([
            "Decreasing (Ma_max at centre)",
            "Increasing (Ma_min at centre)"])
        self.direction_combo.setToolTip(
            "Decreasing: centre gets high Mach (thin), tips get low Mach (thick).\n"
            "  -> More volume (Li et al. Case 3)\n"
            "Increasing: centre gets low Mach (thick), tips get high Mach (thin).\n"
            "  -> Less volume (Li et al. Case 4)")
        layout.addWidget(self.direction_combo, 2, 1)

        group.setLayout(layout)
        self.ma_min_spin.valueChanged.connect(self._update_beta_hint)
        self.ma_max_spin.valueChanged.connect(self._update_beta_hint)
        return group

    def _create_shock_group(self):
        group = QGroupBox("Shock Parameters")
        layout = QGridLayout()

        layout.addWidget(QLabel("Shock Angle (deg):"), 0, 0)
        beta_row = QHBoxLayout()
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(5.0, 45.0)
        self.beta_spin.setValue(13.5)
        self.beta_spin.setSingleStep(0.5)
        self.beta_spin.setDecimals(1)
        self.beta_spin.setToolTip(
            "Conical shock half-angle in degrees.\n"
            "Must be above Mach angle of Ma_min for shock attachment.")
        beta_row.addWidget(self.beta_spin)
        self.auto_beta_btn = QPushButton("Auto")
        self.auto_beta_btn.setFixedWidth(50)
        self.auto_beta_btn.setToolTip("Set shock angle to 1.2x Mach angle of Ma_min")
        self.auto_beta_btn.clicked.connect(self._auto_calculate_beta)
        beta_row.addWidget(self.auto_beta_btn)
        bw = QWidget(); bw.setLayout(beta_row)
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

        layout.addWidget(QLabel("Length (m):"), 0, 0)
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.1, 15.0)
        self.length_spin.setValue(0.6)
        self.length_spin.setSingleStep(0.1)
        self.length_spin.setDecimals(3)
        self.length_spin.setToolTip("Vehicle streamwise length in metres.")
        layout.addWidget(self.length_spin, 0, 1)

        layout.addWidget(QLabel("S (UE position):"), 1, 0)
        self.s_spin = QDoubleSpinBox()
        self.s_spin.setRange(0.05, 0.95)
        self.s_spin.setValue(0.4)
        self.s_spin.setSingleStep(0.05)
        self.s_spin.setDecimals(2)
        self.s_spin.setToolTip(
            "UE start position as fraction of shock radius.\n"
            "S=0.4: UE starts at 40% of shock radius inboard.\n"
            "Smaller S -> narrower vehicle, larger S -> wider.")
        layout.addWidget(self.s_spin, 1, 1)

        layout.addWidget(QLabel("A0 (UE conic):"), 2, 0)
        self.a0_spin = QDoubleSpinBox()
        self.a0_spin.setRange(0.01, 50.0)
        self.a0_spin.setValue(1.7233)
        self.a0_spin.setSingleStep(0.1)
        self.a0_spin.setDecimals(4)
        self.a0_spin.setToolTip(
            "Conic coefficient for upper edge curve.\n"
            "x_ue = A0*y^2 + R0. Must be positive (paper Table 1: 1.7233).\n"
            "Larger -> wider UE; smaller -> narrower UE.")
        layout.addWidget(self.a0_spin, 2, 1)

        group.setLayout(layout)
        return group

    def _create_resolution_group(self):
        group = QGroupBox("Resolution")
        layout = QGridLayout()

        layout.addWidget(QLabel("Spanwise Points:"), 0, 0)
        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(10, 100)
        self.n_points_spin.setValue(30)
        self.n_points_spin.setToolTip("Number of spanwise stations.")
        layout.addWidget(self.n_points_spin, 0, 1)

        layout.addWidget(QLabel("Streamwise Points:"), 1, 0)
        self.n_stream_spin = QSpinBox()
        self.n_stream_spin.setRange(10, 100)
        self.n_stream_spin.setValue(20)
        self.n_stream_spin.setToolTip("Number of streamwise points per streamline.")
        layout.addWidget(self.n_stream_spin, 1, 1)

        group.setLayout(layout)
        return group

    def _create_blunting_group(self):
        group = QGroupBox("Fillet Compensation")
        layout = QGridLayout()

        layout.addWidget(QLabel("Fillet Radius (mm):"), 0, 0)
        self.blunt_radius_spin = QDoubleSpinBox()
        self.blunt_radius_spin.setRange(0.0, 50.0)
        self.blunt_radius_spin.setValue(0.0)
        self.blunt_radius_spin.setSingleStep(0.5)
        self.blunt_radius_spin.setDecimals(1)
        self.blunt_radius_spin.setToolTip(
            "CAD fillet radius you plan to apply in mm.\n"
            "0 = no compensation (sharp LE export).\n"
            "When set, the vehicle is generated slightly\n"
            "longer so that after filleting in CAD, the\n"
            "final length matches your design target.\n\n"
            "Formula: dL = r * (1/sin(theta) - 1)\n"
            "where theta = local LE half-angle.")
        layout.addWidget(self.blunt_radius_spin, 0, 1)

        self.compensation_label = QLabel("Compensation: --")
        self.compensation_label.setStyleSheet(
            "color: #aaaaaa; font-size: 10px; font-family: monospace;")
        layout.addWidget(self.compensation_label, 1, 0, 1, 2)

        self.blunt_radius_spin.valueChanged.connect(
            self._update_compensation_hint)

        group.setLayout(layout)
        return group

    def _update_compensation_hint(self):
        r_mm = self.blunt_radius_spin.value()
        if r_mm < 0.01 or self.waverider is None:
            if r_mm < 0.01:
                self.compensation_label.setText("Compensation: none (sharp)")
            return
        # Estimate nose retreat from centreline cone angle
        ca = self.waverider.cone_angles_deg
        theta_min = np.radians(ca.min())  # smallest half-angle = max retreat
        theta_max = np.radians(ca.max())
        r_m = r_mm / 1000.0
        dL_max = r_m * (1.0 / np.sin(theta_min) - 1.0) * 1000
        dL_min = r_m * (1.0 / np.sin(theta_max) - 1.0) * 1000
        self.compensation_label.setText(
            f"Nose retreat: {dL_min:.1f}-{dL_max:.1f} mm\n"
            f"Auto +{dL_max:.1f} mm length on next generate")

    def _create_export_group(self):
        group = QGroupBox("Export")
        layout = QGridLayout()
        stl_btn = QPushButton("STL")
        stl_btn.clicked.connect(self.export_stl)
        tri_btn = QPushButton("TRI")
        tri_btn.clicked.connect(self.export_tri)
        step_btn = QPushButton("STEP")
        step_btn.clicked.connect(self.export_step)
        step_btn.setEnabled(CADQUERY_AVAILABLE)
        self.half_vehicle_check = QCheckBox("Half vehicle (right side only)")
        layout.addWidget(stl_btn, 0, 0)
        layout.addWidget(tri_btn, 0, 1)
        layout.addWidget(step_btn, 1, 0, 1, 2)
        layout.addWidget(self.half_vehicle_check, 2, 0, 1, 2)
        group.setLayout(layout)
        return group

    # ==============================================================
    #  View mode
    # ==============================================================

    def _set_view_mode(self, mode):
        self.view_mode = mode
        self.btn_ma_min.setChecked(mode == 'ma_min')
        self.btn_vmn.setChecked(mode == 'vmn')
        self.btn_ma_max.setChecked(mode == 'ma_max')
        self._update_3d_plot()

    # ==============================================================
    #  Generation
    # ==============================================================

    def generate_waverider(self):
        try:
            self.status_label.setText("Generating...")
            self.status_label.setStyleSheet("color: black")
            QApplication.processEvents()

            direction = 'decreasing' if self.direction_combo.currentIndex() == 0 else 'increasing'

            # Compute fillet compensation ΔL
            length = self.length_spin.value()
            r_mm = self.blunt_radius_spin.value()
            if r_mm > 0.01:
                from waverider_generator.flowfield import cone_angle as _ca_func
                beta = self.beta_spin.value()
                ma_max = self.ma_max_spin.value()
                theta_min_deg = _ca_func(ma_max, beta, 1.4)
                r_m = r_mm / 1000.0
                dL = r_m * (1.0 / np.sin(np.radians(theta_min_deg)) - 1.0)
                length += dL

            self.waverider = VMNWaverider(
                Ma_min=self.ma_min_spin.value(),
                Ma_max=self.ma_max_spin.value(),
                beta_deg=self.beta_spin.value(),
                L0=1.0,
                S=self.s_spin.value(),
                A0=self.a0_spin.value(),
                length=length,
                direction=direction,
                n_points=self.n_points_spin.value(),
                n_streamwise=self.n_stream_spin.value(),
            )

            # Generate Ma_min and Ma_max reference waveriders (same compensated length)
            try:
                self._ma_min_ref = VMNWaverider(
                    Ma_min=self.ma_min_spin.value(),
                    Ma_max=self.ma_min_spin.value(),
                    beta_deg=self.beta_spin.value(),
                    L0=1.0, S=self.s_spin.value(),
                    A0=self.a0_spin.value(),
                    length=length,
                    direction=direction,
                    n_points=self.n_points_spin.value(),
                    n_streamwise=self.n_stream_spin.value(),
                    )
            except Exception:
                self._ma_min_ref = None

            try:
                self._ma_max_ref = VMNWaverider(
                    Ma_min=self.ma_max_spin.value(),
                    Ma_max=self.ma_max_spin.value(),
                    beta_deg=self.beta_spin.value(),
                    L0=1.0, S=self.s_spin.value(),
                    A0=self.a0_spin.value(),
                    length=length,
                    direction=direction,
                    n_points=self.n_points_spin.value(),
                    n_streamwise=self.n_stream_spin.value(),
                    )
            except Exception:
                self._ma_max_ref = None

            # Enable UI
            self.canvas_stack.setCurrentIndex(1)
            self.btn_ma_min.setEnabled(True)
            self.btn_vmn.setEnabled(True)
            self.btn_ma_max.setEnabled(True)
            self.view_mode = 'vmn'
            self.btn_vmn.setChecked(True)
            self.btn_ma_min.setChecked(False)
            self.btn_ma_max.setChecked(False)

            self._update_3d_plot()
            self.canvas_mach.update_profile(self.waverider)
            self.canvas_cone.update_profile(self.waverider)

            # Derived quantities
            ca = self.waverider.cone_angles_deg
            le = self.waverider.leading_edge
            width = le[:, 2].max() if len(le) > 0 else 0
            height = abs(le[:, 1].min()) if len(le) > 0 else 0
            self.derived_label.setText(
                f"Cone angles: {ca.min():.1f} - {ca.max():.1f} deg\n"
                f"Vehicle half-width: {width:.3f} m\n"
                f"Vehicle height: {height:.3f} m\n"
                f"Stations: {len(self.waverider.upper_surface_streams)} US + "
                f"{len(self.waverider.lower_surface_streams)} LS")

            n_lower = len(self.waverider.lower_surface_streams)
            n_upper = len(self.waverider.upper_surface_streams)
            self.status_label.setText(
                f"Generated: {n_upper} upper + {n_lower} lower streams, "
                f"L={self.waverider.length:.3f}m")
            self.status_label.setStyleSheet("color: green")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red")
            import traceback
            traceback.print_exc()

    # ==============================================================
    #  Visualisation
    # ==============================================================

    def _update_3d_plot(self):
        if self.waverider is None:
            return
        half_only = self.half_vehicle_check.isChecked()

        if self.view_mode == 'ma_min':
            if self._ma_min_ref is not None:
                self.canvas_3d.plot_waverider(
                    self._ma_min_ref, half_only=half_only,
                    title_prefix=f'Ma={self.ma_min_spin.value():.0f} (reference)')
            else:
                self.canvas_3d.plot_waverider(
                    self.waverider, half_only=half_only,
                    title_prefix='Ma_min ref (not available)')
        elif self.view_mode == 'ma_max':
            if self._ma_max_ref is not None:
                self.canvas_3d.plot_waverider(
                    self._ma_max_ref, half_only=half_only,
                    title_prefix=f'Ma={self.ma_max_spin.value():.0f} (reference)')
            else:
                self.canvas_3d.plot_waverider(
                    self.waverider, half_only=half_only,
                    title_prefix='Ma_max ref (not available)')
        else:
            self.canvas_3d.plot_waverider(
                self.waverider, half_only=half_only,
                title_prefix='VMN Waverider')

    # ==============================================================
    #  Validation
    # ==============================================================

    def _auto_calculate_beta(self):
        ma_min = self.ma_min_spin.value()
        mu = math.degrees(math.asin(1.0 / max(ma_min, 1.01)))
        self.beta_spin.setValue(round(mu * 1.2, 1))

    def _update_beta_hint(self):
        ma_min = self.ma_min_spin.value()
        mu = math.degrees(math.asin(1.0 / max(ma_min, 1.01)))
        beta = self.beta_spin.value()

        if beta < mu + 0.1:
            self.beta_hint_label.setText(
                f"WARNING: beta={beta:.1f} < Mach angle {mu:.1f}! "
                f"Shock detached!")
            self.beta_hint_label.setStyleSheet(
                "color: #EF4444; font-size: 10px; font-weight: bold;")
            self.generate_btn.setEnabled(False)
        else:
            self.beta_hint_label.setText(
                f"Mach angle {mu:.1f} | "
                f"Rec: {mu*1.2:.1f} ({mu+0.5:.1f}-{mu*1.5:.1f})")
            self.beta_hint_label.setStyleSheet("color: #4ADE80; font-size: 10px;")
            self.generate_btn.setEnabled(True)

    # ==============================================================
    #  Export
    # ==============================================================

    def export_stl(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save STL", "vmn_waverider.stl", "STL Files (*.stl)")
        if filename:
            try:
                self.waverider.export_stl(filename)
                self.status_label.setText(f"Exported to {filename}")
                QMessageBox.information(self, "Success", f"STL exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed:\n{str(e)}")

    def export_tri(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save TRI", "vmn_waverider.tri", "TRI Files (*.tri)")
        if filename:
            try:
                self.waverider.export_tri(filename)
                self.status_label.setText(f"Exported to {filename}")
                QMessageBox.information(self, "Success", f"TRI exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed:\n{str(e)}")

    def export_step(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save STEP", "vmn_waverider.step", "STEP Files (*.step)")
        if not filename:
            return
        try:
            sides = 'left' if self.half_vehicle_check.isChecked() else 'both'
            self.waverider.to_CAD(
                sides=sides,
                export=True,
                filename=filename,
                scale=1000.0,
            )
            self.status_label.setText(f"Exported to {filename}")
            QMessageBox.information(self, "Success", f"STEP exported to:\n{filename}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Export Failed",
                f"Failed to export STEP:\n{str(e)}\n\n"
                f"See console for full traceback.")

    # ---- Settings save/load -------------------------------------------

    def get_params_dict(self):
        return {
            'ma_min': self.ma_min_spin.value(),
            'ma_max': self.ma_max_spin.value(),
            'beta': self.beta_spin.value(),
            'length': self.length_spin.value(),
            's': self.s_spin.value(),
            'a0': self.a0_spin.value(),
            'direction': self.direction_combo.currentIndex(),
            'n_points': self.n_points_spin.value(),
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

        _s(self.ma_min_spin, d.get('ma_min'))
        _s(self.ma_max_spin, d.get('ma_max'))
        _s(self.beta_spin, d.get('beta'))
        _s(self.length_spin, d.get('length'))
        _s(self.s_spin, d.get('s'))
        _s(self.a0_spin, d.get('a0'))
        _s(self.direction_combo, d.get('direction'))
        _s(self.n_points_spin, d.get('n_points'))
        _s(self.n_stream_spin, d.get('n_streamwise'))
        _s(self.half_vehicle_check, d.get('half_vehicle'))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    tab = VMNWaveriderTab()
    tab.setWindowTitle("VMN Waverider Tab (Standalone Test)")
    tab.resize(1200, 800)
    tab.show()
    sys.exit(app.exec_())
