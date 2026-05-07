"""Hybrid Waverider Tab — GOC blending between varying and uniform shock curvature."""

import math
import sys
import os
import numpy as np

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QGridLayout, QSlider,
                             QDoubleSpinBox, QSpinBox, QCheckBox, QFileDialog,
                             QMessageBox, QSplitter, QApplication, QScrollArea,
                             QComboBox, QTabWidget, QButtonGroup, QStackedWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from waverider_generator.hybrid_generator import GOCWaverider


def blend_h(z, z_max, h0, h1, p):
    """Module-level shim for blend profile preview (before waverider is generated)."""
    if z_max < 1e-10:
        return float(h0)
    t = np.clip(z / z_max, 0.0, 1.0)
    h = h0 * (1.0 - t) ** p + h1 * t ** p
    return float(np.clip(h, 0.0, 1.0))

try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False

# Blend presets: (name, h0, h1, p)
# h=1 → full OC (varying curvature), h=0 → uniform curvature
PRESETS = [
    ("Full OC             (h: 1.0->1.0)",      1.0, 1.0, 1.0),
    ("Uniform Curvature   (h: 0.0->0.0)",      0.0, 0.0, 1.0),
    ("OC Root -> Uniform Tip (h: 1->0)",       1.0, 0.0, 1.0),
    ("Uniform Root -> OC Tip (h: 0->1)",       0.0, 1.0, 1.0),
    ("OC Core + Uniform Tips (h: 1->0, p=2)",  1.0, 0.0, 2.0),
    ("Uniform Core + OC Tips (h: 0->1, p=2)",  0.0, 1.0, 2.0),
    ("Balanced             (h: 0.5->0.5)",     0.5, 0.5, 1.0),
    ("Sharp OC -> Uniform  (h: 1->0, p=4)",    1.0, 0.0, 4.0),
]


# ══════════════════════════════════════════════════════════════
#  Canvas classes
# ══════════════════════════════════════════════════════════════

class HybridCanvas3D(FigureCanvas):
    """3D matplotlib canvas — matches SHADOW waverider tab style."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        self._info_text = None

    def plot_waverider(self, wr, half_only=False, show_upper=True,
                       show_lower=True, show_le=True, show_info=True,
                       title_prefix='Hybrid Waverider'):
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

        upper = wr.upper_surface
        lower = wr.lower_surface

        if show_upper and upper.shape[0] > 1:
            self.ax.plot_surface(
                upper[:, :, 2], upper[:, :, 0], upper[:, :, 1],
                color='steelblue', alpha=0.4, linewidth=0, antialiased=True, shade=True)
        if show_lower and lower.shape[0] > 1:
            self.ax.plot_surface(
                lower[:, :, 2], lower[:, :, 0], lower[:, :, 1],
                color='indianred', alpha=0.4, linewidth=0, antialiased=True, shade=True)

        if not half_only:
            if show_upper and upper.shape[0] > 1:
                self.ax.plot_surface(
                    -upper[:, :, 2], upper[:, :, 0], upper[:, :, 1],
                    color='steelblue', alpha=0.4, linewidth=0, antialiased=True, shade=True)
            if show_lower and lower.shape[0] > 1:
                self.ax.plot_surface(
                    -lower[:, :, 2], lower[:, :, 0], lower[:, :, 1],
                    color='indianred', alpha=0.4, linewidth=0, antialiased=True, shade=True)

        legend_elements = []
        if show_upper:
            legend_elements.append(Patch(facecolor='steelblue', alpha=0.4, label='Upper Surface'))
        if show_lower:
            legend_elements.append(Patch(facecolor='indianred', alpha=0.4, label='Lower Surface'))

        if show_le and hasattr(wr, 'leading_edge'):
            le = wr.leading_edge
            self.ax.plot(le[:, 2], le[:, 0], le[:, 1], 'k-', linewidth=2.5)
            if not half_only:
                self.ax.plot(-le[:, 2], le[:, 0], le[:, 1], 'k-', linewidth=2.5)
            legend_elements.append(Line2D([0], [0], color='black', linewidth=2.5, label='Leading Edge'))

        mach = getattr(wr, 'M_inf', getattr(wr, 'mach', 0))
        beta = getattr(wr, 'beta', getattr(wr, 'shock_angle', 0))
        self.ax.set_xlabel('Z (Span)', color='#FFFFFF')
        self.ax.set_ylabel('X (Streamwise)', color='#FFFFFF')
        self.ax.set_zlabel('Y (Vertical)', color='#FFFFFF')
        self.ax.set_title(f'{title_prefix} (M={mach:.1f}, beta={beta:.1f} deg)', color='#FFFFFF')
        self.ax.tick_params(colors='#888888')
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper left')
        self._set_axes_equal()

        if show_info:
            self._draw_info_panel(wr, title_prefix)
        self.fig.tight_layout()
        self.draw()

    def _draw_info_panel(self, wr, title='HYBRID WAVERIDER'):
        mach = getattr(wr, 'M_inf', getattr(wr, 'mach', 0))
        beta_oc = getattr(wr, 'beta_OC', getattr(wr, 'beta', 0))
        beta_cd = getattr(wr, 'beta_CD', beta_oc)
        length = getattr(wr, 'length', 0)
        width = getattr(wr, 'width', 0)
        height = getattr(wr, 'height', 0)
        lines = [title]
        lines.append(f"  Mach           {mach:.1f}")
        lines.append(f"  beta OC        {beta_oc:.1f} deg")
        if abs(beta_cd - beta_oc) > 0.01:
            lines.append(f"  beta CD        {beta_cd:.1f} deg")
        if hasattr(wr, 'h0'):
            lines.append(f"  h root         {wr.h0:.2f}")
            lines.append(f"  h tip          {wr.h1:.2f}")
        if hasattr(wr, 'x_t'):
            lines.append(f"  x_t            {wr.x_t:.2f}")
            lines.append(f"  dx_t           {wr.dx_t:.2f}")
        lines.append(f"  Length          {length:.4f} m")

        # Volume
        try:
            from waverider_gui import calculate_waverider_volume
            vol = calculate_waverider_volume(wr)
            lines.append(f"  Volume          {vol:.6f} m³")
        except Exception:
            vol = 0.0

        # Planform area
        try:
            from reference_area_calculator import calculate_planform_area_from_waverider
            a_plan, _ = calculate_planform_area_from_waverider(wr)
            lines.append(f"  Planform Area   {a_plan:.6f} m²")
        except Exception:
            a_plan = 0.0

        # Volumetric efficiency: V^(2/3) / A_planform
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
            limits = np.array([self.ax.get_xlim3d(), self.ax.get_ylim3d(), self.ax.get_zlim3d()])
            center = np.mean(limits, axis=1)
            radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
            self.ax.set_xlim3d([center[0] - radius, center[0] + radius])
            self.ax.set_ylim3d([center[1] - radius, center[1] + radius])
            self.ax.set_zlim3d([center[2] - radius, center[2] + radius])
        except Exception:
            pass


class BlendProfileCanvas(FigureCanvas):
    """2D plot showing alpha(z) blending profile."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 3))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._plot_default()

    def _plot_default(self):
        self.ax.clear()
        z = np.linspace(0, 2.0, 100)
        alpha = np.linspace(1.0, 0.0, 100)
        self._draw_profile(z, alpha)

    def update_profile(self, width, alpha_root, alpha_tip, blend_exp,
                        x1=0.0):
        self.ax.clear()
        z = np.linspace(0, width, 100)
        alpha = np.array([blend_h(zi, width, alpha_root, alpha_tip, blend_exp)
                          for zi in z])
        self._draw_profile(z, alpha, width, x1)

    def _draw_profile(self, z, alpha, width=None, x1=0.0):
        self.ax.fill_between(z, alpha, 1.0, alpha=0.15, color='blue', label='OC')
        self.ax.fill_between(z, 0.0, alpha, alpha=0.15, color='orange', label='Uniform curvature')
        self.ax.plot(z, alpha, 'b-', linewidth=2)

        # Flat region boundary
        if width is not None and x1 > 0.001:
            flat_z = x1 * width
            if flat_z > 0 and flat_z < z[-1]:
                self.ax.axvline(x=flat_z, color='white', linestyle='--',
                                linewidth=1.0, alpha=0.6)
                self.ax.text(flat_z * 0.5, 0.95, 'flat\nregion',
                             ha='center', va='top', color='white',
                             fontsize=8, alpha=0.7)
                self.ax.text(flat_z + (z[-1] - flat_z) * 0.15, 0.95,
                             'blended\nregion', ha='center', va='top',
                             color='white', fontsize=8, alpha=0.7)

        self.ax.set_xlabel('Spanwise position z (m)')
        self.ax.set_ylabel('Blend coeff alpha')
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_title('Spanwise Blending Profile')
        self.ax.legend(loc='center right')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


class CrossSectionCanvas(FigureCanvas):
    """2D cross-section comparison: Hybrid vs OC vs CD."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 3))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_title('Generate geometry to see cross-section preview')
        self.ax.set_xlabel('Z (span, m)')
        self.ax.set_ylabel('Y (vertical, m)')
        self.fig.tight_layout()

    def plot_comparison(self, wr, cd_reference=None, oc_reference=None):
        """Plot base-plane (trailing edge) Y-Z cross-section for Hybrid, OC ref, CD ref."""
        self.ax.clear()
        if wr is None:
            self.ax.set_title('Generate geometry to see cross-section preview')
            self.draw()
            return
        self._cd_reference = cd_reference
        self._oc_reference = oc_reference

        def _get_te_profile(obj):
            """Extract TE cross-section: z vs y for all streams."""
            z_upper, y_upper, z_lower, y_lower = [], [], [], []
            for s in obj.upper_surface_streams:
                if s.shape[0] >= 2:
                    z_upper.append(s[-1, 2])
                    y_upper.append(s[-1, 1])
            for s in obj.lower_surface_streams:
                if s.shape[0] >= 2:
                    z_lower.append(s[-1, 2])
                    y_lower.append(s[-1, 1])
            return (np.array(z_upper), np.array(y_upper),
                    np.array(z_lower), np.array(y_lower))

        # Hybrid — filled cross-section
        zu, yu, zl, yl = _get_te_profile(wr)
        # Close the polygon: upper → tip → lower reversed → back to start
        z_poly = np.concatenate([zu, zl[::-1], [zu[0]]])
        y_poly = np.concatenate([yu, yl[::-1], [yu[0]]])
        self.ax.fill(z_poly, y_poly, alpha=0.15, color='#2196F3')
        self.ax.plot(zu, yu, 'b-', linewidth=2, label='Hybrid (upper)')
        self.ax.plot(zl, yl, 'b-', linewidth=1.5, linestyle='-',
                     label='Hybrid (lower)')

        # OC reference
        oc_ref = getattr(self, '_oc_reference', None)
        if oc_ref is not None:
            zu_oc, yu_oc, zl_oc, yl_oc = _get_te_profile(oc_ref)
            self.ax.plot(zu_oc, yu_oc, '--', color='grey', linewidth=1.2,
                         label='OC ref (upper)')
            self.ax.plot(zl_oc, yl_oc, '--', color='grey', linewidth=1.2)

        # CD / Uniform reference
        cd_ref = getattr(self, '_cd_reference', None)
        if cd_ref is not None:
            zu_cd, yu_cd, zl_cd, yl_cd = _get_te_profile(cd_ref)
            self.ax.plot(zu_cd, yu_cd, '--', color='#FF9800', linewidth=1.2,
                         label='Uniform ref (upper)')
            self.ax.plot(zl_cd, yl_cd, '--', color='#FF9800', linewidth=1.2)

        # Design shockwave Bézier profile at base plane
        if hasattr(wr, '_get_augmented_sw_arrays'):
            try:
                z_sw, y_local_sw = wr._get_augmented_sw_arrays()
                y_sw = np.array([wr._local_to_global(float(y))
                                 for y in y_local_sw])
                self.ax.plot(z_sw, y_sw, 'r-', linewidth=1.5, alpha=0.7,
                             label='Shockwave')
            except Exception:
                pass

        self.ax.set_xlabel('Z (span, m)')
        self.ax.set_ylabel('Y (vertical, m)')
        self.ax.set_title('Base-plane Cross-section (x = L)')
        self.ax.set_aspect('equal')
        self.ax.legend(loc='upper right', fontsize=7)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


# ══════════════════════════════════════════════════════════════
#  Main tab widget
# ══════════════════════════════════════════════════════════════

class HybridWaveriderTab(QWidget):
    """GUI tab for hybrid waverider design."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.waverider = None
        self.view_mode = 'hybrid'  # 'oc' | 'cd' | 'hybrid'
        self._updating = False
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Horizontal)

        # ── Left panel ────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(300)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(self._create_preset_group())
        left_layout.addWidget(self._create_shared_group())
        left_layout.addWidget(self._create_oc_group())
        # CD (Shadow) group removed — GOC blends radius of curvature, no Shadow params needed
        left_layout.addWidget(self._create_blend_group())
        left_layout.addWidget(self._create_dome_group())

        # Generate button
        self.generate_btn = QPushButton("Generate Hybrid Waverider")
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 8px; font-size: 14px; }")
        self.generate_btn.clicked.connect(self.generate_waverider)
        left_layout.addWidget(self.generate_btn)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label)

        self.blend_stats_label = QLabel("")
        self.blend_stats_label.setAlignment(Qt.AlignCenter)
        self.blend_stats_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        left_layout.addWidget(self.blend_stats_label)

        left_layout.addWidget(self._create_export_group())
        left_layout.addStretch()
        left_scroll.setWidget(left_widget)
        main_splitter.addWidget(left_scroll)

        # ── Right panel ───────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # View mode toggle buttons
        toggle_layout = QHBoxLayout()
        self.btn_oc = QPushButton("OC")
        self.btn_hybrid = QPushButton("Hybrid")
        self.btn_cd = QPushButton("Uniform")
        for btn in [self.btn_oc, self.btn_hybrid, self.btn_cd]:
            btn.setCheckable(True)
            btn.setEnabled(False)
            btn.setStyleSheet(
                "QPushButton { padding: 4px 12px; } "
                "QPushButton:checked { background-color: #D97706; color: white; font-weight: bold; }")
        self.btn_hybrid.setChecked(True)
        self.btn_oc.clicked.connect(lambda: self._set_view_mode('oc'))
        self.btn_hybrid.clicked.connect(lambda: self._set_view_mode('hybrid'))
        self.btn_cd.clicked.connect(lambda: self._set_view_mode('cd'))
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.btn_oc)
        toggle_layout.addWidget(self.btn_hybrid)
        toggle_layout.addWidget(self.btn_cd)
        toggle_layout.addStretch()
        right_layout.addLayout(toggle_layout)

        # 3D canvas with placeholder
        self.canvas_3d = HybridCanvas3D()
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)

        self.placeholder_label = QLabel(
            "Click 'Generate Hybrid Waverider'\nto create geometry")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet(
            "color: grey; font-style: italic; font-size: 12px;")

        self.canvas_stack = QStackedWidget()
        self.canvas_stack.addWidget(self.placeholder_label)  # index 0
        canvas_widget = QWidget()
        cl = QVBoxLayout(canvas_widget)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.toolbar_3d)
        cl.addWidget(self.canvas_3d)
        self.canvas_stack.addWidget(canvas_widget)  # index 1
        self.canvas_stack.setCurrentIndex(0)

        # Bottom tabs: blend profile + cross-section
        self.bottom_tabs = QTabWidget()
        self.canvas_blend = BlendProfileCanvas()
        self.bottom_tabs.addTab(self.canvas_blend, "Blend Profile")
        self.canvas_xsec = CrossSectionCanvas()
        self.bottom_tabs.addTab(self.canvas_xsec, "Cross-section Preview")

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

        # Initial updates
        self._update_beta_hint()
        self._update_constraint_hints()
        self._update_blend_plot()

    # ══════════════════════════════════════════════════════
    #  Preset group
    # ══════════════════════════════════════════════════════

    def _create_preset_group(self):
        group = QGroupBox("Presets")
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Quick preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("-- Select preset --")
        for name, _, _, _ in PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        layout.addWidget(self.preset_combo, 1)
        group.setLayout(layout)
        return group

    def _apply_preset(self, index):
        if index <= 0:
            return
        _, root, tip, exp = PRESETS[index - 1]
        self.alpha_root_spin.blockSignals(True)
        self.alpha_root_slider.blockSignals(True)
        self.alpha_tip_spin.blockSignals(True)
        self.alpha_tip_slider.blockSignals(True)
        self.blend_exp_spin.blockSignals(True)
        self.blend_exp_slider.blockSignals(True)

        self.alpha_root_spin.setValue(root)
        self.alpha_root_slider.setValue(int(root * 100))
        self.alpha_tip_spin.setValue(tip)
        self.alpha_tip_slider.setValue(int(tip * 100))
        self.blend_exp_spin.setValue(exp)
        self.blend_exp_slider.setValue(int(exp * 10))

        self.alpha_root_spin.blockSignals(False)
        self.alpha_root_slider.blockSignals(False)
        self.alpha_tip_spin.blockSignals(False)
        self.alpha_tip_slider.blockSignals(False)
        self.blend_exp_spin.blockSignals(False)
        self.blend_exp_slider.blockSignals(False)

        self._update_blend_plot()

    # ══════════════════════════════════════════════════════
    #  Parameter groups
    # ══════════════════════════════════════════════════════

    def _create_shared_group(self):
        group = QGroupBox("Shared Parameters")
        layout = QGridLayout()
        row = 0

        # Mach
        layout.addWidget(QLabel("Mach Number:"), row, 0)
        self.m_inf_spin = QDoubleSpinBox()
        self.m_inf_spin.setRange(3.0, 15.0)
        self.m_inf_spin.setValue(7.0)
        self.m_inf_spin.setSingleStep(0.1)
        self.m_inf_spin.setDecimals(6)
        self.m_inf_spin.setToolTip(
            "Freestream Mach number.\nShared between OC and CD generators.\nRange: 3.0 - 15.0")
        layout.addWidget(self.m_inf_spin, row, 1)
        row += 1

        # Shock angle + Auto
        layout.addWidget(QLabel("Shock Angle (deg):"), row, 0)
        beta_row = QHBoxLayout()
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(5.0, 45.0)
        self.beta_spin.setValue(12.0)
        self.beta_spin.setSingleStep(0.5)
        self.beta_spin.setDecimals(6)
        self.beta_spin.setToolTip(
            "Conical shock half-angle in degrees.\n"
            "Must be above Mach angle for shock attachment.\nUse 'Auto' for optimal value.")
        beta_row.addWidget(self.beta_spin)
        self.auto_beta_btn = QPushButton("Auto")
        self.auto_beta_btn.setFixedWidth(50)
        self.auto_beta_btn.setToolTip("Auto-calculate recommended shock angle for current Mach")
        self.auto_beta_btn.clicked.connect(self._auto_calculate_beta)
        beta_row.addWidget(self.auto_beta_btn)
        bw = QWidget(); bw.setLayout(beta_row)
        layout.addWidget(bw, row, 1)
        row += 1

        # Beta hint
        self.beta_hint_label = QLabel("")
        self.beta_hint_label.setStyleSheet("color: #888888; font-size: 10px;")
        self.beta_hint_label.setWordWrap(True)
        layout.addWidget(self.beta_hint_label, row, 0, 1, 2)
        row += 1

        # CD Shock Angle (NEW — for GOC)
        layout.addWidget(QLabel("CD Shock Angle (deg):"), row, 0)
        cd_beta_row = QHBoxLayout()
        self.beta_cd_spin = QDoubleSpinBox()
        self.beta_cd_spin.setRange(5.0, 45.0)
        self.beta_cd_spin.setValue(12.0)
        self.beta_cd_spin.setSingleStep(0.5)
        self.beta_cd_spin.setDecimals(6)
        self.beta_cd_spin.setToolTip(
            "Shock angle for the cone-derived limit (h=0).\n"
            "Independent of OC shock angle.\n"
            "When h=0 everywhere, the vehicle uses this shock angle.")
        cd_beta_row.addWidget(self.beta_cd_spin)
        self.auto_beta_cd_btn = QPushButton("Auto")
        self.auto_beta_cd_btn.setFixedWidth(50)
        self.auto_beta_cd_btn.setToolTip(
            "Auto-calculate recommended CD shock angle for current Mach\n"
            "(uses SHADOW/cone-derived empirical formula)")
        self.auto_beta_cd_btn.clicked.connect(self._auto_calculate_beta_cd)
        cd_beta_row.addWidget(self.auto_beta_cd_btn)
        cd_bw = QWidget(); cd_bw.setLayout(cd_beta_row)
        layout.addWidget(cd_bw, row, 1)
        row += 1

        # CD Beta hint
        self.beta_cd_hint_label = QLabel("")
        self.beta_cd_hint_label.setStyleSheet("color: #888888; font-size: 10px;")
        self.beta_cd_hint_label.setWordWrap(True)
        layout.addWidget(self.beta_cd_hint_label, row, 0, 1, 2)
        row += 1

        # Height (input — matches generator convention)
        layout.addWidget(QLabel("Height (m):"), row, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.01, 20.0)
        self.height_spin.setValue(round(3.0 * np.tan(12.0 * np.pi / 180), 3))
        self.height_spin.setSingleStep(0.01)
        self.height_spin.setDecimals(6)
        self.height_spin.setToolTip(
            "Waverider height (vertical extent of base plane) in metres.\n"
            "Length is derived: length = height / tan(beta).\n"
            "Changing height updates length automatically.")
        layout.addWidget(self.height_spin, row, 1)
        row += 1

        # Length (derived: length = height / tan(beta), read-only display)
        layout.addWidget(QLabel("Length (m):"), row, 0)
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.0, 999.0)
        init_height = round(3.0 * np.tan(12.0 * np.pi / 180), 3)
        self.length_spin.setValue(round(init_height / np.tan(12.0 * np.pi / 180), 3))
        self.length_spin.setSingleStep(0.01)
        self.length_spin.setDecimals(6)
        self.length_spin.setReadOnly(True)
        self.length_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.length_spin.setToolTip(
            "Derived: length = height / tan(beta).\n"
            "Not editable — change Height or Shock Angle instead.")
        self.length_spin.setStyleSheet("background-color: #333333; color: #999999;")
        layout.addWidget(self.length_spin, row, 1)
        row += 1

        # Half-Width
        layout.addWidget(QLabel("Half-Width (m):"), row, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 10.0)
        self.width_spin.setValue(2.0)
        self.width_spin.setSingleStep(0.1)
        self.width_spin.setDecimals(6)
        self.width_spin.setToolTip("Vehicle half-span.")
        layout.addWidget(self.width_spin, row, 1)
        row += 1

        # Osculating Planes
        layout.addWidget(QLabel("Osculating Planes:"), row, 0)
        self.n_planes_spin = QSpinBox()
        self.n_planes_spin.setRange(10, 50)
        self.n_planes_spin.setValue(20)
        self.n_planes_spin.setToolTip("Number of osculating planes (spanwise resolution).")
        layout.addWidget(self.n_planes_spin, row, 1)
        row += 1

        # Streamwise Points
        layout.addWidget(QLabel("Streamwise Points:"), row, 0)
        self.n_stream_spin = QSpinBox()
        self.n_stream_spin.setRange(10, 50)
        self.n_stream_spin.setValue(20)
        self.n_stream_spin.setToolTip("Number of streamwise points per streamline.")
        layout.addWidget(self.n_stream_spin, row, 1)

        group.setLayout(layout)

        # Height and beta update derived length
        self.height_spin.valueChanged.connect(self._on_height_changed)
        self.beta_spin.valueChanged.connect(self._on_beta_changed)

        # Other connections
        self.m_inf_spin.valueChanged.connect(self._update_beta_hint)
        self.beta_cd_spin.valueChanged.connect(self._update_beta_hint)
        self.m_inf_spin.valueChanged.connect(self._update_constraint_hints)
        self.width_spin.valueChanged.connect(self._update_constraint_hints)
        self.width_spin.valueChanged.connect(self._update_blend_plot)
        return group

    def _create_oc_group(self):
        group = QGroupBox("Osculating Cone Parameters")
        layout = QGridLayout()
        tips = [
            "X1: Flat region fraction of half-span.\n0.0 = no flat region.",
            "X2: LE height fraction.\nConstrained by OC design condition.",
            "X3: Upper surface shape parameter.",
            "X4: Upper surface shape parameter (secondary).",
        ]
        for i, (name, default, lo, hi, step) in enumerate([
            ("X1 (Flat Region):", 0.0, 0.0, 0.99, 0.0001),
            ("X2 (LE Height):", 0.2, 0.0, 1.0, 0.0001),
            ("X3 (Upper Surface):", 0.5, 0.0, 1.0, 0.0001),
            ("X4 (Upper Surface):", 0.5, 0.0, 1.0, 0.0001),
        ]):
            layout.addWidget(QLabel(name), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(default)
            spin.setSingleStep(step)
            spin.setDecimals(6)
            spin.setToolTip(tips[i])
            layout.addWidget(spin, i, 1)
            setattr(self, f'x{i+1}_spin', spin)

        self.oc_constraint_label = QLabel("")
        self.oc_constraint_label.setWordWrap(True)
        self.oc_constraint_label.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(self.oc_constraint_label, 4, 0, 1, 2)
        group.setLayout(layout)
        self.x1_spin.valueChanged.connect(self._update_constraint_hints)
        self.x1_spin.valueChanged.connect(self._update_blend_plot)
        self.x2_spin.valueChanged.connect(self._update_constraint_hints)
        return group

    def _create_cd_group(self):
        group = QGroupBox("Uniform curvature Parameters")
        layout = QGridLayout()
        params = [
            ("A3:", 'a3_spin', -100, 100, 0.0, 5.0, 1,
             "3rd-order LE polynomial coeff.\nControls planform sweep curvature."),
            ("A2:", 'a2_spin', -50, 50, -2.0, 0.5, 2,
             "2nd-order LE polynomial coeff.\nTypically negative for aft-swept LE."),
            ("A0:", 'a0_spin', -1.0, 0.0, -0.15, 0.01, 3,
             "0th-order LE offset.\nMust be negative (LE below freestream)."),
            ("Top Sfc Ctrl:", 'cd_top_spin', 0.0, 100.0, 0.0, 1.0, 1,
             "Upper surface thickness control.\n0 = flat top."),
        ]
        for row, (label, attr, lo, hi, default, step, dec, tip) in enumerate(params):
            layout.addWidget(QLabel(label), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(default)
            spin.setSingleStep(step)
            spin.setDecimals(6)
            spin.setToolTip(tip)
            layout.addWidget(spin, row, 1)
            setattr(self, attr, spin)

        # Note: CD length is taken from the shared Length spinbox
        cd_note = QLabel("(CD length = shared Length above)")
        cd_note.setStyleSheet("color: #888888; font-size: 9px; font-style: italic;")
        layout.addWidget(cd_note, len(params), 0, 1, 2)

        self.cd_warning_label = QLabel("")
        self.cd_warning_label.setWordWrap(True)
        self.cd_warning_label.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(self.cd_warning_label, len(params) + 1, 0, 1, 2)
        group.setLayout(layout)
        self.a3_spin.valueChanged.connect(self._update_constraint_hints)
        self.a2_spin.valueChanged.connect(self._update_constraint_hints)
        return group

    def _create_blend_group(self):
        group = QGroupBox("Blending Parameters")
        layout = QGridLayout()

        for row, (label, attr_slider, attr_spin, s_lo, s_hi, s_def,
                  sp_lo, sp_hi, sp_def, sp_step, scale, tip) in enumerate([
            ("OC at Root:", 'alpha_root_slider', 'alpha_root_spin',
             0, 100, 100, 0.0, 1.0, 1.0, 0.01, 0.01,
             "1.0 = pure OC at centreline. 0.0 = pure Uniform curvature."),
            ("OC at Tip:", 'alpha_tip_slider', 'alpha_tip_spin',
             0, 100, 0, 0.0, 1.0, 0.0, 0.01, 0.01,
             "0.0 = uniform curvature at tip. 1.0 = pure OC at tip."),
            ("Blend Exp:", 'blend_exp_slider', 'blend_exp_spin',
             1, 50, 10, 0.1, 5.0, 1.0, 0.1, 0.1,
             "1.0 = linear. <1 = faster near root. >1 = faster near tip."),
        ]):
            layout.addWidget(QLabel(label), row, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(s_lo, s_hi)
            slider.setValue(s_def)
            layout.addWidget(slider, row, 1)
            spin = QDoubleSpinBox()
            spin.setRange(sp_lo, sp_hi)
            spin.setValue(sp_def)
            spin.setSingleStep(sp_step)
            spin.setDecimals(6)
            spin.setToolTip(tip)
            layout.addWidget(spin, row, 2)
            setattr(self, attr_slider, slider)
            setattr(self, attr_spin, spin)
            self._link_slider_spinbox(slider, spin, scale)

        # Streamwise transition params (NEW for GOC)
        row_base = 3

        layout.addWidget(QLabel("Transition Centre (x/L):"), row_base, 0)
        self.xt_spin = QDoubleSpinBox()
        self.xt_spin.setRange(0.0, 1.0)
        self.xt_spin.setValue(0.5)
        self.xt_spin.setSingleStep(0.05)
        self.xt_spin.setDecimals(6)
        self.xt_spin.setToolTip(
            "Streamwise position of OC/CD transition midpoint.\n"
            "0 = nose, 1 = base.\n"
            "At this x-position, h transitions between CD and OC.")
        layout.addWidget(self.xt_spin, row_base, 1, 1, 2)
        self.xt_spin.valueChanged.connect(self._update_blend_plot)
        row_base += 1

        layout.addWidget(QLabel("Transition Width (x/L):"), row_base, 0)
        self.dxt_spin = QDoubleSpinBox()
        self.dxt_spin.setRange(0.01, 1.0)
        self.dxt_spin.setValue(0.3)
        self.dxt_spin.setSingleStep(0.05)
        self.dxt_spin.setDecimals(6)
        self.dxt_spin.setToolTip(
            "Width of the streamwise transition zone.\n"
            "Small = sharp transition, Large = gradual.")
        layout.addWidget(self.dxt_spin, row_base, 1, 1, 2)
        self.dxt_spin.valueChanged.connect(self._update_blend_plot)
        row_base += 1

        # Advanced options (collapsed by default)
        self.per_plane_beta_check = QCheckBox("Per-plane beta correction")
        self.per_plane_beta_check.setChecked(False)
        self.per_plane_beta_check.setToolTip(
            "Apply first-order curvature correction to shock angle per plane.")
        layout.addWidget(self.per_plane_beta_check, row_base, 0, 1, 3)
        row_base += 1

        self.independent_profiles_check = QCheckBox("Independent nose/base profiles")
        self.independent_profiles_check.setChecked(False)
        self.independent_profiles_check.setToolTip(
            "Use separate h(z) profiles at nose and base.\n"
            "When unchecked, nose is always CD (h=0).")
        layout.addWidget(self.independent_profiles_check, row_base, 0, 1, 3)
        row_base += 1

        # h0_nose, h1_nose (only visible when independent profiles is checked)
        self.h0_nose_label = QLabel("h nose root:")
        self.h0_nose_spin = QDoubleSpinBox()
        self.h0_nose_spin.setRange(0.0, 1.0)
        self.h0_nose_spin.setValue(0.0)
        self.h0_nose_spin.setSingleStep(0.05)
        self.h0_nose_spin.setDecimals(6)
        self.h0_nose_spin.setToolTip(
            "Blend coefficient at the nose centreline.\n"
            "1.0 = pure OC at nose root, 0.0 = uniform curvature.\n"
            "Only used when 'Independent nose/base profiles' is enabled.")
        layout.addWidget(self.h0_nose_label, row_base, 0)
        layout.addWidget(self.h0_nose_spin, row_base, 1, 1, 2)
        row_base += 1

        self.h1_nose_label = QLabel("h nose tip:")
        self.h1_nose_spin = QDoubleSpinBox()
        self.h1_nose_spin.setRange(0.0, 1.0)
        self.h1_nose_spin.setValue(0.0)
        self.h1_nose_spin.setSingleStep(0.05)
        self.h1_nose_spin.setDecimals(6)
        self.h1_nose_spin.setToolTip(
            "Blend coefficient at the nose wingtip.\n"
            "1.0 = pure OC at nose tip, 0.0 = uniform curvature.\n"
            "Only used when 'Independent nose/base profiles' is enabled.")
        layout.addWidget(self.h1_nose_label, row_base, 0)
        layout.addWidget(self.h1_nose_spin, row_base, 1, 1, 2)
        row_base += 1

        # Toggle nose spinbox visibility
        def _toggle_nose_params(checked):
            self.h0_nose_label.setVisible(checked)
            self.h0_nose_spin.setVisible(checked)
            self.h1_nose_label.setVisible(checked)
            self.h1_nose_spin.setVisible(checked)
        self.independent_profiles_check.toggled.connect(_toggle_nose_params)
        _toggle_nose_params(False)  # hidden by default

        info = QLabel("h=1.0 -> OC (varying) | h=0.0 -> CD (uniform)")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: grey; font-style: italic;")
        layout.addWidget(info, row_base, 0, 1, 3)
        group.setLayout(layout)
        return group

    def _link_slider_spinbox(self, slider, spinbox, scale):
        def slider_changed(val):
            spinbox.blockSignals(True)
            spinbox.setValue(val * scale)
            spinbox.blockSignals(False)
            self._update_blend_plot()
        def spinbox_changed(val):
            slider.blockSignals(True)
            slider.setValue(int(round(val / scale)))
            slider.blockSignals(False)
            self._update_blend_plot()
        slider.valueChanged.connect(slider_changed)
        spinbox.valueChanged.connect(spinbox_changed)

    # ══════════════════════════════════════════════════════
    #  Validation & hints
    # ══════════════════════════════════════════════════════

    # ── Three-way linking: length <-> height <-> beta ──

    def _on_height_changed(self, value):
        if self._updating:
            return
        self._updating = True
        try:
            beta_rad = self.beta_spin.value() * np.pi / 180
            tan_b = np.tan(beta_rad)
            if abs(tan_b) > 1e-10:
                self.length_spin.setValue(round(value / tan_b, 3))
        finally:
            self._updating = False
        self._update_constraint_hints()
        self._update_blend_plot()

    def _on_beta_changed(self, value):
        if self._updating:
            return
        self._updating = True
        try:
            # Keep height fixed, update length
            beta_rad = value * np.pi / 180
            tan_b = np.tan(beta_rad)
            if abs(tan_b) > 1e-10:
                current_height = self.height_spin.value()
                self.length_spin.setValue(round(current_height / tan_b, 3))
        finally:
            self._updating = False
        self._update_beta_hint()
        self._update_constraint_hints()

    def _optimal_oc_shock_angle(self, mach):
        """OC recommended shock angle from lookup table (same as OC Waverider tab)."""
        beta_table = {
            2.0: 40.0, 2.5: 34.0, 3.0: 26.5, 3.5: 23.5, 4.0: 21.0,
            4.5: 19.5, 5.0: 18.0, 5.5: 17.0, 6.0: 16.0, 7.0: 14.5,
            8.0: 13.5, 10.0: 12.0, 12.0: 11.0, 15.0: 10.0,
        }
        machs = sorted(beta_table.keys())
        if mach <= machs[0]:
            return beta_table[machs[0]]
        if mach >= machs[-1]:
            return beta_table[machs[-1]]
        for i in range(len(machs) - 1):
            if machs[i] <= mach <= machs[i + 1]:
                t = (mach - machs[i]) / (machs[i + 1] - machs[i])
                return beta_table[machs[i]] * (1 - t) + beta_table[machs[i + 1]] * t
        return beta_table[machs[-1]]

    def _optimal_cd_shock_angle(self, mach):
        """CD/SHADOW recommended shock angle (empirical mu * factor)."""
        mu = math.degrees(math.asin(1.0 / max(mach, 1.01)))
        if mach < 6:    factor = 1.3
        elif mach < 10: factor = 1.25
        elif mach < 15: factor = 1.2
        else:           factor = 1.15
        return mu * factor

    def _auto_calculate_beta(self):
        optimal = self._optimal_oc_shock_angle(self.m_inf_spin.value())
        self.beta_spin.setValue(round(optimal, 1))

    def _auto_calculate_beta_cd(self):
        optimal = self._optimal_cd_shock_angle(self.m_inf_spin.value())
        self.beta_cd_spin.setValue(round(optimal, 1))

    def _update_beta_hint(self):
        mach = self.m_inf_spin.value()
        mu = math.degrees(math.asin(1.0 / max(mach, 1.01)))

        # OC hint
        beta_oc = self.beta_spin.value()
        opt_oc = self._optimal_oc_shock_angle(mach)
        rec_lo_oc = max(mu + 0.5, opt_oc - 2.0)
        rec_hi_oc = opt_oc + 2.0
        if beta_oc < mu + 0.1:
            self.beta_hint_label.setText(
                f"WARNING: beta={beta_oc:.1f} < Mach angle {mu:.1f}! Shock detached!")
            self.beta_hint_label.setStyleSheet("color: #EF4444; font-size: 10px; font-weight: bold;")
            self.generate_btn.setEnabled(False)
        elif beta_oc < rec_lo_oc or beta_oc > rec_hi_oc:
            self.beta_hint_label.setText(
                f"OC: Mach angle {mu:.1f} | Rec: {opt_oc:.1f} ({rec_lo_oc:.1f}-{rec_hi_oc:.1f})")
            self.beta_hint_label.setStyleSheet("color: #F59E0B; font-size: 10px;")
            self._check_generate_enabled()
        else:
            self.beta_hint_label.setText(
                f"OC: Mach angle {mu:.1f} | Rec: {opt_oc:.1f} ({rec_lo_oc:.1f}-{rec_hi_oc:.1f})")
            self.beta_hint_label.setStyleSheet("color: #4ADE80; font-size: 10px;")
            self._check_generate_enabled()

        # CD hint
        beta_cd = self.beta_cd_spin.value()
        opt_cd = self._optimal_cd_shock_angle(mach)
        rec_lo_cd = max(mu + 0.5, opt_cd - 2.0)
        rec_hi_cd = opt_cd + 2.0
        if beta_cd < mu + 0.1:
            self.beta_cd_hint_label.setText(
                f"WARNING: beta={beta_cd:.1f} < Mach angle {mu:.1f}! Shock detached!")
            self.beta_cd_hint_label.setStyleSheet("color: #EF4444; font-size: 10px; font-weight: bold;")
        elif beta_cd < rec_lo_cd or beta_cd > rec_hi_cd:
            self.beta_cd_hint_label.setText(
                f"CD: Mach angle {mu:.1f} | Rec: {opt_cd:.1f} ({rec_lo_cd:.1f}-{rec_hi_cd:.1f})")
            self.beta_cd_hint_label.setStyleSheet("color: #F59E0B; font-size: 10px;")
        else:
            self.beta_cd_hint_label.setText(
                f"CD: Mach angle {mu:.1f} | Rec: {opt_cd:.1f} ({rec_lo_cd:.1f}-{rec_hi_cd:.1f})")
            self.beta_cd_hint_label.setStyleSheet("color: #4ADE80; font-size: 10px;")

    def _update_constraint_hints(self):
        W = self.width_spin.value()
        H = max(self.height_spin.value(), 0.01)
        X1 = self.x1_spin.value()
        X2 = self.x2_spin.value()

        denom = max((1.0 - X1) ** 4, 1e-12)
        lhs = X2 / denom
        rhs = (7.0 / 64.0) * (W / H) ** 4 * 0.9
        oc_valid = lhs < rhs

        if oc_valid:
            margin = (1.0 - lhs / max(rhs, 1e-12)) * 100
            max_x2 = rhs * denom
            self.oc_constraint_label.setText(
                f"OC condition OK (margin {margin:.0f}%) | Max X2 ~ {max_x2:.3f}")
            self.oc_constraint_label.setStyleSheet("color: #4ADE80; font-size: 10px;")
        else:
            max_x2 = rhs * denom
            self.oc_constraint_label.setText(
                f"WARNING: OC condition violated! ({lhs:.3f} >= {rhs:.3f})\n"
                f"Max X2 ~ {max_x2:.3f} or reduce height / increase width")
            self.oc_constraint_label.setStyleSheet("color: #EF4444; font-size: 10px; font-weight: bold;")

        # CD sweep metric removed — GOC doesn't use Shadow parameters

        self._check_generate_enabled()

    def _check_generate_enabled(self):
        """Enable Generate only if OC condition is met and beta above Mach angle."""
        mach = self.m_inf_spin.value()
        beta = self.beta_spin.value()
        mu = math.degrees(math.asin(1.0 / max(mach, 1.01)))
        X1 = self.x1_spin.value()
        X2 = self.x2_spin.value()
        W = self.width_spin.value()
        H = max(self.height_spin.value(), 0.01)
        denom = max((1.0 - X1) ** 4, 1e-12)
        lhs = X2 / denom
        rhs = (7.0 / 64.0) * (W / H) ** 4 * 0.9
        oc_ok = lhs < rhs
        beta_ok = beta > mu + 0.1
        self.generate_btn.setEnabled(oc_ok and beta_ok)

    def _create_dome_group(self):
        group = QGroupBox("Volume Loft (Dome)")
        layout = QGridLayout()

        layout.addWidget(QLabel("Dome Height (m):"), 0, 0)
        self.dome_height_spin = QDoubleSpinBox()
        self.dome_height_spin.setRange(0.0, 1.0)
        self.dome_height_spin.setValue(0.0)
        self.dome_height_spin.setSingleStep(0.005)
        self.dome_height_spin.setDecimals(3)
        self.dome_height_spin.setToolTip(
            "Max vertical offset added to upper surface at mid-chord.\n"
            "Increases internal volume without changing planform area.\n"
            "LE and TE stay pinned (shock attachment preserved).\n"
            "0 = no dome (flat upper surface).")
        layout.addWidget(self.dome_height_spin, 0, 1)

        layout.addWidget(QLabel("Spanwise Taper:"), 1, 0)
        self.dome_taper_spin = QDoubleSpinBox()
        self.dome_taper_spin.setRange(0.5, 5.0)
        self.dome_taper_spin.setValue(2.0)
        self.dome_taper_spin.setSingleStep(0.5)
        self.dome_taper_spin.setDecimals(1)
        self.dome_taper_spin.setToolTip(
            "Spanwise decay exponent for dome height.\n"
            "1 = linear taper (root to tip)\n"
            "2 = parabolic taper (default, smooth)\n"
            "4 = concentrated near root")
        layout.addWidget(self.dome_taper_spin, 1, 1)

        group.setLayout(layout)
        return group

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
        self.half_vehicle_check.setToolTip(
            "Export only the right half (positive Z).\nUseful for CFD with symmetry BC.")
        layout.addWidget(stl_btn, 0, 0)
        layout.addWidget(tri_btn, 0, 1)
        layout.addWidget(step_btn, 1, 0, 1, 2)
        layout.addWidget(self.half_vehicle_check, 2, 0, 1, 2)
        self.shock_surface_check = QCheckBox("Include shock surface")
        self.shock_surface_check.setToolTip("Export conical shock as separate STEP body")
        self.shock_surface_check.setEnabled(CADQUERY_AVAILABLE)
        layout.addWidget(self.shock_surface_check, 3, 0, 1, 2)
        group.setLayout(layout)
        return group

    # ══════════════════════════════════════════════════════
    #  View mode toggle
    # ══════════════════════════════════════════════════════

    def _set_view_mode(self, mode):
        self.view_mode = mode
        self.btn_oc.setChecked(mode == 'oc')
        self.btn_hybrid.setChecked(mode == 'hybrid')
        self.btn_cd.setChecked(mode == 'cd')
        self._update_3d_plot()

    # ══════════════════════════════════════════════════════
    #  Generation
    # ══════════════════════════════════════════════════════

    def generate_waverider(self):
        try:
            self.status_label.setText("Generating...")
            self.status_label.setStyleSheet("color: black")
            QApplication.processEvents()

            beta_oc = self.beta_spin.value()
            beta_cd = self.beta_cd_spin.value() if hasattr(self, 'beta_cd_spin') else beta_oc
            height = self.height_spin.value()

            x_t = self.xt_spin.value() if hasattr(self, 'xt_spin') else 0.5
            dx_t = self.dxt_spin.value() if hasattr(self, 'dxt_spin') else 0.3

            # Optional advanced params
            extra_kw = {}
            if hasattr(self, 'per_plane_beta_check') and self.per_plane_beta_check.isChecked():
                extra_kw['use_per_plane_beta'] = True
            if hasattr(self, 'independent_profiles_check') and self.independent_profiles_check.isChecked():
                extra_kw['h0_nose'] = self.h0_nose_spin.value()
                extra_kw['h1_nose'] = self.h1_nose_spin.value()

            self.waverider = GOCWaverider(
                M_inf=self.m_inf_spin.value(),
                beta_OC=beta_oc,
                beta_CD=beta_cd,
                height=height,
                width=self.width_spin.value(),
                dp=[self.x1_spin.value(), self.x2_spin.value(),
                    self.x3_spin.value(), self.x4_spin.value()],
                h0=self.alpha_root_spin.value(),
                h1=self.alpha_tip_spin.value(),
                x_t=x_t,
                dx_t=dx_t,
                n_planes=self.n_planes_spin.value(),
                n_streamwise=self.n_stream_spin.value(),
                blend_exp=self.blend_exp_spin.value(),
                dome_height=self.dome_height_spin.value(),
                dome_taper=self.dome_taper_spin.value(),
                **extra_kw,
            )

            # Generate OC reference (h=1/1) for toggle view
            try:
                self._oc_reference = GOCWaverider(
                    M_inf=self.m_inf_spin.value(),
                    beta_OC=beta_oc, beta_CD=beta_cd,
                    height=height,
                    width=self.width_spin.value(),
                    dp=[self.x1_spin.value(), self.x2_spin.value(),
                        self.x3_spin.value(), self.x4_spin.value()],
                    h0=1.0, h1=1.0,
                    x_t=x_t, dx_t=dx_t,
                    n_planes=self.n_planes_spin.value(),
                    n_streamwise=self.n_stream_spin.value(),
                )
            except Exception:
                self._oc_reference = None

            # Generate CD/uniform reference (h=0/0) for toggle view
            try:
                self._cd_reference = GOCWaverider(
                    M_inf=self.m_inf_spin.value(),
                    beta_OC=beta_oc, beta_CD=beta_cd,
                    height=height,
                    width=self.width_spin.value(),
                    dp=[self.x1_spin.value(), self.x2_spin.value(),
                        self.x3_spin.value(), self.x4_spin.value()],
                    h0=0.0, h1=0.0,
                    x_t=x_t, dx_t=dx_t,
                    n_planes=self.n_planes_spin.value(),
                    n_streamwise=self.n_stream_spin.value(),
                )
            except Exception:
                self._cd_reference = None

            # Switch to 3D canvas, enable toggles
            self.canvas_stack.setCurrentIndex(1)
            self.btn_oc.setEnabled(True)
            self.btn_hybrid.setEnabled(True)
            self.btn_cd.setEnabled(True)
            self.view_mode = 'hybrid'
            self.btn_hybrid.setChecked(True)
            self.btn_oc.setChecked(False)
            self.btn_cd.setChecked(False)

            self._update_3d_plot()
            self._update_blend_plot()
            self.canvas_xsec.plot_comparison(
                self.waverider,
                cd_reference=self._cd_reference,
                oc_reference=self._oc_reference)

            # Stats
            stats = self.waverider.get_blend_stats()
            self.blend_stats_label.setText(
                f"Blend: {stats['oc_fraction']:.0f}% OC  |  "
                f"{stats['cd_fraction']:.0f}% Uniform  (area-weighted)")

            n_lower = len(self.waverider.lower_surface_streams)
            n_upper = len(self.waverider.upper_surface_streams)
            self.status_label.setText(
                f"Generated: {n_upper} upper + {n_lower} lower streams, "
                f"L={self.waverider.length:.3f}")
            self.status_label.setStyleSheet("color: green")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red")
            import traceback
            traceback.print_exc()

    # ══════════════════════════════════════════════════════
    #  Visualisation
    # ══════════════════════════════════════════════════════

    def _update_3d_plot(self):
        if self.waverider is None:
            return
        half_only = self.half_vehicle_check.isChecked()
        wr = self.waverider

        if self.view_mode == 'oc':
            # OC reference (h=1/1) — generated at generate time
            if hasattr(self, '_oc_reference') and self._oc_reference is not None:
                self.canvas_3d.plot_waverider(self._oc_reference, half_only=half_only,
                                              title_prefix='Osculating Cone (reference)')
            else:
                self.canvas_3d.plot_waverider(wr, half_only=half_only,
                                              title_prefix='OC reference (not generated)')
        elif self.view_mode == 'cd':
            # CD reference (h=0/0) — generated at generate time
            if hasattr(self, '_cd_reference') and self._cd_reference is not None:
                self.canvas_3d.plot_waverider(self._cd_reference, half_only=half_only,
                                              title_prefix='Uniform Curvature (reference)')
            else:
                self.canvas_3d.plot_waverider(wr, half_only=half_only,
                                              title_prefix='Uniform reference (not generated)')
        else:
            self.canvas_3d.plot_waverider(wr, half_only=half_only,
                                          title_prefix='Hybrid Waverider')

    def _update_blend_plot(self):
        self.canvas_blend.update_profile(
            self.width_spin.value(),
            self.alpha_root_spin.value(),
            self.alpha_tip_spin.value(),
            self.blend_exp_spin.value(),
            x1=self.x1_spin.value(),
        )

    # ══════════════════════════════════════════════════════
    #  Export
    # ══════════════════════════════════════════════════════

    def export_stl(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save STL", "hybrid_waverider.stl", "STL Files (*.stl)")
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
            self, "Save TRI", "hybrid_waverider.tri", "TRI Files (*.tri)")
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
            self, "Save STEP", "hybrid_waverider.step", "STEP Files (*.step)")
        if not filename:
            return

        try:
            from waverider_generator.cad_export import to_CAD

            sides = 'left' if self.half_vehicle_check.isChecked() else 'both'
            wr = self.waverider

            wr = self.waverider

            to_CAD(
                waverider=wr,
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
        """Return all design parameters as a JSON-serializable dict."""
        d = {
            # Flow conditions
            'mach': self.m_inf_spin.value(),
            'beta_oc': self.beta_spin.value(),
            'beta_cd': self.beta_cd_spin.value(),
            # Geometry
            'height': self.height_spin.value(),
            'width': self.width_spin.value(),
            # OC shape
            'x1': self.x1_spin.value(),
            'x2': self.x2_spin.value(),
            'x3': self.x3_spin.value(),
            'x4': self.x4_spin.value(),
            # Blending
            'alpha_root': self.alpha_root_spin.value(),
            'alpha_tip': self.alpha_tip_spin.value(),
            'blend_exp': self.blend_exp_spin.value(),
            'x_t': self.xt_spin.value(),
            'dx_t': self.dxt_spin.value(),
            # Advanced
            'per_plane_beta': self.per_plane_beta_check.isChecked(),
            'independent_profiles': self.independent_profiles_check.isChecked(),
            'h0_nose': self.h0_nose_spin.value(),
            'h1_nose': self.h1_nose_spin.value(),
            # Resolution
            'n_planes': self.n_planes_spin.value(),
            'n_streamwise': self.n_stream_spin.value(),
            # Dome
            'dome_height': self.dome_height_spin.value(),
            'dome_taper': self.dome_taper_spin.value(),
            # Export options
            'half_vehicle': self.half_vehicle_check.isChecked(),
            'shock_surface': self.shock_surface_check.isChecked(),
        }
        # CD params (only if the group was created)
        if hasattr(self, 'a3_spin'):
            d['a3'] = self.a3_spin.value()
            d['a2'] = self.a2_spin.value()
            d['a0'] = self.a0_spin.value()
            d['cd_top'] = self.cd_top_spin.value()
        return d

    def set_params_dict(self, d):
        """Restore parameters from a dict (e.g. loaded from JSON)."""
        from PyQt5.QtWidgets import QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox

        def _s(widget, value):
            if value is None:
                return
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                widget.setCurrentIndex(idx if idx >= 0 else 0)

        _s(self.m_inf_spin, d.get('mach'))
        _s(self.beta_spin, d.get('beta_oc'))
        _s(self.beta_cd_spin, d.get('beta_cd'))
        _s(self.height_spin, d.get('height'))
        _s(self.width_spin, d.get('width'))
        _s(self.x1_spin, d.get('x1'))
        _s(self.x2_spin, d.get('x2'))
        _s(self.x3_spin, d.get('x3'))
        _s(self.x4_spin, d.get('x4'))
        # CD params (only if the group was created)
        if hasattr(self, 'a3_spin'):
            _s(self.a3_spin, d.get('a3'))
            _s(self.a2_spin, d.get('a2'))
            _s(self.a0_spin, d.get('a0'))
            _s(self.cd_top_spin, d.get('cd_top'))
        _s(self.alpha_root_spin, d.get('alpha_root'))
        _s(self.alpha_tip_spin, d.get('alpha_tip'))
        _s(self.blend_exp_spin, d.get('blend_exp'))
        _s(self.xt_spin, d.get('x_t'))
        _s(self.dxt_spin, d.get('dx_t'))
        _s(self.per_plane_beta_check, d.get('per_plane_beta'))
        _s(self.independent_profiles_check, d.get('independent_profiles'))
        _s(self.h0_nose_spin, d.get('h0_nose'))
        _s(self.h1_nose_spin, d.get('h1_nose'))
        _s(self.n_planes_spin, d.get('n_planes'))
        _s(self.n_stream_spin, d.get('n_streamwise'))
        _s(self.dome_height_spin, d.get('dome_height'))
        _s(self.dome_taper_spin, d.get('dome_taper'))
        _s(self.half_vehicle_check, d.get('half_vehicle'))
        _s(self.shock_surface_check, d.get('shock_surface'))

        # Sync sliders with spinboxes
        self.alpha_root_slider.setValue(int(self.alpha_root_spin.value() * 100))
        self.alpha_tip_slider.setValue(int(self.alpha_tip_spin.value() * 100))
        self.blend_exp_slider.setValue(int(self.blend_exp_spin.value() * 10))


# ===================================================================
# TO ADD THIS TAB TO waverider_gui.py:
#
# 1. Add import at top:
#    from hybrid_waverider_tab import HybridWaveriderTab
#
# 2. In the tab widget initialisation, add after the last existing tab:
#    self.hybrid_tab = HybridWaveriderTab()
#    self.tab_widget.addTab(self.hybrid_tab, "Hybrid Waverider")
# ===================================================================


if __name__ == '__main__':
    app = QApplication(sys.argv)
    tab = HybridWaveriderTab()
    tab.setWindowTitle("Hybrid Waverider Tab (Standalone Test)")
    tab.resize(1200, 800)
    tab.show()
    sys.exit(app.exec_())
