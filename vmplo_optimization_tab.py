"""VMPLO Waverider Optimization Sub-Tab.

Multi-objective optimization of volumetric efficiency using
scipy.optimize.differential_evolution with the VMPLO generator.
Design variables: Ma(z) control points, n(z) control points, X1-X4.
"""

import os
import time
import traceback
import numpy as np

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QGridLayout,
                             QDoubleSpinBox, QSpinBox, QCheckBox, QTextEdit,
                             QSplitter, QApplication, QScrollArea,
                             QTabWidget, QProgressBar, QFileDialog,
                             QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ======================================================================
#  Canvas classes (dark theme)
# ======================================================================

def _style_ax(ax, title=""):
    ax.set_facecolor('#0D0D0D')
    ax.tick_params(colors='#888888')
    ax.xaxis.label.set_color('#CCCCCC')
    ax.yaxis.label.set_color('#CCCCCC')
    if title:
        ax.set_title(title, color='#F59E0B')
    ax.grid(True, alpha=0.15, color='#555555')
    for spine in ax.spines.values():
        spine.set_color('#333333')


class ParetoCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5))
        self.fig.patch.set_facecolor('#1A1A1A')
        self.ax = self.fig.add_subplot(111)
        _style_ax(self.ax, "Evaluated Designs")
        self.ax.set_xlabel("Volumetric Efficiency η")
        self.ax.set_ylabel("Objective J")
        super().__init__(self.fig)
        self.setParent(parent)
        self._etas = []
        self._Js = []
        self._best_eta = None
        self._best_J = None

    def add_point(self, eta, J, is_best=False):
        self._etas.append(eta)
        self._Js.append(J)
        if is_best:
            self._best_eta = eta
            self._best_J = J
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        _style_ax(self.ax, "Evaluated Designs")
        self.ax.set_xlabel("Volumetric Efficiency η")
        self.ax.set_ylabel("Objective J")
        if self._etas:
            colors = np.linspace(0.3, 1.0, len(self._etas))
            self.ax.scatter(self._etas, self._Js, c=colors, cmap='YlOrRd',
                            s=20, alpha=0.7, edgecolors='none')
        if self._best_eta is not None:
            self.ax.plot(self._best_eta, self._best_J, '*',
                         color='#00FF00', markersize=15, zorder=10)
        self.fig.tight_layout()
        self.draw()

    def reset(self):
        self._etas.clear()
        self._Js.clear()
        self._best_eta = None
        self._best_J = None
        self.ax.clear()
        _style_ax(self.ax, "Evaluated Designs")
        self.draw()


class ConvergenceCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5))
        self.fig.patch.set_facecolor('#1A1A1A')
        self.ax = self.fig.add_subplot(111)
        _style_ax(self.ax, "Convergence")
        super().__init__(self.fig)
        self.setParent(parent)
        self._evals = []
        self._Js = []
        self._best_so_far = []

    def add_point(self, eval_num, J):
        self._evals.append(eval_num)
        self._Js.append(J)
        best = min(self._best_so_far[-1], J) if self._best_so_far else J
        self._best_so_far.append(best)
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        _style_ax(self.ax, "Convergence")
        self.ax.set_xlabel("Evaluation #")
        self.ax.set_ylabel("Objective J")
        if self._evals:
            self.ax.plot(self._evals, self._Js, '.', color='#555555',
                         markersize=3, alpha=0.5)
            self.ax.plot(self._evals, self._best_so_far, '-',
                         color='#F59E0B', linewidth=2, label='Best so far')
            self.ax.legend(facecolor='#1A1A1A', edgecolor='#333333',
                           labelcolor='#CCCCCC')
        self.fig.tight_layout()
        self.draw()

    def reset(self):
        self._evals.clear()
        self._Js.clear()
        self._best_so_far.clear()
        self.ax.clear()
        _style_ax(self.ax, "Convergence")
        self.draw()


class DistributionCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6))
        self.fig.patch.set_facecolor('#1A1A1A')
        self.axes = self.fig.subplots(2, 1)
        for ax in self.axes:
            _style_ax(ax)
        super().__init__(self.fig)
        self.setParent(parent)

    def update_best(self, ma_cp, n_cp, half_span):
        for ax in self.axes:
            ax.clear()
            _style_ax(ax)

        z_ma = np.linspace(0, half_span, len(ma_cp))
        self.axes[0].plot(z_ma / half_span, ma_cp, 'o-', color='#4488CC',
                          markersize=6)
        self.axes[0].set_ylabel("Mach number")
        self.axes[0].set_title("Best Design — Ma(z)", color='#F59E0B')

        z_n = np.linspace(0, half_span, len(n_cp))
        self.axes[1].plot(z_n / half_span, n_cp, 's-', color='#CC6644',
                          markersize=6)
        self.axes[1].set_ylabel("Exponent n")
        self.axes[1].set_xlabel("z / half_span")
        self.axes[1].set_title("Best Design — n(z)", color='#F59E0B')

        self.fig.tight_layout()
        self.draw()

    def reset(self):
        for ax in self.axes:
            ax.clear()
            _style_ax(ax)
        self.draw()


# ======================================================================
#  Worker thread
# ======================================================================

class VMPLOOptimizationWorker(QThread):
    progress = pyqtSignal(int, int)
    log_message = pyqtSignal(str)
    new_design = pyqtSignal(dict)
    new_best = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            # NOTE: The legacy VariableMachWaverider (OC-composition) was
            # replaced by the spec-based VMPLOWaverider.  This tab was
            # written against the old API; we adapt here so it still
            # imports and runs.  dp (X1-X4) is ignored by the new method
            # — ICC/L/x_LE are derived from sensible defaults.
            from waverider_generator.vmplo.bspline import BSpline1D
            from waverider_generator.vmplo.osculating import OsculatingAssembly
            from waverider_generator.vmplo.geometry import VMPLOWaverider
            from waverider_generator.distributions import SpanwiseDistribution

            def _dist_to_coeffs(dist, W, n_knots=4):
                """Convert an old SpanwiseDistribution into a new BSpline1D."""
                if dist is None:
                    return None
                sp = BSpline1D(0.0, W, n_internal_knots=n_knots, symmetry=True)
                zs = np.linspace(0.0, W, 50)
                fs = np.array([dist(z) for z in zs])
                sp.fit_values(zs, fs)
                return sp

            def _make_vmplo(M_inf, beta, height, width, dp,
                            n_planes, n_streamwise,
                            Ma_distribution=None, n_distribution=None):
                W = float(width)
                H = float(height)
                x_LE = 0.05
                L = max(3.0 * H, 1.0)
                Ma_sp = (_dist_to_coeffs(Ma_distribution, W)
                         if Ma_distribution is not None
                         else BSpline1D.constant(M_inf, 0.0, W))
                n_sp = (_dist_to_coeffs(n_distribution, W)
                        if n_distribution is not None
                        else BSpline1D.constant(1.0, 0.0, W))
                icc_sp = BSpline1D.linear(0.95 * H, 0.30 * H, 0.0, W)
                assembly = OsculatingAssembly(
                    Ma_spline=Ma_sp, n_spline=n_sp,
                    ICC_spline=icc_sp, US_spline=None,
                    beta_design=beta, L=L, W=W, H=H, x_LE=x_LE)
                return VMPLOWaverider(assembly, n_planes=n_planes,
                                      n_streamwise=n_streamwise)

            VariableMachWaverider = _make_vmplo  # noqa: N806 (adapter alias)

            cfg = self.config
            height = cfg['height']
            width = cfg['width']
            m_inf = cfg['m_inf']
            beta_ref = cfg['beta_ref']
            n_planes = cfg['n_planes']
            n_streamwise = cfg['n_streamwise']
            half_span = width

            # Build bounds and variable map
            bounds = []
            var_names = []

            if cfg['active_ma']:
                n_ma = cfg['n_ma_cp']
                for i in range(n_ma):
                    bounds.append((cfg['ma_lower'], cfg['ma_upper']))
                    var_names.append(f'ma_cp_{i}')

            if cfg['active_n']:
                n_n = cfg['n_n_cp']
                for i in range(n_n):
                    bounds.append((cfg['n_lower'], cfg['n_upper']))
                    var_names.append(f'n_cp_{i}')

            if cfg['active_x1']:
                bounds.append((0.0, 0.95))
                var_names.append('x1')
            if cfg['active_x2']:
                bounds.append((0.0, 0.99))
                var_names.append('x2')
            if cfg['active_x3']:
                bounds.append((0.0, 1.0))
                var_names.append('x3')
            if cfg['active_x4']:
                bounds.append((0.0, 1.0))
                var_names.append('x4')

            n_vars = len(bounds)
            if n_vars == 0:
                self.error.emit("No active design variables!")
                return

            self.log_message.emit(
                f"Starting DE optimization: {n_vars} variables, "
                f"pop={cfg['pop_size']}, maxiter={cfg['max_iter']}")
            self.log_message.emit(f"Variables: {', '.join(var_names)}")

            eval_count = [0]
            best_J = [float('inf')]
            best_result = [None]
            cache = {}

            def objective(x):
                if self._stop:
                    raise StopIteration()

                eval_count[0] += 1

                # Cache check
                key = tuple(np.round(x, 6))
                if key in cache:
                    return cache[key]

                # Decode vector
                idx = 0
                ma_cp = None
                n_cp = None

                if cfg['active_ma']:
                    n_ma = cfg['n_ma_cp']
                    ma_cp = x[idx:idx + n_ma].tolist()
                    idx += n_ma

                if cfg['active_n']:
                    n_n = cfg['n_n_cp']
                    n_cp = x[idx:idx + n_n].tolist()
                    idx += n_n

                x1 = float(x[idx]) if 'x1' in var_names else cfg.get('x1_fixed', 0.0)
                if 'x1' in var_names:
                    idx += 1
                x2 = float(x[idx]) if 'x2' in var_names else cfg.get('x2_fixed', 0.2)
                if 'x2' in var_names:
                    idx += 1
                x3 = float(x[idx]) if 'x3' in var_names else cfg.get('x3_fixed', 0.5)
                if 'x3' in var_names:
                    idx += 1
                x4 = float(x[idx]) if 'x4' in var_names else cfg.get('x4_fixed', 0.5)
                if 'x4' in var_names:
                    idx += 1

                # Son et al. constraint
                denom = max((1.0 - x1) ** 4, 1e-12)
                lhs = x2 / denom
                rhs = (7.0 / 64.0) * (width / max(height, 0.01)) ** 4 * 0.9
                if lhs >= rhs:
                    penalty = 1000.0 + (lhs - rhs) * 100.0
                    cache[key] = penalty
                    self.log_message.emit(
                        f"  [{eval_count[0]}] INFEASIBLE (X1={x1:.3f} X2={x2:.3f})")
                    self.progress.emit(eval_count[0], -1)
                    return penalty

                # Build distributions
                ma_dist = None
                n_dist = None

                if ma_cp is not None:
                    ma_dist = SpanwiseDistribution(
                        ma_cp, half_span=half_span, name="Mach")

                if n_cp is not None:
                    # Check all n values are valid
                    if any(v <= 0.5 or v > 1.0 for v in n_cp):
                        cache[key] = 999.0
                        self.progress.emit(eval_count[0], -1)
                        return 999.0
                    n_dist = SpanwiseDistribution(
                        n_cp, half_span=half_span, name="exponent")

                # Generate waverider (via the _make_vmplo adapter
                # defined at the top of ``run``).
                try:
                    wr = VariableMachWaverider(
                        M_inf=m_inf, beta=beta_ref,
                        height=height, width=width,
                        dp=[x1, x2, x3, x4],
                        n_planes=n_planes,
                        n_streamwise=n_streamwise,
                        Ma_distribution=ma_dist,
                        n_distribution=n_dist)
                except Exception:
                    cache[key] = 998.0
                    self.progress.emit(eval_count[0], -1)
                    return 998.0

                # Compute metrics
                vol = wr.compute_volume()
                if vol <= 0:
                    cache[key] = 997.0
                    self.progress.emit(eval_count[0], -1)
                    return 997.0

                # Planform area
                us = wr.upper_surface_streams
                chords, z_pos = [], []
                for s in us:
                    if s.shape[0] >= 2:
                        chords.append(s[-1, 0] - s[0, 0])
                        z_pos.append(s[0, 2])
                a_half = float(np.trapz(chords, z_pos)) if len(chords) > 2 else 1e-6
                a_plan = 2.0 * max(a_half, 1e-6)  # full vehicle

                eta = vol ** (2.0 / 3.0) / a_plan

                # Objective: minimize -eta (maximize eta)
                J = -eta

                cache[key] = J

                result = {
                    'eval': eval_count[0],
                    'J': J,
                    'eta': eta,
                    'volume': vol,
                    'planform': a_plan,
                    'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
                    'ma_cp': ma_cp,
                    'n_cp': n_cp,
                }

                self.new_design.emit(result)
                self.progress.emit(eval_count[0], -1)

                if J < best_J[0]:
                    best_J[0] = J
                    best_result[0] = result.copy()
                    self.new_best.emit(result)
                    self.log_message.emit(
                        f"  [{eval_count[0]}] ★ NEW BEST η={eta:.5f} "
                        f"V={vol:.4f}m³ A={a_plan:.4f}m²")
                elif eval_count[0] % 20 == 0:
                    self.log_message.emit(
                        f"  [{eval_count[0]}] η={eta:.5f} V={vol:.4f}")

                return J

            # Run DE
            from scipy.optimize import differential_evolution

            seed = cfg.get('seed', 42)
            if seed == 0:
                seed = None

            result = differential_evolution(
                objective, bounds=bounds,
                maxiter=cfg['max_iter'],
                popsize=cfg['pop_size'],
                seed=seed,
                tol=1e-6,
                mutation=(0.5, 1.5),
                recombination=0.7,
                polish=False,
            )

            self.log_message.emit(
                f"\nOptimization complete: {result.message}")
            self.log_message.emit(
                f"Total evaluations: {eval_count[0]}, "
                f"Cache hits: {eval_count[0] - len(cache)}")

            self.finished.emit({
                'best': best_result[0],
                'total_evals': eval_count[0],
                'scipy_result': str(result),
            })

        except StopIteration:
            self.log_message.emit("\nOptimization stopped by user")
            self.finished.emit({'stopped': True, 'best': best_result[0] if 'best_result' in dir() else None})
        except Exception as e:
            self.error.emit(str(e))
            self.log_message.emit(f"\nError: {e}\n{traceback.format_exc()}")


# ======================================================================
#  Main tab widget
# ======================================================================

class VMPLOOptimizationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vmplo_tab = parent  # reference to VMPLOWaveriderTab
        self.worker = None
        self._best_result = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # -- Left panel --
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(450)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(self._create_optimizer_group())
        left_layout.addWidget(self._create_variables_group())
        left_layout.addWidget(self._create_geometry_group())
        left_layout.addWidget(self._create_constraint_group())
        left_layout.addWidget(self._create_controls())
        left_layout.addStretch()

        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # -- Right panel --
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        right_splitter = QSplitter(Qt.Vertical)

        # Plot tabs
        self.plot_tabs = QTabWidget()
        self.pareto_canvas = ParetoCanvas()
        self.plot_tabs.addTab(self.pareto_canvas, "Designs")
        self.convergence_canvas = ConvergenceCanvas()
        self.plot_tabs.addTab(self.convergence_canvas, "Convergence")
        self.dist_canvas = DistributionCanvas()
        self.plot_tabs.addTab(self.dist_canvas, "Best Distributions")
        right_splitter.addWidget(self.plot_tabs)

        # Log console
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet(
            "QTextEdit { background-color: #0D0D0D; color: #CCCCCC; "
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; "
            "border: 1px solid #333333; }")
        self.log_console.setMaximumHeight(250)
        right_splitter.addWidget(self.log_console)

        right_splitter.setSizes([500, 200])
        right_layout.addWidget(right_splitter)
        splitter.addWidget(right_widget)

        splitter.setSizes([380, 700])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

    # ==============================================================
    #  Parameter groups
    # ==============================================================

    def _create_optimizer_group(self):
        group = QGroupBox("Optimizer Settings")
        layout = QGridLayout()
        row = 0

        layout.addWidget(QLabel("α (η weight):"), row, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setToolTip(
            "1.0 = maximize volumetric efficiency only\n"
            "0.0 = maximize L/D only (requires CFD)\n"
            "Currently geometry-only mode uses α=1.0")
        layout.addWidget(self.alpha_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("Population:"), row, 0)
        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(5, 200)
        self.pop_spin.setValue(15)
        self.pop_spin.setSingleStep(5)
        self.pop_spin.setToolTip("DE population size (total = pop × n_vars)")
        layout.addWidget(self.pop_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("Max Iterations:"), row, 0)
        self.maxiter_spin = QSpinBox()
        self.maxiter_spin.setRange(5, 1000)
        self.maxiter_spin.setValue(30)
        self.maxiter_spin.setSingleStep(5)
        layout.addWidget(self.maxiter_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("Seed:"), row, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self.seed_spin.setToolTip("0 = random seed")
        layout.addWidget(self.seed_spin, row, 1)

        group.setLayout(layout)
        return group

    def _create_variables_group(self):
        group = QGroupBox("Design Variables")
        layout = QGridLayout()
        row = 0

        # Header
        layout.addWidget(QLabel("Active"), row, 0)
        layout.addWidget(QLabel("Variable"), row, 1)
        layout.addWidget(QLabel("Lower"), row, 2)
        layout.addWidget(QLabel("Upper"), row, 3)
        row += 1

        # Ma(z) control points
        self.active_ma = QCheckBox()
        self.active_ma.setChecked(True)
        layout.addWidget(self.active_ma, row, 0)
        layout.addWidget(QLabel("Ma(z) CPs"), row, 1)
        self.ma_lower = QDoubleSpinBox()
        self.ma_lower.setRange(1.5, 25.0)
        self.ma_lower.setValue(5.0)
        self.ma_lower.setDecimals(1)
        layout.addWidget(self.ma_lower, row, 2)
        self.ma_upper = QDoubleSpinBox()
        self.ma_upper.setRange(1.5, 25.0)
        self.ma_upper.setValue(15.0)
        self.ma_upper.setDecimals(1)
        layout.addWidget(self.ma_upper, row, 3)
        row += 1

        layout.addWidget(QLabel("  # CPs:"), row, 1)
        self.n_ma_cp_spin = QSpinBox()
        self.n_ma_cp_spin.setRange(2, 8)
        self.n_ma_cp_spin.setValue(5)
        layout.addWidget(self.n_ma_cp_spin, row, 2, 1, 2)
        row += 1

        # n(z) control points
        self.active_n = QCheckBox()
        self.active_n.setChecked(True)
        layout.addWidget(self.active_n, row, 0)
        layout.addWidget(QLabel("n(z) CPs"), row, 1)
        self.n_lower = QDoubleSpinBox()
        self.n_lower.setRange(0.55, 1.0)
        self.n_lower.setValue(0.60)
        self.n_lower.setDecimals(2)
        self.n_lower.setSingleStep(0.05)
        layout.addWidget(self.n_lower, row, 2)
        self.n_upper_spin = QDoubleSpinBox()
        self.n_upper_spin.setRange(0.55, 1.0)
        self.n_upper_spin.setValue(1.0)
        self.n_upper_spin.setDecimals(2)
        self.n_upper_spin.setSingleStep(0.05)
        layout.addWidget(self.n_upper_spin, row, 3)
        row += 1

        layout.addWidget(QLabel("  # CPs:"), row, 1)
        self.n_n_cp_spin = QSpinBox()
        self.n_n_cp_spin.setRange(2, 8)
        self.n_n_cp_spin.setValue(5)
        layout.addWidget(self.n_n_cp_spin, row, 2, 1, 2)
        row += 1

        # X1-X4
        self.active_x = {}
        for name, default_lo, default_hi in [
            ('X1', 0.0, 0.5), ('X2', 0.0, 0.5),
            ('X3', 0.0, 1.0), ('X4', 0.0, 1.0)]:
            cb = QCheckBox()
            cb.setChecked(True)
            self.active_x[name] = cb
            layout.addWidget(cb, row, 0)
            layout.addWidget(QLabel(name), row, 1)
            lo = QDoubleSpinBox()
            lo.setRange(0.0, 1.0)
            lo.setValue(default_lo)
            lo.setDecimals(3)
            lo.setSingleStep(0.01)
            layout.addWidget(lo, row, 2)
            hi = QDoubleSpinBox()
            hi.setRange(0.0, 1.0)
            hi.setValue(default_hi)
            hi.setDecimals(3)
            hi.setSingleStep(0.01)
            layout.addWidget(hi, row, 3)
            setattr(self, f'{name.lower()}_lo', lo)
            setattr(self, f'{name.lower()}_hi', hi)
            row += 1

        # Active count
        self.active_count_label = QLabel("Active variables: --")
        self.active_count_label.setStyleSheet("color: #F59E0B; font-size: 11px;")
        layout.addWidget(self.active_count_label, row, 0, 1, 4)

        for cb in [self.active_ma, self.active_n] + list(self.active_x.values()):
            cb.stateChanged.connect(self._update_active_count)
        self.n_ma_cp_spin.valueChanged.connect(self._update_active_count)
        self.n_n_cp_spin.valueChanged.connect(self._update_active_count)
        self._update_active_count()

        group.setLayout(layout)
        return group

    def _create_geometry_group(self):
        group = QGroupBox("Fixed Geometry")
        layout = QGridLayout()

        for row, (label, attr, default, lo, hi, step, dec) in enumerate([
            ("Height (m):", 'opt_height', 1.34, 0.01, 20.0, 0.01, 3),
            ("Width (m):", 'opt_width', 3.0, 0.1, 20.0, 0.1, 3),
            ("Ref. Mach:", 'opt_mach', 6.0, 1.5, 25.0, 0.5, 1),
            ("Ref. Beta (°):", 'opt_beta', 13.0, 5.0, 45.0, 0.5, 1),
        ]):
            layout.addWidget(QLabel(label), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(default)
            spin.setSingleStep(step)
            spin.setDecimals(dec)
            layout.addWidget(spin, row, 1)
            setattr(self, attr, spin)

        row = 4
        layout.addWidget(QLabel("Osc. Planes:"), row, 0)
        self.opt_planes = QSpinBox()
        self.opt_planes.setRange(10, 50)
        self.opt_planes.setValue(12)
        self.opt_planes.setToolTip("Low for speed during optimization")
        layout.addWidget(self.opt_planes, row, 1)

        row = 5
        layout.addWidget(QLabel("Streamwise:"), row, 0)
        self.opt_stream = QSpinBox()
        self.opt_stream.setRange(10, 50)
        self.opt_stream.setValue(15)
        layout.addWidget(self.opt_stream, row, 1)

        group.setLayout(layout)
        return group

    def _create_constraint_group(self):
        group = QGroupBox("Constraint Status")
        layout = QGridLayout()

        self.constraint_son = QLabel("🟢 Son et al.")
        self.constraint_mach = QLabel("🟢 Mach > arcsin(1/Ma)")
        self.constraint_vars = QLabel("🟢 Variables active")
        layout.addWidget(self.constraint_son, 0, 0)
        layout.addWidget(self.constraint_mach, 1, 0)
        layout.addWidget(self.constraint_vars, 2, 0)

        group.setLayout(layout)
        return group

    def _create_controls(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #AAAAAA;")
        layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #2B5B2B; color: white; "
            "font-weight: bold; padding: 8px; font-size: 13px; }"
            "QPushButton:hover { background-color: #3B7B3B; }")
        self.start_btn.clicked.connect(self.start_optimization)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #5B2B2B; color: white; "
            "padding: 8px; font-size: 13px; }"
            "QPushButton:hover { background-color: #7B3B3B; }")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_optimization)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.load_best_btn = QPushButton("📋 Load Best Design")
        self.load_best_btn.setEnabled(False)
        self.load_best_btn.clicked.connect(self.load_best_design)
        layout.addWidget(self.load_best_btn)

        return widget

    # ==============================================================
    #  Actions
    # ==============================================================

    def _update_active_count(self):
        count = 0
        if self.active_ma.isChecked():
            count += self.n_ma_cp_spin.value()
        if self.active_n.isChecked():
            count += self.n_n_cp_spin.value()
        for name, cb in self.active_x.items():
            if cb.isChecked():
                count += 1
        self.active_count_label.setText(f"Active variables: {count}")
        ok = count > 0
        self.constraint_vars.setText(
            f"{'🟢' if ok else '🔴'} Variables active ({count})")
        self.start_btn.setEnabled(ok and self.worker is None)

    def _build_config(self):
        return {
            'height': self.opt_height.value(),
            'width': self.opt_width.value(),
            'm_inf': self.opt_mach.value(),
            'beta_ref': self.opt_beta.value(),
            'n_planes': self.opt_planes.value(),
            'n_streamwise': self.opt_stream.value(),
            'pop_size': self.pop_spin.value(),
            'max_iter': self.maxiter_spin.value(),
            'seed': self.seed_spin.value(),
            'alpha': self.alpha_spin.value(),
            'active_ma': self.active_ma.isChecked(),
            'active_n': self.active_n.isChecked(),
            'active_x1': self.active_x['X1'].isChecked(),
            'active_x2': self.active_x['X2'].isChecked(),
            'active_x3': self.active_x['X3'].isChecked(),
            'active_x4': self.active_x['X4'].isChecked(),
            'n_ma_cp': self.n_ma_cp_spin.value(),
            'n_n_cp': self.n_n_cp_spin.value(),
            'ma_lower': self.ma_lower.value(),
            'ma_upper': self.ma_upper.value(),
            'n_lower': self.n_lower.value(),
            'n_upper': self.n_upper_spin.value(),
            'x1_fixed': 0.0,
            'x2_fixed': 0.2,
            'x3_fixed': 0.5,
            'x4_fixed': 0.5,
        }

    def start_optimization(self):
        if self.worker is not None:
            return

        config = self._build_config()

        # Reset plots
        self.pareto_canvas.reset()
        self.convergence_canvas.reset()
        self.dist_canvas.reset()
        self.log_console.clear()
        self.progress_bar.setValue(0)

        self.log_console.append(
            f"=== VMPLO Optimization ===\n"
            f"Height={config['height']}, Width={config['width']}\n"
            f"M_inf={config['m_inf']}, Beta={config['beta_ref']}°\n"
            f"Planes={config['n_planes']}, Stream={config['n_streamwise']}\n")

        self.worker = VMPLOOptimizationWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._on_log)
        self.worker.new_design.connect(self._on_new_design)
        self.worker.new_best.connect(self._on_new_best)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Running...")
        self.status_label.setStyleSheet("color: #F59E0B;")

        self.worker.start()

    def stop_optimization(self):
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Stopping...")

    def load_best_design(self):
        if self._best_result is None or self.vmplo_tab is None:
            return
        r = self._best_result
        # Build params dict for the Design tab
        params = {}
        if r.get('ma_cp'):
            ma_cp = r['ma_cp']
            params['ma_center'] = ma_cp[0]
            params['ma_tip'] = ma_cp[-1]
            params['ma_dist'] = 1  # Linear
        if r.get('n_cp'):
            n_cp = r['n_cp']
            params['n_center'] = n_cp[0]
            params['n_tip'] = n_cp[-1]
            params['n_dist'] = 1  # Linear
        params['x1'] = r.get('x1', 0.0)
        params['x2'] = r.get('x2', 0.2)
        params['x3'] = r.get('x3', 0.5)
        params['x4'] = r.get('x4', 0.5)
        params['height'] = self.opt_height.value()
        params['width'] = self.opt_width.value()
        params['beta'] = self.opt_beta.value()

        self.vmplo_tab.set_params_dict(params)
        self.log_console.append("\n📋 Best design loaded into Design tab")

    # ==============================================================
    #  Worker signal handlers
    # ==============================================================

    def _on_progress(self, current, total):
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))

    def _on_log(self, text):
        self.log_console.append(text)
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_new_design(self, result):
        self.pareto_canvas.add_point(result['eta'], result['J'])
        self.convergence_canvas.add_point(result['eval'], result['J'])

    def _on_new_best(self, result):
        self._best_result = result
        self.pareto_canvas.add_point(result['eta'], result['J'], is_best=True)
        self.status_label.setText(
            f"Best: η={result['eta']:.5f}  V={result['volume']:.4f}m³")
        # Update distribution plot
        half_span = self.opt_width.value()
        ma_cp = result.get('ma_cp', [self.opt_mach.value()] * 5)
        n_cp = result.get('n_cp', [1.0] * 5)
        self.dist_canvas.update_best(ma_cp, n_cp, half_span)
        self.load_best_btn.setEnabled(True)

    def _on_finished(self, summary):
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setStyleSheet("color: #4ADE80;")
        if summary.get('stopped'):
            self.status_label.setText("Stopped")
        else:
            best = summary.get('best')
            if best:
                self.status_label.setText(
                    f"Done: η={best['eta']:.5f} ({summary['total_evals']} evals)")
            else:
                self.status_label.setText("Done (no feasible design found)")

    def _on_error(self, msg):
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Error: {msg}")
        self.status_label.setStyleSheet("color: #EF4444;")
