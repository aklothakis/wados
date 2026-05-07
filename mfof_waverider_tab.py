"""GUI tab for the MFOF (Multi-Flowfield Osculating Framework) waverider — Phase 2.

A thin subclass of :class:`Liu2019WaveriderTab` that routes geometry and
aerodynamics through the :mod:`mfof` package with an all-cone factory. Output
is numerically identical (within ``1e-6`` relative) to the Liu 2019 tab when
the same parameters are used; the equivalence is gated by
:func:`mfof.validate.run_equivalence_test`.

The subclass adds:

* a "Flowfield (Phase 2: cone only)" group in the left input panel with a
  disabled :class:`QComboBox` (placeholder for future phase additions),
* a "MFOF equivalence vs Liu 2019" row in the Validation sub-tab.

Future phases will add :class:`PowerLawFlowfield`, :class:`WedgeFlowfield`
options to the combobox and allow mixed-flowfield waveriders.
"""

from __future__ import annotations

import traceback
from typing import Optional

import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

# Reuse the Liu 2019 tab as a base. Its module-level imports already pull in
# matplotlib and the liu2019 package; we layer the mfof routing on top.
from liu2019_waverider_tab import Liu2019WaveriderTab, ACCENT, BG_DARK, TEXT_MUTED

from mfof.cone_flowfield import ConeFlowfield
from mfof.wedge_flowfield import WedgeFlowfield
from mfof.power_law_flowfield import PowerLawFlowfield
from mfof.geometry import build_mfof_waverider
from mfof.aero import MFOFAeroEvaluator
from mfof.validate import run_equivalence_test


# ===========================================================================
#  MFOF-specific worker threads
# ===========================================================================

class _MFOFGeometryWorker(QThread):
    """Builds an MFOF waverider on a worker thread.

    Dispatches by ``flowfield_type``:

    * ``"cone"`` (default): all-cone factory -- byte-identical to the
      Phase 2 production path, so the equivalence test still passes at 1e-13.
    * ``"wedge"``: all-wedge factory using ``WedgeFlowfield``.
    * ``"power-law"``: all-power-law factory with the supplied exponent
      ``n``. Routes through ``mfof.moc`` (axisymmetric MOC); ~2 min for
      a 200-plane sweep at default mesh density.
    """
    finished_ok = pyqtSignal(object)   # emits MFOFWaverider
    failed      = pyqtSignal(str)

    def __init__(self, params: dict, n_z: int, n_x: int,
                 flowfield_type: str = "cone",
                 power_law_n: float = 1.0):
        super().__init__()
        self.params, self.n_z, self.n_x = params, n_z, n_x
        self.flowfield_type = str(flowfield_type)
        self.power_law_n = float(power_law_n)

    def run(self):
        try:
            beta = float(self.params["beta_deg"])
            gamma = float(self.params.get("gamma", 1.4))
            L_w = float(self.params["L_w"])
            L_s = float(self.params["L_s"])
            ftype = self.flowfield_type
            n_pl = self.power_law_n

            # ------------------------------------------------------------
            # Factory dispatch -- uniform per type.
            #
            # * Cone:      Phase 2 production path, byte-identical to
            #              Liu 2019 (equivalence test still 1e-13).
            # * Wedge:     uniform (no apex singularity).
            # * Power-law: uniform. PowerLawFlowfield internally clamps
            #              x_LE to 1e-9 for the singular centerline case
            #              (n<1 has a vertical tangent at x=0), so every
            #              plane goes through the same MOC code path.
            #              That keeps the spanwise streamline stack
            #              continuous: there is no cone-vs-MOC model
            #              boundary inside the half-span. (Earlier
            #              versions used a hybrid -- cone for |z| <= L_s
            #              or for x_LE < eps_apex -- but both produced a
            #              visible step at the cone-MOC handover.)
            # ------------------------------------------------------------
            if ftype == "wedge":
                def factory(z, Ma_z):
                    return WedgeFlowfield(Ma_z, beta, gamma)
            elif ftype == "power-law":
                def factory(z, Ma_z):
                    return PowerLawFlowfield(
                        Ma_z, beta, n=n_pl, L=L_w, gamma=gamma)
            else:                          # "cone" (default)
                def factory(z, Ma_z):
                    return ConeFlowfield(Ma_z, beta, gamma)

            wr = build_mfof_waverider(
                self.params, factory, n_z=self.n_z, n_x=self.n_x)
            # Tag the waverider with the flowfield type so the GUI can show
            # the right header / diagnostics tab content.
            try:
                wr._mfof_flowfield_type = ftype
                wr._mfof_power_law_n = n_pl if ftype == "power-law" else None
            except Exception:
                pass
            self.finished_ok.emit(wr)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class _MFOFAeroWorker(QThread):
    """Aero evaluation on a worker thread using :class:`MFOFAeroEvaluator`."""
    finished_ok = pyqtSignal(object)   # emits list of dicts
    failed      = pyqtSignal(str)
    progress    = pyqtSignal(int)

    def __init__(self, waverider):
        super().__init__()
        self.waverider = waverider

    def run(self):
        try:
            evaluator = MFOFAeroEvaluator(self.waverider)
            rows = evaluator.evaluate_paper_trajectory(
                progress_callback=self.progress.emit)
            self.finished_ok.emit(rows)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# ===========================================================================
#  MFOF tab
# ===========================================================================

class MFOFWaveriderTab(Liu2019WaveriderTab):
    """MFOF waverider tab — Phase 2 architectural-refactor deliverable.

    Same UI as :class:`Liu2019WaveriderTab` plus a "Flowfield" placeholder
    group and an equivalence-check row in the validation table. Geometry
    and aero are routed through the :mod:`mfof` package.
    """

    def __init__(self, parent=None):
        # Pre-create the equivalence-cache attribute so super().__init__'s
        # call to _on_geometry_ready (if any) doesn't AttributeError.
        self._equivalence_pass: Optional[bool] = None
        self._diag_canvas = None        # populated below
        super().__init__(parent)
        # Relabel the 3-D canvas title and info-panel header so MFOF
        # doesn't show the base-class "Liu 2019 Waverider" branding.
        if hasattr(self, "canvas_3d") and self.canvas_3d is not None:
            self.canvas_3d.title_prefix = "MFOF Waverider"
            self.canvas_3d.info_panel_header = "MFOF WAVERIDER"
        # After the base UI is fully constructed, splice the Flowfield group
        # into the left panel (above the feasibility label) and the
        # Flowfield-diagnostics sub-tab into the right panel.
        self._inject_flowfield_group()
        self._inject_diagnostics_subtab()

    # ---------------------------------------------------------------
    #  Left-panel: insert "Flowfield" placeholder group
    # ---------------------------------------------------------------
    def _inject_flowfield_group(self):
        """Insert a new ``QGroupBox`` between the mesh group and the
        feasibility label in the left input panel."""
        layout = self.feasibility_label.parentWidget().layout()
        if layout is None:
            return
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is not None and item.widget() is self.feasibility_label:
                layout.insertWidget(i, self._group_flowfield())
                break

    # Flowfield-combobox value -> internal key
    _FF_COMBO_TO_KEY = {
        "Cone (Taylor-Maccoll)":      "cone",
        "Wedge (2D oblique shock)":   "wedge",
        "Power-law (axisymmetric MOC)": "power-law",
    }

    def _group_flowfield(self) -> QGroupBox:
        """Phase 3: live combobox + conditional ``n`` spinbox.

        The default selection is "Cone (Taylor-Maccoll)", which routes
        through the same code path as the Phase 2 production cone build,
        so the equivalence test still passes at 1e-13 on first open.
        """
        g = QGroupBox("Flowfield")
        grid = QGridLayout(g)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setSpacing(4)

        # ---- Combobox ----
        grid.addWidget(QLabel("Type:"), 0, 0)
        self.flowfield_combo = QComboBox()
        for label in self._FF_COMBO_TO_KEY.keys():
            self.flowfield_combo.addItem(label)
        self.flowfield_combo.setEnabled(True)
        self.flowfield_combo.setCurrentIndex(0)        # default = Cone
        self.flowfield_combo.setToolTip(
            "<b>Basic flowfield</b> for each osculating plane.<br><br>"
            "<b>Cone (Taylor-Maccoll):</b> Sobieczky 1990 / Liu 2019. "
            "Streamline is a straight line at angle delta_c.<br>"
            "<b>Wedge (2D oblique shock):</b> 2D wedge limit. "
            "Streamline is straight at theta_w &lt; delta_c -- a shallower "
            "compression than the cone.<br>"
            "<b>Power-law (axisymmetric MOC):</b> Rodi 2005 / Mazhul 2004. "
            "Curved streamline traced by axisymmetric MOC. "
            "Generation takes ~2 min at default mesh density.<br><br>"
            "Default selection 'Cone' reproduces the Liu 2019 tab to "
            "1e-13.")
        self.flowfield_combo.currentIndexChanged.connect(
            self._on_flowfield_changed)
        grid.addWidget(self.flowfield_combo, 0, 1)

        # ---- Power-law exponent n (visible only when power-law selected) ----
        self.n_powerlaw_label = QLabel("Power-law n:")
        self.n_powerlaw_spin = QDoubleSpinBox()
        self.n_powerlaw_spin.setRange(0.30, 2.00)
        self.n_powerlaw_spin.setSingleStep(0.05)
        self.n_powerlaw_spin.setDecimals(3)
        self.n_powerlaw_spin.setValue(0.7)              # Mazhul-typical
        self.n_powerlaw_spin.setToolTip(
            "<b>Power-law body exponent n</b> in r(x) = R_b * (x/L)^n.<br><br>"
            "<b>n = 1:</b> degenerate cone (use 'Cone' option above for "
            "machine-precision result).<br>"
            "<b>n &lt; 1:</b> ogive (concave-up body, sharp apex).<br>"
            "<b>n &gt; 1:</b> tangent-ogive-like (convex-up body, "
            "blunt apex).<br><br>"
            "Mazhul 2004 typically uses n in [0.6, 0.9].")
        self.n_powerlaw_label.setVisible(False)
        self.n_powerlaw_spin.setVisible(False)
        grid.addWidget(self.n_powerlaw_label, 1, 0)
        grid.addWidget(self.n_powerlaw_spin, 1, 1)

        # ---- Note ----
        note = QLabel(
            "Cone matches Liu 2019 bit-identically. Wedge is shallower. "
            "Power-law uses MOC and takes ~2 min per generate.")
        note.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px; "
                           f"font-style: italic;")
        note.setWordWrap(True)
        grid.addWidget(note, 2, 0, 1, 2)
        return g

    def _on_flowfield_changed(self, _index: int):
        """Show/hide the ``n`` spinbox based on the combobox selection."""
        key = self._current_flowfield_key()
        is_power = (key == "power-law")
        self.n_powerlaw_label.setVisible(is_power)
        self.n_powerlaw_spin.setVisible(is_power)

    def _current_flowfield_key(self) -> str:
        """Return the internal flowfield key (cone/wedge/power-law) for the
        currently-selected combobox entry."""
        return self._FF_COMBO_TO_KEY.get(
            self.flowfield_combo.currentText(), "cone")

    # ---------------------------------------------------------------
    #  Right-panel: insert "Flowfield diagnostics" sub-tab
    # ---------------------------------------------------------------
    def _inject_diagnostics_subtab(self):
        """Add a new sub-tab to the right-panel QTabWidget showing the MOC
        mesh (when power-law) or the streamline trajectory (any flowfield)
        in the local osculating-plane (x, r) frame.
        """
        if not hasattr(self, "sub_tabs"):
            return
        self._diag_canvas = _MFOFDiagnosticsCanvas()
        v = QWidget()
        l = QVBoxLayout(v)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(NavigationToolbar(self._diag_canvas, v))
        l.addWidget(self._diag_canvas)
        # Insert before the Validation tab so the order reads naturally.
        idx = self.sub_tabs.indexOf(self.validation_table)
        if idx < 0:
            idx = self.sub_tabs.count()
        self.sub_tabs.insertTab(idx, v, "Flowfield diagnostics")

    # ---------------------------------------------------------------
    #  Geometry generation: route to MFOF worker, run equivalence test
    # ---------------------------------------------------------------
    def _on_generate(self):
        if not self._check_feasibility():
            return
        params = self._read_params()
        n_z = self.n_z_spin.value()
        n_x = self.n_x_spin.value()
        ftype = self._current_flowfield_key()
        n_pl  = float(self.n_powerlaw_spin.value())

        self.generate_btn.setEnabled(False)
        self.run_aero_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)

        # Hint to the user that power-law takes much longer than cone/wedge
        if ftype == "power-law":
            self.progress_bar.setFormat(
                "Generating power-law waverider (MOC, ~2 min)...")
        else:
            self.progress_bar.setFormat("")

        self._geom_worker = _MFOFGeometryWorker(
            params, n_z, n_x, flowfield_type=ftype, power_law_n=n_pl)
        self._geom_worker.finished_ok.connect(self._on_geometry_ready)
        self._geom_worker.failed.connect(self._on_worker_error)
        self._geom_worker.start()

    def _on_geometry_ready(self, wr):
        """After MFOF geometry is built:

        * If the cone flowfield was used, run the Liu-vs-MFOF equivalence
          test (it should still pass at 1e-13).
        * If wedge or power-law was used, the equivalence test is not
          meaningful (different physics) -- skip it and mark "n/a".

        Then call the base class to refresh canvases and metric cards.
        """
        ftype = self._current_flowfield_key()
        if ftype == "cone":
            try:
                self._equivalence_pass = run_equivalence_test(
                    self._read_params(),
                    n_z=self.n_z_spin.value(),
                    n_x=self.n_x_spin.value(),
                    verbose=False,
                )
            except Exception:
                self._equivalence_pass = False
        else:
            # Equivalence to Liu 2019 only meaningful for the cone path.
            self._equivalence_pass = None
        super()._on_geometry_ready(wr)
        # Update the diagnostics sub-tab if it exists (built in _build_right_panel)
        if hasattr(self, "_diag_canvas") and self._diag_canvas is not None:
            self._diag_canvas.plot(wr, self._read_params(),
                                    flowfield_type=ftype,
                                    power_law_n=float(self.n_powerlaw_spin.value()))

    # ---------------------------------------------------------------
    #  Aero: route to MFOF aero worker
    # ---------------------------------------------------------------
    def _on_run_aero(self):
        if self.waverider is None:
            return
        self.run_aero_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self._aero_worker = _MFOFAeroWorker(self.waverider)
        self._aero_worker.finished_ok.connect(self._on_aero_ready)
        self._aero_worker.failed.connect(self._on_worker_error)
        self._aero_worker.start()

    # ---------------------------------------------------------------
    #  Validation table: append the MFOF equivalence row
    # ---------------------------------------------------------------
    def _update_validation_table(self, wr, aero_rows):
        # Let the base class fill in geometric + aero rows and update the
        # validation_btn text/colour first.
        super()._update_validation_table(wr, aero_rows)

        t = self.validation_table
        n = t.rowCount()
        t.setRowCount(n + 1)

        ok = self._equivalence_pass
        if ok is None:
            ok_str = "—"
            colour = QColor("#888888")
        elif ok:
            ok_str = "PASS"
            colour = QColor("#6CBB6C")
        else:
            ok_str = "FAIL"
            colour = QColor("#E06C6C")

        t.setItem(n, 0, QTableWidgetItem("MFOF equivalence vs Liu 2019"))
        t.setItem(n, 1, QTableWidgetItem("—"))
        t.setItem(n, 2, QTableWidgetItem("—"))
        t.setItem(n, 3, QTableWidgetItem("< 1e-6 rel"))
        status_item = QTableWidgetItem(ok_str)
        status_item.setForeground(colour)
        t.setItem(n, 4, status_item)

        # Refresh the validation button to count this row too.
        passed = 0
        total  = t.rowCount()
        for i in range(total):
            it = t.item(i, 4)
            if it is not None and it.text() == "PASS":
                passed += 1
        self.validation_btn.setText(f"Validation: {passed}/{total} pass")
        all_green = (passed == total)
        bg = "#2B5B2B" if all_green else "#5B2B2B"
        self.validation_btn.setStyleSheet(
            f"QPushButton {{ background-color: {bg}; color: white; "
            "padding: 6px 10px; }}")


# ===========================================================================
#  Flowfield diagnostics canvas (Phase 3)
# ===========================================================================

class _MFOFDiagnosticsCanvas(FigureCanvas):
    """2-panel matplotlib canvas:

    * Left:  ``(x, r)`` streamline at the centerline plane (z=0). Shows
      the LE / TE points and (if power-law) the body and shock curves.
    * Right: ``delta(z)`` distribution -- the LE deflection angle stored on
      each ``OsculatingPlaneData`` across the half-span. Constant for cone
      and wedge; depends on Ma(z) for variable-Mach cone (which is what we
      have).

    For power-law, the centerline plane's most-recent MOC mesh is
    overlaid as a sparse scatter to give the user a sense of the
    underlying characteristics network.
    """

    def __init__(self):
        self.fig = Figure(figsize=(10, 4))
        self.fig.patch.set_facecolor(BG_DARK)
        self.ax_left  = self.fig.add_subplot(1, 2, 1)
        self.ax_right = self.fig.add_subplot(1, 2, 2)
        for ax in (self.ax_left, self.ax_right):
            ax.set_facecolor(BG_DARK)
            for s in ax.spines.values():
                s.set_color("#666666")
            ax.tick_params(colors="#CCCCCC")
            ax.xaxis.label.set_color("#CCCCCC")
            ax.yaxis.label.set_color("#CCCCCC")
            ax.title.set_color("#FFFFFF")
        super().__init__(self.fig)
        self._show_placeholder()

    # --------------------------------------------------------------
    def _reset(self):
        for ax in (self.ax_left, self.ax_right):
            ax.clear()
            ax.set_facecolor(BG_DARK)
            for s in ax.spines.values():
                s.set_color("#666666")
            ax.tick_params(colors="#CCCCCC")
            ax.xaxis.label.set_color("#CCCCCC")
            ax.yaxis.label.set_color("#CCCCCC")
            ax.title.set_color("#FFFFFF")

    def _show_placeholder(self):
        self._reset()
        for ax, msg in ((self.ax_left,  "Generate to see streamline"),
                        (self.ax_right, "Generate to see deflection profile")):
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    color="#888888", style="italic",
                    transform=ax.transAxes)
        self.fig.tight_layout()
        self.draw()

    # --------------------------------------------------------------
    def plot(self, wr, params, flowfield_type: str = "cone",
             power_law_n: float = 1.0):
        """Re-render with the most recent waverider data."""
        self._reset()
        if wr is None or len(wr.planes) == 0:
            self._show_placeholder()
            return

        # ---- Left panel: centerline-plane streamline ----------------
        # Use the centerline plane (index 0 -> z = 0).
        plane0 = wr.planes[0]
        sl3d = plane0.streamline                 # (n_x, 3) in 3D global coords
        # Project to local (x, in-plane r) frame: r = perpendicular distance
        # from y_LE along -n_base. For the centerline plane n_base = (0, 1, 0)
        # so r = y_LE - y; for off-axis planes the projection is the same as
        # the back-projection in mfof.osculating but inverted.
        x_local = sl3d[:, 0]
        ny = float(plane0.n_base[1])
        nz = float(plane0.n_base[2])
        # local descent = (y_LE - y_3d)/n_y if n_y!=0, else (z_LE - z_3d)/n_z.
        if abs(ny) > 1e-9:
            descent = (plane0.P_LE[1] - sl3d[:, 1]) / ny
        else:
            descent = (plane0.P_LE[2] - sl3d[:, 2]) / nz
        r_local = float(plane0.P_LE[1]) - descent * 0  # placeholder; we plot r as descent below LE
        # We plot streamline as (x, r=descent below LE) for clarity.
        self.ax_left.plot(x_local, descent, color=ACCENT, linewidth=2.0,
                           label=f"Streamline ({flowfield_type})")
        self.ax_left.scatter([plane0.P_LE[0]], [0.0],
                              color="white", s=40, zorder=5, label="LE")
        self.ax_left.scatter([plane0.P_TE[0]], [descent[-1]],
                              color="cyan", s=40, zorder=5, label="TE")

        # If power-law was used, overlay the MOC mesh (sparse scatter).
        if flowfield_type == "power-law" and hasattr(plane0, "flowfield"):
            ff = plane0.flowfield
            grid = getattr(ff, "last_grid", lambda: None)()
            if grid is not None:
                pts = grid.all_points()
                if pts:
                    px = [p["x"] for p in pts]
                    pr_raw = [p["r"] for p in pts]
                    # Convert raw r -> descent below LE via the same mirror
                    # trick used in the streamline trace.
                    pr = [float(plane0.P_LE[1]) - r for r in pr_raw]
                    # Show a thin scatter so the user sees the mesh density.
                    self.ax_left.scatter(px, pr, s=4, color="#888888",
                                          alpha=0.4, label="MOC mesh")

        self.ax_left.set_xlabel("x  (streamwise)")
        self.ax_left.set_ylabel("descent below LE  (= r_LE - r)")
        title_left = f"Centerline streamline ({flowfield_type})"
        if flowfield_type == "power-law":
            title_left += f", n = {power_law_n:.3f}"
        self.ax_left.set_title(title_left)
        self.ax_left.grid(True, alpha=0.2, color="#555555")
        try:
            self.ax_left.legend(facecolor=BG_DARK, edgecolor="#555555",
                                labelcolor="white", fontsize=8)
        except Exception:
            pass

        # ---- Right panel: delta_LE(z) across the half-span ----------
        zs = np.array([p.z for p in wr.planes])
        ds = np.array([p.delta_deg for p in wr.planes])
        self.ax_right.plot(zs, ds, color="#7EC8E3", linewidth=1.8,
                            label="LE deflection angle")
        self.ax_right.fill_between(zs, ds, alpha=0.15, color="#7EC8E3")
        self.ax_right.set_xlabel("z  (m, half-span)")
        self.ax_right.set_ylabel("delta_LE  (deg)")
        self.ax_right.set_title(f"LE deflection vs z  ({flowfield_type})")
        self.ax_right.grid(True, alpha=0.2, color="#555555")

        self.fig.tight_layout()
        self.draw()

