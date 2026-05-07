"""PSWR-1: Plasma-Sheath-Shaped Variable-Wedge Waverider.

Phase 1: variable-wedge geometry, oblique-shock thermodynamics, inviscid aero.

Internal coordinate convention (per PSWR-1 spec §5):
    x = streamwise (freestream direction, +x downstream)
    y = spanwise
    z = vertical (lift direction, +z up)
    Origin at the apex (nose).

This is *different* from the rest of the GUI, which uses
(x stream, y vertical, z spanwise). Use ``pswr.geometry.variable_wedge.to_gui_frame``
when feeding geometry to the existing matplotlib canvases.
"""

__all__ = ["geometry", "thermo", "aero"]
