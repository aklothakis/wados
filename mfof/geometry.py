"""3D geometry assembly for the MFOF framework.

:class:`MFOFWaverider` is a thin subclass of :class:`liu2019.geometry.Liu2019Waverider`.
It accepts an :class:`mfof.osculating.OsculatingPlaneSet` (whose
:class:`OsculatingPlaneData` carries the same attributes as Liu's per-plane
record, plus ``flowfield`` and ``delta_deg``). All geometric methods --
``upper_surface``, ``lower_surface``, ``volume``, ``wetted_area``,
``planform_area``, ``base_area``, ``volumetric_efficiency``, ``export_stl``,
``export_obj``, ``export_step``, ``summary`` -- are inherited verbatim.

The single override is :meth:`cone_angle_array`, which Liu's class names after
its only flowfield (``cone_angle_array`` reading ``p.delta_c``); MFOF's
generalised name is :meth:`deflection_angle_array`, but we retain
``cone_angle_array`` as a back-compat alias since per-plane data carries
both ``p.delta_deg`` and the alias ``p.delta_c``.
"""

import numpy as np

from liu2019.geometry import Liu2019Waverider

from .osculating import OsculatingPlaneSet, build_all_osculating_planes


class MFOFWaverider(Liu2019Waverider):
    """3D waverider assembled from an MFOF :class:`OsculatingPlaneSet`.

    Constructed identically to :class:`Liu2019Waverider`. The only added
    method is :meth:`deflection_angle_array`, which is the framework-neutral
    name for the spanwise distribution of post-shock deflection angles
    (== cone half-angle when every plane uses :class:`ConeFlowfield`).
    """

    def __init__(self, planes: OsculatingPlaneSet, params: dict,
                 n_x: int = 100):
        super().__init__(planes, params, n_x=n_x)

    # ------------------------------------------------------------------
    def deflection_angle_array(self):
        """Return ``(z_array, delta_deg_array)`` for the half-span planes.

        Generalises :meth:`Liu2019Waverider.cone_angle_array` to any
        :class:`BasicFlowfield` subclass. For all-cone configurations the
        values are identical to the inherited ``cone_angle_array``.
        """
        zs = np.array([p.z for p in self.planes])
        d  = np.array([p.delta_deg for p in self.planes])
        return zs, d


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_mfof_waverider(params: dict, flowfield_factory,
                         n_z: int = 200, n_x: int = 100) -> MFOFWaverider:
    """Build an :class:`MFOFWaverider` end-to-end with the given flowfield factory.

    Mirrors :func:`liu2019.geometry.build_liu2019_waverider`.

    For Liu 2019 reproduction pass an all-cone factory:

    >>> from mfof.cone_flowfield import ConeFlowfield
    >>> beta = params['beta_deg']
    >>> def all_cone(z, Ma_z):
    ...     return ConeFlowfield(Ma_z, beta, params.get('gamma', 1.4))
    >>> wr = build_mfof_waverider(params, all_cone, n_z=200, n_x=100)
    """
    planes = build_all_osculating_planes(
        params, flowfield_factory, n_z=n_z, n_x_per_strip=n_x)
    return MFOFWaverider(planes, params, n_x=n_x)
