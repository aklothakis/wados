"""Abstract base class for basic flowfields used in osculating-plane construction.

A "basic flowfield" is the 2D axisymmetric flow placed in each osculating plane
of a waverider. Different concrete subclasses represent different generating
bodies:

    ConeFlowfield      -- Sobieczky 1990, Liu 2019 (Taylor-Maccoll cone flow)
    PowerLawFlowfield  -- Rodi 2005, Mazhul 2004 (axisymmetric MOC)        [future]
    WedgeFlowfield     -- 2D oblique-shock (analytical)                    [future]
    BiconicFlowfield   -- two-stage compression                            [future]

The osculating sweep in :mod:`mfof.osculating` instantiates one
``BasicFlowfield`` per spanwise station via a *factory* callable, so future
phases can mix flowfield types in a single waverider.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class StreamlineResult:
    """Streamline traced from leading edge to base plane in one osculating plane.

    Coordinates are in the flowfield's local 2D ``(x, r)`` frame, where ``x``
    is the streamwise direction (parallel to the freestream) and ``r`` is the
    in-plane perpendicular distance from the cone/body axis. The osculating
    sweep is responsible for back-projecting these to global ``(x, y, z)``.
    """
    x_arr: np.ndarray              # x coordinates along the streamline (n_points,)
    r_arr: np.ndarray              # r (in-plane radial) coordinates    (n_points,)
    delta_LE_deg: float            # flow deflection angle at LE (post-shock)
    delta_TE_deg: float            # flow deflection angle at TE (may differ for non-cone)
    Ma_TE: float                   # Mach number at TE (post-shock body-near asymptote)


class BasicFlowfield(ABC):
    """Abstract base for an osculating-plane basic flowfield.

    Each concrete subclass defines one type of generating body and the
    flowfield around it. The osculating sweep instantiates one
    ``BasicFlowfield`` per spanwise station ``z_i`` with the local design
    parameters ``(Ma_i, beta_i)`` plus subclass-specific parameters, then
    asks for the streamline that traces the compression surface in that plane.

    Parameters that are common to all flowfields (``Ma_inf``, ``beta_design``,
    ``gamma``) are stored on the base class. Type-specific parameters
    (cone half-angle for cones, power-law exponent for power-law bodies,
    wedge angle for wedges) are stored on the subclass.

    Subclasses must implement four abstract methods (see below). They MAY
    also expose additional non-abstract methods for diagnostics or
    visualisation (e.g. ``PowerLawFlowfield`` could expose a ``last_grid()``
    method returning the MOC mesh). Such additions are optional and
    type-specific; the osculating sweep only ever calls the four abstract
    methods, so adding diagnostics never breaks the contract.
    """

    def __init__(self, Ma_inf: float, beta_design_deg: float,
                 gamma: float = 1.4):
        self.Ma_inf = float(Ma_inf)
        self.beta_design_deg = float(beta_design_deg)
        self.gamma = float(gamma)

    @abstractmethod
    def name(self) -> str:
        """Short descriptive name (for logging and plots).

        Examples: ``'cone(Ma=6.00,beta=13.00)'``, ``'power-law n=0.7'``.
        """

    @abstractmethod
    def attached_shock_check(self) -> tuple:
        """Verify that an attached shock exists for the current parameters.

        Returns ``(is_valid: bool, message: str)``. ``message`` describes the
        failure reason if ``is_valid`` is ``False``
        (e.g. ``'beta < Mach angle'`` or ``'beta > detachment'``).
        """

    @abstractmethod
    def deflection_angle_deg(self) -> float:
        """Flow deflection angle theta (degrees) immediately downstream of the
        shock at this Mach and shock angle, evaluated specifically AT THE
        LEADING EDGE. This is the angle the streamline makes with the
        freestream immediately after passing through the shock.

        Semantics by subclass
        ---------------------
        * **Cone (Taylor-Maccoll)**: this is the cone half-angle
          ``delta_c``, which is *constant* along the cone surface — the
          streamline keeps this slope all the way to the base plane.
        * **Wedge (2D oblique shock)**: this is the wedge half-angle, which
          is also *constant* throughout the post-shock flow.
        * **Power-law body (n != 1)**: this is the body slope at ``x_LE``
          *only*. The streamline slope changes downstream because the body
          is curved. The osculating sweep MUST NOT assume this angle is
          constant when using a non-cone, non-wedge flowfield.

        Callers that need the streamline's actual shape (not just its
        starting angle) must use :meth:`trace_streamline` to get the full
        ``(x, r)`` trajectory.
        """

    @abstractmethod
    def trace_streamline(self, x_LE: float, r_LE: float, x_end: float,
                         n_points: int = 100) -> StreamlineResult:
        """Trace the leading-edge streamline from ``(x_LE, r_LE)`` to ``x = x_end``
        in the flowfield's local ``(x, r)`` plane.

        For a cone (Taylor-Maccoll) the streamline is a straight line at angle
        ``delta_c`` from the freestream — analytical.

        For a power-law body the streamline is curved and requires numerical
        integration through the MOC mesh.

        Returns a :class:`StreamlineResult` with ``x_arr``, ``r_arr``, and
        post-trace state.
        """
