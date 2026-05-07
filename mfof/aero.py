"""Aerodynamic evaluator for the MFOF framework.

Phase 2: thin alias around :class:`liu2019.aero.Liu2019AeroEvaluator`. The Liu
evaluator is fully duck-typed on the waverider object (it only calls
``wr.upper_surface(mirror=True)`` and ``wr.lower_surface(mirror=True)``), so
:class:`MFOFWaverider` works in place of :class:`Liu2019Waverider` without
modification.

Future phases may override :meth:`evaluate` to use real-gas / shock-expansion
panel methods instead of pure modified-Newtonian impact theory.
"""

from liu2019.aero import Liu2019AeroEvaluator


class MFOFAeroEvaluator(Liu2019AeroEvaluator):
    """Modified-Newtonian impact-theory aero evaluator for an MFOF waverider.

    Constructor and public methods are inherited verbatim:

    * ``__init__(waverider, gamma=1.4, ref_length=6.0, ref_area=1.0,
                  moment_ref=(0,0,0))``
    * ``evaluate(Ma, alpha_deg=0.0, atm_conditions=None)``
    * ``evaluate_paper_trajectory(progress_callback=None)``
    * ``compare_with_paper()``
    """
    pass
