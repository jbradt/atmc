import numpy as np
import atmc.mcopt_wrapper as mcopt
# from atmc.mcopt_wrapper import PadPlane, EventGenerator, Minimizer

# __all__ = ['Tracker', 'Minimizer', 'PadPlane', 'EventGenerator']


# class Tracker(mcopt.Tracker):
#     """A more convenient wrapper around the mcopt functions.
#
#     Parameters
#     ----------
#     mass_num, charge_num : int
#         The mass and charge number of the tracked particle.
#     gas : pytpc gas class
#         The detector gas.
#     efield, bfield : array-like
#         The electric and magnetic fields in SI units.
#     max_en : int, optional
#         The maximum allowable particle energy in MeV. This is used to make the energy lookup
#         table for the tracker.
#     """
#
#     def __init__(self, mass_num, charge_num, gas, efield, bfield, max_en=100):
#         ens = np.arange(0, max_en*1000, dtype='int')
#         eloss = gas.energy_loss(ens / 1000, mass_num, charge_num)
#         super().__init__(mass_num, charge_num, eloss, efield, bfield)
