import numpy as np
import atmc.mcopt_wrapper as mcopt
from atmc.mcopt_wrapper import PadPlane, EventGenerator, Minimizer

__all__ = ['Tracker', 'Minimizer', 'PadPlane', 'EventGenerator']


def find_vertex_energy(beam_intercept, beam_enu0, beam_mass, beam_chg, gas):
    ei = beam_enu0 * beam_mass
    ri = gas.range(ei, beam_mass, beam_chg)  # this is in meters
    rf = ri - (1.0 - beam_intercept)
    ef = gas.inverse_range(rf, beam_mass, beam_chg)
    return ef


class Tracker(mcopt.Tracker):
    """A more convenient wrapper around the mcopt functions.

    Parameters
    ----------
    mass_num, charge_num : int
        The mass and charge number of the tracked particle.
    beam_enu0 : float
        The initial energy per nucleon of the beam projectile.
    beam_mass, beam_charge : int
        The mass and charge numbers of the beam projectile.
    gas : pytpc gas class
        The detector gas.
    efield, bfield : array-like
        The electric and magnetic fields in SI units.
    max_en : int, optional
        The maximum allowable particle energy in MeV. This is used to make the energy lookup
        table for the tracker.
    """

    def __new__(cls, mass_num, charge_num, beam_enu0, beam_mass, beam_charge, gas, efield, bfield, max_en=100):
        ens = np.arange(0, max_en*1000, dtype='int')
        eloss = gas.energy_loss(ens / 1000, mass_num, charge_num)

        zs = np.arange(0, 1000, dtype='int')
        en_vs_z = find_vertex_energy(zs / 1000., beam_enu0, beam_mass, beam_charge, gas)

        mcopt_gas = mcopt.Gas(eloss, en_vs_z)

        return super().__new__(cls, mass_num, charge_num, mcopt_gas, efield, bfield)
