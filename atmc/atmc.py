import numpy as np
import atmc.mcopt_wrapper as mcopt

__all__ = ['ATMCTracker']


class ATMCTracker(object):
    """A more convenient wrapper around the mcopt functions.

    Parameters
    ----------
    mass_num, charge_num : int
        The mass and charge number of the tracked particle.
    gas : pytpc gas class
        The detector gas.
    efield, bfield : array-like
        The electric and magnetic fields in SI units.
    max_en : int, optional
        The maximum allowable particle energy in MeV. This is used to make the energy lookup
        table for the tracker.
    """

    def __init__(self, mass_num, charge_num, gas, efield, bfield, max_en=100):
        self.mass_num = mass_num
        self.charge_num = charge_num
        self.gas = gas

        ens = np.arange(0, max_en*1000, dtype='int')
        self.eloss = self.gas.energy_loss(ens / 1000, mass_num, charge_num)

        self.efield = np.asarray(efield)
        self.bfield = np.asarray(bfield)

    def track_particle(self, x0, y0, z0, enu0, azi0, pol0, bfield=None):
        """Track a particle.

        Parameters
        ----------
        x0, y0, z0 : float
            The initial position of the particle, in meters.
        enu0 : float
            The initial energy, in MeV/u.
        azi0, pol0 : float
            The initial trajectory angles of the track, in radians.
        bfield : array-like, optional
            The magnetic field, in Tesla. If not provided, `self.bfield` will be used instead.

        Returns
        -------
        ndarray
            The simulated track. The columns are x, y, z, time, energy/nucleon, azimuthal angle, polar angle.
            The positions are in meters, the time is in seconds, and the energy is in MeV/u.

        Raises
        ------
        RuntimeError
            If tracking fails for whatever reason.
        """
        bfield = np.asarray(bfield) if bfield is not None else self.bfield
        return mcopt.track_particle(x0, y0, z0, enu0, azi0, pol0,
                                    self.mass_num, self.charge_num,
                                    self.eloss, self.efield, bfield)

    def minimize(self, ctr0, sigma, true_values, num_iters=10, num_pts=200, red_factor=0.8):
        """Perform Monte Carlo track minimization.

        Parameters
        ----------
        ctr0 : array-like
            The initial guess for the track's parameters. These are (x0, y0, z0, enu0, azi0, pol0, bmag0).
        sigma : array-like
            The initial width of the parameter space in each dimension. The distribution will be centered
            on `ctr0` with a width of `sigma / 2` in each direction.
        true_values : array-like
            The experimental data points, as (x, y, z) triples.
        num_iters : int
            The number of iterations to perform before stopping. Each iteration draws `num_pts` samples
            and picks the best one.
        num_pts : int
            The number of samples to draw in each iteration. The tracking function will be evaluated
            `num_pts * num_iters` times.
        red_factor : float
            The factor to multiply the width of the parameter space by on each iteration. Should be <= 1.

        Returns
        -------
        ctr : ndarray
            The fitted track parameters.
        minChis : ndarray
            The minimum chi^2 value at the end of each iteration.
        allParams : ndarray
            The parameters from all generated tracks. There will be `num_iters * num_pts` rows.
        goodParamIdx : ndarray
            The row numbers in `allParams` corresponding to the best points from each iteration, i.e. the ones whose
            chi^2 values are in `minChis`.

        Raises
        ------
        ValueError
            If a provided array has invalid dimensions.
        RuntimeError
            If tracking fails for some reason.
        """
        ctr0 = np.asarray(ctr0)
        sigma = np.asarray(sigma)
        true_values = np.asarray(true_values)
        return mcopt.MCminimize(ctr0, sigma, true_values, self.mass_num, self.charge_num,
                                self.eloss, self.efield, num_iters, num_pts, red_factor)
