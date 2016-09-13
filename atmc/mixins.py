from . import Tracker, EventGenerator, PadPlane
import numpy as np
from pytpc.gases import InterpolatedGas
from pytpc.constants import degrees
from pytpc.utilities import rot_matrix
import h5py


class ATMCBase(object):
    def __init__(self, *args, **kwargs):
        pass


class TrackerMixin(ATMCBase):
    def __init__(self, config):
        self.gas = InterpolatedGas(config['gas_name'], config['gas_pressure'])
        self._efield = np.array(config['efield'])
        self._bfield = np.array(config['bfield'])
        self.mass_num = config['mass_num']
        self.charge_num = config['charge_num']
        self.beam_enu0 = config['beam_enu0']
        self.beam_mass = config['beam_mass']
        self.beam_charge = config['beam_charge']

        self.tracker = Tracker(mass_num=self.mass_num,
                               charge_num=self.charge_num,
                               beam_enu0=self.beam_enu0,
                               beam_mass=self.beam_mass,
                               beam_charge=self.beam_charge,
                               gas=self.gas,
                               efield=self.efield,
                               bfield=self.bfield,
                               max_en=100)

        super().__init__(config)

    @property
    def efield(self):
        return self._efield

    @efield.setter
    def efield(self, value):
        value = np.asarray(value, dtype='float64')
        self.tracker.efield = value
        self._efield = value

    @property
    def bfield(self):
        return self._bfield

    @bfield.setter
    def bfield(self, value):
        value = np.asarray(value, dtype='float64')
        self.tracker.bfield = value
        self._bfield = value

    @property
    def bfield_mag(self):
        return np.linalg.norm(self.bfield)


class EventGeneratorMixin(ATMCBase):
    def __init__(self, config):
        self._vd = np.array(config['vd'])
        self.mass_num = config['mass_num']
        self.pad_rot_angle = config['pad_rot_angle'] * degrees
        self.padrotmat = rot_matrix(self.pad_rot_angle)
        self.ioniz = config['ioniz']
        self.gain = config['micromegas_gain']
        self.clock = config['clock']
        self.shape = float(config['shape'])
        self._tilt = config['tilt'] * degrees
        self.diff_sigma = config['diffusion_sigma']

        with h5py.File(config['lut_path'], 'r') as hf:
            lut = hf['LUT'][:]
        self.padplane = PadPlane(lut, -0.280, 0.0001, -0.280, 0.0001, self.pad_rot_angle)
        self.evtgen = EventGenerator(self.padplane, self.vd, self.clock, self.shape, self.mass_num,
                                     self.ioniz, self.gain, self.tilt, self.diff_sigma)

        super().__init__(config)

    @property
    def vd(self):
        return self._vd

    @vd.setter
    def vd(self, value):
        value = np.asarray(value, dtype='float64')
        self.evtgen.vd = value
        self._vd = value

    @property
    def tilt(self):
        return self._tilt

    @tilt.setter
    def tilt(self, value):
        self.evtgen.tilt = value
        self._tilt = value
