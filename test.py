import sys
import glob
sys.path.insert(1, glob.glob('./build/lib*')[0])
sys.path.append('..')

import numpy as np
import atmc
import pytpc
from pytpc.constants import degrees
import timeit
import montecarlo

x0 = 0.
y0 = 0.
z0 = 0.49
enu0 = 1.36
azi0 = -3.110857
pol0 = 1.951943
mass_num = 1
charge_num = 1
bmag0 = 2.0

efield = np.array([0, 0, 9e3])
# bfield = np.array([0., 0.18346418, 1.74554505])
bfield = np.array([0., 0., bmag0])

gas = pytpc.gases.InterpolatedGas('isobutane', 18)
ens = np.arange(100000, dtype='double')
eloss = gas.energy_loss(ens/1000, mass_num, charge_num)

track = atmc.track_particle(x0, y0, z0, enu0, azi0, pol0, mass_num, charge_num, eloss, efield, bfield)
print('trlen =', track.shape)
print(track)

dataIdx = np.random.choice(len(track), size=500, replace=False)
data = track[dataIdx, :3].copy()
data2 = track[dataIdx, :4].copy()

ctr0 = np.array([x0, y0, z0, enu0, azi0, pol0, bmag0])
sigma = np.array([0., 0., 0.15, 0.5, 10*degrees, 10*degrees, 0.1])

begin = timeit.default_timer()
ctr, minchis, allparams, goodidx = atmc.MCminimize(ctr0, sigma, data, mass_num, charge_num, eloss, efield, 15, 200, 0.8)
end = timeit.default_timer()

print(end - begin)
print(ctr)

consts = {'efield': efield,
          'eloss': eloss,
          'mass_num': mass_num,
          'charge_num': charge_num}

begin = timeit.default_timer()
ctr2 = montecarlo.mcmin(montecarlo.run_track, ctr0, sigma, func_kwargs={'true_values': data2, 'consts': consts},
                        num_iters=15, num_pts=200, red_factor=0.8)
end = timeit.default_timer()

print(end - begin)
print(ctr2)
