
# coding: utf-8

# In[1]:

import atmc
import numpy as np


# In[2]:

import pytpc


# In[3]:

efield = np.array([0, 0, 9e3])
bfield = np.array([0, 0, 1.75])
mass_num = 1
charge_num = 1
gas = pytpc.gases.InterpolatedGas('isobutane', 18.)
ens = np.arange(0, 100*1000, dtype='int')
eloss = gas.energy_loss(ens / 1000, mass_num, charge_num)


# In[ ]:

tr = atmc.mcopt_wrapper.PyTracker(mass_num, charge_num, eloss, efield, bfield)


# In[ ]:



