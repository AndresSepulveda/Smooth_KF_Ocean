import dapper.mods as modelling
import numpy as np
from dapper.mods.Lorenz63 import step
from dapper.mods.Lorenz63.extras import dstep_dx

dt = 0.01
tseq = modelling.Chronology(dt=dt, dko=5, K=2000)
x0 = np.array([5., 5., 5.])
Nx = len(x0)
X0 = modelling.GaussRV(C=4, mu=x0)

Dyn = dict()
Dyn['M']= Nx
Dyn['model']= step
Dyn['linear']= dstep_dx
Dyn['noise']= 0.

@modelling.ens_compatible
def obs(x, t):
    k = np.rint(t/dt)
    if k % 25 == 0:
        return x[[0, 1]]
    if k % 5 == 0:
        return x[[0]]


def linear(x, t):
    k = np.rint(t/dt)
    if k % 25 == 0:
        return modelling.utils.direct_obs_matrix(Nx, np.array([0, 1]))
    if k % 5 == 0:
        return modelling.utils.direct_obs_matrix(Nx, np.array([0]))


Obs = modelling.partial_Id_Obs(Nx, np.arange(3))
Obs['noise'] = 4.
Obs['model'] = obs
Obs['linear'] = linear
