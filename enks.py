import dapper as dpr
import dapper.tools.progressbar
import dapper.da_methods as da
import dapper.mods as modelling

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp
import pickle
import os
import time


def assmEnKS(HMM, xx, yy, N, infl, Lag):
    print('... assimilating EnKS ...')
    xp = da.EnKS('Sqrt', N=N, infl=infl, Lag=Lag)
    E, incr, ff = xp.assimilate(HMM, xx, yy)
    return E, incr, ff

def assmMEnKS(HMM, xx, yy, N, infl, Lag, gamma):
    print('... assimilating MEnKS ...')
    xp = da.MEnKS_Pa('Sqrt', N=N, infl=infl, Lag=Lag, gamma=gamma)
    E, incr, Pa = xp.assimilate(HMM, xx, yy)
    return E, incr, Pa


def assmEnKF(HMM, xx, yy, N, infl):
    # lag -> # of timestep, 14 = 2.8 window
    print('... assimilating EnKF ...')
    xp = da.EnKF('Sqrt', N=N, infl=infl)
    E, incr = xp.assimilate(HMM, xx, yy)
    return E, incr


#====================================================================

if __name__ == '__main__':
    dapper.tools.progressbar.disable_progbar = True

    np.random.seed(2)
    s = np.random.randint(0, 10000, size=4000)
    seeds = np.array(list(set(s)))[:100]

    seed = seeds[0]

    beta = 0.05
    infl = 1.0
    gamma = 0.9
    Lag = 40

    print("")
    print(f"    gamma = {gamma}, Lag = {Lag}, beta = {beta}, infl = {infl}")
    print("")

    os.makedirs('data', exist_ok=True)
    filename_obs=f'data/obs_{seed}'
    filename = f'data/enks_{seed}_Lag{Lag}_infl{infl}_gamma{gamma}.npz'

    from ex_setup import Dyn, Obs, tseq, X0

    # get truth(xx) and observations(yy)
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    np.random.seed(seed)
    xx, yy = HMM.simulate()

    # generate B
    np.random.seed(seed)

#    use same B matrix as in the extended KS - need ro run extks.py first 
    fb =  np.load(f'data/B_clim_8192.npz')
    B_clim = fb["B_clim"]

    # perturb observations
    np.random.seed(seed)
    yy_3D = []
    Obs = HMM.Obs
    for i in range(100):
        yy_3D_in = []
        for k, ko, t, dt in HMM.tseq.ticker:
            if ko is not None:
                yy_3D_in.append(yy[ko])
                Obs.noise = modelling.GaussRV(C=Obs.noise.C.diag[0], M=yy_3D_in[ko].shape[-1])
                yy_3D_in[ko] = yy_3D_in[ko] + Obs.noise.sample(1)
        yy_3D.append(yy_3D_in)

    # run ensemble Kalman smoother
    np.random.seed(seed)
    E_KS, incr_KS, Ef_KS = assmEnKS(HMM, xx, yy, N=100, infl=1.0, Lag=Lag)

    # get initial ensemble 
    x0 = E_KS[0]
    y0 = yy_3D[0]


    # run extended Kalman filter N times
    E_EnKF, incr_EnKF = assmEnKF(HMM, xx, yy, N=100, infl=1.0)

    # run modified ensemble kalman smoother 
    E_MEnKS, incr_MEnKS, Pa_MEnKS = assmMEnKS(HMM, xx, yy, N=100, infl=1.0, Lag=Lag, gamma=gamma)


    np.savez_compressed(filename, xx=xx, E_EnKF=E_EnKF, incr_EnKF=incr_EnKF,
            E_EnKS=E_KS, incr_EnKS=incr_KS,
            E_MEnKS=E_MEnKS, incr_MEnKS=incr_MEnKS,
            Pa_MEnKS=Pa_MEnKS)

