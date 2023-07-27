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



def assmExtKF(HMM, xx, yy, x0, B, beta=0.0, infl=1.0):
    print('... assimilating ExtKF ...')
    xp = da.ExtKF(infl=infl, beta=beta)
    # X0 = modelling.GaussRV(C=HMM.X0.C, mu=x0)
    X0 = modelling.GaussRV(C=B, mu=x0[0])
    HMM = modelling.HiddenMarkovModel(HMM.Dyn, HMM.Obs, HMM.tseq, X0)
    E, incr, Pall = xp.assimilate(HMM, xx, yy)
    return E, incr, Pall


def assmEnExtKF(HMM, xx, yy, x0, B, N, beta=0.0, infl=1.0):
    """ExtKF ensemble"""
    print('... assimilating ensemble ExtKF ...')
    incr = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M))
    E = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M))
    xp = da.ExtKF(infl=infl, beta=beta)
    assert len(yy) == N, 'number of obs. sample must be equal to the ensemble size is not correct!'
    for i, y in enumerate(yy):
        X0 = modelling.GaussRV(C=B, mu=x0[i])
        HMM = modelling.HiddenMarkovModel(HMM.Dyn, HMM.Obs, HMM.tseq, X0)
        E[:, i, :], incr[:, i, :], _ =  xp.assimilate(HMM, xx, y)
    return np.squeeze(E), np.squeeze(incr) 


def assmEnExtKS(HMM, xx, yy, x0, B, N, Lag, beta=0.0, beta_s=0.0, infl=1.0):
    """ExtKS ensemble"""
    print('... assimilating ensemble ExtKS ...')
    incr = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M))
    E = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M))
    Pa = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M, HMM.Dyn.M))    ### smoother analysis error covariance
    xp = da.ExtKS(Lag=Lag, beta=beta, beta_s=beta_s, infl=infl)
    assert len(yy) == N, 'number of obs. sample must be equal to the ensemble size is not correct!'
    for i, y in enumerate(yy):
        X0 = modelling.GaussRV(C=B, mu=x0[i])
        HMM = modelling.HiddenMarkovModel(HMM.Dyn, HMM.Obs, HMM.tseq, X0)
        E[:, i, :], incr[:, i, :], Pa[:, i, :, :] =  xp.assimilate(HMM, xx, y)
    Paa = Pa.mean(axis=1)
    return np.squeeze(E), np.squeeze(incr), np.squeeze(Paa)

###
def assmEnSMKS(HMM, xx, yy, x0, B, N, Lag, beta=0.0, beta_s=0.0, gamma=0.7, infl=1.0):
    """SMKS ensemble"""
    print('... assimilating ensemble SMKS ...')
    incr = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M))
    E = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M))
    Ps = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M, HMM.Dyn.M))    ### smoother analysis error covariance
    Pa = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M, HMM.Dyn.M))    ### filter analysis error covariance
    Pf = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M, HMM.Dyn.M))    ### filter forecast error covariance
    xp = da.SMKS(Lag=Lag, beta=beta, beta_s=beta_s, gamma=gamma, infl=infl)
    assert len(yy) == N, 'number of obs. sample must be equal to the ensemble size is not correct!'
    for i, y in enumerate(yy):
        X0 = modelling.GaussRV(C=B, mu=x0[i])
        HMM = modelling.HiddenMarkovModel(HMM.Dyn, HMM.Obs, HMM.tseq, X0)
        #E[:, i, :], incr[:, i, :] =  xp.assimilate(HMM, xx, y)
        #E[:, i, :], incr[:, i, :], Pa[:, i, :, :] =  xp.assimilate(HMM, xx, y)
        #E[:, i, :], incr[:, i, :], Ps[:, i, :, :], Pa[:, i, :, :] =  xp.assimilate(HMM, xx, y)
        E[:, i, :], incr[:, i, :], Ps[:, i, :, :], Pa[:, i, :, :], Pf[:,i,:,:] =  xp.assimilate(HMM, xx, y)
    print('MKS gamma = ', gamma)
    #return np.squeeze(E), np.squeeze(incr) 
    Pss = Ps.mean(axis=1)
    Paa = Pa.mean(axis=1)
    Pff = Pf.mean(axis=1)
    return np.squeeze(E), np.squeeze(incr), np.squeeze(Pss), np.squeeze(Paa), np.squeeze(Pff)
###


def assmFGAT(HMM, xx, yy, x0, N, Lag, B):
    """3DVar ensemble"""
    print('... assimilating 3DVar-FGAT ...')
    incr = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M))
    E = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M))
    Ef = np.zeros((HMM.tseq.K + 1, N, HMM.Dyn.M))
    xp = da.Var3Dinc(Lag=Lag, B=B)
    dko = HMM.tseq.dko
    assert len(yy) == N, 'number of obs. sample must be equal to the ensemble size is not correct!'
    for i, y in enumerate(yy):
        X0 = modelling.GaussRV(C=HMM.X0.C, mu=x0[i])
        HMM = modelling.HiddenMarkovModel(HMM.Dyn, HMM.Obs, HMM.tseq, X0)
        E[:, i, :], incr[dko::dko, i, :], Ef[:, i, :] =  xp.assimilate(HMM, xx, y)
    return np.squeeze(E), np.squeeze(incr), np.squeeze(Ef)


def assmEnKS(HMM, xx, yy, N, infl, Lag):
    # lag -> # of timestep, 14 = 2.8 window
    print('... assimilating EnKS ...')
    xp = da.EnKS('Sqrt', N=N, infl=infl, Lag=Lag)
    E, incr, ff = xp.assimilate(HMM, xx, yy)
    return E, incr, ff


def assmSmoother(E, incr, gamma, K):
    """smoother"""
    # interpolate observation time step to window time step
    print('... assimilating Smoother ...')
    Es = E.copy()
    for i in range(K, 1, -1):
        Es[i-1] = E[i-1] + gamma*(Es[i] - E[i] + incr[i])
    incr = Es - E
    return Es, incr





if __name__ == '__main__':
    dapper.tools.progressbar.disable_progbar = True

    np.random.seed(2)
    s = np.random.randint(0, 10000, size=4000)
    seeds = np.array(list(set(s)))[:100]

    seed = seeds[0]

    beta = 0.05
    infl = 1.0
    gamma = 0.9
    Lag_EnKS = 40
    Lag = 40 

    print("")
    print(f"    gamma = {gamma}, Lag = {Lag}, beta = {beta}, infl = {infl}")
    print("")

    os.makedirs('data', exist_ok=True)
    filename_obs=f'data/obs_{seed}'
    filename=f'data/ensemble_{seed}_{gamma}_{Lag}_beta{beta}_infl{infl}.npz'

    from ex_setup import Dyn, Obs, tseq, X0

    # get truth(xx) and observations(yy)
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    np.random.seed(seed)
    xx, yy = HMM.simulate()

    # generate B
    np.random.seed(seed)
    _, _, Ef_3DVar = assmFGAT(HMM, xx, [yy], X0.sample(1), N=1, Lag=5, B=10*np.eye(3))
    B_clim = (Ef_3DVar-xx).T@(Ef_3DVar-xx)
    B_clim /= 2001


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


    np.random.seed(seed)


    # get initial ensemble 
    E_KS, incr_KS, Ef_KS = assmEnKS(HMM, xx, yy, N=100, infl=1.0, Lag=Lag_EnKS)
    x0 = E_KS[0]
    y0 = yy_3D[0]

    # run extended Kalman filter
    E_ExtKF, incr_ExtKF, Pall =  assmExtKF(HMM, xx, y0, x0, B=B_clim, beta=beta, infl=infl)


    # run extended Kalman filter with N ensemble size
    E_EnExtKF, incr_EnExtKF = assmEnExtKF(HMM, xx, yy_3D, x0, B=B_clim, N=100, beta=beta, infl=infl)

    # run smoother on EnExtKF
    E_S, incr_S = assmSmoother(E_EnExtKF, incr_EnExtKF, gamma=gamma, K=HMM.tseq.K)

    # run extended Kalman smoother with N ensemble size
    E_EnExtKS, incr_EnExtKS, Ps_EnExtKS = assmEnExtKS(HMM, xx, yy_3D, x0, B=B_clim, N=100, Lag=Lag, beta=beta, beta_s=0.0, infl=infl)


    # run modified  Kalman smoother with N ensemble size
    E_EnSMKS, incr_EnSMKS, Ps_EnSMKS, Pa_EnSMKS, Pf_EnSMKS = assmEnSMKS(HMM, xx, yy_3D, x0, B=B_clim, N=100, Lag=Lag, beta=beta, beta_s=0.0, gamma=gamma, infl=infl)


    np.savez_compressed(filename, xx=xx, 
                                         Ef_3DVar=Ef_3DVar, 
                                         E_KS=E_KS, incr_KS=incr_KS, Ef_KS=Ef_KS,
                                         E_S=E_S, incr_S=incr_S, B_clim=B_clim,
                                         E_EnExtKF=E_EnExtKF,incr_EnExtKF=incr_EnExtKF,
                                         E_ExtKF=E_ExtKF, incr_ExtKF=incr_ExtKF,
                                         E_EnExtKS=E_EnExtKS,incr_EnExtKS=incr_EnExtKS,
                                         E_EnSMKS=E_EnSMKS,incr_EnSMKS=incr_EnSMKS,
                                         Pall=Pall,
                                         Ps_EnExtKS=Ps_EnExtKS, Ps_EnSMKS=Ps_EnSMKS,
                                         Pa_EnSMKS=Pa_EnSMKS,Pf_EnSMKS=Pf_EnSMKS)

    # save observation data
    with open(filename_obs, 'wb') as fp:
        pickle.dump(yy, fp)
