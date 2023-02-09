import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import argrelextrema
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy import signal
from scipy.stats import norm
import pandas as pd
import os
from poincare import PoincareOscillator
from poincare import sdeIntegrator
from poincare import Fits 
from poincare import RhythmicParameters 

######### CHOOSE PARAMETERS FOR SIMULATIONS ###########
#########-----------------------------------###########
initial_conditions = 'LC' #'LC'(fig2 or 4) or 'random'(for3)
n_oscs = 100 #number of oscillators in one 'redox ensemble'
total_simus = 1 #how many realizations to run (and remove noise comp.)
                #1 for 4, 5 for 3A-D, 50 for 2 

# all should be numpy arrays, even if with one entry
Fzg = np.arange(0.000,0.260,0.005) #TTFL input
Kcoup = np.array([0.05]) #inter-redox-oscillator coupling
sigma = np.array([0.05]) #noise in x and y
tausd = np.array([1.50]) #std. dev. of period heterogeneity (in h)
amp_rr = np.array([0.05]) #amplitude relaxation rate
#####################################################
#####################################################

# combinations of Fzg, sigma, tausd, amp_rr and Kcoup values 
# for which stochastic diff. eqs (SDEs) should be solved 
combinations = np.array([(F,i,j,l,k) \
        for F in Fzg for i in sigma for j in tausd \
        for l in amp_rr for k in Kcoup])

amplitudes_x, amplitudes_y = [], []
for c in combinations:
    F, sig, tsd, l, Kc = c
    # construct Poincare Oscillator and stochastic Object
    poincare_obj = PoincareOscillator(amp = 1,
                                      lam = l,
                                      eps = 0.05,
                                      tau = 24.23 + tsd*np.random.randn(n_oscs),
                                      F_zg = F,
                                      K_coup = Kc)
    sdeint_obj = sdeIntegrator(sig_x = sig,
                               sig_y = sig)
    # define initial conditions to solve SDEs
    if initial_conditions == 'LC':
        # initial conditions in the limit cycle
        y0 = np.tile(np.array([1.0,0.0,
            0.10035983,  0.38609676,  2.36251403]),
            #these are limit cycle values from Goodwin-like (Gonze) model
            n_oscs).flatten()
    elif initial_conditions == 'random':
        # random initial conditions for redox (not for Gonze model)
        y0 = np.asarray([np.array([np.random.uniform(-1,1), 
                         np.random.uniform(-1,1), 
                         0.10035983,  0.38609676,  2.36251403]) 
                         for x in range(n_oscs)]).flatten()
    else:
        'check "initial_conditions": must be "LC" or "random"'
    print(y0)

    # define time and time step for integration
    if (len(Fzg) == 1) & (Fzg[0] == 0.0): 
        dt = 0.12 #solver works for this dt when no TTFL input
        t = np.arange(0, 2400.0, dt) 
    else: #for simulations with TTFL input I need lower dt 
        dt = 0.01 #(else we get integration errors in Gonze model)
        t = np.arange(0, 1200.0, dt) 
    N = len(t) #length of time span
    
    # solve SDE
    for i in range(total_simus): # total number of realizations to run
        print(
            'Fzg={}, '.format(
                format(F, '.3f')) +\
            'lam={}, sig={}, tausd={}, Kcoup={} --- running simulation = {}'.format(
            format(l, '.2f'),
            format(sig, '.2f'),
            format(tsd, '.2f'),
            format(Kc, '.3f'),                    
            i+1))
        ss = np.random.SeedSequence()
        generator = np.random.default_rng(ss)

        eqs = poincare_obj.f_PoincareGonze
        diff = sdeint_obj.G_PoincareGonze
        #dW = sdeIntegrator.deltaW(N-1, np.shape(y0)[0], dt, generator)
        m = np.shape(y0)[0]
        dW = sdeint_obj.deltaW(N-1, m, dt, generator)
        #solution = sdeIntegrator.itoEuler(eqs, diff, y0, t, dW=dW)
        solution = sdeint_obj.itoEuler(eqs, diff, y0, t, dW=dW, m=m)

        # save results
        # and create path to save the results
        if (len(Fzg) == 1) & (Fzg[0] == 0.0): #solutions without TTFL input
            solution = solution
            mypath = \
                './results/'+\
                    'no TTFL, ' +\
                    'lambda={}, sigma={}, tausd={}, Kcoup={}, {}_ic/simu{}/'.format( 
                    format(l, '.2f'),
                    format(sig, '.2f'),
                    format(tsd, '.2f'),
                    format(Kc, '.3f'),                    
                    initial_conditions,
                    format(i+1))

        elif len(Fzg) != 1: 
            #solutions with TTFL input: not whole solution is saved, 
            #else insufficient RAM to process the results later on
            solution = solution[0:int(24*14/dt), :] #14 days of solution
            t = t[0:int(24*14/dt)]
            mypath = \
                './results/'+\
                    'TTFL={}, '.format(format(F, '.3f')) +\
                    'lambda={}, sigma={}, tausd={}, Kcoup={}, {}_ic/simu{}/'.format( 
                    format(l, '.2f'),
                    format(sig, '.2f'),
                    format(tsd, '.2f'),
                    format(Kc, '.3f'),                    
                    initial_conditions,
                    format(i+1))

        elif (len(Fzg) == 1) & (Fzg[0] != 0.0) : 
            #for fig4D we need transients to calculate the minimum
            # time for entrainment, so I save more days of the solution
            solution = solution[0:int(24*30/dt), :] #30 days of solution
            t = t[0:int(24*30/dt):]
            mypath = \
                './results/'+\
                    'TTFL={}, '.format(format(F, '.3f')) +\
                    'lambda={}, sigma={}, tausd={}, Kcoup={}, {}_ic/simu{}/'.format( 
                    format(l, '.2f'),
                    format(sig, '.2f'),
                    format(tsd, '.2f'),
                    format(Kc, '.3f'),                    
                    initial_conditions,
                    format(i+1))

        print(mypath)
        isExist = os.path.exists(mypath)
        if not isExist:  
            os.makedirs(mypath)

        # determine amplitudes, periods and phases
        sol_xdf = pd.DataFrame(solution[:,0::5])
        sol_ydf = pd.DataFrame(solution[:,1::5])

        amp = pd.DataFrame(np.sqrt(sol_xdf**2 + sol_ydf**2))
        phi = pd.DataFrame(np.arctan2( sol_ydf , sol_xdf ))
        tau = RhythmicParameters().periods( t, sol_xdf )    
        
        np.save(mypath + '/solution.npy'.format(i+1),solution)
        np.save(mypath + '/amplitudes.npy'.format(i+1),amp)
        np.save(mypath + '/phases.npy'.format(i+1),phi)
        np.save(mypath + '/periods.npy'.format(i+1),tau)



