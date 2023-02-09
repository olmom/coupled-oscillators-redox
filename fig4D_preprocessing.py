#%% Load results and calculate minimum time for entrainment (mtfe)
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import argrelextrema
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy import signal
from scipy.stats import norm
import scipy.stats
import pandas as pd
import os
import cmath
from poincare import Fits 
from poincare import RhythmicParameters 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = '13'
plt.rcParams['legend.fontsize'] = '12'
plt.rcParams['xtick.labelsize'] = '12'
plt.rcParams['ytick.labelsize'] = '12'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

####################################################
####################################################
# choose Kcoup values to calculate minimum time for entrainment (mtfe)
# (it is done by parts and mtfe are saved by parts in order to plot them
# later because else there is no RAM)
# from K=0.00 to K=0.10
Kcoup = np.arange(0.05, 0.08, 0.01)

####################################################
####################################################

Fzg = np.arange(0.000, 0.330, 0.005)

amp_rr = 0.05
sigma = 0.05
tausd = 1.50 

n_oscs = 100
total_simus = 1 
initial_conditions = 'LC' 

order = 20

dt = 0.01
t = np.arange(0, 14*24, dt)

# Load results
list_xs, list_ys = [], []
list_periods, list_phases = [], []

combinations = np.array([(F,k) for F in Fzg for k in Kcoup])

for c in combinations:
    F, Kc = c
    # load solution
    toload = './results/' +\
        'TTFL={}, '.format(format(F, '.3f')) +\
        'lambda={}, sigma={}, tausd={}, Kcoup={}, {}_ic/simu{}/'.format(
            format(amp_rr, '.2f'),
            format(sigma, '.2f'),
            format(tausd, '.2f'),
            format(Kc, '.3f'),
            initial_conditions,
            int(total_simus))

    solu = np.load(toload + '/solution.npy')
    solu = solu[-int(14*24/dt):,:]
    sol_xdf = pd.DataFrame(solu[:,0::5])
    sol_ydf = pd.DataFrame(solu[:,1::5])
    sol_xdf.index = t
    sol_ydf.index = t

    # load periods
    tau = pd.DataFrame(np.load(toload + 'periods.npy', allow_pickle=True)[0])
    phi = pd.DataFrame(np.load(toload + '/phases.npy'))
    amp = pd.DataFrame(np.load(toload + '/amplitudes.npy'))

    print(
        'simulation {}/{} loaded (amplitudes, phases and circstds'.format(
            int(total_simus), total_simus) +
        ' computed) --- F={}, lambda={}, sig={}, tausd={}, Kcoup={}, '.format( 
            format(F, '.3f'),
            format(amp_rr, '.2f'),
            format(sigma, '.2f'),
            format(tausd, '.2f'),
            format(Kc, '.2f')
            )) 

    list_xs.append(sol_xdf)
    list_ys.append(sol_ydf)
    list_periods.append(tau)
    list_phases.append(phi)

np.set_printoptions(suppress=True) 

# compute minimum time for entrainment for each K,F combination
array_mtfe = [] # first K[0] with all Fzg, then K[1] with all Fzg, ...
dataframes = pd.DataFrame([])
threshold = 0.2
for k in range(len(Kcoup)):
    dataframes2 = pd.DataFrame([])
    mtfe_singleK = []
    for f in range(len(Fzg)):
        # periodogram of mean signal -> gives cleaner results than np.fft
        mean_x = list_xs[k::len(Kcoup)][f].mean(axis=1)
        mean_y = list_ys[k::len(Kcoup)][f].mean(axis=1)
        freq_MF_x, pspec_MF_x = signal.periodogram(mean_x, 1/dt,) 
        freq_MF_y, pspec_MF_y = signal.periodogram(mean_y, 1/dt,) 
    
        period_ft = 1/freq_MF_x 
        y_ft = pspec_MF_x 
        max_period, min_period = 45, 15
        y_ft = y_ft[(period_ft < max_period) & (period_ft > min_period)]
        period_ft = period_ft[(period_ft < max_period) & (period_ft > min_period)]
    
        df2 = {
              'Kcoup': np.repeat(Kcoup[k], len(period_ft)),
              'Fzg': np.repeat(Fzg[f], len(period_ft)),
              'period': period_ft,
              'yft': y_ft,
              'yft_norm': y_ft/np.max(y_ft), 
                }
        df2 = pd.DataFrame(df2)
        print('computed power spectrum for Kc={}, TTFL input={}'.format(
                 format(Kcoup[k], '.3f'), format(Fzg[f], '.3f')))
        dataframes2 = dataframes2.append(df2)     

    dataframes = dataframes.append(dataframes2) #format: K[0] (all Fzg), K[1] (all Fzg), ...

    nrows = len(period_ft) 
    ncols = len(Fzg)
    Z = np.asarray(dataframes2.yft_norm).reshape(ncols, nrows).T
    x = np.asarray(dataframes2.Fzg.unique()) 
    y = np.asarray(dataframes2.period.unique())

    # entrainment period + power associated to each Fzg
    idx_period_entr = np.where(Z[:,-1] == 1.0)
    idx_period_entr_check = np.where(y == 24.0)
    idx_period_entr = idx_period_entr if idx_period_entr == \
            idx_period_entr_check else "error"
    power_period_TTFL = Z[idx_period_entr]

    # we look for the Fzg value that gives power == 1.0 to the TTFL period 
    # but power for the rest of the frequencies ~0
    idx_max_period_entr = np.where(power_period_TTFL[0] == 1.)[0]
    d = Z[:, idx_max_period_entr] #all power values once TTFL signal has power==1
    d = pd.DataFrame(d)
    d2 = (d.iloc[6,:] == 1.0) & (d.drop(6) < threshold) #we want all columns in
        # dataframe d to be true (TTFL period has highest power + rest have power < thresh)
    d2_cols_all_true = d2.all() #tells us whether all elements in each col of d2 are T
    d2_filt = [d2_cols_all_true[i:np.shape(d2_cols_all_true)[0]].all() for i in \
            range(np.shape(d2_cols_all_true)[0])] #check from which column on
        # all columns are all==True --> that means entrainment from that TTFL input
    idx_d2_filt = np.where(np.asarray(d2_filt)==True)
    # but we had got rid of some columns of before to define d, so retrieve them
    idx_d2_mtfe = idx_d2_filt[0][0]
    idx_mtfe = idx_d2_mtfe + idx_max_period_entr[0]
    # now we have the minimum time for entrainment
    mtfe = x[idx_mtfe]
    array_mtfe.append(mtfe)


# convert to array and save
array_mtfe = np.asarray(array_mtfe)
np.save('./results/mtfe/mtfe for Kc between {} and {}'.format(
    format(Kcoup[0],'.2f'), format(Kcoup[-1], '.2f')),
        array_mtfe)

