#%% Load results and plot Figure 4B
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

########### PARAMETER CHOICE FOR FIG4 ###############
#####################################################
# for figure 4B: fixed Kc and variable TTFL input 
Fzg = np.arange(0.000, 0.265, 0.005)
Fzg_timeseries = [0.0, 0.07, 0.25]
Kcoup = np.array([0.05])

# for figure 4C: fixed TTFL input and variable Kc
Fzg_C = np.array([0.25])
Kcoup_C = np.arange(0.000, 0.160, 0.005)
Kcoup_timeseries = [0.0, 0.05, 0.105, 0.14]
Kcoup_C = np.around(Kcoup_C,3)

# common
amp_rr = np.array([0.05]) 
sigma = 0.05
tausd = np.array([1.50])

n_oscs = 100
total_simus = 1 
initial_conditions = 'LC' 

order = 20 #number of points around rel. max/min to estimate the extrema
####################################################
####################################################

dt = 0.01 
t = np.arange(0, 14*24, dt)

# Load results
list_xs, list_ys = [], []
list_periods, list_phases = [], []

combinations = np.array([(F,s,l,ts,k) \
        for F in Fzg \
        for s in np.arange(total_simus)+1 \
        for l in amp_rr for ts in tausd for k in Kcoup])

for c in combinations:
    F, simu, l, tsd, Kc = c
    # load solution
    toload = './results/' +\
        'TTFL={}, '.format(format(F, '.3f')) +\
        'lambda={}, sigma={}, tausd={}, Kcoup={}, {}_ic/simu{}/'.format(
            format(l, '.2f'),
            format(sigma, '.2f'),
            format(tsd, '.2f'),
            format(Kc, '.3f'),
            initial_conditions,
            int(simu))

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
            int(simu), total_simus) +
        ' computed) --- F={}, lambda={}, sig={}, tausd={}, Kcoup={}, '.format( 
            format(F, '.3f'),
            format(l, '.2f'),
            format(sigma, '.2f'),
            format(tsd, '.2f'),
            format(Kc, '.2f')
            )) 

    list_xs.append(sol_xdf)
    list_ys.append(sol_ydf)
    list_periods.append(tau)
    list_phases.append(phi)


# PLOT fig4B: how does period of mean-field change as a function of TTFL input?
# in this plot, Kc is fixed to Kc=0.05 (weak coupling)
fig4B = plt.figure(figsize=(11,3.5)) 
ax1 = fig4B.add_subplot(131)

dataframes = pd.DataFrame([])
for f in range(len(Fzg)):
    # periodogram of mean signal -> gives cleaner results than np.fft
    mean_x = list_xs[f].mean(axis=1)
    mean_y = list_ys[f].mean(axis=1)
    freq_MF_x, pspec_MF_x = signal.periodogram(mean_x, 1/dt,) 
    freq_MF_y, pspec_MF_y = signal.periodogram(mean_y, 1/dt,)

    period_ft = 1/freq_MF_x 
    y_ft = pspec_MF_x 
    max_period, min_period = 45, 15
    y_ft = y_ft[(period_ft < max_period) & (period_ft > min_period)]
    period_ft = period_ft[(period_ft < max_period) & (period_ft > min_period)]

    # save power of each period for each TTFL input
    df = {'Fzg': np.repeat(Fzg[f], len(period_ft)),
          'period': period_ft,
          'yft': y_ft,
          'yft_norm': y_ft/np.max(y_ft), 
            }
    df = pd.DataFrame(df)
    print('computed power spectrum for Kc={}, TTFL input={}'.format(
          format(Kcoup[0], '.3f'), format(Fzg[f], '.3f')))
    dataframes = dataframes.append(df)     

# plot left panel: period of mean-field as a function of TTFL input for Kc=0.05
# color-code based on power of the period in the periodogram
nrows = len(period_ft) 
ncols = len(Fzg)
Z = np.asarray(dataframes.yft_norm).reshape(ncols, nrows).T
x = np.asarray(dataframes.Fzg.unique()) 
y = np.asarray(dataframes.period.unique())
im = ax1.pcolormesh(x, y, Z, vmin=Z.min(), vmax=Z.max(), cmap="Blues", 
        shading='nearest')
ax1.set_xlabel('strength of TTFL input (a.u.)')
ax1.set_ylabel('period of\nmean field (h)')
ax1.set_ylim([18,35])
ax1.set_xlim([0, 0.25])
ax1.set_yticks([20,24,28,32])
ax1.set_aspect(0.5/ax1.get_data_ratio(), adjustable='box')
axins = inset_axes(ax1,
                    width="100%",  
                    height="10%",
                    loc='lower center',
                    borderpad=-5.5
                   )
fig4B.colorbar(im, ax=ax1, orientation="horizontal", cax=axins)
ax1.collections[0].colorbar.set_label(
        'power (norm. to max. power)')
ax1.text(0.01,35.99, "fixed inter-osc. coupling $K_c={}$".format(format(Kcoup[0])))

# timeseries for different values of TTFL input (Kc=0.05): middle, right panels
ax2 = fig4B.add_subplot(132)
ax3 = fig4B.add_subplot(133)

idxs = np.where(np.isin(Fzg, np.asarray(Fzg_timeseries)) )[0]

days_shift = 5 #show 5 days in timeseries of middle panel
t_shift = t[0:int(24*days_shift/dt)]
for i in range(n_oscs-1, -1, -1): 
    ax2.plot(t_shift/24, list_xs[idxs[0]].iloc[-int(24*days_shift/dt):,i], 
            c='sandybrown', lw=0.5, alpha=0.05)
    ax2.plot(t_shift/24, list_xs[idxs[2]].iloc[-int(24*days_shift/dt):,i], 
            c='darkseagreen', lw=0.5, alpha=0.05)
    ax3.plot(t/24, list_xs[idxs[1]].iloc[:,i], 
            c='silver', lw=0.5, alpha=0.05)
ax2.plot(t_shift/24, list_xs[idxs[0]].iloc[-int(24*days_shift/dt):].mean(axis=1), 
        c='darkorange', label="TTFL input $={}$\,a.u.".format(format(Fzg[idxs[0]], '.2f')))
ax2.plot(t_shift/24,
        list_xs[idxs[2]].iloc[-int(24*days_shift/dt):].mean(axis=1), 
        c='green', label="TTFL input $={}$\,a.u.".format(format(Fzg[idxs[2]], '.2f')))
ax3.plot(t/24, list_xs[idxs[1]].mean(axis=1), c='k',
        label="TTFL input $={}$\,a.u.".format(format(Fzg[idxs[1]], '.2f')))

ax2.set_xticks([0, 1, 2, 3, 4, 5])
ax2.legend(framealpha=0, loc="upper left")
ax3.legend(framealpha=0, loc="upper left")
ax3.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
ax2.set_ylim([-2., +2.]); ax3.set_ylim([-2., +2.])
ax2.set_ylabel('simulated $x$'); ax3.set_ylabel('simulated $x$')
ax2.set_xlabel('time (days)'); 
ax3.set_xlabel('time (days)'); 
ax2.set_aspect(0.75/ax2.get_data_ratio(), adjustable='box')
ax3.set_aspect(0.75/ax3.get_data_ratio(), adjustable='box')

fig4B.subplots_adjust(
        top=1.0,
        bottom=0.3,
        left=0.105,
        right=0.985,
        hspace=0.2,
        wspace=0.28)

####################################################################################
####################################################################################

#%% Load results and plot Figure 4C
# Load results for fig4C
list_xs_2, list_ys_2 = [], []
list_periods_2, list_phases_2 = [], []

combinations_2 = np.array([(F,s,l,ts,k) \
        for F in Fzg_C \
        for s in np.arange(total_simus)+1 \
        for l in amp_rr for ts in tausd for k in Kcoup_C])

for c in combinations_2:
    F, simu, l, tsd, Kc = c
    # load solution
    toload = './results/' +\
        'TTFL={}, '.format(format(F, '.3f')) +\
        'lambda={}, sigma={}, tausd={}, Kcoup={}, {}_ic/simu{}/'.format(
            format(l, '.2f'),
            format(sigma, '.2f'),
            format(tsd, '.2f'),
            format(Kc, '.3f'),
            initial_conditions,
            int(simu))

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
            int(simu), total_simus) +
        ' computed) --- F={}, lambda={}, sig={}, tausd={}, Kcoup={}, '.format( 
            format(F, '.3f'),
            format(l, '.2f'),
            format(sigma, '.2f'),
            format(tsd, '.2f'),
            format(Kc, '.3f')
            )) 

    list_xs_2.append(sol_xdf)
    list_ys_2.append(sol_ydf)
    list_periods_2.append(tau)
    list_phases_2.append(phi)


# PLOT fig4C: how does period of mean-field change as a function of Kcoup?
# in this plot, TTFL is fixed to TTFL=0.25 (entrains a weakly coupled system)

# PLOT 
fig4C = plt.figure(figsize=(11,3.5)) 
ax4 = fig4C.add_subplot(131)

dataframes2 = pd.DataFrame([])
for k in range(len(Kcoup_C)):
    # periodogram of mean signal -> gives cleaner results than np.fft
    mean_x = list_xs_2[k].mean(axis=1)
    mean_y = list_ys_2[k].mean(axis=1)
    freq_MF_x, pspec_MF_x = signal.periodogram(mean_x, 1/dt,) 
    freq_MF_y, pspec_MF_y = signal.periodogram(mean_y, 1/dt,) 

    period_ft = 1/freq_MF_x 
    y_ft = pspec_MF_x 
    max_period, min_period = 45, 15
    y_ft = y_ft[(period_ft < max_period) & (period_ft > min_period)]
    period_ft = period_ft[(period_ft < max_period) & (period_ft > min_period)]
    
    # save power of each period for each Kc input (varies)
    df2 = {
          'Kcoup': np.repeat(Kcoup_C[k], len(period_ft)),
          'Fzg': np.repeat(Fzg_C[0], len(period_ft)),
          'period': period_ft,
          'yft': y_ft,
          'yft_norm': y_ft/np.max(y_ft), 
            }
    df2 = pd.DataFrame(df2)
    print('computed power spectrum for Kc={}, TTFL input={}'.format(
            format(Kcoup_C[k], '.3f'), format(Fzg_C[0], '.3f')))
    dataframes2 = dataframes2.append(df2)     

# plot left panel: period of mean-field as a function of Kc for TTFL=0.25 
# color-code based on power of the period in the periodogram
nrows = len(period_ft) 
ncols = len(Kcoup_C)
Z2 = np.asarray(dataframes2.yft_norm).reshape(ncols, nrows).T
x2 = np.asarray(dataframes2.Kcoup.unique()) 
y2 = np.asarray(dataframes2.period.unique())
im = ax4.pcolormesh(x2, y2, Z2, vmin=Z2.min(), vmax=Z2.max(), cmap="Reds", 
        shading='nearest')
ax4.set_xlabel('coupling strength $K_c$ (a.u.)')
ax4.set_ylabel('period of\nmean field (h)')
ax4.set_ylim([18,40])
ax4.set_xlim([0, 0.15])
ax4.set_yticks([20,24,28,32,36])
ax4.set_aspect(0.5/ax4.get_data_ratio(), adjustable='box')
axins = inset_axes(ax4,
                    width="100%",  
                    height="10%",
                    loc='lower center',
                    borderpad=-5.5
                   )
fig4C.colorbar(im, ax=ax4, orientation="horizontal", cax=axins)
ax4.collections[0].colorbar.set_label(
        'power (norm. to max. power)')
ax4.text(0.005,40.99, "fixed TTFL input $={}$".format(format(Fzg_C[0])))

# timeseries for different values of Kc input (TTFL=0.25): middle, right panels
ax5 = fig4C.add_subplot(132)
ax6 = fig4C.add_subplot(133)

idxs = np.where(np.isin(Kcoup_C, np.asarray(Kcoup_timeseries)) )[0]

days_shift = 5 #show only 5 days in middle panel
t_shift = t[0:int(24*days_shift/dt)]
for i in range(n_oscs-1, -1, -1): 
    ax5.plot(t_shift/24, list_xs_2[idxs[0]].iloc[-int(24*days_shift/dt):,i], 
            c='cornflowerblue', lw=0.5, alpha=0.05)
    ax5.plot(t_shift/24, list_xs_2[idxs[1]].iloc[-int(24*days_shift/dt):,i], 
            c='lightcoral', lw=0.5, alpha=0.05)
    ax6.plot(t/24, list_xs_2[idxs[2]].iloc[:,i], 
            c='silver', lw=0.5, alpha=0.05)
ax5.plot(t_shift/24, list_xs_2[idxs[0]].iloc[-int(24*days_shift/dt):].mean(axis=1), 
        c='blue', label="$K_c={}$\,a.u.".format(format(Kcoup_C[idxs[0]], '.2f')))
ax5.plot(t_shift/24,
        list_xs_2[idxs[1]].iloc[-int(24*days_shift/dt):].mean(axis=1), 
        c='red', label="$K_c={}$\,a.u.".format(format(Kcoup_C[idxs[1]], '.2f')))
ax6.plot(t/24, list_xs_2[idxs[2]].mean(axis=1), c='k',
        label="$K_c={}$\,a.u.".format(format(Kcoup_C[idxs[2]], '.2f')))

ax5.set_xticks([0, 1, 2, 3, 4, 5])
ax5.legend(framealpha=0, loc="upper left")
ax6.legend(framealpha=0, loc="upper left")
ax6.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
ax5.set_ylim([-2.5, +2.5]); ax6.set_ylim([-2.5, +2.5])
ax5.set_ylabel('simulated $x$'); ax6.set_ylabel('simulated $x$')
ax5.set_xlabel('time (days)'); 
ax6.set_xlabel('time (days)'); 
ax5.set_aspect(0.75/ax5.get_data_ratio(), adjustable='box')
ax6.set_aspect(0.75/ax6.get_data_ratio(), adjustable='box')


# save results
fig4C.subplots_adjust(
        top=1.0,
        bottom=0.3,
        left=0.105,
        right=0.985,
        hspace=0.2,
        wspace=0.28)

#%% Save figures
isExist = os.path.exists('./figures/')
if not isExist:  
    os.makedirs('./figures/')

fig4B.savefig('./figures/fig4B.pdf', format='pdf')
fig4C.savefig('./figures/fig4C.pdf', format='pdf')
