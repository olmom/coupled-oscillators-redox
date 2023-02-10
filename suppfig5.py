#%% Load results and plot Supplementary Figure 5
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
import scipy.stats
import pandas as pd
import os
from poincare import Fits 
from poincare import RhythmicParameters 
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = '13'
plt.rcParams['legend.fontsize'] = '12'
plt.rcParams['xtick.labelsize'] = '12'
plt.rcParams['ytick.labelsize'] = '12'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

########### PARAMETER CHOICE FOR SUPPFIG2 ##############
########################################################

Fzg = 0.00
sigma = 0.05
amp_rr = np.array([0.03, 0.05, 1.00]) 
Kcoup = np.arange(0.000, 0.160, 0.005)
tausd = np.array([1.50]) 
total_simus = 5

initial_conditions = 'random' 
n_oscs = 100

####################################################
####################################################

dt = 0.12 
t = np.arange(0, 2400, dt)

# load results for suppfig5
list_xs = []

combinations = np.array([(s,l,ts,k) for s in np.arange(total_simus)+1 \
        for l in amp_rr for ts in tausd for k in Kcoup])

for c in combinations:
    simu, l, tsd, Kc = c
    # load solution
    toload = './results/' +\
        'no TTFL, ' +\
        'lambda={}, sigma={}, tausd={}, Kcoup={}, {}_ic/simu{}/'.format(
        format(l, '.2f'),
        format(sigma, '.2f'),
        format(tsd, '.2f'),
        format(Kc, '.3f'),
        initial_conditions,
        int(simu))

    solu = np.load(toload + '/solution.npy')
    sol_xdf = pd.DataFrame(solu[:,0::5])
    sol_ydf = pd.DataFrame(solu[:,1::5])
    sol_xdf.index = t
    sol_ydf.index = t

    print(
        'simulation {}/{} loaded (amplitudes, phases and circstds'.format(
            int(simu), total_simus) +
        ' computed) --- lambda={}, sig={}, tausd={}, Kcoup={}, '.format( 
            format(l, '.2f'),
            format(sigma, '.2f'),
            format(tsd, '.2f'),
            format(Kc, '.3f'),
            )) 

    list_xs.append(sol_xdf)



# PLOT SUPPFIG5: effect of different lambdas on network dynamics
# Calculate amplitude of ensemble (mean of all oscillators)
list_xs_ens = [list_xs[i].values.mean(axis=1) for 
        i in range(len(list_xs))] #ensemble timeseries
y = list_xs_ens
ymaxs = [y[i][argrelextrema(y[i], np.greater, order=20)[0]] \
        for i in range(len(list_xs_ens))] #maxima 
ymins = [y[i][argrelextrema(y[i], np.less, order=20)[0]] \
        for i in range(len(list_xs_ens))] #minima
ms = [np.array([len(ymaxs[i]), len(ymins[i])]).min() for 
        i in range(len(list_xs_ens))]
amps = [ymaxs[i][-int(ms[i]/2):] - ymins[i][-int(ms[i]/2):] for i in
        range(len(list_xs_ens))]
amp2 = [abs(amps[i].mean()) for i in range(len(list_xs_ens))]
amp2 = np.asarray(amp2)
amp2 = amp2.reshape(total_simus, len(amp_rr), len(Kcoup))
amp2_meansimu = np.mean(amp2, axis=0)/2

# plot how resonance amplitude changes with Kc for diff. lambda (suppfig5B)
# plot higher lambda values associated with no period lengthening 
# (for K=0.1) (suppfig5C)
suppfig5BC = plt.figure(figsize=(6.5,6))
ax_supp4_5 = suppfig5BC.add_subplot(221)
ax_supp4_6 = suppfig5BC.add_subplot(222)

colors = ['dodgerblue', 'black', 'crimson']
handles, labels = ax_supp4_5.get_legend_handles_labels()

# smooth the mean ensemble amplitude as a function of Kc by
# computing the average of K Lowess models that are fitted to
# 50% of the points (chosen randomly in each iteration of k)
# iterate this for the different lambda values
for l in range(len(amp_rr)):
    xgrid = np.linspace(Kcoup.min(),Kcoup[-2])
    K = 100 # fit 100 Lowes modells
    smooths1 = np.stack([Fits.smooth_curve(Kcoup, amp2_meansimu[l,:], 
        xgrid, frac=0.5, it=2, random_percentage=50) for k in range(K)]).T
    mean1 = np.nanmean(smooths1, axis=1) #mean of the ensemble 
                            #amplitude (for each  lambda)
    stderr1 = scipy.stats.sem(smooths1, axis=1) #stand. error of mean
    stderr1 = np.nanstd(smooths1, axis=1, ddof=0)
    stderr1[stderr1 < 0.0] = 0
    # plot average ensemble amplitude (smoothened by Lowess) as well as the
    # condifence interval for the Lowess models
    ax_supp4_5.fill_between(xgrid, mean1-1.96*stderr1, mean1+1.96*stderr1, 
            color=colors[l], alpha=0.15)    
    ax_supp4_5.plot(xgrid, mean1, color=colors[l], alpha=0.4,
            label='$\lambda={}$ h-1'.format(amp_rr[l]))
    ax_supp4_5.plot(Kcoup, amp2_meansimu[l,:], 'o', markersize=2,
            color=colors[l])
    
ax_supp4_5.set_xlabel('$K_c$')
ax_supp4_5.set_ylabel('ensemble amplitude (a.u.)')
ax_supp4_5.set_xlim([-0.0051, 0.122])
ax_supp4_5.set_ylim([-0.2, 3.2])
ax_supp4_5.set_aspect(1.0/ax_supp4_5.get_data_ratio(), adjustable='box')
ax_supp4_5.legend(loc='upper left', framealpha=0)

period_lam1 = RhythmicParameters().periods(t, list_xs[20+len(Kcoup)*6])[0]
period_lam2 = RhythmicParameters().periods(t, list_xs[20+len(Kcoup)*7])[0]
period_lam3 = RhythmicParameters().periods(t, list_xs[20+len(Kcoup)*8])[0]
periods_lam = np.array([period_lam1, period_lam2, period_lam3])

ax_supp4_6.hist(np.transpose(periods_lam),bins=np.arange(22,35,.75),
        stacked=True, color=colors, alpha=0.8, edgecolor='white')   
ax_supp4_6.legend(loc=2, framealpha=0.0); 
ax_supp4_6.set_xlabel('period (h)'); 
ax_supp4_6.set_ylabel('number of oscillators'); 
ax_supp4_6.set_xticks([24, 28, 32])
ax_supp4_6.set_ylim([0, 102])
ax_supp4_6.set_aspect(1.0/ax_supp4_6.get_data_ratio(), adjustable='box')
ax_supp4_6.text(.05, 0.95, 'for $K_c=0.10$',
        ha='left', va='top', transform=ax_supp4_6.transAxes)

suppfig5BC.subplots_adjust(
        top=0.960,
        bottom=0.08,
        left=0.110,
        right=0.955,
        hspace=0.360,
        wspace=0.425
        )

# suppfig5A: better synchrony for higher lambda values, but less resonance
suppfig5A = plt.figure(figsize=(4.5,6)) 
ax_supp4_1 = suppfig5A.add_subplot(311)
ax_supp4_2 = suppfig5A.add_subplot(312)
ax_supp4_3 = suppfig5A.add_subplot(313)
idx=64

t_shift = np.arange(0, t[-1]-t[-1000], dt)/24 #take only 5 days
for i in range(n_oscs):
    ax_supp4_1.plot(t_shift, list_xs[idx].iloc[(-5800):(-4800),i], 
            c="silver", lw=0.5, alpha=0.3)
    ax_supp4_2.plot(t_shift, list_xs[idx+10].iloc[(-2500):(-1500),i], 
            c="silver", lw=0.5, alpha=0.3)
    ax_supp4_3.plot(t_shift, list_xs[idx+20].iloc[(-3000):(-2000),i], 
            c="silver", lw=0.5, alpha=0.3)    
ax_supp4_1.plot(t_shift, list_xs[idx].iloc[(-5800):(-4800),:].mean(axis=1), 
        c='crimson')
ax_supp4_2.plot(t_shift, list_xs[idx+10].iloc[(-2000):(-1000),:].mean(axis=1), 
        c='crimson')
ax_supp4_3.plot(t_shift, list_xs[idx+20].iloc[(-3000):(-2000),:].mean(axis=1),
        c='crimson')

ax_supp4_1.text(.05, 1.03, 
        '$K_c={}$'.format(format(combinations[idx][-1], '.2f')), 
        ha='left', va='top', transform=ax_supp4_1.transAxes)
ax_supp4_2.text(.05, 1.03, 
        '$K_c={}$'.format(format(combinations[idx+10][-1], '.2f')), 
        ha='left', va='top', transform=ax_supp4_2.transAxes)
ax_supp4_3.text(.05, 1.03, 
        '$K_c={}$'.format(format(combinations[idx+20][-1], '.2f')), 
        ha='left', va='top', transform=ax_supp4_3.transAxes)

ax_supp4_1.set_xticklabels([]); ax_supp4_2.set_xticklabels([])
ax_supp4_2.set_ylabel('simulated $x$'); 
ax_supp4_3.set_xlabel('time (days)')
ax_supp4_1.set_ylim([-2.3, +2.3])
ax_supp4_2.set_ylim([-2.3, +2.3])
ax_supp4_3.set_ylim([-2.3, +2.3])

suppfig5A.subplots_adjust(
        top=0.915,
        bottom=0.08,
        left=0.215,
        right=0.96,
        hspace=0.2,
        wspace=0.2)

#%% Save figure
isExist = os.path.exists('./figures/')
if not isExist:  
    os.makedirs('./figures/')

suppfig5A.savefig('./figures/suppfig5A.pdf', format='pdf')
suppfig5BC.savefig('./figures/suppfig5BC.pdf', format='pdf')

