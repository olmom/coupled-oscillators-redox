#%% Load results and plot Supplementary Figure 2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.signal import argrelextrema
from scipy.integrate import odeint
from scipy import signal
import pandas as pd
import os
from scipy.stats import circstd
from poincare import KineticOscillator
from poincare import RhythmicParameters
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = '13'
plt.rcParams['legend.fontsize'] = '12'
plt.rcParams['xtick.labelsize'] = '12'
plt.rcParams['ytick.labelsize'] = '12'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

############## PARAMETER CHOICE FOR FIG2 ###########
####################################################
Fzg = 0.00
amp_rr = np.array([0.05])
Kcoup = 0.00
sigma = np.arange(0.0, 0.11, 0.01)
sigma_default = 0.05 #for later simulations
tausd = np.array([0.00])

n_oscs =100
total_simus = 10 
initial_conditions = 'LC' #limit cycle initial conditions for
                          #suppfig2 (analysis of desync dynamics)
####################################################
####################################################

dt = 0.12 
t = np.arange(0, 2400, dt)

# Load results
list_phases, list_phases_norm = [], []
list_periods, list_amplitudes = [], []
list_circstds = []
list_xs = []

combinations = np.array([(s,l,k,n) for s in np.arange(total_simus)+1 \
        for l in amp_rr for k in tausd for n in sigma])

for c in combinations:
    simu, l, tsd, n = c
    # load solution
    toload = './results/' +\
        'no TTFL, ' +\
        'lambda={}, sigma={}, tausd={}, Kcoup={}, {}_ic/simu{}/'.format(
            format(l, '.2f'),
            format(n, '.2f'),
            format(tsd, '.2f'),
            format(Kcoup, '.3f'),
            initial_conditions,
            int(simu))
    solu = np.load(toload + '/solution.npy')
    sol_xdf = pd.DataFrame(solu[:,0::5])
    sol_ydf = pd.DataFrame(solu[:,1::5])
    
    # load amplitudes, phases, periods
    amp = pd.DataFrame(np.load(toload + '/amplitudes.npy'))
    phi = pd.DataFrame(np.load(toload + '/phases.npy'))
    tau = pd.DataFrame(np.load(toload + 'periods.npy', allow_pickle=True)[0])

    amp.index = t
    phi.index = t
    sol_xdf.index = t
    sol_ydf.index = t

    # compute standard deviation of phase distribution across oscillators 
    cstds = circstd(phi.values, axis=1)

    print(
        'simulation {}/{} loaded (amplitudes, phases and circstds'.format(
            int(simu), total_simus) +
        ' computed) --- lam={}, sig={}, tausd={}, Kcoup={}, '.format( 
            format(l, '.2f'),
            format(n, '.2f'),
            format(tsd, '.2f'),
            format(Kcoup, '.2f')
            )) 

    list_phases.append(phi)
    list_amplitudes.append(amp)
    list_circstds.append(cstds)
    list_xs.append(sol_xdf)


# PLOT
# suppfig2: mean-field timeseries (A) and growth of std of phase dispersion (B)
# for different values of variances of noise
suppfig2 = plt.figure(figsize=(11,11))
ax1 = suppfig2.add_subplot(221)
ax2 = suppfig2.add_subplot(222)
ax3 = suppfig2.add_subplot(223)
ax4 = suppfig2.add_subplot(224)

color = cm.brg_r(np.linspace(0, 1, len(sigma)))
ax1.set_ylim([-1.45,1.45]); ax2.set_xlim([-.25,20])
for i in range(len(sigma)):
    if (i == 0) or (i == 5) or (i == 10):
        ax1.plot(t[0:int(20*24/dt)]/24, list_xs[i].values[0:int(20*24/dt),:].mean(axis=1), 
            color=color[i])   
    ax2.plot(t/24, np.asarray(list_circstds[i::11]).mean(axis=0)*24/(2*np.pi), 
        color=color[i], 
        label='$\sigma_x={}$'.format(format(combinations[i,3], '.2f')))
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.0)
ax2.set_xlabel('time (days)'); ax1.set_xlabel('time (days)')
ax1.set_ylabel('average signal of 100 oscillators')
ax2.set_ylabel('circular standard deviation of phases (h)')

# check that nothing strange is going on in timeseries for high sigma values
sigma_check, idx = 0.10, 1
sigma_idx = np.where(sigma == round(sigma_check,2))[0][0]
fig_aux = plt.figure(figsize=(11,4))
axA = fig_aux.add_subplot(121)
axB = fig_aux.add_subplot(122)
for i in range(n_oscs):
    axA.plot(t[0:int(30*24/dt)]/24, list_xs[idx*sigma_idx].values[0:int(30*24/dt),i], 
            color='silver', alpha=0.25)
axA.plot(t[0:int(30*24/dt)]/24, list_xs[sigma_idx].values[0:int(30*24/dt),:].mean(axis=1), 
    color='k')
axA.set_title('100 oscs from 1 realization + ' +\
        '\naverage signal ($\sigma_x=${})'.format(sigma[sigma_idx]))
for i in range(total_simus):
    axB.plot(t[0:int(30*24/dt)]/24, 
            list_xs[sigma_idx::11][i].values[0:int(30*24/dt),:].mean(axis=1), 
            color='blue', alpha=0.25)
axB.set_title('mean of 100 oscs from 10 realizations'+\
        '\n($\sigma_x=${})'.format(sigma[sigma_idx]))
axA.set_ylabel('simulated $x$'); axB.set_ylabel('average of simulated $x$')
axA.set_xlabel('time (days)'); axB.set_xlabel('time (days)')


# suppfig2C: power spectral density (PSD) of kinetic and poincare models
# Simulate the redox-Kinetic model and focus on D2 oscillations
kinetic_obj = KineticOscillator(a = 1000,
                                b = 2,
                                d = 0.2,
                                e = 0.1,
                                p = 1,
                                q = 0.1,
                                K_coup = 0,
                                F_zg = 0,
                                )
t = np.arange(0, 2400, dt)
y0 = np.array([3.81698766e-01, 7.56473656e-01, 5.23792394e-01, 
    0.1012619494752, 0.4904989032611, 2.6796504078])

odes = kinetic_obj.Kinetic_Gonze
solution = odeint(odes, y0, t)
D2 = solution[:,1]
D2_norm = D2/D2.mean() - 1

period = RhythmicParameters().determine_period_singleosc(t, D2_norm)
freq, peaks_val = signal.periodogram(D2_norm, 1/dt)
peaks_val = peaks_val/(np.max(peaks_val))
time = 1/freq 
time = time 
freq = 1/time

# show PSD for sigma = 0.05 (default value)
sigma_idx = np.where(sigma == round(sigma_default,2))[0][0] #index for sigma=0.05

freq_noisy, peaks_val_noisy = signal.periodogram(list_xs[sigma_idx].iloc[:,0], 1/dt)
peaks_val_noisy = peaks_val_noisy/(np.max(peaks_val_noisy))
time_noisy = 1/freq_noisy 
time_noisy = time_noisy 
freq_noisy = 1/time_noisy

ax3.set_xlabel('frequency (days-1)')
ax3.set_ylabel('log10(norm. power)')
ax3.set_xlim(-0.25,7.5)
ax3.set_ylim(-9,1)
ax3.plot(freq*24,np.log10(peaks_val), lw=1, c='k', 
        label='deterministic, kinetic $D_2$')#[$V^2 h$]
ax3.plot(freq_noisy*24,np.log10(peaks_val_noisy), lw=1, 
        c=color[sigma_idx], 
        label='stochastic, amplitude-phase $D_2$\nat $\sigma_x=0.05$')
ax3.tick_params(axis='both', which='major')
ax3.set_xticks([0,1,2,3,4,5,6,7])
ax3.legend(framealpha=0)


#suppfig2D: autocovariance function of one noisy Poincare model
#with the parameters estimated from the kinetic model
k = int(144/dt)
D2_norm_norm = D2_norm #normalize
D2_norm_norm[D2_norm_norm > 0] = D2_norm_norm[D2_norm_norm > 0] * \
        (1/D2_norm_norm[argrelextrema(D2_norm_norm, np.greater)[0]].mean())
D2_norm_norm[D2_norm_norm < 0] = D2_norm_norm[D2_norm_norm < 0] * \
        (1/abs(D2_norm_norm[argrelextrema(D2_norm_norm, np.less)[0]].mean()))

x0_noisy  = list_xs[sigma_idx].iloc[:,0].values #single oscillator, stochastic
x0_det = D2_norm_norm #single oscillator, deterministic
autocovs_noisy, autocovs_det = [], []
for i in range(k):
    k_i = int(i)
    C_noisy = RhythmicParameters().autocovariance_signal(x0_noisy, len(x0_noisy), k_i)
    C_det = RhythmicParameters().autocovariance_signal(x0_det, len(x0_det), k_i)
    autocovs_noisy.append(C_noisy)
    autocovs_det.append(C_det)

autocovs_det = np.array(autocovs_det)
autocovs_noisy= np.array(autocovs_noisy)
acf_det = autocovs_det[: k + 1] / autocovs_det[0] 
acf_noisy = autocovs_noisy[: k + 1] / autocovs_noisy[0] 

ax4.axvline(x=period, linestyle='--', c='silver', alpha=0.5)
ax4.plot(t[0:k], acf_det, c='k', label='deterministic, kinetic $D_2$')
ax4.plot(t[0:k], acf_noisy, c=color[sigma_idx], 
        label='stochastic, amplitude-phase $D_2$\nat $\sigma_x=0.05$')
ax4.set_xlabel(r'delay $\tau$ (h)')
ax4.set_ylabel('Autocovariance') 
ax4.legend(framealpha=0, loc="upper right")
ax4.set_xticks([0,24,48,72, 96, 120, 144])
ax4.set_ylim([-1.5, 1.5])
ax4.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])

ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
ax3.set_aspect(1.0/ax3.get_data_ratio(), adjustable='box')
ax4.set_aspect(1.0/ax4.get_data_ratio(), adjustable='box')

suppfig2.subplots_adjust(
    top=0.945,
    bottom=0.09,
    left=0.0,
    right=0.945,
    hspace=0.405,
    wspace=0.03)

#%% Save figure
isExist = os.path.exists('./figures/')
if not isExist:  
    os.makedirs('./figures/')

suppfig2.savefig('./figures/suppfig2.pdf', format='pdf')
