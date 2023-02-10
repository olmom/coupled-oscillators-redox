#%% Compute parameters of kinetic model amp. rel. rate and twist to parametrize
#stochastic amplitude-phase Poincare model: Supplementary Figure 1
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import argrelextrema
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import os
from poincare import KineticOscillator
from poincare import sdeIntegrator
from poincare import Fits 
from poincare import RhythmicParameters 
from poincare import PoincareOscillator 
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = '13'
plt.rcParams['legend.fontsize'] = '12'
plt.rcParams['xtick.labelsize'] = '12'
plt.rcParams['ytick.labelsize'] = '12'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

############
############
n_simus = 50 #number of simus for lambda (amp. rel. rate) and twist determination
d_sd = 10    #percentage of deviation from default parameter value in kinetic model
############
############

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
dt = 0.12
t = np.arange(0, 480, dt)
y0 = np.array([3.81698766e-01, 7.56473656e-01, 5.23792394e-01, 
    0.1012619494752, 0.4904989032611, 2.6796504078])

odes = kinetic_obj.Kinetic_Gonze
solution = odeint(odes, y0, t)
D2 = solution[:,1]
D2_norm = D2/D2.mean() - 1

# calculate period (needed for simulations of Poincare model)
period = RhythmicParameters().determine_period_singleosc(t, D2_norm)

suppfig1 = plt.figure(figsize=(11,4))
ax1 = suppfig1.add_subplot(122)
ax2 = suppfig1.add_subplot(121)

# calculate amplitude rel.rate by fitting exp_decay to different init.
# conditions (suppfig1A)
ax1.set_xlabel('time (h)')
ax1.set_ylabel('$D_2$ conc. (a.u.)')

import matplotlib.cm as cm
color = cm.rainbow(np.linspace(0, 1, 5))
color = 'k'

D2s, popts = [], []
for i in range(n_simus):
    # add a perturbation in D2
    ic = np.array([3.81698766e-01, np.random.uniform(0,8), 5.23792394e-01,
        0.1012619494752, 0.4904989032611, 2.6796504078])
    sol_i = odeint(odes, ic, t)
    D2_i = sol_i[:,1]
    D2s.append(D2_i)
    # calculate maxima of D2
    idx_maxs = argrelextrema(D2_i, np.greater)[0]
    D2i_maxs = D2_i[idx_maxs]
    t_D2i_maxs = t[idx_maxs]
    # fit an exponential decay curve to maxima of D2
    popt_i, pcov_i = curve_fit(Fits.exp_func, t_D2i_maxs, D2i_maxs, p0=(1.5,-0.1,0.81))
    popts.append(popt_i)

ax1.plot(t, D2_i, c=color)
ax1.plot(t_D2i_maxs, D2i_maxs, 'o', c=color)
y_pred = Fits.exp_func(t, popt_i[0], popt_i[1], popt_i[2])
ax1.plot(t, y_pred, lw=0.5, c='silver')
ax1.text(.4, .73, 
    'decay rate $D_2i={}$\,h-1'.format(format(-popt_i[1],'.3f')), 
    ha='left', va='top', transform=ax1.transAxes, color='silver')
ax1.text(.4, .58, 
    'mean decay rate of\n100 $D_2$ oscillators $=0.05$\,h-1', 
    weight='bold', ha='left', va='top', transform=ax1.transAxes)
ax1.set_xlim([0, 120])
ax1.set_xticks([0, 24,48,72, 96,120])
D2s = np.asarray(D2s)
popts = np.asarray(popts)

amp_rr    = np.abs(popts[:,1].mean()) #in h**(-1) 
amp_rr_sd = np.abs(popts[:,1].std())  #in h**(-1) 


# study parametric twist: is there an amplitude-period correlation (suppfig1B)
# when we change parameters in our model? simulate heterogeneity by changing d
d_values = 0.2 + 0.2*d_sd/100*np.random.randn(n_simus)

amplitudes, periods = [], []
for o in range(n_simus):
    kinetic_obj = KineticOscillator(a = 1000,
                                    b = 2,
                                    d = d_values[o],
                                    e = 0.1,
                                    p = 1,
                                    q = 0.1,
                                    K_coup = 0,
                                    F_zg = 0,
                                    )
    # solve ODEs from kinetic deterministic model
    odes = kinetic_obj.Kinetic_Gonze
    y0 = np.array([3.81698766e-01, 7.56473656e-01, 5.23792394e-01, 
        0.1012619494752, 0.4904989032611, 2.6796504078])
    solution = odeint(odes, y0, t)
    sol_m = solution[-int(240/dt):] #remove transients
    sol_m_norm = sol_m/sol_m.mean(axis=0) - 1 #normalize to mean
    t_m = t[-int(240/dt):] #remove transients also from time array

    # compute period & amplitude in timeseries without transients
    tau = RhythmicParameters().determine_period_singleosc(t_m, sol_m_norm[:,1]) 
    idx_D2maxs = argrelextrema(sol_m_norm[:,1], np.greater)[0][-5:]
    idx_D2mins = argrelextrema(sol_m_norm[:,1], np.less)[0][-5:]
    amp = np.mean(sol_m_norm[idx_D2maxs, 1] - sol_m_norm[idx_D2mins, 1]) #p. to t.
    periods.append(tau); amplitudes.append(amp)

periods = np.asarray(periods)
amplitudes = np.asarray(amplitudes)

ax2.set_xlabel('period (h)')
ax2.set_ylabel('peak-to-trough amplitude $D_2$')
ax2.plot(periods, amplitudes, 'o', color='black') #plot twist


#%% Simulate kinetic deterministic vs. amplitude-phase stochastic: Figure 1B
# fig1B: comparison of kinetic deterministic & amplitude-phase
# stochastic model (fig1B)
y0_P = np.array([1.0,0.0, 0.1012619494752, 0.4904989032611, 2.6796504078])
dt_noisy = 0.12
t_noisy = np.arange(0, 480, dt_noisy)

# define amplitude-phase (Poincare) stochastic object
poincare_obj = PoincareOscillator(amp = 1,
                                  lam = amp_rr,
                                  eps = 0.1,
                                  tau = period,
                                  F_zg = 0,
                                  K_coup = 0)
sdeint_obj = sdeIntegrator(sig_x = 0.05,
                           sig_y = 0.05)

m = np.shape(y0_P)[0] 
ss = np.random.SeedSequence()
generator = np.random.default_rng(ss)
eqs = poincare_obj.f_PoincareGonze
diff = sdeint_obj.G_PoincareGonze
dW = sdeint_obj.deltaW(len(t)-1, m, dt_noisy, generator)
solution_Poincare = sdeint_obj.itoEuler(eqs, diff, y0_P, t_noisy, dW=dW, m=m)

fig1B = plt.figure(figsize=(3.66,5))
axB = fig1B.add_subplot(111)

D2_norm_norm = D2_norm #normalize D2 kinetic-deterministic timeseries to compare
D2_norm_norm[D2_norm_norm > 0] = D2_norm_norm[D2_norm_norm > 0] * \
        (1/D2_norm_norm[argrelextrema(D2_norm_norm, np.greater)[0]].mean())
D2_norm_norm[D2_norm_norm < 0] = D2_norm_norm[D2_norm_norm < 0] * \
        (1/abs(D2_norm_norm[argrelextrema(D2_norm_norm, np.less)[0]].mean()))

axB.plot(t, D2_norm_norm, label='deterministic, kinetic $D_2$', c='k')
axB.plot(t_noisy, solution_Poincare[:,0], label='stochastic, amplitude-phase $D_2$', c='red')
axB.set_xlabel('time (h)')
axB.set_ylabel('$D_2$ concentration')
axB.set_xlim([0, 120])
axB.set_xticks([0, 24, 48, 72, 96, 120])
axB.legend(framealpha=0, loc="upper right")

fig1B.subplots_adjust(top=0.88,
        bottom=0.205,
        left=0.21,
        right=0.9,
        hspace=0.2,
        wspace=0.2)

# square plots
ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
axB.set_aspect(1.0/axB.get_data_ratio(), adjustable='box')

suppfig1.subplots_adjust(
        top=0.945,
        bottom=0.17,
        left=0.08,
        right=0.945,
        hspace=0.405,
        wspace=0.36)


#%% Save figures
# make figures/ directory and save figures
isExist = os.path.exists('./figures/')
if not isExist:  
    os.makedirs('./figures/')

suppfig1.savefig('./figures/suppfig1.pdf', format='pdf')
fig1B.savefig('./figures/fig1B.pdf', format='pdf')



