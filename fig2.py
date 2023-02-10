#%% Load results and plot Figure 2A-C, Supplementary Figure 3
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import os
import seaborn as sns
from scipy.stats import circstd
from poincare import Fits 
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = '13'
plt.rcParams['legend.fontsize'] = '12'
plt.rcParams['xtick.labelsize'] = '12'
plt.rcParams['ytick.labelsize'] = '12'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

########## PARAMETER CHOICE FOR FIG2 ###############
####################################################
Fzg = 0.00
amp_rr = np.array([0.05, 1.0, 0.03])
Kcoup = 0.00
sigma = 0.05
tausd = np.array([0.00,1.50])

n_oscs = 100
total_simus = 50 
initial_conditions = 'LC' #limit cycle initial conditions for
                          #fig2 (analysis of desync dynamics)
####################################################
####################################################

dt = 0.12 
t = np.arange(0, 2400, dt)

# Load results
list_phases, list_phases_norm = [], []
list_periods, list_amplitudes = [], []
list_circstds, list_zs, list_sync_idxs = [], [], []

combinations = np.array([(s,l,k) for s in np.arange(total_simus)+1 \
        for l in amp_rr for k in tausd])

for c in combinations:
    simu, l, tsd = c
    # load solution
    toload = './results/' +\
        'no TTFL, ' +\
        'lambda={}, sigma={}, tausd={}, Kcoup={}, {}_ic/simu{}/'.format(
            format(l, '.2f'),
            format(sigma, '.2f'),
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

    # determine synchronization index of the 100 oscillators 
    z = amp * np.exp( 1j * phi ) 
    si = abs(pd.DataFrame(np.exp(1j * np.angle(z))).sum(axis='columns')/n_oscs)

    # compute standard deviation of phase distribution across oscillators 
    cstds = circstd(phi.values, axis=1)

    print(
        'simulation {}/{} loaded (amplitudes, phases and circstds'.format(
            int(simu), total_simus) +
        ' computed) --- lam={}, sig={}, tausd={}, Kcoup={}, '.format( 
            format(l, '.2f'),
            format(sigma, '.2f'),
            format(tsd, '.2f'),
            format(Kcoup, '.2f')
            )) 

    list_phases.append(phi)
    list_amplitudes.append(amp)
    list_zs.append(z)
    list_sync_idxs.append(si)
    list_circstds.append(cstds)


# PLOT
# fig2A: heatmap of desynchronization of noisy oscillators
fig2A = plt.figure(figsize=(11,4))
ax1 = fig2A.add_subplot(111)

xticks = np.arange(0, 3001, 200, 
        dtype=np.int) #idx position of yticks
xticklabels = np.array([t[idx] \
        for idx in xticks])/24 #content of labels
xticklabels = ['$' + \
        str(np.int(np.round(xticklabels[i],1))) + '$' \
        for i in range(len(xticklabels))]
ax1 = sns.heatmap(
        np.transpose(list_phases[0].iloc[0:3000,:]*24/(2*np.pi)),
        xticklabels=xticklabels, yticklabels=False, 
        vmin=-12, vmax=12, ax=ax1 )
ax1.set_xticks(xticks)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 0)
ax1.set_xlabel('time (days)')
ax1.set_ylabel('redox oscillator number')
ax1.collections[0].colorbar.set_label('phase (h)')

# figs 2B, C
fig2BC = plt.figure(figsize=(11,4))
ax2 = fig2BC.add_subplot(121, projection='polar')
ax3 = fig2BC.add_subplot(122, aspect='equal')

# fig2B: polar plots of phases at t=1, 5 and 50 days 
degrees = np.random.randint(-180, 180, size=200)
radians = np.deg2rad(degrees)

bin_size = 2 #in hours 
a1, b1 =np.histogram(list_phases[0].iloc[200,:]*24/(2*np.pi), 
        bins=np.arange(-12, 12+bin_size, bin_size))
a2, b2 =np.histogram(list_phases[0].iloc[1000,:]*24/(2*np.pi), 
        bins=np.arange(-12, 12+bin_size, bin_size))
a3, b3 =np.histogram(list_phases[0].iloc[-10000,:]*24/(2*np.pi), 
        bins=np.arange(-12, 12+bin_size, bin_size))
centers1 = np.deg2rad(np.ediff1d(b1)//2 + b1[:-1])
centers2 = np.deg2rad(np.ediff1d(b2)//2 + b2[:-1])
centers3 = np.deg2rad(np.ediff1d(b3)//2 + b3[:-1])
centers1 = 2*np.pi/24 * (np.ediff1d(b1)//2 + b1[:-1])
centers2 = 2*np.pi/24 * (np.ediff1d(b2)//2 + b2[:-1])
centers3 = 2*np.pi/24 * (np.ediff1d(b3)//2 + b3[:-1])

ax2.bar(centers1, a1, width=2*np.pi/24*(bin_size), bottom=0.0, 
        edgecolor='k', alpha=0.4, 
        label='$t={}\,$day'.format(int(round(t[200]/24,1))))
ax2.bar(centers2, a2, width=2*np.pi/24*(bin_size), bottom=0.0, 
        edgecolor='k', alpha=0.4,
        label='$t={}\,$days'.format(int(round(t[1000]/24,1))))
ax2.bar(centers3, a3, width=2*np.pi/24*(bin_size), bottom=0.0, 
        edgecolor='k', alpha=0.4,
        label='$t={}\,$days'.format(int(round(t[-10000]/24,1))))
ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)
ax2.legend(loc='center right', bbox_to_anchor=(1.65, 0.50),
        framealpha=0)#,ncol=3)
ax2.set_thetagrids(angles=np.arange(0,360,45), 
        labels=['$0$h','$3$h','$6$h','$9$h','$12/-12$h',
        '$-9$h','$-6$h','$-3$h'])
y = np.asarray(list_circstds[0::6]).mean(axis=0)*24/(2*np.pi)

# fig2C: desynchronization dynamics. How does the phase spreading
# of the oscillators grow over time?
ax3.set_xlim([-1,60])
for i in range(total_simus):
    ax3.plot(t/24, np.asarray(list_circstds[6*i])*24/(2*np.pi), 
            color='lightsalmon', alpha=0.08)
    ax3.plot(t/24, np.asarray(list_circstds[6*i+1])*24/(2*np.pi), 
            color='silver', alpha=0.1)
ax3.plot(t/24, np.asarray(list_circstds[0::6]).mean(axis=0)*24/(2*np.pi), 
        color='crimson', label=r'average SD, periods $=24.23\pm {}$ h'.format(
            format(tausd[0], '.1f')))
ax3.plot(t/24, np.asarray(list_circstds[1::6]).mean(axis=0)*24/(2*np.pi), 
        color='k', label=r'average SD, periods $=24.23\pm {}$ h'.format(
            format(tausd[1], '.1f')))
ax3.set_xlabel('time (days)') 
ax3.set_ylabel('circular standard deviation of phases (h)')

# fit mean sd curve to a sqrt function (Stratonovich)
y1 = np.asarray(list_circstds[0::6]).mean(axis=0)*24/(2*np.pi)#lam=0.05, tsd=0
y2 = np.asarray(list_circstds[1::6]).mean(axis=0)*24/(2*np.pi)#lam=0.05, tsd=1.5
popt1, pcov1 = curve_fit(Fits.sigmoidal_func, t[1:]/24, y1[1:], p0=(8,10*24,1))
popt2, pcov2 = curve_fit(Fits.sigmoidal_func, t[1:]/24, y2[1:], p0=(8,10*24,1))
popt3, pcov3 = curve_fit(Fits.sqrt_func, t[1:]/24, y1[1:], p0=(2,1))
popt4, pcov4 = curve_fit(Fits.sqrt_func, t[1:]/24, y2[1:], p0=(2,1))

y1_pred = Fits.sigmoidal_func(t/24, popt1[0], popt1[1], popt1[2])
y2_pred = Fits.sigmoidal_func(t/24, popt2[0], popt2[1], popt2[2])
y3_pred = Fits.sqrt_func(t/24, popt3[0], popt3[1])
y4_pred = Fits.sqrt_func(t/24, popt4[0], popt4[1])

# compute goodness of fit
ss_res1 = np.sum((y1[1:] - y1_pred[1:]) ** 2) # residual sum of squares
ss_res2 = np.sum((y2[1:] - y2_pred[1:]) ** 2) 
ss_res3 = np.sum((y1[1:] - y3_pred[1:]) ** 2)
ss_res4 = np.sum((y2[1:] - y4_pred[1:]) ** 2)
ss_tot1 = np.sum((y1[1:] - np.mean(y1[1:])) ** 2) # total sum of squares
ss_tot2 = np.sum((y2[1:] - np.mean(y2[1:])) ** 2) 
ss_tot3 = np.sum((y1[1:] - np.mean(y1[1:])) ** 2) 
ss_tot4 = np.sum((y2[1:] - np.mean(y2[1:])) ** 2) 
r2_1 = 1 - (ss_res1 / ss_tot1) # r-squared
r2_2 = 1 - (ss_res2 / ss_tot2) 
r2_3 = 1 - (ss_res3 / ss_tot3)
r2_4 = 1 - (ss_res4 / ss_tot4)

ax3.set_ylim(ymax=15.2)
ax3.set_aspect(1.0/ax3.get_data_ratio(), adjustable='box')
ax3.legend(loc='upper left', framealpha=0)

fig2A.subplots_adjust(
        top=0.88,
        bottom=0.175,
        left=0.11,
        right=0.9,
        hspace=0.51,
        wspace=0.33)
fig2BC.subplots_adjust(
        top=0.90,
        bottom=0.16,
        left=0.05,
        right=0.95,
        hspace=0.62,
        wspace=0.44)

#%% Save figure
# save fig2
# make figures/ directory and save figures
isExist = os.path.exists('./figures/')
if not isExist:  
    os.makedirs('./figures/')
    
fig2A.savefig('./figures/fig2A.png', format='png', dpi=1200)
fig2BC.savefig('./figures/fig2BC.pdf', format='pdf')



#########################
#########################
#%% Plot Supplementary Figure 3
# speed of desynchronization also depends on relaxation rate (suppfig3)
suppfig3 = plt.figure(figsize=(11,4))
sax1 = suppfig3.add_subplot(121)
sax2 = suppfig3.add_subplot(122)

# desync dynamics for different lambda values, all oscs. identical
sax1.plot(t/24, np.asarray(list_circstds[4::6]).mean(axis=0)*24/(2*np.pi), 
         label='$\lambda={}$ h-1'.format(amp_rr[2]), c='dodgerblue')
sax1.plot(t/24, np.asarray(list_circstds[0::6]).mean(axis=0)*24/(2*np.pi), 
         label='$\lambda={}$ h-1'.format(amp_rr[0]), c='k')
sax1.plot(t/24, np.asarray(list_circstds[2::6]).mean(axis=0)*24/(2*np.pi), 
         label='$\lambda={}$ h-1'.format(amp_rr[1]), c='crimson')
y = np.asarray(list_circstds[2::6]).mean(axis=0)*24/(2*np.pi)#l=1, tsd=1.5
popt, pcov = curve_fit(Fits.sqrt_func, t[1:]/24, y[1:], p0=(1,1))
y_pred = Fits.sqrt_func(t/24, popt[0], popt[1])
ss_res = np.sum((y[1:] - y_pred[1:]) ** 2) # residual sum of squares
ss_tot = np.sum((y[1:] - np.mean(y[1:])) ** 2) # total sum of squares
r2 = 1 - (ss_res / ss_tot) # r-squared
sax1.plot(t/24, y_pred, color='crimson',lw=.75, linestyle='--')
sax1.text(40, 4, '$\sigma=k\sqrt{Dt}, D=\,' + 
         str(round(popt[1],2)) + '$' + 
         '\n$R^{2}=\,' + str(round(r2,2)) + '$', color="crimson")

# desync dynamics for different lambda values, heterogeneous oscillators
sax2.plot(t/24, np.asarray(list_circstds[5::6]).mean(axis=0)*24/(2*np.pi), 
         label='$\lambda={}$ h-1'.format(amp_rr[2]), c='dodgerblue')
sax2.plot(t/24, np.asarray(list_circstds[1::6]).mean(axis=0)*24/(2*np.pi), 
         label='$\lambda={}$ h-1'.format(amp_rr[0]), c='k')
sax2.plot(t/24, np.asarray(list_circstds[3::6]).mean(axis=0)*24/(2*np.pi), 
         label='$\lambda={}$ h-1'.format(amp_rr[1]), c='crimson')
sax1.set_xlabel('time (days)'); 
sax1.set_ylabel('mean circular std. dev. of phases (h)')
sax1.set_ylim(ymax=11.2); 
sax1.set_aspect(1.0/sax1.get_data_ratio(), adjustable='box')
sax1.legend(loc='lower right', framealpha=0)
sax2.set_xlabel('time (days)'); 
sax2.set_ylabel('mean circular std. dev. of phases (h)')
sax2.set_ylim(ymax=11.2); 
sax2.set_aspect(1.0/sax1.get_data_ratio(), adjustable='box')
sax2.legend(loc='lower right', framealpha=0)

sax2.text(.05, 0.95, 'periods $=24.23\pm {}$\,h'.format(tausd[1]), 
        ha='left', va='top', transform=sax2.transAxes)
sax1.text(.05, 0.95, 'periods $=24.23\pm {}$\,h'.format(tausd[0]), 
        ha='left', va='top', transform=sax1.transAxes)

suppfig3.subplots_adjust(
    top=0.89,
    bottom=0.16,
    left=0.05,
    right=0.97,
    hspace=0.2,
    wspace=0.36)

#%% Save figure
suppfig3.savefig('./figures/suppfig3.pdf', format='pdf')
