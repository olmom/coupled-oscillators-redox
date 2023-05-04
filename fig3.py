#%% Load results and plot Figure 3A, B
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
import scipy.stats
import pandas as pd
import os
from scipy.stats import circstd
from poincare import Fits 
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = '13'
plt.rcParams['legend.fontsize'] = '12'
plt.rcParams['xtick.labelsize'] = '12'
plt.rcParams['ytick.labelsize'] = '12'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

############## PARAMETER CHOICE FOR FIG3 ###########
####################################################

# FOR FIGURES 3A-3B
# -----------------
Fzg = 0.00 #strength of the Goodwlin-like input (Gonze) to Poincare
amp_rr = np.array([0.05]) 
Kcoup = np.array([0.00, 0.05, 0.10])
sigma = 0.05
tausd = np.array([1.50])

# FOR FIG 3C, D and SUPPFIG4 
# --------------------------
Kcoup_S3 = np.arange(0.000, 0.160, 0.005)
tausd_S3 = np.array([1.50, 0.00]) 

total_simus = 5
initial_conditions = 'random' 
n_oscs = 100

####################################################
####################################################

dt = 0.12  
t = np.arange(0, 2400, dt) 

# FOR FIGURES 3A-C
# ----------------
# Load results
list_phases, list_phases_norm = [], []
list_periods, list_amplitudes = [], []
list_circstds, list_zs, list_sync_idxs = [], [], []
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

    # load amplitudes, phases, periods
    amp = pd.DataFrame(np.load(toload + '/amplitudes.npy'))
    phi = pd.DataFrame(np.load(toload + '/phases.npy'))
    tau = pd.DataFrame(np.load(toload + 'periods.npy', allow_pickle=True)[0])

    amp.index = t; phi.index = t
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
        ' computed) --- lambda={}, sig={}, tausd={}, Kcoup={}, '.format( 
            format(l, '.2f'),
            format(sigma, '.2f'),
            format(tsd, '.2f'),
            format(Kc, '.2f')
            )) 

    list_phases.append(phi)
    list_periods.append(tau)
    list_amplitudes.append(amp)
    list_zs.append(z)
    list_sync_idxs.append(si)
    list_circstds.append(cstds)
    list_xs.append(sol_xdf)



# PLOT
# fig3A: timeseries for three values of coupling Kc
fig3A = plt.figure(figsize=(4.5,6)) 

ax1 = fig3A.add_subplot(311)
ax2 = fig3A.add_subplot(312)
ax3 = fig3A.add_subplot(313)

norm = [plt.Normalize(list_periods[k][0].min(), 
    list_periods[k][0].max()) \
                for k in range(len(Kcoup))]
cm = plt.cm.rainbow
z = [list_periods[k][0] for k in range(len(Kcoup))]

t_shift = np.arange(0, t[-1]-t[-1000], dt)/24
for i in range(n_oscs): #idxs 0,1,2: noise only; 3,4,5: noise+heterogeneity
    ax1.plot(t_shift, list_xs[0].iloc[(-5800):(-4800),i], 
            c=cm(norm[0](z[0][i])), lw=0.5, alpha=0.5)
    ax2.plot(t_shift, list_xs[1].iloc[(-2000):(-1000),i], 
            c=cm(norm[1](z[1][i])), lw=0.5, alpha=0.5)
    ax3.plot(t_shift, list_xs[2].iloc[(-3000):(-2000),i], 
            c=cm(norm[2](z[2][i])), lw=0.5, alpha=0.5)
   
ax1.plot(t_shift, list_xs[0].iloc[(-5800):(-4800),:].mean(axis=1), c='k')
ax2.plot(t_shift, list_xs[1].iloc[(-2000):(-1000),:].mean(axis=1), c='k')
ax3.plot(t_shift, list_xs[2].iloc[(-3000):(-2000),:].mean(axis=1), c='k')

ax1.text(.05, 1.03, '$K_c={}$'.format(format(Kcoup[0], '.2f')), 
        ha='left', va='top', transform=ax1.transAxes)
ax2.text(.05, 1.03, '$K_c={}$'.format(format(Kcoup[1], '.2f')), 
        ha='left', va='top', transform=ax2.transAxes)
ax3.text(.05, 1.03, '$K_c={}$'.format(format(Kcoup[2], '.2f')), 
        ha='left', va='top', transform=ax3.transAxes)

sm1 = plt.cm.ScalarMappable(norm=norm[0], cmap=cm)
sm2 = plt.cm.ScalarMappable(norm=norm[1], cmap=cm)
sm3 = plt.cm.ScalarMappable(norm=norm[2], cmap=cm)
sm1.set_array([]); sm2.set_array([]); sm3.set_array([])
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins1 = inset_axes(ax1,width="3%",height="90%",loc='right',borderpad=-1.5)
axins2 = inset_axes(ax2,width="3%",height="90%",loc='right',borderpad=-1.5)
axins3 = inset_axes(ax3,width="3%",height="90%",loc='right',borderpad=-1.5)
fig3A.colorbar(sm1, ticks=[22, 25, 28], 
              boundaries=np.arange(21.5, 28.5, 0.05),
              cax=axins1, orientation='vertical', label='period (h)')  
fig3A.colorbar(sm2, ticks=[22, 24, 26], 
              boundaries=np.arange(21.5, 26.5, 0.05),
              cax=axins2, orientation='vertical', label='period (h)')               
fig3A.colorbar(sm3, ticks=[28, 29, 30], 
              boundaries=np.arange(27.8, 30.2, 0.005),
              cax=axins3, orientation='vertical', label='period (h)')                

ax1.set_xticklabels([]); ax2.set_xticklabels([])
ax2.set_ylabel('simulated $x$'); 
ax3.set_xlabel('time (days)')
ax1.set_ylim([-2.3, +2.3])
ax2.set_ylim([-2.3, +2.3])
ax3.set_ylim([-2.3, +2.3])

fig3A.subplots_adjust(
        top=0.915,
        bottom=0.08,
        left=0.135,
        right=0.825,
        hspace=0.2,
        wspace=0.2)

# Figures 3B
fig3BCD = plt.figure(figsize=(6.5,6))

my_colors = ['#31688e', '#35b779', 'goldenrod'] 
my_labels = [
        '$K_c={}$'.format(Kcoup[0]), 
        '$K_c={}$'.format(Kcoup[1]),
        '$K_c={}$'.format(Kcoup[2])]

ax4 = fig3BCD.add_subplot(221)

periods = np.array([
    list_periods[0][0],
    list_periods[1][0],
    list_periods[2][0]])

ax4.hist(np.transpose(periods), bins=np.arange(20,36.5,.5), 
        stacked=True, color=my_colors, alpha=0.8, edgecolor='white',
        label=my_labels)   
ax4.legend(loc=2, framealpha=0.0); 
ax4.set_xlabel('period (h)'); 
ax4.set_ylabel('number of oscillators'); 
ax4.set_aspect(1.0/ax4.get_data_ratio(), adjustable='box')

##############################
#%% Load results and plot Figure 3C,D, Supplementary Figure 4

# FIGURE 3C: how do sync_index (R0) and amplitude change with Kc?
# SUPPFIG4: how does R0 change with Kc when oscillators are all identical? 
# ------------------------------------------------------------------------
list_xs_S3, list_sync_idxs_S3 = [], []

combinations_S3 = np.array([(s,l,ts,k) for s in np.arange(total_simus)+1 \
        for l in amp_rr for ts in tausd_S3 for k in Kcoup_S3])

for c in combinations_S3:
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

    # load amplitudes, phases, periods
    amp = pd.DataFrame(np.load(toload + '/amplitudes.npy'))
    phi = pd.DataFrame(np.load(toload + '/phases.npy'))
    amp.index = t; phi.index = t

    # determine synchronization index of the 100 oscillators 
    z = amp * np.exp( 1j * phi ) 
    si = abs(pd.DataFrame(np.exp(1j * np.angle(z))).sum(axis='columns')/n_oscs)

    print(
        'simulation {}/{} loaded (amplitudes, phases and circstds'.format(
            int(simu), total_simus) +
        ' computed) --- lambda={}, sig={}, tausd={}, Kcoup={}, '.format( 
            format(l, '.2f'),
            format(sigma, '.2f'),
            format(tsd, '.2f'),
            format(Kc, '.3f'),
            )) 

    list_xs_S3.append(sol_xdf)
    list_sync_idxs_S3.append(si)


# ensemble timeseries and sync. indeces -- separate in noisy+identical
# or noisy+heterog datasets
list_xs_ens_S3 = [list_xs_S3[i].values.mean(axis=1) for 
        i in range(len(list_xs_S3))] 
    #mean-field of all datasets: noisy+identical & noisy+heterogeneous

list_xs_ens_noisehet = [list_xs_ens_S3[(2*len(Kcoup_S3)*s) : \
        (2*len(Kcoup_S3)*s+len(Kcoup_S3))] for\
        s in range(total_simus)] #mean-field for noisy and heterogeneous
list_xs_ens_noise = [list_xs_ens_S3[(2*len(Kcoup_S3)*s+len(Kcoup_S3)) : \
        (2*len(Kcoup_S3)*s+2*len(Kcoup_S3))] for\
        s in range(total_simus)] #mean-field for noisy and identical
list_SI_noisehet = [list_sync_idxs_S3[(2*len(Kcoup_S3)*s) : \
        (2*len(Kcoup_S3)*s+len(Kcoup_S3))] for\
        s in range(total_simus)] #sync_indeces for noisy and heterogeneous
list_SI_noise = [list_sync_idxs_S3[(2*len(Kcoup_S3)*s+len(Kcoup_S3)) : \
        (2*len(Kcoup_S3)*s+2*len(Kcoup_S3))] for\
        s in range(total_simus)] #sync_indeces for noisy and identical 


# Calculate amplitude of ensemble (mean of all oscillators)
y = np.stack(list_xs_ens_noisehet).reshape(total_simus*len(Kcoup_S3), len(t))
ymaxs = [y[i][argrelextrema(y[i], np.greater, order=20)[0]] \
        for i in range(len(y))]
ymins = [y[i][argrelextrema(y[i], np.less, order=20)[0]] \
        for i in range(len(y))]
ms = [np.array([len(ymaxs[i]), len(ymins[i])]).min() for 
        i in range(len(y))]
amps = [ymaxs[i][-int(ms[i]/2):] - ymins[i][-int(ms[i]/2):] for i in
        range(len(y))]
np.hstack(amps)
amp2 = [abs(amps[i].mean()) for i in range(len(y))]
amp2 = np.asarray(amp2).reshape( total_simus, len(Kcoup_S3) ) 
amp2_meansimu = np.mean(amp2, axis=0)/2

# smooth the mean ensemble amplitude as a function of Kc by
# computing the average of K Lowess models that are fitted to
# 50% of the points (chosen randomly in each iteration of k)
xgrid = np.linspace(Kcoup_S3.min(),Kcoup_S3[-2])
smooths = np.stack([Fits.smooth_curve(Kcoup_S3, amp2_meansimu, 
    xgrid, frac=0.3, it=2, random_percentage=50) for k in range(100)]).T
mean = np.nanmean(smooths, axis=1) #mean of the Lowess models
stderr = scipy.stats.sem(smooths, axis=1) #standard error of mean
stderr = np.nanstd(smooths, axis=1, ddof=0)
mean_minus_sd, mean_plus_sd = mean-1.96*stderr, mean+1.96*stderr
mean_minus_sd[mean_minus_sd < 0 ] = 0 #confidence interval

# mean synchronization indeces
y2 = np.stack(list_SI_noisehet)
si_50days = y2[:,:,10000] #sync index at day 50
si_50days = si_50days.mean(axis=0)

# smooth the mean synchronization index as a function of Kc by
# computing the average of K Lowess models that are fitted to
# 50% of the points (chosen randomly in each iteration of k)
xgrid = np.linspace(Kcoup_S3.min(),Kcoup_S3.max())
smooths2 = np.stack([Fits.smooth_curve(Kcoup_S3, si_50days, 
    xgrid, frac=0.4, it=2, random_percentage=50) for k in range(100)]).T
mean2 = np.nanmean(smooths2, axis=1) #mean of the Lowess models
stderr2 = scipy.stats.sem(smooths2, axis=1) #standard error of mean
stderr2 = np.nanstd(smooths2, axis=1, ddof=0)
mean2_minus_sd, mean2_plus_sd = mean2-1.96*stderr2, mean2+1.96*stderr2
mean2_minus_sd[mean2_minus_sd < 0 ] = 0 #confidence interval

# FIGURE 3C, D: change of amplitude and sync idx as a function of Kc
# also show the smoothened average of the 100 Lowess models and
# the confidence intervals
ax5 = fig3BCD.add_subplot(223)
ax6 = fig3BCD.add_subplot(224)

ax5.fill_between(xgrid, mean2_minus_sd, mean2_plus_sd, color='k', alpha=0.15)  
ax5.plot(xgrid, mean2, color='k', alpha=0.4)
ax5.plot(Kcoup_S3, si_50days, 'o', markersize=2.5, color='k')
ax5.set_xlabel('$K_c$')
ax5.set_ylabel('$R_0$ synchr. index')
ax5.set_ylim([-0.01, 1.01])
ax5.set_xticks([0.00, 0.05, 0.10, 0.15])
ax5.set_xlim([-0.005, 0.152])
ax5.set_ylim([-0.04, 1.02])
ax5.set_aspect(1.0/ax5.get_data_ratio(), adjustable='box')

ax6.fill_between(xgrid, mean_minus_sd, mean_plus_sd, color='k', alpha=0.15)  
ax6.plot(xgrid, mean, color='k', alpha=0.4)
ax6.plot(Kcoup_S3[0:31], amp2_meansimu[0:31], 'o', markersize=2.5, color='k')
ax6.set_xlabel('$K_c$')
ax6.set_ylabel('ensemble amplitude (a.u.)')
ax6.set_xticks([0.00, 0.05, 0.10, 0.15])
ax6.set_xlim([-0.0051, 0.152])
ax6.set_ylim([-0.05, 2.57])
ax6.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0,2.5])
ax6.set_aspect(1.0/ax6.get_data_ratio(), adjustable='box')


# SUPP FIGURE 4: change of sync idx as a function of Kc: 
# noisy + identical vs noisy + heterogeneous
suppfig4 = plt.figure(figsize=(4,4))
ax_supp3 = suppfig4.add_subplot(111)

# mean synchronization indeces
y2 = np.stack(list_SI_noise)
si_50days_S3 = y2[:,:,10000] #sync index at day 50
si_50days_S3 = si_50days_S3.mean(axis=0)

ax_supp3.plot(xgrid, mean2, color='k', alpha=0.4,
        label='heterogeneous ensemble')
ax_supp3.plot(Kcoup_S3, si_50days, 'o', markersize=2.5, color='k')
ax_supp3.plot(Kcoup_S3, si_50days_S3, color='chocolate', alpha=0.4,
        label='homogeneous ensemble')
ax_supp3.plot(Kcoup_S3, si_50days_S3, 'o', markersize=2.5, color='chocolate')

ax_supp3.legend(loc="lower right", framealpha=0.0)
ax_supp3.set_xlabel('$K_c$')
ax_supp3.set_ylabel('$R_0$ synchr. index')
ax_supp3.set_xlim([-0.005, 0.152])
ax_supp3.set_ylim([-0.04, 1.02])
ax_supp3.set_xticks([0.00, 0.05, 0.10, 0.15])
ax_supp3.set_aspect(1.0/ax_supp3.get_data_ratio(), adjustable='box')


# adjust plots and save
suppfig4.subplots_adjust(
        top=0.96,
        bottom=0.135,
        left=0.160,
        right=0.935,
        hspace=0.36,
        wspace=0.425,
        )

fig3BCD.subplots_adjust(
        top=0.960,
        bottom=0.08,
        left=0.110,
        right=0.955,
        hspace=0.360,
        wspace=0.425
        )

#%% Save figures
isExist = os.path.exists('./figures/')
if not isExist:  
    os.makedirs('./figures/')

#fig3A.savefig('./figures/fig3A.pdf', format='pdf')
#fig3BCD.savefig('./figures/fig3BCD.pdf', format='pdf')
#suppfig4.savefig('./figures/suppfig4.pdf', format='pdf')


