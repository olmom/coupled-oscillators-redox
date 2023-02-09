#%% Load results and plot Figure 4D
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
import glob
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

Kcoup = np.arange(0.00, 0.11, 0.01)

# load the minimum times for entrainment (mtfe) 
# for each Kc (calculated in fig4D_preprocessing.py)
mtfe_solutions = {}
for np_name in glob.glob('./results/mtfe/*.np[yz]'):
    print(np_name)
    mtfe_solutions[np_name] = np.load(np_name)
mtfe = []
for ms in range(len(mtfe_solutions.keys())):
    solu = list(mtfe_solutions.values())[ms]
    mtfe.append(solu)
mtfe = np.asarray(mtfe)
mtfe = np.sort(np.hstack(mtfe))

fig4D = plt.figure(figsize=(4.5,4.5))
axD = fig4D.add_subplot(111)
axD.plot(Kcoup, mtfe,'o',c='k')
axD.set_xlabel('mean-field coupling strength $K_c$')
axD.set_ylabel('min. TTFL input for entrainment (a.u.)')
axD.set_aspect(1.0/axD.get_data_ratio(), adjustable='box')
fig4D.subplots_adjust(
        top=0.88,
        bottom=0.180,
        left=0.195,
        right=0.760,
        hspace=0.2,
        wspace=0.2)

#%% Save figure
isExist = os.path.exists('./figures/')
if not isExist:  
    os.makedirs('./figures/')

fig4D.savefig('./figures/fig4D.pdf', format='pdf')
