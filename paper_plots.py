import numpy as np
import matplotlib.pyplot as plt

print(plt.rcParams)
plt.rcParams.update({'font.size': 15, 'xtick.labelsize': 'small', 'ytick.labelsize': 'small',})
dir = 'paper_plots/'
figsize1 = (10,5)
figsize_onecolumn = (15,7.5)

dpi = 200
do_all = True

### Figure 1: lightcurve, gaps and shit ###
if False or do_all:
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    fig, ax = plt.subplots(figsize = figsize1, dpi = dpi)
    ax.plot(time, flux, 'ko', markersize = 0.75)
    ax.set_xlabel('Time - 2457000 [BTJD days]')
    ax.set_ylabel('normalized flux')
    orbits = [1596.77203, 1610.77620, 1640.03312, 1624.94979, 1653.91505, 1668.61921]
    cadence_start = 265908
    cadence = [265908, 275991]
    guiding = [1599.94148, 1614.19842]
    cadence_removed = [266841, 277043]
    lim = ax.set_ylim()
    for t in orbits:
        ax.plot([t, t], lim, 'r-')
    for i in range(2):
        ax.plot([1596.77203 + (cadence_removed[i]-cadence_start)*2/60/24, 1596.77203 + (cadence_removed[i]-cadence_start)*2/60/24], lim, 'b--')

    plt.tight_layout()
    plt.savefig(dir + 'Figure1_full_lc_p_gaps.pdf', bbox_inches = 0)


### Figure 2: Zoom in non transit ###
if True or do_all:
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    print(len(time))
    gap_time = 0
    for i in range(len(time)-1):
        if time[i+1]-time[i] > 0.6:
            gap_time+= time[i+1]-time[i]
    print((84.28-gap_time)/84.28)
    fig, ax = plt.subplots(figsize = figsize_onecolumn, dpi = dpi)
    ax.plot(time, flux, 'ko', markersize = 1)
    ax.set_xlabel('Time - 2457000 [BTJD days]')
    ax.set_ylabel('normalized flux')
    ax.set_xlim(1632.85, 1636.9)
    ax.set_ylim(0.98, 1.03)
    plt.tight_layout()
    plt.savefig(dir + 'Figure2_zoom_in_lc.pdf', bbox_inches = 0)