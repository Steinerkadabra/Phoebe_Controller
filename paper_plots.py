import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

print(plt.rcParams)
plt.rcParams.update({'font.size': 15, 'xtick.labelsize': 'small', 'ytick.labelsize': 'small',})
dir = 'paper_plots/'
figsize1 = (10,5)
figsize_onecolumn = (15,7.5)

dpi = 200
do_all = False



### Figure 1: lightcurve, gaps and shit ###
if False or do_all:
    data = np.loadtxt('endurance_data/' + 'RS_Cha_lightcurve.txt').T
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
    start = np.min(time[:int(len(time)/3)])
    end = np.max(time[:int(len(time)/3)])
    print(end)
    #ax.errorbar((start+end)/2, 0.4, xerr = (start-end)/2, capsize = 4)
    plt.tight_layout()
    plt.savefig(dir + 'Figure1_full_lc_p_gaps.pdf', bbox_inches = 0)
    #plt.show()
    plt.close()


### Figure 2: Zoom in non transit ###
if False or do_all:
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

### Figure 3
def gaus(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c

def line(x, a, b):
    return a * x + b

if False or do_all:
    from scipy.optimize import curve_fit
    import lightkurve as lk
    data = np.loadtxt('endurance_data/' + 'Removed_Pulsations_from_first_run.txt').T
    time = data[0]
    flux = data[1]
    data_ = np.loadtxt('endurance_data/' + 'RS_Cha_lightcurve.txt').T
    time_m = data_[0]
    flux_m = data_[1]

    period = 1.66987725

    #import phoebe
    #import phoebe_controller

    #d = {'ecc@binary': 0.01}
    #chi, m_flux = phoebe_controller.chi_square_multi(d, flux, time, np.ones(len(time)))


    new_lc = lk.LightCurve(time, flux)
    new_lc.normalize()
    ntf = new_lc.time / (0.5 * period)
    ntf2 = time_m / (0.5 * period)
    fts = new_lc.flux # sig.savgol_filter(new_lc.flux, 15, 3)
    fts2 = flux_m
    #print(mflux)
    o_p = []
    c_p = []
    o_s = []
    c_s = []
    t0 = (1599.397275) / (0.5 * period) % 1
    for i in range(1914, 2015):
        if i in [1998, 1994, 1979]:
            continue
        try:
            ind = np.where(np.logical_and(ntf >= i, ntf <= i + 1))
            ind2 = np.where(np.logical_and(ntf2 >= i, ntf2 <= i + 1))
            minimum = np.where(fts[ind] == np.amin(fts[ind]))
            popt, pcov = curve_fit(gaus, ntf[ind], fts[ind], p0=[-0.5, t0 + i, 0.2, 1])
            popt2, pcov2 = curve_fit(gaus, ntf2[ind], fts2[ind], p0=[-0.5, t0 + i, 0.2, 1])
            #print(popt, popt2)
            plt.plot(ntf[ind], fts[ind], 'k-')
            plt.plot(ntf2[ind2], fts2[ind2], 'r-')
            #plt.plot(ntf2[ind], fts2[ind], 'r-')
            plt.plot(ntf[ind], gaus(ntf[ind], *popt), 'b-')
            plt.plot(ntf2[ind2], gaus(ntf[ind2], *popt2), 'c-')
            #ax.plot(ntf[ind][minimum], np.amin(fts[ind]), 'kx')
            plt.plot([t0 + i, t0 + i], [0.59, 0.7], 'k-')
            if 0.62 > np.amin(fts[ind]) > 0.58:
                c_p.append(t0 + i)
                #o_p.append(popt2[1])
                o_p.append(popt[1])
            elif 0.66 > np.amin(fts[ind]) > 0.63:
                c_s.append(t0 + i)
                #o_s.append(popt2[1])
                o_s.append(popt[1])
            else:
                pass
        except ValueError:
            pass
        except RuntimeError:
            pass
        except TypeError:
            pass
    plt.show()

    fig, ax = plt.subplots(figsize = figsize1, dpi = dpi)

    o_p = np.array(o_p)*(0.5 * period)
    c_p = np.array(c_p)*(0.5 * period)
    o_s = np.array(o_s)*(0.5 * period)
    c_s = np.array(c_s)*(0.5 * period)
    ax.plot(o_p, np.array(o_p) - np.array(c_p), 'go')
    ax.plot(o_s, np.array(o_s) - np.array(c_s), 'b^')
    ax.plot(ax.set_xlim(), [0.0, 0.0], 'k--')
    popt_p, pcov_s = curve_fit(line, o_p, np.array(o_p) - np.array(c_p), p0=[0, 0])
    popt_s, pcov_p = curve_fit(line, o_s, np.array(o_s) - np.array(c_s), p0=[0, 0])
    ax.plot(o_p, line(o_p, *popt_p), 'g-')
    ax.plot(o_s, line(o_s, *popt_s), 'b-')

    ax.set_xlabel('Time - 2457000 [BTJD days]')
    ax.set_ylabel('O-C days')
    plt.tight_layout()
    plt.savefig(dir + 'Figure3_O_C_diagram.pdf', bbox_inches = 0)


#### Mail #####

if True:
    fig, ax = plt.subplots(3,1, figsize = figsize1, dpi = dpi)
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    ax[0].plot(time, flux, 'ko', markersize = 0.75)
    data2 = np.loadtxt('endurance/' + 'lets_try_a_third_of_the_lightcurve/binary_model.txt').T
    time2 = data2[0]
    flux2 = data2[1]
    ax[0].plot(time2, flux2, 'r-', lw = 0.75)
    ax[1].plot(time2, flux - flux2, 'ko', markersize = 0.75)
    time = data[0]
    flux = data[1]
    ax[0].set_xlabel('Time - 2457000 [BTJD days]')
    ax[0].set_ylabel('normalized flux')
    ax[1].set_xlabel('Time - 2457000 [BTJD days]')
    ax[1].set_ylabel('normalized flux')
    ax[0].set_xlim(1600, 1610)
    ax[1].set_xlim(1600, 1610)
    lc = lk.LightCurve(time, flux - flux2)
    pdg=lc.to_periodogram()
    ax[2].plot(pdg.frequency, pdg.power, 'k-')
    for i in range(1, 60):
        ax[2].plot([i * 1/1.66987725, i / 1.66987725], ax[2].set_ylim(), 'b--', lw = 0.5)
    ax[2].set_xlim(8, 35)
    ax[2].set_xlabel('frequency $d^{-1}')
    ax[2].set_ylabel('power')
    plt.tight_layout(h_pad=0, w_pad=0)

    plt.savefig(dir + 'mail.png', bbox_inches = 0)