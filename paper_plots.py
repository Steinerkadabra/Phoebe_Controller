import numpy as np
import phase_analysis as pha
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.optimize import curve_fit
import scipy.interpolate as interp
import scipy.integrate as integ
from tabulate import tabulate
import time as timey

print(plt.rcParams)
plt.rcParams.update({'font.size': 20, 'xtick.labelsize': 'small', 'ytick.labelsize': 'small',})
dir = 'paper_plots/'
figsize1 = (10,5)
figsize_onecolumn = (20,7.5)

def figsize(n):
    return (10,5*n)

dpi = 200
do_all = False
do_mail = False
do_talk = False



### Figure 1: lightcurve, gaps and shit ###
if False or do_all:
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    fig, ax = plt.subplots(figsize = figsize_onecolumn, dpi = dpi)
    ax.plot(time, flux, 'ko', markersize = 0.75, rasterized = True)
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
    plt.savefig(dir + 'Figure1_full_lc_p_gaps_raster_stretch.pdf', bbox_inches = 0)
    plt.show()
    plt.close()

## fig2 evolutionary tracks
if False or do_all:
    import os
    files = 'evol_tracks/tracks_final/LOGS/'
    num = len(os.listdir(files))
    fig, ax = plt.subplots(figsize = figsize1, dpi = dpi)
    Rp = 2.15
    Rs = 2.36
    Rerr = 0.02
    t_c12 = []
    l_c12 = []
    for i in range(1, num):
        hist = np.genfromtxt(files + f'grid_{i}/history.data', skip_header=5, names = True)
        ms = -1
        for j in range(2, len(hist['star_age'])):
            if hist['center_c12'][j] < 1e-3:
                t_c12.append(hist['log_Teff'][j-1])
                l_c12.append(hist['log_L'][j-1])
                break
        for j in range(2, len(hist['star_age'])):
            if hist['center_h1'][j]<0.725:
                ms = j
                break
        ax.plot(hist['log_Teff'][50:ms], hist['log_L'][50:ms], 'k-', lw =1, alpha = 0.5)
    ax.plot(t_c12, l_c12, 'g-.', lw =3)

    files = 'evol_tracks/tracks5/LOGS/'
    num = len(os.listdir(files))
    for i in range(1, num):
        hist = np.genfromtxt(files + f'grid_{i}/history.data', skip_header=5, names = True)
        ms = -1
        for j in range(len(hist['star_age'])):
            if hist['center_h1'][j]<0.725:
                ms = j
                break
        #ax.plot(hist['log_Teff'][50:ms], hist['log_L'][50:ms], 'r-', lw =0.5 )
        fittingtp = []
        fittingts = []
        fittinglp = []
        fittingls = []
        ages_p= []
        ages_s=[]
        ptlp = 0
        ptls = 0
        if abs(hist['star_mass'][-1] -  1.89) < 0.01:
            for j in range(len(hist['star_age'])):
                if Rp-Rerr < 10**hist['log_R'][j] < Rp +Rerr and hist['log_Teff'][j]>3.8 and 9.17 > hist['star_age'][j]/1e6 > 9.125:
                    fittingtp.append(hist['log_Teff'][j])
                    fittinglp.append(hist['log_L'][j])
                    ages_p.append(hist['star_age'][j])
                    print('primary', hist['star_mass'][-1])
                    ptlp = j
                    break
        if abs(hist['star_mass'][-1] - 1.87) < 0.01:
            for j in range(len(hist['star_age'])):
                if Rs-Rerr < 10**hist['log_R'][j] < Rs +Rerr and hist['log_Teff'][j]>3.8and 9.17 > hist['star_age'][j]/1e6 > 9.125:
                    fittingts.append(hist['log_Teff'][j])
                    fittingls.append(hist['log_L'][j])
                    ages_s.append(hist['star_age'][j])
                    print('secondary', hist['star_mass'][-1])
                    ptls = j
                    break
        if ptlp != 0:
            ax.plot(hist['log_Teff'][50:ptlp], hist['log_L'][50:ptlp], 'b-', lw=2)
            ax.plot(hist['log_Teff'][ptlp-1], hist['log_L'][ptlp-1], 'bo')
            print(hist['log_L'][ptlp])
        if ptls != 0:
            ax.plot(hist['log_Teff'][50:ptls], hist['log_L'][50:ptls], 'r-', lw=2)
            ax.plot(hist['log_Teff'][ptls-1], hist['log_L'][ptls-1], 'ro')
            print(hist['log_L'][ptls])
        try:
            print('primary:', np.min(ages_p)/1e6, np.max(ages_p)/1e6)
        except ValueError:
            pass
        try:
            print('secondary:', np.min(ages_s)/1e6, np.max(ages_s)/1e6)
        except ValueError:
            pass
        #ax.plot(fittingtp, fittinglp, 'co')
        #ax.plot(fittingts, fittingls, 'mo')
    lp = 1.15
    lp_err = 0.09
    teffp = 7638
    teffp_err = 76
    ls = 1.13
    ls_err = 0.09
    teffs = 7228
    teffs_err = 72
    lsp = [lp -lp_err, lp+lp_err, lp+lp_err,  lp-lp_err,  lp-lp_err]
    lss = [ls -ls_err, ls+ls_err, ls+ls_err,  ls-ls_err,  ls-ls_err]
    tsp = np.log10(np.array([teffp -teffp_err, teffp -teffp_err, teffp +teffp_err, teffp +teffp_err, teffp -teffp_err]))
    tss = np.log10(np.array([teffs -teffs_err, teffs -teffs_err, teffs +teffs_err, teffs +teffs_err, teffs -teffs_err]))
    ax.plot(tsp, lsp, 'b-', lw = 2)
    ax.plot(tss, lss, 'r-', lw = 2)
    ax.set_xlabel('log($T_{\\rm eff}$)')
    ax.set_ylabel('log($L/L_\odot$)')
    ax.invert_xaxis()
    ax.set_ylim([0.5, 1.625])
    ax.set_xlim([3.975, 3.75])
    plt.tight_layout()
    plt.savefig(dir + 'Figure1_evolutionary_status.pdf', bbox_inches = 0)
    # plt.show()



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
    plt.savefig(dir + 'Figure2_zoom_in_lc_stretched.pdf', bbox_inches = 0)

### Figure 3
def gaus(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c

def line(x, a, b):
    return a * x + b

if False or do_all:
    data = np.loadtxt('endurance/' + 'Removed_Pulsations_from_first_run.txt').T
    time = data[0]
    flux = data[1]
    data_ = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
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
    #plt.show()

    fig, ax = plt.subplots(figsize = figsize1, dpi = dpi)

    o_p = np.array(o_p)*(0.5 * period)*24*3600
    c_p = np.array(c_p)*(0.5 * period)*24*3600
    o_s = np.array(o_s)*(0.5 * period)*24*3600
    c_s = np.array(c_s)*(0.5 * period)*24*3600
    ax.plot(o_p, np.array(o_p) - np.array(c_p), 'go')
    ax.plot(o_s, np.array(o_s) - np.array(c_s), 'b^')
    ax.plot(ax.set_xlim(), [0.0, 0.0], 'k--')
    popt_p, pcov_s = curve_fit(line, o_p, np.array(o_p) - np.array(c_p), p0=[0, 0])
    popt_s, pcov_p = curve_fit(line, o_s, np.array(o_s) - np.array(c_s), p0=[0, 0])
    ax.plot(o_p, line(o_p, *popt_p), 'g-')
    ax.plot(o_s, line(o_s, *popt_s), 'b-')

    ax.set_xlabel('Time - 2457000 [BTJD days]')
    ax.set_ylabel('O-C seconds')
    plt.tight_layout()
    plt.savefig(dir + 'Figure3_O_C_diagram_seconds.pdf', bbox_inches = 0)


def binning(time, mag, period):
    time = (time%period)/period# (1/0.5988495842998753)*0.5988495842998753
    bins = []
    means = []
    sds = []
    num = 600
    bins.append(0)
    ind = np.where(np.logical_and(time >= 0, time <= 0.5 / num))
    means.append(np.mean(mag[ind]))
    sds.append(np.std(mag[ind]))
    for i in range(num - 1):
        ind = np.where(np.logical_and(time >= (i + 0.5) / num, time <= (i + 1.5) / num))
        if ind[0].size > 0:
            bins.append((i + 1) / num)
            means.append(np.mean(mag[ind]))
            sds.append(np.std(mag[ind]))
    bins.append(1)
    ind = np.where(np.logical_and(time >= (num - 0.5) / num, time <= 1))
    means.append(np.mean(mag[ind]))
    sds.append(np.std(mag[ind]))
    return bins, means, sds

### Figure 4: Result of Binary Modelling ###
if False or do_all:
    period = 1.66987725
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    bins, means, sds = binning(time, flux, period)
    data = np.loadtxt('endurance/' + 'Removed_Pulsations_from_first_run_irrad_gravbol/binary_model.txt').T
    time_mod = data[0]
    flux_mod = data[1]
    data = np.loadtxt('endurance/' + 'Fourier_residual_model.txt').T
    time_res = data[0]
    flux_res = data[1]
    phase = time_mod%period/period
    sort = np.argsort(phase)
    rbins, rmeans, rsds = binning(time, flux-flux_mod, period)
    rrbins, rrmeans, rrsds = binning(time, flux-flux_mod - flux_res, period)
    fig= plt.figure(constrained_layout=True, figsize = figsize1, dpi = dpi)
    gs = fig.add_gridspec(9, 1)
    ax2 = fig.add_subplot(gs[0:-4, 0])
    ax3 = fig.add_subplot(gs[-4:-2, 0])
    ax4 = fig.add_subplot(gs[-2:, 0])
    ax2.plot(bins, means, 'ko', ms = 2)
    ax2.plot(phase[sort], flux_mod[sort], 'r-', lw = 0.75)
    ax3.plot(rbins, rmeans, 'ko', ms = 2)
    ax4.plot(rrbins, rrmeans, 'ko', ms = 2)
    ax2.set_xticks([])
    ax3.set_ylim([-0.004, 0.0035])
    ax4.set_ylim([-0.004, 0.0035])
    ax3.set_yticks([-0.002, 0, 0.002])
    ax4.set_yticks([-0.002, 0, 0.002])
    ax4.set_xlabel('orbital phase')
    ax2.set_ylabel('flux')
    ax3.set_ylabel('flux')
    ax4.set_ylabel('flux')
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.savefig(dir + '/Figure4_binary_modelling.pdf', bbox_inches=0)
    plt.show()

    figsi, axi, = plt.subplots()
    resid = lk.LightCurve(rrbins, rrmeans).to_periodogram()
    axi.plot(resid.frequency, resid.power)
    plt.show()


### Figure5: Different Amplitude Spectra ###

if False or do_all:

    plt.rcParams.update({'font.size': 16, 'xtick.labelsize': 'small', 'ytick.labelsize': 'small','xtick.minor.visible': True })
    freq = 1/1.66987725
    period = 1.66987725
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    data2 = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
    time2 = data2[0]
    flux2 = data2[1]
    ind = np.where(np.logical_and(abs(time % period / period - 0.3) >= 0.1, abs(time % period / period - 0.775) >= 0.125))
    pdg_orig = lk.LightCurve(time, flux).to_periodogram()
    pdg_it1 = lk.LightCurve(time2, flux2).to_periodogram()
    pdg_ooe = lk.LightCurve(time2[ind], flux2[ind]).to_periodogram()
    fig = plt.figure(constrained_layout=True,  figsize = figsize_onecolumn, dpi = dpi)
    gs = fig.add_gridspec(3, 3)
    ax00 = fig.add_subplot(gs[0, :-1])
    ax01 = fig.add_subplot(gs[1, :-1])
    ax02 = fig.add_subplot(gs[2, :-1])
    ax10 = fig.add_subplot(gs[0, -1:])
    ax11 = fig.add_subplot(gs[1, -1:])
    ax12 = fig.add_subplot(gs[2, -1:])
    ax00.plot(pdg_orig.frequency, pdg_orig.power, 'k-', lw = 0.5, rasterized=True)
    ax01.plot(pdg_it1.frequency, pdg_it1.power, 'k-', lw = 0.5, rasterized=True)
    ax02.plot(pdg_ooe.frequency, pdg_ooe.power, 'k-', lw = 0.5, rasterized=True)
    ax10.plot(time%period/period, flux, 'ko', ms = 0.25, rasterized=True)
    ax11.plot(time%period/period, flux2, 'ko', ms = 0.25, rasterized=True)
    ax12.plot(time[ind]%period/period, flux2[ind], 'ko', ms = 0.25, rasterized=True)
    ax11.set_ylim(ax12.set_ylim())
    ax01.set_ylim(ax02.set_ylim())

    # fig.text(0.05, 0.5, "power (relative flux)", ha="center", va="center", rotation=90)
    ax01.set_ylabel('amplitude (relative flux)')
    ax11.set_ylabel('relative flux')
    ax02.set_xlabel('frequency ${\\rm d}^{-1}$')
    ax12.set_xlabel('orbital phase')

    for a in [ax00, ax01, ax11, ax10]:
        a.set_xticks([])
    for a in [ax00, ax01, ax02]:
        # a.set_xlabel('frequency ${\\rm d}^{-1}$')
        # a.set_ylabel('power (relative flux)')
        a.set_xlim(0, 34.9)
        # a.plot(a.set_xlim(), [0.000165, 0.000165], 'r:', lw = 0.25)
        for i in range(80):
            a.plot([i*freq, i*freq], [0, a.set_ylim()[1]], 'b--', lw = 0.25)
    for a in [ax10, ax11, ax12]:
        # a.set_xlabel('Phase')
        # a.set_ylabel('flux')

        a.yaxis.tick_right()
        a.yaxis.set_label_position("right")
        a.set_xlim([0,1])
    for a in [ ax11, ax12]:
        a.set_yticks([-0.01, 0, 0.01])
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    ax02zoom = zoomed_inset_axes(ax02, 2, loc=2)  # zoom-factor: 2.5, location: upper-left
    ax01zoom = zoomed_inset_axes(ax01, 2, loc=2)  # zoom-factor: 2.5, location: upper-left
    ax02zoom.plot(pdg_ooe.frequency, pdg_ooe.power, 'k-', lw = 0.5, rasterized=True)
    ax01zoom.plot(pdg_it1.frequency, pdg_it1.power, 'k-', lw = 0.5, rasterized=True)
    for a in [ax02zoom, ax01zoom]:
        a.set_xlim(1, 5)
        a.set_ylim(0, 0.0005)
        a.xaxis.set_visible('False')
        a.yaxis.set_visible('False')
        a.set_xticks([])
        a.set_yticks([])
        for i in range(80):
            a.plot([i*freq, i*freq], [0, ax02zoom.set_ylim()[1]], 'b--', lw = 0.25)


    mark_inset(ax02, ax02zoom, loc1=2, loc2=4, fc="none", ec="0.5")
    mark_inset(ax01, ax01zoom, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout(h_pad=0, w_pad=0)

    plt.subplots_adjust(hspace=0)
    plt.savefig(dir + '/Figure5_amplitude_spectra_label.pdf', bbox_inches=0)
    #plt.show()

    plt.rcParams.update({'font.size': 20, 'xtick.labelsize': 'small', 'ytick.labelsize': 'small', })

### Figure6: Why use Gaussian drop in Ampplitude?###
if False or do_all:
    freq = 1/1.66987725
    period = 1.66987725
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    data2 = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
    time2 = data2[0]
    flux2 = data2[1]
    fs = [11.0703910326469, 11.624895962024176, 12.79520550704062, 20.122516352277714,
          18.924907836258388, ]
    amps = [0.0016715568284938562, 0.0016435120422416158, 0.0014987863342293011, 0.0009152175177519561,
            0.000851788328761722, ]
    modes = []
    for f, a in zip(fs, amps):
        modes.append(pha.mode(f, a, 0, 0))
    for p in np.linspace(0, 1, 50):
        modes = pha.amplitude_in_phase(time2, flux2, modes, phase=p)

    fig = plt.figure(constrained_layout=True, figsize = figsize(3), dpi = dpi)
    gs = fig.add_gridspec(5, 1)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[2,0])
    ax4 = fig.add_subplot(gs[4,0])
    ax5 = fig.add_subplot(gs[3,0])
    for a, i in zip([ax1, ax2, ax3], range(3)):
        a.errorbar(modes[i].phase_bins, modes[i].phase_amps, yerr=modes[i].phase_amps_, ls='', marker='o', color= 'k', ms = 5, capsize = 3)

    for a in [ax1, ax2, ax3, ax4]:
        a.set_xlim(0,1)
    for a in [ax1, ax2, ax3]:
        a.set_ylabel('Amplitude')
        a.set_xticks([])
    ax4.plot(time % period / period, flux, 'ko', ms=0.75)
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Flux')
    for a, i in zip([ax1, ax2, ax3], range(3)):
        x = np.array(modes[i].phase_bins)
        y = np.array(modes[i].phase_amps)
        sds = np.array(modes[i].phase_amps_)
        ind = np.where(abs(x-0.3) <= 0.2)
        popt, pcov = curve_fit(gaus, x[ind], y[ind], sigma = sds[ind], absolute_sigma=True,  p0=[1, 0.3, 0.05, 0.002])
        xda = np.linspace(0.3-0.2,0.3+0.2, 500)
        a.plot(xda, gaus(xda, *popt), 'r-')
        Nexp = gaus(x[ind], *popt)
        r = np.array(y[ind]) - Nexp
        chisq = np.sum((r / sds[ind]) ** 2)
        df = len(x[ind]) - 4
        string = f'F{i+1}: {np.round(modes[i].f,2)}' +' d$^{-1}$\n$\\frac{\chi^2}{\mathrm{red}} = $' + str(np.round(chisq/df, 2))
        a.text(0.775, a.set_ylim()[0]+ 0.1*(a.set_ylim()[1]-a.set_ylim()[0]), string, fontsize = 'small')

    x = np.array(modes[0].phase_bins)
    y1 = np.array(modes[0].phase_amps)
    y2 = np.array(modes[2].phase_amps)
    y3 = np.array(modes[1].phase_amps)
    y = (y1/np.mean(y1)+ y2/np.mean(y2))/2
    sds1 = np.array(modes[0].phase_amps_)
    sds2 = np.array(modes[2].phase_amps_)
    sds_1 = sds1/np.mean(y1)
    sds_2 = y1*np.std(y1)/(np.mean(y1)**2*np.sqrt(len(x)))
    sds_3 = sds2/np.mean(y2)
    sds_4 = y2*np.std(y2)/(np.mean(y2)**2*np.sqrt(len(x)))
    sds_5 = np.sqrt((y1/np.mean(y1)- y)**2 + (y2/np.mean(y2)- y)**2)
    sds = np.sqrt(sds_1**2 + sds_2**2 + sds_3**2 + sds_4**2+ sds_5**2)/2
    #sds = np.sqrt(sds_1**2 + sds_3**2 )/2
    ax5.errorbar(x, y, yerr = sds,  ls='', marker='o', color= 'k', ms = 5, capsize = 3)

    popt, pcov = curve_fit(gaus, x, y, sigma=sds, absolute_sigma=True, p0=[1, 0.3, 0.05, 0.002])
    xda = np.linspace(0,1, 500)
    ax5.plot(xda, gaus(xda, *popt), 'r-')
    Nexp = gaus(x, *popt)
    r = np.array(y) - Nexp
    chisq = np.sum((r / sds) ** 2)
    df = len(x) - 4
    string = '$\\frac{\chi^2}{\mathrm{red}} = $' + str(np.round(chisq/df, 2))
    ax5.text(0.825, ax5.set_ylim()[0]+ 0.1*(ax5.set_ylim()[1]-ax5.set_ylim()[0]), string)

    ax5.set_ylabel('Amplitude / Mean\n F1 + F3')
    ax5.set_xticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(dir + '/Figure6_model_motivation_label.pdf', bbox_inches=0)


### Figure6: Why use Gaussian drop in Ampplitude? Appenix Revised###
if True or do_all:
    freq = 1/1.66987725
    period = 1.66987725
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    data2 = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
    time2 = data2[0]
    flux2 = data2[1]
    fs = [11.0703910326469, 11.624895962024176, 12.79520550704062, 20.122516352277714,
          18.924907836258388, ]
    amps = [0.0016715568284938562, 0.0016435120422416158, 0.0014987863342293011, 0.0009152175177519561,
            0.000851788328761722, ]
    modes = []
    for f, a in zip(fs, amps):
        modes.append(pha.mode(f, a, 0, 0))
    for p in np.linspace(0, 1, 50):
        modes = pha.amplitude_in_phase(time2, flux2, modes, phase=p)

    fig = plt.figure(constrained_layout=True, figsize = figsize(3), dpi = dpi)
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[2,0])
    ax4 = fig.add_subplot(gs[3,0])
    # ax5 = fig.add_subplot(gs[3,0])
    for a, i in zip([ax1, ax2, ax3], range(3)):
        a.errorbar(modes[i].phase_bins, modes[i].phase_amps, yerr=modes[i].phase_amps_, ls='', marker='o', color= 'k', ms = 5, capsize = 3)

    for a in [ax1, ax2, ax3, ax4]:
        a.set_xlim(0,1)
    for a in [ax1, ax2, ax3]:
        a.set_ylabel('Amplitude')
        a.set_xticks([])
    ax4.plot(time % period / period, flux, 'ko', ms=0.75)
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Flux')
    for a, i in zip([ax1,ax3], [0,2]):
        x = np.array(modes[i].phase_bins)
        y = np.array(modes[i].phase_amps)
        sds = np.array(modes[i].phase_amps_)
        ind = np.where(abs(x-0.3) <= 0.2)
        popt, pcov = curve_fit(gaus, x[ind], y[ind], sigma = sds[ind], absolute_sigma=True,  p0=[1, 0.3, 0.05, 0.002])
        xda = np.linspace(0.3-0.2,0.3+0.2, 500)
        a.plot(xda, gaus(xda, *popt), 'r-')
        Nexp = gaus(x[ind], *popt)
        r = np.array(y[ind]) - Nexp
        chisq = np.sum((r / sds[ind]) ** 2)
        df = len(x[ind]) - 4
        string = f'F{i+1}: {np.round(modes[i].f,2)}' +' d$^{-1}$\n$\\frac{\chi^2}{\mathrm{red}} = $' + str(np.round(chisq/df, 2))
        a.text(0.775, a.set_ylim()[0]+ 0.1*(a.set_ylim()[1]-a.set_ylim()[0]), string, fontsize = 'small')
    for a, i in zip([ax2], [1]):
        x = np.array(modes[i].phase_bins)
        y = np.array(modes[i].phase_amps)
        sds = np.array(modes[i].phase_amps_)
        ind = np.where(abs(x-0.35) <= 0.1)
        popt, pcov = curve_fit(gaus, x[ind], y[ind], sigma = sds[ind], absolute_sigma=True,  p0=[1, 0.3, 0.05, 0.002])
        xda = np.linspace(0.3-0.2,0.3+0.2, 500)

        ind = np.where(abs(x-0.3) <= 0.2)
        a.plot(xda, gaus(xda, *popt), 'r-')
        Nexp = gaus(x[ind], *popt)
        r = np.array(y[ind]) - Nexp
        chisq = np.sum((r / sds[ind]) ** 2)
        df = len(x[ind]) - 4
        string = f'F{i+1}: {np.round(modes[i].f,2)}' +' d$^{-1}$\n$\\frac{\chi^2}{\mathrm{red}} = $' + str(np.round(chisq/df, 2))
        a.text(0.775, a.set_ylim()[0]+ 0.1*(a.set_ylim()[1]-a.set_ylim()[0]), string, fontsize = 'small')

    # x = np.array(modes[0].phase_bins)
    # y1 = np.array(modes[0].phase_amps)
    # y2 = np.array(modes[2].phase_amps)
    # y3 = np.array(modes[1].phase_amps)
    # y = (y1/np.mean(y1)+ y2/np.mean(y2))/2
    # sds1 = np.array(modes[0].phase_amps_)
    # sds2 = np.array(modes[2].phase_amps_)
    # sds_1 = sds1/np.mean(y1)
    # sds_2 = y1*np.std(y1)/(np.mean(y1)**2*np.sqrt(len(x)))
    # sds_3 = sds2/np.mean(y2)
    # sds_4 = y2*np.std(y2)/(np.mean(y2)**2*np.sqrt(len(x)))
    # sds_5 = np.sqrt((y1/np.mean(y1)- y)**2 + (y2/np.mean(y2)- y)**2)
    # sds = np.sqrt(sds_1**2 + sds_2**2 + sds_3**2 + sds_4**2+ sds_5**2)/2
    # #sds = np.sqrt(sds_1**2 + sds_3**2 )/2
    # ax5.errorbar(x, y, yerr = sds,  ls='', marker='o', color= 'k', ms = 5, capsize = 3)
    #
    # popt, pcov = curve_fit(gaus, x, y, sigma=sds, absolute_sigma=True, p0=[1, 0.3, 0.05, 0.002])
    # xda = np.linspace(0,1, 500)
    # ax5.plot(xda, gaus(xda, *popt), 'r-')
    # Nexp = gaus(x, *popt)
    # r = np.array(y) - Nexp
    # chisq = np.sum((r / sds) ** 2)
    # df = len(x) - 4
    # string = '$\\frac{\chi^2}{\mathrm{red}} = $' + str(np.round(chisq/df, 2))
    # ax5.text(0.825, ax5.set_ylim()[0]+ 0.1*(ax5.set_ylim()[1]-ax5.set_ylim()[0]), string)
    #
    # ax5.set_ylabel('Amplitude / Mean\n F1 + F3')
    # ax5.set_xticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(dir + '/Appendix_model_motivation_label.pdf', bbox_inches=0)
    plt.show()



### Figure6: Phase_dependent_amplitudes###
if False or do_all:
    freq = 1/1.66987725
    period = 1.66987725
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    data2 = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
    time2 = data2[0]
    flux2 = data2[1]
    # fs = [11.0703910326469, 11.624895962024176, 12.79520550704062, 20.122516352277714,
    #       18.924907836258388, 12.8225, 10.427359]
    # amps = [0.0016715568284938562, 0.0016435120422416158, 0.0014987863342293011, 0.0009152175177519561,
    #         0.000851788328761722, 0.000689, 0.000584]
    # modes = []
    # for f, a in zip(fs, amps):
    #     modes.append(pha.mode(f, a, 0, 0))
    # for p in np.linspace(0, 1, 50):
    #     modes = pha.amplitude_in_phase(time2, flux2, modes, phase=p)

    fig = plt.figure(constrained_layout=True, figsize = figsize(3), dpi = dpi)
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[2,0])
    ax4 = fig.add_subplot(gs[3,0])
    # for a, i in zip([ax1, ax2, ax3], range(3)):
    #     a.errorbar(modes[i].phase_bins, modes[i].phase_amps, yerr=modes[i].phase_amps_, ls='', marker='^', color= 'r', ms = 3, capsize = 2)

    fs = [11.0703910326469, 11.624895962024176, 12.79520550704062, 20.122516352277714,
          18.924907836258388]
    amps = [0.0016715568284938562, 0.0016435120422416158, 0.0014987863342293011, 0.0009152175177519561,
            0.000851788328761722]
    modes = []
    for f, a in zip(fs, amps):
        modes.append(pha.mode(f, a, 0, 0))
    for p in np.linspace(0, 1, 50):
        modes = pha.amplitude_in_phase(time2, flux2, modes, phase=p)
    for a, i in zip([ax1, ax2, ax3], range(3)):
        a.errorbar(modes[i].phase_bins, modes[i].phase_amps, yerr=modes[i].phase_amps_, ls='', marker='o', color= 'k', ms = 5, capsize = 3)



    for a in [ax1, ax2, ax3, ax4]:
        a.set_xlim(0,1)
    for a in [ax1, ax2, ax3]:
        a.set_ylabel('Amplitude')
        a.set_xticks([])
    ax4.plot(time % period / period, flux, 'ko', ms=0.75)
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Flux')
    for a, i in zip([ax1, ax2, ax3], range(3)):
    #     x = np.array(modes[i].phase_bins)
    #     y = np.array(modes[i].phase_amps)
    #     sds = np.array(modes[i].phase_amps_)
    #     ind = np.where(abs(x-0.3) <= 0.2)
    #     popt, pcov = curve_fit(gaus, x[ind], y[ind], sigma = sds[ind], absolute_sigma=True,  p0=[1, 0.3, 0.05, 0.002])
    #     xda = np.linspace(0.3-0.2,0.3+0.2, 500)
    #     a.plot(xda, gaus(xda, *popt), 'r-')
    #     Nexp = gaus(x[ind], *popt)
    #     r = np.array(y[ind]) - Nexp
    #     chisq = np.sum((r / sds[ind]) ** 2)
    #     df = len(x[ind]) - 4
    #     string = f'F{i+1}: {np.round(modes[i].f,2)}' +' d$^{-1}$\n$\\frac{\chi^2}{\mathrm{red}} = $' + str(np.round(chisq/df, 2))
        string = f'F{i+1}: {np.round(modes[i].f,2)}'+' d$^{-1}$'
        a.text(0.775, a.set_ylim()[0]+ 0.1*(a.set_ylim()[1]-a.set_ylim()[0]), string, fontsize = 'small')

    # x = np.array(modes[0].phase_bins)
    # y1 = np.array(modes[0].phase_amps)
    # y2 = np.array(modes[2].phase_amps)
    # y3 = np.array(modes[1].phase_amps)
    # y = (y1/np.mean(y1)+ y2/np.mean(y2))/2
    # sds1 = np.array(modes[0].phase_amps_)
    # sds2 = np.array(modes[2].phase_amps_)
    # sds_1 = sds1/np.mean(y1)
    # sds_2 = y1*np.std(y1)/(np.mean(y1)**2*np.sqrt(len(x)))
    # sds_3 = sds2/np.mean(y2)
    # sds_4 = y2*np.std(y2)/(np.mean(y2)**2*np.sqrt(len(x)))
    # sds_5 = np.sqrt((y1/np.mean(y1)- y)**2 + (y2/np.mean(y2)- y)**2)
    # sds = np.sqrt(sds_1**2 + sds_2**2 + sds_3**2 + sds_4**2+ sds_5**2)/2
    # #sds = np.sqrt(sds_1**2 + sds_3**2 )/2
    # ax5.errorbar(x, y, yerr = sds,  ls='', marker='o', color= 'k', ms = 5, capsize = 3)
    #
    # popt, pcov = curve_fit(gaus, x, y, sigma=sds, absolute_sigma=True, p0=[1, 0.3, 0.05, 0.002])
    # xda = np.linspace(0,1, 500)
    # ax5.plot(xda, gaus(xda, *popt), 'r-')
    # Nexp = gaus(x, *popt)
    # r = np.array(y) - Nexp
    # chisq = np.sum((r / sds) ** 2)
    # df = len(x) - 4
    # string = '$\\frac{\chi^2}{\mathrm{red}} = $' + str(np.round(chisq/df, 2))
    # ax5.text(0.825, ax5.set_ylim()[0]+ 0.1*(ax5.set_ylim()[1]-ax5.set_ylim()[0]), string)
    #
    # ax5.set_ylabel('Amplitude / Mean\n F1 + F3')
    # ax5.set_xticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    plt.savefig(dir + '/Figure6_Binary_phase_dependent_amplitudes.pdf', bbox_inches=0)



## Table Frequency list#
def sig_digit(num):
    s = '{:.10f}'.format(num)
    for i in range(len(s)):
        val = s[i]
        if val =='.':
            place = i
    for i in range(len(s)):
        val = s[i]
        try:
            if int(val) > 2:
                return i-place, int(np.round(float(s[i:i+2])/10,0))
            elif int(val) > 0:
                return i+1 - place, int(np.round(float(s[i:i+3])/10,0))
        except:
            continue

class frequency():
    def __init__(self, f, ferr, amp, amperr, phase, phaseerr, snr, key, star):
        self.f = f
        self.ferr = ferr
        self.amp = amp
        self.amperr = amperr
        self.phase = phase
        self.phaseerr = phaseerr
        self.snr = snr
        self.combination = ''
        self.key = key
        self.num = int(key[1:])
        self.star = star

class frequency2():
    def __init__(self, f, ferr, amp, amperr, phase, phaseerr, snr, key):
        self.f = f
        self.ferr = ferr
        self.amp = amp
        self.amperr = amperr
        self.phase = phase
        self.phaseerr = phaseerr
        self.snr = snr
        self.combination = ''
        self.key = key
        self.num = int(key[1:])

    def find_combination(self, dic, a, b):
        rl = 0.011865383643105324
        for key1 in range(1, self.num):
            if dic[f'F{key1}'].f == self.f:
                continue
            f1 = dic[f'F{key1}'].f
            for key2 in range(1, self.num):
                f2 = dic[f'F{key2}'].f
                if f2 == self.f:
                    continue
                else:
                    dif = a*f2 + b*f1 -self.f
                    if abs(dif)<= rl/2:
                        self.combination = self.combination + f'{"%+d" % (a)} F{key2} {"%+d" % (b)} F{key1};'

    def print(self):
        print(self.key +':    {:.3f}    {:.6f}$    {:.3f}    {:.3f}'.format(self.f, self.amp, self.phase, self.snr))
        print('Found Combinations:', self.combination)

def comb_frequencies(r):
    r = r.sort_values(by=['Amplitude'], ascending=False)
    l = {}
    num = 1
    for f, ferr, amp, amperr, phase, phaseerr, snr in zip(r['Frequency'], r['Frequency_error'], r['Amplitude'],
                                                     r['Amplitude_error'], r['Phase'], r['Phase_error'],r['SNR']):
        if amp > 0.000000:
            l[f'F{num}'] = frequency2(f, ferr, amp, amperr, phase, phaseerr, snr, f'F{num}')
            num+=1
    for i in range(2, len(l)+1):
        for (a, b) in [(1,1), (1, -1), (-1, 1)]:
                l[f'F{i}'].find_combination(l, a, b)
        for (a, b) in [(2,1), (2, -1), (1, 2), (-1, 2), (-2, 1), (1,-2), (2,2), (2, -2)]:
                l[f'F{i}'].find_combination(l, a, b)
        for (a, b) in [(3,0), (4,0)]:
                l[f'F{i}'].find_combination(l, a, b)
    return l



def comb_frequencies_pda(r):
    r = r.sort_values(by=['Amplitude'], ascending=False)
    l = {}
    num = 1
    for f, ferr, amp, amperr, phase, phaseerr, snr, star in zip(r['Frequency'], r['Frequency_error'], r['Amplitude'],
                                                     r['Amplitude_error'], r['Phase'], r['Phase_error'],r['SNR'] , r['Component']):
        if amp > 0.000000:
            l[f'F{num}'] = frequency(f, ferr, amp, amperr, phase, phaseerr, snr, f'F{num}',star)
            num+=1
    for i in range(2, len(l)+1):
        for (a, b) in [(1,1), (1, -1), (-1, 1)]:
                l[f'F{i}'].find_combination(l, a, b)
        for (a, b) in [(2,1), (2, -1), (1, 2), (-1, 2), (-2, 1), (1,-2), (2,2), (2, -2)]:
                l[f'F{i}'].find_combination(l, a, b)
        for (a, b) in [(3,0), (4,0)]:
                l[f'F{i}'].find_combination(l, a, b)
    return l



#### table sin ###
if False or do_all:
    import mini_smurfs as ms
    r = ms.load_result('minismurfs/result.csv')
    r = r.sort_values(by=['Amplitude'], ascending=False)
    l = comb_frequencies(r)

    period = 1.66987725

    forb = 1/period
    forb_err = 0.00000005/period**2
    table = []
    total_count = 0
    above_count = 0
    timebase = 1/0.011865383643105324
    amp_err = np.sqrt(2/55309)*0.000165
    f_err =np.sqrt(6/55309)*0.000165/timebase/np.pi
    for i in range(1, len(l)+1):
        mode = l[f'F{i}']
        total_count += 1
        if mode.amp > 0.000000:
            above_count += 1
            line = []
            line.append(mode.key + ' = ' + mode.combination.split(';')[0])
            dig, err = sig_digit(np.sqrt(mode.ferr**2+(f_err/mode.amp)**2))
            string = '${:.' + str(dig) + 'f}({:d})$'
            line.append(string.format(mode.f, err))
            dig, err = sig_digit(np.sqrt(mode.amperr**2+amp_err**2))
            string = r'${:' + str(dig) +'d}({:d})$'
            line.append(string.format(int(1000000*mode.amp), err))
            dig, err = sig_digit(np.sqrt(mode.phaseerr**2+(amp_err/mode.amp)**2 ))
            string = r'${:.' + str(dig) + 'f}({:d})$'
            line.append(string.format(mode.phase, err))
            # for dominic
            dif = np.min([mode.f%forb, forb - mode.f%forb])
            dig, err = sig_digit(np.sqrt(mode.ferr**2+(f_err/mode.amp)**2+ forb_err**2))
            print(dig, err)
            string = r'${:.' + str(dig) + 'f}({:d})$'
            line.append(string.format(dif, err))
            table.append(line)

    print(tabulate(table, tablefmt="latex_raw"))
    # print(len(r))
    # print(above_count)


### Table pda ###
if False or do_all:
    import mini_smurfs as ms
    r = ms.load_result('minismurfs/result_pda_convergence_5_times.csv')
    r = r.sort_values(by=['Amplitude'], ascending=False)
    l = comb_frequencies_pda(r)

    table = []
    total_count = 0
    above_count = 0
    timebase = 1/0.011865383643105324
    amp_err = np.sqrt(2/55309)*0.000165
    f_err =np.sqrt(6/55309)*0.000165/timebase/np.pi
    prim= 0
    secon =0
    for i in range(1, len(l)+1):
        mode = l[f'F{i}']
        total_count += 1
        if mode.amp > 0.000000:
            above_count += 1
            line = []
            line.append(mode.key + ' = ' + mode.combination.split(';')[0])
            dig, err = sig_digit(np.sqrt(mode.ferr**2+(f_err/mode.amp)**2))
            string = '${:.' + str(dig) + 'f}({:d})$'
            line.append(string.format(mode.f, err))
            dig, err = sig_digit(np.sqrt(mode.amperr**2+amp_err**2))
            string = r'${:' + str(dig) +'d}({:d})$'
            line.append(string.format(int(1000000*mode.amp), err))
            dig, err = sig_digit(np.sqrt(mode.phaseerr**2+(amp_err/mode.amp)**2))
            string = r'${:.' + str(dig) + 'f}({:d})$'
            line.append(string.format(mode.phase, err))
            if mode.star == "Primary":
                string = "P"
                prim += 1
            else:
                string = 'S'
                secon += 1
            line.append(string)
            table.append(line)

    print(tabulate(table, tablefmt="latex_raw"))
    print(len(table))
    print(len(r['Amplitude']))
    print(prim)
    print(secon)

#### Figure:7: compare bics aics #####

def pda_flux(time, amp, frequency, phase, phase0, sigma, defect):
    period = 1.66987725
    phases = time%period/period
    return amp*(1- defect*np.exp(-(phases - phase0) ** 2 / (2 * sigma ** 2)))*np.sin(2. * np.pi * (frequency * time + phase))


def sin_multiple(x, *params):
    y = np.zeros(len(x))
    for i in range(0, len(params), 3):
        y += sin(x, params[i], params[i + 1], params[i + 2])
    return y

def sin(x, amp, f, phase):
    return amp * np.sin(2. * np.pi * (f * x + phase))

if False or do_all:
    import mini_smurfs as ms
    r = ms.load_result('minismurfs/result.csv')
    r = r.sort_values(by=['Amplitude'], ascending=False)
    pda = ms.load_result('minismurfs/result_pda_convergence_5_times.csv')
    pda = pda.sort_values(by=['Amplitude'], ascending=False)
    data = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
    time = data[0]
    flux = data[1]
    n = len(time)
    fs = np.array(r['Frequency'])
    amps = np.array(r['Amplitude'])
    phis =  np.array(r['Phase'])
    fs_pda = np.array(pda['Frequency'])
    amps_pda = np.array(pda['Amplitude'])
    phis_pda =  np.array(pda['Phase'])
    phase0_pda = np.array(pda['phase0'])
    defects_pda = np.array(pda['defects'])
    sigma_pda = np.array(pda['sigma'])
    component_pda = np.array(pda['Component'])
    p = lk.LightCurve(time, flux).to_periodogram(maximum_frequency=40, minimum_frequency=0.1)
    pr_start =integ.simps(p.power, p.frequency)
    var_start = np.var(flux)
    bic_sin = []
    aic_sin = []
    ks_sin = []
    pr_sin = []
    vr_sin = []
    bic_pda = []
    aic_pda = []
    ks_pda = []
    pr_pda = []
    vr_pda = []
    for i in range(1, len(fs)+1):

        print(i)
        arr = []
        k = 3*i
        ks_sin.append(k)
        for j in range(0, i):
            arr.append(amps[j])
            arr.append(fs[j])
            arr.append(phis[j])
        mod = sin_multiple(time, *arr)
        RSS = n*np.log(np.sum((flux-mod)**2)/n)
        bic_sin.append(RSS + k*np.log(n))
        aic_sin.append(RSS + 2*k + n)
        p = lk.LightCurve(time, flux-mod).to_periodogram(maximum_frequency=40, minimum_frequency=0.1)
        pr_sin.append(1-integ.simps(p.power, p.frequency) / pr_start)

        var = np.var(flux-mod)
        vr_sin.append(1 - var / var_start)

    fu,au = plt.subplots(2,1)

    au[0].plot(time, flux-mod, 'ko', ms = .75)
    fun = np.random.normal(0, 0.000165, len(time))
    au[0].plot(time, fun)
    p = lk.LightCurve(time, flux-mod).to_periodogram(maximum_frequency=40, minimum_frequency=0.1)
    p2 = lk.LightCurve(time, fun + 0.0006*np.sin(time*200)).to_periodogram(maximum_frequency=40, minimum_frequency=0.1)
    au[0].set_title(f'{np.round(np.std(flux-mod)*1000000, 1)}         {np.round(np.std(fun)*1000000, 1)}')

    au[1].plot(p.frequency, p.power)
    au[1].plot(p2.frequency, p2.power)
    au[1].set_title(f'{np.round(np.mean(p.power)*1000000, 1)}         {np.round(np.mean(p2.power)*1000000, 1)}')
    #plt.show(fu)


    secondary = 0
    primary = 0
    mod = np.zeros(len(flux))
    for i in range(0, len(fs_pda)):
        print(i)
        arr = []
        if secondary == 0 and component_pda[i] == 'Secondary':
            secondary = 1
            print('First secondary: ', i)
        if primary == 0 and component_pda[i] == 'Primary':
            primary = 1
            print('First primary: ', i)
        k = 3*(i+1) +3*secondary + 3* primary
        ks_pda.append(k)
        mod = mod + pda_flux(time, amps_pda[i], fs_pda[i], phis_pda[i], phase0_pda[i], sigma_pda[i], defects_pda[i])
        RSS = n*np.log(np.sum((flux-mod)**2)/n)
        bic_pda.append(RSS + k*np.log(n))
        aic_pda.append(RSS + 2*k + n)
        p = lk.LightCurve(time, flux-mod).to_periodogram(maximum_frequency=40, minimum_frequency=0.1)
        pr_pda.append(1-integ.simps(p.power, p.frequency) / pr_start)
        var = np.var(flux-mod)
        vr_pda.append(1 - var / var_start)




    fig, ax = plt.subplots(3,1,figsize = figsize(2), dpi = dpi)
    ms = 4
    mp = 5
    ax[0].plot(ks_sin, 100*(1-(bic_sin[0] - np.array(bic_sin))/abs(bic_sin[0])), 'ko', ms = ms, mfc = 'none')
    ax[0].plot(ks_sin, 100*(1-(aic_sin[0] - np.array(aic_sin))/abs(aic_sin[0])), 'ro', ms = ms)
    ax[0].plot(ks_pda, 100*(1-(bic_pda[0] - np.array(bic_pda))/abs(bic_pda[0])), 'kP', ms = mp, mfc = 'none')
    ax[0].plot(ks_pda, 100*(1-(aic_pda[0] - np.array(aic_pda))/abs(aic_pda[0])), 'rP', ms = mp)
    ax[1].plot(ks_sin, 100*(np.array(vr_sin)), 'ko', ms = ms)
    ax[2].plot(ks_sin, 100*(np.array(pr_sin)), 'ko', ms = ms)
    ax[1].plot(ks_pda, 100*(np.array(vr_pda)), 'kP', ms = mp, mfc = 'none')
    ax[2].plot(ks_pda, 100*(np.array(pr_pda)), 'kP', ms = mp, mfc = 'none')
    ax[0].xaxis.set_visible(False)
    ax[1].xaxis.set_visible(False)
    ax[0].set_ylabel('BIC & AIC (%)' )
    ax[1].set_ylabel('VR (%)' )
    ax[2].set_ylabel('PR (%)' )
    ax[2].set_xlabel('number of free parameters')
    ax[0].plot([45*3], [81.75], 'k^')
    ax[0].plot([45*3], [100], 'kv')
    ax[0].plot([46*3+6], [100], 'kv', mfc = 'none')
    ax[0].plot([46*3+6], [81.75], 'k^', mfc = 'none')
    ax[1].plot([45*3], [15], 'k^')
    ax[1].plot([45*3], [90], 'kv')
    ax[1].plot([46*3+6], [90], 'kv', mfc = 'none')
    ax[1].plot([46*3+6], [15], 'k^', mfc = 'none')
    ax[2].plot([45*3], [4], 'k^')
    ax[2].plot([45*3], [50], 'kv')
    ax[2].plot([46*3+6], [50], 'kv', mfc = 'none')
    ax[2].plot([46*3+6], [4], 'k^', mfc = 'none')

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    import matplotlib
    # ax2zoom = zoomed_inset_axes(ax[1], 5, loc=4, axes_kwargs={'aspect': 10, })  # zoom-factor: 2.5, location: upper-left
    ax1zoom = inset_axes(ax[1], width='50%', height='50%', loc=4)
    ax2zoom = inset_axes(ax[2], width='50%', height='50%', loc=4)
    ax0zoom = inset_axes(ax[0], width='50%', height='50%', loc=1)
    ax0zoom.plot(ks_sin, 100*(1-(bic_sin[0] - np.array(bic_sin))/abs(bic_sin[0])), 'ko', ms = 1.5*ms, mfc = 'none')
    ax0zoom.plot(ks_sin, 100*(1-(aic_sin[0] - np.array(aic_sin))/abs(aic_sin[0])), 'ro', ms = 1.5*ms)
    ax0zoom.plot(ks_pda, 100*(1-(bic_pda[0] - np.array(bic_pda))/abs(bic_pda[0])), 'kP', ms = 1.5*mp, mfc = 'none')
    ax0zoom.plot(ks_pda, 100*(1-(aic_pda[0] - np.array(aic_pda))/abs(aic_pda[0])), 'rP', ms = 1.5*mp)
    ax1zoom.plot(ks_sin, 100*(np.array(vr_sin)), 'ko', ms = 1.5*ms)
    ax1zoom.plot(ks_pda, 100*(np.array(vr_pda)), 'kP', ms = 1.5*mp, mfc = 'none')
    ax2zoom.plot(ks_sin, 100*(np.array(pr_sin)), 'ko', ms = 1.5*ms)
    ax2zoom.plot(ks_pda, 100*(np.array(pr_pda)), 'kP', ms = 1.5*mp, mfc = 'none')
    ax1zoom.set_xlim(215, 330)
    ax1zoom.set_ylim(85.75, 90)
    ax2zoom.set_xlim(215, 330)
    ax2zoom.set_ylim(44, 50.5)
    ax0zoom.set_xlim(215, 330)
    ax0zoom.set_ylim(81.2, 85.75)
    for a in [ax0zoom, ax1zoom, ax2zoom]:
        a.xaxis.set_visible('False')
        a.yaxis.set_visible('False')
        a.set_xticks([])
        a.set_yticks([])

    mark_inset(ax[2], ax2zoom, loc1=2, loc2=1, fc="none", ec="0.5")
    mark_inset(ax[0], ax0zoom, loc1=3, loc2=4, fc="none", ec="0.5")
    mark_inset(ax[1], ax1zoom, loc1=2, loc2=1, fc="none", ec="0.5")



    plt.tight_layout(h_pad=0, w_pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(dir + '/Figure7test_statistics.pdf', bbox_inches=0)
    # plt.show()


###Figure:8: perturbed modes ####
class mode:
    def __init__(self, f, amp, phi, star = 'none', name = ''):
        self.f = f
        self.amp = amp
        self.phi = phi
        self.star = star
        self.name = name

def exp(f, i, a):
    period = 1.66987725
    forb = 1 / period
    l = []
    for j in i:
        l.append(f+j*forb)
    l = np.array(l)
    a.plot(l%forb, l, 'ks', ms = 5, mfc = 'none')


if False or do_all:
    import mini_smurfs as ms
    r = ms.load_result('minismurfs/result.csv')
    r = r.sort_values(by=['Amplitude'], ascending=False)
    pda = ms.load_result('minismurfs/result_pda_convergence_5_times.csv')
    pda = pda.sort_values(by=['Amplitude'], ascending=False)
    # data = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
    # time = data[0]
    # flux = data[1]
    fs = np.array(r['Frequency'])
    amps = np.array(r['Amplitude'])
    phis =  np.array(r['Phase'])
    fs_pda = np.array(pda['Frequency'])
    amps_pda = np.array(pda['Amplitude'])
    phis_pda =  np.array(pda['Phase'])
    phase0_pda = np.array(pda['phase0'])
    defects_pda = np.array(pda['defects'])
    sigma_pda = np.array(pda['sigma'])
    component_pda = np.array(pda['Component'])
    sin_a165 = []
    sin_b165 = []
    secon_a165 = []
    secon_b165 = []
    prim_a165 = []
    prim_b165 = []
    for i in range(len(fs)):
        if amps[i] > 0.000165:
            sin_a165.append(mode(fs[i], amps[i], phis[i], name = f'F{i+1}') )
        else:
            sin_b165.append(mode(fs[i], amps[i], phis[i], name = f'F{i+1}') )
    for i in range(len(fs_pda)):
        if amps_pda[i] > 0.000165:
            if component_pda[i] == 'Primary':
                prim_a165.append(mode(fs_pda[i], amps_pda[i], phis_pda[i], component_pda[i], name = f'f{i+1}'))
            else:
                secon_a165.append(mode(fs_pda[i], amps_pda[i], phis_pda[i], component_pda[i], name = f'f{i+1}'))
        else:
            if component_pda[i] == 'Primary':
                prim_b165.append(mode(fs_pda[i], amps_pda[i], phis_pda[i], component_pda[i], name = f'f{i+1}'))
            else:
                secon_b165.append(mode(fs_pda[i], amps_pda[i], phis_pda[i], component_pda[i], name = f'f{i+1}'))

    period = 1.66987725
    forb = 1/period
    ms = 6


    fig = plt.figure(constrained_layout=True,  figsize = figsize(1.7), dpi = dpi)
    gs = fig.add_gridspec(4, 1)
    axtop = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1:, 0])
    axtop.set_xticks([])
    axtop.set_yticks([])
    perturbeds = [[0], [-1, 1], [-1, 0, 1], [-2, 0, 2], [-2, -1, 1, 2], [-2, -1, 0, 1, 2], [-3, -1, 1, 3], [-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 1, 2, 3]]
    strings1 = ['$l=0$', '$l=1$', '$l=1$', '$l=2$', '$l=2$', '$l=2$', '$l=3$', '$l=3$', '$l=3$']
    strings2 = ['', '$|\\tilde{m}|=0$', '$|\\tilde{m}|=1$', '$|\\tilde{m}|=0$', '$|\\tilde{m}|=1$', '$|\\tilde{m}|=2$', '$|\\tilde{m}|=0$', '$|\\tilde{m}|=1,3$', '$|\\tilde{m}|=2$']
    step = forb/(len(perturbeds)+1)
    num = 0
    for p,s1,s2 in zip(perturbeds, strings1, strings2):
        exp(step+num*step, p, axtop)
        axtop.text((step+num*step)%forb, 4.5, s1, ha = 'center', fontsize = 'x-small')
        axtop.text((step+num*step)%forb, 3.25, s2, ha = 'center', fontsize = 'x-small')
        num += 1

    ax.set_ylim([8, 40])
    axtop.set_ylim(-3.5, 6.7)




    import matplotlib.patches as mpatches

    import matplotlib.legend_handler


    # fig, ax = plt.subplots(1,figsize = figsize(1.25), dpi = dpi)
    fs = np.array([x.f for x in sin_a165])
    sin1, = ax.plot(fs%forb, fs, 'ko', ms = 1.2*ms, label = '$M_1$')
    fs = np.array([x.f for x in sin_b165])
    sin2, = ax.plot(fs%forb, fs, 'ko', ms = 1.2*ms, mfc = 'none')
    fs = np.array([x.f for x in prim_a165])
    prim1, = ax.plot(fs%forb, fs, 'gX', ms = ms, label = '$M_2$ primary')
    fs = np.array([x.f for x in prim_b165])
    prim2, = ax.plot(fs%forb, fs, 'gX', ms = ms, mfc = 'none')
    fs = np.array([x.f for x in secon_a165])
    sec1, = ax.plot(fs%forb, fs, 'mX', ms = ms, label = '$M_2$ secondary')
    fs = np.array([x.f for x in secon_b165])
    sec2, = ax.plot(fs%forb, fs, 'mX', ms = ms, mfc = 'none')

    handles = [(sin1, sin2), (prim1, prim2), (sec1, sec2)]
    _, labels = ax.get_legend_handles_labels()

    ax.legend(handles=handles, labels=labels, loc='upper center', ncol = 3,
              handler_map={tuple: matplotlib.legend_handler.HandlerTuple(None)})

    ax.set_xlabel('frequency modulo $f_\mathrm{orb}$  (${\\rm d}^{-1}$)')
    ax.set_ylabel('frequency (${\\rm d}^{-1}$)')

    ### for analysis ###
    # for f in sin_a165:
    #     ax.text(f.f%forb+ 0.001, f.f, f.name, fontsize = 'xx-small')
    # for f in sin_b165:
    #     ax.text(f.f%forb+ 0.001, f.f, f.name, fontsize = 'xx-small')
    # for f in prim_a165:
    #     ax.text(f.f%forb- 0.001, f.f, f.name, fontsize = 'xx-small')
    # for f in prim_b165:
    #     ax.text(f.f%forb- 0.001, f.f, f.name, fontsize = 'xx-small')
    # for f in secon_a165:
    #     ax.text(f.f%forb- 0.001, f.f, f.name, fontsize = 'xx-small')
    # for f in secon_b165:
    #     ax.text(f.f%forb- 0.001, f.f, f.name, fontsize = 'xx-small')

    plt.tight_layout(h_pad=0, w_pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(dir + '/Figure8_perturbed_modes.pdf', bbox_inches=0)
    plt.show()


if False or do_all:
    import mini_smurfs as ms
    r = ms.load_result('minismurfs/result.csv')
    r = r.sort_values(by=['Amplitude'], ascending=False)
    fs = np.array(r['Frequency'])
    amps = np.array(r['Amplitude'])
    phis =  np.array(r['Phase'])
    sin_a165 = []
    sin_b165 = []
    for i in range(len(fs)):
        if amps[i] > 0.000165:
            sin_a165.append(mode(fs[i], amps[i], phis[i], name = f'F{i+1}') )
        else:
            sin_b165.append(mode(fs[i], amps[i], phis[i], name = f'F{i+1}') )
    period = 1.66987725
    forb = 1/period
    ms = 6


    fig = plt.figure(constrained_layout=True,  figsize = figsize(1.7), dpi = dpi)
    gs = fig.add_gridspec(4, 1)
    axtop = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1:, 0])
    axtop.set_xticks([])
    axtop.set_yticks([])
    perturbeds = [[0], [-1, 1], [-1, 0, 1], [-2, 0, 2], [-2, -1, 1, 2], [-2, -1, 0, 1, 2], [-3, -1, 1, 3], [-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 1, 2, 3]]
    strings1 = ['$l=0$', '$l=1$', '$l=1$', '$l=2$', '$l=2$', '$l=2$', '$l=3$', '$l=3$', '$l=3$']
    strings2 = ['', '$|\\tilde{m}|=0$', '$|\\tilde{m}|=1$', '$|\\tilde{m}|=0$', '$|\\tilde{m}|=1$', '$|\\tilde{m}|=2$', '$|\\tilde{m}|=0$', '$|\\tilde{m}|=1,3$', '$|\\tilde{m}|=2$']
    step = forb/(len(perturbeds)+1)
    num = 0
    for p,s1,s2 in zip(perturbeds, strings1, strings2):
        exp(step+num*step, p, axtop)
        axtop.text((step+num*step)%forb, 4.5, s1, ha = 'center', fontsize = 'x-small')
        axtop.text((step+num*step)%forb, 3.25, s2, ha = 'center', fontsize = 'x-small')
        num += 1

    ax.set_ylim([8, 35])
    axtop.set_ylim(-2.9, 6.1)




    import matplotlib.patches as mpatches

    import matplotlib.legend_handler


    # fig, ax = plt.subplots(1,figsize = figsize(1.25), dpi = dpi)
    fs = np.array([x.f for x in sin_a165])
    sin1, = ax.plot(fs%forb, fs, 'ko', ms = 1.2*ms, label = '$M_1$')
    fs = np.array([x.f for x in sin_b165])
    sin2, = ax.plot(fs%forb, fs, 'ko', ms = 1.2*ms, mfc = 'none')


    ax.set_xlabel('frequency modulo $f_\mathrm{orb}$  (${\\rm d}^{-1}$)')
    ax.set_ylabel('frequency (${\\rm d}^{-1}$)')

    ### for analysis ###
    for f in sin_a165:
        ax.text(f.f%forb+ 0.001, f.f, f.name, fontsize = 'xx-small')
    for f in sin_b165:
        ax.text(f.f%forb+ 0.001, f.f, f.name, fontsize = 'xx-small')
    # for f in prim_a165:
    #     ax.text(f.f%forb- 0.001, f.f, f.name, fontsize = 'xx-small')
    # for f in prim_b165:
    #     ax.text(f.f%forb- 0.001, f.f, f.name, fontsize = 'xx-small')
    # for f in secon_a165:
    #     ax.text(f.f%forb- 0.001, f.f, f.name, fontsize = 'xx-small')
    # for f in secon_b165:
    #     ax.text(f.f%forb- 0.001, f.f, f.name, fontsize = 'xx-small')

    plt.tight_layout(h_pad=0, w_pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(dir + '/Figure8_perturbed_modes_without_M2.pdf', bbox_inches=0)
    plt.show()





### Appendix ###

if False or do_all:
    import phase_dependent_amplitude as pda
    fig1, ax1 = pda.plot_amplitude_statistics(figsize1, dpi)
    plt.tight_layout()
    fig1.savefig(dir + '/App2_peak_amplitudes.pdf', bbox_inches=0)
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    fig2, ax2 = plt.subplots(2,1, figsize = figsize(1.5), dpi = dpi)
    pda_f = pda_flux(time, 0.001500, 12, 0.5, 0.3, 0.05, 0.7)+ np.random.normal(0, 0.000165, len(time))
    ax2[0].plot(time, pda_f, 'ko', ms = 0.75)
    ax2[0].set_xlim([1599, 1608])
    pdg = lk.LightCurve(time, pda_f).to_periodogram()
    ax2[1].plot(pdg.frequency, pdg.power, 'k-')
    ax2[1].set_xlim([7, 17])
    ax2[0].set_xlabel('Time - 2457000 [BTJD days]')
    ax2[0].set_ylabel('relative flux')
    ax2[1].set_xlabel('frequency $d^{-1}$')
    ax2[1].set_ylabel('amplitude (relative flux)')
    period = 1.66987725
    forb = 1/period
    xs = np.array([12+i*forb for i in [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]])
    ax2[1].plot(xs, 0.000500 * np.ones(len(xs)), 'kv', ms = 4)
    for x, i  in zip(xs, [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]):
        ax2[1].text(x, 0.000600, i, fontsize = 'x-small', ha = 'center')
    ax2[1].errorbar([12-forb/2, 12+forb/2], [0.001500, 0.001500], xerr = forb/2, capsize = 5, ls='',  color = 'k')
    ax2[1].text(12-forb/2, 0.001380, r'$f_\mathrm{orb}$', fontsize = 'x-small', ha = 'center')
    ax2[1].text(12+forb/2, 0.001380, r'$f_\mathrm{orb}$', fontsize = 'x-small', ha = 'center')
    plt.tight_layout()
    fig2.savefig(dir + '/App1_lc_spectra_label.pdf', bbox_inches=0)


### Figure 4: Result of Binary Modelling ###
if False or do_all:
    period = 1/0.5988495842998753
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    data = np.loadtxt('endurance/' + 'Removed_Pulsations_from_first_run_irrad_gravbol/binary_model.txt').T
    time_mod = data[0]
    flux_mod = data[1]


    import mini_smurfs as ms
    r = ms.load_result('minismurfs/result.csv')
    r = r.sort_values(by=['Amplitude'], ascending=False)
    fs = np.array(r['Frequency'])
    amps = np.array(r['Amplitude'])
    phis =  np.array(r['Phase'])
    arr = []
    for j in range(0, len(fs)):
        if abs(fs[j]*period - np.round(fs[j]*period, 0)) < 0.02:
            print(fs[j], fs[j]*period, abs(fs[j]*period - np.round(fs[j]*period, 0)))
            continue
        else:
            arr.append(amps[j])
            arr.append(fs[j])
            arr.append(phis[j])
    mod = sin_multiple(time, *arr)

    se = np.where(np.logical_and(abs(time%period/period - 0.3) >= 0.1, abs(time%period/period - 0.775) >= 0.125))
    resid_lc = lk.LightCurve(time%period/period , flux - flux_mod - mod).to_periodogram()
    resid_lc_oof = lk.LightCurve((time%period/period )[se], flux[se] - flux_mod[se] - mod[se]).to_periodogram()

    phase = time_mod%period/period
    sort = np.argsort(phase)
    fig= plt.figure(constrained_layout=True, figsize = figsize1, dpi = dpi)
    gs = fig.add_gridspec(9, 1)
    ax2 = fig.add_subplot(gs[0:-4, 0])
    ax3 = fig.add_subplot(gs[-4:, 0])
    ax2.plot(time%period, flux - flux_mod - mod, 'ko', ms = 0.75)
    ax2.plot((time%period)[se], (flux - flux_mod - mod)[se], 'ro', ms = 0.75)
    ax3.plot(resid_lc.frequency, resid_lc.power, 'k-')
    ax3.plot(resid_lc_oof.frequency, resid_lc_oof.power, 'r-')
    ax2.set_xticks([])
    ax2.set_xlabel('orbital phase')
    ax3.set_xlabel('frequency 1/Period')
    ax2.set_ylabel('flux')
    ax3.set_ylabel('ampltitude')
    ax3.set_xlim(0,20)
    plt.tight_layout(h_pad=0, w_pad=0)
    #plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(dir + '/App3_tidally_excited.pdf', bbox_inches=0)
    plt.show()

#### Mail #####

if False or do_mail:
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



def load_weird_csv(name):
    data = np.loadtxt(f'endurance/alecian_data/{name}', dtype=np.str, delimiter=';', ).T
    result1 = []
    for s in data[0]:
        result1.append(float(s.strip().replace(',', '.')))
    result2 = []
    for s in data[1]:
        result2.append(float(s.strip().replace(',', '.')))
    return np.array([result1, result2])

def get_lfs_l0(data):
    x_vals = data[0]
    return np.mean(x_vals[1:]-x_vals[0:-1])/11.57407407

def get_lfs_l1(data):
    x_vals = data[0][1::3]
    return np.mean(x_vals[1:]-x_vals[0:-1])/11.57407407

def load_results(path: str):
    """
    Loads the pandas dataframes from the results file
    :param path: exact path of the results file
    :return: 3 pandas dataframes, settings, statistics and results
    """
    import os
    from pandas import read_csv
    from io import StringIO
    from uncertainties import ufloat_fromstr
    if not os.path.exists(path) or not path.endswith("result.csv"):
        raise IOError("You need to provide the exact path to the results file")

    with open(path, 'r') as f:
        content = f.read()

    settings = read_csv(StringIO(content.split("\n\n")[0]), skiprows=1)
    statistics = read_csv(StringIO(content.split("\n\n")[1]), skiprows=2)
    results = read_csv(StringIO(content.split("\n\n")[2]), skiprows=2,
                       converters={'frequency': ufloat_fromstr,
                                   'amp': ufloat_fromstr,
                                   'phase': ufloat_fromstr})
    return settings, statistics, results


if False or do_mail:
    l0_modes_secondary = load_weird_csv('l0_modes_secondary.csv')
    l0_modes_primary = load_weird_csv('l0_modes_primary.csv')
    l1_modes_secondary = load_weird_csv('l1_modes_secondary.csv')
    l1_modes_primary = load_weird_csv('l1_modes_primary.csv')
    lfs_l0_primary =  get_lfs_l0(l0_modes_primary)
    lfs_l0_secondary =  get_lfs_l0(l0_modes_secondary)
    lfs_l1_primary =  get_lfs_l1(l1_modes_primary)
    lfs_l1_secondary =  get_lfs_l1(l1_modes_secondary)
    data = np.loadtxt('endurance/' + 'Removed_Binary_plus_savgol_from_original.txt').T
    time = data[0]
    flux = data[1]
    rl = 1/(time[-1]-time[0])
    settings, statistics, results = load_results('endurance/Removed_Binary_plus_savgol_from_original/data/result.csv')
    period = 1.66987725
    fs = results['frequency'].values
    amps = results['amp'].values
    snrs = results['snr'].values
    phis = results['phase'].values
    freqs = []
    amp = []
    freqs_b = []
    amp_b = []
    freqs_la = []
    amp_la = []
    for i in range(len(fs)):
        if snrs[i] > 4 and not  ((2*rl > fs[i].n % (1/period)) or (fs[i].n % (1/period) >(1/period)-2*rl)) and amps[i].n > 0.0002:
            freqs.append(fs[i].n)
            amp.append(amps[i].n)
        elif snrs[i] > 4 and not  ((2*rl > fs[i].n % (1/period)) or (fs[i].n % (1/period) >(1/period)-2*rl)) :
            freqs_la.append(fs[i].n)
            amp_la.append(amps[i].n)
        elif snrs[i] > 4:
            freqs_b.append(fs[i].n)
            amp_b.append(amps[i].n)

    #plt.plot(np.array(freqs)%(1/period), freqs, 'ko')
    lc = lk.LightCurve(time, flux).to_periodogram()
    plt.plot(lc.frequency, lc.power, 'k-')
    for f in range(len(freqs)):
        plt.plot([freqs[f], freqs[f]], [0, amp[f]], 'r-')
    for f in range(len(freqs_b)):
        plt.plot([freqs_b[f], freqs_b[f]], [0, amp_b[f]], 'r--')
    plt.show()
    plt.close()
    #weird = lk.LightCurve(sorted(freqs), np.ones(len(freqs))).to_periodogram()
    #plt.plot(weird.frequency, weird.power, 'k-')
    print(lfs_l0_primary, lfs_l0_secondary, lfs_l1_primary, lfs_l1_secondary)
    count = 0
    f_sec = [11.074, 11.624, 12.7956]
    from tqdm import tqdm
    for shit in tqdm(np.linspace(3.5, 4.5, 500)):
        fig, ax = plt.subplots()
        ax.plot(np.array(freqs)%shit, freqs, 'ro', markersize = 10,label = 'Amp >= 0.0002')
        ax.plot(np.array(f_sec)%shit, f_sec, 'b', markersize = 8,label = 'secondary')
        lsfhj = ax.scatter(np.array(freqs_la)%shit, freqs_la, marker = 'o', color = 'k', s= 10,label = 'Amp < 0.0002')
        ax.set_xlabel('Frequency mod ' + str(shit)[:5] + '$d^{-1}$')
        ax.set_ylabel('Frequency $d^{-1}$')
        plt.tight_layout()
        lsfhj.set_facecolor('none')
        ax.legend(loc = 3)
        #fig.canvas.draw()
        #plt.pause(0.2)
        #ax.clear()
        plt.savefig(dir + '/video/%03d.png' %count, bbox_inches=0)
        count += 1
        plt.close()

    plt.show()

if False:
    R1 = 2.15*696340
    R2 = 2.36*696340
    period = 1.66988
    period = 2.598


    vsin1 = 66
    vsin2 = 72

    i = 83.4
    ierr = 0.3

    u1 = 2*np.pi * R1
    u2 = 2*np.pi * R2
    u1err = 2*np.pi * 0.06
    u2err = 2*np.pi * 0.06
    per1 = u1/vsin1/3600/24/np.sin(i/360*2*np.pi)
    per2 = u2/vsin2/3600/24/np.sin(i/360*2*np.pi)

    print(per1, per2)
    print(u1/period/3600/24/np.sin(i/360*2*np.pi), np.sqrt((u1err/period/3600/24/np.sin(i/360*2*np.pi))**2 + (ierr*u1/period/3600/24/np.cos(i/360*2*np.pi)/360*2*np.pi)**2))
    print(u2/period/3600/24/np.sin(i/360*2*np.pi), np.sqrt((u2err/period/3600/24/np.sin(i/360*2*np.pi))**2 + (ierr*u2/period/3600/24/np.cos(i/360*2*np.pi)/360*2*np.pi)**2))

if False:
    G = 6.67430*10**-11
    c = 299792458
    msun = 1.98847*10**30

    dtos = 24*3600
    m1 = 1.89*msun
    m2 = 1.87*msun
    i = 83.4
    sini = np.sin(i/360*2*np.pi)
    Porb = 1.66988*dtos
    P_puls = 1/12*dtos
    alpha = (2*np.pi*G)**(1/3)/c*Porb**(2/3)/P_puls*m2*sini/(m1+m2)**(2/3)
    print(alpha)

    q = m2/m1

    alpha2 = (2*np.pi*G*msun)**(1/3)/c*(m1/msun)**(1/3)*q*(1+q)**(-2/3)*Porb**(2/3)/P_puls*sini
    print(alpha2)

if False or do_mail:
    freq = 1/1.66987725
    period = 1.66987725
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    data2 = np.loadtxt('endurance/' + 'Removed_Binary_plus_savgol_from_original.txt').T
    time2 = data2[0]
    flux2 = data2[1]
    data3 = np.loadtxt('endurance/' + 'Removed_Pulsations_from_first_run_new_teff50/binary_model.txt').T
    time3 = data3[0]
    flux3 = data3[1]
    ind = np.where(np.logical_and(abs(time % period / period - 0.3) >= 0.1, abs(time % period / period - 0.8) >= 0.1))
    pdg_orig = lk.LightCurve(time, flux).to_periodogram()
    pdg_it1 = lk.LightCurve(time2, flux2).to_periodogram()
    pdg_it2 = lk.LightCurve(time3, flux - flux3).to_periodogram()
    pdg_ooe = lk.LightCurve(time2[ind], flux2[ind]).to_periodogram()
    pdg_ooe2 = lk.LightCurve(time2[ind], flux[ind]-flux3[ind]).to_periodogram()
    fig = plt.figure(constrained_layout=True,  figsize = figsize1, dpi = dpi)
    gs = fig.add_gridspec(3, 3)
    ax00 = fig.add_subplot(gs[0, :-1])
    ax01 = fig.add_subplot(gs[1, :-1])
    ax02 = fig.add_subplot(gs[2, :-1])
    ax10 = fig.add_subplot(gs[0, -1:])
    ax11 = fig.add_subplot(gs[1, -1:])
    ax12 = fig.add_subplot(gs[2, -1:])
    ax00.plot(pdg_orig.frequency, pdg_orig.power, 'k-', lw = 0.5)
    ax01.plot(pdg_it1.frequency, pdg_it1.power, 'k-', lw = 0.5)
    ax02.plot(pdg_ooe.frequency, pdg_ooe.power, 'k-', lw = 0.5)
    ax10.plot(time%period/period, flux, 'ko', ms = 0.25)
    ax11.plot(time%period/period, flux2, 'ko', ms = 0.25)
    ax12.plot(time[ind]%period/period, flux2[ind], 'ko', ms = 0.25)
    for a in [ax00, ax01, ax02]:
        a.set_xlabel('frequency $d^{-1}$')
        a.set_ylabel('power')
        a.set_xlim(0, 35)
        for i in range(80):
            a.plot([i*freq, i*freq], [0, a.set_ylim()[1]], 'b--', lw = 0.25)
    for a in [ax10, ax11, ax12]:
        a.set_xlabel('Phase')
        a.set_ylabel('flux')
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig(dir + '/mail2_amplitude_spectra.png', bbox_inches=0)
    plt.show()

if False or do_mail:
    freq = 1/1.66987725
    period = 1.66987725
    data2 = np.loadtxt('endurance/' + 'Removed_Binary_plus_savgol_from_original.txt').T
    time2 = data2[0]
    flux2 = data2[1]
    pdg_it1 = lk.LightCurve(time2, flux2).to_periodogram()
    settings, statistics, results = load_results('endurance/Removed_Binary_plus_savgol_from_original/data/result.csv')
    fs = results['frequency'].values
    amps = results['amp'].values
    snrs = results['snr'].values
    phis = results['phase'].values
    freqs_f1 = []
    amp_f1 = []
    freqs_f2 = []
    amp_f2 = []
    freqs_f3 = []
    amp_f3 = []
    rl = 1/(time2[-1]-time2[0])
    for i in range(len(fs)):
        if snrs[i] > 4 and not (
                (2 * rl > fs[i].n % (1 / period)) or (fs[i].n % (1 / period) > (1 / period) - 2 * rl)):
            if  (rl + 11.0704 % (1 / period)  >= fs[i].n % (1 / period) >= 11.0704 % (1 / period)-rl):
                #print( rl + 11.0704 % (1 / period), fs[i].n % (1 / period),11.0704 % (1 / period)-rl, fs[i].n)
                freqs_f1.append(fs[i].n)
                amp_f1.append(amps[i].n)
            elif (rl + 11.624 % (1 / period)  >= fs[i].n % (1 / period) >= 11.624 % (1 / period)-rl):
                freqs_f2.append(fs[i].n)
                amp_f2.append(amps[i].n)
            elif (rl + 12.7956 % (1 / period)  >= fs[i].n % (1 / period) >= 12.7956 % (1 / period)-rl):
                freqs_f3.append(fs[i].n)
                amp_f3.append(amps[i].n)


    fig , ax= plt.subplots( figsize = figsize1, dpi = dpi)
    ax.plot(pdg_it1.frequency, pdg_it1.power, 'k-', lw = 0.5)
    ax.set_xlabel('frequency $d^{-1}$')
    ax.set_ylabel('power')
    ax.set_xlim(8, 15.5)
    for i in range(80):
        ax.plot([i*freq, i*freq], [0, ax.set_ylim()[1]], 'b--', lw = 0.25)
    for f in range(len(freqs_f1)):
        ax.plot([freqs_f1[f],freqs_f1[f]], [0, amp_f1[f]], 'b-', lw = 1)
    for f in range(len(freqs_f2)):
        ax.plot([freqs_f2[f],freqs_f2[f]], [0, amp_f2[f]], 'r-', lw=1)
    for f in range(len(freqs_f3)):
        ax.plot([freqs_f3[f],freqs_f3[f]], [0, amp_f3[f]], 'g-', lw=1)
    ax.annotate('f2', (11.0704, 0.0016), (11.304, 0.0015), arrowprops=dict(arrowstyle= '-', facecolor='black', lw = 0.75))
    ax.annotate('f1', (11.624, 0.0016), (11.904, 0.0015), arrowprops=dict(arrowstyle= '-', facecolor='black', lw = 0.75))
    ax.annotate('f3', (12.7956, 0.0015), (13.0956, 0.0014), arrowprops=dict(arrowstyle= '-', facecolor='black', lw = 0.75))
    for f in range(len(freqs_f1)):
        if amp_f1[f] < 0.0014:
            num = int(np.around(((freqs_f1[f] - 11.0704) / freq)))
            print(num)
            if num > 0:
                sig = '+'
            else:
                sig = '-'
            ax.annotate(f'f2{sig}{abs(num)}' + '$\Omega$', (freqs_f1[f], 0), (freqs_f1[f] + 0.15, -0.000025),
                        arrowprops=dict(arrowstyle='-', facecolor='black', lw=0.5), fontsize=(5))
    for f in range(len(freqs_f2)):
        if amp_f2[f] < 0.0014:
            num = int(np.around((freqs_f2[f] - 11.624) / freq))
            print(num)
            if num > 0:
                sig = '+'
            else:
                sig = '-'
            ax.annotate(f'f1{sig}{abs(num)}' + '$\Omega$', (freqs_f2[f], amp_f2[f]), (freqs_f2[f] - 0.5, 0.0004),
                        arrowprops=dict(arrowstyle='-', facecolor='black', lw=0.5), fontsize=(5))

    plt.savefig(dir + '/mail2_splitted.png', bbox_inches=0)
    plt.show()

### Timeline ###

def element(l, h, x0, y0, ax, c, txt='', dir = 1, fill = True):
    inset = 0.5
    x = [x0 - l/2, x0 + l/2, x0 + l/2 + dir*inset, x0+l/2, x0-l/2, x0-l/2+dir*inset]
    y = [y0 + h/2, y0 + h/2, y0, y0 -h/2, y0-h/2, y0]
    if fill:
        ax.fill(x, y, fc = c)
        ax.text(x0+dir*inset/2, y0, txt.format(fname), ha = 'center', va = 'center_baseline', color = 'k', fontproperties = prop3)
    else:
        ax.fill(x, y, fc = c, ls = '-', ec = 'k', lw = 2, fill = False)
    return

def round(l, h, x0, y0, ax, c, dir  = 1):
    radius = 2.5
    inset = 0.5
    x = [x0-dir*l/2]
    y = [y0+h/2]
    for i in range(101):
        x.append(x0 + dir*(radius + h/2) * np.sin(i*np.pi/100))
        y.append(y0-radius + (radius + h/2) * np.cos(i*np.pi/100))
    x.append(x0 - dir*l / 2 )
    y.append(y0 - 2 * radius -h/2)
    x.append(x0-dir*(l/2 +inset))
    y.append(y0 - 2*radius)
    x.append(x0 - dir*l / 2 )
    y.append(y0 - 2 * radius +h/2)
    for i in range(101):
        x.append(x0 + dir*(radius - h/2) * np.sin((100-i)*np.pi/100))
        y.append(y0-radius + (radius - h/2) * np.cos((100-i)*np.pi/100))
    x.append(x0 - dir*l / 2 )
    y.append(y0 - h/2)
    x.append(x0-dir*(l/2-inset))
    y.append(y0)
    # ax.plot(x, y, 'ko', ms = 0.5)
    ax.fill(x, y, fc = c)
    return

def text(l, h, x0, y0, ax, txt1='', txt2 = '', dir = 1):
    inset1 = 0.1
    inset2 = 0.5
    x = [x0-inset1+dir*inset2/2, x0+dir*inset2/2, x0+inset1+dir*inset2/2]
    y = [y0+h/2, y0+h/2-inset1, y0+h/2]
    ax.fill(x, y, fc = 'white', ls = '-', ec = 'white')

    hfont = {'fontname': 'Times New Roman'}
    ax.text(x0+dir*inset2/2, y0+2.5, txt1.format(fname), ha = 'center', va = 'center_baseline', color = 'k', fontproperties = prop)
    ax.text(x0+dir*inset2/2, y0+1.25, txt2.format(fname), ha = 'center', va = 'center_baseline', color = '#989696', fontproperties = prop2)

if False or do_talk:
    from matplotlib import font_manager as fm
    import os
    fpath = os.path.join(plt.rcParams["datapath"], "/Users/thomas/Documents/Projects/RS_Cha/fonts/Helvetica Neu Bold.ttf")
    prop = fm.FontProperties(fname=fpath, size = 18)
    prop2 = fm.FontProperties(fname=fpath, size = 12)
    prop3 = fm.FontProperties(fname=fpath, size = 20)
    fname = os.path.split(fpath)[1]

    fig, ax = plt.subplots(figsize = (15,7.5))
    ax.set_xlim(-15, 15)
    ax.set_ylim(-3, 12)
    ax.axis('off')
    grey = '#e7e6e6'
    h=1.3
    plt.tight_layout()
    element(3, h, -12.5, 8, ax, '#991b1b' , txt = '1960')
    text(3, h, -12.5, 8, ax, txt1 = 'slightly variable \n star', txt2  =  'Cousins 1960')
    plt.savefig(dir + '/talk_timeline1.pdf', bbox_inches=0)
    element(2, h, -10., 8, ax, grey)
    element(3, h, -7.5, 8, ax, '#d82e2e', txt = '1964' )
    text(3, h, -7.5, 8, ax, txt1 = 'binary nature\n discovered', txt2  =  'Strohmeier 1964')
    plt.savefig(dir + '/talk_timeline2.pdf', bbox_inches=0)
    element(2, h, -5., 8, ax, grey)
    element(3, h, -2.5, 8, ax, '#e38730', txt = '1967' )
    text(3, h, -2.5, 8, ax, txt1 = 'first orbit\n determination', txt2  =  'e.g. Wild & Lagerweij 1967')
    plt.savefig(dir + '/talk_timeline3.pdf', bbox_inches=0)
    element(2, h, -0., 8, ax, grey)
    element(3, h, 2.5, 8, ax, '#f1d825', txt = '1975' )
    text(3, h, 2.5, 8, ax, txt1 = 'equally bright\n components', txt2  =  'Jorgensen 1975')
    plt.savefig(dir + '/talk_timeline4.pdf', bbox_inches=0)
    element(2, h, 5., 8, ax, grey)
    element(3, h, 7.5, 8.0, ax, '#9fe321', txt = '1977')
    text(3, h, 7.5, 8, ax, txt1 = 'short period\n variability', txt2  =  'McInally 1977')
    plt.savefig(dir + '/talk_timeline5.pdf', bbox_inches=0)
    round(2, h, 10, 8, ax, grey)
    element(3, h, 7.5, 3.0, ax, '#21e321', txt = '1980', dir = -1 )
    text(3, h, 7.5, 3, ax, txt1 = 'both components\n $\delta$ Scuti', txt2  =  'Clausen & Nordström 1980', dir = -1)
    plt.savefig(dir + '/talk_timeline6.pdf', bbox_inches=0)
    element(2, h, 5., 3.0, ax, grey, dir = -1)
    element(3, h, 2.5, 3.0, ax, '#21e38b', txt = '1999', dir = -1 )
    text(3, h, 2.5, 3, ax, txt1 = 'pre-main\n sequence star', txt2  =  'Mamajek et al. 1999', dir = -1)
    plt.savefig(dir + '/talk_timeline7.pdf', bbox_inches=0)
    element(2, h, 0., 3.0, ax, grey, dir = -1)
    element(3, h, -2.5, 3.0, ax, '#21e3ce', txt = '2005', dir = -1 )
    text(3, h, -2.5, 3, ax, txt1 = 'circularized & \n synchronized orbit ', txt2  =  'Alecian et al. 2005', dir = -1)
    plt.savefig(dir + '/talk_timeline8.pdf', bbox_inches=0)
    element(2, h, -5., 3.0, ax, grey, dir = -1)
    element(3, h, -7.5, 3.0, ax, '#2197e3', txt = '2007', dir = -1 )
    text(3, h, -7.5, 3, ax, txt1 = 'detailed stellar\n modelling ', txt2  =  'Alecian et al. 2007', dir = -1)
    plt.savefig(dir + '/talk_timeline9.pdf', bbox_inches=0)
    round(2, h, -10, 3, ax, grey, dir =-1)
    element(3, h, -7.5, -2, ax, '#4945c8' , txt = '2009')
    text(3, h, -7.5, -2, ax, txt1 = 'first detection\n of frequencies ', txt2  =  'Böhm et al. 2009')
    plt.savefig(dir + '/talk_timeline10.pdf', bbox_inches=0)
    element(2, h, -5., -2, ax, grey)
    element(3, h, -2.5, -2, ax, '#9245c8', txt = '2013' )
    text(3, h, -2.5, -2, ax, txt1 = 'third\n component ', txt2  =  'Woollands et al. 2013')
    plt.savefig(dir + '/talk_timeline11.pdf', bbox_inches=0)
    element(2, h, 0., -2, ax, grey)
    element(3, h, 2.5, -2, ax, '#c845bc', txt = '2020' )
    text(3, h, 2.5, -2, ax, txt1 = 'tidally perturbed \n modes ', txt2  =  'Steindl et al. in preparation')
    plt.savefig(dir + '/talk_timeline12.pdf', bbox_inches=0)
    element(7, h, 7.5, -2, ax, grey)
    plt.savefig(dir + '/talk_timeline13.pdf', bbox_inches=0)
    plt.show()

### Talk Pulsation LC ###
if False or do_all:
    data = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
    time = data[0]
    flux = data[1]
    fig, ax = plt.subplots(figsize = figsize1, dpi = dpi)
    ax.plot(time, flux, 'ko', markersize = 0.75)
    ax.set_xlabel('Time - 2457000 [BTJD days]')
    ax.set_ylabel('normalized flux')
    plt.tight_layout()
    plt.savefig(dir + 'talk_pulsation_lc.png', bbox_inches = 0)
    plt.show()
    plt.close()

## additioanl stuff.
if False:

    l1_m0_primary = np.array([(26.475 +25.278)/2, (11.947 + 13.142)/2, (30.091 +31.286)/2, (19.435 + 20.631)/2, ])
    l1_m0_secondary =np.array( [(20.123 +18.925)/2, (25.636 + 24.438)/2, (12.005 + 10.807)/2, (26.024 + 27.222)/2, (27.3055 + 26.108)/2, (25.796 + 24.597)/2,])
    l0_secondary = np.array([21.06, 21.15, 18.79, 26.3, 25.36])

    l1_sec_alecian = np.array([132.07084101938864, 138.5667123211755, 144.69919484447118, 171.0603193704428, 179.36125911440809, 185.85950550625046, 215.82126655635696, 222.680526636635, 230.26418918383993, 260.94797761081776, 268.165876289476, 275.02988654986507, 306.0746886652786, 314.0193649009192, 320.1542225142704,349.75496987594113, 356.97286855459936,364.5517809216933])
    l1_sec_alecian = l1_sec_alecian*24*3600*1e-6

    print(l1_m0_primary)
    print(l1_m0_secondary)
    for f in np.linspace(3, 5, 100):
        fig, ax = plt.subplots()
        ax.plot(l1_m0_primary%f, l1_m0_primary, 'ko')
        ax.plot(l1_m0_secondary%f, l1_m0_secondary, 'ro')
        ax.plot(l1_sec_alecian%f, l1_sec_alecian, 'rx')
        ax.plot(l0_secondary%f, l0_secondary, 'r^')
        ax.set_title(f)
        plt.show()
    plt.close()