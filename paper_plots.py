import numpy as np
import phase_analysis as pha
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.optimize import curve_fit
from tabulate import tabulate

print(plt.rcParams)
plt.rcParams.update({'font.size': 15, 'xtick.labelsize': 'small', 'ytick.labelsize': 'small',})
dir = 'paper_plots/'
figsize1 = (10,5)
figsize_onecolumn = (15,7.5)

def figsize(n):
    return (10,5*n)

dpi = 200
do_all = False
do_mail = False



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
    plt.show()

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
    num = 350
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
    ax4.set_xlabel('Phase')
    ax2.set_ylabel('flux')
    ax3.set_ylabel('flux')
    ax4.set_ylabel('flux')
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(dir + '/Figure4_binary_modelling.pdf', bbox_inches=0)
    #plt.show()


### Figure5: Different Amplitude Spectra ###
if True or do_all:
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

    for a in [ax00, ax01, ax02]:
        a.set_xlabel('frequency $d^{-1}$')
        a.set_ylabel('power')
        a.set_xlim(0, 35)
        a.plot(a.set_xlim(), [0.000165, 0.000165], 'r:', lw = 0.25)
        for i in range(80):
            a.plot([i*freq, i*freq], [0, a.set_ylim()[1]], 'b--', lw = 0.25)
    for a in [ax10, ax11, ax12]:
        a.set_xlabel('Phase')
        a.set_ylabel('flux')
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
    # plt.savefig(dir + '/Figure5_amplitude_spectra.pdf', bbox_inches=0)
    plt.show()

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
        string = f'F{i+1}: {np.round(modes[i].f,2)}' +' d$^{-1}$\n$\\frac{\chi^2}{\mathrm{DoF}} = $' + str(np.round(chisq/df, 2))
        a.text(0.825, a.set_ylim()[0]+ 0.1*(a.set_ylim()[1]-a.set_ylim()[0]), string)

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
    string = '$\\frac{\chi^2}{\mathrm{DoF}} = $' + str(np.round(chisq/df, 2))
    ax5.text(0.855, ax5.set_ylim()[0]+ 0.1*(ax5.set_ylim()[1]-ax5.set_ylim()[0]), string)

    ax5.set_ylabel('Amplitude / Mean')
    ax5.set_xticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(dir + '/Figure6_model_motivation.pdf', bbox_inches=0)


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
                                                     r['Amplitude_error'], r['Phase'], r['Phase_error'],r['SNR'] ):
        if amp > 0.000165:
            l[f'F{num}'] = frequency(f, ferr, amp, amperr, phase, phaseerr, snr, f'F{num}')
            num+=1
    for i in range(2, len(l)+1):
        for (a, b) in [(1,1), (1, -1), (-1, 1)]:
                l[f'F{i}'].find_combination(l, a, b)
        for (a, b) in [(2,1), (2, -1), (1, 2), (-1, 2), (-2, 1), (1,-2), (2,2), (2, -2)]:
                l[f'F{i}'].find_combination(l, a, b)
        for (a, b) in [(3,0), (4,0)]:
                l[f'F{i}'].find_combination(l, a, b)
    # print('key   Frequency  Amplitude    Phase     SNR')
    # for i in range(1, len(l)+1):
    #     l[f'F{i}'].print()
    return l


if False or do_all:
    import mini_smurfs as ms
    r = ms.load_result('minismurfs/result.csv')
    r = r.sort_values(by=['Amplitude'], ascending=False)
    l = comb_frequencies(r)

    table = []
    total_count = 0
    above_count = 0
    timebase = 1/0.011865383643105324
    amp_err = np.sqrt(2/55309)*0.000165
    f_err =np.sqrt(6/55309)*0.000165/timebase/np.pi
    # for f, ferr, amp, amperr, phase, phaseerr in zip(r['Frequency'],r['Frequency_error'], r['Amplitude'],r['Amplitude_error'], r['Phase'], r['Phase_error']):
    #     total_count += 1
    #     if amp > 0.000165:
    #         above_count += 1
    #         line = []
    #         dig, err = sig_digit(np.sqrt(ferr**2+(f_err/amp)**2))
    #         string = '${:' + str(dig) + 'f}({:d})$'
    #         line.append(string.format(f, err))
    #         dig, err = sig_digit(np.sqrt(amperr**2+amp_err**2))
    #         string = r'${:' + str(dig) +'d}({:d})$'
    #         line.append(string.format(int(1000000*amp), err))
    #         dig, err = sig_digit(np.sqrt(phaseerr**2+(amp_err/amp)**2))
    #         string = r'${:.' + str(dig) + 'f}({:d})$'
    #         line.append(string.format(phase, err))
    #         table.append(line)
    for i in range(1, len(l)+1):
        mode = l[f'F{i}']
        total_count += 1
        if mode.amp > 0.000165:
            above_count += 1
            line = []
            line.append(mode.key + ' = ' + mode.combination)
            dig, err = sig_digit(np.sqrt(mode.ferr**2+(f_err/mode.amp)**2))
            string = '${:.' + str(dig) + 'f}({:d})$'
            line.append(string.format(mode.f, err))
            dig, err = sig_digit(np.sqrt(mode.amperr**2+amp_err**2))
            print(mode.amp, np.sqrt(mode.amperr**2+amp_err**2))
            print(dig, err)
            string = r'${:' + str(dig) +'d}({:d})$'
            line.append(string.format(int(1000000*mode.amp), err))
            dig, err = sig_digit(np.sqrt(mode.phaseerr**2+(amp_err/mode.amp)**2))
            string = r'${:.' + str(dig) + 'f}({:d})$'
            line.append(string.format(mode.phase, err))
            table.append(line)

    print(tabulate(table, tablefmt="latex_raw"))
    # print(len(r))
    # print(above_count)

#### Figure? compare bics aics #####

def sin_multiple(x, *params):
    y = np.zeros(len(x))
    for i in range(0, len(params), 3):
        y += sin(x, params[i], params[i + 1], params[i + 2])
    return y

def sin(x, amp, f, phase):
    return amp * np.sin(2. * np.pi * (f * x + phase))

if True or do_all:
    import mini_smurfs as ms
    r = ms.load_result('minismurfs/result.csv')
    r = r.sort_values(by=['Amplitude'], ascending=False)
    data = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
    time = data[0]
    flux = data[1]
    n = len(time)
    fs = np.array(r['Frequency'])
    amps = np.array(r['Amplitude'])
    phis =  np.array(r['Phase'])
    print(amps[76:85])
    print(fs[76:85])
    print(phis[76:85])
    rsss = []
    bic = []
    aic = []
    ks = []
    for i in range(1, len(fs)+1):
        print(i)
        arr = []
        k = 1+3*i
        ks.append(k)
        for j in range(0, i):
            arr.append(amps[j])
            arr.append(fs[j])
            arr.append(phis[j])
        mod = sin_multiple(time, *arr)
        RSS = n*np.log(np.sum((flux-mod)**2)/n)
        # plt.plot(time, flux, 'ko', ms = 0.75)
        # plt.plot(time, mod, 'r-')
        # plt.title(RSS)
        # plt.show()
        # plt.close()
        rsss.append(RSS)
        bic.append(RSS + k*np.log(n))
        aic.append(RSS + 2*k + n)
    fig, ax = plt.subplots(2,1,figsize = figsize1, dpi = dpi)
    ax[0].plot(ks, bic, 'ko')
    ax[0].plot(ks, aic, 'ro')
    ax[1].plot(ks, rsss, 'ro')
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

if True:
    R1 = 2.15*696340
    R2 = 2.36*696340

    vsin1 = 68
    vsin2 = 72

    i = 83.4

    u1 = 2*np.pi * R1
    u2 = 2*np.pi * R2
    per1 = u1/vsin1/3600/24/np.sin(i)
    per2 = u2/vsin2/3600/24/np.sin(i)
    print(per1, per2)



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


