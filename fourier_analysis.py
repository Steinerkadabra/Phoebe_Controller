import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from tqdm import tqdm
from pandas import read_csv
from io import StringIO
from uncertainties import ufloat_fromstr
import os
from scipy.optimize import curve_fit
#from maelstrom.utils import amplitude_spectrum
import astropy as ap
import lightkurve as lk


def load_lc(lc_file = 'RS_CHA_lightcurve.txt', plot= True, npoints = 5000, points_step = 5, parts_of_lc = True):
    data = np.loadtxt(lc_file).T
    times = data[0]
    flux = data[1]

    if parts_of_lc:
        times = times[:npoints:points_step]
        flux = flux[:npoints:points_step]
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, flux, 'ko', markersize = 0.75)
        plt.show()
        plt.close()
    return  (times, flux)

class mode():
    def __init__(self, f, amp, phase, snr):
        self.f = f
        self.amp = amp
        self.phase = phase
        self.snr = snr

class lc():
    def __init__(self, time, flux):
        #lc_data = np.genfromtxt(path + '/data/LC.txt', skip_header=1, delimiter=',', usecols=[0,1]).T
        self.time = time #lc_data[0]
        self.mag = flux #lc_data[1]

def load_results(path: str):
    """
    Loads the pandas dataframes from the results file
    :param path: exact path of the results file
    :return: 3 pandas dataframes, settings, statistics and results
    """

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

def get_freqs_to_remove(file):
    print(file + 'data/results.csv')
    settings, statistics, results = load_results(file + 'data/result.csv')
    fs = results['frequency'].values
    amps = results['amp'].values
    snrs = results['snr'].values
    phis = results['phase'].values
    freqs_init = []
    for i in range(len(fs)):
        if snrs[i]>3.8:
            freqs_init.append(mode(fs[i].n, amps[i].n, phis[i].n, snrs[i]))

    freqs = []
    freqs_rl = []
    for i in range(1, len(freqs_init)):
        fnew = freqs_init[i]
        new = True
        if fnew.f < 0.25:
            new = False
        for f in freqs_init[:i]:
            if abs(fnew.f-f.f) < 0.01:
                new = False
        if new:
            freqs.append(fnew)
        else:
            freqs_rl.append(fnew)
    freqs_binary = []
    freqs_to_remove = []
    for f in freqs:
        if  (0.025 > f.f % (0.5988495842998753) and f.f < 15)  or  (f.f % (0.5988495842998753)  >(0.5988495842998753)-0.025 and f.f < 15):
            freqs_binary.append(f)
        else :
            freqs_to_remove.append(f)

    fig, ax = plt.subplots(2,1, figsize=(15,8))
    for i in range(1,25):
        ax[0].plot([i*0.5988495842998753, i*0.5988495842998753], [0, 0.001], 'r--')
    for f in freqs_binary:
        ax[0].plot([f.f, f.f], [0, f.amp], 'k--')
    for f in freqs_rl:
        ax[0].plot([f.f, f.f], [0, f.amp], 'k:')
    fs = []
    for f in freqs_to_remove:
        ax[0].plot([f.f, f.f], [0, f.amp], 'k-')
        fs.append(f.f)


    ax[0].set_xlabel('Frequency ($d^{-1}$')
    ax[0].set_ylabel('SNR')

    ax[0].plot(0, 0, 'k--', label = 'probably from binary')
    ax[0].plot(0, 0, 'k-', label = 'probably pulsations (to remove)')
    ax[0].plot(0, 0, 'r--', label = 'binary multiple')

    ax[1].plot(np.array(fs) % 0.598849584299875, fs, 'ko')
    #ax[2].plot(np.array(fs) % (2*0.598849584299875), fs, 'ko')
    ax[0].legend()


    #plt.show()
    return freqs_to_remove, freqs_binary, freqs_rl


def sin(x: np.ndarray, amp: float, f: float, phase: float) -> np.ndarray:
    """
    Sinus function, used for fitting.

    :param x: Time axis
    :param amp: amplitude
    :param f: frequency
    :param phase: phase, normed to 1
    """
    return amp * np.sin(2. * np.pi * (f * x + phase))

def fourier_model(time, fs, amps, phis):
    mag = np.zeros(len(time))
    for i in range(0, len(fs)):
        mag += sin(time, amps[i], fs[i], phis[i])
    return mag


def sin_multiple(x: np.ndarray, *params) -> np.ndarray:
    """
    Multiple sinii summed up

    :param x: Time axis
    :param params: Params, see *sin* for signature
    """
    y = np.zeros(len(x))
    for i in range(0, len(params), 3):
        y += sin(x, params[i], params[i + 1], params[i + 2])

    return y


def remove_pulsations(starting_lc, smurfs_result, residuals_lc, output_dir):
    orig_lc = np.loadtxt(starting_lc).T


    ftr, fb, frl = get_freqs_to_remove(smurfs_result)
    time, flux = load_lc(residuals_lc, plot = False, parts_of_lc=False)
    flux_mod = np.zeros(len(time))
    for f in ftr:
        flux_mod = flux_mod + fourier_model(time, [f.f], [f.amp], [f.phase])
    #for f in fb:
    #    flux_mod = flux_mod + fourier_model(time, [f.f], [f.amp], [f.phase])
    for f in frl:
        flux_mod = flux_mod + fourier_model(time, [f.f], [f.amp], [f.phase])
    flux_mod = flux_mod + fourier_model(time, [11.0704029], [0.001677], [0.3697])

    fig3 = plt.figure(constrained_layout=True, figsize=(15,8))
    gs = fig3.add_gridspec(3, 3)
    ax00 = fig3.add_subplot(gs[0, :-1])
    ax01 = fig3.add_subplot(gs[1, :-1])
    ax02 = fig3.add_subplot(gs[2, :-1])
    ax10 = fig3.add_subplot(gs[0, -1:])
    ax11 = fig3.add_subplot(gs[1, -1:])
    ax12 = fig3.add_subplot(gs[2, -1:])
    ax00.plot(time, flux, 'ko', markersize = 0.75)
    ax10.plot(time, flux, 'ko', markersize = 0.75)
    ax00.plot(time, flux_mod, 'r-')
    ax10.plot(time, flux_mod, 'r-')
    for ax in [ax00, ax01, ax10, ax11, ax02, ax12]:
        ax.set_xlabel('Time - 2457000 [BTJD days]')
        ax.set_ylabel('normalized flux')
    for ax in [ax10, ax11, ax12]:
        ax.set_xlim(np.min(time), np.min(time) + 3.2)
    ax01.plot(time, flux-flux_mod, 'ko', markersize = 0.75)
    ax11.plot(time, flux-flux_mod, 'ko', markersize = 0.75)
    ax02.plot(time, orig_lc[1], 'ko', markersize = 0.75)
    ax12.plot(time, orig_lc[1], 'ko', markersize = 0.75)
    ax02.plot(time, orig_lc[1]+flux_mod, 'ro', markersize = 0.5)
    ax12.plot(time, orig_lc[1]+flux_mod, 'ro', markersize = 0.5)
    plt.tight_layout()
    #plt.show()
    np.savetxt(f'endurance/{output_dir}/removed_pulsation.txt', np.array([time, flux-flux_mod,]).T)
    np.savetxt(f'endurance/{output_dir}/result_{output_dir}.txt', np.array([time, orig_lc[1]+flux_mod]).T)
    plt.savefig(f'endurance/{output_dir}/result_{output_dir}.png')
    plt.show()

def plot_set(fs, ax, fmt):
    for f in fs:
        ax.plot([f.f, f.f], [0, f.amp], fmt)

def boxed_frequencies(fmod, f, size, ax1, ax2, color, lof, size2 = 0.01):
    ax2.plot([fmod - size2, fmod + size2, fmod + size2, fmod - size2, fmod - size2],
             [f - size, f - size, f + size, f + size, f - size], color + '-')
    fitting = []
    for lis in lof:
        for fs in lis:
            if f-size <= fs.f <=  f+ size and abs(fs.f%0.5988495842998753 -fmod) < 0.01:
                fitting.append(fs)
    plot_set(fitting, ax1, color + '-')

plt.rcParams.update({'font.size': 20.0, 'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small'})

#remove_pulsations('endurance/RS_Cha_lightcurve.txt', 'endurance/RS_Cha_minus_phoebe_minus_binary_mutiples/', 'endurance/RS_Cha_minus_phoebe_minus_binary_mutiples.txt', output_dir='iteration1_minus_binary_pulsation_analysis')


def echelle_diagramm():
    resids = np.genfromtxt( 'endurance/iteration1_minus_binary_pulsation_analysis/removed_pulsation_1/data/PS_residual.txt', skip_header=1, delimiter=',').T
    flux_res = resids[1]
    for i in range(int(len(flux_res)/7),len(flux_res)):
        if resids[0][i]>50:
            stop = i
            break
    noise_level = np.mean([flux_res[stop]])
    print(np.mean([flux_res[:stop]]), np.median([flux_res[:stop]]))

    settings, statistics, results = load_results('endurance/RS_Cha_minus_phoebe_minus_binary_mutiples/data/result.csv')
    fs = results['frequency'].values
    amps = results['amp'].values
    snrs = results['snr'].values
    phis = results['phase'].values
    freqs_38= []
    freqs_35 = []
    freqs_b35 = []
    for i in range(len(fs)):
        if fs[i].n>0.5:
            if amps[i]>=0.0003:
                freqs_38.append(mode(fs[i].n, amps[i].n, phis[i].n, snrs[i]))
            elif snrs[i]>=3.8:
                freqs_35.append(mode(fs[i].n, amps[i].n, phis[i].n, snrs[i]))
            else:
                freqs_b35.append(mode(fs[i].n, amps[i].n, phis[i].n, snrs[i]))
    freqs_38.append(mode(11.0704, 0.001677, 0.3737, 8))

    fig = plt.figure(constrained_layout=True, figsize=(15, 9.5))
    gs = fig.add_gridspec(5, 8)
    ax1 = fig.add_subplot(gs[0,:])
    ax3 = fig.add_subplot(gs[1,:])
    ax2 = fig.add_subplot(gs[2:,:-1])
    ax_cb = fig.add_subplot(gs[2:,-1])
    sig = ax2.scatter(np.array([x.f for x in freqs_38])%0.5988495842998753, np.array([x.f for x in freqs_38]), s= 30,
                        c = np.array([x.amp for x in freqs_38]), marker = 'o', linewidths =  2, vmin = 0,
                        vmax= np.max([x.amp for x in freqs_38]))
    b4  = ax2.scatter(np.array([x.f for x in freqs_35])%0.5988495842998753, np.array([x.f for x in freqs_35]), s= 30,
                        c = np.array([x.amp for x in freqs_35]), marker = 'o', linewidths =  2, vmin = 0,
                        vmax= np.max([x.amp for x in freqs_38]))
    b4.set_facecolor('none')
    b35  = ax2.scatter(np.array([x.f for x in freqs_b35])%0.5988495842998753, np.array([x.f for x in freqs_b35]),
                         s= 30, c = np.array([x.amp for x in freqs_b35]), marker = 'o', linewidths =  1, vmin = 0, #
                         vmax= np.max([x.amp for x in freqs_38]))
    ax2.plot([0,0], [5, 5 + 0.5988495842998753], 'k-')
    ax2.plot([0.01,0.01], [5, 5 + 2*0.5988495842998753], 'k-')
    ax2.plot([0.02,0.02], [5, 5 + 3*0.5988495842998753], 'k-')
    ax1.plot([7.5,7.5 + 0.5988495842998753], [0.0015, 0.0015], 'k-')
    ax1.plot([7.5,7.5 + 2*0.5988495842998753], [0.0014, 0.0014], 'k-')
    ax1.plot([7.5,7.5 + 3*0.5988495842998753], [0.0013, 0.0013], 'k-')
    ax3.plot([10.5,10.5 + 0.5988495842998753], [0.0005, 0.0005], 'k-')
    ax3.plot([10.5,10.5 + 2*0.5988495842998753], [0.0004, 0.0004], 'k-')
    ax3.plot([10.5,10.5 + 3*0.5988495842998753], [0.0003, 0.0003], 'k-')
    b35.set_facecolor('none')
    fig.colorbar(b4, ax = ax_cb, label = 'Amplitude')
    ax_cb.axis('off')
    ax2.set_xlabel('Frequency modulo orbital frequency $d^{-1}$')
    ax2.set_ylabel('Frequency $d^{-1}$')
    ax1.set_xlabel('Frequency $d^{-1}$')
    ax3.set_xlabel('Frequency $d^{-1}$')
    ax1.set_ylabel('Amplitude')
    ax3.set_ylabel('Amplitude')

    fmod = 0.29
    f = 12
    size = 5
    boxed_frequencies(fmod, f, size, ax1, ax2, 'r', [freqs_38, freqs_35])
    boxed_frequencies(0.246, 12.5, 3.75, ax1, ax2, 'g', [freqs_38, freqs_35], size2 = 0.005)
    boxed_frequencies(0.218, 10.5, 4, ax1, ax2, 'b', [freqs_38, freqs_35], size2 = 0.0075)
    boxed_frequencies(0.042, 22, 11, ax3, ax2, 'm', [freqs_38, freqs_35], size2 = 0.0075)
    boxed_frequencies(0.014, 25, 11, ax3, ax2, 'c', [freqs_38, freqs_35], size2 = 0.0075)

    plt.tight_layout(w_pad = 0, h_pad=0)
    plt.show()


def generate_ooe_lc():
    fig, ax = plt.subplots(3,1)
    data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
    time = data[0]
    flux = data[1]
    mod = np.loadtxt('endurance/iteration1/binary_model.txt').T
    mod_time = mod[0]
    mod_flux = mod[1]
    res = np.loadtxt('endurance/iteration1_remove_binary_multiples_step1/removed_binary.txt').T
    res_time = res[0]
    res_flux = res[1]
    ooe = np.where(mod_flux>0.995)
    ax[0].plot(time, flux, 'ko', ms = 0.75)
    ax[0].plot(mod_time, mod_flux, 'r-')

    smooth = sig.savgol_filter(mod_flux[ooe]-flux[ooe], 10001, 2)
    ax[1].plot(time[ooe], mod_flux[ooe]-flux[ooe], 'ko', ms = 0.75)
    ax[1].plot(time[ooe], smooth , 'r-')
    lco = lk.LightCurve(time[ooe],mod_flux[ooe]-flux[ooe]- smooth)
    pdg = lco.to_periodogram()
    ax[2].plot(pdg.frequency, pdg.power, 'k-')
    lco = lk.LightCurve(time, mod_flux-flux)
    pdg = lco.to_periodogram()
    ax[2].plot(pdg.frequency, pdg.power, 'r--')
    np.savetxt('endurance/iteration1_removed_binary_out_of_eclipse_savgol.txt', np.array([time[ooe],mod_flux[ooe]-flux[ooe]- smooth -np.mean(mod_flux[ooe]-flux[ooe]- smooth)]).T)
    plt.show()


'''
period_res = np.loadtxt('endurance/iteration1_removed_binary_out_of_eclipse_period04_result/period04_out_of_eclipse_frequencies.per', usecols=[1,2,3]).T
fs = period_res[0]
amps = period_res[1]
phis = period_res[2]
snrs = [15.05000, 19.73500, 20.19227, 24.86000, 20.82400, 17.52000, 10.59000, 13.62300, 7.29334, 6.84200, 9.96000,
        8.32000, 7.68900, 5.21000, 13.80000, 7.76500, 8.81900, 6.82000, 10.09000, 8.39000, 5.27000, 8.10000,
        5.69000, 9.50900, 7.42570]
freqs_init = []
for i in range(len(fs)):
    if snrs[i] > 3.8:
        freqs_init.append(mode(fs[i], amps[i], phis[i], snrs[i]))


freqs_binary = []
freqs_to_remove = []
for f in freqs_init:
    if  ((0.025 > f.f % (0.5988495842998753) and f.f < 15)  or  (f.f % (0.5988495842998753)  >(0.5988495842998753)-0.025 and f.f < 15)) and f.f> 0.4:
        freqs_binary.append(f)
    else :
        freqs_to_remove.append(f)


fig, ax = plt.subplots(3,1)
dataooe = np.loadtxt('endurance/' + 'iteration1_removed_binary_out_of_eclipse.txt').T
timeooe = dataooe[0]
fluxooe = dataooe[1]
flux_mod = np.zeros(len(timeooe))
for f in freqs_binary:
    print(f.f, f.amp, f.phase)

print('######')
for f in freqs_to_remove:
    print(f.f)
#for f in freqs_binary:
#    flux_mod = flux_mod + fourier_model(timeooe, [f.f], [f.amp], [f.phase])

ax[0].plot(timeooe, fluxooe, 'ko', ms =0.75)
ax[0].plot(timeooe, flux_mod, 'r-')
for f in freqs_to_remove:
    flux_mod = flux_mod + fourier_model(timeooe, [f.f], [f.amp], [f.phase])
ax[0].plot(timeooe, flux_mod, 'b-', lw = 0.5)
ax[1].plot(timeooe, fluxooe-flux_mod, 'ko', ms  = 0.75)

data = np.loadtxt('endurance/' + 'RS_Cha_lightcurve.txt').T
time = data[0]#(data[0]% (1/0.5988495842998753))*0.5988495842998753
flux = data[1]

flux_mod = np.zeros(len(time))
for f in freqs_binary:
    flux_mod = flux_mod + fourier_model(time, [f.f], [f.amp], [f.phase])


ax[2].plot(time, flux, 'ro', ms = 0.25)
ax[2].plot(time, flux+flux_mod + 0.05, 'go', ms = 0.25)

ftr, fb, frl = get_freqs_to_remove('endurance/RS_Cha_minus_phoebe_minus_binary_mutiples/')


for f in ftr:
    flux_mod = flux_mod + fourier_model(time, [f.f], [f.amp], [f.phase])

for f in frl:
    flux_mod = flux_mod + fourier_model(time, [f.f], [f.amp], [f.phase])

for f in fb:
    flux_mod = flux_mod + fourier_model(time, [f.f], [f.amp], [f.phase])

print(flux_mod)
datab = np.loadtxt('endurance/' + 'iteration2/binary_model.txt').T
timeb = datab[0]#(data[0]% (1/0.5988495842998753))*0.5988495842998753
fluxb = datab[1]
ax[2].plot(time, flux+flux_mod - 0.05, 'bo', ms = 0.25)
ax[2].plot(time, fluxb + 0.05, 'k-')

plt.show()
'''