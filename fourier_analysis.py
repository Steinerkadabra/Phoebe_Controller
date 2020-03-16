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
        if  (0.025 > f.f % (0.5988495842998753) and f.f < 10)  or  (f.f % (0.5988495842998753)  >(0.5988495842998753)-0.025 and f.f < 10):
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


    plt.show()
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

remove_pulsations('endurance/RS_Cha_lightcurve.txt', 'endurance/iteration1/residuals/', 'endurance/iteration1/residuals.txt', output_dir='iteration1')
