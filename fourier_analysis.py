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

def get_freqs_to_remove():
    settings, statistics, results = load_results('removed_binary_first_time/data/result.csv')
    fs = results['frequency'].values
    amps = results['amp'].values
    snrs = results['snr'].values
    phis = results['phase'].values
    freqs = []
    for i in range(len(fs)):
        if snrs[i]>3.8:
            freqs.append(mode(fs[i].n, amps[i].n, phis[i].n, snrs[i]))

    fig, ax = plt.subplots(2,1, figsize = (12,8))
    freqs_to_remove = []
    fs = []
    for f in freqs:
        #fs.append(f.f)
        if  (0.025 > f.f % (0.5988495842998753) and f.f < 10)  or  (f.f % (0.5988495842998753)  >(0.5988495842998753)-0.025 and f.f <10) or f.f < 0.25 :
            ax[0].plot([f.f, f.f], [0, f.snr], 'k--')
        else:
            fs.append(f.f)
            freqs_to_remove.append(f)
            ax[0].plot([f.f, f.f], [0, f.snr], 'k-')
    for i in range(1,17):
        ax[0].plot([i*0.5988495842998753, i*0.5988495842998753], [0, 5], 'r--')
    #for i in range(1,60):
    #    ax.plot([i*1/1.6699, i*1/1.6699], [0, 2.5], 'b--')

    ax[0].set_xlabel('Frequency ($d^{-1}$')
    ax[0].set_ylabel('SNR')

    ax[0].plot(0, 0, 'k--', label = 'probably from binary')
    ax[0].plot(0, 0, 'k-', label = 'probably pulsations (to remove)')
    ax[0].plot(0, 0, 'r--', label = 'binary multiple')

    ax[1].plot(np.array(fs) % 0.598849584299875, fs, 'ko')
    #ax[2].plot(np.array(fs) % (2*0.598849584299875), fs, 'ko')
    ax[0].legend()


    plt.show()
    return freqs_to_remove


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

ftr = get_freqs_to_remove()
time, flux = load_lc('removed_binary_first_time.txt', plot = False, parts_of_lc=False)
print(time, flux)
mag = np.zeros(len(time))
for f in ftr:
    mag = mag + fourier_model(time, [f.f], [f.amp], [f.phase])

fig, ax = plt.subplots(2,1, figsize = (12,8))
ax[0].plot(time, flux, 'ko', markersize = 0.75)
ax[0].plot(time, mag, 'r-')
ax[1].plot(time, flux-mag, 'ko', markersize = 0.75)
plt.show()


fig, ax = plt.subplots( figsize = (12,8))
time, flux = load_lc('RS_Cha_lightcurve.txt', plot = False, parts_of_lc=False)
ax.plot(time, flux, 'ko', markersize = 0.75)
ax.plot(time, flux - mag, 'r-')
plt.show()


np.savetxt('LC_iteration_1.txt', np.array([time, flux - mag]).T)