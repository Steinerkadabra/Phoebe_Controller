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
import astropy as ap
import phoebe_controller



class lightcurve_to_fold():
    def __init__(self, time, flux):
        self.time = time
        self.mag = flux


    def phase_fold(self, rot_freq,  ax = None, plot = True):
        self.time_folded =self.time % (1/rot_freq)
        sort = np.argsort(self.time_folded)
        time= self.time_folded[sort]*rot_freq
        mag = self.mag[sort]
        bins = []
        means = []
        sds = []
        num = 200
        bins.append(0)
        ind = np.where(np.logical_and(time>=0, time<=0.5/num))
        means.append(np.mean(mag[ind]))
        sds.append(np.std(mag[ind]))
        for i in range(num-1):
            ind = np.where(np.logical_and(time>=(i+0.5)/num, time<=(i+1.5)/num))
            if ind[0].size > 0:
                bins.append((i+1)/num)
                means.append(np.mean(mag[ind]))
                sds.append(np.std(mag[ind]))
        bins.append(1)
        ind = np.where(np.logical_and(time>=(num-0.5)/num, time<=1))
        means.append(np.mean(mag[ind]))
        sds.append(np.std(mag[ind]))
        try:
            smooth = sig.savgol_filter(means, 5, 3)
        except ValueError:
            return 1.0
        inter = interp1d(bins, means, 'cubic')
        vals = np.linspace(0, 1, 1000)
        if plot:
            ax.plot(time, mag, 'ko', markersize = 0.75, zorder = 0)
            ax.plot(vals, inter(vals), 'r-', lw = 0.5, zorder = 1)
            ax.errorbar(bins, means, sds, fmt =  'ro', markersize = 2, zorder=1)
            ax.set_xlabel(f'Phase with period of {np.round(1/rot_freq, 10)} days')
            ax.set_ylabel('magnitude (mag)')
        dif = mag - inter(time)
        return np.std(dif)/np.std(self.mag)

    def phase_plot_minimization(self, min, max, res = 0.0005, plot_sigmas = True, ax = None, fmt = 'k-', sig_guess = 0.01, plot_fit = False, plot_expected = False, expected = 7):
        fs = np.linspace(min, max, int((max-min)/res))
        difs = []
        for f in tqdm(range(len(fs))):
            dif = self.phase_fold(fs[f], plot = False)
            difs.append(dif)
        difs = np.array(difs)
        if plot_sigmas:
            ax.plot(fs, difs, fmt)
        nans = np.logical_not(np.isnan(difs))
        difs = difs[nans]
        const_guess = np.mean(difs)
        mean_guess = fs[np.argmin(difs)]
        amp_guess = np.min(difs) - np.max(difs)
        try:
            popt, pcov = curve_fit(gaus, fs[nans], difs, p0=[amp_guess, mean_guess, sig_guess, const_guess])
            if plot_fit:
                ax.plot(fs, gaus(fs, *popt), 'r-', label=f'$\mu_f$: {np.round(popt[1],4)}      $\sigma_f$: {np.round(popt[2], 4)} \n'
                                                         f'$\mu_P$: {np.round(1/popt[1],4)}        $\sigma_P$: {np.round(popt[2]/popt[1]**2, 4)}')
                ax.set_xlabel('Frequency ($d^{-1}$)')
                ax.set_ylabel('$\sigma_\mathrm{residual}$ (mag)')
                ax.legend()
                if plot_expected:
                    up = 1 + 0.05*(1-np.min(difs))
                    for n in range(expected):
                        ax.plot([popt[1]/n] , up, 'kv', markersize = 5)
            return mean_guess, popt
        except RuntimeError:
            print('no gauss fit possible')
            pass
            return mean_guess, []

    def set_up_binary_model(self, rot_freq):
        self.time_folded = self.time % (1 / rot_freq)
        sort = np.argsort(self.time_folded)
        time = self.time_folded[sort] * rot_freq
        mag = self.mag[sort]
        bins = []
        means = []
        sds = []
        num = 500
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
        try:
            smooth = sig.savgol_filter(means, 5, 3)
        except ValueError:
            return 1.0
        self.binary_model = interp1d(bins, means, 'cubic')
        self.binary_freq = rot_freq
        return bins, means, sds

    def get_binned_lc(self, rot_freq):
        bins, means, sds = self.set_up_binary_model(rot_freq)
        bins = np.array(bins)/rot_freq
        return bins, means, sds


def gaus(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c

