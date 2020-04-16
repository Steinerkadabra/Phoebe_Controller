import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.integrate as integ
import scipy.interpolate as interp


class Mode:
    def __init__(self, f, amp, phase, snr):
        self.f = f
        self.amp = amp
        self.phase = phase
        self.snr = snr
        self.aic = None
        self.bic = None
        self.vr = None
        self.pr = None
        self.iter = None
        self.nofp = None

class Iteration:
    def __init__(self, time, flux, mode):
        self.time = time.copy()
        self.flux = flux.copy()
        self.lc = lk.LightCurve(self.time, self.flux)
        self.mode = mode



class MiniSmurfs:
    def __init__(self, time, flux, fmax=40, fmin=0.000001):
        self.time = time.copy()
        self.flux = flux.copy()
        self.lc = lk.LightCurve(self.time, self.flux)
        self.resid = flux.copy()
        self.resid_lc = lk.LightCurve(self.time, self.resid)
        self.iterations = []
        self.iters = 0
        self.fmax = fmax
        self.fmin = np.min([fmin, 0.00001])
        self.result = []
        self.nofp = 0
        self.var = np.var(flux)
        pdg = lk.LightCurve(times, flux).to_periodogram(maximum_frequency=fmax, minimum_frequency=self.fmin)
        pdg_func = interp.interp1d(pdg.frequency, pdg.power, fill_value='extrapolate')
        self.power_val = integ.quad(pdg_func, self.fmin, self.fmax, limit=5000, epsabs=1.49e-05, epsrel=1.49e-03)[0]
        self.n = len(time)
        self.aic_val = self.n * np.log(np.sum(self.flux**2)/self.n) +self.n
        self.bic_val =  self.n * np.log(np.sum(self.flux**2)/self.n)


    def run(self, snr=4, ws=2):
        stop = False
        while not stop and self.iters < 100:
            stop = self.take_step(snr, ws)

    def take_step(self, snr_limit, ws):
        self.iters += 1
        guess = self.get_guess(ws)
        mode = scipy_fit(self.resid_lc, guess)
        print(f'found frequency: F{self.iters}', mode.f, mode.amp, mode.phase, mode.snr)
        self.resid = self.resid - sin(self.time, mode.amp.n, mode.f.n, mode.phase.n)
        self.resid_lc = lk.LightCurve(self.time, self.resid)
        self.iterations.append(Iteration(self.time, self.resid, mode))
        if guess.snr < snr_limit:
            return True
        else:
            mode.iter = self.iters
            self.nofp+= 3
            self.calc_statistics(mode)
            self.result.append(mode)
            self.result = _scipy_fit(self.lc, self.result)
            return False

    def get_guess(self, ws):
        pdg = self.resid_lc.to_periodogram(maximum_frequency=self.fmax, minimum_frequency=self.fmin)
        amp = np.amax(pdg.power)
        f = pdg.frequency[np.where(pdg.power == amp)[0][0]].value
        sur = np.where(abs(pdg.frequency.value - f ) <= ws/2)
        mean = np.mean(pdg.power[sur])

        snr = amp/mean
        return Mode(f, amp, 0, snr)

    def calc_statistics(self, mode):
        var = np.var(self.resid)
        mode.vr = var/self.var
        p = lk.LightCurve(times, flux).to_periodogram(maximum_frequency=self.fmax, minimum_frequency=self.fmin)
        pfunc = interp.interp1d(p.frequency, p.power, fill_value='extrapolate')
        mode.pr = integ.quad(pfunc, self.fmin, self.fmax, limit=5000, epsabs=1.49e-05, epsrel=1.49e-03)[0] / self.power_val
        mode.aic = (self.n * np.log(np.sum((self.flux-self.resid)**2)/self.n) +self.n + 2*self.nofp) /self.aic_val
        mode.bic = (self.n * np.log(np.sum((self.flux-self.resid)**2)/self.n) + self.nofp*np.log(self.n)) /self.bic_val
        mode.nofp = self.nofp
        return mode

def scipy_fit(lc, mode):
    f_guess = mode.f
    amp_guess = mode.amp

    arr = [amp_guess,  # amplitude
           f_guess,  # frequency
           0  # phase --> set to center
           ]
    limits = [[0, 0.95 * f_guess, 0], [2 * amp_guess, 1.05 * f_guess, 1]]
    try:
        popt, pcov = curve_fit(sin, lc.time, lc.flux, p0=arr, bounds=limits)
    except RuntimeError:
        try:
            popt, pcov = curve_fit(sin, lc.time, lc.flux, p0=arr, bounds=limits,
                                   maxfev=400 * (len(lc.time) + 1))
        except RuntimeError:
            return 0, 0, 0

    perr = np.sqrt(np.diag(pcov))
    return Mode(ufloat(popt[1], perr[1]), ufloat(popt[0], perr[0]), ufloat(popt[2], perr[2]), mode.snr)


def _scipy_fit(lc, modes):
    if len(lc.flux) <= 3 * len(modes):
        return 0

    arr = []
    for r in modes:
        arr.append(r.amp.n)
        arr.append(r.f.n)
        arr.append(r.phase.n)

    try:
        popt, pcov = curve_fit(sin_multiple, lc.time, lc.flux, p0=arr, method='trf')
    except RuntimeError:
        print(f"Failed to improve first {len(modes)} frequencies. Skipping fit improvement.")
        return 0

    perr = np.sqrt(np.diag(pcov))
    for r, vals in zip(modes,
                       [[ufloat(popt[i + j], perr[i + j]) for j in range(0, 3)] for i in range(0, len(popt), 3)]):
        r.amp = vals[0]
        r.f = vals[1]
        r.phase = vals[2]
    return modes

def sin_multiple(x, *params):
    y = np.zeros(len(x))
    for i in range(0, len(params), 3):
        y += sin(x, params[i], params[i + 1], params[i + 2])

    return y

def sin(x, amp, f, phase):
    return amp * np.sin(2. * np.pi * (f * x + phase))

lc_file = 'endurance/Final_Pulsation_LC.txt'
data = np.loadtxt(lc_file).T
times = data[0]
flux = data[1]

s = MiniSmurfs(times, flux)
s.run(5, 2)
arr = []
for m in s.result:
    print(m.f, m.amp, m.phase)
    arr.append(m.amp.n)
    arr.append(m.f.n)
    arr.append(m.phase.n)
mod_flux =sin_multiple(s.time, *arr)

fig, ax = plt.subplots(3,1, figsize = (12,8))
ax[0].plot(s.time, s.flux, 'ko', ms = 0.75)
ax[0].plot(s.time, mod_flux, 'r-')
pdg = s.lc.to_periodogram()
pdg2 = s.resid_lc.to_periodogram()
ax[1].plot(pdg.frequency, pdg.power, 'k-')
ax[1].plot(pdg2.frequency, pdg2.power, 'r-')
ax[2].plot([m.nofp for m in s.result], [m.aic for m in s.result], 'ko')
ax[2].plot([m.nofp for m in s.result], [m.bic for m in s.result], 'ro')
plt.show()