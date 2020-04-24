import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.integrate as integ
import scipy.interpolate as interp
import time
import pandas
import sys

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
        self.secondary = False
        self.primary = False
        self.defect = None
        self.phase0 = None
        self.sigma = None

    def component(self):
        if self.secondary:
            return 'Secondary'
        if self.primary:
            return 'Primary'

class Iteration:
    def __init__(self, time, flux, mode):
        self.time = time.copy()
        self.flux = flux.copy()
        self.lc = lk.LightCurve(self.time, self.flux)
        self.mode = mode



class MiniSmurfs:
    def __init__(self, time, flux, fmax=40, fmin=0.000001, model = 'sin'):
        models =['sin', 'pda', 'sin_analytic']
        if model not in models:
            sys.exit(f'{model} not an available option for model. Try one in {models}')
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
        self.nofp = 1
        self.var = np.var(flux)
        pdg = lk.LightCurve(time, flux).to_periodogram(maximum_frequency=fmax, minimum_frequency=self.fmin)
        pdg_func = interp.interp1d(pdg.frequency, pdg.power, fill_value='extrapolate')
        self.power_val = integ.quad(pdg_func, self.fmin, self.fmax, limit=5000, epsabs=1.49e-05, epsrel=1.49e-03)[0]
        self.n = len(time)
        self.aic_val = self.n * np.log(np.sum(self.flux**2)/self.n) +self.n+ 2*self.nofp
        self.bic_val =  self.n * np.log(np.sum(self.flux**2)/self.n)+ self.nofp*np.log(self.n)
        self.stop = 0
        self.model = model
        if self.model == "pda":
            self.have_primary = False
            self.have_secondary = True

    def run(self, snr=4, ws=2, ef=1):
        while self.stop < ef and self.iters < 200:
            if self.model == 'pda':
                self.take_step_pda(snr, ws)
            elif self.model == 'sin_analytic':
                self.take_step_sin_analytic(snr, ws)
            else:
                self.take_step_sin(snr, ws)
        if self.model == 'pda':
            self.result = _pda_fit2(self.lc, self.result, final=True)
        print(f'Stopping after {len(self.result)} significant frequencies.')

    def save_result(self):
        prs = [m.pr for m in self.result]
        vrs = [m.vr for m in self.result]
        aics = [m.aic for m in self.result]
        bics = [m.bic for m in self.result]
        nofps = [m.nofp for m in self.result]
        fs = [m.f.n for m in self.result]
        fserr = [m.f.s for m in self.result]
        amps = [m.amp.n for m in self.result]
        ampserr = [m.amp.s for m in self.result]
        phis = [m.phase.n for m in self.result]
        phiserrs = [m.phase.s for m in self.result]
        snrs = [m.snr for m in self.result]
        if self.model == "pda":
            phase0s = [m.phase0.n for m in self.result]
            sigmas = [m.sigma.n for m in self.result]
            defects = [m.defect.n for m in self.result]
            phase0serrs = [m.phase0.s for m in self.result]
            sigmaserrs = [m.sigma.s for m in self.result]
            defectserrs = [m.defect.s for m in self.result]
            components = [m.component() for m in self.result]
            df = pandas.DataFrame({'Frequency': fs, 'Frequency_error': fserr, 'Amplitude': amps, 'Amplitude_error': ampserr,
                                   'Phase': phis, 'Phase_error': phiserrs, 'SNR': snrs, 'phase0': phase0s, 'phase0_error': phase0serrs,
                                   'sigma': sigmas, 'sigma_error': sigmaserrs, 'defects': defects,  'defects_error': defectserrs,
                                   'Component': components,
                                   'Power_Reduction': prs,'Variance Reduction': vrs, 'AIC': aics, 'BIC': bics, 'Free_params': nofps})
            df.to_csv("minismurfs/result_pda_convergence_at_end.csv", index=False)
        else:
            df = pandas.DataFrame({'Frequency': fs, 'Frequency_error': fserr, 'Amplitude': amps, 'Amplitude_error': ampserr,
                                   'Phase': phis, 'Phase_error': phiserrs, 'SNR': snrs, 'Power_Reduction': prs,
                                   'Variance Reduction': vrs, 'AIC': aics, 'BIC': bics, 'Free_params': nofps})
            if self.model == 'sin_analytic':
                df.to_csv("minismurfs/result_analytic.csv", index=False)
            else:
                df.to_csv("minismurfs/result.csv", index=False)



    def take_step_sin(self, snr_limit, ws):
        self.iters += 1
        guess = self.get_guess(ws)
        mode = scipy_fit(self.resid_lc, guess)
        print(f'found frequency: F{self.iters}', mode.f, mode.amp, mode.phase, mode.snr)
        self.resid = self.resid - sin(self.time, mode.amp.n, mode.f.n, mode.phase.n)
        self.resid_lc = lk.LightCurve(self.time, self.resid)
        self.iterations.append(Iteration(self.time, self.resid, mode))
        if guess.snr < snr_limit:
            print('The last frequency was not significant')
            self.stop += 1
            return
        else:
            self.stop = 0
            mode.iter = self.iters
            self.nofp+= 3
            self.calc_statistics(mode)
            self.result.append(mode)
            self.result = _scipy_fit(self.lc, self.result)
            return

    def take_step_sin_analytic(self, snr_limit, ws):
        self.iters += 1
        guess = self.get_guess(ws)
        mode = scipy_fit_analytic(self.resid_lc, guess)
        print(f'found frequency: F{self.iters}', mode.f, mode.amp, mode.phase, mode.snr)
        self.resid = self.resid - sin(self.time, mode.amp.n, mode.f.n, mode.phase.n)
        self.resid_lc = lk.LightCurve(self.time, self.resid)
        self.iterations.append(Iteration(self.time, self.resid, mode))
        if guess.snr < snr_limit:
            print('The last frequency was not significant')
            self.stop += 1
            return
        else:
            self.stop = 0
            mode.iter = self.iters
            self.nofp+= 3
            self.calc_statistics(mode)
            self.result.append(mode)
            self.result = _scipy_fit_analytic(self.lc, self.result)
            return

    def take_step_pda(self, snr_limit, ws):
        self.iters += 1
        guess = self.get_guess(ws)
        mode_primary = pda_fit_primary(self.resid_lc, guess)
        mode_secondary = pda_fit_secondary(self.resid_lc, guess)
        mode, resid = self.decide_star(mode_primary, mode_secondary)
        print(f'found frequency: F{self.iters}', mode.f, mode.amp, mode.phase, mode.snr, mode.phase0, mode.sigma, mode.defect, mode.component())
        self.resid = resid
        self.resid_lc = lk.LightCurve(self.time, self.resid)
        self.iterations.append(Iteration(self.time, self.resid, mode))
        if guess.snr < snr_limit:
            print('The last frequency was not significant')
            self.stop += 1
            return
        else:
            self.stop = 0
            mode.iter = self.iters
            self.nofp+= 3
            self.calc_statistics(mode)
            self.result.append(mode)
            self.result = _pda_fit2(self.lc, self.result)
            return

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
        mode.vr = 1-var/self.var
        p = lk.LightCurve(self.time, self.resid).to_periodogram(maximum_frequency=self.fmax, minimum_frequency=self.fmin)
        pfunc = interp.interp1d(p.frequency, p.power, fill_value='extrapolate')
        mode.pr = 1-integ.quad(pfunc, self.fmin, self.fmax, limit=5000, epsabs=1.49e-05, epsrel=1.49e-05)[0] / self.power_val
        const = self.n * np.log(np.sum(self.resid**2)/self.n)
        mode.aic = (const +self.n + 2*self.nofp)
        mode.bic = (const + self.nofp*np.log(self.n))
        mode.nofp = self.nofp
        return mode

    def calc_statistics_pda(self, mode, resid):
        var = np.var(resid)
        mode.vr = 1-var/self.var
        p = lk.LightCurve(self.time, resid).to_periodogram(maximum_frequency=self.fmax, minimum_frequency=self.fmin)
        pfunc = interp.interp1d(p.frequency, p.power, fill_value='extrapolate')
        mode.pr = 1-integ.quad(pfunc, self.fmin, self.fmax, limit=5000, epsabs=1.49e-05, epsrel=1.49e-05)[0] / self.power_val
        add_nofp = 0
        if self.have_primary:
            add_nofp+=3
        if self.have_secondary:
            add_nofp += 3
        mode.aic = (self.n * np.log(np.sum(resid**2)/self.n) +self.n + 2*(self.nofp+add_nofp))
        mode.bic = (self.n * np.log(np.sum(resid**2)/self.n) + self.nofp*np.log(self.n+add_nofp))
        mode.nofp = self.nofp
        return mode

    def decide_star(self, prim, secon):
        r_prim = self.resid - pda(self.time, prim.amp.n, prim.f.n, prim.phase.n, prim.phase0.n, prim.sigma.n, prim.defect.n)
        r_sec = self.resid - pda(self.time, secon.amp.n, secon.f.n, secon.phase.n, secon.phase0.n, secon.sigma.n, secon.defect.n)
        prim = self.calc_statistics_pda(prim, r_prim)
        secon = self.calc_statistics_pda(secon, r_sec)
        decision = 0
        if prim.bic < secon.bic:
            decision -= 1
        else:
            decision =+ 1
        if prim.aic < secon.aic:
            decision -= 1
        else:
            decision =+ 1
        if decision > 0:
            secon.secondary = True
            self.have_secondary = True
            return secon, r_sec
        if prim.pr > secon.pr:
            decision -= 1
        else:
            decision =+ 1
        if prim.vr > secon.vr:
            decision -= 1
        else:
            decision =+ 1
        if decision < 0:
            prim.primary = True
            self.have_primary = True
            return prim, r_prim
        else:
            secon.secondary = True
            self.have_secondary = True
            return secon, r_sec



def scipy_fit(lc, mode):
    f_guess = mode.f
    amp_guess = mode.amp

    arr = [amp_guess,  # amplitude
           f_guess,  # frequency
           0.5  # phase --> set to center
           ]
    limits = [[0.5*amp_guess, 0.95 * f_guess, -100], [1.5 * amp_guess, 1.05 * f_guess, 100]]
    try:
        popt, pcov = curve_fit(sin, lc.time, lc.flux, p0=arr, bounds=limits, sigma = 0.000165*np.ones(len(lc.time)), absolute_sigma=True)
    except RuntimeError:
        try:
            popt, pcov = curve_fit(sin, lc.time, lc.flux, p0=arr, bounds=limits,
                                   maxfev=400 * (len(lc.time) + 1))
        except RuntimeError:
            return 0, 0, 0

    #print(popt, np.sqrt(np.diag(pcov)) )
    perr = np.sqrt(np.diag(pcov))
    return Mode(ufloat(popt[1], perr[1]), ufloat(popt[0], perr[0]), ufloat((popt[2]+100)%1, perr[2]), mode.snr)

def scipy_fit_analytic(lc, mode):
    f_guess = mode.f
    amp_guess = mode.amp

    arr = [amp_guess,  # amplitude
           f_guess,  # frequency
           0.5  # phase --> set to center
           ]
    limits = [[0.5*amp_guess, 0.95 * f_guess, -100], [1.5 * amp_guess, 1.05 * f_guess, 100]]
    try:
        popt, pcov = curve_fit(sin, lc.time, lc.flux, p0=arr, bounds=limits, sigma = 0.000165*np.ones(len(lc.time)), absolute_sigma=True, jac = jac)
    except RuntimeError:
        try:
            popt, pcov = curve_fit(sin, lc.time, lc.flux, p0=arr, bounds=limits,
                                   maxfev=400 * (len(lc.time) + 1))
        except RuntimeError:
            return 0, 0, 0

    #print(popt, np.sqrt(np.diag(pcov)) )
    perr = np.sqrt(np.diag(pcov))
    return Mode(ufloat(popt[1], perr[1]), ufloat(popt[0], perr[0]), ufloat((popt[2]+100)%1, perr[2]), mode.snr)


def jac(x, *fitParams):
    params = np.array(fitParams)
    As = params[::3]
    fs = params[1::3]
    offsets = params[2::3]
    phases = 2 * np.pi * (np.outer(x, fs) + offsets)
    sins = np.sin(phases)
    coss = np.cos(phases)

    jacA = sins
    jacF = x[:, np.newaxis] * coss * As * 2 * np.pi
    jacO = coss * As * 2 * np.pi

    jac = np.stack([jacA, jacF, jacO], 2).reshape((len(x), len(params)))
    return jac

def _scipy_fit(lc, modes):
    if len(lc.flux) <= 3 * len(modes):
        return 0

    arr = []
    limits = [[], []]
    for r in modes:
        arr.append(r.amp.n)
        limits[0].append(0.5*r.amp.n)
        limits[1].append(1.5*r.amp.n)
        arr.append(r.f.n)
        limits[0].append(0.9*r.f.n)
        limits[1].append(1.1*r.f.n)
        arr.append(r.phase.n)
        limits[0].append(-100)
        limits[1].append(100)

    try:
        popt, pcov = curve_fit(sin_multiple, lc.time, lc.flux, p0=arr, bounds = limits, sigma = 0.000165*np.ones(len(lc.time)), absolute_sigma=True)
    except RuntimeError:
        print(f"Failed to improve first {len(modes)} frequencies. Skipping fit improvement.")
        return 0
    perr = np.sqrt(np.diag(pcov))
    for r, vals in zip(modes,
                       [[ufloat(popt[i + j], perr[i + j]) for j in range(0, 3)] for i in range(0, len(popt), 3)]):
        r.amp = vals[0]
        r.f = vals[1]
        r.phase = (vals[2]+100)%1
    return modes

def _scipy_fit_analytic(lc, modes):
    if len(lc.flux) <= 3 * len(modes):
        return 0

    arr = []
    limits = [[], []]
    for r in modes:
        arr.append(r.amp.n)
        limits[0].append(0.5*r.amp.n)
        limits[1].append(1.5*r.amp.n)
        arr.append(r.f.n)
        limits[0].append(0.9*r.f.n)
        limits[1].append(1.1*r.f.n)
        arr.append(r.phase.n)
        limits[0].append(-100)
        limits[1].append(100)

    try:
        popt, pcov = curve_fit(sin_multiple, lc.time, lc.flux, p0=arr, bounds = limits, sigma = 0.000165*np.ones(len(lc.time)), absolute_sigma=True, jac = jac)
    except RuntimeError:
        print(f"Failed to improve first {len(modes)} frequencies. Skipping fit improvement.")
        return 0
    perr = np.sqrt(np.diag(pcov))
    for r, vals in zip(modes,
                       [[ufloat(popt[i + j], perr[i + j]) for j in range(0, 3)] for i in range(0, len(popt), 3)]):
        r.amp = vals[0]
        r.f = vals[1]
        r.phase = (vals[2]+100)%1
    return modes

def sin_multiple(x, *params):
    y = np.zeros(len(x))
    for i in range(0, len(params), 3):
        y += sin(x, params[i], params[i + 1], params[i + 2])

    return y

def sin(x, amp, f, phase):
    return amp * np.sin(2. * np.pi * (f * x + phase))

def pda(time, amp, frequency, phase, phase0, sigma, defect):
    period = 1.66987725
    phases = time%period/period
    return amp*(1- defect*np.exp(-(phases - phase0) ** 2 / (2 * sigma ** 2)))*np.sin(2. * np.pi * (frequency * time + phase))

def pda_multiple(time, *params):
    y = np.zeros(len(time))
    for i in range(3, len(params), 3):
        y += pda(time, params[i], params[i + 1], params[i + 2], params[0], params[1], params[2])
    return y

def pda_fit_primary(lc, mode):
    f_guess = mode.f
    amp_guess = mode.amp

    arr = [amp_guess,  # amplitude
           f_guess,  # frequency
           0., # phase --> set to center
           0.8, #phase0 guess for primary
           0.05, #sigma
           0.75, #defect
           ]
    limits = [[0, 0.95 * f_guess, -1000, 0.75, 0.025, 0.6], [2 * amp_guess, 1.05 * f_guess, 1000, 0.85, 0.075, 0.9]]
    try:
        popt, pcov = curve_fit(pda, lc.time, lc.flux, p0=arr, bounds=limits, sigma = 0.000165*np.ones(len(lc.time)), absolute_sigma=True, x_scale = "jac")
    except RuntimeError:
        try:
            popt, pcov = curve_fit(pda, lc.time, lc.flux, p0=arr, bounds=limits, sigma = 0.000165*np.ones(len(lc.time)), absolute_sigma=True,
                                   maxfev=400 * (len(lc.time) + 1), x_scale = "jac")
        except RuntimeError:
            return 0, 0, 0

    perr = np.sqrt(np.diag(pcov))
    rmod = Mode(ufloat(popt[1], perr[1]), ufloat(popt[0], perr[0]), ufloat((popt[2]+100)%1, perr[2]), mode.snr)
    rmod.phase0 = ufloat(popt[3], perr[3])
    rmod.sigma = ufloat(popt[4], perr[4])
    rmod.defect = ufloat(popt[5], perr[5])

    return rmod


def pda_fit_secondary(lc, mode):
    f_guess = mode.f
    amp_guess = mode.amp

    arr = [amp_guess,  # amplitude
           f_guess,  # frequency
           0.,  # phase --> set to center
           0.3,  # phase0 guess for primary
           0.05,  # sigma
           0.75,  # defect
           ]
    limits = [[0, 0.95 * f_guess, -100, 0.25, 0.025, 0.6], [2 * amp_guess, 1.05 * f_guess, 100, 0.35, 0.075, 0.9]]
    try:
        popt, pcov = curve_fit(pda, lc.time, lc.flux, p0=arr, bounds=limits, sigma = 0.000165*np.ones(len(lc.time)), absolute_sigma=True, x_scale = "jac")
    except RuntimeError:
        try:
            popt, pcov = curve_fit(pda, lc.time, lc.flux, p0=arr, bounds=limits, maxfev=400 * (len(lc.time) + 1), sigma = 0.000165*np.ones(len(lc.time)), absolute_sigma=True, x_scale = "jac")
        except RuntimeError:
            return 0, 0, 0

    perr = np.sqrt(np.diag(pcov))
    rmod = Mode(ufloat(popt[1], perr[1]), ufloat(popt[0], perr[0]), ufloat((popt[2] + 100) % 1, perr[2]), mode.snr)
    rmod.phase0 = ufloat(popt[3], perr[3])
    rmod.sigma = ufloat(popt[4], perr[4])
    rmod.defect = ufloat(popt[5], perr[5])

    return rmod


def get_arr_pda(modes, phase0):
    arr = [phase0, 0.05, 0.5]
    limits = [[phase0-0.05, 0.025, 0.4], [phase0+0.05, 0.075, 0.9]]
    for r in modes:
        arr.append(r.amp.n)
        limits[0].append(0.5*r.amp.n)
        limits[1].append(1.5*r.amp.n)
        arr.append(r.f.n)
        limits[0].append(0.9*r.f.n)
        limits[1].append(1.1*r.f.n)
        arr.append(r.phase.n)
        limits[0].append(-100)
        limits[1].append(100)
    return arr, limits

def _pda_plot(time, result):
    y = np.zeros(len(time))
    for r in result:
        y += pda(time, r.amp.n, r.f.n, r.phase.n, r.phase0.n, r.sigma.n, r.defect.n)
    return y

def _pda_fit(lc, modes):
    if len(lc.flux) <= 3 * len(modes) + 6:
        return 0

    modes_p = []
    modes_s = []
    for r in modes:
        if r.primary:
            modes_p.append(r)
        else:
            modes_s.append(r)

    if len(modes_p) > 0 :
        arr, limits = get_arr_pda(modes_p, 0.8)
        popt_p, pcov_p = curve_fit(pda_multiple, lc.time, lc.flux, p0=arr, bounds = limits, sigma = 0.000165*np.ones(len(lc.time)), absolute_sigma=True, x_scale = "jac")

        lc_new = lk.LightCurve(lc.time, lc.flux - pda_multiple(lc.time, *popt_p))
    else:
        lc_new = lc

    arr, limits = get_arr_pda(modes_s, 0.3)
    popt_s, pcov_s = curve_fit(pda_multiple, lc_new.time, lc_new.flux, p0=arr, bounds = limits, sigma = 0.000165*np.ones(len(lc.time)), absolute_sigma=True, x_scale = "jac")


    if len(modes_p) > 0 :
        perr_p = np.sqrt(np.diag(pcov_p))

        for r, vals in zip(modes_p,
                           [[ufloat(popt_p[i + j], perr_p[i + j]) for j in range(0, 3)] for i in range(3, len(popt_p), 3)]):
            r.amp = vals[0]
            r.f = vals[1]
            r.phase = (vals[2]+100)%1
            r.phase0 = ufloat(popt_p[0], perr_p[0])
            r.sigma = ufloat(popt_p[1], perr_p[1])
            r.defect = ufloat(popt_p[2], perr_p[2])



    perr_s = np.sqrt(np.diag(pcov_s))
    for r, vals in zip(modes_s,
                       [[ufloat(popt_s[i + j], perr_s[i + j]) for j in range(0, 3)] for i in range(3, len(popt_s), 3)]):
        r.amp = vals[0]
        r.f = vals[1]
        r.phase = (vals[2]+1000)%1
        r.phase0 = ufloat(popt_s[0], perr_s[0])
        r.sigma = ufloat(popt_s[1], perr_s[1])
        r.defect = ufloat(popt_s[2], perr_s[2])
    result = []
    for r in modes_s:
        result.append(r)
    for r in modes_p:
        result.append(r)
    return sorted(result, key=lambda x: x.iter)

def _pda_fit2(lc, modes, final = False):
    if len(lc.flux) <= 3 * len(modes) + 6:
        return 0

    modes_p = []
    modes_s = []
    for r in modes:
        if r.primary:
            modes_p.append(r)
        else:
            modes_s.append(r)

    if len(modes_p) == 0:
        arr, limits = get_arr_pda(modes_s, 0.3)
        popt_s, pcov_s = curve_fit(pda_multiple, lc.time, lc.flux, p0=arr, bounds=limits,
                                   sigma=0.000165 * np.ones(len(lc.time)), absolute_sigma=True, x_scale="jac")

        perr_s = np.sqrt(np.diag(pcov_s))
        for r, vals in zip(modes_s,
                           [[ufloat(popt_s[i + j], perr_s[i + j]) for j in range(0, 3)] for i in range(3, len(popt_s), 3)]):
            r.amp = vals[0]
            r.f = vals[1]
            r.phase = (vals[2] + 1000) % 1
            r.phase0 = ufloat(popt_s[0], perr_s[0])
            r.sigma = ufloat(popt_s[1], perr_s[1])
            r.defect = ufloat(popt_s[2], perr_s[2])
        result = []
        for r in modes_s:
            result.append(r)
        return sorted(result, key=lambda x: x.iter)


    else:
        arr_p, limits_p = get_arr_pda(modes_p, 0.8)
        arr_s, limits_s = get_arr_pda(modes_s, 0.3)
        lc_primary = lk.LightCurve(lc.time, lc.flux)
        popt_p, pcov_p = _pda_fit_step(lc_primary, arr_p, limits_p)
        lc_secondary = lk.LightCurve(lc.time, lc.flux)
        popt_s, pcov_s = _pda_fit_step(lc_secondary, arr_s, limits_s)
        #print('Starting improvement')
        converged = False
        num =1
        max = 5
        if final:
            max = 10000
        while not converged and num <= max:
            popt_p_old = np.array(popt_p.copy())
            popt_s_old = np.array(popt_s.copy())
            lc_primary.flux = lc.flux - pda_multiple(lc.time, *popt_s)
            popt_p, pcov_p = _pda_fit_step(lc_primary, popt_p, limits_p)


            lc_secondary.flux = lc.flux - pda_multiple(lc.time, *popt_p)
            popt_s, pcov_s = _pda_fit_step(lc_secondary, arr_s, limits_s)
            dif_p = abs(popt_p_old - np.array(popt_p))
            dif_s = abs(popt_s_old - np.array(popt_s))
            mean_p = np.mean(dif_p)
            mean_s = np.mean(dif_s)
            max_p = np.max(dif_p)
            max_s = np.max(dif_s)
            if np.max([max_s, max_p]) < 10**-6 and  np.max([mean_s, mean_p]) < 10**-7:
                converged = True
                print(f'Mean change it {num}:  Primary: ' + r'{:.3e}'.format(mean_p), 'Secondary:' + r'{:.3e}'.format(mean_s))
                print(f'Max change it {num}:  Primary: ' + r'{:.3e}'.format(max_p), 'Secondary:' + r'{:.3e}'.format(max_s))
            else:
                num += 1

    if len(modes_p) > 0 :
        perr_p = np.sqrt(np.diag(pcov_p))

        for r, vals in zip(modes_p,
                           [[ufloat(popt_p[i + j], perr_p[i + j]) for j in range(0, 3)] for i in range(3, len(popt_p), 3)]):
            r.amp = vals[0]
            r.f = vals[1]
            r.phase = (vals[2]+100)%1
            r.phase0 = ufloat(popt_p[0], perr_p[0])
            r.sigma = ufloat(popt_p[1], perr_p[1])
            r.defect = ufloat(popt_p[2], perr_p[2])


    perr_s = np.sqrt(np.diag(pcov_s))
    for r, vals in zip(modes_s,
                       [[ufloat(popt_s[i + j], perr_s[i + j]) for j in range(0, 3)] for i in range(3, len(popt_s), 3)]):
        r.amp = vals[0]
        r.f = vals[1]
        r.phase = (vals[2]+1000)%1
        r.phase0 = ufloat(popt_s[0], perr_s[0])
        r.sigma = ufloat(popt_s[1], perr_s[1])
        r.defect = ufloat(popt_s[2], perr_s[2])
    result = []
    for r in modes_s:
        result.append(r)
    for r in modes_p:
        result.append(r)
    return sorted(result, key=lambda x: x.iter)

def _pda_fit_step(lc, arr, limits):

    popt, pcov = curve_fit(pda_multiple, lc.time, lc.flux, p0=arr, bounds = limits, sigma = 0.000165*np.ones(len(lc.time)), absolute_sigma=True, x_scale = "jac")

    return popt, pcov


def load_result(file):
    return pandas.read_csv(file)


# start = time.time()
# lc_file = 'endurance/Final_Pulsation_LC.txt'
# data = np.loadtxt(lc_file).T
# times = data[0]
# flux = data[1]
#
# s = MiniSmurfs(times, flux)
# s.run(4, 2, ef = 5)
# s.save_result()
# arr = []
# for m in s.result:
#     print(m.f, m.amp, m.phase)
#     arr.append(m.amp.n)
#     arr.append(m.f.n)
#     arr.append(m.phase.n)
# mod_flux =sin_multiple(s.time, *arr)
#
# fig, ax = plt.subplots(4,1, figsize = (12,8))
# ax[0].plot(s.time, s.flux, 'ko', ms = 0.75)
# ax[0].plot(s.time, mod_flux, 'r-')
# pdg = s.lc.to_periodogram()
# pdg2 = s.resid_lc.to_periodogram()
# ax[1].plot(pdg.frequency, pdg.power, 'k-')
# ax[1].plot(pdg2.frequency, pdg2.power, 'r-')
# aic =  [m.aic for m in s.result]
# bic =   [m.bic for m in s.result]
# ax[2].plot([m.nofp for m in s.result], [m.aic for m in s.result], 'ko')
# ax[2].plot([m.nofp for m in s.result], [m.bic for m in s.result], 'ro')
# ax[3].plot([m.nofp for m in s.result], [m.pr for m in s.result], 'go')
# ax[3].plot([m.nofp for m in s.result], [m.vr for m in s.result], 'bo')
# print(s.model)
# print(time.time()-start)
# plt.show()
#
