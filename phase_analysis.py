import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.optimize import curve_fit
from uncertainties import ufloat
from tqdm import tqdm
import scipy.integrate as integ
import scipy.interpolate as interp
import os
from pandas import read_csv
from io import StringIO
from uncertainties import ufloat_fromstr

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
            return (0,0,0)

    perr = np.sqrt(np.diag(pcov))
    return ufloat(popt[0], perr[0]), ufloat(popt[1], perr[0]), ufloat(popt[2], perr[0])


def _scipy_fit(lc, modes):
    """
    stolen from smurfs
    """

    if len(lc.flux) <= 3 * len(modes):
        return 0

    result = []
    single_lc = lk.LightCurve(lc.time, lc.flux)
    for m in modes:
        amp, f, phi = scipy_fit(single_lc, m)
        result.append(mode(f.n, amp.n, phi.n, 0))
        single_lc.flux = single_lc.flux - sin(single_lc.time, amp.n, f.n, phi.n)

    arr = []
    boundaries = [[], []]
    for r in result:
        arr.append(r.amp)
        arr.append(r.f)
        arr.append(r.phase)


    try:
        popt, pcov = curve_fit(sin_multiple, lc.time, lc.flux, p0=arr, method= 'trf')
    except RuntimeError:
        print(f"Failed to improve first {len(modes)} frequencies. Skipping fit improvement.")
        return 0

    perr = np.sqrt(np.diag(pcov))

    Nexp = sin_multiple(lc.time, *popt)
    r = np.array(lc.flux) - Nexp
    chisq = np.sum(r ** 2)
    df = len(lc.time) - 3*len(modes)
    print("chisq =", chisq, "df =", df, 'chisq/DoF', chisq/df)


    for r, vals in zip(result,
                       [[ufloat(popt[i + j], perr[i + j]) for j in range(0, 3)] for i in range(0, len(popt), 3)]):
        r.amp = vals[0]
        r.f = vals[1]
        r.phase = vals[2]
    return result, np.sqrt(2/len(lc.time))*0.000165

def sin_multiple(x: np.ndarray, *params) -> np.ndarray:
    """
    Multiple sinuses summed up

    :param x: Time axis
    :param params: Params, see *sin* for signature
    """
    y = np.zeros(len(x))
    for i in range(0, len(params), 3):
        y += sin(x, params[i], params[i + 1], params[i + 2])

    return y

def sin(x: np.ndarray, amp: float, f: float, phase: float) -> np.ndarray:
    """
    Sinus function, used for fitting.

    :param x: Time axis, days
    :param amp: amplitude, mag
    :param f: frequency, c/d
    :param phase: phase, normed to 1
    """
    return amp * np.sin(2. * np.pi * (f * x + phase))

class mode():
    def __init__(self, f, amp, phase, snr):
        self.f = f
        self.amp = amp
        self.phase = phase
        self.snr = snr
        self.phase_bins = []
        self.phase_amps = []
        self.phase_amps_ = []
        self.phase_phases = []
        self.phase_phases_ = []
        self.time_bins = []
        self.time_amps = []
        self.time_amps_ = []
        self.time_phases = []
        self.time_phases_ = []

def amplitude_in_phase(time, flux, modes, period = 1.66987725, phase = 0.5, delta_phase = 0.05, n = 1):
    p = time%(n*period)/period
    start = (n+phase - delta_phase)%n
    end = (phase + delta_phase)%n
    print(start, end)
    if end > start:
        ind = np.where(np.logical_and(p >= start, p <= end))
    else:
        ind = np.where(np.logical_or(p >= start, p <= end))
    lc = lk.LightCurve(time[ind], flux[ind])
    result, amp_err = _scipy_fit(lc, modes)
    if result == 0:
        return modes
    for mode, mode_r in zip(modes, result):
        mode.phase_bins.append(phase)
        mode.phase_amps.append(mode_r.amp.nominal_value)
        mode.phase_amps_.append(mode_r.amp.std_dev + amp_err)
        mode.phase_phases.append(mode_r.phase.nominal_value)
        mode.phase_phases_.append(mode_r.phase.std_dev)
    return modes



def test_multiple_modes():
    lc_file = 'endurance/Final_Pulsation_LC.txt'
    data = np.loadtxt(lc_file).T
    times = data[0]
    flux = data[1]
    modes = [mode(11.0704, 0.001628, 0.8671, 10), mode(11.624, 0.001647, 0.603, 10), mode(12.7956, 0.001494, 0.281, 10)]
    modes_add1 = [mode(20.122, 0.000901, 0.7524, 10), mode(18.925287, 0.000830, 0.0615, 10), mode(25.635161, 0.000575, 0.580, 10)]
    modes_add2 = [mode(12.821734, 0.000737,0.000737, 10 ),
                 mode(10.427299, 0.000679, 0.517, 10)]
    mode_addb = [mode(1.204336, 0.000298 ,0.023, 10),mode(2.40155, 0.000175 ,0.161, 10) ]

    period = 1.66987725
    import phase_dependent_amplitude as pda
    flux2 = np.zeros(len(times))
    for m in modes:
        flux2 += pda.pda(times, 1.075*m.amp, m.f, m.phase, sigma=0.02, defect = 0.8,phase0=0.295)
    for m in modes_add1:
        flux2 += pda.pda(times, 1.075*m.amp, m.f, m.phase, sigma=0.02, defect = 0.8,phase0=0.795)
    for m in modes_add2:
        flux2 += pda.pda(times, 1.075*m.amp, m.f, m.phase, sigma=0.02, defect = 0.8,phase0=0.295)
    flux2 += + np.random.normal(0, 0.000165, len(times))
    fig, ax = plt.subplots(2,1)
    ax[0].plot(times%period/period, flux2, 'ko', ms = 0.75)
    ax[0].plot(times%period/period, flux, 'ro', ms = 0.75)
    pdg = lk.LightCurve(times, flux2).to_periodogram()
    pdg2 = lk.LightCurve(times, flux).to_periodogram()
    ax[1].plot(pdg2.frequency, pdg2.power, 'r-')
    ax[1].plot(pdg.frequency, pdg.power, 'k-')
    ax[1].set_xlim(0, 30)
    plt.show()
    plt.close()

    modes2 = [mode(11.0704, 0.001628, 0.8671, 10), mode(11.624, 0.001647, 0.603, 10), mode(12.7956, 0.001494, 0.281, 10)]


    fig, ax = plt.subplots(3, 1, figsize=(10,7))

    for x in np.linspace(0, 1, 100):
        modes = amplitude_in_phase(times, flux, modes, phase = x, n = 1, delta_phase=0.05)
        modes2 = amplitude_in_phase(times, flux2, modes2, phase = x, n = 1, delta_phase=0.05)


    for i in range(3):
        ax[i].errorbar(modes[i].phase_bins, modes[i].phase_amps, yerr= modes[i].phase_amps_,  fmt = 'ko')
        ax[i].errorbar(modes2[i].phase_bins, modes2[i].phase_amps, yerr= modes2[i].phase_amps_,  fmt = 'ro')

    plt.show()

def pda(time, amp, frequency, phase, phase0, sigma, defect):
    period = 1.66987725
    phases = time%period/period
    return amp*(1- defect*np.exp(-(phases - phase0) ** 2 / (2 * sigma ** 2)))*np.sin(2. * np.pi * (frequency * time + phase))

def pda_multiple(time, *params):
    y = np.zeros(len(time))
    for i in range(3, len(params), 3):
        y += pda(time, params[i], params[i + 1], params[i + 2], params[0], params[1], params[2])
    return y

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

'''
modes = [mode(11.0704, 0.001628, 0.3, 10), mode(11.624, 0.001647, 0.6, 10), mode(12.7956, 0.001494, 0.14, 10), mode(12.821734, 0.000737, 0.46, 10),
              mode(10.427299, 0.000679, 0.98, 10), mode(20.122, 0.000901, 0.65, 10), mode(18.925287, 0.000830, 0.87, 10),
              mode(25.635161, 0.000575, 0.78, 10)]


lc_file = 'endurance/Final_Pulsation_LC.txt'
data = np.loadtxt(lc_file).T
times = data[0]
arr = [0.3, 0.05, 0.5]
for p in modes:
    arr.append(p.amp)
    arr.append(p.f)
    arr.append(p.phase)

flux_thingie = pda_multiple(times, *arr)+ np.random.normal(0, 0.000165, len(times))
plt.plot(times, flux_thingie, 'ko', ms = 0.75)
plt.show()
np.savetxt('test_lc.txt', np.array([times, flux_thingie]).T)
import sys
sys.exit()

'''

def fat_weird_thingie():
    lc_file = 'endurance/Final_Pulsation_LC.txt'
    data = np.loadtxt(lc_file).T
    times = data[0]
    flux = data[1]

    settings, statistics, results = load_results('endurance/Final_Pulsation_LC/data/result.csv')

    fs = results['frequency'].values
    amps = results['amp'].values
    snrs = results['snr'].values
    phis = results['phase'].values
    modes = []
    for i in range(len(fs)):
        if amps[i] >= 0.00005 and fs[i] > 8 and snrs[i]>4:
            modes.append(mode(fs[i].n, amps[i].n, phis[i].n, snrs[i]))

    #modes = [mode(11.0704, 0.001628, 0, 10), mode(11.624, 0.001647, 0, 10), mode(12.7956, 0.001494, 0, 10), mode(12.821734, 0.000737, 0., 10),
    #              mode(10.427299, 0.000679, 0, 10), mode(20.122, 0.000901, 0, 10), mode(18.925287, 0.000830, 0, 10),
    #              mode(25.635161, 0.000575, 0, 10)]


    fig, ax = plt.subplots(2,1,figsize = (12,6))
    #ax.plot(times, flux2, 'ko', ms=0.75)
    ax[0].plot(times, flux, 'ko', ms=0.75)
    result_p = []
    result_s = []
    flux_new = flux.copy()
    aic_old = 0
    for m in modes:
        print(m.f)
        pdg = lk.LightCurve(times, flux_new).to_periodogram(maximum_frequency = 40)
        test = interp.interp1d(pdg.frequency, pdg.power, fill_value='extrapolate')
        pr = integ.quad(test, 0, 40, limit = 5000, epsabs=1.49e-03, epsrel=1.49e-03)[0]
        limits = [[0.5*m.amp, 0.9*m.f, 0, 0.2, 0.0, 0.4], [1.5*m.amp, 1.1*m.f, 1000, 0.4, 0.1, 0.8]]
        try:
            popt, pcov = curve_fit(pda, times, flux_new, p0=[m.amp, m.f, m.phase, 0.3, 0.05, 0.5], bounds = limits)
            aic_s = np.log(sum((flux_new -  pda(times, *popt))**2))
            pdg_s = lk.LightCurve(times, flux_new -  pda(times, *popt)).to_periodogram(maximum_frequency = 40)
            test1 = interp.interp1d(pdg_s.frequency, pdg_s.power, fill_value='extrapolate')
            pr_s = integ.quad(test1, 0, 40, limit = 5000, epsabs=1.49e-03, epsrel=1.49e-03)[0]/pr
        except RuntimeError:
            pr_s = 1
            aic_s = 0
        try:
            limits = [[0.5*m.amp, 0.9*m.f, 0, 0.7, 0.0, 0.4], [1.5*m.amp, 1.1*m.f, 1000, 0.9, 0.1, 0.8]]
            popt2, pcov2 = curve_fit(pda, times, flux_new, p0=[m.amp, m.f, m.phase, 0.8, 0.05, 0.5], bounds = limits)
            aic_p = np.log(sum((flux_new -  pda(times, *popt2))**2))
            pdg_p = lk.LightCurve(times, flux_new -  pda(times, *popt2)).to_periodogram(maximum_frequency = 40)
            test2 = interp.interp1d(pdg_p.frequency, pdg_p.power, fill_value='extrapolate')
            pr_p = integ.quad(test2, 0, 40, limit = 5000, epsabs=1.49e-03, epsrel=1.49e-03)[0]/pr
        except RuntimeError:
            pr_p = 1
            aic_p = 0
        print('AIC s/p', aic_s, aic_p, 'PR s/p', pr_s, pr_p)
        if np.min([aic_s, aic_p]) < aic_old:
            if aic_s < aic_p:
                result_s.append(popt)
                flux_new = flux_new -  pda(times, *popt)
                aic_old = aic_s
            else:
                result_p.append(popt2)
                flux_new = flux_new -  pda(times, *popt2)
                aic_old = aic_p
    print('Secondary')
    for p in result_s:
        print('F:', p[1], 'A:', p[0], 'phi:', p[2], 'phi_0:', p[3], 'sigma:', p[4], 'defect:', p[5])
    print('##########################')
    print('Primary')
    for p in result_p:
        print('F:', p[1], 'A:', p[0], 'phi:', p[2], 'phi_0:', p[3], 'sigma:', p[4], 'defect:', p[5])
    print('##########################')
    arr = [0.3, 0.05, 0.5]
    limits = [[0.2, 0.01, 0], [0.4, 0.1, 1]]
    for p in result_s:
        arr.append(p[0])
        limits[0].append(0.5*p[0])
        limits[1].append(1.5*p[0])
        arr.append(p[1])
        limits[0].append(0.9*p[1])
        limits[1].append(1.1*p[1])
        arr.append(p[2])
        limits[0].append(-np.inf)
        limits[1].append(np.inf)
    popt_s, pcov_s = curve_fit(pda_multiple, times, flux, p0=arr, bounds = limits)
    print('Secondary')
    print('phi_0:', popt_s[0], 'sigma:', popt_s[1], 'defect:', popt_s[2])
    freqs_secondary = []
    for i in range(3, len(popt_s), 3):
        freqs_secondary.append(popt_s[i+1])
        print('F:', popt_s[i+1], 'A:', popt_s[i], 'phi:', popt_s[i+2])


    mod_flux_s = pda_multiple(times, *popt_s)
    new_flux = flux - mod_flux_s
    print('##########################')
    arr = [0.8, 0.05, 0.5]
    limits = [[0.7, 0.01, 0], [0.9, 0.1, 1]]
    for p in result_p:
        arr.append(p[0])
        limits[0].append(0.5*p[0])
        limits[1].append(1.5*p[0])
        arr.append(p[1])
        limits[0].append(0.9*p[1])
        limits[1].append(1.1*p[1])
        arr.append(p[2])
        limits[0].append(-np.inf)
        limits[1].append(np.inf)
    print(arr)
    print(limits)
    popt_p, pcov_p = curve_fit(pda_multiple, times, new_flux, p0=arr, bounds = limits)
    print('Primary')
    print('phi_0:', popt_p[0], 'sigma:', popt_p[1], 'defect:', popt_p[2])
    freqs_primary = []
    for i in range(3, len(popt_p), 3):
        freqs_primary.append(popt_p[i+1])
        print('F:', popt_p[i+1], 'A:', popt_p[i], 'phi:', popt_p[i+2])
    mod_flux_p = pda_multiple(times, *popt_p)
    new_flux = new_flux - mod_flux_p
    print('##########################')


    ax[0].plot(times, mod_flux_s + mod_flux_p, 'ro', ms = 0.5)
    pdg = lk.LightCurve(times, flux).to_periodogram()
    res_pdg = lk.LightCurve(times, new_flux).to_periodogram()
    mod_s_pdg = lk.LightCurve(times, mod_flux_s).to_periodogram()
    mod_p_pdg = lk.LightCurve(times, mod_flux_p).to_periodogram()
    ax[1].plot(pdg.frequency, pdg.power, 'k', lw = 1)
    ax[1].plot(mod_s_pdg.frequency, mod_s_pdg.power, 'g--', lw = 0.5)
    ax[1].plot(mod_p_pdg.frequency, mod_p_pdg.power, 'm--', lw = 0.5)
    ax[1].plot(res_pdg.frequency, res_pdg.power, 'r', lw = 0.8)
    plt.show()
    plt.close()

    period = 1.66987725
    fig, ax = plt.subplots(figsize = (12,6))
    f = [m.f for m in modes]
    ax.plot(np.array(f)%(1/period), f, 'ko', ms = 5)
    ax.plot(np.array(freqs_primary)%(1/period), freqs_primary, 'rx', ms = 5)
    ax.plot(np.array(freqs_secondary)%(1/period), freqs_secondary, 'bx', ms = 5)
    plt.show()



'''
lc_file = 'endurance/Final_Pulsation_LC.txt'
data = np.loadtxt(lc_file).T
times = data[0]
flux = data[1]
result = []
flux_new = flux.copy()
count = 0
stop = False
while count < 20:
    pdg = lk.LightCurve(times, flux_new).to_periodogram(minimum_frequency = 8, maximum_frequency = 40)
    amp = np.amax(pdg.power)
    f = pdg.frequency[np.where(pdg.power == amp)[0][0]].value
    print(f, amp)
    limits = [[0.5 * amp, 0.9 * f, 0, 0.2, 0.0, 0.4], [1.5 * amp, 1.1 * f, 1000, 0.4, 0.1, 0.8]]
    popt, pcov = curve_fit(pda, times, flux_new, p0=[amp, f, 0, 0.3, 0.05, 0.5], bounds = limits)
    result.append(popt)
    flux_new = flux_new -  pda(times, *popt)
    count += 1

for p in result:
    print('F:', p[1], 'A:', p[0], 'phi:', p[2], 'phi_0:', p[3], 'sigma:', p[4], 'defect:', p[5])
fig, ax = plt.subplots(2,1, figsize = (12,6))
ax[0].plot(times, flux, 'ko', ms = 0.75)
ax[0].plot(times, flux-flux_new, 'r-')
ax[1].plot(pdg.frequency, lk.LightCurve(times, flux).to_periodogram(minimum_frequency = 8, maximum_frequency = 40).power, 'k-')
ax[1].plot(pdg.frequency, pdg.power, 'r-')
plt.show()







lc_file = 'endurance/Final_Pulsation_LC.txt'
data = np.loadtxt(lc_file).T
times = data[0]
flux = data[1]

fs = [11.0703910326469 ,11.624895962024176 ,12.79520550704062, 20.122516352277714 ,18.924907836258388,]#12.822593234768355 ,10.427356235437156 ,25.63617123640961]
amps = [0.0016715568284938562, 0.0016435120422416158, 0.0014987863342293011,0.0009152175177519561, 0.000851788328761722, ]#0.0007491355996644594,0.0006890738769746643,0.0005867220290295697]

modes = []
for f, a in zip(fs, amps):
    modes.append(mode(f, a, 0, 0))

for p in np.linspace(0, 1, 100):
    modes = amplitude_in_phase(times, flux, modes, phase = p)

fig, ax = plt.subplots(4, figsize = (12,7))
for i in range(3):
    ax[i].errorbar(modes[i].phase_bins, modes[i].phase_amps, yerr=modes[i].phase_amps_, ls = '', marker = 'o', color = 'k')
period = 1.66987725
ax[3].plot(times%period/period, flux, 'ko', ms = 0.75)
plt.show()
'''
