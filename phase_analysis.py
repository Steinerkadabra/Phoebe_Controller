import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.optimize import curve_fit
from uncertainties import ufloat

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

def _scipy_fit(lc, result):
    """
    Performs a combination fit for all found frequencies using *scipy.optimize.curve_fit*.

    :param result: List of found frequencies
    :return: List of improved frequencies
    """
    arr = []
    boundaries = [[], []]
    for r in result:
        arr.append(r.amp)
        arr.append(r.f)
        arr.append(r.phase)

    try:
        popt, pcov = curve_fit(sin_multiple, lc.time, lc.flux, p0=arr)
    except RuntimeError:
        print(f"Failed to improve first {len(result)} frequencies. Skipping fit improvement.")
        return result
    perr = np.sqrt(np.diag(pcov))
    for r, vals in zip(result,
                       [[ufloat(popt[i + j], perr[i + j]) for j in range(0, 3)] for i in range(0, len(popt), 3)]):
        r.amp = vals[0]
        r.f = vals[1]
        r.phase = vals[2]
        #print(r.amp, r.f, r.phase)
    return result

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


time, flux = load_lc(lc_file = 'endurance_data/Removed_Binary_plus_savgol_from_original.txt', parts_of_lc=False, plot = False)
phase =  time%(1/0.5988495842998753)*0.5988495842998753

time_f, flux_f = load_lc(lc_file = 'endurance_data/Removed_Pulsations_from_first_run.txt', parts_of_lc=False, plot = False)
phase_f =  time_f%(1/0.5988495842998753)*0.5988495842998753

amp1 = []
amp2 = []
amp3 = []
amp4 = []
amp5 = []
amp1_ = []
amp2_= []
amp3_ = []
amp4_ = []
amp5_ = []

phase1 = []
phase2 = []
phase3 = []
phase4 = []
phase5 = []
phase1_ = []
phase2_= []
phase3_ = []
phase4_ = []
phase5_ = []

points = np.linspace(0.00, 1.00, 100)
for i in points:
    start = (1+i - 0.05)%1
    end = (i + 0.05)%1
    if end > start:
        ind = np.where(np.logical_and(phase >= start, phase <= end))
    else:
        ind = np.where(np.logical_or(phase >= start, phase <= end))
    print(start, end)
    #plt.plot(time[ind], flux[ind], 'ko', markersize = 0.75)
    lc = lk.LightCurve(time[ind], flux[ind])
    result = [mode(11.0704, 0.001633, 0.867, 10), mode(11.624, 0.001643, 0.6029, 10), mode(12.7956, 0.001495, 0.2818, 10), mode(20.1225, 0.000901, 0.7518, 10), mode(18.9252, 0.000829, 0.0622, 10)]
    result = _scipy_fit(lc, result)

    amp1.append(result[0].amp.nominal_value/0.001633)
    amp2.append(result[1].amp.nominal_value/0.001643)
    amp3.append(result[2].amp.nominal_value/0.001495)
    amp4.append(result[3].amp.nominal_value/0.000901)
    amp5.append(result[4].amp.nominal_value/0.000829)
    amp1_.append(result[0].amp.std_dev/0.001633)
    amp2_.append(result[1].amp.std_dev/0.001643)
    amp3_.append(result[2].amp.std_dev/0.001495)
    amp4_.append(result[3].amp.std_dev/0.000901)
    amp5_.append(result[4].amp.std_dev/0.000829)

    phase1.append(result[0].phase.nominal_value)
    phase2.append(result[1].phase.nominal_value)
    phase3.append(result[2].phase.nominal_value)
    phase4.append(result[3].phase.nominal_value)
    phase5.append(result[4].phase.nominal_value)
    phase1_.append(result[0].phase.std_dev)
    phase2_.append(result[1].phase.std_dev)
    phase3_.append(result[2].phase.std_dev)
    phase4_.append(result[3].phase.std_dev)
    phase5_.append(result[4].phase.std_dev)

result = [mode(11.0704, 0.001633, 0.867, 10), mode(11.624, 0.001643, 0.6029, 10), mode(12.7956, 0.001495, 0.2818, 10),
          mode(20.1225, 0.000901, 0.7518, 10), mode(18.9252, 0.000829, 0.0622, 10)]

fig, ax = plt.subplots(5, 1, figsize=(7,14))

ax[0].errorbar(points, amp1, yerr= amp1_,  fmt = 'ko')
ax[1].errorbar(points, amp2, yerr= amp2_, fmt = 'ko')
ax[2].errorbar(points, amp3, yerr= amp3_, fmt = 'ko')

ax[4].errorbar(phase_f, flux_f, fmt = 'ko', ms = 0.75)
ax[3].errorbar(points, amp5, yerr= amp5_, fmt = 'ko-')
for i in range(4):
    ax[i].set_xlabel('Phase')
    ax[i].set_ylabel('$\Delta$A/A')
    x_lim = ax[i].set_xlim()
    y_lim = ax[i].set_ylim()
    ax[i].text(x_lim[0]+0.55*(x_lim[1]-x_lim[0]), y_lim[0]+0.1*(y_lim[1]-y_lim[0]), f' F_{i}: {result[i].f} Amplitude: {result[i].amp} ')
ax[4].set_xlabel('Phase')
ax[4].set_ylabel('flux')
plt.tight_layout(w_pad=0, h_pad=0)
plt.savefig('Phase_dependence_of_amplitude.png')




