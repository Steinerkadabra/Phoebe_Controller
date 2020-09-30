import numpy as np
import phase_analysis as pha
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.optimize import curve_fit
import scipy.interpolate as interp
import scipy.integrate as integ
from tabulate import tabulate
import time as timey
from tqdm import tqdm


import mini_smurfs as ms
freq = 1/1.66987725
period = 1.66987725
data = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
time = data[0]
flux = data[1]

r = ms.load_result('minismurfs/result.csv')
r = r.sort_values(by=['Amplitude'], ascending=False)
fs = np.array(r['Frequency'])
amps = np.array(r['Amplitude'])
phis = np.array(r['Phase'])
print(phis)
num_modes = 5
flux_mod = np.zeros(len(time))
# multiplet_fs = [11.0704, 12.86744, 12.26757, 13.46483,14.06425, 10.47140, 9.27411, 14.66291, 15.26153, 11.67081]
# fit_fs = [11.0704, 12.86744]
# fit_amps = [0.001671, 0.000226]

# multiplet_fs = [11.624913, 12.822585, 10.427359, 11.025846,13.42164, 14.02060, 12.22351]
# fit_fs = [11.624913, 12.822585,10.427359, 11.025846]
# fit_amps = [0.001644, 0.000751, 0.000689, 0.000331]
# fit_fs = [11.624913]
# fit_amps = [0.001644]

# multiplet_fs = [12.795260, 11.59805, 10.99512, 9.80338]
# fit_fs = [12.795260]
# fit_amps = [0.001450]

multiplet_fs = [11.0704, 12.86744, 12.26757, 13.46483,14.06425, 10.47140, 9.27411, 14.66291, 15.26153, 11.67081, 11.624913, 12.822585, 10.427359, 11.025846,13.42164, 14.02060, 12.22351, 12.795260, 11.59805, 10.99512, 9.80338]
fit_fs = [11.0704, 11.624913, 12.795260]
fit_amps = [0.001671, 0.001644, 0.001450]


modes = []
for f, a in zip(fit_fs, fit_amps):
    modes.append(pha.mode(f, a, 0.5, 0))

for f, a, phi in zip(fs, amps, phis):
    include = True
    for mf in multiplet_fs:
        if abs(f-mf) < 0.0001:
            include = False
    if include:
        flux_mod += pha.sin(time, a, f, phi)


removed_flux = flux-flux_mod

fig, ax = plt.subplots(3,1)
ax[0].plot(time, flux, 'ko', ms = 0.75)
ax[0].plot(time, flux_mod, 'r-')
ax[1].plot(time%period, removed_flux, 'ko', ms = 0.75)
pdg = lk.LightCurve(time, removed_flux).to_periodogram()
ax[2].plot(pdg.frequency, pdg.power, 'k-')
plt.show()
plt.close()

segments_time = []
segments_amp  = []
segments_phi = []
segments_amp_err = []
segments_f = []

lc = lk.LightCurve(time, removed_flux)
for i in tqdm(np.linspace(time[0], time[-1], int((time[-1]- time[0])/0.1))[1:-1]):
    start = i -0.2
    end = i+ 0.2
    ind = np.where(np.logical_and(time >= start, time <= end))
    if len(time[ind]) < 200:
        continue
    lc = lk.LightCurve(time[ind], (removed_flux)[ind])

    try:
        result, amp_err = pha._scipy_fit_fixed_f(lc, modes)
        segments_time.append(i)
        segments_amp.append([x.amp.n for x in result])
        segments_phi.append([x.phase.n%1 for x in result])
        segments_f.append([x.f.n for x in result])
        segments_amp_err.append(amp_err)
        # plt.plot(lc.time, lc.flux, 'ko', ms = 1)
        # plt.plot(lc.time, pha.sin(lc.time, result[0].amp.n, result[0].f.n, result[0].phase.n), 'r-')
        # plt.show()

    except TypeError:
        # plt.plot(lc.time, lc.flux)
        # plt.show()
        pass
segments_amp = np.array(segments_amp).T
segments_phi = np.array(segments_phi).T
segments_f = np.array(segments_f).T
print(segments_amp)

number = len(segments_amp)
fig, ax = plt.subplots(number, 1)
if number > 1:
    for k in range(len(segments_amp)):
        ax[k].plot(np.array(segments_time)%(2*period)/period, segments_amp[k], 'ko')
else:
    ax.plot(np.array(segments_time) % (2*period)/period, segments_amp[0], 'ko')

plt.show()






