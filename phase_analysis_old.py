import mini_smurfs as ms
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk


lc_file = 'endurance/Final_Pulsation_LC.txt'
data = np.loadtxt(lc_file).T
times = data[0]
flux = data[1]

s = ms.MiniSmurfs(times, flux, model='pda')
s.run(4, 2, 5)
s.save_result()
mod_flux = ms._pda_plot(s.time, s.result)

fig, ax = plt.subplots(4,1, figsize = (12,8))
ax[0].plot(s.time, s.flux, 'ko', ms = 0.75)
ax[0].plot(s.time, mod_flux, 'r-')
pdg = s.lc.to_periodogram()
pdg2 = s.resid_lc.to_periodogram()
ax[1].plot(pdg.frequency, pdg.power, 'k-')
ax[1].plot(pdg2.frequency, pdg2.power, 'r-')
aic =  [m.aic for m in s.result]
bic =   [m.bic for m in s.result]
ax[2].plot([m.nofp for m in s.result], [m.aic for m in s.result], 'ko')
ax[2].plot([m.nofp for m in s.result], [m.bic for m in s.result], 'ro')
ax[3].plot([m.nofp for m in s.result], [m.pr for m in s.result], 'go')
ax[3].plot([m.nofp for m in s.result], [m.vr for m in s.result], 'bo')
plt.show()
'''

r = ms.load_result('minismurfs/result.csv')
fig, ax = plt.subplots(3,1, figsize= (12,7))

lc_file = 'endurance/Final_Pulsation_LC.txt'
data = np.loadtxt(lc_file).T
times = data[0]
flux = data[1]
period = 1.66987725


pdg = lk.LightCurve(times, flux).to_periodogram()
ax[0].plot(times, flux, 'ko', ms=0.75)
ax[2].plot(pdg.frequency, pdg.power, 'k-')
y = np.zeros(len(times))
for f, amp, phase in zip(r['Frequency'], r['Amplitude'], r['Phase']):
    if amp > 0.000165:
        print(f, amp, phase)
    y += ms.sin(times, amp, f, phase)
    ax[2].plot([f, f], [0, amp], 'r-')
ax[0].plot(times, y, 'r-')
pdg = lk.LightCurve(times, flux-y).to_periodogram()
ax[1].plot(times, flux-y, 'ro', ms=0.75)
ax[2].plot(pdg.frequency, pdg.power, 'r-', alpha=0.5)
ax[2].set_xlim(0, 40)
fig2, ax2 = plt.subplots(figsize=(12,7))
ax2.plot(r['Frequency']%(1/period), r['Frequency'], 'ro', ms=5)



r2 = ms.load_result('minismurfs/result_pda.csv')

lc_file = 'endurance/Final_Pulsation_LC.txt'
data = np.loadtxt(lc_file).T
times = data[0]
flux = data[1]
period = 1.66987725


y = np.zeros(len(times))
for f, amp, phase, phase0, sigma, defect in zip(r2['Frequency'], r2['Amplitude'], r2['Phase'], r2['phase0'],  r2['sigma'], r2['defects']):
    y += ms.pda(times, amp, f, phase, phase0, sigma, defect)
    ax[2].plot([f, f], [0, amp], 'b--')
ax[0].plot(times, y, 'b-')
pdg = lk.LightCurve(times, flux-y).to_periodogram()
ax[1].plot(times, flux-y, 'bo', ms=0.75)
ax[2].plot(pdg.frequency, pdg.power, 'r-', alpha=0.5)
ax[2].set_xlim(0, 40)

for f, comp in zip(r2['Frequency'], r2['Component']):
    if comp == 'Secondary':
        ax2.plot(f%(1/period), f, 'cx', ms=5, mew = 2)
    else:
        ax2.plot(f%(1/period), f, 'mx', ms=5, mew = 2)



fig3, ax3 = plt.subplots(2,1, figsize= (12,7))
ax3[0].plot(r2['Free_params'], r2['BIC'], 'bx', mew=2)
ax3[0].plot(r['Free_params'], r['BIC'], 'rx', mew=2)
ax3[0].plot(r2['Free_params'], r2['AIC'], 'bo', mew=2)
ax3[0].plot(r['Free_params'], r['AIC'], 'ro', mew=2)
ax3[1].plot(r2['Free_params'], r2['Power_Reduction'], 'bx', mew=2)
ax3[1].plot(r['Free_params'], r['Power_Reduction'], 'rx', mew=2)
ax3[1].plot(r2['Free_params'], r2['Variance Reduction'], 'bo', mew=2)
ax3[1].plot(r['Free_params'], r['Variance Reduction'], 'ro', mew=2)

plt.show()
'''