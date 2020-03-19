import numpy as np
import matplotlib.pyplot as plt

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




time, flux = load_lc(lc_file = 'endurance/RS_Cha_minus_phoebe_minus_binary_mutiples.txt', parts_of_lc=False, plot = False)
phase = (time % 0.5988495842998753)/0.5988495842998753
ind = np.where(np.logical_and(phase >= 0, phase <= 0.05))
plt.plot(time[ind], flux[ind], 'ko', markersize = 0.75)
num = 0
print(ind)
for f in ind[0]:
    if f > 0:
        num += 1
print(num)





plt.show()