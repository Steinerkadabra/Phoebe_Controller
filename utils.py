import numpy as np
import phoebe_controller
import ast
import matplotlib.pyplot as plt

def load_history(folder):
    file = folder + 'Simplex_History.txt'
    with open(file, 'r') as file:
        data = file.read().replace('\n', '')
    return ast.literal_eval(data)

def get_best_vals(dict):
    res = {}
    for key in dict.keys():
        if key not in ['steps', 'simplex_sd', 'chi**2']:
           res[key] = dict[key][-1]
    return res

def load_lc(lc_file = 'RS_CHA_lightcurve.txt', plot= False, npoints = 5000, points_step = 1, parts_of_lc = False):
    data = np.loadtxt(lc_file).T
    times = data[0]
    flux = data[1]
    try:
        sigmas = data[2]
    except:
        pass
    if parts_of_lc:
        times = times[:npoints:points_step]
        flux = flux[:npoints:points_step]
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, flux, 'ko', markersize = 0.75)
        plt.show()
        plt.close()
    try:
        return  times, flux, sigmas
    except:
        return times, flux, np.ones(len(flux))


def remove_binary_signal(folder, endurance_dir, compare_lc,  exp_time = False):
    history = load_history(folder)
    best_vals = get_best_vals(history)
    times, fluxes, sigmas = load_lc(endurance_dir + compare_lc, npoints=10000, parts_of_lc=False )
    fig3 = plt.figure(constrained_layout=True, figsize=(15,8))
    gs = fig3.add_gridspec(2, 3)
    ax00 = fig3.add_subplot(gs[0, :-1])
    ax01 = fig3.add_subplot(gs[1, :-1])
    ax10 = fig3.add_subplot(gs[0, -1:])
    ax11 = fig3.add_subplot(gs[1, -1:])
    ax00.plot(times, fluxes, 'ko', markersize = 0.75)
    ax10.plot(times, fluxes, 'ko', markersize = 0.75)
    for ax in [ax00, ax01, ax10, ax11]:
        ax.set_xlabel('Time - 2457000 [BTJD days]')
        ax.set_ylabel('normalized flux')
    for ax in [ax10, ax11]:
        ax.set_xlim(np.min(times), np.min(times) + 3.2)
    chi2, mod_flux = phoebe_controller.chi_square_multi(best_vals, fluxes, times, sigmas, exp_time= exp_time)
    ax00.plot(times, mod_flux, 'r-')
    ax10.plot(times, mod_flux, 'r-')
    ax01.plot(times, mod_flux-fluxes, 'ko', markersize = 0.75)
    ax11.plot(times, mod_flux-fluxes, 'ko', markersize = 0.75)
    plt.tight_layout()
    plt.savefig(folder + 'LC_p_binary_model.png')
    np.savetxt(folder+ 'binary_model.txt', np.array([times, mod_flux]).T)
    np.savetxt(folder+ 'residuals.txt', np.array([times, np.array(mod_flux)- np.array(fluxes)]).T)
    plt.show()
    plt.close()

plt.rcParams.update({'font.size': 20.0, 'xtick.labelsize': 'x-small'})
#print(plt.rcParams)
phoebe_controller.initialize_phoebe(mpi_ncores=16, logger=False)

endurance_dir = 'data/'
#remove_binary_signal(endurance_dir + 'iteration2/', endurance_dir, 'result_iteration1.txt', exp_time = False)
remove_binary_signal(endurance_dir + 'iteration2_minus_sigma/', endurance_dir, 'result_iteration1.txt', exp_time = False)
