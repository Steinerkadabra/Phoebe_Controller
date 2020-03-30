import phoebe

import numpy as np
import matplotlib.pyplot as plt
import time
import lightkurve as lk
import multiprocessing as mp  # import Pool
import sys
import os
import shutil
import json




def initialize_phoebe(output_dir = None, mpi_ncores = 0, logger = True):
    if output_dir != None:
        try:
            os.mkdir(output_dir)
        except:
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)
    if logger:
        sys.stdout = Logger(output_dir)
    phoebe.logger(clevel='ERROR')  # ignore warnings - for tqdm to work properly
    phoebe.interactive_checks_off()
    phoebe.check_visible_off()
    phoebe.interactive_constraints_off()
    if mpi_ncores != 0:
        phoebe.mpi_on(nprocs=mpi_ncores)
    return output_dir

def load_lc(lc_file = 'RS_CHA_lightcurve.txt', plot= False, npoints = 5000, points_step = 5, parts_of_lc = False):
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
        return  (times, flux, sigmas)
    except:
        return (times, flux, np.ones(len(flux)))


def standard_binary():
    dict = {
        'vgamma@system': 15.7,
        'mass@primary': 1.89,
        'mass@secondary': 1.89,
        'period@binary': 1.66987725,
        'incl@binary': 83.4,
        'vsini@primary': 64,
        'vsini@secondary': 70,
        'requiv@primary': 2.15,
        'requiv@secondary': 2.36,
        'teff@primary': 7638,
        'teff@secondary': 7228,
        'ecc@binary': 0,
        't0_supconj@binary@component': 1599.397275,


    }
    return dict


def binary_history(dict):
    hist_dict = {}
    for key in dict.keys():
        hist_dict[key] = [dict[key]]
    return hist_dict




def units():
    dict = {
        'vgamma@system': 'km/s',
        'mass@primary': r'$M_\odot$',
        'mass@secondary': r'$M_\odot$',
        'period@binary': 'days',
        'incl@binary': 'deg',
        'vsini@primary': 'km/s',
        'vsini@secondary': 'km/s',
        'requiv@primary': r'$R_\odot$',
        'requiv@secondary': r'$R_\odot$',
        'teff@primary': 'K',
        'teff@secondary': 'K',
        'ecc@binary': 'deg',
        't0_supconj@binary@component': 'days',
        'passed calculation time': 's'

    }
    return dict

def get_binary(dict):
    binary = phoebe.default_binary()
    binary.set_value('vgamma@system', dict['vgamma@system'])
    mass1 = dict['mass@primary']
    mass2 = dict['mass@secondary']


    P = dict['period@binary']
    mu_sun = 1.32712440018e20  # = G M_sun [m3 s^-2], Wiki Standard_gravitational_parameter
    R_sun = 695700000  # [m] Wiki Sun

    sma = (mu_sun * (mass1 + mass2) * (P * 86400 / (2 * np.pi)) ** 2) ** (1. / 3) / R_sun  # Kepler equation

    incl = dict['incl@binary']
    vp_sini = dict['vsini@primary']
    vs_sini = dict['vsini@secondary']

    Rp = dict['requiv@primary']    # [R_sun]
    Rs = dict['requiv@secondary']    # [R_sun]

    sini = np.sin(np.pi * incl / 180)

    vp = vp_sini * 86400 / sini  # [km/s]
    vs = vs_sini * 86400 / sini  # [km/s]

    Pp = 2 * np.pi * Rp * R_sun / 1000 / vp
    Ps = 2 * np.pi * Rs * R_sun / 1000 / vs

    Fp = P / Pp
    Fs = P / Ps

    binary.set_value('q', mass2 / mass1)
    binary.set_value('incl@binary', incl)
    binary.set_value('sma@binary', sma)  # calculated
    binary.set_value('ecc@binary', abs(dict['ecc@binary']))

    binary.set_value('period@binary', P)  # calculated

    binary.set_value('syncpar@primary', Fp)  # calculated
    binary.set_value('syncpar@secondary', Fs)  # calculated

    binary.set_value('requiv@primary', Rp)
    binary.set_value('requiv@secondary', Rs)

    binary.set_value('teff@primary', dict['teff@primary'])
    binary.set_value('teff@secondary', dict['teff@secondary'])

    binary.set_value('syncpar@secondary@component', 1)
    binary.set_value('syncpar@primary@component', 1)

    binary.set_value('t0_supconj@binary@component', dict['t0_supconj@binary@component'])



    return binary

def chi_square_multi(input_vals, flux, times, sigmas, exp_time = False):
    dict = standard_binary()
    for key in input_vals.keys():
        dict[key] = input_vals[key]
    binary = get_binary(dict)
    binary.add_dataset('lc', times=times, dataset='lc01', fluxes = flux,  overwrite=True, ld_func='logarithmic', passband='TESS:T')
    binary.set_value('pblum_mode', 'dataset-scaled')
    if exp_time:
        ###third light####
        #binary.set_value('l3_mode', 'fraction')
        #binary.set_value('l3_frac', 0.0297703)
        #binary.run_compute(model = 'mod', overwrite= True)
        #### exposure time ####
        #binary['exptime'] = 2, 's'
        #binary.run_compute(fti_method='oversample',  model = 'mod', overwrite= True)
        ### grav_bol####
        binary['gravb_bol@primary'] = 0.8
        binary['gravb_bol@secondary'] = 0.8
        binary.run_compute(irrad_method='horvat', model='mod', overwrite=True)
        #### irrad method
        #binary['irrad_frac_refl_bol@primary'] = 0.9
        #binary['irrad_frac_refl_bol@secondary'] = 0.9
        #binary.run_compute(irrad_method='horvat',   model = 'mod', overwrite= True)
        ### ld mode ####
        #binary['ld_mode@primary'] = 'interp'
        #binary['ld_mode@secondary'] = 'interp'
        #binary.run_compute(model = 'mod', overwrite= True)
    else:
        binary.run_compute(model = 'mod', overwrite= True)
    fluxes = binary.get_model(model = 'mod')['fluxes'].value
    times = binary.get_model(model = 'mod')['times'].value
    mod = lk.LightCurve(time = times, flux = fluxes)
    chi_square = np.sum(np.array(flux - mod.flux)**2/np.array(sigmas))
    return (chi_square, mod.flux)

def calc_an_plot(input_vals, flux, times, exp_time = False, ax = None):
    if ax == None:
        fig, ax = plt.subplots(2, 1, figsize = (10,6))
    dict = standard_binary()
    for key in input_vals.keys():
        dict[key] = input_vals[key]
    binary = get_binary(dict)

    binary.add_dataset('lc', times=times, dataset='lc01', overwrite=True, ld_func='logarithmic', passband='TESS:T')
    if exp_time:
        binary['exptime'] = 2, 's'
        binary.run_compute(fti_method='oversample', ltte=False, model = 'mod', overwrite= True)
    else:
        binary.run_compute(ltte=False, model = 'mod', overwrite= True)
    fluxes = binary.get_model(model = 'mod')['fluxes'].value
    times = binary.get_model(model = 'mod')['times'].value
    mod = lk.LightCurve(time = times, flux = fluxes)
    mod = mod.normalize()
    chi_square = np.sum(np.array(flux - mod.flux)**2)
    ax[0].set_title('chi_square=', chi_square)
    ax[0].plot(times, mod.flux, 'ro', markersize = 0.75)
    ax[0].plot(times, flux, 'ko', markersize = 0.75)
    ax[1].plot(times, flux - mod.flux, 'ko', markersize = 0.75)



class vertex_multi:
    def __init__(self, input_vals, flux, times, sigmas,  exp_time):
        self.flux = flux
        self.times = times
        self.vals = input_vals
        self.exp_time = exp_time
        self.sigmas = sigmas
        self.sd, self.flux_model = chi_square_multi(self.vals, self.flux, self.times, self.sigmas,  exp_time = self.exp_time)


class simplex_multi:
    def __init__(self, input_vals, input_sigmas, flux, times, sigmas, settings= 'standard', exp_time = False, ncores = 32):
        if settings == 'standard':
            self.alpha = 1
            self.gamma = 2
            self.rho = 0.5
            self.sigma = 0.5
        elif settings == 'agressive':
            self.alpha = 2
            self.gamma = 2
            self.rho = 0.95
            self.sigma = 0.95
        self.ncores = ncores
        self.history = binary_history(input_vals)
        self.start = time.time()
        self.vertices = []
        self.units = units()
        self.chi_history = []
        self.sd_history = []
        self.steps = []
        self.dims = len(input_vals) + 1
        self.vals = input_vals
        self.input_sigmas = input_sigmas
        self.flux = flux
        self.times = times
        self.sigmas = sigmas
        self.exp_time = exp_time
        self.start_nprocs = int(self.ncores/self.dims)
        if self.dims * self.start_nprocs > self.ncores:
            self.start_nprocs = self.start_nprocs -1
        self.shrink_nprocs = int(self.ncores/(self.dims-1))
        if (self.dims-1) * self.shrink_nprocs > self.ncores:
            self.shrink_nprocs = self.shrink_nprocs-1
        self.nprocs = int(self.ncores/3)
        if 3 * self.nprocs > self.ncores:
            self.shrink_nprocs = self.nprocs-1
        vert_vals = [input_vals.copy()]
        for key in input_vals.keys():
            ex_vert_vals = input_vals.copy()
            ex_vert_vals[key] = ex_vert_vals[key] + input_sigmas[key]
            vert_vals.append(ex_vert_vals)
        mult_input = []
        for each in vert_vals:
            mult_input.append((each, flux, times, sigmas, exp_time))
        phoebe.mpi_on(nprocs=self.start_nprocs)
        with mp.Pool(self.dims) as p:
            self.vertices = p.starmap(vertex_multi, mult_input)
        self.sort()
        self.initial_vertex = self.vertices[0]


    def sort(self):
        newlist = sorted(self.vertices, key=lambda x: x.sd)
        self.vertices = newlist


    def calculate_centroid(self):
        centroid = self.vertices[0].vals.copy()
        for vert in self.vertices[1:-1]:
            for key in self.vertices[0].vals.keys():
                centroid[key] = centroid[key] + vert.vals[key]
        for key in self.vertices[0].vals.keys():
            centroid[key] = centroid[key]/(self.dims-1)
        self.centroid_vals = centroid

    def calculate_reflected_point(self, alpha):
        self.reflected_point_vals = self.centroid_vals.copy()
        for key in self.centroid_vals.keys():
            self.reflected_point_vals[key] = self.centroid_vals[key] + alpha * (self.centroid_vals[key] - self.vertices[-1].vals[key])

    def reflection(self):
        self.vertices = self.vertices[:-1]
        new_vertex = self.reflected_point
        self.vertices.append(new_vertex)
        self.standard_deviation()
        self.last_step = 'Reflection'

    def calculate_expanded_point(self, gamma):
        self.expanded_point_vals = self.centroid_vals.copy()
        for key in self.centroid_vals.keys():
            self.expanded_point_vals[key] = self.centroid_vals[key] + gamma * (self.reflected_point_vals[key] - self.centroid_vals[key])

    def expansion(self):
        self.vertices = self.vertices[:-1]
        if self.expanded_point.sd < self.reflected_point.sd:
            new_vertex = self.expanded_point
        else:
            new_vertex = self.reflected_point
        self.vertices.append(new_vertex)
        self.standard_deviation()
        self.last_step = 'Expansion'

    def calculate_contracted_point(self, rho):
        self.contracted_point_vals = self.centroid_vals.copy()
        for key in self.centroid_vals.keys():
            self.contracted_point_vals[key] = self.centroid_vals[key] + rho * (self.vertices[-1].vals[key] - self.centroid_vals[key])

    def contraction(self):
        self.vertices = self.vertices[:-1]
        new_vertex = self.contracted_point
        self.vertices.append(new_vertex)
        self.standard_deviation()
        self.last_step = 'Contraction'

    def calculate_points(self):
        self.calculate_reflected_point(self.alpha)
        self.calculate_expanded_point(self.gamma)
        self.calculate_contracted_point(self.rho)
        mult_input = []
        mult_input.append((self.reflected_point_vals, self.flux, self.times, self.sigmas, self.exp_time))
        mult_input.append((self.expanded_point_vals, self.flux, self.times, self.sigmas, self.exp_time))
        mult_input.append((self.contracted_point_vals, self.flux, self.times, self.sigmas,  self.exp_time))
        phoebe.mpi_on(nprocs=self.nprocs)
        with mp.Pool(processes=3) as p:
            calculated_points = p.starmap(vertex_multi, mult_input)
        self.reflected_point = calculated_points[0]
        self.expanded_point = calculated_points[1]
        self.contracted_point = calculated_points[2]

    def shrink(self):
        phoebe.mpi_on(nprocs=self.shrink_nprocs)
        new_vertices = [self.vertices[0]]
        mult_input = []
        for vert in self.vertices[1:]:
            new_vals = self.centroid_vals.copy()
            for key in self.centroid_vals.keys():
                new_vals[key] = self.vertices[0].vals[key] + self.sigma * (vert.vals[key] - self.vertices[0].vals[key])
            mult_input.append((new_vals, self.flux, self.times, self.sigmas,  self.exp_time))
        with mp.Pool(self.dims-1) as p:
            new_calculated = p.starmap(vertex_multi, mult_input)
        for each in new_calculated:
            new_vertices.append(each)
        self.vertices = new_vertices
        self.last_step = 'Shrink'
        self.standard_deviation()

    def standard_deviation(self):
        sds = []
        for vert in self.vertices:
            sds.append(vert.sd)
        self.sd = np.std(sds)

    def print_best_vertex(self):
        table = {}
        for key in self.vertices[0].vals.keys():
            table[key] = str(self.vertices[0].vals[key])
        table['$\chi^2$'] = str(self.vertices[0].sd)
        table['passed calculation time'] = str(time.time() - self.start)
        for name, value in table.items():
            try:
                print(f'{name:30} ==> {value:20} {self.units[name]}')
            except KeyError:
                print(f'{name:30} ==> {value:20}')

    def save_summary_plot(self, output_dir= None):
        fig, ax = plt.subplots(4,1, figsize = (10,16))
        ax[0].plot(range(1, len(self.steps)+1), self.chi_history, 'ko')
        ax[1].plot(range(1, len(self.steps)+1), self.steps, 'ko-')
        ax[2].plot(range(1, len(self.steps)+1), self.sd_history, 'ko')
        for a in ax:
            a.set_xlabel('Iteration')
        ax[0].set_ylabel(r'$\chi^2$ of best vertex')
        ax[0].set_yscale('log')
        ax[2].set_ylabel(r'$\sigma$ of vertices in simplex')
        ax[2].set_yscale('log')
        ax[1].set_ylabel(r'Step Type')
        ax[3].plot(self.times, self.flux, 'ko', markersize = 0.75)
        ax[3].plot(self.times, self.vertices[0].flux_model, 'r-')
        ax[3].plot(self.times, self.initial_vertex.flux_model, 'g-')
        ax[3].set_xlabel('Time - 2457000 [BTJD days]')
        ax[3].set_ylabel('Normalized Flux')
        plt.tight_layout()
        if output_dir != None:
            plt.savefig(output_dir +'/Simplex_History.png')
            plt.close()
        else:
            plt.savefig(f'Simplex_History{time.time()}.png')
            plt.close()

    def save_summary(self, output_dir = None):
        if output_dir != None:
            np.savetxt(output_dir +'/Simplex_History.txt', np.array([self.steps, self.chi_history, self.sd_history]).T, fmt = [self.steps[0].dtype, self.chi_history[0].dtype, self.sd_history[0].dtype] )
        else:
            np.savetxt(output_dir +'/Simplex_History.txt', np.array([self.steps, self.chi_history, self.sd_history]).T)

    def start_history(self):
        self.history = binary_history(self.vals)
        self.history['steps'] = ['None']
        self.history['simplex_sd'] = [self.sd]
        self.history['chi**2'] = [self.vertices[0].sd]


    def update_history(self):
        for key in self.vals.keys():
            self.history[key].append(self.vertices[0].vals[key])
        self.history['steps'].append(self.last_step)
        self.history['simplex_sd'].append(self.sd)
        self.history['chi**2'].append(self.vertices[0].sd)

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def nelder_mead_opt_multi(input_vals, input_sigmas, flux, times, sigmas,  max_iter=20, settings = 'standard', exp_time = False, ncores = 32, output_dir = None):
    simp = simplex_multi(input_vals, input_sigmas, flux, times, sigmas, settings = settings, exp_time = exp_time, ncores= ncores)
    simp.standard_deviation()
    simp.sort()
    print(color.BOLD +'Initial Simplex with standard deviation', simp.sd, ' and best vertex:' + color.END)
    simp.print_best_vertex()
    simp.start_history()
    for i in range(max_iter):
        try:
            string= f'Iteration {i+1}: '
            simp.sort()
            simp.calculate_centroid()
            simp.calculate_points()
            if simp.vertices[-2].sd > simp.reflected_point.sd >= simp.vertices[0].sd:
                simp.reflection()
                string = string + 'Reflection. '
                step = 'Reflection'
            elif simp.reflected_point.sd < simp.vertices[0].sd:
                simp.expansion()
                string = string + 'Expansion. '
                step = 'Expansion'
            else:
                if simp.contracted_point.sd < simp.vertices[-1].sd:
                    simp.contraction()
                    string = string + 'Contraction. '
                    step = 'Contraction'
                else:
                    simp.shrink()
                    string = string + 'Shrink. '
                    step = 'Shrink'
            simp.sort()
            simp.chi_history.append(simp.vertices[0].sd)
            simp.sd_history.append(simp.sd)
            simp.steps.append(step)
            print(color.BOLD + string, 'We now have a standard deviation of',  simp.sd, 'and best vertex:' + color.END)
            simp.update_history()
            f = open(output_dir + "/Simplex_History.txt", "w")
            f.write(str(simp.history))
            f.close()
            simp.print_best_vertex()
            #simp.save_summary_plot(output_dir = output_dir)
	        #simp.save_summary(output_dir = output_dir)
        except json.decoder.JSONDecodeError:
            continue

    #print(simp.history)
    #simp.save_summary_plot(output_dir = output_dir)
    import pickle
    save = open(output_dir + "/history.pkl", "wb")
    pickle.dump(dict, save)
    save.close()

class Logger(object):
    def __init__(self, output_dir):
        self.terminal = sys.stdout
        self.log = open(f"{output_dir}/output.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

