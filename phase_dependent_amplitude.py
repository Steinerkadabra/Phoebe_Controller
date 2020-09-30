import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from smurfs import Smurfs
from tqdm import tqdm, trange
import multiprocessing as mp
import ast
import os
from scipy.optimize import curve_fit
from lmfit import Model


class mode():
    def __init__(self, f, amp, phase, snr, label = ""):
        self.f = f
        self.amp = amp
        self.phase = phase
        self.snr = snr
        self._label = label

    @property
    def label(self) -> str:
        """
        Returns the label of the found frequency
        """
        return self._label

def pda(time, amp, frequency, phase, phase0 = 0.3, sigma = 0.05, defect = 0.5):
    period = 1.66987725
    phases = time%period/period
    return amp*(1- defect*np.exp(-(phases - phase0) ** 2 / (2 * sigma ** 2)))*np.sin(2. * np.pi * (frequency * time + phase))

def get_result(list, dict):
    freq = 1/1.66987725
    f0 = list[0].f.n
    used = []
    for f in list:
        num = int(np.round((f.f.n-f0)/freq, 0))
        if num in used:
            continue
        else:
            used.append(num)
            try:
                if num != 0:
                    dict[f"amp_{num}"].append(f.amp.n/list[0].amp.n)
                    dict[f"phase_{num}"].append(f.phase.n-list[0].phase.n)
                else:
                    dict[f"amp_{num}"].append(f.amp.n)
                    dict[f"phase_{num}"].append(f.phase.n)
            except:
                continue
    for i in range(-5, 6):
        if i not in used:
            dict[f"amp_{i}"].append(0)
            dict[f"phase_{i}"].append(0)
    return dict



def simulate(run, time, folder, defect):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    result = {}
    for i in range(-5, 6):
        result[f"amp_{i}"] = []
        result[f"phase_{i}"] = []

    for _ in trange(200, desc = f'Run {run+1}', position = run+1):
        flux = pda(time, np.random.uniform(0.001, 0.003,1), np.random.uniform(10, 20, 1), np.random.uniform(0,1,1), defect = defect)\
               + np.random.normal(0, 0.000165, len(time))
        s = Smurfs(time = time, flux = flux, quiet_flag=True)
        s.run(improve_fit=True, f_max = 40, mode='scipy')
        #s.improve_result()
        res = s.result["f_obj"].values
        result = get_result(res, result)
    f = open(f'pda_simulations/{folder}/' + f'{run}.txt', "w")
    f.write(str(result))
    f.close()


def simulate_multiple(folder, defect= 0.5):
    n = 5
    if not os.path.exists('pda_simulations/'+str(folder)):
        os.mkdir('pda_simulations/'+str(folder))
    pool = mp.Pool(processes=n)
    data = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
    time = data[0][:20000]
    runs = []
    for i in range(n):
        runs.append((i, time, folder, defect))
    for _ in pool.starmap(simulate, runs):
        pass


def load_dict(folder, number):
    file = folder + f'{number}.txt'
    with open(file, 'r') as file:
        data = file.read().replace('\n', '')
    return ast.literal_eval(data)


def load_all(folder):
    result = {}
    for i in range(-5, 6):
        result[f"amp_{i}"] = []
        result[f"phase_{i}"] = []

    for i in range(0,5):
        step = load_dict(f'pda_simulations/{folder}/', i)
        for key in step.keys():
            result[key].extend(step[key])
    return result

def plot_simulations(folder, ax_amp, ax_phase, fc = 'gray'):#Initiate two figures acc to fig_phase, ax_phase = plt.subplots(3, 4, figsize=(12,7))
    result = load_all(folder)


    amp_key = [f"amp_{i}" for i in range(-5, 6)]
    phase_key = [f"phase_{i}" for i in range(-5, 6)]
    filterByKey = lambda keys: {x: result[x] for x in keys}

    count = 0
    for key in filterByKey(amp_key):
        data = np.array(result[key])
        data = data[data != 0]
        ax = ax_amp[int(count/4)][count%4]
        if key[-1] == "0":
            ax.hist(np.array(data), bins = 50, edgecolor = 'k', facecolor = fc, label = key, alpha = 0.5)
            ax.legend(fontsize = 'x-small')
            ax.set_xlabel('$A_0$')
            ax.set_ylabel('occurence')
        else:
            ax.hist(np.array(data), bins = 50, edgecolor = 'k', facecolor = fc, label = key, alpha = 0.5)
            ax.legend(fontsize = 'x-small')
            ax.set_xlabel('$A$/$A_0$')
            ax.set_ylabel('occurence')
        count+=1

    count = 0
    for key in filterByKey(phase_key):
        data = np.array(result[key])
        data = data[data != 0]
        ax = ax_phase[int(count/4)][count%4]
        ax.hist((np.array(data)+1)%1, bins = 50, edgecolor = 'k', facecolor = fc, label = key, alpha = 0.5)
        ax.legend(fontsize = 'x-small')
        ax.set_xlabel('phase')
        ax.set_ylabel('occurence')
        count += 1

    #fig_amp.tight_layout(h_pad=0, w_pad=0)
    #fig_phase.tight_layout(h_pad=0, w_pad=0)
    #plt.show()
    #plt.close()
    return ax_amp, ax_phase

class run():
    def __init__(self, num, time):
        self.n = num
        self.time = time

def plot_mult_sims(folders, colors):
    fig_amp, ax_amp = plt.subplots(3, 4, figsize=(12,7))
    fig_phase, ax_phase = plt.subplots(3, 4, figsize=(12,7))
    for f, c in zip(folders, colors):
        ax_amp, ax_phase = plot_simulations(f, ax_amp, ax_phase, fc = c)
    plt.show()

def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def line(x, a, ):
    return a * x

def square(x, a, b):
    return a * x + b*x**2

def get_amplitude_value(folder, key):
    result = load_all(folder)
    data = np.array(result[key])
    data = data[data != 0]
    vals, edges =np.histogram(data, 30)
    xs = (edges[1:]+edges[:-1])/2
    try:
        popt, pcov = curve_fit(gauss, xs, vals, p0=[100, np.mean(data), np.std(data)])
        return popt[1], popt[2], len(data)
    except RuntimeError:
        errfig, errax = plt.subplots(figsize = (12,7))
        x = np.linspace(np.min(data), np.max(data), 500)
        errax.plot(x, gauss(x, *[100, np.mean(data), np.std(data)]), 'r-')
        errax.bar(xs, vals, width = edges[1]-edges[0], edgecolor = 'k', facecolor = 'gray')
        errax.set_title(key)
        return 0, 0, 0

def plot_amplitude_statistics(figsize, dpi):
    c = ['#FF5733', '#FFB900', '#93FF00', '#019539', '#00D1EA', '#000BEA', '#BF00EA', '#806979', '#000000', '#3e197c', '#ffb600']
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    amp_key = [f"amp_{i}" for i in range(-5, 6)]
    count = 0
    for key in range(-5, 6):
        if key != 0:
            means = []
            sds = []
            defects = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            for folder in  range(4, 12):
                m, s, length = get_amplitude_value(folder, f'amp_{key}')
                means.append(m)
                sds.append(s)
            marker = '^'
            if key > 0:
                marker = 's'
            #popt, pcov = curve_fit(square, defects, means, sigma=sds,  p0=[(means[-1]-means[0])/0.7, 0], absolute_sigma=True)
            popt, pcov = curve_fit(line, defects, means, sigma=sds,  p0=[(means[-1]-means[0])/0.7])#, absolute_sigma=True)
            x = np.linspace(0,0.8, 500)
            print(f"$A_{ {key} }$")
            N = np.array(means)
            #Nexp = square(np.array(defects), *popt)
            Nexp = line(np.array(defects), *popt)
            r = N - Nexp
            chisq = np.sum((r / np.array(sds)) ** 2)
            df = len(means) - len(popt)
            print("chisq =", chisq, "df =", df, "reduced chisq", chisq/df, 'num= ', length)
            print('###################')
            ax.errorbar(defects, means, yerr = sds, color = c[key+5], marker = marker,  markerfacecolor= 'none',
                        markeredgewidth = 1, capsize = 2, lw =0, elinewidth = 1, label = f"$\\tilde{{A}}_{ {key} }$")#   $"+ "$\\frac{\chi^2}{DoF}=$" +f"{np.round(chisq/df, 3)}")
            #ax.plot(x, square(x, *popt), color = c[count])
            lw ='-'
            if key > 0:
                lw ='--'
            ax.plot(x, line(x, *popt), color = c[key+5], ls = lw)
            count+=1
    ax.legend(fontsize = 'small', ncol = 4, labelspacing=0.05, handletextpad = 0.1)
    ax.set_xlabel('Defect')
    ax.set_ylabel('relative Amplitude')
    # plt.tight_layout()
    #plt.savefig('pda_simulations/plots/line.png', bbox_inches = 0)
    # plt.show()
    return fig, ax

def sin(x: np.ndarray, amp: float, f: float, phase: float) -> np.ndarray:

    return amp * np.sin(2. * np.pi * (f * x + phase))


def _lmfit_fit(result, time, flux):
    models = []
    for f in result:
        m = Model(sin, prefix=f._label)

        m.set_param_hint(f.label + 'amp', value=f.amp, min=0 * f.amp,
                         max=2 * f.amp)
        m.set_param_hint(f.label + 'f', value=f.f, min=0.8*f.f,
                         max=1.2 * f.f)
        m.set_param_hint(f.label + 'phase', value=f.phase, min=0,
                         max=1)
        models.append(m)

    model: Model = np.sum(models)
    fit_result = model.fit(flux, x=time)

    for f in result:
        #sigma_amp, sigma_f, sigma_phi = m_od_uncertainty(self.lc, fit_result.values[f._label + 'amp'])
        f.amp = fit_result.values[f.label + 'amp']
        f.f = fit_result.values[f.label + 'f']
        f.phase =fit_result.values[f.label + 'phase']

    return result




def simulate_fit(run, time, folder, defect):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    result_sim = {}
    f_orb = 1/1.66987725
    result_sim[f"amp"] = []
    result_sim[f"phase"] = []
    result_sim[f"freq"] = []
    for i in range(-5, 6):
        result_sim[f"amp_{i}"] = []
        result_sim[f"phase_{i}"] = []

    for _ in trange(200, desc = f'Run {run+1}', position = run+1):
        amp = np.random.uniform(0.001, 0.003,1)[0]
        freq =  np.random.uniform(10, 20, 1)[0]
        phi = np.random.uniform(0,1,1)[0]
        result_sim[f"amp"].append(amp)
        result_sim[f"phase"].append(phi)
        result_sim[f"freq"].append(freq)
        flux = pda(time, amp, freq, phi, defect = defect)\
               + np.random.normal(0, 0.000165, len(time))
        result = []
        for i in range(-5,6):
            if i ==0:
                result.append(mode(freq + i* f_orb, amp, phi, 10, label="F"+str(i).replace('-', '_')))
            else:
                result.append(mode(freq + i * f_orb, amp*0.05, phi, 10, label="F" + str(i).replace('-', '_')))
        result = _lmfit_fit(result, time, flux)
        for f in result:
            f._label =  f._label.replace('_', '-')[1:]
            if f._label == '0':
                f0 =  f
        for f in result:
            num = int(f._label)
            if num != 0:
                result_sim[f"amp_{num}"].append(f.amp / f0.amp)
                result_sim[f"phase_{num}"].append(f.phase - f0.phase)
            else:
                result_sim[f"amp_{num}"].append(f.amp)
                result_sim[f"phase_{num}"].append(f.phase)
    f = open(f'pda_simulations/fit/{folder}/' + f'{run}.txt', "w")
    f.write(str(result_sim))
    f.close()

def simulate_multiple_fit(folder, defect= 0.5):
    n = 5
    if not os.path.exists('pda_simulations/fit/'+str(folder)):
        os.mkdir('pda_simulations/fit/'+str(folder))
    pool = mp.Pool(processes=n)
    data = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
    time = data[0][:20000]
    runs = []
    for i in range(n):
        runs.append((i, time, folder, defect))
    for _ in pool.starmap(simulate_fit, runs):
        pass


def get_phase_tda(folder):
    tda_input = []
    result = load_all(folder)
    amps = result['amp_0']
    ph_keys = []
    for i in range(-5, 6):
        if i !=0:
            ph_keys.append(f"phase_{i}")
    num = len(amps)
    for i in range(num):
        vert = []
        for key in ph_keys:
            vert.append((1+result[key][i])%1)
        if 0 in vert:
            print(vert)
        else:
            tda_input.append(vert)

    return np.array(tda_input)







'''
f8 = get_phase_tda(8)
f7 = get_phase_tda(6)


from ripser import ripser
from persim import plot_diagrams

dgm8 = ripser(np.array(f8))['dgms'][1]
dgm7 = ripser(np.array(f7))['dgms'][1]
#print(dgm7, dgm8)

plot_diagrams([dgm8, dgm7] , labels=['Clean $H_1$', 'Noisy $H_1$'], show = True)
'''

