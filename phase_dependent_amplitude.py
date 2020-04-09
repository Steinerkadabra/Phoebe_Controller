import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from smurfs import Smurfs
from tqdm import tqdm, trange
import multiprocessing as mp
import ast
import os
import ray


class mode():
    def __init__(self, f, amp, phase, snr):
        self.f = f
        self.amp = amp
        self.phase = phase
        self.snr = snr

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
                    dict[f"amp_{num}"].append(f.amp.n.amp.n)
                    dict[f"phase_{num}"].append(f.phase.n.phase.n)
            except:
                continue
    for i in range(-5, 6):
        if i not in used:
            dict[f"amp_{i}"].append(0)
            dict[f"phase_{i}"].append(0)
    return dict

@ray.remote
def s(run, time):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    result = {}
    for i in range(-5, 6):
        result[f"amp_{i}"] = []
        result[f"phase_{i}"] = []

    for i in tqdm(range(100)):
        flux = pda(time, np.random.uniform(0.001, 0.003,1), np.random.uniform(10, 20, 1), np.random.uniform(0,1,1))\
               + np.random.normal(0, 0.000165, len(time))
        s = Smurfs(time = time, flux = flux, quiet_flag=True)
        s.run(improve_fit=False)
        s.improve_result()
        res = s.result["f_obj"].values
        result = get_result(res, result)
    f = open('pda_simulations/1/' + f'{run}.txt', "w")
    f.write(str(result))
    f.close()


def simulate2(run, time):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    result = {}
    for i in range(-5, 6):
        result[f"amp_{i}"] = []
        result[f"phase_{i}"] = []

    for _ in tqdm(range(100)):
        flux = pda(time, np.random.uniform(0.001, 0.003,1), np.random.uniform(10, 20, 1), np.random.uniform(0,1,1))\
               + np.random.normal(0, 0.000165, len(time))
        s = Smurfs(time = time, flux = flux, quiet_flag=True)
        s.run(improve_fit=False)
        s.improve_result()
        res = s.result["f_obj"].values
        result = get_result(res, result)
    f = open('pda_simulations/1/' + f'{run}.txt', "w")
    f.write(str(result))
    f.close()

def simulate(run):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    result = {}
    for i in range(-5, 6):
        result[f"amp_{i}"] = []
        result[f"phase_{i}"] = []

    for _ in tqdm(range(100)):
        flux = pda(run.time, np.random.uniform(0.001, 0.003,1), np.random.uniform(10, 20, 1), np.random.uniform(0,1,1))\
               + np.random.normal(0, 0.000165, len(run.time))
        s = Smurfs(time = run.time, flux = flux, quiet_flag=True)
        s.run(improve_fit=False)
        s.improve_result()
        res = s.result["f_obj"].values
        result = get_result(res, result)
    f = open('pda_simulations/1/' + f'{run.n}.txt', "w")
    f.write(str(result))
    f.close()

def simulate_multiple():
    n = 6
    pool = mp.Pool(processes=n)
    data = np.loadtxt('data/' + 'Final_Pulsation_LC.txt').T
    time = data[0][:20000]
    runs = []
    for i in range(n):
        runs.append((i, time))
    for _ in pool.starmap(simulate2, runs):
        pass

def simulate_multiple_ray():
    n = 12
    ray.init(num_cpus=n)
    data = np.loadtxt('endurance/' + 'Final_Pulsation_LC.txt').T
    time = data[0][:20000]
    for _ in range(n):
        ray.get([s.remote(i, time) for i in range(n)])

#simulate_multiple()
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

    for i in range(0,8):
        step = load_dict(f'pda_simulations/{folder}/', i)
        for key in step.keys():
            result[key].extend(step[key])
    return result

def plot_simulations(folder):
    result = load_all(folder)


    fig_amp, ax_amp = plt.subplots(3, 4, figsize=(12,7))
    fig_phase, ax_phase = plt.subplots(3, 4, figsize=(12,7))
    amp_key = [f"amp_{i}" for i in range(-5, 6)]
    phase_key = [f"phase_{i}" for i in range(-5, 6)]
    filterByKey = lambda keys: {x: result[x] for x in keys}

    count = 0
    for key in filterByKey(amp_key):
        data = np.array(result[key])
        data = data[data != 0]
        ax = ax_amp[int(count/4)][count%4]
        if key[-1] == 0:
            ax.hist(np.array(data), bins = np.linspace(0.0005, 0.003), edgecolor = 'k', facecolor = 'gray', label = key)
            ax.legend(fontsize = 'x-small')
            ax.set_xlabel('phase')
            ax.set_ylabel('occurence')
        else:
            ax.hist(np.array(data), bins = np.linspace(0, 0.15, 50), edgecolor = 'k', facecolor = 'gray', label = key)
            ax.legend(fontsize = 'x-small')
            ax.set_xlabel('phase')
            ax.set_ylabel('occurence')
        count+=1

    count = 0
    for key in filterByKey(phase_key):
        data = np.array(result[key])
        data = data[data != 0]
        ax = ax_phase[int(count/4)][count%4]
        ax.hist((np.array(data)+1)%1, bins = np.linspace(0,1, 50), edgecolor = 'k', facecolor = 'gray', label = key)
        ax.legend(fontsize = 'x-small')
        ax.set_xlabel('phase')
        ax.set_ylabel('occurence')
        count += 1

    fig_amp.tight_layout(h_pad=0, w_pad=0)
    fig_phase.tight_layout(h_pad=0, w_pad=0)
    plt.show()
    plt.close()

class run():
    def __init__(self, num, time):
        self.n = num
        self.time = time

simulate_multiple()
#simulate(1)
