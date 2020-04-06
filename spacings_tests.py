import numpy as np
import matplotlib.pyplot as plt
import math
import time

f1 = 2#np.random.uniform(0, 10)
f2 = 3#np.random.uniform(0, 10)
f3 = 4#np.random.uniform(0, 10)
Ps = []
for i in range(8):
    Ps.append(f1+i*0.1315)
for i in range(6):
    Ps.append(f2+i*0.4332)
for i in range(3):
    Ps.append(f3+i*0.3579)
for i in range(8):
    Ps.append(np.random.uniform(0.5, 10, 1)[0])
Ps = np.array(Ps)+np.random.normal(0, 0.0005, len(Ps))

difs = []
for i in range(len(Ps)):
    for k in range(i+1, len(Ps)):
        difs.append(abs(Ps[i]-Ps[k]))

difs = np.array(difs)

def s(difs, dP):
    s = []
    for d in difs:
        x = d/dP
        s.append(d-(int(math.ceil(x))-1)*dP)
    return np.array(s)


def s2(difs, dP):
    return difs - (np.ceil(difs/dP)-1)*dP
    #return np.int(np.ceil(difs/dP))

def s0_1(s, epsilon, dP):
    num = 0
    for i in s:
        if (0 <= i <= epsilon) or (dP-epsilon <= i <= dP):
            num +=1
    return num

def matrix_test(difs, dP_max, eps):
    res = []
    dPs = np.linspace(0.05, dP_max, 10000)
    for dP in dPs:
        t = s(difs, dP)
        s0 = s0_1(t, eps, dP)
        res.append(s0)
    return np.array(res), dPs

def load_results(path: str):
    """
    Loads the pandas dataframes from the results file
    :param path: exact path of the results file
    :return: 3 pandas dataframes, settings, statistics and results
    """
    import os
    from pandas import read_csv
    from io import StringIO
    from uncertainties import ufloat_fromstr
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

settings, statistics, results = load_results('endurance/Removed_Binary_plus_savgol_from_original/data/result.csv')
period = 1.66987725
rl = 0.002
fs = results['frequency'].values
amps = results['amp'].values
snrs = results['snr'].values
phis = results['phase'].values
freqs = []
amp = []
freqs_b = []
amp_b = []
freqs_la = []
amp_la = []
for i in range(len(fs)):
    if snrs[i] > 4 and not  ((2*rl > fs[i].n % (1/period)) or (fs[i].n % (1/period) >(1/period)-2*rl)) and amps[i].n > 0.0001:
        freqs.append(fs[i].n)
        amp.append(amps[i].n)
    elif snrs[i] > 4:
        freqs_b.append(fs[i].n)
        amp_b.append(amps[i].n)

print(len(freqs))
Ps = np.array(freqs)
difs = []
for i in range(len(Ps)):
    for k in range(i+1, len(Ps)):
        difs.append(abs(Ps[i]-Ps[k]))
difs = np.array(difs)


eps = 0.002
res, dPs = matrix_test(difs, 5, eps)
se =  2*eps/dPs*len(difs)
p= eps/dPs
fig, ax = plt.subplots(figsize = (10,7))
#ax.plot(dPs, (se-res), 'k-')
#print(se)
ax.plot(dPs, (se-res)/np.sqrt(se*(len(difs)-se)/len(difs)), 'k-')
#ax.plot(dPs, p, 'r-')
#ax.plot(dPs, se, 'g-')
#ax.plot(dPs, 0.02/dPs*len(Ps), 'b-')
#ax.plot(dPs, 0.01/dPs, 'g-')
ax.plot([0.1315, 0.1315], ax.set_ylim(), 'k-', lw = 3, alpha = 0.3)
ax.plot([0.4332, 0.4332], ax.set_ylim(), 'k-', lw = 3, alpha = 0.3)
ax.plot([0.3579, 0.3579], ax.set_ylim(), 'k-', lw = 3, alpha = 0.3)
plt.show()