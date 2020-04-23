import numpy as np
import matplotlib.pyplot as plt
from smurfs import Smurfs
from scipy.optimize import least_squares, curve_fit
from time import time
import math


def gen_data(t, *params):
    y = np.zeros(len(t))
    for i in range(0, len(params), 3):
        y += sin(t, params[i], params[i + 1], params[i + 2])
    #rnd = np.random.RandomState(0)
    #error = 0.000165 * rnd.randn(t.size)
    #outliers = rnd.randint(0, t.size, 10)
    #error[outliers] *= 10
    return y #+ error


def jac(fitParams, x, y):
    params = np.array(fitParams)
    As = params[::3]
    fs = params[1::3]
    offsets = params[2::3]
    phases = 2 * np.pi * (np.outer(x, fs) + offsets)
    sins = np.sin(phases)
    coss = np.cos(phases)

    jacA = sins
    jacF = x[:, np.newaxis] * coss * As * 2 * np.pi
    jacO = coss * As * 2 * np.pi

    jac = np.stack([jacA, jacF, jacO], 2).reshape((len(x), len(params)))
    return jac

def jactest(x, *fitParams):
    params = np.array(fitParams)
    As = params[::3]
    fs = params[1::3]
    offsets = params[2::3]
    phases = 2 * np.pi * (np.outer(x, fs) + offsets)
    sins = np.sin(phases)
    coss = np.cos(phases)

    jacA = sins
    jacF = x[:, np.newaxis] * coss * As * 2 * np.pi
    jacO = coss * As * 2 * np.pi

    jac = np.stack([jacA, jacF, jacO], 2).reshape((len(x), len(params)))
    return jac


def jac2(params, x, y):
    pi2 = 2. * np.pi
    r = []
    for j in range(len(x)):
        s = []
        for i in range(0, len(params), 3):
            fe = pi2 * (params[i + 1] * x[j] + params[i + 2])
            acosfe = params[i]*math.cos(fe)*pi2
            s.append(math.sin(fe))
            s.append(x[j] * acosfe)
            s.append(acosfe)
        r.append(s)
    return np.array(r)



def sin_multiple(x, *params):
    y = np.zeros(len(x))
    for i in range(0, len(params), 3):
        y += sin(x, params[i], params[i + 1], params[i + 2])
    return y

def sin(x, amp, f, phase):
    return amp * np.sin(2. * np.pi * (f * x + phase))


def fun(x, t, y):
    return sin_multiple(t, *x) - y

def fun2(x, t, y):
    return msin(t, x) - y

def msin(x, *fitParams):
    params = np.array(fitParams)
    As = params[::3]
    fs = params[1::3]
    offsets = params[2::3]
    #phases = 2 * np.pi * (np.outer(x, fs) + offsets)
    sins = np.sin(2 * np.pi * (np.outer(x, fs) + offsets))
    return np.dot(sins, As)

np.random.seed(1)
t_min = 0
t_max = 10
n_points = 50000
numf= 20
#fs = [15, 10.3654, 13.56848]
fs = np.random.uniform(10,20, numf)
#amps = [0.000165 *1.98, 0.000165 *2.68, 0.000165 *3.25]
amps = 0.000165 * np.random.uniform(0.5, 5, numf)
#phis = [0.12, 0.658, 0.486]
phis = np.random.uniform(0, 1, numf)
arr = []
for f, a, phi in zip(fs, amps, phis):
    arr.append(a)
    arr.append(f)
    arr.append(phi)
t_train = np.linspace(t_min, t_max, n_points)
y_train = msin(t_train, *arr)


#x0 = np.array([0.000165 *1.95,14.98,  0.5,0.000165 *2.6, 10.35,   0.5,0.000165 *3, 13.57,   0.5 ])
x0 = []
limits = [[], []]
for f, a, phi in zip(fs, amps, phis):
    x0.append(a *np.random.uniform(0.95, 1.05, 1)[0])
    limits[0].append(0.5*a)
    limits[1].append(1.5*a)
    x0.append(f *np.random.uniform(0.9975, 1.0025, 1)[0])
    limits[0].append(0.9*f)
    limits[1].append(1.1*f)
    x0.append(0.0)
    limits[0].append(-100)
    limits[1].append(100)



# start = time()
# res_lsq = least_squares(fun, x0, args=(t_train, y_train), bounds= limits)#, jac = jac)
# print(res_lsq.njev)
# print(time()-start)

# start = time()
# res_lsq = least_squares(fun2, x0, args=(t_train, y_train), bounds= limits)#, jac = jac)
# print(res_lsq.njev)
# print(time()-start)


popt, pcov = curve_fit(msin, t_train, y_train, p0=arr, bounds=limits, sigma=0.000165 * np.ones(len(t_train)),
                       absolute_sigma=True, jac=jactest)
print(popt)
start = time()
res_lsq_ana = least_squares(fun, x0, args=(t_train, y_train), jac = jac, bounds = limits)
print(res_lsq_ana.njev)
print(time()-start)



#print(res_lsq.optimality, res_lsq_ana.optimality)
#for i in res_lsq.jac:
#    print(i)
for i in range(len(arr)):
    print(np.round(arr[i], 5),  np.round(res_lsq_ana.x[i], 5))







plt.plot(t_train, y_train, 'ko', ms = 0.75)
ytest = gen_data(t_train, *res_lsq_ana.x)
plt.plot(t_train, ytest, 'b-', lw = 0.5)
plt.show()
