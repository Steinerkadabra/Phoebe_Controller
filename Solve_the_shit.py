import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
import scipy.misc as misc
import scipy.integrate as int

profile = '/Users/thomas/MESA/mesa-r12115/star/test_suite/1.5M_with_diffusion/LOGS/profile4.data'
model = np.genfromtxt(profile, skip_header=5, names = True)
mass_values = model['mass']
radius_values = model['radius']
m = inter.interp1d(radius_values, mass_values, fill_value= 'extrapolate', kind = 'cubic')

def dm_dr(r):
    return misc.derivative(m, r, dx = 10**(-8))

def y1(r, y):
    return y[0]

def y2(r, y):
    return 4/r**2*y[0] -2*(dm_dr(r)/m(r)-1/r)*y[1]

def f(r, y):
    return(np.array([y1(r, y), y2(r, y)]))

R1 = np.max(radius_values)
def rk(init):
    result = int.RK45(f, 0.00000001, np.array(init), R1, max_step=0.0001, first_step= 0.0000001 )
    rs_vals = []
    eps_vals = []
    eps_dot_vals = []
    while result.status == 'running':
        rs_vals.append(result.t)
        eps_vals.append(result.y[0])
        eps_dot_vals.append(result.y[1])
        result.step()

    eps = inter.interp1d(rs_vals, eps_dot_vals, fill_value= 'extrapolate', kind = 'cubic')
    eps_dot = inter.interp1d(rs_vals, eps_dot_vals, fill_value= 'extrapolate', kind = 'cubic')
    return eps, eps_dot

def eps_div_r(r):
    return eps(r)/r

def eps_div_r_dot(r):
    return misc.derivative(eps_div_r, r, dx = 10**(-8))

def bc(r):
    return eps_div_r_dot(r) + 2/r* eps_div_r(r) -5/(2*r)

fig, ax = plt.subplots(figsize = (14,8))
x = []
x_dot = []
bcs = []
for i in np.linspace(0.0008,0.00082, 1):
    for j in np.linspace(100, 500, 1):
        eps, eps_dot = rk(np.array([i, j]))
        x.append(i)
        x_dot.append(j)
        bcs.append(np.log10(abs(bc(R1))))
        vals = np.linspace(0, R1, 1000)
        ax.plot(vals, eps(vals), 'k-')
        #print(bc(R1))

#fu = plt.scatter(x, x_dot, c = bcs)#, vmin = -0.01, vmax = 0.01)
#plt.colorbar(fu)
plt.show()
#plt.plot(rs_vals, eps_vals, 'k-')
#plt.show()


