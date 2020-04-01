import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
import scipy.misc as misc
import scipy.integrate as int
import scipy.optimize as opt
import scipy.special as spc

'''
profile = '/Users/thomas/MESA/mesa-r12115/star/test_suite/1.5M_with_diffusion/LOGS/profile4.data'
model = np.genfromtxt(profile, skip_header=5, names = True)
mass_values = model['mass']
radius_values = model['radius']
m = inter.interp1d(radius_values, mass_values, fill_value= 'extrapolate', kind = 'cubic')
R1 = mass_values[0]
print(R1)

def m(r):
    return 4/3*np.pi*r**3

def dm_dr(r):
    return misc.derivative(m, r, dx = 10**(-8))

def y1(r, y):
    return y[1]

def y2(r, y):
    return 4/r**2*y[0] -2*(dm_dr(r)/m(r)-1/r)*y[1]

def fun(r, y):
    return np.vstack((y1(r, y), y2(r, y)))

def bc(ya, yb):
    return np.array([ya[0], yb[1]/R1 + yb[0]/R1**2 +5/2/R1])

'''

G =1#6.67430*10**(-11)
rho_val = 1 # 0.5490201231825738
min = 0

def create_derivative(x_vals, func):
    dif_vals = []
    for val in x_vals:
        dif_vals.append(misc.derivative(func, val, dx = 10**(-6)))
    return inter.interp1d(x_vals, dif_vals, fill_value= 'extrapolate', kind = 'cubic')


class stellar_model():
    xi_st : callable = None
    xi_st_div_r : callable = None
    eta_st : callable = None
    eta_st_div_r : callable = None


    def __init__(self, mass_profile):
        #self.x : np.ndarray= None
        #self.R1 = None
        self.mass_profile=  self.get_mass_profile(mass_profile)

    def get_mass_profile(self, profile):
        model = np.genfromtxt(profile, skip_header=5, names = True)
        mass_values = model['mass']
        radius_values = model['radius']
        self.m_dummy = inter.interp1d(radius_values, mass_values, fill_value= 'extrapolate', kind = 'cubic')
        self.R1 = radius_values[0]

    def m(self, r):
        return 4 / 3 * np.pi * r ** 3

    def dm_dr(self, r):
        return misc.derivative(self.m, r, dx=10 ** (-8))

    def y1(self, r, y):
        return y[1]

    def y2(self, r, y):
        return 4 / r ** 2 * y[0] - 2 * (self.dm_dr(r) / self.m(r) - 1 / r) * y[1]

    def fun(self, r, y):
        return np.vstack((self.y1(r, y), self.y2(r, y)))

    def bc(self, ya, yb):
        return np.array([ya[0], yb[1] / self.R1 + yb[0] / self.R1 ** 2 + 5 / 2 / self.R1])

    def solve_bvp(self):
        self.x = np.linspace(0.0000001, self.R1, 500)
        y_0 = np.zeros((2, self.x.size))
        y_0[0] = 10
        y_0[1] = 100
        res = int.solve_bvp(self.fun, self.bc, self.x, y_0, tol = 3e-7, max_nodes=np.inf)
        y = res.sol(self.x)[0]
        print(res.message)
        fig, ax = plt.subplots(figsize = (12, 6))
        ax.set_xlabel('Radius ($R_\odot$)')
        ax.set_ylabel(r'$\xi_\mathrm{st}$ / $\eta_\mathrm{st}$')
        self.xi_st = inter.interp1d(self.x, y, fill_value= 'extrapolate', kind = 'cubic')
        self.xi_st_div_r = create_derivative(self.x, self.xi_st)
        eta_vals = 1/6 * (2*self.x * self.xi_st(self.x) + self.x**2 * self.xi_st_div_r(self.x))
        self.eta_st= inter.interp1d(self.x, eta_vals, fill_value= 'extrapolate', kind = 'cubic')
        self.eta_st_div_r = create_derivative(self.x, self.eta_st)
        x_plot = np.linspace(0, self.R1, 1000)
        ax.plot(x_plot, self.eta_st(x_plot), 'g-')
        ax.plot(x_plot, self.xi_st(x_plot), 'k-')
        ax.plot(self.x, -5 / 4 * self.x, 'r--')
        ax.plot(self.x, 1/6 * -5 / 4 * 3 * self.x**2, 'r--')
        ax.plot()
        #plt.show()
        plt.close()

class Tidal_Perturbation():
    def __init__(self, n, l, star_object):
        self.n = n
        self.l = l
        self.star = star_object
        self.eps_nl_div_r = create_derivative(self.star.x, self.eps_nl)
        self.eta_nl_div_r = create_derivative(self.star.x, self.eta_nl)
        self.P0_div_r = create_derivative(self.star.x, self.P0)
        self.rho0_div_r = create_derivative(self.star.x, self.rho)
        self.g4_n_l_const = -((self.l+1)*self.star.xi_st(self.star.R1)-3*self.star.eta_st(self.star.R1)
                              /self.star.R1)*self.rho(self.star.R1)*self.eps_nl(self.star.R1)/self.star.R1**self.l
        self.phi_mark_n_l_div_r = create_derivative(self.star.x, self.phi_mark_n_l)
        self.sigma_n = np.sqrt(16*np.pi*G*rho_val/15)
        self.cos=create_derivative(self.star.x, np.sin)

    def P0(self, r):
        return 2*np.pi*G*self.rho(r)**2/3*self.star.R1**2*(1-r**2/self.star.R1**2)

    def c0(self, r): ##sound speed**2 = partial P / partial rho
        return np.sqrt(4*np.pi*G*self.rho(r)/3*self.star.R1**2*(1-r**2/self.star.R1**2))

    def eps_nl(self, r):
        return r

    def eta_nl(self, r):
        return r ** 2 / 2

    def rho(self, r):
        return rho_val

    def N_func(self, r):
        return self.rho(r)*(self.eps_nl(r)**2+self.l*(self.l+1)/r**2*self.eta_nl(r)**2)*r**2

    def N(self):
        return int.quad(self.N_func, min, self.star.R1)[0]

    def Y_l_m(self, l, m, theta, phi):   ### TESTED. WORKS###
        return spc.sph_harm(m, l, theta, phi)

    def A_l_func(self, theta, phi): ### TESTED. WORKS###
        return self.Y_l_m(self.l, 0, theta, phi) * np.conjugate(self.Y_l_m(self.l, 0, theta, phi)) * (
                    self.Y_l_m(2, 0, theta, phi) - self.Y_l_m(2, 2, theta, phi) / 4 - self.Y_l_m(2, -2, theta, phi) / 4)\
                    * np.sin(phi)

    def A_l(self):  ### TESTED. WORKS###
        return int.nquad(self.A_l_func, [[0, 2*np.pi], [0, np.pi]])[0]

    def h1_n_l(self, r):
        return 0 ##homology test case - analytic solution
        ### implementation seems correct ### 1d-7
        #return (2*self.star.xi_st(r) - 9 * self.star.eta_st(r)/r + 3*self.star.eta_st_div_r(r))*self.eps_nl(r) \
        #       + 3*(4*self.star.eta_st(r)/r-(1+self.l*(self.l+1)/3)*self.star.xi_st(r)-self.star.eta_st_div_r(r))*self.eta_nl(r)/r \
        #       + 3*(self.star.xi_st(r)-self.star.eta_st(r)/r)*self.eta_nl_div_r(r) #+ r**2 *self.star.xi_st_div_r(r)*self.eps_nl_div_r(r)

    def h2_n_l(self, r):
        return -35/16*r**3##homology test case - analytic solution
        ### implementation seems correct ### 1d-7
        #return r**2*self.star.xi_st_div_r(r)*self.eps_nl(r) + 3 * (self.star.xi_st(r)-self.star.eta_st(r)/r)*self.eta_nl(r)

    def F1_n_l_func(self, r):
        ### seems to work ### no real test possible
        return self.eps_nl(r)*self.P0_div_r(r)*(self.h1_n_l(r)+self.rho0_div_r(r)/self.rho(r)*self.h2_n_l(r))

    def F1_n_l(self):
        return -int.quad(self.F1_n_l_func, min, self.star.R1, limit = 500)[0]

    def alpha_n_l(self, r):
        ### works ### 1d-10
        return 0
        #return 2/r*self.eps_nl(r)+self.eps_nl_div_r(r)-self.l*(self.l+1)/r**2*self.eta_nl(r)

    def F2_n_l_func(self, r):
        return (self.alpha_n_l(r)*(self.rho(r)*self.c0(r)**2*self.h1_n_l(r)+self.P0_div_r(r)*self.h2_n_l(r)))

    def F2_n_l(self):
        return -int.quad(self.F2_n_l_func, min, self.star.R1, limit = 500)[0] + self.eps_nl(self.star.R1)\
               *(self.rho(self.star.R1)*self.c0(self.star.R1)**2*self.h1_n_l(self.star.R1)+
                 self.P0_div_r(self.star.R1)*self.h2_n_l(self.star.R1))

    def rho_mark_n_l(self, r):
        return -self.rho(r)*(self.alpha_n_l(r)+self.rho0_div_r(r)/self.rho(r)*self.eps_nl(r))

    def phi_mark_n_l_funk_below(self, r):
        return self.rho_mark_n_l(r)*r**(self.l+2)

    def phi_mark_n_l_funk_above(self, r):
        return self.rho_mark_n_l(r)*r**(-self.l+1)

    def phi_mark_n_l(self, r):
        return -4*np.pi*G/(2*self.l+1)*(r**(-self.l+1)*int.quad(self.phi_mark_n_l_funk_below, min, r)[0] + r**self.l
                                        * int.quad(self.phi_mark_n_l_funk_above, r, self.star.R1)[0]
                                        + self.rho(self.star.R1)*self.eps_nl(self.star.R1)/self.star.R1**(self.l-1))

    def g1_n_l_funk_below(self, r):
        return self.rho_mark_n_l(r)*r**(self.l+1)*(self.l*self.star.xi_st(r)+3*self.star.eta_st(r)/r)

    def g1_n_l_funk_above(self, r):
        return self.rho_mark_n_l(r)*r**(-self.l)*(-(self.l+1)*self.star.xi_st(r)+3*self.star.eta_st(r)/r)

    def g1_n_l(self, r):
        return r**(-(self.l+1))*int.quad(self.g1_n_l_funk_below, min, r, limit = 500)[0]\
               + r**self.l*int.quad(self.g1_n_l_funk_above, r, self.star.R1, limit = 500)[0]

    def g2_n_l_funk_below(self, r):
        return self.rho(r)*r**(self.l-1)*self.h2_n_l(r)

    def g2_n_l_funk_above(self, r):
        return self.rho(r)*r**(-self.l-2)*self.h2_n_l(r)

    def g2_n_l(self, r):
        return self.l*r**(-self.l-1)*int.quad(self.g2_n_l_funk_below, min, r)[0]\
               - (self.l+1)*r**self.l*int.quad(self.g2_n_l_funk_above, r, self.star.R1)[0]

    def h3_n_l(self, r):
        return 3*(self.star.eta_st_div_r(r)/r-self.star.eta_st(r)/r**2)*self.eps_nl(r)\
               + ((self.l*(self.l+1)-3)*self.star.xi_st(r)+9*self.star.eta_st(r)/r)*self.eta_nl(r)/r**2

    def g3_n_l_funk_below(self, r):
        return self.rho(r)*r**(self.l+1)*self.h3_n_l(r)

    def g3_n_l_funk_above(self, r):
        return self.rho(r)*r**(-self.l)*self.h3_n_l(r)

    def g3_n_l(self, r):
        return r**(-self.l-1)*int.quad(self.g3_n_l_funk_below, min, r)[0] \
               + r**self.l*int.quad(self.g3_n_l_funk_above, r, self.star.R1)[0]

    def g4_n_l(self, r):
        return self.g4_n_l_const*r**self.l

    def f3_n_l_func_part(self, r):
        return (self.star.xi_st(r)*self.phi_mark_n_l_div_r(r)+3*self.star.eta_st(r)*self.phi_mark_n_l(r)/r**2)\
               -4*np.pi*G/(2*self.l+1)*(self.g1_n_l(r)-self.g2_n_l(r)-self.g3_n_l(r)+self.g4_n_l(r))

    def f3_n_l_func(self, r):
        return self.rho_mark_n_l(r)*r**2*self.f3_n_l_func_part(r)

    def F3_n_l(self):
        return self.rho(self.star.R1)*self.star.R1**2*self.eps_nl(self.star.R1)*self.f3_n_l_func(self.star.R1)\
               +int.quad(self.f3_n_l_func, min, self.star.R1)[0]

    def F4_n_l_func(self, r):
        return -self.rho(r)*(self.eps_nl(r)*self.h2_n_l(r)+r*self.eta_nl(r)*self.h3_n_l(r))

    def F4_n_l(self):
        return int.quad(self.F4_n_l_func, min, self.star.R1)[0]

    def H_n_0_n_0(self):
        return self.A_l()*(2*self.l+1)/(4*np.pi*self.N())*((self.F1_n_l()+self.F2_n_l()+self.F3_n_l())/self.sigma_n**2+self.F4_n_l())


primary = stellar_model('/Users/thomas/MESA/mesa-r12115/star/test_suite/1.5M_with_diffusion/LOGS/profile4.data')
primary.solve_bvp()
print(primary.R1)
primary_10_2 = Tidal_Perturbation(10, 2, primary)


x = primary.x
x_f = np.linspace(np.min(x), np.max(x), 2000)
#y = []
#for i in x:
#    y.append(primary_10_2.phi_mark_n_l(i))
#plt.plot(x, y, 'ro')
fig, ax = plt.subplots(2,1)
implemented = primary_10_2.F2_n_l_func(x_f)
expected = 0*x_f

ax[0].plot(x_f,implemented , 'ko')
ax[0].plot(x_f, expected, 'r-')
ax[1].plot(x_f,expected -  implemented, 'r-')
plt.show()

print(primary_10_2.F2_n_l())
print(4*np.pi/3*35/16*primary.R1**4)
#plt.plot(x_f, primary_10_2.g1_n_l(x_f), 'g-')
#plt.show()

#print(primary_10_2.sigma_n)
#print(primary_10_2.A_l())
#result = primary_10_2.H_n_0_n_0()
#print(result, result / (65/28))


