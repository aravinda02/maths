'''Implementation of Euler's method with regards to a free partical moving
in Geodesic motion. The mathematical functions are as follows:
r is the radial coordinate
phi is the angular position
t is the coordinate time
tao is the partical time (as experience by the partical and not the coordinate)'
Veff is the effective potential, used to calculate the orbits of objects (in this case a partical)'''

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

M = 1 #mass of an object that the partical will fall into. using 1 for simplicity here
L = 4 #angular momentum per unit mass
E = 0.98 #energy per unit mass

r, tau = sp.symbols('r tau')

dphi = L / r**2
dt = E / (1 - (2*M / r))
dVeff = (2*M)/r**2 - (2*L**2)/r**3 + (6*M*L**2)/r**4 

#sp sqrt doesn't work with the previous version of eulers,
#hence I convert them to numeric functions
dVeff_func = sp.lambdify(r, dVeff, 'numpy')
dphi_func = sp.lambdify(r, dphi, 'numpy')
dt_func = sp.lambdify(r, dt, 'numpy')

def eulers_method(h, interval):
    '''Eulers method modified again to work with np instead of sp'''
    rj = 20.0
    uj = -np.sqrt(E**2 - (1 - (2*M)/rj)*(1 + (L**2)/(rj**2)))  
    phij = 0.0
    tj = 0.0
    tauj = interval[0]

    r_vals = []
    phi_vals = []
    t_vals = []
    tau_vals = []

    while tauj <= interval[1]:
        r_vals.append(rj)
        phi_vals.append(phij)
        t_vals.append(tj)
        tau_vals.append(tauj)

        dphi_val = float(dphi_func(rj))
        dt_val = float(dt_func(rj))
        du = -0.5 * float(dVeff_func(rj))

        rj += h * uj
        uj += h * du
        phij += h * dphi_val
        tj += h * dt_val
        tauj += h

    return np.array(r_vals), np.array(phi_vals), np.array(t_vals), np.array(tau_vals)

r_vals, phi_vals, t_vals, tau_vals = eulers_method(h=0.01, interval=[0, 20_000])
x_vals = r_vals * np.cos(phi_vals)
y_vals = r_vals * np.sin(phi_vals)

plt.plot(x_vals, y_vals)
plt.xlabel('x(tau)')
plt.ylabel('y(tau)')
plt.title('Trajectory')
plt.show()
