# This code is a subset of MC_one
import os
import numpy as np
import matplotlib.pyplot as plt

""" MODEL AND INTEGRATION FUNCTIONS """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Definition of the Lorenz96 model
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is our model, L96 does not explicitly depend on time
def L96(x,F):
    # compute state derivatives
    d = np.zeros(D)
    # first the 3 edge cases (initial conditions)
    d[  0] = (x[1] - x[D-2]) * x[D-1] - x[  0]
    d[  1] = (x[2] - x[D-1]) * x[  0] - x[  1]
    d[D-1] = (x[0] - x[D-3]) * x[D-2] - x[D-1]
    # then the general case
    for i in range(2, D-1):
        d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
    # return the state derivatives + forcing term
    return d+F

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This function integrates x forward in time
# Updates the previous state variables using their time derivatives
# Here we are using the simple RK2/4, set up for L96 now, will be generalized later
# RK4 is used in data generation and RK2 is used (faster) in annealing
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def stepForward_RK4(x,F):
    a1 = L96(x            ,F)
    a2 = L96(x+a1*dt_gen/2,F)
    a3 = L96(x+a2*dt_gen/2,F)
    a4 = L96(x+a3*dt_gen  ,F)
    return x+(a1/6+a2/3+a3/3+a4/6)*dt_gen

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Performs the integration, gives us the full data set x
# Returns state as array
# Uses RK4 integration
# Ideally data should be generated for very small timesteps
# for high-quality data, then down-sampled and fed into the annealer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def generateData(x,F):
    xt = np.array(x)
    for k in range(1,M_gen):
        x  = stepForward_RK4(x,F)
        if k%int(dt/dt_gen) == 0:
            xt = np.vstack((xt,x))
    return xt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Discards the first few hundred data point
# Makes the system forget transients and sample around the attractor
# Same as generateData but nothing is saved
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def burnData(x,F):
    xt = np.array(x)
    for k in range(0,int(T/dt_gen)):
        x  = stepForward_RK4(x,F)
    return x

#####################################################
# Setup of Initial Conditions for Data Generation
#####################################################

# Using seed for repeatability
# np.random.seed(9001)

# Constants
F_real = 8.17
T = 5
dt_gen = 0.001
dt     = 0.025
M_gen = int(T/dt_gen)
M     = int(T/dt)
D = 5
times = np.transpose(np.array([np.arange(M)*dt]))

# Perturbing from equilibrium
x0_real = F_real*np.ones(D)+0.01*(np.random.rand(D)-0.5)

# Burning the first few hundred time steps
x0_real = burnData(x0_real,F_real)

# Y_real is the original data without noise
Y_real = generateData(x0_real,F_real)
# Output = np.hstack((times,Y_real))

print Y_real.shape
Y = Y_real + 1.0*(2.0*np.random.random_sample(Y_real.shape)-1.0)
np.savetxt('L96_D_5_T_5_dt_0p025.dat'  ,Y_real,fmt='%9.6f',delimiter='\t\t')
np.savetxt('L96_D_5_T_5_dt_0p025.noise',Y     ,fmt='%9.6f',delimiter='\t\t')

fig1, axs1 = plt.subplots(D, 1, sharex=True)
plt.suptitle('Generated Data');
fig1.subplots_adjust(hspace=0)
for i in range(0,D):
    axs1[i].plot(times,Y_real[:,i]);
    axs1[i].plot(times,Y     [:,i]);
plt.xlabel('TimeStep')
plt.show()
