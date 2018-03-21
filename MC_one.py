import os
import numpy as np
import matplotlib.pyplot as plt
import time

""" IO SUPPORT FUNCTIONS """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Functions for quality of life improvements
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Disable all print statements
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Enable all print statements
def enablePrint():
    sys.stdout = sys.__stdout__

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Function to quickly initiate batch jobs
# Parses the input arguments
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def initParser():
    if (len(sys.argv)!=5) or isinstance(sys.argv[1],int) or isinstance(sys.argv[2],int):
        print("Please enter proper arguments:")
        print("D integer")
        print("M integer")
        print("dt float")
        print("L_frac 0<float<1")
        sys.exit(0)
    global D2
    global M
    global dt
    global L_frac
    D,M,dt,L_frac  = np.asarray(sys.argv[1:])
    D = D.astype(int)
    M = M.astype(int)
    dt = dt.astype(float)
    L_frac = L_frac.astype(float)
    print "Running annealSW for D = "+str(D)+ ", M = "+str(M)+ \
            ", dt = "+str(dt)+", L_frac = "+str(L_frac)

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
# These functions integrate x forward in time
# Updates the previous state variables using their time derivatives
# Here we are using the simple RK2/4, set up for L96 now, will be generalized later
# RK4 is used in data generation and RK2 is used (faster) in annealing
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def stepForward_RK4(x,F):
    a1 = L96(x        ,F)
    a2 = L96(x+a1*dt/2,F)
    a3 = L96(x+a2*dt/2,F)
    a4 = L96(x+a3*dt  ,F)
    return x+(a1/6+a2/3+a3/3+a4/6)*dt

# Used in annealing only, quite a lot faster and 'good enough'
def stepForward_RK2(x,F):
    a1 = L96(x        ,F)
    a2 = L96(x+a1*dt/2,F)
    return x+a2*dt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Performs entire integration, gives us the full data set x
# Returns state as array
# Uses RK4 integration
# Ideally data should be generated for very small timesteps
# for high-quality data, then down-sampled and fed into the annealer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def generateData(x,F):
    xt = np.array(x)
    for k in range(0,M-1):
        x  = stepForward_RK4(x,F)
        xt = np.vstack((xt,x))
    return xt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Discards the first few hundred data points
# Makes the system forget transients and sample around the attractor
# Same as generateData but nothing is saved
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def burnData(x,F):
    xt = np.array(x)
    for k in range(0,1000):
        x  = stepForward_RK4(x,F)
    return x

""" ANNEALING AND ITS SUPPORT FUNCTIONS """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Calculates the Measurement Error for the entire time-series
# We only sum over Lidx (observed variables)
# NOTE1: This does not take into account Rm
# NOTE2: Y is not pass as an argument and needs to be defined globally
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def calcMeasError(X):
    error = 0.0
    for i in range(0,M):
        for j in Lidx:
            error+=pointmeasError(X,i,j)
    return error

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Calculates the Model Error for the entire time-series
# We have to sum over all variables
# NOTE1: This does not take into account Rf
# NOTE2: Right now this uses a weird and inconsistent discretization
#        which is (selectively) either forward- or backward- Euler
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def calcModelError(X,F):
    error = 0.0
    for i in range(0,M):
        error += pointmodelError(X,F,i)
    return error

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is the point-wise Measurement Error which is sometimes more useful
# This is also called in calcMeasError()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def pointmeasError(X,i,j):
    if j in Lidx:
        return abs(X[i,j]-Y[i,j])**2
    else:
        return 0

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is the point-wise Model Error which is more useful
# This is called in calcModelError() above
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def pointmodelError(X,F,idx):
    youWantSlowCode = False
    if youWantSlowCode == True:
        # Integral Version (Fourth Order RK)
        if idx == 0: # if we picked the first element
            error =  abs(X[1,:]-stepForward_RK4(X[0,:],F))**2
        elif idx == M-1: # if we picked the last element
            error =  abs(X[M-1,:]-stepForward_RK4(X[M-2,:],F))**2
        else: # else we have to vary in both directions
            error =  abs(X[idx+1,:]-stepForward_RK4(X[idx-0,:],F))**2 \
                    +abs(X[idx+0,:]-stepForward_RK4(X[idx-1,:],F))**2
        return sum(error)
    else:
        # Integral Version (Second Order RK), really this is good enoough
        if idx == 0: # if we picked the first element
            error =  abs(X[1,:]-stepForward_RK2(X[0,:],F))**2
        elif idx == M-1: # if we picked the last element
            error =  abs(X[M-1,:]-stepForward_RK2(X[M-2,:],F))**2
        else: # else we have to vary in both directions
            error =  abs(X[idx+1,:]-stepForward_RK2(X[idx-0,:],F))**2 \
                    +abs(X[idx+0,:]-stepForward_RK2(X[idx-1,:],F))**2
        return sum(error)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is our annealing routine and the workhorse of the algorithm
# Right now, this is taken for F as a constant-in-time scalar which
# will be generalized for vector F later on
# For now, the code uses a 'hard' rejection criterion in the MC
# It can be generalized for the 'soft/probabilistic' rejection with
# little effort, which is said to be better for many problems
# NOTE1: Lidx and notLidx are called from the global definition, be careful
# The difference is that this perturbs the measured state as well as
# the unmeasured one, which has show better results empirically
# This is, in a sense, stage 2 of the annealing, with annealAll() is the first stage
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def annealAll(X,F,Rm,Rf,jump,it,flag):
    exit_count = 0
    accept_count = 0
    count = 0

    # Just copying the shapes
    X_new = np.copy(X)
    F_new = np.copy(F)

    # Calculating the initial action for inputs
    measError  = calcMeasError(X)
    modelError = calcModelError(X,F)
    Action     = Rm*measError+Rf*modelError

    # Here I am doing this a fixed number of times instead of a while loop for now
    while(1):

        # Choose a random element of the trajectory
        if flag == 0: # Only perturbing unmeasured states (experimental)
            idx2 = np.random.choice(notLidx)
        elif flag == 1: # Purturbing all states (standard)
            idx2 = np.random.choice(np.arange(D))

        # Propose a new path, by perturbing the chosen element
        if idx2 == D: # perturbing forcing parameter (not doing this yet)
            F_new = F + 0.1*jump*(np.random.rand(1)-0.5)
            X_new = np.copy(X)
            # Calculate change in the Action
            delta_measError = 0
            delta_modelError = calcModelError(X_new,F_new)-calcModelError(X,F)
        else: # perturbing states only
            idx  = np.random.choice(np.arange(M))
            F_new = np.copy(F)
            X_new[idx,idx2] = X[idx,idx2] + jump*(2.0*np.random.rand(1)-1.0)
            # Calculate change in the Action
            delta_measError  = pointmeasError(X_new,idx,idx2) - pointmeasError(X,idx,idx2)
            delta_modelError = pointmodelError(X_new,F_new,idx)-pointmodelError(X,F,idx)

        delta_Action =  Rm*delta_measError + Rf*delta_modelError

        # If the trial Action is lower, accept it probabilistically
        if(np.exp(-delta_Action)>np.random.rand()):
            if idx2==D:
                F = F_new
            else:
                X[idx,idx2] = X_new[idx,idx2]

            measError  += delta_measError
            modelError += delta_modelError
            Action     += delta_Action
            # Reseting exit count
            exit_count = 0
            # For calculating acceptance rate
            accept_count += 1
        else:
            if idx2 == D:
                F_new = F
            else:
                X_new[idx,idx2] = X[idx,idx2]
            # One strike for exit count
            exit_count += 1

        # Number of times the while loop has run
        count += 1

        # After a certain number of consecutive failures, let's say that we are at a local min
        if (exit_count == it):
            # Exit the while loop
            break
        if (count == 5*10**4):
            print "Max Iterations Hit!"
            # Exit the while loop
            break
    return X,F,measError,modelError,Action,accept_count,count

#####################################################
# Setup of Initial Conditions for Data Generation
#####################################################
# Using seed for repeatability
# np.random.seed(9001)

# Constants
F_real = 8.17
M = 200
dt = 0.025
D = 5
t = np.arange(M)*dt

#####################################################
# Setting up Initial Conditions for Annealing
#####################################################

# Annealing Constants
beta  = 50
alpha = 2.0
jump  = 5.1
damp  = 1.2
Rm    = 0.05
Rf0   = 0.01
Rf    = Rf0*alpha**np.arange(beta)
it    = int(float(M*D)/10.0)

# Initial Values
F_init = F_real

# Loading Data
Y      = np.loadtxt('L96_D_5_T_5_dt_0p025.noise')

# Replacing unmeasured states with noise
Y[:,0] = 5.0*(2.0*np.random.rand()-1.0)*np.ones(M)
# Y[:,1] = 5.0*(2.0*np.random.rand(M)-1.0)*np.ones(M)
Y[:,2] = 5.0*(2.0*np.random.rand()-1.0)*np.ones(M)
# Y[:,3] = 5.0*(2.0*np.random.rand(M)-1.0)*np.ones(M)
# Y[:,4] = 5.0*(2.0*np.random.rand()-1.0)*np.ones(M)

# NOTE: Change this accordingly when deciding which states are measured
Lidx = [1,3,4]
notLidx = [0,2]

# Initializing the Path and Action (Rf = 0 step)
X  = np.copy(Y)
F  = np.array([F_init])
Xt = np.ones((beta,M,D))

#####################################################
# Rf = 0
#####################################################
measError   = np.array([])
modelError  = np.array([])
modelError2 = np.array([])
Action      = np.array([])

#####################################################
# Annealling First One
#####################################################
start = time.time()

accept = 0
count = 0
print "Annealing All States"
print "Rf0 =", Rf0
for i in range(1,beta+1):
    flag = 1 # Experiemental Stuff, Don't worry about it
    [X,F1,measError1,modelError1,Action1,accept1,count1]=annealAll(X,F[-1],Rm,Rf0,jump,it,flag)
    # Appending Quantities of Interest
    measError   = np.append(measError,measError1)
    modelError = np.append(modelError,calcModelError(X,F1))
    Action      = np.append(Action,Action1)
    F           = np.append(F,F1)
    # Modifying parameters for next annealing step
    Rf0     = alpha*Rf0
    jump    = jump/damp
    Xt[i-1]   = X # appending paths
    count  += count1
    accept += accept1
    it     += 1
    if i%5==0:
        print 'Anneal Step:', i
        print 'Accetance Rate:', float(accept)/count*100.0,'%'
        print 'Trials:', count
        count = 0
        accept = 0

#####################################################
# Timing and Misc Plotting
#####################################################

print 'Rf Final:', Rf0/alpha
print 'jump Final:', jump*damp
print 'measError Final:', measError[-1]
print 'modelError Final:', modelError[-1]
print 'TotalAction Final:', Action[-1]
print 'Elapsed Time:', time.time()-start,'sec'

Y_real = np.loadtxt('L96_D_5_T_5_dt_0p025.dat')
np.savetxt('L96_D_5_T_5_dt_0p025.firstanneal',X,fmt='%9.6f',delimiter='\t\t')

fig1, axs1 = plt.subplots(D, 1, sharex=True)
plt.suptitle('End of Annealling Phase');
fig1.subplots_adjust(hspace=0)
for i in range(0,D):
    axs1[i].plot(X[:,i],label='Estimated');
    axs1[i].plot(Y_real[:,i],label='Actual');
plt.xlabel('TimeStep')
plt.legend()

plt.figure(D)
plt.suptitle('LogLog through Annealing Process');
plt.loglog(Rf,abs(measError),label='measError = Sum[(x-y)^2]');
plt.loglog(Rf,calcMeasError(Y_real)*np.ones(Action.size),label='measError Actual');
plt.loglog(Rf,abs(modelError),label='modelError = Sum[(xdot-F(x))^2]');
plt.loglog(Rf,calcModelError(Y_real,F_real)*np.ones(Action.size),label='modelError Actual');
plt.loglog(Rf,abs(Action),label='TotalAction = Rm*measError + Rf*modelError');
plt.xlabel('AnnealStep')
plt.legend()

# plt.figure(D+1)
# plt.suptitle('Forcing Parameter vs Annealin Step');
# plt.plot(F);
# plt.xlabel('AnnealStep')

plt.figure(D+2)
plt.suptitle('Error in Variables');
for i in range(0,D):
    plt.plot(X[:,i]-Y_real[:,i],label = 'x'+str(i));
plt.xlabel('TimeStep')
plt.legend()

fig2, axs2 = plt.subplots(D, 1, sharex=True)
plt.suptitle('Path Evolution through Annealing Process');
fig2.subplots_adjust(hspace=0)
for i in range(0,D):
    axs2[i].plot(np.transpose(Xt[:,:,i]));
    axs2[i].plot(Y_real[:,i],color='k',label='Actual');
plt.xlabel('TimeStep')
plt.legend()


plt.show()
