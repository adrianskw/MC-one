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
# Parsing the input arguments
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def initParser():
    if (len(sys.argv)!=5) or isinstance(sys.argv[1],int) or isinstance(sys.argv[2],int):
        print("Please enter proper arguments:")
        print("D integer")
        print("M integer")
        print("dt float")
        print("L_frac 0<float<1")
        sys.exit(0)
    global D
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
# This function integrates x forward in time
# Updates the previous state variables using their time derivatives
# Here we are using the simple RK4, set up for L96 now, will be generalized later
# This is generally used for data generation
# Will be generalized for use in the annealing steps at later time
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def stepForward_RK4(x,F):
    a1 = L96(x        ,F)
    a2 = L96(x+a1*dt/2,F)
    a3 = L96(x+a2*dt/2,F)
    a4 = L96(x+a3*dt  ,F)
    return x+(a1/6+a2/3+a3/3+a4/6)*dt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Performs the integration, gives us the full data set x
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
# Discards the first few hundred data point
# Makes the system forget transients and sample around the attractor
# Same as generateData but nothing is saved
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def burnData(x,F):
    xt = np.array(x)
    for k in range(0,500):
        x  = stepForward_RK4(x,F)
    return x

""" ANNEALING FUNCTIONS """
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
            error+=pointMeasError(X,i,j)
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
    # for i in range(0,M-1):
    #     error +=abs((X[i+1,:]-X[i,:])/dt - L96(X[i,:],F))**2
    # error +=abs((X[M-1,:]-X[M-2,:])/dt - L96(X[M-1,:],F))**2
    for i in range(0,M):
        error += pointModelError(X,F,i)
    return error

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is the point-wise Measurement Error which is sometimes more useful
# This is also called in calcMeasError()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def pointMeasError(X,i,j):
    return abs(X[i,j]-Y[i,j])**2

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is the point-wise Model Error which is sometimes more useful
# This is called in calcModelError() above
# Again, the discretization here is very flawed since it is not conservative
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def pointModelError(X,F,idx):
    # NOTE: idx is the time index, aka n
    if idx == M-1: # if we picked the last element
        error =abs((X[M-1,:]-X[M-2,:])/dt - (L96(X[M-1,:],F)))**2
    elif idx == 0: # if we picked the first element
        error =abs((X[1,:]-X[0,:])/dt - (L96(X[0,:],F)))**2
    else: # else we have to vary in both directions
        error =abs((X[idx+1,:]-X[idx,:])/dt - (L96(X[idx,:],F)))**2   \
                +abs((X[idx,:]-X[idx-1,:])/dt - (L96(X[idx-1,:],F)))**2
    return sum(error)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is our annealing routine and the workhorse of the algorithm
# Right now, this is taken for F as a constant-in-time scalar which
# will be generalized for vector F later on
# For now, the code uses a 'hard' rejection criterion in the MC
# It can be generalized for the 'soft/probabilistic' rejection with
# little effort, which is said to be better for many problems
# NOTE1: Lidx and notLidx are called from the global definition, be careful
# NOTE2: For now, flag = True tell the code to anneal F which is buggy
#        and will be fixed later
# Please note the flags ***
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
    Action = Rm*calcMeasError(X)+Rf*calcModelError(X,F)

    # Here I am doing this a fixed number of times instead of a while loop for now
    while(1):

        # Choose a random element of the trajectory
        if flag == 0:
            idx2 = np.random.choice(np.arange(D))
        elif flag == 1:
            idx2 = np.random.choice(notLidx)
        elif flag == 2: # Not Working Ideally this should be the deafult flag
            idx2 = np.random.choice(np.arrange(D+1))
        elif flag == 3: # Not Working
            idx2 = np.random.choice([notLidx,D])
        else:
            print "ERROR: Undefined Flag"

        # Propose a new path, by perturbing the chosen element
        if idx2 == D:
            F_new = F + 0.02*jump*(np.random.rand(1)-0.5)
            X_new = np.copy(X)
        elif idx2 in Lidx: # perturbing the observed states
            idx  = np.random.choice(np.arange(M))
            F_new = np.copy(F)
            X_new[idx,idx2] = X[idx,idx2] + 0.05*jump*(2.0*np.random.rand(1)-1.0)
        else: # perturbing the unobserved states
            idx  = np.random.choice(np.arange(M))
            F_new = np.copy(F)
            X_new[idx,idx2] = X[idx,idx2] + jump*(2.0*np.random.rand(1)-1.0)

        # Calculate change in the Action
        Action_new = Rm*pointMeasError(X_new,idx,idx2)+Rf*pointModelError(X_new,F_new,idx)
        Action_old = Rm*pointMeasError(X,idx,idx2) + Rf*pointModelError(X,F,idx)
        delta =  Action_new - Action_old

        # If the trial Action is lower, accept it
        if(delta<0):
            if idx2==D:
                F = F_new
            else:
                X[idx,idx2] = X_new[idx,idx2]
            Action += delta
            # Reseting exit count
            exit_count = 0
            accept_count += 1
        else:
            if idx2 == D:
                F_new = F
            else:
                X_new[idx,idx2] = X[idx,idx2]
            # One strike for exit count
            exit_count += 1

        # After a certain number of consecutive failures, let's say that we are at a local min
        if exit_count == it:
            # Exit the while loop
            break

        count += 1
#     print 'accepted steps:',accept_count
#     print 'acceptance rate:',format(float(accept_count)/count*100.0, '.2f'),'%'
    return X,F,Action


#####################################################
# Setup of Initial Conditions for Data Generation
#####################################################

# Using seed for repeatability
# np.random.seed(9001)

# Constants
F = 8.17
M = 50
dt = 0.025
D = 5
Lidx = [0]
notLidx = [1,2,3,4]

# Perturbing from equilibrium
x0_real = F*np.ones(D)+(np.random.rand(D)-0.5)

# Burning the first few hundred time steps
x0_real = burnData(x0_real,F)

# ytreal is the original data without noise
Y_real = generateData(x0_real,F)

# yt is the original data with noise
Y = Y_real+1.0*(np.random.random_sample((M,D))-0.5)


#####################################################
# Setting up Initial Conditions for Annealing
#####################################################

F_init = F
X_init = (np.random.random_sample((M,D))-0.5)*20.0
X_init[:,Lidx]=Y[:,Lidx]

X = np.copy(X_init)
Action = np.array([])
ModelAction = np.array([])
MeasAction = np.array([])
F = np.array([F_init])

Rm = 0.05
Rf = 10**-8
jump = 0.5
it = 40
steps = 80
Xt = np.ones((steps,M,D))
alpha = 1.22

# Starting Timer
start = time.time()

# Initiating the Path and Action
[X,F1,Action1]=annealAll(X,F[-1],Rm,Rf,jump,it,1) # flag set to 1 for burn-in
MeasAction = np.append(MeasAction,Rm*calcMeasError(X))
ModelAction = np.append(ModelAction,Rf*calcModelError(X,F1))
Action = np.append(Action,Action1) # Here I just want to track the Action vs annealing step
F = np.append(F,F1) # Here I just want to track the Forcing term vs annealing step

#####################################################
# Annealling and Plotting
#####################################################

print "Annealing Starting"
print "Rf0 =", Rf
for i in range(0,steps):
    if (i+1)%10==0:
        print 'Anneal Step:', i+1
    Rf = alpha*Rf
    [X,F1,Action1]=annealAll(X,F[-1],Rm,Rf,jump,it,0) # flag set to 0, default
    MeasAction = np.append(MeasAction,Rm*calcMeasError(X))
    ModelAction = np.append(ModelAction,Rf*calcModelError(X,F1))
    Action = np.append(Action,Action1) # Here I just want to track the Action vs annealing step
    F = np.append(F,F1) # Here I just want to track the Forcing term vs annealing step
    Xt[i]=X

fig1, axs1 = plt.subplots(D, 1, sharex=True)
plt.suptitle('End of Annealling Phase');
fig1.subplots_adjust(hspace=0)
for i in range(0,D):
    axs1[i].plot(X[:,i],label='Estimated');
    axs1[i].plot(Y_real[:,i],label='Actual');
plt.legend()

#####################################################
# Timing and Misc Plotting
#####################################################

end = time.time()
print 'Rf Final:', Rf
print 'MeasAction Final:', MeasAction[-1]
print 'ModelAction Final:', ModelAction[-1]
print 'elapsed time:', end-start,'sec'

plt.figure(D)
plt.suptitle('Log(Action) through Annealing Process');
plt.plot(np.log10(abs(Action)),label='Action');
plt.plot(np.log10(abs(MeasAction+0.01)),label='MeasAction');
plt.plot(np.log10(abs(ModelAction)),label='ModelAction');
plt.legend()

fig2, axs2 = plt.subplots(D, 1, sharex=True)
plt.suptitle('Path Evolution through Annealing Process');
fig2.subplots_adjust(hspace=0)
for i in range(0,D):
    axs2[i].plot(np.transpose(Xt[:,:,i]));
    axs2[i].plot(Y_real[:,i],color='k',label='Actual');
plt.legend()

plt.show()
