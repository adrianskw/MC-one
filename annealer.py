from libMCone import *
import matplotlib.pyplot as plt

""" EXAMPLE FUNCTION THAT CALLS CLASS FUNCTIONS """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is our annealing routine and the workhorse of the algorithm
# The point of this code is that the user is free to design their
# own annealing routine. Here is a preannealer as example:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# I will call the main annealer annealAll.. well because it anneals ALL of the variables
def annealAll(MC):
    # Useful Tallies
    count              = 0
    low_count          = 0
    block_count        = 0
    accept_count       = 0
    accept_block_count = 0
    # Quantities for adaptive delta
    lowestAction = MC.oldAction
    Xlow = np.copy(MC.Xold)
    deltaArray = np.array([])
    # Annealing Routine
    while(1):
        MC.perturbState()
        MC.evalSoftAcceptance()
        # Acceptance
        if(MC.isAccepted):
            MC.keepNewState()
            accept_count += 1 # For calculating acceptance rate
            accept_block_count += 1 # For adaptive delta
            # Saving the lowest action and associated path
            if(MC.oldAction) < lowestAction:
                Xlow = np.copy(MC.Xold)
                lowestAction = MC.oldAction
                low_count += 1
        else:
            MC.keepOldState()
        count += 1
        block_count += 1
        if (block_count == 100):
            deltaArray = np.append(deltaArray,MC.delta)
            MC.updateDelta(MC.delta*(1+0.3*(float(accept_block_count)/block_count-0.5)))
            block_count = 0 # Reseting the block-related tallies
            accept_block_count = 0 # Reseting the block-related tallies
        if (count>= MC.maxIt):
            break
    MC.replaceXold(Xlow)
    return accept_count,count,deltaArray

# I will call the preannealer annealSome.. well because it anneals SOME of the variables
def annealSome(MC):
    # Useful Tallies
    count              = 0
    low_count          = 0
    block_count        = 0
    accept_count       = 0
    accept_block_count = 0
    # Quantities for adaptive delta
    deltaArray = np.array([])
    # Annealing Routine
    while(1):
        MC.perturbState()
        MC.evalSoftAcceptance()
        if(MC.isAccepted):
            MC.keepNewState()
            accept_count += 1 # For calculating acceptance rate
            accept_block_count += 1 # For adaptive delta
        else:
            MC.keepOldState()
        count += 1
        block_count += 1
        if (block_count == 100):
            deltaArray = np.append(deltaArray,MC.delta)
            MC.updateDelta(MC.delta*(1+0.3*(float(accept_block_count)/block_count-0.5)))
            block_count = 0 # Reseting the block-related tallies
            accept_block_count = 0 # Reseting the block-related tallies
        if (count>= MC.maxIt):
            break
    return accept_count,count,deltaArray

""" EXAMPLE ROUTINE USING annealAll(MC) """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Quick example that should get you familiar with the code.
# By the way, 'overflow encountered in exp' error is expected.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Setting Random Seed (for repeatability)
np.random.seed(123)

# Model Constants
M = 200
D = 5
dt = 0.025
F = 8.3
Lidx = [0,2,4]
notLidx  = np.setxor1d(np.arange(D),Lidx)

# Twin Experiment Data
Y = np.loadtxt('./data/L96_D_5_T_5_dt_0p025.noise')
Z = np.loadtxt('./data/L96_D_5_T_5_dt_0p025.dat')

# Calculating Error
# NOTE: do this before over-writing unmeasured states
[realMeasError, realModelError] = calcRealError(Z,Y,M,D,dt,Lidx)

# Replacing unmeasured states with noise
for i in notLidx:
    Y[:,i] = 5.0*(2.0*np.random.rand()-1.0)*np.ones(M)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Preannealing Phase
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Preannealing Constants
Rm    = 1.0
Rf    = Rm/10.0
maxIt = 10*M*D
delta = 5.0
beta1  = 15

# Tracking Error and Action
measError1  = np.array([])
modelError1 = np.array([])
action1     = np.array([])
deltaArray1 = np.array([])
Xt1         = np.ones((beta1,M,D))

# Initializing the MC object
MC = Annealer(Y,L96,dt,F,Lidx,Rm,Rf,maxIt,delta,True)
Xinit = np.copy(MC.Xold)

# Initializing the MC object
for i in range(0,beta1):
    if 0<= i <= 3:
        MC.updateMaxIt(25000)
    else:
        MC.updateMaxIt(15000)
    [accept,count,deltaArr]=annealSome(MC)
    print 'beta =',i,'accept rate = ',float(accept)/count,'Rf = ',MC.Rf,'delta = ',MC.delta
    Xt1[i]      = MC.Xold
    deltaArray1 = np.append(deltaArray1,deltaArr)
    MC.updateRf(1.9*MC.Rf)

measError1  = np.copy(MC.measErrorArray)
modelError1  = np.copy(MC.modelErrorArray)
action1  = np.copy(MC.actionArray)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Main Annealing Phase
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Preannealing Constants
MC.updateRf(1.0)
MC.updateMaxIt(10*M*D)
MC.updateDelta(2.0)
MC.resetErrors()
MC.resetArrays()
MC.setPreFalse()
beta2  = 40

# Tracking Error and Action
measError2  = np.array([])
modelError2 = np.array([])
action2     = np.array([])
deltaArray2 = np.array([])
Xt2         = np.ones((beta2,M,D))

# Initializing the MC object
for i in range(0,beta2):
    [accept,count,deltaArr]=annealSome(MC)
    print 'beta =',i,'accept rate = ',float(accept)/count,'Rf = ',MC.Rf,'delta = ',MC.delta
    Xt2[i]      = MC.Xold
    deltaArray2 = np.append(deltaArray2,deltaArr)
    MC.updateRf(1.15*MC.Rf)

measError2  = np.copy(MC.measErrorArray)
modelError2  = np.copy(MC.modelErrorArray)
action2  = np.copy(MC.actionArray)
print "Final Forcing",MC.Fold

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Plotting Part
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
f, ax = plt.subplots(5, sharex=True, sharey=True)
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

plt.figure(0)
for i in range(0,5):
    plt.subplot(5,1,i+1)
    plt.plot(Z[:,i],'k',label='True Solution')
    plt.plot(Xinit[:,i],label='Initial Guess')
plt.suptitle('After Initialization Step')
plt.legend()

plt.figure(1)
for i in range(0,5):
    plt.subplot(5,1,i+1)
#     plt.plot(np.transpose(Xt1[:-2,:,i]))
    plt.plot(np.transpose(Xt1[-1,:,i]),label='Latest Proposed Solution')
    plt.plot(Z[:,i],'k',label='True Solution')
plt.suptitle('After Preannealing Step')
plt.legend()

plt.figure(2)
for i in range(0,5):
    plt.subplot(5,1,i+1)
#     plt.plot(np.transpose(Xt2[:-2,:,i]))
    plt.plot(np.transpose(Xt2[-1,:,i]),label='Latest Proposed Solution')
    plt.plot(Z[:,i],'k',label='True Solution')
plt.suptitle('After Main Annealing Step')
plt.legend()

plt.figure(3)
plt.semilogy(modelError1,label='pre')
plt.semilogy(modelError2,label='main')
plt.semilogy(realModelError*np.ones(max(modelError1.shape,modelError2.shape)),label='real')
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('Model Error')
plt.title('Model Error vs Total Iteration')
plt.legend()

plt.figure(4)
plt.semilogy(measError1,label='pre')
plt.semilogy(measError2,label='main')
plt.semilogy(realMeasError*np.ones(max(measError1.shape,measError2.shape)),label='real')
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('Meas Error')
plt.ylim([100,300])
plt.title('Meas Error vs Total Iteration')
plt.legend()

plt.figure(5)
plt.semilogy(action1,label='pre')
plt.semilogy(action2,label='main')
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('Action')
plt.title('Action vs Total Iteration')
plt.legend()

plt.figure(6)
plt.semilogy(deltaArray1,label='pre')
plt.semilogy(deltaArray2,label='main')
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('delta')
plt.title('delta vs Total Iteration')
plt.legend()

plt.show()
