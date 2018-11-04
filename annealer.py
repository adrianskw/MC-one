from libMCone import *
import matplotlib.pyplot as plt
import sys

""" EXAMPLE FUNCTION THAT CALLS CLASS FUNCTIONS """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is our annealing routine and the workhorse of the algorithm
# The point of this code is that the user is free to design their
# own annealing routine. Here is a preannealer as example:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is the annealer function used for both the pre- and main- annealing
# Three different versions
def annealLatest(MC):
    # Useful Tallies
    count              = 0
    low_count          = 0
    accept_count       = 0
    # Quantities for adaptive delta
    deltaArray = np.array([])
    block_count        = 0
    accept_block_count = 0
    # Annealing Routine
    while(1):
        MC.perturbState()
        MC.evalSoftAcceptance()
        if(MC.isAccepted):
            MC.keepNewState()
            accept_count       += 1 # For calculating acceptance rate
            accept_block_count += 1 # For adaptive delta
        else:
            MC.keepOldState()
        count += 1
        block_count += 1
        # Adaptive delta subroutine
        if (block_count == 200):
            MC.updateDelta(MC.delta*(1+0.3*(float(accept_block_count)/block_count-0.5)))
            deltaArray         = np.append(deltaArray,MC.delta)
            block_count        = 0 # Reseting the block-related tallies
            accept_block_count = 0 # Reseting the block-related tallies
        if (count>= MC.maxIt):
            break
    return accept_count,count,deltaArray,MC.Xold,MC.Fold

def annealAverage(MC):
    # Useful Tallies
    count              = 0
    low_count          = 0
    accept_count       = 0
    # Quantities for adaptive delta
    deltaArray = np.array([])
    block_count        = 0
    accept_block_count = 0
    # Annealing Routine
    Xtot = np.zeros(MC.Xold.shape)
    Ftot = np.zeros(MC.Fold.shape)
    while(1):
        MC.perturbState()
        MC.evalSoftAcceptance()
        if(MC.isAccepted):
            MC.keepNewState()
            accept_count       += 1 # For calculating acceptance rate
            accept_block_count += 1 # For adaptive delta
        else:
            MC.keepOldState()
        Xtot += MC.Xold
        Ftot += MC.Fold
        count += 1
        block_count += 1
        # Adaptive delta subroutine
        if (block_count == 200):
            MC.updateDelta(MC.delta*(1+0.3*(float(accept_block_count)/block_count-0.5)))
            deltaArray         = np.append(deltaArray,MC.delta)
            block_count        = 0 # Reseting the block-related tallies
            accept_block_count = 0 # Reseting the block-related tallies
        if (count>= MC.maxIt):
            break
    return accept_count,count,deltaArray,Xtot/count,Ftot/count

def annealLowest(MC):
    # Useful Tallies
    count              = 0
    low_count          = 0
    accept_count       = 0
    # Quantities for adaptive delta
    deltaArray = np.array([])
    block_count        = 0
    accept_block_count = 0
    # Annealing Routine
    Xlow = np.copy(MC.Xold)
    Flow = np.copy(MC.Fold)
    while(1):
        lowestAction = MC.oldAction
        MC.perturbState()
        MC.evalSoftAcceptance()
        if(MC.isAccepted):
            MC.keepNewState()
            accept_count       += 1 # For calculating acceptance rate
            accept_block_count += 1 # For adaptive delta
        else:
            MC.keepOldState()
        if(MC.oldAction < lowestAction):
            Xlow = np.copy(MC.Xold)
            Flow = np.copy(MC.Fold)
        count += 1
        block_count += 1
        # Adaptive delta subroutine
        if (block_count == 200):
            MC.updateDelta(MC.delta*(1+0.3*(float(accept_block_count)/block_count-0.5)))
            deltaArray         = np.append(deltaArray,MC.delta)
            block_count        = 0 # Reseting the block-related tallies
            accept_block_count = 0 # Reseting the block-related tallies
        if (count>= MC.maxIt):
            break
    return accept_count,count,deltaArray,Xlow,Flow

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
F = 8.5 #not a vector, so 'const'
Lidx = [0,2,4]
notLidx  = np.setxor1d(np.arange(D),Lidx)
NL = len(Lidx)

# Twin Experiment Data
Z = np.loadtxt('./data/L96_D5_Fconst_truepath.dat')[0:M,:]
Y = Z + (2.0*np.random.rand(M,D)-1.0)

# Calculating Error
# NOTE: do this before over-writing unmeasured states
[realMeasError, realModelError] = calcRealError(Z,Y,M,D,dt,Lidx)

# Replacing unmeasured states with noise
for i in notLidx:
    Y[:,i] = 10.0*(2.0*np.random.rand()-1.0)*np.ones(M)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Preannealing Phase
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Preannealing Constants
Rm    = 3.0
Rf    = Rm/10.0
maxIt = 20*M*D
delta = 5.0
beta1  = 10
alpha1 = 1.8

# Tracking Error and Action
deltaArray1 = np.array([])
FArray1     = np.array([])
Xt1         = np.ones((beta1,M,D))
Ft1         = np.ones(beta1)

# Initializing the MC object
MC = Annealer(Y,L96,dt,F,Lidx,Rm,Rf,maxIt,delta,True,True,True)
Xinit = np.copy(MC.Xold)

print 'Sit tight. This should take no more than 10 mins to run. '
print "Preannealing..."
# Initializing the MC object
for i in range(0,beta1):
    if 0 <= i <= 6:
        MC.updateMaxIt(50*M*D)
    else:
        MC.updateMaxIt(20*M*D)
    [accept,count,deltaArr,Xt1[i],Ft1[i]]=annealLatest(MC)
    # MC.updateOld(Xt1[i],Ft1[i]) # burn in trials only really need the latest state
    print 'beta =',i,'accept rate = ',float(accept)/count,'Rf = ',MC.Rf,'delta = ',MC.delta
    deltaArray1 = np.append(deltaArray1,deltaArr)
    MC.updateRf(alpha1*MC.Rf)

measError1   = Rm*np.copy(MC.measErrorArray)/M/NL
modelError1  = np.copy(MC.modelErrorArray)/M/D
modelAction1 = np.copy(MC.modelActionArray)/M/D
action1      = np.copy(MC.actionArray)/M/NL

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Main Annealing Phase
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Preannealing Constants
MC.updateRf(1.0)
MC.updateMaxIt(20*M*D)
MC.updateDelta(2.0)
MC.resetErrors()
MC.resetArrays()
MC.setPreFalse()
beta2  = 30
alpha2 = 1.4

# Tracking Error and Action
deltaArray2 = np.array([])
FArray2     = np.array([])
Xt2         = np.ones((beta2,M,D))
Ft2         = np.ones(beta2)

print "Main Annealing..."
# Initializing the MC object
for i in range(0,beta2):
    [accept,count,deltaArr,Xt2[i],Favg]=annealAverage(MC)
    MC.updateOld(Xt2[i],Favg)
    Ft2[i] = np.copy(Favg)
    print 'beta =',i,'accept rate = ',float(accept)/count,'Rf = ',MC.Rf,'delta = ',MC.delta
    deltaArray2 = np.append(deltaArray2,deltaArr)
    MC.updateRf(alpha2*MC.Rf)

measError2   = Rm*np.copy(MC.measErrorArray)/M/NL
modelError2  = np.copy(MC.modelErrorArray)/M/D
modelAction2 = np.copy(MC.modelActionArray)/M/D
action2      = np.copy(MC.actionArray)/M/NL

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Sampler Annealing Phase
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Preannealing Constants
MC.updateRf(25000.0)
MC.updateMaxIt(100*M*D)
MC.resetErrors()
MC.resetArrays()
MC.setPreFalse()
beta3  = 1

# Tracking Error and Action
deltaArray3 = np.array([])
FArray3     = np.array([])

print "Sample Annealing..."
[accept,count,deltaArr,Xt3,Ft3]=annealAverage(MC)
print 'beta =',1,'accept rate = ',float(accept)/count,'Rf = ',MC.Rf,'delta = ',MC.delta
deltaArray3 = np.append(deltaArray3,deltaArr)
FArray3     = np.append(FArray3,MC.Fold)

measError3   = Rm*np.copy(MC.measErrorArray)/M/NL
modelError3  = np.copy(MC.modelErrorArray)/M/D
modelAction3 = np.copy(MC.modelActionArray)/M/D
action3      = np.copy(MC.actionArray)/M/NL
print "Final Forcing",Ft3

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Saving Data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# np.savetxt('./data/L96_D'+str(D)+'_L'+str(len(Lidx))+'_Fconst_path.dat',Xt2[-1])
# np.save('./data/L96_D'+str(D)+'_L'+str(len(Lidx))+'_Fconst_path.npy',Xt2)
# np.save('./data/L96_D'+str(D)+'_L'+str(len(Lidx))+'_Fconst_sampled.npy',Xt3)
# np.save('./data/L96_D'+str(D)+'_L'+str(len(Lidx))+'_Fconst_misc.npy',[FArray2,measError2,modelError2,modelAction2,action2])

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
    plt.plot(Z[:,i],'k',label='True Solution')
    plt.plot(np.transpose(Xt1[-1,:,i]),label='Latest Proposed Solution')
plt.suptitle('After Preannealing Step')
plt.legend()

plt.figure(2)
for i in range(0,5):
    plt.subplot(5,1,i+1)
    plt.plot(Z[:,i],'k',label='True Solution')
    plt.plot(np.transpose(Xt2[-1,:,i]),label='Latest Proposed Solution')
plt.suptitle('After Main Annealing Step')
plt.legend()

plt.figure(3)
for i in range(0,5):
    plt.subplot(5,1,i+1)
    plt.plot(Z[:,i],'k',label='True Solution')
    plt.plot(np.transpose(Xt3[:,i]),label='Latest Proposed Solution')
plt.suptitle('After Sample Annealing Step')
plt.legend()

plt.figure(4)
plt.plot(FArray1,label='pre')
plt.plot(FArray2,label='main')
plt.plot(8.17*np.ones(len(FArray2)),label='real')
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('F')
plt.title('Forcing Parameter vs Total Iteration')
plt.legend()

plt.figure(5)
plt.semilogy(modelError1,label='pre')
plt.semilogy(modelError2,label='main')
plt.semilogy(realModelError*np.ones(max(modelError1.shape,modelError2.shape)),label='real')
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('Model Error')
plt.title('Model Error vs Total Iteration')
plt.legend()

plt.figure(6)
plt.semilogy(modelAction1,label='pre')
plt.semilogy(modelAction2,label='main')
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('Model Action')
plt.title('Model Action vs Total Iteration')
plt.legend()

plt.figure(7)
plt.semilogy(measError1,label='pre')
plt.semilogy(measError2,label='main')
plt.semilogy(Rm*realMeasError*np.ones(max(measError1.shape,measError2.shape)),label='real')
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('Meas Error')
plt.title('Meas Error vs Total Iteration')
plt.ylim([0.01,2])
plt.legend()

plt.figure(8)
plt.semilogy(action1,label='pre')
plt.semilogy(action2,label='main')
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('Action')
plt.title('Action vs Total Iteration')
plt.legend()

plt.figure(9)
plt.semilogy(deltaArray1,label='pre')
plt.semilogy(deltaArray2,label='main')
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('delta')
plt.title('delta vs Total Iteration')
plt.legend()

plt.show()
