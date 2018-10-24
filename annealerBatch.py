from libMCone import *
import sys

""" EXAMPLE FUNCTION THAT CALLS CLASS FUNCTIONS """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is our annealing routine and the workhorse of the algorithm
# The point of this code is that the user is free to design their
# own annealing routine. Here is a preannealer as example:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is the annealer function used for both the pre- and main- annealing
def annealSome(MC):
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
            block_count        = 0 # Reseting the block-related tallies
            accept_block_count = 0 # Reseting the block-related tallies
        if (count>= MC.maxIt):
            break
    return accept_count,count,deltaArray

""" EXAMPLE ROUTINE USING annealAll(MC) """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Quick example that should get you familiar with the code.
# By the way, 'overflow encountered in exp' error is expected.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Model Constants
M = 200
D = 5
dt = 0.025
F = 8.5 #not a vector, so 'const'
Lidx = [0,2,4]
notLidx  = np.setxor1d(np.arange(D),Lidx)
NL = len(Lidx)

# Setting Random Seed when adding noise (for repeatability)
np.random.seed(123)

# Twin Experiment Data
Z = np.loadtxt('./data/L96_D5_Fconst_truepath.dat')[0:M,:]
Y = Z + (2.0*np.random.rand(M,D)-1.0)

# Calculating Error
# NOTE: do this before over-writing unmeasured states
[realMeasError, realModelError] = calcRealError(Z,Y,M,D,dt,Lidx)

# Replacing unmeasured states with noise
# These two lines below are for batch runs on a cluster
ID = int(sys.argv[1])
np.random.seed(ID)
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
measError1   = np.array([])
modelError1  = np.array([])
modelAction1 = np.array([])
action1      = np.array([])
deltaArray1  = np.array([])
FArray1      = np.array([])
Xt1          = np.ones((beta1,M,D))

# Initializing the MC object
MC = Annealer(Y,L96,dt,F,Lidx,Rm,Rf,maxIt,delta,True,True,False)
Xinit = np.copy(MC.Xold)


# Initializing the MC object
for i in range(0,beta1):
    if 0 <= i <= 6:
        MC.updateMaxIt(50*M*D)
    else:
        MC.updateMaxIt(20*M*D)
    annealSome(MC)
    # Xt1[i]       = MC.Xold
    # FArray1      = np.append(FArray1,MC.Fold)
    # measError1   = np.append(measError1,Rm*MC.oldMeasError/M/NL)
    # modelError1  = np.append(modelError1,MC.oldModelError/M/D)
    # modelAction1 = np.append(modelAction1,MC.oldModelAction/M/D)
    # action1      = np.append(action1,MC.oldAction/M/NL)
    MC.updateRf(alpha1*MC.Rf)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Main Annealing Phase
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Preannealing Constants
MC.updateRf(1.0)
MC.updateMaxIt(12*M*D)
MC.updateDelta(2.0)
MC.resetErrors()
MC.resetArrays()
MC.setPreFalse()
beta2  = 30
alpha2 = 1.3

# Tracking Error and Action
measError2   = np.array([])
modelError2  = np.array([])
modelAction2 = np.array([])
action2      = np.array([])
deltaArray2  = np.array([])
FArray2      = np.array([])
Xt2          = np.ones((beta2,M,D))

# Initializing the MC object
for i in range(0,beta2):
    annealSome(MC)
    Xt2[i]       = MC.Xold
    FArray2      = np.append(FArray2,MC.Fold)
    measError2   = np.append(measError2,Rm*MC.oldMeasError/M/NL)
    modelError2  = np.append(modelError2,MC.oldModelError/M/D)
    modelAction2 = np.append(modelAction2,MC.oldModelAction/M/D)
    action2      = np.append(action2,MC.oldAction/M/NL)
    MC.updateRf(alpha2*MC.Rf)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Saving Data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Paths
np.savetxt('./data/L96_D'+str(D)+'_L'+str(len(Lidx))+'_Fconst_path_'+str(ID)+'.dat',Xt2[-1])
np.save('./data/L96_D'+str(D)+'_L'+str(len(Lidx))+'_Fconst_path_'+str(ID)+'.npy',Xt2)

# Misc
np.savetxt('./data/L96_D'+str(D)+'_L'+str(len(Lidx))+'_Fconst_misc_'+str(ID)+'.dat',[FArray2,measError2,modelError2,modelAction2,action2])
