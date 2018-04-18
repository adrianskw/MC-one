from libMCone import *
import matplotlib.pyplot as plt

""" EXAMPLE FUNCTION THAT CALLS CLASS FUNCTIONS """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is our annealing routine and the workhorse of the algorithm
# Right now, this is taken for F as a constant-in-time scalar which
# will be generalized for vector F later on
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def annealAll(MC):
    accept_count = 0
    count = 0
    while(1):
        MC.perturbState()
        MC.evalSoftAcceptance()
        if(MC.isAccepted):
            MC.keepNewState()
            MC.appendErrors()
            # For calculating acceptance rate
            accept_count += 1
        else:
            MC.keepOldState()
        # Number of times the while loop has run
        count += 1
        if (count >= MC.maxIt):
            break
    MC.calcActionArray()
    return accept_count,count


""" EXAMPLE ROUTINE USING annealAll(MC) """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Quick example that should get you familiar with the code
# Note that preannealing is set to True here
# By the way, 'overflow encountered in exp' error is expected
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
M = 200
D = 5
dt = 0.025
F = 8.17
Rm = 0.05
Rf = Rm
maxIt = 1000
delta = 3.0
Y = np.loadtxt('./data/L96_D_5_T_5_dt_0p025.noise')
Z = np.loadtxt('./data/L96_D_5_T_5_dt_0p025.dat')
Lidx = [0,1,2,3]
notLidx = [4]
measError = np.array([])
modelError = np.array([])
Action = np.array([])

# Replacing unmeasured states with noise
for i in notLidx:
    Y[:,i] = 5.0*(2.0*np.random.rand()-1.0)*np.ones(M)

# Initializing the MC object
MC = Annealer(Y,L96,dt,F,Lidx,Rm,Rf,maxIt,delta)
MC.setPreannealingTrue()

beta = 25
Xt = np.ones((beta+1,M,D))
Xt[0] = np.copy(Y)
for i in range(1,beta+1):
    [accept,count]=annealAll(MC)
    print 'beta =',i,'accept rate = ',float(accept)/count
    Xt[i]     = MC.Xnew
    measError = np.append(measError,MC.measErrorArray)
    modelError = np.append(modelError,MC.modelErrorArray)
    Action = np.append(Action,MC.ActionArray)
    MC.updateRf(1.9*MC.Rf)
    MC.updateDelta(MC.delta/1.15)
    MC.updateMaxIt(int(MC.maxIt*1.15))

np.savetxt(''.join(['./data/L96.path.dat']),Xt[-1],fmt='%9.6f',delimiter='\t\t')
np.savetxt(''.join(['./data/L96.misc.dat']),np.stack((measError,modelError,Action),axis=1)\
                    ,fmt='%9.6e',delimiter='\t\t')

# Plotting Stuff
plt.figure(0)
plt.semilogy(MC.modelErrorArray)
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('Model Error')

plt.figure(1)
for i in range(0,5):
    plt.subplot(5,1,i+1)
    plt.plot(np.transpose(Xt[:,:,i]))
    plt.plot(Z[:,i],'k')

plt.show()
