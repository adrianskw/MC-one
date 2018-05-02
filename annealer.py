from libMCone import *
import matplotlib.pyplot as plt

""" EXAMPLE FUNCTION THAT CALLS CLASS FUNCTIONS """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is our annealing routine and the workhorse of the algorithm
# The point of this code is that the user is free to design their
# own annealing routine. Here is a preannealer as example:
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
# Quick example that should get you familiar with the code.
# Note that preannealing is set to True here, therefore we are
# only doing preannealing and not the main annealer.
# By the way, 'overflow encountered in exp' error is expected.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
M = 200
D = 5
dt = 0.025
F = 8.25
Rm = 5.0
Rf = Rm
maxIt = 2*M*D
delta = 1.0
Y = np.loadtxt('./data/L96_D_5_T_5_dt_0p025.noise')
Z = np.loadtxt('./data/L96_D_5_T_5_dt_0p025.dat')
Lidx = [1,3]
notLidx = [0,2,4]
measError = np.array([])
modelError = np.array([])
Action = np.array([])
np.random.seed(1234)

# Replacing unmeasured states with noise
for i in notLidx:
    Y[:,i] = 5.0*(2.0*np.random.rand()-1.0)*np.ones(M)

# Initializing the MC object
MC = Annealer(Y,L96,dt,F,Lidx,Rm,Rf,maxIt,delta,True)
Xinit = np.copy(MC.Xold)

beta = 15
Xt = np.ones((beta,M,D))
for i in range(0,beta):
    [accept,count]=annealAll(MC)
    print 'beta =',i,'accept rate = ',float(accept)/count,'Rf = ',MC.Rf,'delta = ',MC.delta
    Xt[i]     = MC.Xnew
    measError = np.append(measError,MC.measErrorArray)
    modelError = np.append(modelError,MC.modelErrorArray)
    Action = np.append(Action,MC.ActionArray)
    MC.updateRf(1.9*MC.Rf)
    MC.updateDelta(MC.delta/1.2)
    MC.updateMaxIt(int(MC.maxIt*1.2))

np.savetxt(''.join(['./data/L96.path.dat']),Xt[-1],fmt='%9.6f',delimiter='\t\t')
np.savetxt(''.join(['./data/L96.misc.dat']),np.stack((measError,modelError,Action),axis=1)\
                    ,fmt='%9.6e',delimiter='\t\t')

# Plotting Stuff
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
    plt.plot(np.transpose(Xt[:-2,:,i]))
    plt.plot(np.transpose(Xt[-1,:,i]),label='Latest Proposed Solution')
plt.suptitle('After Preannealing Step')
plt.legend()

plt.figure(2)
plt.semilogy(MC.modelErrorArray)
plt.xlabel('Total Iterations (roughly proportional to beta)')
plt.ylabel('Model Error')
plt.title('Model Error vs Total Iteration')

print "Final Forcing",MC.Fold
plt.show()
