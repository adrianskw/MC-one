
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
