import os, sys
import numpy as np
import time

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# The goal of the Annealer class is to provide a 'container' that
# houses all the necessary functions and variables securely, without
# having to explicitly pass the arguments to functions.
# When using class functions, the appropriate variables are called
# and saved automatically within the object. This results in neat
# and readable code. You should be able to implement any Monte-Carlo
# annealing schedule/strategy with this class. [See "annealer.py"]
# NOTE: Do not attempt to use this code if any of the EXTERNAL
# VARIABLES or DERIVED VARIABLES are not understood. Please refer to
# some material instead, say Jingxin Ye and John C. Quinn's theses
# on escholarship.org.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class Annealer:
    """ INITIALIZING ANNEALER OBJECT """
    def __init__(self,Y,model,dt,F,Lidx,Rm,Rf,maxIt,delta,pre):

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # EXTERNAL VARIABLES
        # These are explicitly passed into the object container
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        self.Y       = Y
        self.model   = model
        self.dt      = dt
      # self.F not used, saved as Fnew and Fold (see below)
        self.Lidx    = Lidx
        self.Rm      = Rm
        self.Rf      = Rf
        self.maxIt   = maxIt
        self.delta   = delta
        self.pre     = pre

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # DERIVED VARIABLES
        # Implicitly used in the object but are derived from external variables
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        self.M       = Y.shape[0]
        self.D       = Y.shape[1]
        self.Fold    = np.copy(F) # ALWAYS COPY
        self.Fnew    = np.copy(F) # ALWAYS COPY
        self.notLidx = np.setxor1d(np.arange(self.D),Lidx)
        if   self.pre == True:
            self.initializeData() # this sets Xnew and Xold
        elif self.pre == False:
            self.Xold = np.copy(Y)
            self.Xnew = np.copy(Y)
        else:
            print "self.pre not initialized properly"

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # INTERNAL VARIABLES
        # These are calculated implicitly, but only during an annealing step
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        self.Didx    = 9999 # initialized as garbage values
        self.Midx    = 9999 # initialized as garbage values
        self.calcOldModelError()
        self.calcOldMeasError()
        self.deltaMeasError  = 0
        self.deltaModelError = 0
        self.isAccepted = False
        self.measErrorArray = np.array([self.oldMeasError])
        self.modelErrorArray = np.array([self.oldModelError])

    """ CORE FUNCTIONS """
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # POINT-WISE ERRORS
    # Calculates point-wise error which are used in the total error
    # 'new' refers to the Xnew trial state and vice versa for 'old'
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def oldPointMeasError(self,Midx,Didx):
        if Didx in self.Lidx: # fail-safe check
            return abs(self.Xold[Midx,Didx]-self.Y[Midx,Didx])**2
        else:
            return 0

    def newPointMeasError(self,Midx,Didx):
        if Didx in self.Lidx: # fail-safe check
            return abs(self.Xnew[Midx,Didx]-self.Y[Midx,Didx])**2
        else:
            return 0

    # NOTE: These two are summed over all states, but with more information
    #       and some tweaking we can sum over the 'connected' variables only
    def oldPointModelError(self,Midx):
        if Midx == 0: # if we picked the first element
            error =  abs(self.Xold[       1,:]-self.RK2(self.Xold[       0,:],self.Fold))**2
        elif Midx == self.M-1: # if we picked the last element
            error =  abs(self.Xold[self.M-1,:]-self.RK2(self.Xold[self.M-2,:],self.Fold))**2
        else: # else we have to vary in both directions
            error =  abs(self.Xold[Midx+1,:]-self.RK2(self.Xold[Midx-0,:],self.Fold))**2 \
                    +abs(self.Xold[Midx+0,:]-self.RK2(self.Xold[Midx-1,:],self.Fold))**2
        return sum(error)

    def newPointModelError(self,Midx):
        if Midx == 0: # if we picked the first element
            error =  abs(self.Xnew[       1,:]-self.RK2(self.Xnew[       0,:],self.Fnew))**2
        elif Midx == self.M-1: # if we picked the last element
            error =  abs(self.Xnew[self.M-1,:]-self.RK2(self.Xnew[self.M-2,:],self.Fnew))**2
        else: # else we have to vary in both directions
            error =  abs(self.Xnew[Midx+1,:]-self.RK2(self.Xnew[Midx-0,:],self.Fnew))**2 \
                    +abs(self.Xnew[Midx+0,:]-self.RK2(self.Xnew[Midx-1,:],self.Fnew))**2
        return sum(error)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # INTEGRATORS
    # These are used for time marching models forward in time
    # RK2 is used in main annealer (2x faster), and RK4 for initializer
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # RK2 integration used in point model error
    def RK2(self,x,F):
        a1 = self.model(x             ,F)
        a2 = self.model(x+a1*self.dt/2,F)
        return x+a2*self.dt

    # RK4 integration only used for initializing routine
    def RK4(self,x,F):
        a1 = self.model(x             ,F)
        a2 = self.model(x+a1*self.dt/2,F)
        a3 = self.model(x+a2*self.dt/2,F)
        a4 = self.model(x+a3*self.dt  ,F)
        return x+(a1/6+a2/3+a3/3+a4/6)*self.dt

    """ CONVENIENT WRAPPINGS OF CORE FUNCTIONS """
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # TOTAL ERRORS
    # Calculates the error of the entire time-series by summing over
    # the point-wise version of these functions
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def calcOldMeasError(self):
        error = 0.0
        for Midx in range(0,self.M):
            for Didx in self.Lidx:
                error+=self.oldPointMeasError(Midx,Didx)
        self.oldMeasError = error

    def calcNewMeasError(self):
        error = 0.0
        for Midx in range(0,self.M):
            for Didx in self.Lidx:
                error+=self.newPointMeasError(Midx,Didx)
        self.newMeasError = error

    def calcOldModelError(self):
        error = 0.0
        for Midx in range(0,self.M):
            error += self.oldPointModelError(Midx)
        self.oldModelError = error

    def calcNewModelError(self):
        error = 0.0
        for Midx in range(0,self.M):
            error += self.newPointModelError(Midx)
        self.newModelError = error

    def calcActionArray(self):
        self.ActionArray = self.Rm*np.array(self.measErrorArray)\
                         + self.Rf*np.array(self.modelErrorArray)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # POINT-WISE DELTA ERRORS
    # Calculates change in error before and after pertubations/trials
    # Only use after a perturbation is proposed and saved
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def calcDeltaMeasError(self):
        self.deltaMeasError = self.newPointMeasError(self.Midx,self.Didx)\
                            - self.oldPointMeasError(self.Midx,self.Didx)

    def calcDeltaModelError(self):
        self.deltaModelError = self.newPointModelError(self.Midx)\
                             - self.oldPointModelError(self.Midx)

    def calcDeltaAction(self):
        self.calcDeltaMeasError()
        self.calcDeltaModelError()
        self.deltaAction = self.Rm*self.deltaMeasError + self.Rf*self.deltaModelError

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # CORE MONTE CARLO FUNCTIONS
    # All RNG based functions and acceptance/rejection functions are here
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def rollDidx(self):
        if self.pre == True:
            self.Didx = np.random.choice(self.notLidx)
        else:
            self.Didx = np.random.choice(np.arange(self.D))

    def rollMidx(self):
        self.Midx = np.random.choice(np.arange(self.M))

    def perturbState(self):
        # Roll indices to perturb
        self.rollMidx()
        self.rollDidx()
        # Perturb old to get new
        self.Xnew[self.Midx,self.Didx] = self.Xold[self.Midx,self.Didx]\
                                       + self.delta*(2.0*np.random.rand()-1.0)

    def evalSoftAcceptance(self):
        self.calcDeltaAction()
        self.isAccepted = (np.exp(-self.deltaAction)>np.random.rand())

    def evalHardAcceptance(self):
        self.calcDeltaAction()
        self.isAccepted = (self.deltaAction<0)

    def keepOldState(self):
        self.Xnew[self.Midx,self.Didx] = self.Xold[self.Midx,self.Didx]

    def keepNewState(self):
        self.Xold[self.Midx,self.Didx] = self.Xnew[self.Midx,self.Didx]

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # UPDATING FUNCTIONS
    # Used to update annealing parameters as desired
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def updateRf(self,Rf):
        self.Rf = Rf

    def updateDelta(self,delta):
        self.delta = delta

    def updateMaxIt(self,maxIt):
        self.maxIt = maxIt

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Misc Functions
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Appends the errors into an array that stores ALL errors from previous beta
    def appendErrors(self):
        self.measErrorArray = np.append(self.measErrorArray,\
                                        self.measErrorArray[-1]+self.deltaMeasError)
        self.modelErrorArray = np.append(self.modelErrorArray,\
                                         self.modelErrorArray[-1]+self.deltaModelError)

    # This calculates action array from measurement and model errors
    def calcActionArray(self):
        self.ActionArray = self.Rm*np.array(self.measErrorArray)\
                         + self.Rf*np.array(self.modelErrorArray)

    # Henry's initialization trick, by default set to True
    # Calculates the trajectories forward with bad initial conditions
    # but this trajectory is still 'close' to the actual solution
    def initializeData(self):
        X = np.copy(self.Y)
        for k in range(0,self.M-1):
            X[k+1,self.notLidx]  = self.RK4(X[k],self.Fold)[self.notLidx]
        self.Y    = np.copy(X)
        self.Xold = np.copy(X)
        self.Xnew = np.copy(X)

    """ END ANNEALER CLASS """

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
    global D
    global M
    global dt
    global L_frac
    D,M,dt,L_frac  = np.asarray(sys.argv[1:])
    D = D.astype(int)
    M = M.astype(int)
    dt = dt.astype(float)
    L_frac = L_frac.astype(float)
    print "Running annealling for D = "+str(D)+ ", M = "+str(M)+ \
            ", dt = "+str(dt)+", L_frac = "+str(L_frac)

def initAnnealingParser():
    if (len(sys.argv)!=2) or isinstance(sys.argv[1],int):
        print("Please enter proper arguments:")
        print("ID has to be an integer.")
        sys.exit(0)
    global ID
    ID = np.asarray(sys.argv[1])
    ID = ID.astype(int)

""" MODEL AND INTEGRATION FUNCTIONS """
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Definition of the Lorenz96 model
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This is our model, L96 does not explicitly depend on time
def L96(x,F):
    # compute state derivatives
    D = x.size
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
# Here we are using the simple RK4, set up for L96 now
# RK4 is used in data generation and RK2 is used (faster) in annealing
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def stepForward_RK4(x,F):
    a1 = L96(x        ,F)
    a2 = L96(x+a1*dt/2,F)
    a3 = L96(x+a2*dt/2,F)
    a4 = L96(x+a3*dt  ,F)
    return x+(a1/6+a2/3+a3/3+a4/6)*dt

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
