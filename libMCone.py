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
    def __init__(self,Y,model,dt,F,Lidx,Rm,Rf,maxIt,delta,flagPre=True,flagF=True,flagAppend=True):

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # EXTERNAL VARIABLES
        # These are explicitly passed into the object container
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
      # self.F not used, saved as Fnew and Fold (see below)
        self.Y          = np.copy(Y)
        self.model      = model
        self.dt         = dt
        self.Lidx       = Lidx
        self.Rm         = Rm
        self.maxIt      = maxIt
        self.flagPre    = flagPre
        self.flagF      = flagF
        self.flagAppend = flagAppend
        self.Rf         = np.copy(Rf)
        self.delta      = np.copy(delta)

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # DERIVED VARIABLES
        # Implicitly used in the object but are derived from external variables
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        if isinstance(F,int) or isinstance(F,float):
            self.Fold = np.array([F])
            self.Fnew = np.array([F])
        else:
            self.Fold = np.copy(F) # ALWAYS COPY
            self.Fnew = np.copy(F) # ALWAYS COPY
        self.NF       = self.Fnew.size
        if self.flagF == False:
            self.NF   = 0
        self.M        = Y.shape[0]
        self.D        = Y.shape[1]
        self.notLidx  = np.setxor1d(np.arange(self.D),Lidx)
        # self.Xold     = np.copy(Y)
        # self.Xnew     = np.copy(Y)
        self.initializeData() # this sets Xnew and Xold
        self.resetContainer()

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # INTERNAL VARIABLES
        # These are calculated implicitly, but ONLY during an annealing step
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        self.Didx             = 9999 # initialized as garbage values
        self.Midx             = 9999 # initialized as garbage values
        self.deltaMeasError   = 0
        self.deltaModelError  = 0
        self.isAccepted       = False
        self.calcOldMeasError()
        self.calcOldModelError()
        self.oldAction        = self.Rm*self.oldMeasError + self.Rf*self.oldModelError
        self.oldModelAction   = self.Rf*self.oldModelError
        self.measErrorArray   = np.array([self.oldMeasError])
        self.modelErrorArray  = np.array([self.oldModelError])
        self.actionArray      = np.array([self.oldAction])
        self.modelActionArray = np.array([self.oldModelAction])
        self.set1             = np.append(np.arange(self.M),self.M+self.notLidx)
        self.set2             = np.arange(self.M+self.NF)

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
        # if Midx   == 0: # if we picked the first element
        #     error =  abs(self.Xold[       1,:]-self.RK2(self.Xold[       0,:],self.Fold))**2
        # elif Midx == self.M-1: # if we picked the last element
        #     error =  abs(self.Xold[self.M-1,:]-self.RK2(self.Xold[self.M-2,:],self.Fold))**2
        if Midx   == 0: # if we picked the first element
            error =  abs(self.Xold[Midx+1,:]-self.RK2(self.Xold[Midx  ,:],self.Fold))**2\
                    # +abs(self.Xold[Midx+2,:]-self.RK2(self.RK2(self.Xold[Midx  ,:],self.Fold),self.Fold))**2
        elif Midx == self.M-1: # if we picked the last element
            error =  abs(self.Xold[Midx  ,:]-self.RK2(self.Xold[Midx-1,:],self.Fold))**2\
                    # +abs(self.Xold[Midx  ,:]-self.RK2(self.RK2(self.Xold[Midx-2,:],self.Fold),self.Fold))**2
        else: # else we have to vary in both directions
            error =  abs(self.Xold[Midx+1,:]-self.RK2(self.Xold[Midx  ,:],self.Fold))**2 \
                    +abs(self.Xold[Midx  ,:]-self.RK2(self.Xold[Midx-1,:],self.Fold))**2
        return sum(error)

    def newPointModelError(self,Midx):
        # if Midx   == 0: # if we picked the first element
        #     error =  abs(self.Xnew[       1,:]-self.RK2(self.Xnew[       0,:],self.Fnew))**2
        # elif Midx == self.M-1: # if we picked the last element
        #     error =  abs(self.Xnew[self.M-1,:]-self.RK2(self.Xnew[self.M-2,:],self.Fnew))**2
        if Midx   == 0: # if we picked the first element
            error =  abs(self.Xnew[Midx+1,:]-self.RK2(self.Xnew[Midx  ,:],self.Fnew))**2\
                    # +abs(self.Xnew[Midx+2,:]-self.RK2(self.RK2(self.Xnew[Midx  ,:],self.Fnew),self.Fnew))**2
        elif Midx == self.M-1: # if we picked the last element
            error =  abs(self.Xnew[Midx  ,:]-self.RK2(self.Xnew[Midx-1,:],self.Fnew))**2\
                    # +abs(self.Xnew[Midx  ,:]-self.RK2(self.RK2(self.Xnew[Midx-2,:],self.Fnew),self.Fnew))**2
        else: # else we have to vary in both directions
            error =  abs(self.Xnew[Midx+1,:]-self.RK2(self.Xnew[Midx  ,:],self.Fnew))**2 \
                    +abs(self.Xnew[Midx  ,:]-self.RK2(self.Xnew[Midx-1,:],self.Fnew))**2
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
                error += abs(self.Xold[Midx,Didx]-self.Y[Midx,Didx])**2
        self.oldMeasError = error

    def calcNewMeasError(self):
        error = 0.0
        for Midx in range(0,self.M):
            for Didx in self.Lidx:
                error += abs(self.Xnew[Midx,Didx]-self.Y[Midx,Didx])**2
        self.newMeasError = error

    def calcOldModelError(self):
        error = 0.0
        for Midx in range(0,self.M-1):
            error += abs(self.Xold[Midx+1,:]-self.RK2(self.Xold[Midx,:],self.Fnew))**2
        self.oldModelError = sum(error)

    def calcNewModelError(self):
        error = 0.0
        for Midx in range(0,self.M-1):
            error += abs(self.Xnew[Midx+1,:]-self.RK2(self.Xnew[Midx,:],self.Fnew))**2
        self.newModelError = sum(error)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # POINT-WISE DELTA ERRORS
    # Calculates change in error before and after pertubations/trials
    # Only use after a perturbation is proposed and saved
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def calcDeltaMeasError(self):
        if self.Midx < self.M:
            self.deltaMeasError = self.newPointMeasError(self.Midx,self.Didx)\
                                - self.oldPointMeasError(self.Midx,self.Didx)
        elif self.Midx >= self.M:
            self.deltaMeasError = 0.0

    def calcDeltaModelError(self):
        if self.Midx < self.M:
            self.deltaModelError = self.newPointModelError(self.Midx)\
                                 - self.oldPointModelError(self.Midx)
        elif self.Midx >= self.M:
            self.calcNewModelError()
            self.deltaModelError = (self.newModelError - self.oldModelError)

    def calcDeltaAction(self):
        self.calcDeltaMeasError()
        self.calcDeltaModelError()
        self.deltaAction = self.Rm*self.deltaMeasError + self.Rf*self.deltaModelError

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # CORE MONTE CARLO FUNCTIONS
    # All RNG based functions and acceptance/rejection functions are here
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def rollDidx(self):
        if self.flagPre == True:
            self.Didx = np.random.choice(self.notLidx)
        else:
            self.Didx = np.random.choice(np.arange(self.D))

    def rollMidx(self):
        if self.flagPre == True and self.NF > 1:
            self.Midx = np.random.choice(self.set1) # refer to 'Internal Variables'
        else:
            self.Midx = np.random.choice(self.set2) # refer to 'Internal Variables'

    def perturbState(self):
        # Roll indices to perturb
        self.rollMidx()
        self.rollDidx()
        # Perturb old to get new
        if self.Midx < self.M:
            self.Xnew[self.Midx,self.Didx] = self.Xold[self.Midx,self.Didx]\
                                           + self.delta*(2.0*np.random.rand()-1.0)
        elif self.Midx >= self.M:
            self.Fnew[self.Midx-self.M] = self.Fold[self.Midx-self.M] + self.delta*(2.0*np.random.rand()-1.0)

    def evalSoftAcceptance(self):
        self.calcDeltaAction()
        self.isAccepted = (np.exp(-self.deltaAction)>np.random.rand())

    def evalHardAcceptance(self):
        self.calcDeltaAction()
        self.isAccepted = (self.deltaAction<0)

    def keepOldState(self):
        if self.Midx < self.M:
            self.Xnew[self.Midx,self.Didx] = self.Xold[self.Midx,self.Didx]
        elif self.Midx >= self.M:
            self.Fnew = np.copy(self.Fold)

    def keepNewState(self):
        if self.Midx < self.M:
            self.Xold[self.Midx,self.Didx] = self.Xnew[self.Midx,self.Didx]
        elif self.Midx >= self.M:
            self.Fold = np.copy(self.Fnew)
        self.appendErrors()

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

    def updateXold(self,Xold):
        self.Xold = np.copy(Xold)

    def resetContainer(self):
        self.Xcontainer = np.zeros(self.Xnew.shape)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Misc Functions
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # For switching on/off the preconditioning routine
    def setPreTrue(self):
        self.flagPre = True

    def setPreFalse(self):
        self.flagPre = False

    # Reset Errors
    def resetErrors(self):
        self.calcOldMeasError()
        self.calcOldModelError()
        self.oldAction = self.Rm*self.oldMeasError + self.Rf*self.oldModelError

    # Reset Arrays
    def resetArrays(self):
        self.measErrorArray   = np.array([self.oldMeasError])
        self.modelErrorArray  = np.array([self.oldModelError])
        self.actionArray      = np.array([self.oldAction])
        self.modelActionArray = np.array([self.oldModelAction])

    # Appends the errors into an array that stores ALL errors from previous beta
    # Also calculates the latest (rolling) value of the action/errors
    def appendErrors(self):
        # Latest Values
        self.oldMeasError    = self.oldMeasError + self.deltaMeasError
        self.oldModelError   = self.oldModelError + self.deltaModelError
        self.oldAction       = self.Rm*self.oldMeasError + self.Rf*self.oldModelError
        self.oldModelAction  = self.Rf*self.oldModelError
        # Cumulative Arrays
        if self.flagAppend == True:
            self.measErrorArray  = np.append(self.measErrorArray,self.oldMeasError)
            self.modelErrorArray = np.append(self.modelErrorArray,self.oldModelError)
            self.actionArray     = np.append(self.actionArray,self.oldAction)
            self.modelActionArray= np.append(self.modelActionArray,self.oldModelAction)

    def replaceXold(self,Xlow):
        self.Xold = np.copy(Xlow)
        self.Xnew = np.copy(Xlow)
        self.resetErrors()

    # Henry's initialization trick, by default set to True
    # Calculates the trajectories forward with 'bad' initial conditions
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
# Performs entire integration, gives us the full data set x
# Returns state as array
# Uses RK4 integration
# Ideally data should be generated for very small timesteps
# for high-quality data, then down-sampled and fed into the annealer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def generateData(x,F,dt):
    xt = np.array(x)
    for k in range(0,M-1):
        x  = RK4(x,F,dt)
        xt = np.vstack((xt,x))
    return xt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Discards the first few hundred data points
# Makes the system forget transients and sample around the attractor
# Same as generateData but nothing is saved
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def burnData(x,F,dt):
    xt = np.array(x)
    for k in range(0,1000):
        x  = RK4(x,F,dt)
    return x

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Integrators, From Class Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# RK2 integration used in point model error
def RK2(x,F,dt):
    a1 = L96(x        ,F)
    a2 = L96(x+a1*dt/2,F)
    return x+a2*dt

# RK4 integration only used for genrating data
def RK4(x,F,dt):
    a1 = L96(x             ,F)
    a2 = L96(x+a1*dt/2,F)
    a3 = L96(x+a2*dt/2,F)
    a4 = L96(x+a3*dt  ,F)
    return x+(a1/6+a2/3+a3/3+a4/6)*dt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Expected (Normalized) Error Calculation, From Class Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def calcRealError(Z,Y,M,D,dt,Lidx):
    notLidx  = np.setxor1d(np.arange(D),Lidx)
    NL       = len(Lidx)

    modelError = 0
    for i in range(0,M):
        modelError += pointModelErrorRK2(Z,Y,M,D,i,8.17,dt)

    # Replacing unmeasured states with noise
    for i in notLidx:
        Y[:,i] = 5.0*(2.0*np.random.rand()-1.0)*np.ones(M)

    measError = 0
    for i in range(0,M):
        for j in Lidx:
            measError += pointMeasError(Z,Y,M,i,j,Lidx)

    return measError,modelError

def pointModelErrorRK2(X,Y,M,D,Midx,F,dt):
    if Midx == 0: # if we picked the first element
        error =  abs(X[       1,:]-RK2(X[       0,:],F,dt))**2
    elif Midx == M-1: # if we picked the last element
        error =  abs(X[M-1,:]-RK2(X[M-2,:],F,dt))**2
    else: # else we have to vary in both directions
        error =  abs(X[Midx+1,:]-RK2(X[Midx-0,:],F,dt))**2 \
                +abs(X[Midx+0,:]-RK2(X[Midx-1,:],F,dt))**2
    return sum(error)/M/D

def pointModelErrorRK4(X,Y,M,D,Midx,F,dt):
    if Midx == 0: # if we picked the first element
        error =  abs(X[       1,:]-RK4(X[       0,:],F,dt))**2
    elif Midx == M-1: # if we picked the last element
        error =  abs(X[M-1,:]-RK4(X[M-2,:],F,dt))**2
    else: # else we have to vary in both directions
        error =  abs(X[Midx+1,:]-RK4(X[Midx-0,:],F,dt))**2 \
                +abs(X[Midx+0,:]-RK4(X[Midx-1,:],F,dt))**2
    return sum(error)/M/D

def pointMeasError(X,Y,M,Midx,Didx,Lidx):
    if Didx in Lidx: # fail-safe check
        return abs(X[Midx,Didx]-Y[Midx,Didx])**2/M/len(Lidx)
    else:
        return 0
