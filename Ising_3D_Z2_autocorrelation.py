from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from numba import jitclass
from numba import boolean
from numba.numpy_support import from_dtype

def Exponential(x, a, b, c):
    return a * np.exp(-x/b) + c

def autocorrelation_Ising(K_val, L_val, max_time):

    max_step = 100

    L_val = np.int64(L_val)
    K_val = np.float64(K_val)

    state = np.random.randint(0, 2, (L_val, L_val, L_val))
    state[state==0] = -1
    state = np.float64(state)
    #state = np.ones((L_val, L_val, L_val, 3))

    spec = [
        ('K', from_dtype(K_val.dtype)),          
        ('length', from_dtype(L_val.dtype)),
        ('state', from_dtype(state.dtype)[:,:,:])       
    ]
        
    @jitclass(spec)
    class Lattice(object): 
 
        def __init__(self, K, L, state):  

            self.K = K
            self.length = L
            self.state = state

        def pbc(self, x):
            # Periodic boundary conditions
 
            if x > (self.length)-1:
                return 0
            if x < 0:
                return (self.length)-1
            else:
                return x

        def GetFlipEnergy(self, X, Y, Z):

            return -2*self.state[X, Y, Z]*(
                    self.state[self.pbc(X-1), Y, Z] +
                    self.state[self.pbc(X+1), Y, Z] +
                    self.state[X, self.pbc(Y-1), Z] +
                    self.state[X, self.pbc(Y+1), Z] +
                    self.state[X, Y, self.pbc(Z-1)] +
                    self.state[X, Y, self.pbc(Z+1)]
                )

        def GetInternalEnergy(self):

            E=0

            for X in range(self.length):
                for Y in range(self.length):
                    for Z in range(self.length):
                        E += self.state[X, Y, Z]*(
                                self.state[self.pbc(X+1), Y, Z] +
                                self.state[X, self.pbc(Y+1), Z] +
                                self.state[X, Y, self.pbc(Z+1)]
                            )

            return (1 - (E/(self.length)**3)/3)

        def DoSweep(self):

            X_range = np.arange(self.length)
            Y_range = np.arange(self.length)
            Z_range = np.arange(self.length)
            np.random.shuffle(X_range)
            np.random.shuffle(Y_range)
            np.random.shuffle(Z_range)

            for X in X_range:
                for Y in Y_range:
                    for Z in Z_range:

                        # Calculate the energy if you flip the spin at that site
                        E = -1*self.GetFlipEnergy(X,Y,Z)

                        # Is the spin flipped or not?
                        if E <= 0:
                            self.state[X,Y,Z] *= -1
                        elif np.exp(-E*self.K) > np.random.rand():
                            self.state[X,Y,Z] *= -1

    
    print ("L =", L_val)
    print ("K =", K_val)

    rand_lattice = Lattice(K_val, L_val, state)

    print ("Get to equilibrium first")
    for i in range(2000):
        if i%500==0:
            print ("Sweep #:", i)
        rand_lattice.DoSweep()

    ac_M = []
    ac_avgM  = 0
    ac_avgM2 = 0

    print ("Now calculate autocorrelation")
    for i in range(max_time):
    
        if i%1000==0:
            print ("Sweep #:", i)

        M = rand_lattice.GetInternalEnergy()

        ac_M.append(M)
        ac_avgM += M
        ac_avgM2 += M**2
        rand_lattice.DoSweep()

    ac_avgM = ac_avgM/max_time
    ac_avgM2 = ac_avgM2/max_time

    ac_t = []
    ac_M0t = []
    #for t in range(max_time-1):
    for t in range(max_step):
        ac_t.append(t)
        if t%1000==0: print ("Step:", t)
        iter = 0
        M0t  = 0
        for j in range(len(ac_M)-t):
            iter += 1
            M0t += ac_M[j]*ac_M[j+t]

        ac_M0t.append(M0t/iter)

    ac_function = [(x - ac_avgM**2)/(ac_avgM2 - ac_avgM**2) for x in ac_M0t]

    fig = plt.figure()
    plt.xlabel("# of MCS")
    plt.ylabel("Autocorrelation of internal energy")
    plt.title("Autocorrelation of 3D Ising model, L={}, K={}".format(L_val, K_val))

    op, pcov = opt.curve_fit(Exponential, ac_t[:max_step], ac_function[:max_step])

    print ("OP:", op)
    print ("pcov:", pcov)

    plt.plot(np.array(ac_t[:max_step]), Exponential(np.array(ac_t[:max_step]), *op), label="y = {} * exp(-x/{}) + {}".format(round(op[0],4), round(op[1],4), round(op[2],4)), color="blue")
    plt.scatter(ac_t[:max_step], ac_function[:max_step], s=4, color="red")

    plt.legend()
    fig.savefig("3D_Ising_model_autocorrelation_L_{}_K_{}.png".format(L_val, K_val))
    plt.show()

if __name__ == "__main__":

    autocorrelation_Ising(0.22, 5, 50000)
    
