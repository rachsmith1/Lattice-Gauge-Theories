from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from numba import jitclass
from numba import boolean
from numba.numpy_support import from_dtype

def Exponential(x, a, b, c):
    return a * np.exp(-x/b) + c

def autocorrelation_Matter(K_val, B_val, L_val, max_time):

    max_step = 100

    L_val = np.int64(L_val)
    K_val = np.float64(K_val)
    B_val = np.float64(B_val)

    state = np.random.randint(0, 2, (L_val, L_val, L_val, 4))
    state[state==0] = -1
    state = np.float64(state)

    spec = [
        ('K', from_dtype(K_val.dtype)), 
        ('B', from_dtype(B_val.dtype)),
        ('length', from_dtype(L_val.dtype)),
        ('state', from_dtype(state.dtype)[:,:,:,:]) 
    ]

    @jitclass(spec)
    class Lattice:

        def __init__(self, K, B, L, state):

            self.length = L
            self.K = K
            self.B = B
            self.state = state

        def pbc(self, x):
            # Periodic boundary conditions

            if x > (self.length)-1:
                return 0
            if x < 0:
                return (self.length)-1
            else:
                return x

        def GetFlipEnergy(self, X, Y, Z, b):

            e = 0

            # Spin is on the x-axis bond
            if b==0:
                e += -2*self.state[X,Y,Z,0]*(
                            self.K*self.state[X,Y,Z,2] * self.state[X,Y,self.pbc(Z+1),0] * self.state[self.pbc(X+1),Y,Z,2] +
                            self.K*self.state[X,Y,Z,1] * self.state[X,self.pbc(Y+1),Z,0] * self.state[self.pbc(X+1),Y,Z,1] +
                            self.K*self.state[X,Y,self.pbc(Z-1),2] * self.state[X,Y,self.pbc(Z-1),0] * self.state[self.pbc(X+1),Y,self.pbc(Z-1),2] +
                            self.K*self.state[X,self.pbc(Y-1),Z,1] * self.state[X,self.pbc(Y-1),Z,0] * self.state[self.pbc(X+1),self.pbc(Y-1),Z,1]
                            +
                            self.B*self.state[X,Y,Z,3]*self.state[self.pbc(X+1),Y,Z,3] 
                        )

            # Spin is on the y-axis bond
            if b==1:
                e += -2*self.state[X,Y,Z,1]*(
                            self.K*self.state[X,Y,Z,2] * self.state[X,Y,self.pbc(Z+1),1] * self.state[X,self.pbc(Y+1),Z,2] +
                            self.K*self.state[X,Y,Z,0] * self.state[self.pbc(X+1),Y,Z,1] * self.state[X,self.pbc(Y+1),Z,0] +
                            self.K*self.state[X,Y,self.pbc(Z-1),2] * self.state[X,Y,self.pbc(Z-1),1] * self.state[X,self.pbc(Y+1),self.pbc(Z-1),2] +
                            self.K*self.state[self.pbc(X-1),Y,Z,0] * self.state[self.pbc(X-1),Y,Z,1] * self.state[self.pbc(X-1),self.pbc(Y+1),Z,0]
                            +
                            self.B*self.state[X,Y,Z,3]*self.state[X,self.pbc(Y+1),Z,3]
                        )

            # Spin is on the z-axis bond
            if b==2:
                e += -2*self.state[X,Y,Z,2]*(
                            self.K*self.state[X,Y,Z,1] * self.state[X,self.pbc(Y+1),Z,2] * self.state[X,Y,self.pbc(Z+1),1] +
                            self.K*self.state[X,Y,Z,0] * self.state[self.pbc(X+1),Y,Z,2] * self.state[X,Y,self.pbc(Z+1),0] +
                            self.K*self.state[X,self.pbc(Y-1),Z,1] * self.state[X,self.pbc(Y-1),Z,2] * self.state[X,self.pbc(Y-1),self.pbc(Z+1),1] +
                            self.K*self.state[self.pbc(X-1),Y,Z,0] * self.state[self.pbc(X-1),Y,Z,2] * self.state[self.pbc(X-1),Y,self.pbc(Z+1),0]
                            +
                            self.B*self.state[X,Y,Z,3]*self.state[X,Y,self.pbc(Z+1),3]
                        )

            # Spin is on the vertex 
            if b==3:
                e += -2*self.state[X,Y,Z,3]*(
                            self.B*self.state[X,Y,Z,0]*self.state[self.pbc(X+1),Y,Z,3] +
                            self.B*self.state[self.pbc(X-1),Y,Z,3]*self.state[self.pbc(X-1),Y,Z,0] +
                            self.B*self.state[X,Y,Z,1]*self.state[X,self.pbc(Y+1),Z,3] +
                            self.B*self.state[X,self.pbc(Y-1),Z,3]*self.state[X,self.pbc(Y-1),Z,1] +
                            self.B*self.state[X,Y,Z,2]*self.state[X,Y,self.pbc(Z+1),3] +
                            self.B*self.state[X,Y,self.pbc(Z-1),3]*self.state[X,Y,self.pbc(Z-1),2]
                        )

            return e

        def GetInternalEnergy(self):
            E=0

            for X in np.arange(self.length):
                for Y in np.arange(self.length):
                    for Z in np.arange(self.length):
                        E += self.K*self.state[X,Y,Z,1]*self.state[X,Y,Z,2]*self.state[X,Y,self.pbc(Z+1),1]*self.state[X,self.pbc(Y+1),Z,2]
                        E += self.K*self.state[X,Y,Z,0]*self.state[X,Y,Z,2]*self.state[X,Y,self.pbc(Z+1),0]*self.state[self.pbc(X+1),Y,Z,2]
                        E += self.K*self.state[X,Y,Z,0]*self.state[X,Y,Z,1]*self.state[X,self.pbc(Y+1),Z,0]*self.state[self.pbc(X+1),Y,Z,1]
                        
                        E += self.B*self.state[X,Y,Z,3]*self.state[X,Y,Z,0]*self.state[self.pbc(X+1),Y,Z,3]
                        E += self.B*self.state[X,Y,Z,3]*self.state[X,Y,Z,1]*self.state[X,self.pbc(Y+1),Z,3]
                        E += self.B*self.state[X,Y,Z,3]*self.state[X,Y,Z,2]*self.state[X,Y,self.pbc(Z+1),3]

            return E

        def DoSweep(self):

            X_length = np.arange(self.length)
            Y_length = np.arange(self.length)
            Z_length = np.arange(self.length)
            np.random.shuffle(X_length)
            np.random.shuffle(Y_length)
            np.random.shuffle(Z_length)

            for w in np.arange(4):
                for X in X_length:
                    for Y in Y_length:
                        for Z in Z_length:
                            b = np.random.randint(0,4)
                            E = -1*self.GetFlipEnergy(X,Y,Z,b)

                            # Is the spin flipped or not?
                            if E<=0:
                                self.state[X,Y,Z,b] *= -1
                            elif np.exp(-E) > np.random.rand():
                                self.state[X,Y,Z,b] *= -1

    print ("L =", L_val)
    print ("K =", K_val)
    print ("B =", B_val)

    rand_lattice = Lattice(K_val, B_val, L_val, state)

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
    plt.title("Autocorr. of 3D LGT model coupled to matter, L={}, K={}, B={}".format(L_val, K_val, B_val))

    op, pcov = opt.curve_fit(Exponential, ac_t[:max_step], ac_function[:max_step])

    print ("OP:", op)
    print ("pcov:", pcov)

    plt.plot(np.array(ac_t[:max_step]), Exponential(np.array(ac_t[:max_step]), *op), label="y = {} * exp(-x/{}) + {}".format(round(op[0],4), round(op[1],4), round(op[2],4)), color="blue")
    plt.scatter(ac_t[:max_step], ac_function[:max_step], s=4, color="red")

    plt.legend()
    fig.savefig("3D_LGT_matter_model_autocorrelation_L_{}_K_{}_B_{}.png".format(L_val, K_val, B_val))
    plt.show()

if __name__ == "__main__":

    K = 1.15296
    B = 0.22
    L = 15

    autocorrelation_Matter(K, B, L, 50000)
    
