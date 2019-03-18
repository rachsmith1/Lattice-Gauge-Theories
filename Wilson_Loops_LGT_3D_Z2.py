from __future__ import division
import numpy as np
from numba import jitclass
from numba import boolean
from numba.numpy_support import from_dtype
import multiprocessing

def Run(vals):

    dt = 80
    N = 3000
    eq_time = 2000

    L_val = np.int64(vals[0])
    K_val = np.float64(vals[1])

    state = np.random.randint(0, 2, (L_val, L_val, L_val, 3))
    state[state==0] = -1
    state = np.float64(state)

    sizes = np.array([])
    for i in np.arange(1,L_val):
        sizes = np.append(sizes, i)
    sizes = np.int64(sizes)
    values = np.float64(np.zeros(np.size(sizes)))

    spec = [
        ('K', from_dtype(K_val.dtype)),          
        ('length', from_dtype(L_val.dtype)),
        ('state', from_dtype(state.dtype)[:,:,:,:]),     
        ('sizes', from_dtype(sizes.dtype)[:]),
        ('values', from_dtype(values.dtype)[:])
    ]
        
    @jitclass(spec)
    class Lattice(object): 
 
        def __init__(self, K, L, state, sizes, values):  

            self.K = K
            self.length = L
            self.state = state
            self.sizes = sizes
            self.values = values

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

            # Bond is on the x-axis
            if b==0:
                e += -2*self.state[X,Y,Z,0]*(
                            self.state[X,Y,Z,2] * self.state[X,Y,self.pbc(Z+1),0] * self.state[self.pbc(X+1),Y,Z,2] +
                            self.state[X,Y,Z,1] * self.state[X,self.pbc(Y+1),Z,0] * self.state[self.pbc(X+1),Y,Z,1] +
                            self.state[X,Y,self.pbc(Z-1),2] * self.state[X,Y,self.pbc(Z-1),0] * self.state[self.pbc(X+1),Y,self.pbc(Z-1),2] +
                            self.state[X,self.pbc(Y-1),Z,1] * self.state[X,self.pbc(Y-1),Z,0] * self.state[self.pbc(X+1),self.pbc(Y-1),Z,1]
                        )

            # Bond is on the y-axis
            if b==1:
                e += -2*self.state[X,Y,Z,1]*(
                            self.state[X,Y,Z,2] * self.state[X,Y,self.pbc(Z+1),1] * self.state[X,self.pbc(Y+1),Z,2] +
                            self.state[X,Y,Z,0] * self.state[self.pbc(X+1),Y,Z,1] * self.state[X,self.pbc(Y+1),Z,0] +
                            self.state[X,Y,self.pbc(Z-1),2] * self.state[X,Y,self.pbc(Z-1),1] * self.state[X,self.pbc(Y+1),self.pbc(Z-1),2] +
                            self.state[self.pbc(X-1),Y,Z,0] * self.state[self.pbc(X-1),Y,Z,1] * self.state[self.pbc(X-1),self.pbc(Y+1),Z,0]
                        )

            # Bond is on the z-axis
            if b==2:
                e += -2*self.state[X,Y,Z,2]*(
                            self.state[X,Y,Z,1] * self.state[X,self.pbc(Y+1),Z,2] * self.state[X,Y,self.pbc(Z+1),1] +
                            self.state[X,Y,Z,0] * self.state[self.pbc(X+1),Y,Z,2] * self.state[X,Y,self.pbc(Z+1),0] +
                            self.state[X,self.pbc(Y-1),Z,1] * self.state[X,self.pbc(Y-1),Z,2] * self.state[X,self.pbc(Y-1),self.pbc(Z+1),1] +
                            self.state[self.pbc(X-1),Y,Z,0] * self.state[self.pbc(X-1),Y,Z,2] * self.state[self.pbc(X-1),Y,self.pbc(Z+1),0]
                        )

            return e

        def GetInternalEnergy(self):
            E=0
 
            for X in np.arange(self.length):
                for Y in np.arange(self.length):
                    for Z in np.arange(self.length):
                        E += self.state[X,Y,Z,1]*self.state[X,Y,Z,2]*self.state[X,Y,self.pbc(Z+1),1]*self.state[X,self.pbc(Y+1),Z,2]
                        E += self.state[X,Y,Z,0]*self.state[X,Y,Z,2]*self.state[X,Y,self.pbc(Z+1),0]*self.state[self.pbc(X+1),Y,Z,2]
                        E += self.state[X,Y,Z,0]*self.state[X,Y,Z,1]*self.state[X,self.pbc(Y+1),Z,0]*self.state[self.pbc(X+1),Y,Z,1]
 
            return E/(self.length)**3

        def DoSweep(self):

            X_length = np.arange(self.length)
            Y_length = np.arange(self.length)
            Z_length = np.arange(self.length)
            np.random.shuffle(X_length)
            np.random.shuffle(Y_length)
            np.random.shuffle(Z_length)

            #for b in np.arange(3):
            for X in X_length:
                for Y in Y_length:
                    for Z in Z_length:
                        
                        b = np.random.randint(0, 3)

                        E = -1*self.GetFlipEnergy(X,Y,Z,b)
 
                        # Is the spin flipped or not?
                        if E<=0:
                            self.state[X,Y,Z,b] *= -1
                        elif np.exp(-E*self.K) > np.random.rand():
                            self.state[X,Y,Z,b] *= -1

        def DoWilsonLoopUpdate(self):

            new_values = []

            for s in self.sizes:
                s = int(s)
    
                iter = np.float64(0)
                corr = np.float64(0)
    
                for X in np.arange(self.length - s):
                    for Y in np.arange(self.length - s):
                        for Z in np.arange(self.length - s):
                            iter += 3

                            X_corr = np.float64(1)
                            Y_corr = np.float64(1)
                            Z_corr = np.float64(1)

                            for i in np.arange(s):
                                X_corr *= self.state[X,Y+i,Z,1] * self.state[X,Y,Z+i,2] * self.state[X,Y+i,Z+s,1] * self.state[X,Y+s,Z+i,2]
                                Y_corr *= self.state[X+i,Y,Z,0] * self.state[X,Y,Z+i,2] * self.state[X+i,Y,Z+s,0] * self.state[X+s,Y,Z+i,2]
                                Z_corr *= self.state[X+i,Y,Z,0] * self.state[X,Y+i,Z,1] * self.state[X+i,Y+s,Z,0] * self.state[X+s,Y+i,Z,1]

                            corr += X_corr
                            corr += Y_corr
                            corr += Z_corr

                corr = corr/iter
                new_values.append(corr)

            self.values += np.array(new_values)

    print ("L =", L_val)
    print ("K =", K_val)

    HT_lattice = Lattice(K_val, L_val, state, sizes, values)

    # Get to equilibrium
    for i in range(eq_time):
        if i%1000==0:
            print ("Equilibriating:", i)
        HT_lattice.DoSweep()

    HT_E = 0
    iter = 0
    for i in range(N*dt):
        if i%dt==0:

            iter += 1
            HT_lattice.DoWilsonLoopUpdate()
        
        if i%5000==0:
            print ("Sweep #:", i)
            print ("K =", K_val)
            print ("L =", L_val)
            for s, v in zip(HT_lattice.sizes, HT_lattice.values):
               print ("{}x{}: {}".format(s, s, v/iter))
            print (" ")


        #LT_lattice.DoSweep()
        HT_lattice.DoSweep()

    return HT_lattice.values/iter

if __name__ == "__main__":

    L = 20
    K_min = 0.5
    K_max = 0.6

    K = np.linspace(K_min, K_max, 10)
    K_vals = [[L,K_i] for K_i in K]

    all_wls = []
    for K_i in K_vals:
        wls = Run(K_i)
        all_wls.append(wls)

    print ("K:")
    for K_i in K:
       print (K_i)

    print (" ")

    for s in np.arange(len(all_wls[0])):
       print ("{}x{}:".format(s+1, s+1))

       for K_val in all_wls:
          print(K_val[int(s)])

       print (" ")





    
