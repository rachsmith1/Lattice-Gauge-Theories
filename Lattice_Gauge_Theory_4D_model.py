from __future__ import division
import numpy as np
from numba import jitclass
from numba import boolean
from numba.numpy_support import from_dtype

def DoTemperatureSweep():

    L_val = np.int64(10)
    K_min = np.float64(0.37)
    K_max = np.float64(0.50)
    isRandom = True

    K_going_up = np.linspace(K_min, K_max, 10000)
    K_going_down = np.flip(K_going_up)
    K_up = K_going_up[::int(len(K_going_up)/30)]
    K_down = np.flip(K_up)

    """
    if isRandom:
       state = np.random.randint(0, 2, (L_val, L_val, L_val, L_val, 4))
       state[state==0] = -1
    else:
       state = np.ones((L_val, L_val, L_val, L_val, 4))
    """
    state = np.ones((L_val, L_val, L_val, L_val, 4))

    spec = [
        ('K', from_dtype(K_min.dtype)),          
        ('length', from_dtype(L_val.dtype)),
        ('isRandom', boolean),
        ('state', from_dtype(state.dtype)[:,:,:,:,:])       
    ]

    @jitclass(spec)
    class Lattice:

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

        def GetFlipEnergy(self, X, Y, Z, W, f):

            e = 0

            if f==0:
                e += -2*(
                    self.state[X,Y,Z,W,0] * self.state[X,Y,Z,W,1] * self.state[X,self.pbc(Y+1),Z,W,0] * self.state[self.pbc(X+1),Y,Z,W,1] +
                    self.state[X,self.pbc(Y-1),Z,W,0] * self.state[X,self.pbc(Y-1),Z,W,1] * self.state[X,Y,Z,W,0] * self.state[self.pbc(X+1),self.pbc(Y-1),Z,W,1] +

                    self.state[X,Y,Z,W,2] * self.state[X,Y,Z,W,0] * self.state[self.pbc(X+1),Y,Z,W,2] * self.state[X,Y,self.pbc(Z+1),W,0] +
                    self.state[X,Y,self.pbc(Z-1),W,2] * self.state[X,Y,self.pbc(Z-1),W,0] * self.state[self.pbc(X+1),Y,self.pbc(Z-1),W,2] * self.state[X,Y,Z,W,0] +
                    
                    self.state[X,Y,Z,W,0] * self.state[X,Y,Z,W,3] * self.state[X,Y,Z,self.pbc(W+1),0] * self.state[self.pbc(X+1),Y,Z,W,3] +
                    self.state[X,Y,Z,self.pbc(W-1),0] * self.state[X,Y,Z,self.pbc(W-1),3] * self.state[X,Y,Z,W,0] * self.state[self.pbc(X+1),Y,Z,self.pbc(W-1),3]
                )
           
            elif f==1:
                e += -2*(
                    self.state[X,Y,Z,W,0] * self.state[X,Y,Z,W,1] * self.state[X,self.pbc(Y+1),Z,W,0] * self.state[self.pbc(X+1),Y,Z,W,1] +
                    self.state[self.pbc(X-1),Y,Z,W,0] * self.state[self.pbc(X-1),Y,Z,W,1] * self.state[self.pbc(X-1),self.pbc(Y+1),Z,W,0] * self.state[X,Y,Z,W,1] +
                    
                    self.state[X,Y,Z,W,1] * self.state[X,Y,Z,W,2] * self.state[X,Y,self.pbc(Z+1),W,1] * self.state[X,self.pbc(Y+1),Z,W,2] +
                    self.state[X,Y,self.pbc(Z-1),W,1] * self.state[X,Y,self.pbc(Z-1),W,2] * self.state[X,Y,Z,W,1] * self.state[X,self.pbc(Y+1),self.pbc(Z-1),W,2] +
                    
                    self.state[X,Y,Z,W,1] * self.state[X,Y,Z,W,3] * self.state[X,Y,Z,self.pbc(W+1),1] * self.state[X,self.pbc(Y+1),Z,W,3] +
                    self.state[X,Y,Z,self.pbc(W-1),1] * self.state[X,Y,Z,self.pbc(W-1),3] * self.state[X,Y,Z,W,1] * self.state[X,self.pbc(Y+1),Z,self.pbc(W-1),3]
                )

            elif f==2:
                e += -2*(
                    self.state[X,Y,Z,W,1] * self.state[X,Y,Z,W,2] * self.state[X,Y,self.pbc(Z+1),W,1] * self.state[X,self.pbc(Y+1),Z,W,2] +
                    self.state[X,self.pbc(Y-1),Z,W,1] * self.state[X,self.pbc(Y-1),Z,W,2] * self.state[X,self.pbc(Y-1),self.pbc(Z+1),W,1] * self.state[X,Y,Z,W,2] +
                
                    self.state[X,Y,Z,W,2] * self.state[X,Y,Z,W,0] * self.state[self.pbc(X+1),Y,Z,W,2] * self.state[X,Y,self.pbc(Z+1),W,0] +
                    self.state[self.pbc(X-1),Y,Z,W,2] * self.state[self.pbc(X-1),Y,Z,W,0] * self.state[X,Y,Z,W,2] * self.state[self.pbc(X-1),Y,self.pbc(Z+1),W,0] +
                
                    self.state[X,Y,Z,W,2] * self.state[X,Y,Z,W,3] * self.state[X,Y,Z,self.pbc(W+1),2] * self.state[X,Y,self.pbc(Z+1),W,3] +
                    self.state[X,Y,Z,self.pbc(W-1),2] * self.state[X,Y,Z,self.pbc(W-1),3] * self.state[X,Y,Z,W,2] * self.state[X,Y,self.pbc(Z+1),self.pbc(W-1),3]
                )
        
            elif f==3:
                e += -2*(
                    self.state[X,Y,Z,W,0] * self.state[X,Y,Z,W,3] * self.state[X,Y,Z,self.pbc(W+1),0] * self.state[self.pbc(X+1),Y,Z,W,3] +
                    self.state[self.pbc(X-1),Y,Z,W,0] * self.state[self.pbc(X-1),Y,Z,W,3] * self.state[self.pbc(X-1),Y,Z,self.pbc(W+1),0] * self.state[X,Y,Z,W,3] +
                
                    self.state[X,Y,Z,W,1] * self.state[X,Y,Z,W,3] * self.state[X,Y,Z,self.pbc(W+1),1] * self.state[X,self.pbc(Y+1),Z,W,3] +
                    self.state[X,self.pbc(Y-1),Z,W,1] * self.state[X,self.pbc(Y-1),Z,W,3] * self.state[X,self.pbc(Y-1),Z,self.pbc(W+1),1] * self.state[X,Y,Z,W,3] +
                    
                    self.state[X,Y,Z,W,2] * self.state[X,Y,Z,W,3] * self.state[X,Y,Z,self.pbc(W+1),2] * self.state[X,Y,self.pbc(Z+1),W,3] +
                    self.state[X,Y,self.pbc(Z-1),W,2] * self.state[X,Y,self.pbc(Z-1),W,3] * self.state[X,Y,self.pbc(Z-1),self.pbc(W+1),2] * self.state[X,Y,Z,W,3]
                )

            return e

        def GetInternalEnergy(self):
            E=0

            for X in np.arange(self.length):
               for Y in np.arange(self.length):
                  for Z in np.arange(self.length):
                     for W in np.arange(self.length):
                        E += self.state[X,Y,Z,W,0] * self.state[X,Y,Z,W,1] * self.state[X,self.pbc(Y+1),Z,W,0] * self.state[self.pbc(X+1),Y,Z,W,1]
                        E += self.state[X,Y,Z,W,1] * self.state[X,Y,Z,W,2] * self.state[X,Y,self.pbc(Z+1),W,1] * self.state[X,self.pbc(Y+1),Z,W,2]
                        E += self.state[X,Y,Z,W,2] * self.state[X,Y,Z,W,0] * self.state[self.pbc(X+1),Y,Z,W,2] * self.state[X,Y,self.pbc(Z+1),W,0]

                        E += self.state[X,Y,Z,W,0] * self.state[X,Y,Z,W,3] * self.state[X,Y,Z,self.pbc(W+1),0] * self.state[self.pbc(X+1),Y,Z,W,3]
                        E += self.state[X,Y,Z,W,1] * self.state[X,Y,Z,W,3] * self.state[X,Y,Z,self.pbc(W+1),1] * self.state[X,self.pbc(Y+1),Z,W,3]
                        E += self.state[X,Y,Z,W,2] * self.state[X,Y,Z,W,3] * self.state[X,Y,Z,self.pbc(W+1),2] * self.state[X,Y,self.pbc(Z+1),W,3]
    
            return E/(self.length)**4

        def DoSweep(self):
    
            X_length = np.arange(self.length)
            Y_length = np.arange(self.length)
            Z_length = np.arange(self.length)
            W_length = np.arange(self.length)
            np.random.shuffle(X_length)
            np.random.shuffle(Y_length)
            np.random.shuffle(Z_length)
            np.random.shuffle(W_length)

            for f in np.arange(4):
                for X in X_length:
                    for Y in Y_length:
                        for Z in Z_length:
                           for W in W_length:
                        
                                E = -1*self.GetFlipEnergy(X,Y,Z,W,f)

                                # Is the spin flipped or not?
                                if E<=0:
                                    self.state[X,Y,Z,W,f] *= -1
                                elif np.exp(-E*self.K) > np.random.rand():
                                    self.state[X,Y,Z,W,f] *= -1


    rand_lattice = Lattice(K_min, L_val, state)

    E_up = np.zeros(len(K_up))
    E_down = np.zeros(len(K_down))
    E2_up = np.zeros(len(K_up))
    E2_down = np.zeros(len(K_down))

    iter = 0.0
    for i in np.arange(1):
        if i%1==0:
            print ("Sweep:", i)

        iter += 1.0
        
        for K_i, K_v in enumerate(K_going_down):
            if K_i%50==0:
               print (K_i, K_v)
            rand_lattice.K = K_v
            rand_lattice.DoSweep()
            for Kwrite_i, Kwrite_v in enumerate(K_down):
                if K_v == Kwrite_v:
                    internal_energy = rand_lattice.GetInternalEnergy()
                    E_down[Kwrite_i] += internal_energy
                    E2_down[Kwrite_i] += (internal_energy)**2
                    print ("K = {} | E = {}".format(K_v, E_down[Kwrite_i]/iter))

        for K_i, K_v in enumerate(K_going_up):
            if K_i%50==0:
               print (K_i, K_v)
            rand_lattice.K = K_v
            rand_lattice.DoSweep()
            for Kwrite_i, Kwrite_v in enumerate(K_up):
                if K_v == Kwrite_v:
                    internal_energy = rand_lattice.GetInternalEnergy()
                    E_up[Kwrite_i] += internal_energy
                    E2_up[Kwrite_i] += (internal_energy)**2
                    print ("K = {} | E = {}".format(K_v, E_up[Kwrite_i]/iter))

        

        print (" ")
        print ("K up")
        for K in K_up:
            print (K)

        print (" ")
        print ("E up")
        for E in E_up:
            print (E/iter)

        print (" ")
        print ("E2 up")
        for E2 in E2_up:
            print (E2/iter)

        print (" ")
        print ("E down")
        for E in np.flip(E_down):
            print (E/iter)

        print (" ")
        print ("E2 down")
        for E2 in np.flip(E2_down):
            print (E2/iter)

    
