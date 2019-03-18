from __future__ import division
import numpy as np
from numba import jitclass
from numba import boolean
from numba.numpy_support import from_dtype
import multiprocessing

def JackKnife(e, e2, K, L):

    c_array = []
    for i in np.arange(len(e)):
        temp_e = e
        temp_e2 = e2
        temp_e = np.delete(temp_e, i)
        temp_e2 = np.delete(temp_e2, i)

        c = (K**2) / (L**3) * (np.mean(temp_e2) - (np.mean(temp_e))**2)

        c_array.append(c)

    c_array = np.array(c_array)
    
    c_mean = np.mean(c_array)
    c_std = np.sqrt( ((len(c_array)-1)/(len(c_array))) * np.sum( (c_array - c_mean)**2 ) )
    return c_mean, c_std

def Run(vals):

    dt = 50
    N = 6000
    eq_time = 2000
    run_time = dt*N

    L_val = np.int64(vals[0])
    K_val = np.float64(vals[1])
    B_val = np.float64(vals[2])

    #state = np.random.randint(0, 2, (L_val, L_val, L_val, 4))
    #state[state==0] = -1

    state = np.ones((L_val, L_val, L_val, 4))
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
                            self.K*self.B*self.state[X,Y,Z,3]*self.state[self.pbc(X+1),Y,Z,3] 
                        )

            # Spin is on the y-axis bond
            if b==1:
                e += -2*self.state[X,Y,Z,1]*(
                            self.K*self.state[X,Y,Z,2] * self.state[X,Y,self.pbc(Z+1),1] * self.state[X,self.pbc(Y+1),Z,2] +
                            self.K*self.state[X,Y,Z,0] * self.state[self.pbc(X+1),Y,Z,1] * self.state[X,self.pbc(Y+1),Z,0] +
                            self.K*self.state[X,Y,self.pbc(Z-1),2] * self.state[X,Y,self.pbc(Z-1),1] * self.state[X,self.pbc(Y+1),self.pbc(Z-1),2] +
                            self.K*self.state[self.pbc(X-1),Y,Z,0] * self.state[self.pbc(X-1),Y,Z,1] * self.state[self.pbc(X-1),self.pbc(Y+1),Z,0]
                            +
                            self.K*self.B*self.state[X,Y,Z,3]*self.state[X,self.pbc(Y+1),Z,3]
                        )

            # Spin is on the z-axis bond
            if b==2:
                e += -2*self.state[X,Y,Z,2]*(
                            self.K*self.state[X,Y,Z,1] * self.state[X,self.pbc(Y+1),Z,2] * self.state[X,Y,self.pbc(Z+1),1] +
                            self.K*self.state[X,Y,Z,0] * self.state[self.pbc(X+1),Y,Z,2] * self.state[X,Y,self.pbc(Z+1),0] +
                            self.K*self.state[X,self.pbc(Y-1),Z,1] * self.state[X,self.pbc(Y-1),Z,2] * self.state[X,self.pbc(Y-1),self.pbc(Z+1),1] +
                            self.K*self.state[self.pbc(X-1),Y,Z,0] * self.state[self.pbc(X-1),Y,Z,2] * self.state[self.pbc(X-1),Y,self.pbc(Z+1),0]
                            +
                            self.K*self.B*self.state[X,Y,Z,3]*self.state[X,Y,self.pbc(Z+1),3]
                        )

            # Spin is on the vertex 
            if b==3:
                e += -2*self.state[X,Y,Z,3]*(
                            self.K*self.B*self.state[X,Y,Z,0]*self.state[self.pbc(X+1),Y,Z,3] +
                            self.K*self.B*self.state[self.pbc(X-1),Y,Z,3]*self.state[self.pbc(X-1),Y,Z,0] +
                            self.K*self.B*self.state[X,Y,Z,1]*self.state[X,self.pbc(Y+1),Z,3] +
                            self.K*self.B*self.state[X,self.pbc(Y-1),Z,3]*self.state[X,self.pbc(Y-1),Z,1] +
                            self.K*self.B*self.state[X,Y,Z,2]*self.state[X,Y,self.pbc(Z+1),3] +
                            self.K*self.B*self.state[X,Y,self.pbc(Z-1),3]*self.state[X,Y,self.pbc(Z-1),2]
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

    # Get to equilibrium
    for i in range(eq_time):
        if i%500==0:
            print ("Sweep #:", i)
        rand_lattice.DoSweep()

    # Sample internal energy every dt sweeps
    # Print info every 2000 sweeps
    iter = 0
    E = 0
    E2 = 0
   
    E_array = []
    E2_array = []

    for i in range(run_time):
           
        if i%dt==0:
            e = rand_lattice.GetInternalEnergy()
            E += e
            E2 += e**2
            iter += 1

            E_array.append(e)
            E2_array.append(e**2)

        if i%2000==0:
            print ("Sweep #:", i)
            print ("Energy:", E/iter)

        rand_lattice.DoSweep()    
            
    print ("L = {}, K = {}, B = {}, E: {}".format(L_val, K_val, B_val, E/iter))
    print ("L = {}, K = {}, B = {}, E2: {}".format(L_val, K_val, B_val, E2/iter))

    E_array = np.array(E_array)
    E2_array = np.array(E2_array)

    C, C_std = JackKnife(E_array, E2_array, K_val, L_val)

    return [np.mean(E_array), np.mean(E2_array), np.std(E_array), np.std(E2_array), C, C_std] 

if __name__ == "__main__":

    L = 10

    B = np.array([0.141739])
    K = np.linspace(0.7, 0.8, 30)

    #B = np.linspace(0.2, 0.25, 30)
    #K = np.array([5])

    K_vals = []
    for K_i in K:
       for B_i in B:
          K_vals.append([L, K_i, B_i])
    
    """
    E_E2 = []
    for K_v in K_vals:
       E_E2.append(Run(K_v))
    """    
    pool = multiprocessing.Pool(processes=3)
    E_E2 = pool.map(Run, K_vals)
    pool.close()
    print (E_E2)
    
   
    print (" ")
    print ("K:")
    for K_i in K:
       print (K_i)

    print (" ")
    print ("B:")
    for B_i in B:
       print (B_i)

    print (" ")
    print ("E:")
    for K_i in E_E2:
        print (K_i[0])

    print (" ")
    print ("E2:")
    for K_i in E_E2:
        print (K_i[1])

    print (" ")
    print ("E std:")
    for K_i in E_E2:
        print (K_i[2])  

    print (" ")
    print ("E2 std:")
    for K_i in E_E2:
        print (K_i[3])

    print (" ")
    print ("C:")
    for K_i in E_E2:
        print (K_i[4])

    print (" ")
    print ("C std:")
    for K_i in E_E2:
        print (K_i[5])





