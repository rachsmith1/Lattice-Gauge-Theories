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

    dt = 80
    N = 3000
    eq_time = 100
    run_time = dt*N

    L_val = np.int64(vals[0])
    K_val = np.float64(vals[1])

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

            return E

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
            
    print ("L = {}, K = {}, E: {}".format(L_val, K_val, E/iter))
    print ("L = {}, K = {}, E2: {}".format(L_val, K_val, E2/iter))

    E_array = np.array(E_array)
    E2_array = np.array(E2_array)

    C, C_std = JackKnife(E_array, E2_array, K_val, L_val)

    return [np.mean(E_array), np.mean(E2_array), np.std(E_array), np.std(E2_array), C, C_std]  
 

if __name__ == "__main__":

    L = 25
    K_min = 0.2
    K_max = 0.25

    K = np.linspace(K_min, K_max, 30)
    K_vals = [[L,K_i] for K_i in K]
    pool = multiprocessing.Pool(processes=3)
    E_E2 = pool.map(Run, K_vals)
    pool.close()
    print (E_E2)
   
    print (" ")
    print ("K:")
    for K_i in K:
       print (K_i)

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

