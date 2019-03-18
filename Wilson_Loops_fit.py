import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
   return a * np.exp(-b*(x*x)-c*2*(x+x))

def linear(x, a, b):
   return a + b*x

xdata = np.array([1, 2, 3, 4])

#K = 0.5
data0 = np.array([0.50209, 0.06691, 0.00257, -1.09158E-4])

#K = 0.51111
data1 = np.array([0.51491, 0.07434, 0.00296, -4.34028E-6])

#K = 0.52222
data2 = np.array([0.52823, 0.08283, 0.00395, 1.37153E-4])

#K = 0.53333
data3 = np.array([0.54147, 0.09198, 0.005, 3.87912E-5])

#K = 0.54444
data4 = np.array([0.55492, 0.10157, 0.00654, 1.96777E-4])

#K = 0.55556
data5 = np.array([0.56869, 0.11293, 0.0083, 2.51411E-4])

#K = 0.56667
data6 = np.array([0.5827, 0.12493, 0.01064, 3.23079E-4])

#K = 0.57778
data7 = np.array([0.59739, 0.1388, 0.01327, 5.56749E-4])

#K = 0.58889
data8 = np.array([0.61224, 0.15378, 0.01717, 8.09191E-4])

#K = 0.6
data9 = np.array([0.62741, 0.17053, 0.02162, 0.00151])

K = np.array([0.51111, 0.52222, 0.53333, 0.54444, 0.55556, 0.56667, 0.57778, 0.58889, 0.6])
data = np.array([data1, data2, data3, data4, data5, data6, data7, data8, data9])

sigma = []
sigma_err = []

mu = []
mu_err = []

for i in range(len(K)):
   popt, pcov = curve_fit(func, xdata, data[i])
   perr = np.sqrt(np.diag(pcov))

   sigma.append(popt[1])
   sigma_err.append(perr[1])

   mu.append(popt[2])
   mu_err.append(perr[2])

   print(popt[1], popt[2])

T = K
sigma = np.array(sigma)
sigma_err = np.array(sigma_err)
mu = np.array(mu)
mu_err = np.array(mu_err)

popt, pcov = curve_fit(linear, T, sigma)
perr = np.sqrt(np.diag(pcov))
print (" ")
print (popt)
print (perr)

wilson = plt.figure()
plt.xlabel("K")
plt.ylabel("$\sigma$, $\mu$")
plt.title("Contributions to ave. Wilson loop")
plt.yscale("log")
plt.errorbar(T, sigma, yerr=sigma_err, fmt="o", color="red", label="area")
plt.errorbar(T, mu, yerr=mu_err, fmt="o", color="blue", label="perimeter")
plt.legend()
plt.show()
wilson.savefig("wilson-log.png")


