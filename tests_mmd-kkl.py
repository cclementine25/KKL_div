import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt

import divergences as dv
import kernels as kl
import gradient_descent as gd

d = 2 #dimension of the particles 
n = 100 # nombre de particules pour p
m = 100 # nombre de particules pour q
#T = 100 # nombre d'itérations

#kernel
k = lambda x,y : kl.k_gauss(x,y,1)
#dk = lambda x,y : kl.dk_gauss(x, y, 1)




#######SIMULATION OF THE DATA#########

#Simulation of (X_i)_i<n ~ p
mux = np.array([10,10])
Mux = np.array([1/k *np.array([10,10]) for k in range(1,17)])
Lx = np.array([[1/2,1/3],[1/4,-2]])
Sigmax = Lx @ Lx.transpose()
SS = np.array([Sigmax * k for k in [6,5,4,3,2,1]])

#Simulation of (Y_i)_i<n ~ q 
muy = np.array([0,0])
Ly = np.array([[1/5, -1],[1/2,1/2]])
Sigmay = Sigmax #Lx @ Lx.transpose()
Y = scs.multivariate_normal.rvs(muy,Sigmay,n)

#######################################################


########PLOTS#########





""" Here we plot two gaussian distributions with same variance and
different mean and we draw the evolution of MMD, KKL and KDE when the
 means of one of the  distribution become closer to the other one"""

mmd = []
kkl = []
kde = []
k_trace = []

fig, axs = plt.subplots(4, 4, figsize=(20,20))
for j in range(16):
    axs[j//4,j%4].axis([-3,23,-10,30])
    X = scs.multivariate_normal.rvs(Mux[j],Sigmax,n)
    mmd.append(dv.MMD(X,Y,k))
    kkl.append(dv.KKL(X,Y,k))
    #k_trace.append(dv.K_trace(X, k) - dv.K_trace(Y, k))
    kde.append(dv.KDE(X, Y, k))
    axs[j//4,j%4].scatter(X[:,0],X[:,1],color = "green")
    axs[j//4,j%4].scatter(Y[:,0],Y[:,1],color = "blue")

    

    
plt.figure()
plt.plot(mmd,label = "mmd")
plt.title("evolution of mmd for 2 distribution of same variance when their means get closer ")

plt.figure()
plt.plot(kkl,label = "kkl")
#plt.plot(k_trace,label = "k_trace")
plt.legend()
plt.title("kkl / Tr - Tr")
#plt.title("evolution of kkl for 2 distribution of same variance when their means get closer ")

plt.figure()
plt.plot(kde,label = "kde")
plt.title("evolution of kde for 2 distribution of same variance when their means get closer ")


""" same experience switching the roles of the mean and variance"""

mmds = []
kkls = []

fig, axs = plt.subplots(2, 3, figsize=(20,20))
for j in range(6):
    axs[j//3,j%3].axis([-3,23,-10,30])
    X = scs.multivariate_normal.rvs(muy,SS[j],n)
    mmds.append(dv.MMD(X,Y,k))
    kkls.append(dv.KKL(X,Y,k))
    axs[j//3,j%3].scatter(X[:,0],X[:,1],color = "green")
    axs[j//3,j%3].scatter(Y[:,0],Y[:,1],color = "blue")


plt.figure()
plt.plot(mmds,label = "mmd")
plt.plot(kkls,label = "kkl")
plt.legend()
    
plt.figure()
plt.plot(mmds,label = "mmd")
plt.title("evolution of mmd for 2 distribution of same mean when their variances get closer ")

plt.figure()
plt.plot(kkls,label = "kkl")
plt.title("evolution of kkl for 2 distribution of same mean when their variances get closer ")



##########################################################



