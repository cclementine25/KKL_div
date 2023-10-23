import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import scipy

import divergences as dv
import kernels as kl
import gradient_descent as gd

d = 2 #dimension of the particles 
n = 100 # nombre de particules pour q
m = 100 # nombre de particules pour p
T = 100 # nombre d'itérations

#kernel
k = lambda x,y : kl.k_poly2(x,y)
dk = lambda x,y : kl.dk_poly2(x, y)



######################################
mux = np.array([5,5])
Lx = np.array([[1/2,1/3],[1/4,-2]])
Sigmax = Lx @ Lx.transpose()
x0 = scs.multivariate_normal.rvs(mux,Sigmax,n)
x00 = np.concatenate([x0[:,0],x0[:,1]])


#Simulation of (Y_i)_i<n ~ p -> objective distribution
muy = np.array([0,0])
Ly = np.array([[1/5, -1],[1/2,1/2]])
Sigmay = Ly @ Ly.transpose()
y = scs.multivariate_normal.rvs(muy,Sigmax,n)


########################
#divergencve
J = lambda x : dv.KKL(np.array([x[:n],x[n:]]).transpose(), y, k) #np.array([x[:n//2],x[n//2:]])
dJ = lambda x : dv.grad_MMD(x, y, k, dk)


###########################################
############ GRADIENT DESCENT #############
###########################################


####### MMD #######


res = scipy.optimize.minimize(J, x00)
x_fin = res.x

plt.scatter(y[:,0],y[:,1],color = "red")
plt.scatter(x_fin[:n],x_fin[n:],color = "blue")




# X,l_f,Grad = gd.gradient_descent(J, dJ, x0, 1, 0.001, 500)

# for i in range(7):
#     plt.figure()
#     plt.scatter(X[i,:,0], X[i,:,1])
#     plt.scatter(y[:,0],y[:,1])

# plt.figure()    
# plt.plot(l_f,".")

# plt.figure()
# plt.plot(Grad,".")
    

