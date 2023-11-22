import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import scipy

import divergences as dv
import kernels as kl
import gradient_descent as gd

d = 2 #dimension of the particles 
n = 40 # nombre de particules pour q
m = 50 # nombre de particules pour p
T = 100 # nombre d'itérations

#kernel
k = lambda x,y : kl.k_gauss(x,y,5)
dk = lambda x,y : kl.dk_gauss(x, y,5)


######################################
mux = np.array([3,5])
Lx = np.array([[1/2,1/3],[1/4,-2]])
Sigmax = Lx @ Lx.transpose()
x0 = scs.multivariate_normal.rvs(mux,Sigmax,n)
#x00 = np.concatenate([x0[:,0],x0[:,1]])


#Simulation of (Y_i)_i<n ~ p -> objective distribution
muy = np.array([0,0])
Ly = np.array([[1/5, -1],[1/2,1/2]])
Sigmay = Ly @ Ly.transpose()
y = scs.multivariate_normal.rvs(muy,Sigmay,m)

# #### mixture de gaussienne ####
# MU = np.array([[0,0],[5,0]])
# Z = np.random.choice([0,1],m,p=[1/4,3/4])
# y = np.array([scs.multivariate_normal.rvs(MU[Z[i]],np.identity(2)) for i in range(m)])


########################
#divergencve
sigm = lambda x,y : max(2,np.linalg.norm(np.mean(x) - np.mean(y)))
J = lambda x : dv.KKL(x, y, lambda a,b : kl.k_gauss(a,b, sigm(x,y))) #np.array([x[:n//2],x[n//2:]])
dJ = lambda x : np.array([dv.WGrad_KKL(x[i],x, y,lambda a,b : kl.k_gauss(a,b, sigm(x,y)), lambda a,b : kl.dk_gauss(a,b, sigm(x,y))) for i in range(n)])


###########################################
############ GRADIENT DESCENT #############
###########################################


####### MMD #######


# res = scipy.optimize.minimize(J, x0)
# x_fin = res.x

# plt.scatter(y[:,0],y[:,1],color = "red")
# plt.scatter(x_fin[:n],x_fin[n:],color = "blue")




X,l_J,Grad = gd.gradient_descent(J, dJ, x0, 0.5, 0.0001, 100)

fig, axs = plt.subplots(5, 4, figsize=(20,20))
for i in range(0,len(X)-1,5):
    j = i//5
    axs[j//4,j%4].axis([-3,6,-2,10])
    axs[j//4,j%4].scatter(X[i,:,0], X[i,:,1])
    axs[j//4,j%4].scatter(y[:,0],y[:,1])

plt.figure()    
plt.scatter(X[-1,:,0], X[-1,:,1])
plt.scatter(y[:,0],y[:,1])


# for i in range(0,len(X),5):
#     plt.figure()
#     plt.scatter(X[i,:,0], X[i,:,1])
#     plt.scatter(y[:,0],y[:,1])

plt.figure()    
plt.plot(l_J)
plt.title("values of J")

plt.figure()
plt.plot(Grad)
plt.title("Values of the gradient of J")
    

