import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import scipy

import divergences as dv
import kernels as kl
import gradient_descent as gd
import generate_y as gy

##############################
######## PARAMETERS ##########
##############################

d = 2 #dimension of the particles 
n = 50 # nombre de particules pour q
m = 500 # nombre de particules pour p
T = 200 # nombre d'itérations
h = 0.5 # stepsize gradient descent
eps = 0.0001

config_y = lambda : gy.shape("heart.png",m)




####### INITIAL DISTRIBUTIONS P AND Q  ########

x0 = scs.multivariate_normal.rvs(np.array([np.cos(np.pi/4),np.sin(np.pi/4)]),0.02 * np.identity(2),n) 
y = config_y()


### KERNEL ###
sigm = lambda X,Y : 4 #np.max(np.linalg.norm(X-Y,axis = 1)) / (np.sqrt(1000 * np.log(10))) # np.abs(np.mean(np.linalg.norm(X,axis = 1)) - np.mean(np.linalg.norm(Y,axis = 1)))#max(2,np.linalg.norm(np.mean(x) - np.mean(y)))
k = lambda x,y,s :  kl.k_gauss(x,y,s)
dk = lambda x,y,s : kl.dk_gauss(x, y, s)


#### Matrice Ky, eigenvalues and eigenvectors ####
Ky = 1/m * np.array([[k(y[i],y[j],sigm(x0,y)) for i in range(m)] for j in range(m)])
Ly,V = np.linalg.eig(Ky)
V = V.transpose()
Ly = np.real(Ly)
Packy = [Ky,Ly,V]

#### DIVERGENCE ####
J = lambda x : dv.KKL(x, y, lambda u,v : k(u,v,sigm(x,y)),Packy) 
dJ = lambda x : np.array([dv.WGrad_KKL(x[i],x, y,lambda u,v : k(u,v,sigm(x,y)), lambda u,v : dk(u,v,sigm(x,y)),Packy) for i in range(n)])



###########################################
############ GRADIENT DESCENT #############
###########################################



X,l_J,Grad = gd.gradient_descent(J, dJ, x0, h, eps, T)


############################
########## PLOTS ###########
############################

fig, axs = plt.subplots(5, 4, figsize=(20,20))
for i in range(0,len(X)-1,T//20):
    j = i//(T//20)
    #axs[j//4,j%4].axis([-3,6,-2,10])
    axs[j//4,j%4].scatter(y[:,0],y[:,1],color = "orange")
    axs[j//4,j%4].scatter(X[i,:,0], X[i,:,1], color = "blue")
    

plt.figure()  
plt.scatter(y[:,0],y[:,1],color = "orange")  
plt.scatter(X[-1,:,0], X[-1,:,1])
plt.title("sigma = " + str(sigm(x0,y)) + ", h = " + str(h))




plt.figure()    
plt.plot(l_J)
plt.title("values of J")

plt.figure()
plt.plot(Grad)
plt.title("Values of the gradient of J")
    

