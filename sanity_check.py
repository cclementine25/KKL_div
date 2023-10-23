import numpy as np
import matplotlib.pyplot as plt 

import divergences as dv
import kernels as kl

#x~p, y~q, p  = sum alpha_i delta_x_i and q = 1/n sum delta x_i (n = 5 ici)


#les x_i suppports de p et q
vects = np.array([np.array([1,1]),np.array([0,1]),np.array([-1,1]),np.array([3,2]),np.array([-1/2,0])])

#proba de chaque x_i pour la distribution p
alpha = np.array([0.1,0.3,0.1,0.2,0.3]) 


#nombre total de particules = 100 pour chaque distrib 
y = np.repeat(vects, [20,20,20,20,20], axis=0)
x = np.repeat(vects,[10,30,10,20,30],axis = 0)

#kernel
k = lambda x,y : kl.k_gauss(x,y,1)


def KL(alpha):
    return np.sum(alpha * (np.log( alpha * len(alpha))))
    

print(KL(alpha))
print(np.real(dv.KKL(x, y, k)))
print(np.log(len(alpha) * np.max(alpha)))