import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

def gradient_descent(J,dJ,x0,h,eps,n_it_max):
    x = x0 
    grad = dJ(x)
    X = [x0]
    i = 0 
    liste_J = []
    Grad = []
    while np.linalg.norm(grad) > eps and i < n_it_max:
        liste_J.append(J(x))
        #y = x + 0.1 * scs.multivariate_normal.rvs(np.zeros(2),np.identity(2),len(x))
        grad = dJ(x)
        Grad.append(np.linalg.norm(grad))
        x = x - h * grad
        X.append(x)
        i = i + 1
    return np.array(X), liste_J,Grad
        
