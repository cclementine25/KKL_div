import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f,df,x0,h,eps,n_it_max):
    x = x0 
    grad = df(x)
    X = [x0]
    i = 0 
    liste_f = []
    Grad = []
    while np.linalg.norm(grad) > eps and i < n_it_max:
        liste_f.append(f(x))
        grad = df(x)
        Grad.append(np.linalg.norm(grad))
        x = x - h * grad
        X.append(x)
        i = i + 1
    return np.array(X), liste_f,Grad
        

