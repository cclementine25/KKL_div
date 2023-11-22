import numpy as np
import scipy.stats as scs
import kernels as kl


#################################################################



def MMD(x,y,k):
    n = len(x)
    Kxx = np.array([[k(x[i],x[j]) for i in range(n)] for j in range(n)])
    Kyy = np.array([[k(y[i],y[j]) for i in range(n)] for j in range(n)])
    Kxy = np.array([[k(x[i],y[j]) for i in range(n)] for j in range(n)])
    A = 1/((n-1)*n) * (np.sum(Kxx) - np.sum(np.diag(Kxx)))
    C = 1/((n-1)*n) * (np.sum(Kyy) - np.sum(np.diag(Kyy)))
    B = 1/n**2* np.sum(Kxy)
    return A - B + C


#gradient in x of MMD 
def grad_MMD(x,y,k,dk):
    d = len(x[0])
    n = len(x)
    m = len(y)
    dKx = np.array([[dk(x[i],x[j]) for j in range(n)] for i in range(n)])
    dKx[:,:,0] = dKx[:,:,0] - np.diag(np.diag(dKx[:,:,0]))
    dKx[:,:,1] = dKx[:,:,1] - np.diag(np.diag(dKx[:,:,1]))
    dKy = np.array([[dk(x[i],y[j]) for j in range(m)] for i in range(n)])
    R = np.zeros((n,d))
    R[:,0] = 2/(n * (n-1)) * dKx[:,:,0] @ np.ones(n) - 2/m**2 * dKy[:,:,0] @ np.ones(m)
    R[:,1] = 2/(n * (n-1)) * dKx[:,:,1] @ np.ones(n) - 2/m**2 * dKy[:,:,1] @ np.ones(m)
    return R



def log_ou_0(t):
    t_log = np.zeros(len(t))
    for i in range(len(t)):
        if t[i] > 0:
            t_log[i] = np.log(t[i])
    return t_log




# def KKL(x,y,k):
#     n = len(x)
#     m = len(y)
#     Kx = 1/n * np.array([[k(x[i],x[j]) for i in range(n)] for j in range(n)])
#     Ky = 1/m * np.array([[k(y[i],y[j]) for i in range(m)] for j in range(m)])
#     regx = 1e-9*np.eye(n)
#     regy = 1e-9*np.eye(m)
#     Kx = Kx +regx
#     Ky = Ky+regy
#     Lx,U = np.linalg.eig(Kx)
#     U = np.real(U).transpose()
#     Lx = np.real(Lx)
#     Ly,V = np.linalg.eig(Ky)
#     V = np.real(V).transpose()
#     Ly = np.real(Ly)
#     Trxy = 0
#     Kxy = np.array([[k(x[i],y[j]) for j in range(m)] for i in range(n)])
#     Trxx = np.sum(Lx * log_ou_0(Lx))
#     for s in range(n):
#         for t in range(m):
#             Trxy = Trxy + (Lx[s] * log_ou_0([Ly[t]])[0] * (U[s] @ Kxy @ V[t])**2) / ((U[s] @ (n * Kx) @ U[s]) * (V[t] @ (m*Ky) @ V[t]))
#             # if n*Lx[s] != (U[s] @ (n * Kx) @ U[s]):
#             #     print((U[s] @ (n * Kx) @ U[s]) / (n*Lx[s]))
#                 #print(U[s])#np.abs((U[s] @ Kxy @ V[t])**2 / ((U[s] @ (n * Kx) @ U[s]) * (V[t] @ (m * Ky) @ V[t]))) > 0: #(n*Lx[s]*m*Ly[t])) > 1:#
#                 # print(" n x lambda_x = " + str(n*Lx[s]))
#                 # print("m x lambda_y = " + str(m*Ly[t]))
#                 # print(" produit scalire = " + str((U[s] @ Kxy @ V[t])**2))
#                 # print("norme^2 de f = " + str((U[s] @ (n * Kx) @ U[s])))
#                 # print("norme^2 de g = " + str((V[t] @ (m * Ky) @ V[t])))
#             #print((U[s] @ Kxy @ V[t])**2 / ((U[s] @ (n * Kx) @ U[s]) * (V[t] @ (m*Ky) @ V[t])))
            
#     Trxx = np.sum(Lx * log_ou_0(Lx))
#     #print(UU)
#     return Trxx - Trxy


# def WGrad_KKL(w,x,y,k,dk):
#     n = len(x)
#     m = len(y)
#     Kx = 1/n * np.array([[k(x[i],x[j]) for i in range(n)] for j in range(n)])
#     Ky = 1/m * np.array([[k(y[i],y[j]) for i in range(m)] for j in range(m)])
#     Lx,U = np.linalg.eig(Kx)
#     U = U.transpose()
#     Lx = np.real(Lx)
#     Ly,V = np.linalg.eig(Ky)
#     V = V.transpose()
#     Ly = np.real(Ly)
#     Kwx = np.array([k(w,x[i]) for i in range(n)]).transpose()
#     Kwy = np.array([k(w,y[j]) for j in range(m)]).transpose()
#     DKx = np.array([dk(w,x[i]) for i in range(n)]).transpose()
#     DKy = np.array([dk(w,y[j]) for j in range(m)]).transpose()
#     Trwx = 0
#     Trwy = 0 
#     for s in range(n):
#         Trwx = Trwx + log_ou_0([Lx[s]])[0] * 2 * (U[s] @ Kwx)* (DKx @ U[s]) / (U[s] @ (n * Kx) @ U[s])
#         #print(U[s] @ (n * Kx) @ U[s])
#     for t in range(m):
#         Trwy = Trwy + log_ou_0([Ly[t]])[0] * 2 * (V[t] @ Kwy)* (DKy @ V[t]) / (V[t] @ (n * Ky) @ V[t])
#     return Trwx - Trwy


def KKL(x,y,k):
    n = len(x)
    m = len(y)
    Kx = 1/n * np.array([[k(x[i],x[j]) for i in range(n)] for j in range(n)])
    Ky = 1/m * np.array([[k(y[i],y[j]) for i in range(m)] for j in range(m)])
    regx = 1e-9*np.eye(n)
    regy = 1e-9*np.eye(m)
    Kx = Kx +regx
    Ky = Ky+regy
    Lx,U = np.linalg.eig(Kx)
    U = np.real(U).transpose()
    Lx = np.real(Lx)
    Ly,V = np.linalg.eig(Ky)
    V = np.real(V).transpose()
    Ly = np.real(Ly)
    Trxy = 0
    Kxy = np.array([[k(x[i],y[j]) for j in range(m)] for i in range(n)])
    Trxx = np.sum(Lx * log_ou_0(Lx))
    for s in range(n):
        for t in range(m):
            Trxy = Trxy + log_ou_0([Ly[t]])[0] / Ly[t] * (U[s] @ Kxy @ V[t])**2 
    Trxx = np.sum(Lx * log_ou_0(Lx))
    
    return Trxx - 1/(n*m) * Trxy


def WGrad_KKL(w,x,y,k,dk):
    n = len(x)
    m = len(y)
    Kx = 1/n * np.array([[k(x[i],x[j]) for i in range(n)] for j in range(n)])
    Ky = 1/m * np.array([[k(y[i],y[j]) for i in range(m)] for j in range(m)])
    Lx,U = np.linalg.eig(Kx)
    U = U.transpose()
    Lx = np.real(Lx)
    Ly,V = np.linalg.eig(Ky)
    V = V.transpose()
    Ly = np.real(Ly)
    Kwx = np.array([k(w,x[i]) for i in range(n)]).transpose()
    Kwy = np.array([k(w,y[j]) for j in range(m)]).transpose()
    DKx = np.array([dk(w,x[i]) for i in range(n)]).transpose()
    DKy = np.array([dk(w,y[j]) for j in range(m)]).transpose()
    Trwx = 0
    Trwy = 0 
    for s in range(n):
        Trwx = Trwx + log_ou_0([Lx[s]])[0] / Lx[s] * 2 * (U[s] @ Kwx)* (DKx @ U[s]) 
        #print(U[s] @ (n * Kx) @ U[s])
    for t in range(m):
        Trwy = Trwy + log_ou_0([Ly[t]])[0] / Ly[t] * 2 * (V[t] @ Kwy)* (DKy @ V[t]) 
    return 1/n * Trwx - 1/ m * Trwy
    
                                   
    
       
            



#gradient KKL

# def DxK(dk,dkk,x):
#     n = len(x)
#     d = len(x[0])
#     M = np.zeros((((n,n,n,d))))
#     for i in range(n):
#         M[i,:,i] = np.array([dk(x[i],x[l],1) for l in range(n)])
#         M[i,i,i] = dkk(x[i],x[i],1)
#     return M
    


######## Kernel density estimation ###############

#base distribution sample
x_tau = scs.multivariate_normal.rvs(np.zeros(2),np.identity(2),100)    


def h(x,y,k):
    return np.mean(np.array([k(x,x_tau[i]) * k(y,x_tau[i]) * np.exp(np.linalg.norm(x_tau[i])) /(np.sqrt(2 * np.pi)) for i in range(len(x_tau))]))
    
    

def DE(x,k,y):
    n = len(x)
    return 1/n * np.sum(np.array([h(x[i],y,k) for i in range(n)]))

def KDE(x, y, k):
    n = len(x)
    Q = np.array([DE(x,k,x[i]) for i in range(n)])
    P = np.array([DE(y,k,x[i]) for i in range(n)])
    return 1/n * np.sum(np.log(Q) * Q - np.log(P) * Q)
    
    
    
######### TRACE #######################

def K_trace(x,k):
    n = len(x)
    Kx = 1/n * np.array([[k(x[i],x[j]) for i in range(n)] for j in range(n)])
    Lambdx,_ = np.linalg.eig(Kx)
    return np.sum(Lambdx)
    


