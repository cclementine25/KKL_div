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


def KKL(x, y, k):
    #x~q, y~p
    n = len(x)
    Kx = 1/n * np.array([[k(x[i],x[j]) for i in range(n)] for j in range(n)])
    Ky = 1/n * np.array([[k(y[i],y[j]) for i in range(n)] for j in range(n)])
    Lambdx,_ = np.linalg.eig(Kx)
    Lambdx = np.real(Lambdx)
    Lambdy,Py = np.linalg.eig(Ky)
    Lambdy = np.real(Lambdy)
    logDy = np.diag(log_ou_0(Lambdy)) 
    
    trxx = np.sum(Lambdx * log_ou_0(Lambdy)) #Tr Sigma_q log Sigma_q
    trxy = np.trace(Kx @ Py @ logDy @ Py.transpose())  # Tr Sigma q log Sigma p
    return trxx - trxy 




#gradient KKL

def DxK(dk,dkk,x):
    n = len(x)
    d = len(x[0])
    M = np.zeros((((n,n,n,d))))
    for i in range(n):
        M[i,:,i] = np.array([dk(x[i],x[l],1) for l in range(n)])
        M[i,i,i] = dkk(x[i],x[i],1)
    return M
    


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
    


