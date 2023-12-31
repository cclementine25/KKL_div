import numpy as np
import scipy.stats as scs 
import shapes as sh



########################################
def gaussian(muy,sigmay,m):
    y = scs.multivariate_normal.rvs(muy,sigmay,m)
    return y


# ###### GAUSSIAN ######
#initial distribution
mux = np.array([5,5])
Lx = np.array([[1/2,1/3],[1/4,-2]])
Sigmax = Lx @ Lx.transpose()
#x0 = scs.multivariate_normal.rvs(mux,Sigmax,n)

#Simulation of (Y_i)_i<n ~ p -> objective distribution
muy = np.array([0,0])
Ly = np.array([[1/5, -1],[1/2,1/2]])
Sigmay = Ly @ Ly.transpose()
#y = scs.multivariate_normal.rvs(muy,Sigmay,m)

#########################################

def mixt_gauss(Muy,Sigmay,py,m):
    Zy = np.random.choice(np.arange(len(py)),m,p=py)
    y = np.array([scs.multivariate_normal.rvs(Muy[Zy[i]],Sigmay[Zy[i]]) for i in range(m)])
    return y


MU = np.array([[0,0],[10,0]])

############################################

def rings(a,b,_delta,nb_rings,m):
    m = m//nb_rings
    y = np.c_[a * np.cos(np.linspace(0, 2 * np.pi, m + 1)), b * np.sin(np.linspace(0, 2 * np.pi, m + 1))][:-1]  
    for i in range(nb_rings - 1):#, 2]:
         y = np.r_[y, y[:m, :]-(i+1)*np.array([0, (2 + _delta) * b])]
    return y

############################################
    

from PIL import Image
def _load_img(fn='heart.png', size=200, max_samples=None):
    r"""Returns x,y of black pixels (between -1 and 1)
    """
    pic = np.array(Image.open(fn).resize((size,size)).convert('L'))
    y_inv, x = np.nonzero(pic<=128)
    y = size - y_inv - 1
    if max_samples and x.size > max_samples:
        ixsel = np.random.choice(x.size, max_samples, replace=False)
        x, y = x[ixsel], y[ixsel]
    return np.stack((x, y), 1) / size * 2 - 1

def shape(img,m):
    rs = np.random.RandomState(42)
    Y = _load_img(img)
    Y = Y[rs.choice(len(Y), m, replace=False)]
    return Y
    


    
    
