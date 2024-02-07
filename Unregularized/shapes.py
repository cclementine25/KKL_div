import logging 
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs


def generate_rings(n,m,a,b,_delta,nb_rings):
    m = m//nb_rings
    y = np.c_[a * np.cos(np.linspace(0, 2 * np.pi, m + 1)), b * np.sin(np.linspace(0, 2 * np.pi, m + 1))][:-1]  # noqa
    for i in range(nb_rings - 1):#, 2]:
         y = np.r_[y, y[:m, :]-(i+1)*np.array([0, (2 + _delta) * a])]

    rs = np.random.RandomState(42)
    #Y = scs.multivariate_normal.rvs(np.zeros(2),r/2 * np.identity(2),m)
    x = rs.randn(n, 2) / 100 - np.array([a/np.sqrt(2), b/np.sqrt(2)])
    #Y = rs.randn(N*(2+1), 2) / 100 - np.array([0, r])

    return x, y


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

def _get_shapes_data(img,m):
    rs = np.random.RandomState(42)
    #X = _load_img('img/spiral3d.jpg')
    #X = X[rs.choice(len(X), 2000, replace=False)]

    Y = _load_img(img)
    Y = Y[rs.choice(len(Y), m, replace=False)]

    #Y = torch.from_numpy(Y).float()
    #X = torch.from_numpy(X).float()
    #Y.requires_grad = True
    return  Y









