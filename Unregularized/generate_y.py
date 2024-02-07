import numpy as np
import scipy.stats as scs 
import shapes as sh
import sklearn
import sklearn.datasets



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


#################################

def inf_train_gen(data, rng=None, batch_size=200):
  """Sample batch of synthetic data."""
  if rng is None:
    rng = np.random.RandomState()

  if data == "swissroll":
    data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
    data = data.astype("float32")[:, [0, 2]]
    data /= 5
    return data

  elif data == "circles":
    data = sklearn.datasets.make_circles(
        n_samples=batch_size, factor=.5, noise=0.08)[0]
    data = data.astype("float32")
    data *= 3
    return data

  elif data == "moons":
    data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
    data = data.astype("float32")
    data = data * 2 + np.array([-1, -0.2])
    return data

  elif data == "8gaussians":
    scale = 4.
    centers = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
    centers = [(scale * x, scale * y) for x, y in centers]

    dataset = []
    for _ in range(batch_size):
      point = rng.randn(2) * 0.5
      idx = rng.randint(8)
      center = centers[idx]
      point[0] += center[0]
      point[1] += center[1]
      dataset.append(point)
    dataset = np.array(dataset, dtype="float32")
    dataset /= 1.414
    return dataset

  elif data == "pinwheel":
    radial_std = 0.3
    tangential_std = 0.1
    num_classes = 5
    num_per_class = batch_size // 5
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = rng.randn(
        num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles),
                          np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

  elif data == "2spirals":
    n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
    d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += np.random.randn(*x.shape) * 0.1
    return x

  elif data == "checkerboard":
    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

  elif data == "line":
    x = rng.rand(batch_size) * 5 - 2.5
    y = x
    return np.stack((x, y), 1)
  elif data == "cos":
    x = rng.rand(batch_size) * 5 - 2.5
    y = np.sin(x) * 2.5
    return np.stack((x, y), 1)
  else:
    raise NotImplementedError

    


    
    
