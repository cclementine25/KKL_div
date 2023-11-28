


000000000000000000000000000
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




000000000000000000
# ###### GAUSSIAN ######
# #initial distribution
# mux = np.array([2,5])
# Lx = np.array([[1/2,1/3],[1/4,-2]])
# Sigmax = Lx @ Lx.transpose()
# #x0 = scs.multivariate_normal.rvs(mux,Sigmax,n)

# #Simulation of (Y_i)_i<n ~ p -> objective distribution
# muy = np.array([0,0])
# Ly = np.array([[1/5, -1],[1/2,1/2]])
# Sigmay = Ly @ Ly.transpose()
# #y = scs.multivariate_normal.rvs(muy,Sigmay,m)



# # ########### Mixture de gaussienne ###########
# MU = np.array([[-2,-1],[5,0]])
# Z = np.random.choice([0,1],m,p=[1/2,1/2])
# #y = np.array([scs.multivariate_normal.rvs(MU[Z[i]],0.5 * np.identity(2)) for i in range(m)])


# ########## RINGS ##########
# y,x0 = rg.generate_rings(n,m, 0.5,1, 0.5,1)

# ########################

# x0 = scs.multivariate_normal.rvs(muy,0.2 * np.diag([1,2]),n)


0000000000000000000
# def GD_k_gauss(J,dJ,x0,h,eps,n_it_max):
#     x = x0  
#     grad = dJ(x)
#     X = [x0]
#     i = 0 
#     liste_J = []
#     Grad = []
#     while np.linalg.norm(grad) > eps and i < n_it_max:
#         liste_J.append(J(x))
#         grad = dJ(x)
#         Grad.append(np.linalg.norm(grad))
#         x = x - h * grad
#         X.append(x)
#         i = i + 1
#     return np.array(X), liste_J,Grad


0000000000000000000000000
