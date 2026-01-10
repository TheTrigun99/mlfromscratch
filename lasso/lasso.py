import numpy as np
class lasso:
    def __init__(self,X):
        self.X       =  X
        self.mu_X    =  self.X.mean(axis=0)
        self.sigma_X =  self.X.std(axis=0, ddof=0)
        self.sigma_X[self.sigma_X == 0] = 1.0
        self.Xc      =  (self.X - self.mu_X) / self.sigma_X
        self.n,self.p=np.shape(self.Xc)
        self.Xn2 = (self.Xc ** 2).sum(axis=0)
    
    def s_threshold(self,z, a):
        return np.sign(z) * max(abs(z) - a, 0.0)
        

    def cd_lasso(self,iter,a,y,b_i,tol):
        y = np.asarray(y).reshape(-1)
        yc= y - y.mean()
        beta = b_i
        for i in range(iter):
            beta_old= beta.copy()
            for j in range(self.p):
                
                x_j=self.Xc[:,j]
                
                r_j=yc-self.Xc@beta+x_j*beta[j]
                
                xr=np.dot(x_j,r_j) 

                beta[j]=self.s_threshold(xr,a)/self.Xn2[j]
            if np.max(np.abs(beta - beta_old)) < tol: #critère avec norme inf pour pas faire d'iter inutile
                break
        return beta #,y.mean()-X.mean(axis=0)@beta
    
    def lasso_path(self,y,iter,tol):
        y = np.asarray(y).reshape(-1)
        yc= y - y.mean()
        a_max = np.max(np.abs(self.Xc.T @ yc))
        a_all = a_max * np.logspace(0, np.log10(1e-3), 200)
        beta=np.zeros(self.p)
        path=[]
        for a in a_all:
            beta = self.cd_lasso(iter,a,y,beta,tol)
            path.append(beta.copy())
        return a_all, np.array(path).T

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X = data.data
y = data.target
model=lasso(X)
a_all,path=model.lasso_path(y,2000,1e-6)
plt.figure(figsize=(8, 5))
for j in range(path.shape[0]):
    plt.plot(np.log10(a_all), path[j, :], lw=1)
plt.xlabel(r'$\log_{10}(\alpha)$')
plt.ylabel(r'Coefficient $\beta_j$ (sur X standardisé)')
plt.title('LASSO regularization path (coordinate descent)')
plt.axhline(0, color='black', lw=0.5)
plt.gca().invert_xaxis()
plt.show()