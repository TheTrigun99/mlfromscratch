import numpy as np
class ridge:
    def __init__(self,X): #on va pas l'appeler lambda, car en python ça définit des fonctions
        #normalisation
        self.X=X
        self.mu_X = self.X.mean(axis=0)
        self.sigma_X = self.X.std(axis=0, ddof=0)
        self.Xc = (self.X - self.mu_X) / self.sigma_X
        #expliquer full_matrices=False
        self.U, self.d, self.Vt = np.linalg.svd(self.Xc, full_matrices=False) #on a des petits soucis si on garde les directions colinéaires
        self.V=self.Vt.T
        self.Xct=self.Xc.T

    def fit(self, y,alpha):

        self.mu_y = y.mean()
        yc = y - self.mu_y
        shrink = self.d / (self.d**2 + alpha)
        beta_s = self.Vt.T @ (shrink * (self.U.T @ yc))
        self.coef = beta_s / self.sigma_X
        self.intercept = self.mu_y - self.mu_X @ self.coef #expliquer
        return self.coef, self.intercept
    
    def predict(self, Xn=None):
        if Xn is not None:
            return Xn @ self.coef + self.intercept    
        return self.X @ self.coef + self.intercept
    def rss(self, X_t=None, y_t=None):
        if X_t is None or y_t is None:
            raise ValueError("X_t and y_t must be provided")

        y_pred = X_t @ self.coef + self.intercept
        return np.sum((y_t - y_pred)**2)
    def find(self, mx, n, X_val, y_val, y_train):
        
        J=np.logspace(-4, np.log10(mx), n)

        rssm=np.inf
        self.rm=[]
        self.v=[]
        self.best_alpha=None

        for alpha in J:
            self.fit(y_train, alpha)
            b=self.rss(X_val, y_val)
            self.rm.append(b)
            self.v.append(alpha)
            if b<rssm:
                rssm=b
                self.best_alpha=alpha

        return self.best_alpha,rssm


