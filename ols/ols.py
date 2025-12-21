import numpy as np

class ols:
    def __init__(self,X):
        """X defines all the model"""
        self.X=X
        self.XtX = X.T @ X
        self.XtX_inv = np.linalg.inv(self.XtX)
    def fit(self,y):
        self.y=y
        self.beta_hat= self.XtX_inv @ self.X.T @ y
        return self.beta_hat
    def predict(self,X_n=None):
        if X_n is not None:
            return X_n @ self.beta_hat
        return self.X @ self.beta_hat

    def residuals (self):
        return self.y-self.predict()
    
    def condi(self):
        return np.linalg.cond(self.XtX)
    
    def R2(self):
        y_hat = self.predict()
        y_bar = self.y.mean()

        sse = np.sum((self.y - y_hat)**2)
        sst = np.sum((self.y - y_bar)**2)

        return 1 - sse / sst
    
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

data = load_diabetes()

X_raw = data.data        
y = data.target          
n = X_raw.shape[0]
X = np.column_stack((np.ones(n), X_raw))  # intercept + features
model = ols(X)
beta_hat = model.fit(y)
y_hat = model.predict(X)
residuals = model.residuals()
print("Condition number :", model.condi())
print("R2 :", model.R2())
plt.figure(figsize=(5,5))
plt.scatter(y, y_hat, alpha=0.6)
m = min(y.min(), y_hat.min())
M = max(y.max(), y_hat.max())
plt.plot([m, M], [m, M], 'r--')
plt.show()