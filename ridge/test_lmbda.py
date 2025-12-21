import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from ridge import ridge 
from sklearn.linear_model import LinearRegression

data = fetch_california_housing()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = ridge(X_train)

# Recherche de alpha/lambda (en python lambda correspond à une fonction)
mx = 1000      # alpha max
n = 200        # nombre de points testés
best_alpha, best_rss = model.find(
    mx=1000,
    n=200,
    X_val=X_test,
    y_val=y_test,
    y_train=y_train
)

ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred = ols.predict(X_test)
resid = y_test - y_pred

print("RSS OSL  :", np.sum(resid**2))
print("Best alpha :", best_alpha)
print("Ridge RSS   :", best_rss)
plt.plot(model.v,model.rm)
plt.title("RSS en fonction de lambda ")
plt.xlabel("lambda ")
plt.ylabel("RSS (erreur)")
plt.show()