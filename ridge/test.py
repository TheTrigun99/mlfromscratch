import matplotlib.pyplot as plt
from ridge import ridge
from sklearn.linear_model import LinearRegression
import numpy as np
X_train = np.array([[0.5], [1.0]])
y_train = np.array([0.5, 1.0])

alpha = 5
X_test = np.array([[0],[2]]) #np.linspace(0, 2, 200).reshape(-1, 1)

np.random.seed(0)
noises = 0.1 * np.random.normal(size=(6,) + X_train.shape)
fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
ax = axes[0]
ols = LinearRegression()

for eps in noises:
    this_X = X_train + eps
    ols.fit(this_X, y_train)
    ax.plot(X_test, ols.predict(X_test), color="gray", alpha=0.6)
    ax.scatter(this_X, y_train, s=12, c="gray", zorder=10)

ols.fit(X_train, y_train)
ax.plot(X_test, ols.predict(X_test), color="blue", linewidth=2)
ax.scatter(X_train, y_train, s=40, c="red", marker="+", zorder=11)

ax.set_title("OLS")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_xlim(0, 2)
ax.set_ylim(0, 1.6)

# ====== Ridge maison ======
ax = axes[1]

for eps in noises:
    this_X = X_train + eps
    rd = ridge(this_X)
    rd.fit(y_train, alpha)
    if rd.coef>0.6:
        ax.plot(X_test, rd.predict(X_test), color="blue", alpha=0.6)
        ax.scatter(this_X, y_train, s=12, c="yellow", zorder=10)
    else:
        ax.plot(X_test, rd.predict(X_test), color="gray", alpha=0.6)

        ax.scatter(this_X, y_train, s=12, c="gray", zorder=10)

po = ridge(X_train)
po.fit(y_train, alpha)
print(po.coef,po.intercept)
ax.plot(X_test, po.predict(X_test), color="green", linewidth=2)
ax.scatter(X_train, y_train, s=40, c="red", marker="+", zorder=11)

ax.set_title(f"Ridge (alpha={alpha})")
ax.set_xlabel("X")
ax.set_xlim(0, 2)

plt.tight_layout()
plt.show()

