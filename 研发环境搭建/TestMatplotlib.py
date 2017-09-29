from matplotlib.pyplot import *

n = 1024
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)
scatter(X, Y, s=75, c=T, alpha=0.5)
xlim(-1.5, 1.5)
ylim(-1.5, 1.5)
show()
