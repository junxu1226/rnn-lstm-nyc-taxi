import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt


def gendata():
    obs = np.concatenate((2.6*np.random.randn(300, 2), 1.6*np.random.randn(300, 2), 6 + 1.3*np.random.randn(300, 2), np.array([-5, 5]) + 1.3*np.random.randn(200, 2), np.array([2, 7]) + 1.1*np.random.randn(200, 2)))
    return obs



def gaussian_2d(x, y, x0, y0, sig):
    return 1/(2*np.pi*np.sqrt(sig)) * np.exp(-0.5*((x-x0)**2 + (y-y0)**2) / sig)


def gengmm(nc, n_iter):
    g = mixture.GMM(n_components=nc)  # number of components
    g.init_params = ""  # No initialization
    g.n_iter = n_iter   # iteration of EM method
    return g


def plotGMM(g, n, pt):
    delta = 0.01
    x = np.arange(-10, 10, delta)
    y = np.arange(-6, 12, delta)
    X, Y = np.meshgrid(x, y)

    if pt == 1:
        for i in xrange(n):
            Z1 = gaussian_2d(X, Y, g.means_[i, 0], g.means_[i, 1], g.covars_[i, 0] * g.covars_[i, 1])
            plt.contour(X, Y, Z1, linewidths=0.5)

    #print g.means_
    # plt.plot(g.means_[0][0],g.means_[0][1], '+', markersize=13, mew=3)
    # plt.plot(g.means_[1][0],g.means_[1][1], '+', markersize=13, mew=3)
    # plt.plot(g.means_[2][0],g.means_[2][1], '+', markersize=13, mew=3)
    # plt.plot(g.means_[3][0],g.means_[3][1], '+', markersize=13, mew=3)

    # plot the GMM with mixing parameters (weights)
    i=0
    Z2= g.weights_[i]*gaussian_2d(X, Y, g.means_[i, 0], g.means_[i, 1], g.covars_[i, 0] * g.covars_[i, 1])
    for i in xrange(1,n):
       Z2 = Z2+ g.weights_[i]*gaussian_2d(X, Y, g.means_[i, 0], g.means_[i, 1], g.covars_[i, 0] * g.covars_[i, 1])
    plt.contour(X, Y, Z2)


# obs = gendata()
fig = plt.figure(1)
g = gengmm(10, 100)

# for i in range(10):
mean = np.random.rand(2,10)*10
sigma = np.random.rand(10) + 2
mixing = np.random.rand(10)

g.means_=mean
g.weights_=mixing
g.covariances_ = sigma

delta = 0.1
x = np.arange(-10, 10, delta)
y = np.arange(-10, 10, delta)
X, Y = np.meshgrid(x, y)


i=0
Z2= g.weights_[i]*gaussian_2d(X, Y, g.means_[0, i], g.means_[1, i], g.covariances_[i])
for i in xrange(1,10):
    Z2 = Z2+ g.weights_[i]*gaussian_2d(X, Y, g.means_[0, i], g.means_[1, i], g.covariances_[i])

c =  plt.contour(X, Y, Z2)
plt.clabel(c)

plt.title('Gaussian Mixture Model')
plt.show()
