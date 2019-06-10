from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize as opt

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np

import csv
import math
import os

import os

X1 = np.array([])
X2 = np.array([])
Y = np.array([])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(params, X):
    hipot = np.array([])

    for i in range(len(X)):
        paramet = np.array([])
        for j in range(len(X[i])):
            nelem = X[i][j] * params[j]
            paramet = np.append(paramet, nelem)
        hipot = np.append(hipot, paramet)

    return sigmoid(hipot)

def costvector(theta,X, y):
    hipTX = sigmoid(np.dot(X, theta))

    return - (np.dot(y, np.log(hipTX).T) + np.dot(np.log(1 - hipTX).T, (1 - y))) / len(y)


def gradiante(theta,X, y):
    hipTX = sigmoid(np.dot(X, theta))

    return np.dot(X.T, (hipTX -y)) / len(y)

def mostrar(X, Y):
    pos = np.where(Y == 1)
    posn = np.where(Y == 0)

    plt.scatter(X[pos, 0], X[pos, 1], marker='.', c='g')
    plt.scatter(X[posn, 0], X[posn, 1], marker='x', c='r')


def mostrarSol(X, Y, theta):
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    mostrar(X, Y)

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

    plt.show()


file = open("ex2data1.csv", "r")

datos = file.read().replace("\n",",").split(",")
file.close()

del datos[-1]

i = 0

while i < len(datos):
    X1 = np.append(X1, float(datos[i]))
    X2 = np.append(X2, float(datos[i+1]))
    Y = np.append(Y, float(datos[i+2]))

    i = i + 3

Xp = np.array([X1,X2])
Xp = np.transpose(Xp)
X = np.hstack((np.ones((Xp.shape[0],1)), Xp))
params = np.zeros(X.shape[1])

res = sigmoid(X)
cost = costvector(params,X,Y)
grad = gradiante(params,X,Y)

print(X.shape, Y.shape, params.shape)
print(cost)
print(grad)

result = opt.fmin_tnc(costvector, params, gradiante, args=(X,Y))
params_opt = result[0]

coste_opt = costvector(params_opt, X, Y)
print(coste_opt)

mostrarSol(Xp,Y,params_opt)