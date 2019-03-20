from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
    ret = np.array([])
    for row in z:
        r = np.array([])
        for elem in row:
            aux = 1.0 / (1.0 + math.exp(float(-elem)))
            r = np.append(r, aux)
        ret = np.append(ret, r)
    return ret


def h(params, X):
    hipot = np.array([])

    for i in range(len(X)):
        paramet = np.array([])
        for j in range(len(X[i])):
            nelem = X[i][j] * params[j]
            paramet = np.append(paramet, nelem)
        hipot = np.append(hipot, paramet)

    return sigmoid(hipot)


def h(params, X):
    paramet = np.dot(X, params)


def costvector(X, Y, params):
    logs = np.array([])
    ilogs = np.array([])

    for row in X:
        s = sigmoid([[np.dot(row, params)]])

        l = math.log(s[0][0])
        il = math.log(1 - s[0][0])

        logs = np.append(logs, l)
        ilogs = np.append(ilogs, il)

    logs = np.transpose(logs)
    ilogs = np.transpose(ilogs)

    ones = np.dot(logs, Y)
    zeros = np.dot(ilogs, np.ones(len(Y)) - Y)
    return -1.0 / (len(X)) * (ones + zeros)


def gradiante(X, Y, params):
    s = sigmoid([[np.dot()]])

    return 1.0 / len(X)


with open("ex2data1.csv") as file:
    csvr = csv.reader(file, delimiter=',')

    for linea in csvr:
        X1 = np.append(X1, float(linea[0]))
        X2 = np.append(X2, float(linea[1]))
        Y = np.append(Y, float(linea[2]))

X = np.array([X1, X2])

pos = np.where(Y == 1)
posn = np.where(Y == 0)
plt.scatter(X[0, pos], X[1, pos], marker='.', c='g')
plt.scatter(X[0, posn], X[1, posn], marker='x', c='r')

X = [[1, 2, 3, 22, 5]]
Y= [10]
Z = [[2,2,3],[2,5,7]]

res = sigmoid(X)
print(res)

plt.title(res)
plt.show()