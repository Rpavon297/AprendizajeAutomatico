import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures
from scipy.io import loadmat

from checkNNGradients import checkNNGradients
from displayData import displayData



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def der_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    m = len(X)

    theta1_s = np.square(theta1[:, 1:])
    theta2_s = np.square(theta2[:, 1:])

    X = np.hstack((np.ones((X.shape[0], 1)), X))

    res1 = np.dot(X, theta1.T)
    sig_res1 = sigmoid(res1)
    sig_res1 = np.hstack((np.ones((X.shape[0], 1)), sig_res1))

    res2 = np.dot(sig_res1, theta2.T)
    sig_res2 = sigmoid(res2)

    Y = np.zeros((m, num_etiquetas))
    for i in range(m):
        Y[i, y[i] - 1] = 1

    cost = 0

    for i in range(m):
        cost += -(np.sum(np.dot(Y[i,], np.log(sig_res2[i,])) + np.dot((1 - Y[i,]), np.log(1 - sig_res2[i,]))) / m)
        
    cost += reg / (2 * m) * (np.sum(theta1_s) + np.sum(theta2_s))
    

    #gradiente
    d3 = sig_res2 - Y
    D2 = np.dot(d3.T, sig_res1)

    res1 = np.hstack((np.ones((X.shape[0], 1)), res1))
    d2 = np.dot(d3, theta2) * der_sigmoid(res1)
    d2 = d2[:, 1:]
    D1 = np.dot(d2.T, X)

    theta1Grad = 1.0 * D1 / m
    theta1Grad[:, 1:] = theta1Grad[:, 1:] + 1.0 * reg / m * theta1[:, 1:]

    theta2Grad = 1.0 * D2 / m
    theta2Grad[:, 1:] = theta2Grad[:, 1:] + 1.0 * reg / m * theta2[:, 1:]

    grad = np.concatenate([theta1Grad.ravel(), theta2Grad.ravel()])

    return cost, grad


def pesosAleatorios(L_in, L_out):
    epsilon_init = 0.12

    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

if __name__ == "__main__":
    data = loadmat('ex4data1.mat')

    y = data['y'].ravel()
    X = data['X']

    m = len(X)

    weights = loadmat('ex4weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']

    params_rn = np.hstack((theta1.flatten(), theta2.flatten()))

    #Selecciona un numero aleatorio de casos
    rand_indices = np.random.choice(m, 100, replace=False)
    sel = X[rand_indices, :]

    print(sel)

    #fig, ax = displayData(sel)
    cost, grad = backprop(params_rn, 400, 25, 10, X, y, 1.0)

    print(cost)

    print("COMPROBACION GRADIANTE:-----")
    initial_theta_1 = pesosAleatorios(400, 25)
    initial_theta_2 = pesosAleatorios(25, 10)
    initial_params_rn = np.hstack((initial_theta_1.ravel(), initial_theta_2.ravel()))

    reg = 3
    checkNNGradients(backprop, reg)
    J, G = backprop(initial_params_rn, 400, 25, 10, X, y, 3)
    print(J, checkNNGradients(backprop, reg))