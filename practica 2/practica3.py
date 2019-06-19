from scipy . io  import loadmat
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costvector2(theta, X, y, l):
    hipTX = sigmoid(np.dot(X, theta))

    return - ((np.dot(y, np.log(hipTX).T) + np.dot(np.log(1 - hipTX).T, (1 - y))) / len(y)) + l / (2 * len(y)) * np.sum(
        np.square(theta[1:]))

def gradiante2(theta, X, y, l):
    hipTX = sigmoid(np.dot(X, theta))
    thetaSinCero = np.insert(theta[1:], 0, 0)

    return (np.dot(X.T, (hipTX -y)) / len(y)) + (l * thetaSinCero) / len(y)

def costvector(theta,X, y, l):
    m = len(y)
    hipTX = sigmoid(np.dot(X, theta))
    thetaSinCero = np.insert(theta[1:], 0, 0)
    c = - ((np.dot(y, np.log(hipTX).T) + np.dot(np.log(1 - hipTX).T, (1 - y))) / m) + l / (2 * m) * np.sum(np.square(theta[1:]))
    g = (np.dot(X.T, (hipTX - y)) + l * thetaSinCero) / m
    return c, g

def oneVsAll(X, y, num_etiquetas, reg):
    all_theta = np.zeros((num_etiquetas, X.shape[1] + 1))
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    initial_theta = np.zeros(X.shape[1])

    for i in range(0, num_etiquetas):
        label = 10 if i == 0 else i
        result = opt.fmin_tnc(costvector, initial_theta, args=(X, (y == label).astype(int), reg))
        all_theta[i, :] = result[0]

    return all_theta

def prediccion(theta, X):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    p = np.argmax(X.dot(theta.T), axis=1)

    p[p == 0] = 10

    return p

def red(X, theta1, theta2):
    capa1 = sigmoid(np.dot(X, theta1.T))
    capa1 = np.hstack((np.ones((X.shape[0], 1)), capa1))

    capa2 = sigmoid(np.dot(capa1, theta2.T))
    salida = np.argmax(capa2, axis=1)
    #Los indices empiezan en 1, no en 0
    salida += 1

    return salida

def parte1(X, Y, data):

    print(data)
    print(X.shape)
    print(y.flatten())

    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()
    print("Calculando parametros:------")
    theta = oneVsAll(X, y, 10, 0.1)
    print("Realizando prediccion:------")
    p = prediccion(theta, X)
    print('Acieto de:', np.mean(p == y) * 100)

if __name__ == "__main__":
    data = loadmat(os.path.abspath("ex3data1.mat"))
    y = data['y'].ravel()
    X = data['X']


    X = np.hstack((np.ones((X.shape[0], 1)), X))

    weights = loadmat('ex3weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']

    pred = red(X, theta1, theta2)
    print('Acieto de:', np.mean(pred == y) * 100)










