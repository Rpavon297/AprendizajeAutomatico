import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures
from scipy.io import loadmat


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costvector(theta, X, y, l):
    hipTX = sigmoid(np.dot(X, theta))

    return - ((np.dot(y, np.log(hipTX).T) + np.dot(np.log(1 - hipTX).T, (1 - y))) / len(y)) + l / (2 * len(y)) * np.sum(
        np.square(theta[1:]))

def gradiante(theta, X, y, l):
    hipTX = sigmoid(np.dot(X, theta))
    thetaSinCero = np.insert(theta[1:], 0, 0)

    return (np.dot(X.T, (hipTX -y)) / len(y)) + (l * thetaSinCero) / len(y)

""""""""""""


def computeCost(X, y, theta):
    """NAC"""
    m = len(y)
    diff = np.matmul(X, theta) - y
    J = 1 / (2 * m) * np.matmul(diff, diff)
    return J


def regLinealFuncCoste(X, y, theta, l):
    """NAC"""
    m = len(y)  # number of training examples

    J = computeCost(X, y, theta)
    J += l / (2 * m) * np.matmul(theta[1:], theta[1:])
    grad = 1 / m * np.matmul(X.transpose(), np.matmul(X, theta) - y)
    grad[1:] += l / m * theta[1:]
    return J, grad


def min_lineal_reg(X, y, l):
    initial_theta = np.zeros(X.shape[1])

    costFunction = lambda t: regLinealFuncCoste(X, y, t, l)[0]
    gradFunction = lambda t: regLinealFuncCoste(X, y, t, l)[1]

    # metodo TNC
    options = {'maxiter': 200, 'disp': True}
    res = opt.minimize(costFunction, initial_theta, method='TNC', jac=gradFunction, options=options)
    theta = res.x
    return theta

def curvaAprendizaje(X, y, Xval, yval, l):
    m = X.shape[0]
    errro_entren = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(1, m + 1):
        theta = min_lineal_reg(X[:i, ], y[:i, ], l)
        errro_entren[i - 1] = 1.0 / (2 * i) * np.sum(np.square(X[:i, ].dot(theta) - y[:i, ]))
        error_val[i - 1] = 1.0 / (2 * Xval.shape[0]) * np.sum(np.square(Xval.dot(theta) - yval))

    return errro_entren, error_val


def polyCaracteristicas(X, p):
    X_poly = X

    # Iterate over the polynomial power.
    for i in range(1, p):
        # Add the i-th power column in X.
        X_poly = np.column_stack((X_poly, np.power(X, i + 1)))

    return X_poly

def normalizar_caract(X):
    """ recibe una matriz de dimensión m x p y
    devuelve: otra matriz de la misma dimensión
    normalizada columna por columna a media 0 y
    desviación estándar 1 """

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def plotEntrenado(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)

    # valore de X nuevos de 1 hastq p
    x = np.arange(min_x - 15, max_x + 25, 0.05)
    x = np.reshape(x, (len(x), 1))

    # Map X and normalize
    X_poly = polyCaracteristicas(x, p)
    X_poly = (X_poly - mu) / sigma
    X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))

    plt.plot(x, np.dot(X_poly, theta), '-', lw=1)

def graficaEntre(X, y, X_poly, l):
    plt.figure()
    theta = min_lineal_reg(X_poly, y, l)

    plt.plot(X, y, 'rx', ms=8)

    plotEntrenado(np.min(X), np.max(X), mu, sigma, theta, p)

    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression Fit ($\lambda$ = %d)' % l)
    plt.ylim([-9, 60])
    plt.xlim([-60, 50])


def plotCurve(X_poly, y, X_poly_val, yval, l):
    plt.figure()

    error_entren, error_val = curvaAprendizaje(X_poly, y, X_poly_val, yval, l)

    plt.plot(np.arange(1, 1 + m), error_entren, np.arange(1, 1 + m), error_val)

    plt.title('Learning curve for linear regression ($\lambda$ = %d)' % l)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend(['Train', 'Cross Validation'])

    print('Regresión polinomial (lambda = %f)\n\n' % l);
    print('# Ejemplos de entrenamiento\tError entrenado\tError validacion\n');
    for i in range(0, m):
        print('  \t%d\t\t%f\t%f\n' % (i + 1, error_entren[i], error_val[i]))

def validacionCurvaLambda (X, y, Xval, yval):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train = []
    error_val = []

    for lambda_ in lambda_vec:
        theta = min_lineal_reg(X, y, lambda_)
        error_train.append(regLinealFuncCoste(X, y, theta, 0)[0])
        error_val.append(regLinealFuncCoste(Xval, yval, theta, 0)[0])

    return lambda_vec, error_train, error_val

if __name__ == "__main__":
    data = loadmat('ex5data1.mat')

    y = data['y'].ravel()
    X = data['X']
    Xtest, ytest = data['Xtest'], data['ytest'].ravel()
    Xval, yval = data['Xval'], data['yval'].ravel()

    # m = numero de ejemplos de entrenamiento
    m = X.shape[0]

    theta = np.array([1, 1])
    J, grad = regLinealFuncCoste(np.hstack((np.ones((X.shape[0], 1)), X)), y.ravel(), theta, 1)
    print("Coste de : ", J)
    print("Gradiente : ", grad)

    X2 = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = min_lineal_reg(X2, y, 0)
    print(theta)
    """
    plt.plot(X, y, 'rx', ms=8)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.plot(X, np.dot(X2, theta), '-', lw=1)
    plt.show()
    """
    #PARTE 2
    error_entren, error_val = curvaAprendizaje(np.hstack((np.ones((m, 1)), X)), y, np.hstack((np.ones((Xval.shape[0], 1)), Xval)), yval, 0)

    """
    plt.plot(np.arange(1, m+1), error_entren, np.arange(1, m+1), error_val, lw=2)
    plt.title('Learning curve for linear regression')
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.show()
    """

    print('# Ejemplos de entrenamiento\tError entrenado\tError validacion\n');
    for i in range(0, m):
        print('  \t%d\t\t%f\t%f\n' % (i+1, error_entren[i], error_val[i]))

    #PARTE 3
    p = 8

    # X_poly caracteristicas polinomicas y  normalizar
    X_poly = polyCaracteristicas(X, p)
    print(X_poly.shape)
    X_poly, mu, sigma = normalizar_caract(X_poly)
    X_poly = np.hstack((np.ones((m, 1)), X_poly))

    # X_poly_test normalizado, usando mu y sigma
    X_poly_test = polyCaracteristicas(Xtest, p)
    X_poly_test = (X_poly_test - mu) / sigma
    X_poly_test = np.hstack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))

    # X_poly_val normalizado usando mu y sigma
    X_poly_val = polyCaracteristicas(Xval, p)
    X_poly_val = (X_poly_val - mu) / sigma
    X_poly_val = np.hstack((np.ones((X_poly_val.shape[0], 1)), X_poly_val))

    print("Ejemplo de entrenamiento normalizado :")
    """
    print(X_poly[0, :])

    graficaEntre(X, y, X_poly, 0)
    plotCurve(X_poly, y, X_poly_val, yval, 0)

    graficaEntre(X, y, X_poly, 1)
    plotCurve(X_poly, y, X_poly_val, yval, 1)

    graficaEntre(X, y, X_poly, 100)
    plotCurve(X_poly, y, X_poly_val, yval, 100)
    """

    #PARTE 4
    lambda_vec, error_train, error_val = validacionCurvaLambda(X_poly, y, X_poly_val, yval)

    plt.plot(lambda_vec, error_train, '-', lambda_vec, error_val, '-', lw=2)
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('lambda')
    plt.ylabel('Error')
    print('# lambda / Error entrenamiento / VErro Validacion')
    for i in range(len(lambda_vec)):
        print('  {0:<8} {1:<13.8f} {2:<.8f}'.format(lambda_vec[i], error_train[i], error_val[i]))

    l = 3

    theta = min_lineal_reg(X_poly, y, l)
    h_test = np.dot(X_poly_test, theta)
    test_error = np.sum(np.square(h_test - ytest)) / (2 * Xtest.shape[0])
    print(f'test error = {test_error}')
