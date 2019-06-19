import numpy as np

import matplotlib.pyplot as plt
from sklearn.svm import SVC as svc
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import sklearn.svm
import math
from process_email import email2TokenList
from get_vocab_dict import getVocabDict
import warnings
import codecs
warnings.filterwarnings('ignore')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def mostrar(X,y) :
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    plt.figure(figsize=(10, 8))

    plt.scatter(X[pos, 0], X[pos, 1], s=50, c='k', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], s=50, c='y', marker='o')

def mostrarFrontera(X, y, svc):
    # visualizar datos
    mostrar(X, y)

    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    val = np.zeros(X1.shape)
    for ii in range(100):
        this_X = np.vstack((X1[:, ii], X2[:, ii])).T
        val[:, ii] = svc.predict(this_X)

    plt.contour(X1, X2, val, [0, 5], linewidths=1, colors='g')


def mostrarF(X, Y, theta):
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    mostrar(X, Y)

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

def kernelGaussiano(x1, x2, sigma):
    x1 = x1.reshape((x1.size, 1))
    x2 = x2.reshape((x1.size, 1))

    return np.exp(-np.sum((x1 - x2) ** 2) / 2 / sigma / sigma)


def eleccionParametro(X, y, Xval, yval):
    C_vec = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]);
    sigma_vec = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]);

    actual_error = math.inf

    for cur_C in C_vec:
        for cur_sigma in sigma_vec:
            gamma = 1 / 2 / cur_sigma ** 2
            clf = sklearn.svm.SVC(C=cur_C, kernel='rbf', gamma=gamma)
            clf.fit(X, y.ravel())

            prediction = clf.predict(Xval)

            error = np.mean(prediction != yval.flatten());

            if (error < actual_error):
                actual_error = error
                C = cur_C
                sigma = cur_sigma

    print('Mejor combinacion:, C = %f sigma = %f: error = %f' % (C, sigma, actual_error))

    return C, sigma


def crearVectorEmail(raw_email, vocab_dic):
    vector = np.zeros((len(vocab_dic), 1))

    tokenList = email2TokenList(raw_email)
    index_list = [vocab_dic[token] for token in tokenList if token in vocab_dic]

    for idx in index_list:
        vector[idx] = 1
    return vector

if __name__ == "__main__":
    data = scipy.io.loadmat('ex6data1.mat')
    X = data['X']
    y = data['y']
    """
    # C = 1.0
    clf = sklearn.svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, y.ravel())

    mostrarFrontera(X, y, clf)
    plt.title('C=1')
    plt.show()
    # C = 100

    clf = sklearn.svm.SVC(kernel='linear', C=100.0)
    clf.fit(X, y.ravel())

    mostrarFrontera(X, y, clf)
    plt.title('C=100')
    plt.show()


    #kernel gaussiano
    data2 = scipy.io.loadmat('ex6data2.mat')
    X = data2['X']
    y = data2['y']
    mostrar(X,y)
    plt.show()

    C = 1
    sigma = 0.1
    gamma = 1 / 2 / sigma ** 2

    clf = sklearn.svm.SVC(kernel='rbf', C=C, gamma=gamma)
    clf.fit(X, y.ravel())

    mostrarFrontera(X, y, clf)
    plt.show()
    #elegir parametros
    data3 = scipy.io.loadmat('ex6data3.mat')
    X = data3['X']
    y = data3['y']
    Xval = data3['Xval']
    yval = data3['yval']

    C, sigma = eleccionParametro(X, y, Xval, yval)
    gamma = 1 / 2 / sigma ** 2

    clf = sklearn.svm.SVC(C=C, kernel='rbf', gamma=gamma)
    clf.fit(X, y.ravel())

    mostrarFrontera(Xval, yval, clf)
    plt.show()
    """

    #control de spam

    email_contents = codecs.open('spam/0001.txt', 'r', encoding='utf-8', errors='ignore').read()
    tokens = email2TokenList(email_contents)
    vocab_dic = getVocabDict()

    directorio = "spam"
    i = 1
    email_spam = codecs.open('{0}/{1:04d}.txt'.format(directorio, i), 'r', encoding='utf-8', errors='ignore').read()

    directorio = "easy_ham"
    i = 1
    email_easy_ham = codecs.open('{0}/{1:04d}.txt'.format(directorio, i), 'r', encoding='utf-8', errors='ignore').read()

    directorio = "hard_ham"
    i = 1
    email_hard_ham = codecs.open('{0}/{1:04d}.txt'.format(directorio, i), 'r', encoding='utf-8', errors='ignore').read()

    #Entrenamiento
    num_spam_train = int(len(email_spam) * 0.6)
    num_easyham_train = int(len(email_easy_ham) * 0.6)
    num_hardham_train = int(len(email_hard_ham) * 0.6)

    spam_train = [crearVectorEmail(x, vocab_dic)
                  for x in email_spam[:num_spam_train]]

    easyham_train = [crearVectorEmail(x, vocab_dic)
                     for x in email_easy_ham[:num_easyham_train]]

    hardham_train = [crearVectorEmail(x, vocab_dic)
                     for x in email_hard_ham[:num_hardham_train]]

    Xtrain = np.concatenate(hardham_train + easyham_train + spam_train, axis=1).T
    ytrain = np.concatenate((np.zeros((num_hardham_train + num_easyham_train, 1)),
                             np.ones((num_spam_train, 1))
                             ), axis=0)

    #Validacion
    num_spam_val = int(len(email_spam) * 0.2)
    num_easyham_val = int(len(email_easy_ham) * 0.2)
    num_hardham_val = int(len(email_hard_ham) * 0.2)

    spam_val = [crearVectorEmail(x, vocab_dic)
                for x in email_spam[num_spam_train:num_spam_train + num_spam_val]]

    easyham_val = [crearVectorEmail(x, vocab_dic)
                   for x in email_easy_ham[num_easyham_train:num_easyham_train + num_easyham_val]]

    hardham_val = [crearVectorEmail(x, vocab_dic)
                   for x in email_hard_ham[num_hardham_train:num_hardham_train + num_hardham_val]]

    Xval = np.concatenate(hardham_val + easyham_val + spam_val, axis=1).T
    yval = np.concatenate((np.zeros((num_hardham_val + num_easyham_val, 1)),
                           np.ones((num_spam_val, 1))
                           ), axis=0)

    #Test
    num_spam_test = len(email_spam) - num_spam_val - num_spam_train
    num_easyham_test = len(email_easy_ham) - num_easyham_val - num_easyham_train
    num_hardham_test = len(email_hard_ham) - num_hardham_val - num_hardham_train

    spam_test = [crearVectorEmail(x, vocab_dic)
                 for x in email_spam[-num_spam_test:]]

    easyham_test = [crearVectorEmail(x, vocab_dic)
                    for x in email_easy_ham[-num_easyham_test:]]

    hardham_test = [crearVectorEmail(x, vocab_dic)
                    for x in email_hard_ham[-num_hardham_test:]]

    Xtest = np.concatenate(hardham_test + easyham_test + spam_test, axis=1).T
    ytest = np.concatenate((np.zeros((num_easyham_test + num_hardham_test, 1)),
                            np.ones((num_spam_test, 1))
                            ), axis=0)


    m_clf = sklearn.svm.SVC(C=0.1, kernel='linear')
    m_clf.fit(Xtrain, ytrain.ravel())

    prediccion = m_clf.predict(Xtest).reshape((ytest.shape[0], 1))
    test = 100. * float(sum(prediccion == ytest)) / ytest.shape[0]
    print('Precicion del conjunto de prueba = %0.2f%%' % test)