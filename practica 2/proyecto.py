import numpy as np

if __name__ == "__main__":
    file = open("training.csv", "r")
    datos = file.read().replace("\n", ",").split(",")
    file.close()

    sanos = np.zeros([0,5])
    enfermos = np.zeros([0,5])

    i = 6

    while i < len(datos)-1:
        aux = np.array([datos[i + 1], datos[i + 2], datos[i + 3], datos[i + 4], datos[i + 5]])

        if (datos[i] == 'n'):
            sanos = np.vstack([sanos, aux])

        else:
            enfermos = np.vstack([enfermos, aux])

        i = i + 6

    print(sanos)
    print(enfermos)