# coding: utf-8
import numpy as np

def accuracy(y_test, y_pred):
    """Calcula métrica acurácia para classificação"""
    n = len(y_test)
    corrects = sum([bool(y1 == y2) for y1, y2 in zip(y_test, y_pred)])
    return corrects/n

def r2(y_test, y_pred):
    """Cálcula coeficiente de ajuste r² para regressão"""
    y_mean = np.mean(y_test)
    n = len(y_test)
    SQE = sum((y_test - y_pred) ** 2)
    Syy = sum((y_test - y_mean) ** 2)
    r = SQE / Syy
    r2 = 1 - r

    return r2

def concat(X, y):
    n = len(X)
    if len(X.shape) == 1:
        X = X.reshape(n, 1)

    if len(y.shape) == 1:
        y = y.reshape(n, 1)

    return np.concatenate([X, y], axis=1)


def hold_out(X, y, test_size=0.30):
    shape = y.shape
    n = len(y)
    c = shape[1] if len(shape) > 1 else 1
    dataset = concat(X, y)
    # dataset embaralhado (shuffled)
    np.random.shuffle(dataset)
    X_s, y_s = dataset[:, :-c], dataset[:, -c:]

    test_index = round(test_size * n)
    X_train = X_s[test_index:]
    y_train = y_s[test_index:]
    X_test = X_s[:test_index]
    y_test = y_s[:test_index]

    return X_train, X_test, y_train, y_test
