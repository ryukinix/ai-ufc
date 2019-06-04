# coding: utf-8
import numpy as np

def accuracy(y_test, y_pred):
    """Calcula métrica acurácia para classificação"""
    n = len(y_test)
    corrects = sum([bool(y1 == y2) for y1, y2 in zip(y_test, y_pred)])
    return corrects/n

def hold_out(X, y, test_size=0.25):
    n, c = y.shape

    dataset = np.concatenate([X, y], axis=1)
    # dataset embaralhado (shuffled)
    np.random.shuffle(dataset)
    X_s, y_s = dataset[:, :-c], dataset[:, -c:]

    test_index = round(test_size * n)
    X_train = X_s[test_index:]
    y_train = y_s[test_index:]
    X_test = X_s[:test_index]
    y_test = y_s[:test_index]

    return X_train, X_test, y_train, y_test
