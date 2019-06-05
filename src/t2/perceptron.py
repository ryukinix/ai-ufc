#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional UFC 2019.1 - Redes Neurais
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

import load
import processing
import testing
import numpy as np

def train(X, y, learning_rate=0.01, max_iterations=1000):
    """Função de treinamento da Rede Neural Perceptron.

    Função de ativação no neurônio é a sigmoid.
    """
    X = processing.add_bias(X)
    n, m = X.shape
    _, c = y.shape
    W = np.zeros((m, c))

    epochs = 0
    while epochs <= max_iterations:
        for xi, d in zip(X, y):
            x = processing.column_vector(xi)
            yi = processing.sigmoid(W.T @ xi) # activation
            e = d - yi
            deltaW = learning_rate * e * x
            W = W + deltaW
        epochs += 1
    return W

def predict(X, W):
    """Função de predição baseado na memória W da Rede Neural Perceptron.

    X é a matriz de features e não deve conter o bias, pois é
    adicionado nessa função.
    """
    X = processing.add_bias(X)
    u = W.T @ X.T
    return processing.encode_label(u.T)


def main():
    X, y = load.iris()
    X_train, X_test, y_train, y_test = testing.hold_out(X, y)
    W = train(X_train, y_train)
    y_pred = predict(X_test, W)
    y_test = processing.encode_label(y_test)
    acc = testing.accuracy(y_test, y_pred)

    print("ACC: {:.2f} %".format(acc * 100) )


if __name__ == '__main__':
    main()
