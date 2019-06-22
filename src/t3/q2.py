#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional UFC 2019.1 - Redes Neural RBF
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

"""Rede Neural RBF

Um algoritmo baseado em redes neurais e mínimos quadrados.
Os pesos da camada oculta são unitários.
Os neurônios da camada oculta são especiais: RBF -> Radial Base Function.
Pontos aleatórios do treinamento são assumidos como centróides da camada oculta.
"""


import numpy as np
from matplotlib import pyplot as plt
import load
import processing
import testing

Q = 10
TEST_SIZE = 0.2

def phi(X, T):
    """Neurônio RBF da camada oculta"""
    n, _ = X.shape
    q, p = T.shape
    matrix = np.zeros(shape=(n, q))
    for i in range(n):
        x = X[i]
        for j in range(q):
            t = T[j]
            matrix[i][j] = np.exp(np.linalg.norm(x - t))

    return matrix

def train(X, y, q=Q, activation=None):
    """Algoritmo de treinamento para Rede Neural RBF

    Parâmetros
    ---------
    X: Vetor de características
    y: Vetor de rótulos
    q: número de neurônios ocultos

    Return
    ------
    T: Centróides da camada oculta com tamanho q
    G: Matriz de pesos da camada de saída.


    """
    # rótulos
    n, p = X.shape
    index = np.arange(0, n - 1)
    D = y


    # Adicionar bias
    X = processing.add_bias(X)

    # training
    # Centróides aleatórios da camada oculta
    T = X[np.random.choice(index, q)]

    # Calcular saída da camada oculta
    PHI = phi(X, T)
    if activation is not None:
        PHI = activation(PHI)
    PHI = processing.add_bias(PHI, axis=1)
    # Calcular pesos G para camada de saída (aprendizado)
    # Utiliza-se mínimos quadrados
    # FIXME: G = D @ (PHI.T @ (np.linalg.inv(PHI @ PHI.T)))
    # Singular Matrix! Usando mínimos quadrados optimizados do Numpy
    G, *_ = np.linalg.lstsq(PHI, D)

    return T, G.T


def predict(X, T, G, activation=None):
    """Algoritmo de predição para ELM (Extreme Learning Machine)

    Parâmetros
    ----------
    T: Centróides da camada oculta
    G: Vetor de pesos da camada de saída para

    Return
    ------
    np.array

    """
    X = processing.add_bias(X)
    PHI = phi(X, T)
    if activation is not None:
        PHI = activation(PHI)
    PHI = processing.add_bias(PHI, axis=1)
    Y = G @ PHI.T

    return Y.T

def eval_classification(X, y, q=Q):
    X_train, X_test, y_train, y_test = testing.hold_out(X, y, test_size=TEST_SIZE)
    T, G = train(X_train, y_train, q=q, activation=processing.sigmoid)
    D_teste = predict(X_test, T, G, activation=processing.sigmoid)
    y_test = processing.encode_label(y_test)
    y_pred = processing.encode_label(D_teste)
    acc = round(testing.accuracy(y_test, y_pred), ndigits=2)
    return acc


def main():
    np.random.seed(processing.SEED)
    X, y = load.iris()
    experiments = 50
    accs = np.empty(experiments)
    for e in range(experiments):
        accs[e] = eval_classification(X, y)

    print("-- Rede Neural RBF")
    print("Experiments: ", experiments)
    print("X: ", X.shape)
    print("y: ", y.shape)
    print("Mean(ACC): ", accs.mean())
    print("Std(ACC): ", accs.std())
    print("Max(ACC): ", accs.max())
    print("Min(ACC): ", accs.min())


if __name__ == '__main__':
    main()
