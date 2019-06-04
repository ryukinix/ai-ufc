#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional UFC 2019.1 - Redes Neurais
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

"""Extreme Learning Machine

Um algoritmo baseado em redes neurais e mínimos quadrados.
Os pesos são aleatórios com uma distribuição normal.

Os tipos de teste são:

+ Hold-Out (splt train test)
+ K-Fold (cross-validation)
+ Leave-One-Out (cross-validation com k=n)

1. Camada oculta com pesos aleatórios de Z = W * X

q = 10 -> 0.89 acc
q = 20 -> 0.97 acc (sorte do cassete)

"""


import numpy as np
from matplotlib import pyplot as plt
from load import iris
from testing import hold_out, accuracy
from processing import sigmoid, add_bias, encode_label

seed = 42

def train(X, y, q=10, activation=None):
    """Algoritmo de treinamento para ELM (Extreme Learning Machine)

    Parâmetros
    ---------
    X: Vetor de características
    y: Vetor de rótulos
    q: número de neurônios ocultos

    Return
    ------
    W: pesos aleatórios da camada oculta

    """
    # rótulos
    # torna vetor linha em coluna
    n, p = X.shape
    D = y.T

    # training
    # Pesos aleatórios da camada oculta
    W = np.random.randn(q, p+1)
    # Adicionar bias
    X = add_bias(X)
    # Calcular saída da camada oculta
    Z = W @ X.T
    if activation is not None:
        Z = activation(Z)
    Z = add_bias(Z, axis=0)
    # Calcular pesos M para camada de saída (aprendizado)
    # Utiliza-se mínimos quadrados
    M = D @ (Z.T @ (np.linalg.inv(Z @ Z.T)))

    return W, M


def predict(X, W, M, activation=None):
    """Algoritmo de predição para ELM (Extreme Learning Machine)

    Parâmetros
    ----------
    W: Vetor de pesos da camada oculta utilizados no treinamento
    M: Vetor de pesos da camada de saída para

    Return
    ------
    np.array

    """
    X = add_bias(X)
    Z = W @ X.T
    if activation is not None:
        Z = activation(Z)
    Z = add_bias(Z, axis=0)
    D = M @ Z

    return D.T


def main():
    print()
    X, y = iris()
    q = 30
    X_train, X_test, y_train, y_test = hold_out(X, y)
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    W, M = train(X_train, y_train, q=q, activation=sigmoid)
    D_teste = predict(X_test, W, M, activation=sigmoid)

    y_test = encode_label(y_test)
    y_pred = encode_label(D_teste)

    print("ACC: ", round(accuracy(y_test, y_pred), ndigits=2))


if __name__ == '__main__':
    main()
