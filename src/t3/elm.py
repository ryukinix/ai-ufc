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
import load
import processing
import testing


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
    X = processing.add_bias(X)
    # Calcular saída da camada oculta
    Z = W @ X.T
    if activation is not None:
        Z = activation(Z)
    Z = processing.add_bias(Z, axis=0)
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
    X = processing.add_bias(X)
    Z = W @ X.T
    if activation is not None:
        Z = activation(Z)
    Z = processing.add_bias(Z, axis=0)
    Y = M @ Z

    return Y.T

def eval_classification(X, y, q=20):
    X_train, X_test, y_train, y_test = testing.hold_out(X, y)
    W, M = train(X_train, y_train, q=q, activation=processing.sigmoid)
    D_teste = predict(X_test, W, M, activation=processing.sigmoid)
    y_test = processing.encode_label(y_test)
    y_pred = processing.encode_label(D_teste)
    acc = round(testing.accuracy(y_test, y_pred), ndigits=2)
    return acc


def main():
    np.random.seed(processing.SEED)
    X, y = load.iris()
    acc = eval_classification(X, y)
    print("-- Extreme Learning Machine")
    print("X: ", X.shape)
    print("y: ", y.shape)
    print("ACC: ", acc )


if __name__ == '__main__':
    main()
