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


def train(X, y, q=10, activation=None):
    """Algoritmo de treinamento para Rede Neural RBF

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


def predict(X, W, G, activation=None):
    """Algoritmo de predição para ELM (Extreme Learning Machine)

    Parâmetros
    ----------
    G: Vetor de pesos da camada de saída para

    Return
    ------
    np.array

    """
    X = processing.add_bias(X)
    Z = W @ X.T
    if activation is not None:
        Z = activation(Z)
    Z = processing.add_bias(Z, axis=0)
    Y = G @ Z

    return Y.T

def eval_classification(X, y, q=10):
    X_train, X_test, y_train, y_test = testing.hold_out(X, y, test_size=0.2)
    W, M = train(X_train, y_train, q=q, activation=processing.sigmoid)
    D_teste = predict(X_test, W, M, activation=processing.sigmoid)
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
    print("Max(ACC): ", accs.max())
    print("Min(ACC): ", accs.min())


if __name__ == '__main__':
    main()
