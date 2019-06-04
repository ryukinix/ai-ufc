#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Extreme Learning Machine Algorithm
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


seed = 42

def activation(x):
    """Função de ativação sigmoid"""
    return 1 / (1 + np.exp(-x))


def add_bias(X, axis=1):
    """Adiciona o vetor de bias em X como uma sequência de -1"""
    # Fix vector classification bias addition
    if axis==1 and (len(X.shape) == 1):
        axis = 0
    return np.insert(X, 0, values=-1, axis=axis)

def train(X, y, q=30):
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
    Z = activation(W @ X.T)
    Z = add_bias(Z, axis=0)
    # Calcular pesos M para camada de saída (aprendizado)
    # Utiliza-se mínimos quadrados
    M = D @ (Z.T @ (np.linalg.inv(Z @ Z.T)))

    return W, M


def predict(X, W, M):
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
    Z = activation(W @ X.T)
    Z = add_bias(Z, axis=0)
    D = M @ Z

    return D.T

def encode_label(X):
    """Transforma representação classes com one-hot-encoding em labels.

    Exemplos de IO
    --------------
    [0, 0, 1] -> 2
    [1, 0, 0] -> 0
    """
    n = len(X)
    labels = np.empty(n)
    for i, x in enumerate(X):
        x = list(x)
        labels[i] = x.index(max(x))
    return labels


def accuracy(y_test, y_pred):
    """Calcula métrica acurácia para classificação"""
    n = len(y_test)
    corrects = sum([bool(y1 == y2) for y1, y2 in zip(y_test, y_pred)])
    return corrects/n


def main():
    X, y = iris()
    W, M = train(X, y)
    D_teste = predict(X, W, M)

    y_test = encode_label(y)
    y_pred = encode_label(D_teste)
    print("ACC: ", round(accuracy(y_test, y_pred), ndigits=2))


if __name__ == '__main__':
    main()
