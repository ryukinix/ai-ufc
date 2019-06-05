#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional UFC 2019.1 - Redes Neurais
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

"""Módulo de processamento auxiliar genérico

+ Operação com matrizes
+ Codificação de classes
+ Funções de ativação
+ Adição de bias

"""

import numpy as np

SEED = 32

def sigmoid(x):
    """Função de ativação sigmoid"""
    return 1 / (1 + np.exp(-x))


def step(x):
    """Função de ativação degrau"""
    return np.vectorize(lambda x: 1 if x >= 0 else 0)(x)


def add_bias(X, axis=1):
    """Adiciona o vetor de bias em X como uma sequência de -1"""
    # If X is a simple vector, add the 1 sequence in axis=0 (horizontal)
    if axis==1 and (len(X.shape) == 1):
        axis = 0
    return np.insert(X, 0, values=-1, axis=axis)


def encode_label(X):
    """Transforma representação classes com one-hot-encoding em labels.

    Exemplos de entrada e saída
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


def column_vector(X):
    n = len(X)
    if len(X.shape) == 1:
        return X.reshape(n, 1)
    return X


def concat(X, y):
    """Concat concatena X e y independente de serem vetor linha ou coluna"""
    n = len(X)
    if len(X.shape) == 1:
        X = X.reshape(n, 1)

    if len(y.shape) == 1:
        y = y.reshape(n, 1)

    return np.concatenate([X, y], axis=1)
