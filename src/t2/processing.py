# coding: utf-8

import numpy as np

"""Módulo de processamento auxiliar genérico

+ Operação com matrizes
+ Codificação de classes
+ Funções de ativação
+ Adição de bias

"""

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
