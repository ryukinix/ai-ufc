#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional UFC 2019.1 - Redes Neurais
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#


"""
Módulo com algoritmos de teste e métricas de avaliação.

Para classificação:
+ accuracy

Para regressão:
+ r2

Algoritmos de separação de treinamento/teste:
+ hold_out
+ kfold
+ leave_one_out

Ambos algoritmos de recorte devolvem partições de N coleções de X e y
na forma (X, y).
"""

import numpy as np
from processing import concat

def accuracy(y_test, y_pred):
    """Calcula métrica de acurácia para classificação."""
    n = len(y_test)
    corrects = sum([bool(y1 == y2) for y1, y2 in zip(y_test, y_pred)])
    return corrects/n

def r2(y_test, y_pred):
    """Computa o coeficiente de ajuste de curva r² para regressão."""
    y_mean = np.mean(y_test)
    n = len(y_test)
    SQE = sum((y_test - y_pred) ** 2)
    Syy = sum((y_test - y_mean) ** 2)
    r = SQE / Syy
    r2 = 1 - r

    return r2

def hold_out(X, y, test_size=0.30):
    """Esquema de particionamento de dados train/test split.

    Particiona X,y de forma ordenada após embaralhamento baseado no ponto
    de corte `test_size`.
    """
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
