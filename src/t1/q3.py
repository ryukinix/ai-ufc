#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: AI-UFC 2019.1 T1
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

"""Regressão Múltipla

Notas
-----
Dataset: uma função linear com um ruído gaussiano de media 0 e
variancia não-unitária.

Beta = (X'X)-¹X'*y
ŷ(X) = X * beta

Legenda:
+ n é a quantidade de amostras
+ y é a função a ser estimada
  + m é a inclinação da curva
  + x é a variável independente (entrada)
+ ŷ é a função estimadora
  + X é a matriz de variáveis com n x (k + 1)
  + beta é um vetor de coeficientes
  + k é o número de variáveis independentes


Apêndice
--------
Instale: python e pip
Dependências: pip install numpy matplotlib

Author: Manoel Vilela

Aumento do grau de Polinômios

k(1) = 0.93
k(2) = 0.94
k(3) = 0.97
k(3) = 0.969
k(4) = 0.9737242
k(5) = 0.9737256


Dataset de aula
----------------

x = velocidade do vento em aquiraz
y = potencia de energia gerada
"""

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from math import sqrt


def dataset():
    "Carrega o dataset e devolve dois arrays: x e y."
    with open('aerogerador.dat') as f:
        data = []
        for line in f.readlines():
            x, y = line.split()
            data.append([float(x), float(y)])
        array = np.array(data)
        return array[:, 0], array[:, 1]


def regression(x, y, k=1):
    "Gera uma regressão de y em relação a x de grau k, retorna vetor beta e ŷ."
    X = np.array([x ** i for i in range(k+1)]).T
    beta = (inv(X.T @ X) @ X.T) @ y
    y_pred = X @ beta
    return beta, y_pred


def regression_report(x, y, k):
    """Gera um relatório de uma regressão de grau k, dado x e y."""
    y_mean = np.mean(y)
    n = len(x)
    beta, y_pred = regression(x, y, k)
    SQE = sum((y - y_pred) ** 2)
    Syy = sum((y - y_mean) ** 2)
    r = SQE / Syy
    r2 = 1 - r
    p = k + 1
    aj = (n - 1) / (n - p)
    r2aj = 1 - r * aj

    rmse = sqrt(sum((y - y_pred) ** 2)/n)
    print()
    print("RESULTS REGRESSION k={}".format(k))
    print("-----------------------")
    print("RMSE:\t", round(rmse, ndigits=5))
    print("  R2:\t", round(r2, ndigits=5))
    print("R2aj:\t", round(r2aj, ndigits=5))
    print("   B:\t", np.round(beta, decimals=3))
    label = 'k={} /  r2aj = {}'.format(k, round(r2aj, ndigits=5))
    plt.plot(x, y_pred, label=label)
    return beta


def main():
    """Função principal."""
    print(__doc__)
    plt.close("all")
    x, y = dataset()
    regression_report(x, y, 2)
    regression_report(x, y, 3)
    regression_report(x, y, 4)
    regression_report(x, y, 5)
    plt.scatter(x, y, color='black', s=2)
    ax = plt.gca()
    ax.set_title("Regressão")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.show(block=False)
    plt.savefig("q3-regression-4-5.png", figsize=(10, 8))

main()
