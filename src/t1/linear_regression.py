#!/usr/bin/env python
# coding: utf-8

"""Regressão linear

Notas
-----
Dataset: uma função linear com um ruído gaussiano de media 0 e variancia não-unitária.

N = σ * randn(n) + μ
y = m * x + N
ŷ = beta_0 + beta_1 * x

Legenda:
+ N uma variável aleatória normal.
  + σ é a variância
  + μ é a média
  + n é a quantidade de amostras
+ y é a função a ser estimada
  + m é a inclinação da curva
  + x é a variável independente (entrada)
+ ŷ é a função estimadora
  + beta_0 é o coeficiente que intercepta o eixo y
  + beta_1 é o coeficiente que controla a inclinação de y
+ x é a entrada,

beta_0 e beta_1 os coeficientes usados no estimador


Apêndice
--------
Instale: python e pip
Dependências: pip install numpy matplotlib

Author: Manoel Vilela
"""

import numpy as np
from matplotlib import pyplot as plt
from math import sqrt

def dataset():
    with open('aerogerador.dat') as f:
        data = []
        for line in f.readlines():
            x, y = line.split()
            data.append([float(x), float(y)])
        array = np.array(data)
        return array[:, 0], array[:, 1]


x, y = dataset()
n = len(x)
y_mean = np.mean(y)
x_mean = np.mean(x)
beta_1 = (sum(x*y) - y_mean*sum(x)) / (np.sum(x * x) - x_mean * sum(x))
beta_0 = y_mean - beta_1 * x_mean

y_pred = beta_0 + beta_1 * x

syy = sum(y - np.mean(y))
r2 = 1 - (sum((y - y_pred) ** 2) / sum((y - y_mean) ** 2))
rmse = sqrt(sum((y - y_pred) ** 2)/n)
print(__doc__)
print("RESULTS")
print("-------")
print("RMSE: ", rmse)
print("R2: ", r2)
print("B0: ", beta_0)
print("B1: ", beta_1)
print(f"EQ: Ŷ(x) = {beta_0:.2f}  + {beta_1:.2f}*x")
print(f"")
plt.close('all')
plt.scatter(x, y, color='r')
plt.plot(x, y_pred)
ax = plt.gca()
ax.set_title("Regressão Linear")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show(block=False)
hack
