#!/usr/bin/env python
# coding: utf-8

"""Regressão Múltipla

Notas
-----
Dataset: uma função linear com um ruído gaussiano de media 0 e variancia não-unitária.

N = σ * randn(n) + μ
y(X) = X * beta + e
ŷ(X) = X * beta

Legenda:
+ N uma variável aleatória normal.
  + σ é a variância
  + μ é a média
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
from matplotlib import pyplot as plt
from math import sqrt

mean = 0
var = 10
m = 4
n = 100
rv = var * np.random.randn(n) + mean
x = np.linspace(0, 100, n)
y = m * x + rv

y_mean = np.mean(y)
x_mean = np.mean(x)
ones = np.ones(n)
X = np.array((ones, x))
beta = (np.linalg.inv(X.T @ X) @ X.T) * y
beta_0 = beta[0]
beta_1 = beta[1]
y_pred = beta_0 + beta_1 * x

syy = sum(y - np.mean(y))
r2 = 1 - (sum((y - y_pred) ** 2) / sum((y - y_mean) ** 2))
# p = k + 1 # k = ordem do polinomio
# aj = (n - p) / (n - 1)
# r2aj = 1 - (sum((y - y_pred) ** 2) )/ sum((y - y_mean) ** 2))

rmse = sqrt(sum((y - y_pred) ** 2)/n)
print(__doc__)
print("RESULTS")
print("-------")
print("RMSE: ", rmse)
print("R2: ", r2)
print("B0: ", beta_0)
print("B1: ", beta_1)
print(f"EQ: Y(x) = N(0, {var})+ {m}*x")
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
plt.savefig("regression.png")
