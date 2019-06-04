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
Determine um modelo de regressão usando rede neural Extreme Learning
Machine (ELM) para o conjunto de dados do aerogerador (variável de
entrada: velocidade do vento, variável de saída: potência
gerada). Avalie a qualidade do modelo pela métrica R 2 (equação 48,
slides sobre Regressão Múltipla) para diferentes quantidades de
neurônios ocultos.
"""

import load
import processing
import testing
import elm
from processing import column_vector as colv
import numpy as np
from matplotlib import pyplot as plt

def train(X, y, q=2):
    W, M = elm.train(X, y, q=q)
    y_pred = elm.predict(X, W, M)
    r2 = testing.r2(y.flatten(), y_pred.flatten())

    return r2, y_pred


def main():
    print(__doc__)
    X, y = load.aerogerador()
    X, y = colv(X), colv(y)
    # Plotting
    # fig 1: models
    plt.figure("Inteligência Computacional 2019.1 UFC")
    for q in range(1, 10, 2):
        r2, y_pred = train(X, y, q=q)
        label = 'q={} r²={}'.format(q, round(r2, ndigits=3))
        plt.plot(X, y_pred.flatten(), label=label)

    plt.scatter(X, y, color='black', s=3)
    ax = plt.gca()
    ax.set_title("Regressão via Extreme Learning Machine")
    ax.set_xlabel("x: velocidade do vento (m/s)")
    ax.set_ylabel("y: potência gerada (KWatt)")
    ax.legend()
    plt.savefig("pics/q1-elm.png", figsize=(10, 8))
    plt.show()




if __name__ == '__main__':
    main()
