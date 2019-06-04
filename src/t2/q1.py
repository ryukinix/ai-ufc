#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: ELM @ Aerogerador - Regressão
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

def main():
    X, y = load.aerogerador()
    q = 3
    X_train, X_test, y_train, y_test = testing.hold_out(X, y, test_size=0.50)
    W, M = elm.train(X_train, y_train, q=q)
    y_pred = elm.predict(X_test, W, M)

    r2 = testing.r2(y_test.flatten(), y_pred.flatten())
    print("Q: ", q)
    print("R2: ", r2)


if __name__ == '__main__':
    main()
