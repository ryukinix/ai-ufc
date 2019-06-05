#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional UFC 2019.1 - Redes Neurais
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

import load
import processing
import testing
import numpy as np

def error(d, y):
    return d - y

def train(X, y, learning_rate=0.01, max_iterations=100):
    X = processing.add_bias(X)
    n, m = X.shape
    _, c = y.shape
    W = np.zeros((m, c))

    epochs = 0
    while epochs <= max_iterations:
        for xi, d in zip(X, y):
            x = processing.column_vector(xi)
            yi = processing.step(W.T @ xi) # activation
            e = error(d, yi)
            deltaW = learning_rate * e * x
            W = W + deltaW
        epochs += 1
    return W

def predict(X, W):
    X = processing.add_bias(X)
    u = W.T @ X.T
    return processing.encode_label(u.T)


def main():
    X, y = load.iris()
    W = train(X, y)
    y_pred = predict(X, W)
    y_test = processing.encode_label(y)
    acc = testing.accuracy(y, y_pred)

    print("ACC: ", acc)

main()

if __name__ == '__main__':
    main()
