#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional - ANN & GA
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#


import elm
import processing
import load
import testing
import numpy as np
from matplotlib import pyplot as plt

# Truncate results of sigmoid to -1 and 1
step = np.vectorize(lambda x: 1 if x >= 0 else -1)
colorize = np.vectorize(lambda x: 'r' if x == 1 else 'b')

def main():

    # Train / test ELM
    print("\n-- Extreme Learning Machine")
    X, y = load.twomoons()
    q = 10
    (X_train, X_test, y_train, y_test) = testing.hold_out(X, y)
    W, M = elm.train(X_train, y_train, q=10, activation=processing.sigmoid)
    y_pred = elm.predict(X_test, W, M, activation=processing.sigmoid)
    y_pred = step(y_pred)
    acc = round(testing.accuracy(y_test, y_pred), ndigits=2)

    acc_text = "ACC={:.2f}%".format(acc * 100)
    print("W.shape:", W.shape)
    print("M.shape: ", M.shape)
    print(acc_text)

    ## SURFACE DECISON
    x1, x2 = X[:, 0], X[:, 1]
    x_min = min(min(x1), min(x2))
    x_max = max(max(x1), max(x2))
    n = 200
    linspace = np.linspace(x_min, x_max, n)

    points = []
    for xi in linspace:
        for xj in linspace:
            points.append((xi, xj))

    X_decision = np.array(points)
    y_elm = elm.predict(X_decision, W, M, activation=processing.sigmoid)
    x1_curve = []
    x2_curve = []
    for yi, p in zip(y_elm, points):
        if abs(yi) < 0.05:
            x1_curve.append(p[0])
            x2_curve.append(p[1])

    ax = plt.gca()
    ax.set_title("Two Moons: ELM / {}".format(acc_text))
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.scatter(x1_curve, x2_curve, c='k', s=1)
    plt.scatter(x1, x2, c=colorize(y))
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
