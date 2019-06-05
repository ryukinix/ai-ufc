#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: <project>
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#


"""
Classifique o conjunto de dados disponível no arquivo iris_log.dat
usando: Perceptron, MLP, e ELM. Utilize as estratégias de validação:
hold-out (70% treino, 30% teste), 10-fold e leave-one-out.
"""

import load
import processing
import testing
import elm
import perceptron
import numpy as np

def train_test_perceptron(X, y):
    pass

def evaluate_elm(X_train, X_test, y_train, y_test, q=10):
    # train
    W, M = elm.train(X_train, y_train, q=q, activation=processing.sigmoid)

    # teste
    y_pred_class = elm.predict(X_test, W, M, activation=processing.sigmoid)
    y_test = processing.encode_label(y_test)
    y_pred = processing.encode_label(y_pred_class)
    acc = round(testing.accuracy(y_test, y_pred), ndigits=2)
    return acc


def train_test_elm(X, y, q=40):
    """Treina, testa e calcúla acurácia do ELM para q.

    Retorna uma vetor com 3 elementos das acurácias calculadas por:

    + hold-out, proporção de 30% de teste
    + k-fold, k=10
    + leave-one-out
    """
    accs = []
    # ==> HOLD OUT
    X_train, X_test, y_train, y_test = testing.hold_out(X, y, test_size=0.3)
    acc_hold_out = evaluate_elm(X_train, X_test, y_train, y_test, q=q)
    accs.append(acc_hold_out)

    # ==> K-FOLD, k=10
    accs_kfold = []
    folds = testing.kfold(X, y, k=10)
    for X_train, X_test, y_train, y_test in folds:
        acc = evaluate_elm(X_train, X_test, y_train, y_test, q=q)
        accs_kfold.append(acc)
    acc_kfold = np.mean(np.array(accs_kfold))
    accs.append(acc_kfold)


    # ==> LEAVE ONE OUT
    accs_leave = []
    folds = testing.leave_one_out(X, y)
    for X_train, X_test, y_train, y_test in folds:
        acc = evaluate_elm(X_train, X_test, y_train, y_test, q=q)
        accs_leave.append(acc)
    acc_leave = np.mean(np.array(accs_leave))
    accs.append(acc_leave)

    return accs

def train_test_mlp(X, y):
    pass

def main():
    np.random.seed(32)
    X, y = load.iris()
    q_range = (3, 41, 5)
    print(":: Q RANGE: ", q_range)

    print(":: Evaluating ELM ")
    print(":: Q | hold-out | 10-fold | leave-one-out")
    accs_elm = []
    for q in range(*q_range):
        acc_elm = train_test_elm(X, y, q=q)
        accs_elm.append([q] + acc_elm)
    accs_elm = np.array(accs_elm)
    print(np.around(accs_elm, decimals=2))


if __name__ == '__main__':
    main()
