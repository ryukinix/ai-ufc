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
from sklearn.neural_network import MLPClassifier

def mlp(q):
    """Cria uma instância do classificador Multi-Layer Perceptron com
    q neurônios na camada oculta.

    Como optimizador utilize SGD: Sthocastic Gradient Descent

    Implementação utilizada: sklearn

    Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    """
    return MLPClassifier(solver='sgd',
                         max_iter=3000,
                         hidden_layer_sizes=(q,),
                         random_state=processing.SEED)

def evaluate_mlp(X_train, X_test, y_train, y_test, q):
    clf = mlp(q)
    clf.fit(X_train, processing.encode_label(y_train))
    y_pred = clf.predict(X_test)
    y_test = processing.encode_label(y_test)
    acc = testing.accuracy(y_test, y_pred)
    return acc




def train_test_mlp(X, y, q):
    accs = []

    # HOLD-OUT
    X_train, X_test, y_train, y_test = testing.hold_out(X, y, test_size=0.3)
    acc = evaluate_mlp(X_train, X_test, y_train, y_test, q)
    accs.append(acc)


    # ==> K-FOLD, k=10
    accs_kfold = []
    folds = testing.kfold(X, y, k=10)
    for X_train, X_test, y_train, y_test in folds:
        acc = evaluate_mlp(X_train, X_test, y_train, y_test, q)
        accs_kfold.append(acc)
    acc_kfold = np.mean(np.array(accs_kfold))
    accs.append(acc_kfold)


    # ==> LEAVE ONE OUT
    accs_leave = []
    folds = testing.leave_one_out(X, y)
    for X_train, X_test, y_train, y_test in folds:
        acc = evaluate_mlp(X_train, X_test, y_train, y_test, q)
        accs_leave.append(acc)
    acc_leave = np.mean(np.array(accs_leave))
    accs.append(acc_leave)


    return accs

def evaluate_elm(X_train, X_test, y_train, y_test, q):
    # train
    W, M = elm.train(X_train, y_train, q=q, activation=processing.sigmoid)

    # teste
    y_pred_class = elm.predict(X_test, W, M, activation=processing.sigmoid)
    y_test = processing.encode_label(y_test)
    y_pred = processing.encode_label(y_pred_class)
    acc = round(testing.accuracy(y_test, y_pred), ndigits=2)

    return acc

def evaluate_perceptron(X_train, X_test, y_train, y_test):
    W = perceptron.train(X_train, y_train)
    y_pred = perceptron.predict(X_test, W)
    y_test = processing.encode_label(y_test)
    acc = testing.accuracy(y_test, y_pred)

    return acc

def train_test_perceptron(X, y):
    accs = []

    # HOLD-OUT
    X_train, X_test, y_train, y_test = testing.hold_out(X, y, test_size=0.3)
    acc = evaluate_perceptron(X_train, X_test, y_train, y_test)
    accs.append(acc)


    # ==> K-FOLD, k=10
    accs_kfold = []
    folds = testing.kfold(X, y, k=10)
    for X_train, X_test, y_train, y_test in folds:
        acc = evaluate_perceptron(X_train, X_test, y_train, y_test)
        accs_kfold.append(acc)
    acc_kfold = np.mean(np.array(accs_kfold))
    accs.append(acc_kfold)


    # ==> LEAVE ONE OUT
    accs_leave = []
    folds = testing.leave_one_out(X, y)
    for X_train, X_test, y_train, y_test in folds:
        acc = evaluate_perceptron(X_train, X_test, y_train, y_test)
        accs_leave.append(acc)
    acc_leave = np.mean(np.array(accs_leave))
    accs.append(acc_leave)


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


def run_elm(X, y, q_range):
    print(":: Evaluating ELM ")
    print(":: Q | hold-out | 10-fold | leave-one-out")
    accs_elm = []
    for q in range(*q_range):
        acc_elm = train_test_elm(X, y, q=q)
        accs_elm.append([q] + acc_elm)
    accs_elm = np.array(accs_elm)
    print(np.around(accs_elm, decimals=2))
    return accs_elm

def run_mlp(X, y, q_range):
    print(":: Evaluating MLP")
    print(":: Q | hold-out | 10-fold | leave-one-out")
    accs_mlp = []
    for q in range(*q_range):
        acc_mlp = train_test_mlp(X, y, q=q)
        accs_mlp.append([q] + acc_mlp)
    accs_mlp = np.array(accs_mlp)
    print(np.around(accs_mlp, decimals=2))


def run_perceptron(X, y):
    print(":: Evaluating Perceptron")
    print(":: hold-out | 10-fold | leave-one-out")
    acc_perceptron = train_test_perceptron(X, y)
    acc_perceptron = np.array(accs_perceptron)
    print(np.around(accs_perceptron, decimals=2))



def main():
    print(__doc__)
    np.random.seed(processing.SEED)
    X, y = load.iris()
    q_range = (3, 41, 5)
    print(":: Q RANGE: ", q_range)
    print()
    run_perceptron(X, y)
    # run_mlp(X, y, q_range)
    # run_elm(X, y, q_range)



if __name__ == '__main__':
    main()
