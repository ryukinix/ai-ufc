#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: AI-UFC 2019.1 T1
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

"""
Q1) Encontre o máximo da função f(x,y) = |xsen(yπ/4) + ysen(xπ/4)| por
meio do algoritmo hill-climbing. As variáveis x e y pertencem ao
intervalo entre 0 e 20. Os vizinhos de determinado estado (x, y) são
(x ± 0,01, y ± 0,01). Por exemplo, os vizinhos do estado (1, 1) são
(1,01, 1,01), (0,99, 0,99), (0,99, 1,01) e (1,01, 0,99).
"""

import numpy as np
from matplotlib import pyplot
from math import sin as sen
from math import pi
from random import randint


def f(node):
    "A função objetiva que será optimizada."
    x, y = node
    return abs(x * sen (y * pi / 4) + y * sen(x * pi / 4))

def neighbors(node, dt=0.01):
    "Calcula os vizinhos do nó node com passo dt."
    x, y = node
    return [
        (x + dt, y + dt),
        (x + dt, y - dt),
        (x - dt, y + dt),
        (x - dt, y - dt)
    ]


def hill_climbing(f, interval=(0, 20), max_steps=1000, digits=2):
    """Tenta otimizar localmente f no intervalo interval, com o máximo
    de passos de max_steps.
    """

    # [-1, 1] random normal variable to avoid integer values
    k = round(np.random.randn(), ndigits=digits)
    node = randint(*interval) + k, randint(*interval) + k
    print("== ALGORITHM: Hill climbing")
    print(":interval ", interval)

    # Valores iniciais
    start_node = node
    start_value = f(node)
    a, b = interval
    i = 0
    best_result = (node, f(node))
    results = [best_result]

    # Loop de iteração e busca
    while max_steps >= 0:
        nodes = neighbors(node)
        next_results = list(map(lambda v: (v, f(v)),  nodes))
        valid_results = []
        # Restrição dos valores válidos a partir do intervalo
        for (x,y), f_xy in next_results:
            if a <= x <= b  and a <= y <= b:
                valid_results.append(((x, y), f_xy))

        # Ordenar pelos melhores resultados em relação a f(x, y)
        best_results = sorted(valid_results, key=lambda x: x[1], reverse=True)

        # Se não há nenhum resultado melhor que os anteriores, parar o algoritmo
        if not best_results or best_results[0][1] < best_result[1]:
            break

        # Atualizar loop de iteração
        best_result = best_results[0]
        node = best_result[0]
        results.append(best_result)
        max_steps -= 1
        i += 1

    print(":iterations",  i)
    n = len(results)
    x = np.array(range(n))
    y = np.array([f_xy for node, f_xy in results])

    # Cálculo dos melhores nós e valores encontrados
    best_node, best_value = results.pop()
    best_x = round(best_node[0], ndigits=digits)
    best_y = round(best_node[1], ndigits=digits)
    best_value = round(best_value, ndigits=digits)
    start_value = round(start_value, ndigits=digits)
    best_node = (best_x, best_y)

    # Impressão do Report no terminal e a geração de um gráfico do hill climbing.
    print("(x, y) -> f(x, y)")
    print("START NODE: {} -> {} ".format(start_node, start_value))
    print("BEST NODE: {} -> {}".format(best_node, best_value))
    pyplot.plot(x, y, label="f(x,y) = |xsen(yπ/4) + ysen(xπ/4)|")
    label = "Start: {} → {} \n Best: {} → {}  \n".format(start_node, start_value, best_node, best_value)
    plt.plot([], [], '*', label=label)
    ax = plt.gca()
    ax.set_title("Hill Climbing")
    ax.set_xlabel("nodes(t)")
    ax.set_ylabel("objective function")
    ax.legend()
    pyplot.show()


def main():
    print(__doc__)
    hill_climbing(f)
    pass

main()
