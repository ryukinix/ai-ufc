#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional - ANN & GA
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#


"""
O indíviduo possui 20 bits tal que 10 bits para x e 10 bits para y.
A função objetiva f deve ser maximizada tal que (x, y) pertence a [0,
20] assim como sua saída.
"""

import math

def f(x, y):
    return abs(x * math.sin(y * ((math.pi)/4)) + y * math.sin(x * ((math.pi)/4)))

def decode(i):
    """Decodifica o indivíduo i em (x, y)"""
    pass

def evaluation(population):
    """Avalia a pontução de cada indíviduo da população"""
    pass


def crossover(population):
    """Corte de um ponto"""
    pass


def mutate(population, prob=0.005):
    """Mutação de indíviduos"""
    pass


def selection(population):
    """Roleta RUSSA"""
    pass


def main():
    pass


if __name__ == '__main__':
    main()
