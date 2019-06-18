#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional - ANN & GA
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#


"""G E N E T I C  --  A L G O R I T H M S
O indíviduo possui 20 bits tal que 10 bits para x e 10 bits para y.
A função objetiva f deve ser maximizada tal que (x, y) pertence a subfaixa [0,
20] do conjunto R².
"""

import math
import numpy as np

N_BITS = 20
POPULATION_SIZE = 100
CONST_SCALE = 20/((2 ** (N_BITS//2)) - 1)

def f(x, y):
    return abs(x * math.sin(y * ((math.pi)/4)) + y * math.sin(x * ((math.pi)/4)))

def random_population(size=POPULATION_SIZE):
    return np.random.random_integers(0, 1, (size, N_BITS)).astype('float')

def bits_to_int(bits):
    n = len(bits)
    integer = 0
    for i, b in enumerate(bits):
        if b:
            integer += 2 ** (n - i - 1)

    return integer

def decode(bits):
    """Decodifica o indivíduo i em (x, y)"""
    x = bits_to_int(bits[N_BITS//2:]) * CONST_SCALE
    y = bits_to_int(bits[:-N_BITS//2]) * CONST_SCALE
    return (x, y)

def evaluation(population):
    """Avalia a pontução de cada indíviduo da população"""
    return [f(*decode(individual)) for individual in population]


def crossover(population):
    """Cruzamento entre os indíviduos"""
    # TODO: Implement a crossover method of one-point cut
    return population


def mutate(population, prob=0.005):
    """Mutação de indíviduos"""
    # TODO: try to swap the state of one bit
    return population


def selection(population_augmented):
    """Roleta RUSSA. Quem é mais apto tem mais chance de sobreviver."""
    # TODO: Implement selection
    return population_augmented

def evolution_step(population):
    # EVOLUTION
    points = evaluation(population)
    population_augmented = np.insert(population, N_BITS, points, axis=1)
    best_individuals = selection(population_augmented)
    new_population = crossover(best_individuals)
    return new_population

def population_report(population):
    points = evaluation(population)
    best_individual = population[0]
    best_point = points[0]
    for ind, point in zip(population, points):
        if point > best_point:
            best_individual = ind
            best_points = point
    print("MEAN: ", np.mean(points))
    print("STD: ", np.std(points))
    print("MAX: ", np.max(points))
    print("MIN: ", np.min(points))
    print("BEST_SOLUTION: ", decode(best_individual))



def main():
    print("---- G E N E T I C -- A L G O R I T H M S ---")
    population = random_population()
    max_iterations = 10
    i = 0
    while i <= max_iterations:
        print("---- G E N E R A T I O N = {} ---".format(i))
        population_report(population)
        population = evolution_step(population)
        i += 1


if __name__ == '__main__':
    main()
