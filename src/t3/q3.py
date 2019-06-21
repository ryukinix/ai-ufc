#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional - ANN & GA
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#


"""---- G E N E T I C  --  A L G O R I T H M S ----
O indíviduo possui 20 bits tal que 10 bits para x e 10 bits para y.  A
função objetiva f deve ser maximizada tal que (x, y) pertence a
subfaixa [0, 20] do conjunto R².

f(x, y) = |x * sin(y * (pi/4)) + y * sin(x * (pi/4)))|
"""

import math
import numpy as np

N_BITS = 20
POPULATION_SIZE = 100
CONST_SCALE = 20/(2 ** (N_BITS//2) - 1)

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
    x = bits_to_int(bits[:N_BITS//2]) * CONST_SCALE
    y = bits_to_int(bits[N_BITS//2:]) * CONST_SCALE
    return (x, y)

def evaluation(population):
    """Avalia a pontução de cada indíviduo da população"""
    evaluations = [f(*decode(individual)) for individual in population]
    population_scored = np.insert(population, N_BITS, evaluations, axis=1)
    return population_scored


def crossover(population):
    """Cruzamento entre os indíviduos: corte de um ponto"""

    np.random.shuffle(population)

    # remove a última coluna com pontuação
    population = population[:, :-1]

    children = []
    queue = list(population)
    while queue:
        father, mother = queue.pop(), queue.pop()
        cut_point = np.random.randint(N_BITS)

        fchromo_y = father[cut_point:]
        mchromo_x = mother[:cut_point]
        fchromo_x = father[:cut_point]
        mchromo_y = mother[cut_point:]

        child1 = np.concatenate([fchromo_x, mchromo_y])
        child2 = np.concatenate([fchromo_y, mchromo_x])

        children.append(child1)
        children.append(child2)

    return np.array(children)


def random_event(probability):
    """Realiza um experimento pseudo-aleatório com probabilidade parametrizada.

    Parâmetros
    ----------
    probability: deve ser um número real e estar no conjunto (0, 1]

    """
    if not probability > 0:
        raise ValueError("probability should be greater than 0.")
    max_value = 1/probability
    if np.random.randint(0, max_value) == 0:
        return True
    return False



def mutate(population, probability=0.005):
    """Mutação de indíviduos"""
    for i in range(len(population)):
        if random_event(probability):
          print("MUTATION OCCURS!")
          j = np.random.randint(0, N_BITS)
          population[i][j] = 1 - population[i][j]
    return population


def selection(population_augmented):
    """Roleta RUSSA. Quem é mais apto tem mais chance de sobreviver."""
    # TODO: Implement selection
    return population_augmented

def evolution_step(population):
    # EVOLUTION
    population_scored = evaluation(population)
    best_individuals = selection(population_scored)
    new_population = mutate(crossover(best_individuals))
    return new_population

def population_report(population, verbose=True):
    # get best individual
    population_scored = evaluation(population)
    evaluations = population_scored[:, N_BITS]
    population_sorted = population_scored[evaluations.argsort()]
    best_individual = population_sorted[-1, :-1]
    best_point = population_sorted[-1]

    # metrics current population
    mean = np.mean(evaluations)
    std = np.std(evaluations)
    max_evaluation = np.max(evaluations)
    min_evaluation = np.min(evaluations)
    x_best, y_best = decode(best_individual)
    if verbose:
        print("MEAN: ", mean)
        print("STD: ", std)
        print("MAX: ", max_evaluation)
        print("MIN: ", min_evaluation)
        print("BEST_SOLUTION: ", (x_best, y_best))

    return [x_best, y_best, mean, std, min_evaluation, max_evaluation]


def main():
    print(__doc__)
    population = random_population()
    reports = []
    max_iterations = 100
    i = 0
    while i <= max_iterations:
        print("---- G E N E R A T I O N [{}] ---".format(i))
        report = population_report(population, False)
        reports.append(report)
        population = evolution_step(population)


        i += 1

    solutions = np.array(reports)
    _, columns = solutions.shape
    evaluation_idx = solutions[:, columns - 1].argsort()
    solutions_sorted = solutions[evaluation_idx]
    x = solutions_sorted[-1, 0]
    y = solutions_sorted[-1, 0]

    print("------ B E S T -- S O L U T I O N ------")
    print("X: {:.2f}".format(x))
    print("Y: {:.2f}".format(y))



if __name__ == '__main__':
    main()
