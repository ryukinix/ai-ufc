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
O indíviduo possui 20 bits tal que 10 bits é reservado para x e 10
bits para y.  A função objetiva f deve ser maximizada tal que a tupla
(x, y) pertença a subfaixa [0, 20] no conjunto R².

f(x, y) = |x * sin(y * (pi/4)) + y * sin(x * (pi/4))|
"""

import math
import numpy as np
import sys

# DOC: quantidade de bits do indíviduo
# aumente para ter uma maior precisão na casa decimal (sempre um número par).
N_BITS = 20

# DOC: tamanho da população
# fixa em todo o programa.
POPULATION_SIZE = 100

# DOC: constante de transformação.
# representação binária para a escala real.
CONST_SCALE = 20/(2 ** (N_BITS//2) - 1)

# DOC: parâmetro de verbosidade do programa
# Se verdadeiro imprime maiores detalhes durante a execução.
VERBOSE = False

# DOC: Quantidade máxima de iterações
MAX_ITERATIONS = 100

# DOC: Probabilidade de mutação de um gêne para um indíviduo
MUTATION_PROBABILITY = 0.005

def f(x, y):
    """Função-objetiva para maximização."""
    return abs(x * math.sin(y * ((math.pi)/4)) + y * math.sin(x * ((math.pi)/4)))

def random_population(size=POPULATION_SIZE):
    """Gera uma população aleatória de tamamho 'size'."""
    return np.random.random_integers(0, 1, (size, N_BITS)).astype('float')

def bits_to_int(bits):
    """Recebe uma sequência de bits e codifica em um número decimal"""
    n = len(bits)
    integer = 0
    for i, b in enumerate(bits):
        if b:
            integer += 2 ** (n - i - 1)

    return integer

def decode(bits):
    """Decodifica o indivíduo i representado em bits para (x, y)."""
    x = bits_to_int(bits[:N_BITS//2]) * CONST_SCALE
    y = bits_to_int(bits[N_BITS//2:]) * CONST_SCALE
    return (x, y)

def encode(bits):
    """Codifica numa string a sequência de bits do indíviduo."""
    fstring = "#0{}b".format(N_BITS)
    return format(bits_to_int(bits), fstring)

def evaluation(population):
    """Avalia a pontução de cada indíviduo da população.

    Parâmetros
    ---------
    population: numpy.ndarray(shape=(POPULATION_SIZE, N_BITS))
    - uma matriz de indíviduos: linhas são indíviduos, colunas são genes.

    Return
    -------
    np.ndarray(shape=(POPULATION_SIZE, N_BITS + 1))
    Retorna uma matriz aumentada com a pontuação de cada indíviduo.
    """
    evaluations = [f(*decode(individual)) for individual in population]
    population_scored = np.insert(population, N_BITS, evaluations, axis=1)
    return population_scored


def crossover(population):
    """Cruzamento entre os indíviduos: corte de um ponto.

    Parâmetros
    ---------
    population: numpy.ndarray(shape=(POPULATION_SIZE, N_BITS))
    - uma matriz de indíviduos: linhas são indíviduos, colunas são genes.

    Return
    ------
    children: numpy.ndarray(shape=(POPULATION_SIZE, N_BITS))
    Retorna uma nova população cruzada com a mesma dimensionalidade anterior.
    """

    np.random.shuffle(population)

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
    probability: float
    - deve ser um número real e estar no conjunto (0, 1]

    Return
    ------
    experiment: boolean.
    - se True experimento foi realizado com sucesso, False do contrário.
    """
    if not probability > 0:
        raise ValueError("probability should be greater than 0.")
    max_value = 1/probability
    if np.random.randint(0, max_value) == 0:
        return True
    return False



def mutate(population, probability=MUTATION_PROBABILITY):
    """Mutação de indíviduos

    NOTA: SIDE-EFFECT!
    essa função modifica o argumento population!

    Parâmetros
    ----------
    population: numpy.ndarray(shape=(POPULATION_SIZE, N_BITS))
    - uma matriz de indíviduos: linhas são indíviduos, colunas são genes.
    probability: float
    - deve ser um número real e estar no conjunto (0, 1]

    Return
    ------
    population: numpy.ndarray(shape=(POPULATION_SIZE, N_BITS))
    - matriz de indíviduos após a operação aleatória de mutação.
    """
    fstring = "[MUTATION] individual({}) gene({}) swapped from {} to {}"
    for i in range(len(population)):
        if random_event(probability):
          j = np.random.randint(0, N_BITS)
          gene = int(population[i][j])
          new_gene = int(1 - gene)
          population[i][j] = new_gene
          if VERBOSE:
              print(fstring.format(i, j, gene, new_gene))
    return population


def selection(population_scored):
    """Roleta RUSSA VICIADA. Quem é mais apto tem mais chance de sobreviver."""
    evaluations = population_scored[:, -1]
    probability = evaluations/sum(evaluations)
    population = population_scored[:, :-1]
    natural_selection = np.empty(shape=(POPULATION_SIZE, N_BITS))
    # define a precisão da roleta
    roleta_size = 1000
    roleta = []

    # CRIA ROLETA COM VIÉS SOBRE A PONTUAÇÃO DO INDIVÍDUO
    for idx, c in enumerate(roleta_size * probability):
        for _ in range(math.ceil(c)):
            roleta.append(population[idx])

    for i in range(POPULATION_SIZE):
        event = np.random.randint(roleta_size)
        selected = roleta[event]
        natural_selection[i] = selected

    return natural_selection

def evolution_step(population):
    """Núcleo do algoritmo genético: realiza um passo evolutivo.

    Passos:
    1. Avaliação
    2. Seleção
    3. CrossOver
    4. Mutação

    Parâmetros
    ---------
    population: numpy.ndarray(shape=(POPULATION_SIZE, N_BITS))
    - uma matriz de indíviduos: linhas são indíviduos, colunas são genes.

    Return
    ------
    new_population: numpy.ndarray(shape=(POPULATION_SIZE, N_BITS))
    - Uma nova população: a evolução de uma geração.
    """
    # EVOLUTION
    population_scored = evaluation(population)
    best_individuals = selection(population_scored)
    new_population = mutate(crossover(best_individuals))
    return new_population

def population_report(i, population):
    """Extrai métricas da população e a melhor solução atual"""
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
    string_best_individual = encode(best_individual)
    if VERBOSE:
        print("\n---- G E N E R A T I O N [{}] ---".format(i))
        print("MEAN: {:.4f}".format(mean))
        print(" STD: {:.4f}".format(std))
        print(" MAX: {:.4f}".format(max_evaluation))
        print(" MIN: {:.4f}".format(min_evaluation))
        print("BEST_INDIVIDUAL: ", string_best_individual)
        print("  BEST_SOLUTION: ({:.4f}, {:.4f})".format(x_best, y_best))

    return [i, x_best, y_best, mean, std, min_evaluation, max_evaluation]

def description():
    print(__doc__)
    print("--- P A R A M E T E R S ---")
    print("N_BITS: ", N_BITS)
    print("POPULATION_SIZE: ", POPULATION_SIZE)
    print("SELECTION: ROLETA")
    print("CROSSOVER: 1-cut point")
    print("MUTATION PROBABILITY: ", MUTATION_PROBABILITY)
    print("VERBOSE: ", VERBOSE)

def parse_flags():
    global VERBOSE
    for flag in sys.argv[1:]:
        if flag == '-v' or flag == '--verbose':
            VERBOSE = True

def main():
    parse_flags()
    description()
    population = random_population()
    reports = []
    i = 0
    while i <= MAX_ITERATIONS:
        report = population_report(i, population)
        reports.append(report)
        population = evolution_step(population)

        i += 1

    solutions = np.array(reports)
    _, columns = solutions.shape
    evaluation_idx = solutions[:, columns - 1].argsort()
    solutions_sorted = solutions[evaluation_idx]
    best_i = solutions_sorted[-1, 0]
    x = solutions_sorted[-1, 1]
    y = solutions_sorted[-1, 2]
    fxy = solutions_sorted[-1, columns - 1]

    print("------ B E S T -- S O L U T I O N ------")
    print("(x, y) = ({:.4f}, {:.4f})".format(x, y))
    print("f(x,y) = {:.4f}".format(fxy))
    print("Best generation = {}".format(int(best_i)))




if __name__ == '__main__':
    main()
