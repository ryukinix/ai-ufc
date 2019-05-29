# coding: utf-8

"""
Algoritmos Genéticos, Wed 15 May 2019 05:02:42 PM -03

IC, 2019.1. UFC, Sobral.

Author: Manoel Vilela.
"""

from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np

# função a ser optimizada, global ótimo: 2.5
# objetivo: maximizar
f = lambda x: -(x ** 2)+ 5 * x - 6

# população inicial
population = [0b101,
              0b100,
              0b010,
              0b111]

# mapeamento da codificação binária da população para a solução real
v_real = lambda v_bin: (v_bin*3.2)/7

# função para mapear a nota para um número positivo não-nulo
fit = lambda x: x + 2.5

# calcular nota de ajuste da população sobre f(x)
fitness =[round(fit(f(v_real(x))), ndigits=2)
          for x in population]

sum_fit = sum(fitness)
notes = [round(x/sum_fit, ndigits=2) for x in fitness]
mean = np.mean(fitness)

# apresentando valores
print()
print("Population: ", population)
print("Notes: ", notes)
print("Mean: ", mean)


x = np.linspace(0, 5, 100)
plt.plot(x, [f(v) for v in x])
plt.show()
