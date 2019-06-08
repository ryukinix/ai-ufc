#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional UFC 2019.1 - Redes Neurais
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

import numpy as np


def dataset(fname):
    """Load a .dat file as numpy array-matrix"""
    with open(fname) as f:
        data = []
        for line in f.readlines():
            numbers = line.split()
            data.append([float(num) for num in numbers])
        return np.array(data)



def aerogerador():
    """
    Columns description
    -------------------

    0: velocidade do vento (m/s)
    1: potência gerada (kW)

    x -> 0
    y -> 1
    """
    d = dataset('data/aerogerador.dat')
    x, y = d[:, 0], d[:, 1]
    return x, y


def iris():
    """
    Columns description
    -------------------
    0: caracteristica 1
    1: caracteristica 2
    2: caracteristica 3
    3: caracteristica 4
    4: classe 1
    5: classe 2
    6: classe 3

    X -> [0, 1, 2, 3]
    y -> [4, 5, 6]
    """
    d = dataset('data/iris_log.dat')
    X, y = d[:, 0:4], d[:, -3:]
    return X, y

def twomoons():
    d = dataset('data/twomoons.dat')
    X, y = d[:, 0:2], d[:, -1]
    return X, y
