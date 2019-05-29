#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: AI-UFC 2019.1 T1
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

"""== LÓGICA FUZZY
Q2) Construa um programa baseado em lógica fuzzy (inferência de Mamdani) que receba
três valores: pressão no pedal, velocidade da roda e velocidade do carro e que devolva
a pressão no freio. Siga as regras disponibilizadas nos slides sobre Lógica Fuzzy.
"""

import numpy as np

# Funções de pertencimento da Pressão no Pedal

def pressao_pedal_baixa(x):
    """Função de pertencimento BAIXA: x -> pressão no pedal"""
    if x < 0:
        return 1
    elif x > 50:
        return 0
    else:
        return 1 - x/50


def pressao_pedal_media(x):
    """Função de pertencimento MEDIA: x -> pressão no pedal"""
    if 30 <= x < 50:
        return (x/20) - 3/2
    elif 50 <= x <= 70:
        return 7/2 - (x/20)
    else:
        return 0

def pressao_pedal_alta(x):
    """Função de pertencimento ALTA: x -> pressão no pedal"""
    if x < 50:
        return 0
    elif x > 100:
        return 1
    else:
        return x/50 - 1


# Funções de pertencimento de Velocidade da Roda

def velocidade_roda_baixa(x):
    """Função de pertencimento BAIXA: x -> velocidade da roda"""
    if x < 0:
        return 1
    elif x > 60:
        return 0
    else:
        return 1 - x/60


def velocidade_roda_media(x):
    """Função de pertencimento MEDIA: x -> velocidade da roda"""
    if 20 <= x < 50:
        return (x/30) - 3/2
    elif 50 <= x <= 80:
        return 8/3 - (x/30)
    else:
        return 0


def velocidade_roda_alta(x):
    """Função de pertencimento ALTA: x -> velocidade da roda"""
    if x < 40:
        return 0
    elif x > 100:
        return 1
    else:
        return x/60 - 2/3


# Funções de pertencimento de Velocidade do Carro (igual velocidade da roda)


def velocidade_carro_baixa(x):
    """Função de pertencimento BAIXA: x -> velocidade do carro"""
    return velocidade_roda_baixa(x)

def velocidade_carro_media(x):
    """Função de pertencimento MEDIA: x -> velocidade do carro"""
    return velocidade_roda_media(x)

def velocidade_carro_alta(x):
    """Função de pertencimento ALTA: x -> velocidade do carro"""
    return velocidade_roda_alta(x)

# Funções de Pertencimento de Ação de Freio: LIBERE e APERTE

def acao_freio_aperte(x):
    """Função de pertencimento APERTE: x -> aperte o freio"""
    if x > 100:
        return 1
    elif x < 0:
        return 0
    else:
        return x/100


def acao_freio_libere(x):
    """Função de pertencimento LIBERE: x -> libere o freio"""
    if x > 100:
        return 0
    elif x < 0:
        return 1
    else:
        return 1 - x/100

def agg(aperte, libere):
    "Função agregada com pontos de corte aperte e libere."
    return lambda x: max(
        min(aperte, acao_freio_aperte(x)),
        min(libere, acao_freio_libere(x))
    )

def orf(*args):
    "Or fuzzy"
    return max(*args)

def andf(*args):
    "And fuzzy"
    return min(*args)


def implicacao(a, b):
    "Implicação de Godel"
    return orf(int(a <= b), b)


def defuzzy(aperte, libere):
    X = np.linspace(0, 100, 1000)
    P = agg(aperte, libere) # gerar função agregadora
    return sum([x * P(x) for x in X]) / sum([P(x) for x in X])

def sistema_fuzzy(pp, vc, vr):
    """
    Valores nítidos:
    + pp: pressão no pedal
    + vr: velocidade da roda
    + vc: velocidade do carro

    Saída:
    + pf: pressão no freio
    """
    pp_alta = pressao_pedal_alta(pp)
    pp_media = pressao_pedal_media(pp)
    pp_baixa = pressao_pedal_baixa(pp)

    vc_alta = velocidade_carro_alta(vc)
    vc_media = velocidade_carro_media(vc)
    vc_baixa = velocidade_carro_baixa(vc)

    vr_alta = velocidade_roda_alta(vr)
    vr_media = velocidade_roda_media(vr)
    vr_baixa = velocidade_roda_baixa(vr)


    r1 = pp_media # aperte
    r2 = andf(pp_alta, vc_alta, vr_alta) # aperte
    r3 = andf(pp_alta, vc_alta, vr_baixa) # libere
    r4 = pp_baixa # libere


    print("R1: ", r1)
    print("R2: ", r2)
    print("R3: ", r3)
    print("R4: ", r4)

    aperte = r1 + r2
    libere = r3 + r4
    print("APERTE: ", aperte)
    print("LIBERE: ", libere)

    pf = defuzzy(aperte, libere)
    return pf



def main():
    print(__doc__)
    pp = 60
    vc = 80
    vr = 55
    print("== ENTRADAS")
    print("PRESSÃO NO PEDAL: ", pp)
    print("VELOCIDADE DO CARRO: ", vc)
    print("VELOCIDADE DA RODA: ", vr)
    print("== FUZZY SYSTEM")
    pf = sistema_fuzzy(pp, vc, vr)
    print("== SAÍDA")
    print("PRESSÃO NO FREIO: ", pf)


main()
