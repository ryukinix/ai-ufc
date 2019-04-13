import numpy as np


def dataset():
    with open('aerogerador.dat') as f:
        data = []
        for line in f.readlines():
            x, y = line.split()
            data.append([float(x), float(y)])
        array = np.array(data)
        return array[:, 0], array[:, 1]
