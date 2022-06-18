import numpy as np


# definite matrix
import sklearn


def matGen(size, scale=1):
    D = sklearn.datasets.make_spd_matrix(size) * scale
    print(D)
    return D

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
