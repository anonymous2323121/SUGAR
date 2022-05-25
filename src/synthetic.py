import numpy as np
from copy import deepcopy
from sklearn.preprocessing import PolynomialFeatures

def generate_W(d=6, prob=0.5, low=0.5, high=2.0):
    """
    generate a random weighted adjaceecy matrix
    :param d: number of nodes
    :param prob: prob of existing an edge
    :return:
    """
    g_random = np.float32(np.random.rand(d,d)<prob)
    g_random = np.tril(g_random, -1)
    U = np.round(np.random.uniform(low=low, high=high, size=[d, d]), 1)
    U[np.random.randn(d, d) < 0] *= -1
    W = (g_random != 0).astype(float) * U
    return W