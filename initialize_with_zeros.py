import numpy as np                   #importing the library for scientific computation

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b
