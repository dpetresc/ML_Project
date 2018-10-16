# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_training = indices[: index_split]
    index_test = indices[index_split:]
    # ***************************************************
    x_training = x[index_training]
    x_test = x[index_test]
    y_training = y[index_training]
    y_test = y[index_test]
    return x_training, x_test, y_training, y_test
