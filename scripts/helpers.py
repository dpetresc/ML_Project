# -*- coding: utf-8 -*-
"""some helpers functions."""
import numpy as np


#******************* Helpers data processing ************************

def build_model_data(y_feature, x_feature):
    """Form (y,tX) to get regression data in matrix form."""
    y = y_feature
    x = x_feature
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly_basis = np.ones((len(x), 1))
    for deg in range(1, degree + 1) :
        poly_basis = np.c_[poly_basis, np.power(x, deg)]
    return poly_basis

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


#******************* Helpers used for regression techniques ************************
def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    return -((tx.T).dot(e))/len(y)

def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))
