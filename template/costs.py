# -*- coding: utf-8 -*-
"""A function to compute the cost."""


import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    temp = np.dot(tx,w)
    e = y - temp
    mse = e.dot(e) / (2 * len(e))
    return mse
