# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    A = tx.T.dot(tx) 
    B = tx.T.dot(y)
    lambda_prime =  lambda_ * 2 * len(y)
    w = np.linalg.solve(A + lambda_prime * np.identity(len(A)), B)
    loss = compute_mse(y, tx, w)
    return w, loss
    
