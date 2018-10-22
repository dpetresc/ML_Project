# -*- coding: utf-8 -*-
"""This file contains the different regression techniques."""
import numpy as np
from costs import *
from helpers import *

# ******************************************************
# REGRESSION METHODS
# ******************************************************
def least_squares_GD(y, tx, initial_w,max_iters, gamma) :
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # compute gradient and loss
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        
	# update w by gradient
        w = w - gamma*gradient
        
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma) :
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    w = initial_w
    loss = 0

    batch_size = 1
    for n_iter in range(max_iters):
        for y_b, tx_b in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            loss = compute_loss(y, tx, w)
            gradient = compute_gradient(y_b, tx_b, w)
            w = w - gamma*gradient
    return w, loss

def least_squares(y, tx) :
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_) :
    A = tx.T.dot(tx) 
    B = tx.T.dot(y)
    lambda_prime =  lambda_ * 2 * len(y)
    w = np.linalg.solve(A + lambda_prime * np.identity(len(A)), B)
    loss = compute_loss(y, tx, w)
    return w, loss

# ********* MISSING IMPLEMENTATIONS ***********************

#def logistic_regression(y, tx, initial_w, max_iters, gamma) :

#def reg_logistic regression(y, tx, lambda_, initial_w, max_iters, gamma) :

