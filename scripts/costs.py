# -*- coding: utf-8 -*-
"""Functions used to compute the loss and weights."""
import numpy as np
import helpers as h

#************ Standard Cost Functions  **********************

def mse(e) :
    return 0.5 * np.mean(e**2)

def mae(e) :
    return np.mean(np.abs(e))

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return mse(e) #you can use mae instead


#************ Cost Functions for logistic regression **********************

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = h.sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = h.sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad
