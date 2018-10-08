# -*- coding: utf-8 -*-
"""Functions used to compute the loss and weights."""

# ******************************************************
# HELPER FUNCTIONS 
# ******************************************************
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
    for deg in range((1, deg + 1)) :
        poly_basis = np.c_[poly, np.power(x, deg)]
    return poly_basis

def mse(e) :
    return 0.5 * np.mean(e**2)

def mae(e) :
    return np.mean(np.abs(e))

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return mse(e)

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    return -((tx.T).dot(e))/len(y)

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


# ******************************************************


# ******************************************************
# REGRESSION METHODS
# ******************************************************
def least squares GD(y, tx, initial w,max_iters, gamma) :
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # compute gradient and loss
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        
        w = w - gamma*gradient
        # TODO: update w by gradient
        ws.append(w)
        losses.append(loss)
    return w, loss

def least squares SGD(y, tx, initial w, max_iters, gamma) :
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        for y_b, tx_b in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            loss = compute_loss(y, tx, w)
            gradient = compute_gradient(y_b, tx_b, w)
            w = w - gamma*gradient
    return w, loss

def least squares(y, tx) :
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge regression(y, tx, lambda_) :
    A = tx.T.dot(tx) 
    B = tx.T.dot(y)
    lambda_prime =  lambda_ * 2 * len(y)
    w = np.linalg.solve(A + lambda_prime * np.identity(len(A)), B)
    loss = compute_loss(y, tx, w)
    return w, loss






    