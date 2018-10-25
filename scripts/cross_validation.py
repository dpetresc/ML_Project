# -*- coding: utf-8 -*-
"""Functions used to run cross validation for our models and to plot the obtain results."""
import numpy as np
from plot import *
from helpers import *
from proj1_helpers import *
from regressions import *
from ipywidgets import IntProgress
from IPython.display import display
import time


# ********* helpers functions **************************


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]

    return np.array(k_indices)



# ******************************************************
# CROSS VALIDATION FUNCTIONS FOR LINEAR REGRESSION
# ******************************************************
def cross_validation(y, x, k_indices, k, regression_technique, **args):
    
    #Build test and training set
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_test = y[te_indice]
    y_train = y[tr_indice]
    X_test = x[te_indice]
    X_train = x[tr_indice]    

    #********************** CHOOSE REGRESSION TECHNIQUE *****************
    w, loss = regression_technique(y=y_train, tx=X_train, **args)
    #calculate the loss for train and test data
    loss_tr = np.sqrt(2 * loss)
    loss_te = np.sqrt(2 * compute_loss(y_test, X_test, w))
    
    
    return loss_tr, loss_te, w

def cross_validation_demo(y, x, regression_technique, **args):
    f = IntProgress(min=0, max=30) # instantiate the bar
    display(f) # display the bar

    seed = 12
    k_fold = 4
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    count = 0
    min_rmse_tr = []
    min_rmse_te = []
    for k in range(k_fold) :
        loss_tr, loss_te, _ = cross_validation(y, x, k_indices, k, regression_technique, **args)
        min_rmse_tr.append(loss_tr)
        min_rmse_te.append(loss_te)

    f.value += 1 # signal to increment the progress bar
    time.sleep(.1)
    count += 1
    cross_validation_visualization(np.arange(k_fold), min_rmse_tr, min_rmse_te, 'folds') #TO CHANGE


# ******************************************************
# CROSS VALIDATION FUNCTIONS FOR RIDGE REGRESSION
# ******************************************************
def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
    
    #Build test and training set
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_test = y[te_indice]
    y_train = y[tr_indice]
    X_test = x[te_indice]
    X_train = x[tr_indice]
    
    #form data with polynomial degree
    tx_training = build_poly(X_train, degree)
    tx_test = build_poly(X_test, degree)

    #********************** RIDGE REGRESSION *****************
    w, loss = ridge_regression(y_train, tx_training, lambda_)

    #calculate the loss for train and test data
    loss_tr = np.sqrt(2 * loss)
    loss_te = np.sqrt(2 * compute_loss(y_test, tx_test, w))
    
    
    return loss_tr, loss_te, w

def cross_validation_ridge_demo(y, x):
    f = IntProgress(min=0, max=4) # instantiate the bar
    display(f) # display the bar
    seed = 12
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
     
    count = 0
    degree = [1,9]
    global_min_tr = []
    global_min_te = []
    best_lambdas = []
    global_w = []
    for deg in degree:
        rmse_tr = []
        rmse_te = []
        w_list = []
        for lambda_ in lambdas :
            min_rmse_tr = []
            min_rmse_te = []
            w_k = []
            for k in range(k_fold) :
                loss_tr, loss_te, w = cross_validation_ridge(y, x, k_indices, k,lambda_ , deg)
                min_rmse_tr.append(loss_tr)
                min_rmse_te.append(loss_te)
                w_k.append(w)
            rmse_tr.append(np.mean(min_rmse_tr))
            rmse_te.append(np.mean(min_rmse_te))
            w_list.append(np.mean(w_k, axis=0))
            
        indice = np.argmin(rmse_te)
        global_min_tr.append(rmse_tr[indice])
        global_min_te.append(rmse_te[indice])
        best_lambdas.append(lambdas[indice])
        global_w.append(w_list[indice])
        
        
        f.value += 1 # signal to increment the progress bar
        time.sleep(.1)
        count += 1
     
    cross_validation_visualization(degree, global_min_tr, global_min_te, 'degrees')
    return best_lambdas, global_min_tr, global_min_te, global_w


# ******************************************************
# CROSS VALIDATION FUNCTIONS FOR LOGISTIC REGRESSION
# ******************************************************


