# -*- coding: utf-8 -*-
"""Functions used to compute the loss and weights."""
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import load_csv_data


#****************** HELPERS *********************** 
def standardize(x):
    
    x_std = np.std(x, axis=0)
    x_mean = np.mean(x, axis=0)
    return (x - x_mean)/x_std, x_mean, x_std

def de_standardize(x, mean_x, std_x):
    x = x * std_x
    x = x + mean_x
    return x

#****************** END HELPERS ******************* 

def process_data(path) :
    y, X, ids = load_csv_data(path)
    new_X = X

    for j in range(len(new_X[0])) :
        col = new_X[:, j]
        m = np.mean(col[col >= -900]) #compute mean of the right columns
        #m = np.median(col[col >= -900]) #compute median of the right columns
        col[np.where(col < -900)] = m
        new_X[:, j] = col
    new_X, x_mean, x_std  = standardize(new_X)
    return y, new_X, x_mean, x_std, ids





    
