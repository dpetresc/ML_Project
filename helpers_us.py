# -*- coding: utf-8 -*-
"""additional helper functions used for project 1."""

from proj1_helpers import load_csv_data
import numpy as np

def build_model_data(y_feature, x_feature):
    """Form (y,tX) to get regression data in matrix form."""
    y = y_feature
    x = x_feature
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def standardize(x):
    """Standardize the original data set.
    
    :param x: 
    :return: 
    """""
    x_std = np.std(x, axis=0)
    x_mean = np.mean(x, axis=0)
    return (x - x_mean)/x_std, x_mean, x_std


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization.

    :param x:
    :param mean_x:
    :param std_x:
    :return:
    """
    x = x * std_x
    x = x + mean_x
    return x


# def process_data(path):
#     """
#
#     :param path:
#     :return:
#     """
#     #y, X, ids = load_csv_data(path)
#     #new_X = X
#
#     #for j in range(len(new_X[0])) :
#     #    col = new_X[:, j]
#     #    m = np.mean(col[col >= -900]) #compute mean of the right columns
#     #    col[np.where(col < -900)] = m
#     #    new_X[:, j] = col
#     #new_X, x_mean, x_std = standardize(new_X)
#     #return y, new_X, x_mean, x_std, id
#     y, X, ids = load_csv_data(path)
#     new_X = X
#
#     for j in range(len(new_X[0])):
#         col = new_X[:, j]
#         m = np.mean(col[col >= -900])  # compute mean of the right columns
#         # m = np.median(col[col >= -900]) #compute median of the right columns
#         col[np.where(col < -900)] = m
#         new_X[:, j] = col
#
#     new_X, x_mean, x_std = standardize(new_X)
#     new_X = np.hstack((np.ones((new_X.shape[0], 1)), new_X))
#     return y, new_X, x_mean, x_std, ids

def column_weighting(X) :
    remove_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    w = 0.9
    for j in range(X.shape[1]) :
        if(j in remove_ind):
            X[j] = (1-w)*X[j]
        else :
            X[j] = w*X[j]
    return X

def inv_log_f(x) :
    inv_log_cols = [0, 2, 5, 7, 9, 10, 13, 16, 19, 21, 23, 26]
    others = [1, 3, 5, 6, 8, 11, 12, 14, 15, 17, 18, 20, 22, 25, 27, 28, 29]
    # test = [0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26]
    # Create inverse log values of features which are positive in value.
    # x_inv_log_cols = np.log(1 / (1 + x[:, inv_log_cols]))
    x_inv_log_cols = np.log(x[:, inv_log_cols])
    # x[:, inv_log_cols] = np.log(x[:, inv_log_cols])
    # x[:, others] = np.log(x[:, others] + 1 - np.min(x[:, others]))

    # temp = x[:, others]
    # x_others = np.log(temp + 1 - np.min(temp))
    x_inv = np.hstack((x, x_inv_log_cols))
    # x_inv = np.hstack((x_inv_log_cols, x_others))
    # x_inv = np.hstack((x, x_inv))
    return x_inv


def get_jet_masks(x):
    dictionnary = {0: x[:, 22] == 0, 1: x[:, 22] == 1, 2: np.logical_or(x[:, 22] == 2, x[:, 22] == 3)}
    return dictionnary

def process_data(path, inv_log=False):
    y, X, ids = load_csv_data(path)
    new_X = X

    for j in range(len(new_X[0])):
        col = new_X[:, j]
        m = np.mean(col[col >= -900])  # compute mean of the right columns
        col[np.where(col < -900)] = m
        new_X[:, j] = col

    if (inv_log):
        new_X = inv_log_f(new_X)

    new_X, x_mean, x_std = standardize(new_X)
    new_X = np.hstack((np.ones((new_X.shape[0], 1)), new_X))
    return y, new_X, x_mean, x_std, ids

def process_data2(path, inv_log=False):
    y, X, ids = load_csv_data(path)
    new_X = X

    for j in range(len(new_X[0])):
        col = new_X[:, j]
        m = np.mean(col[col >= -900])  # compute mean of the right columns
        col[np.where(col < -900)] = m
        new_X[:, j] = col

    if (inv_log):
        new_X = inv_log_f(new_X)

    dict_mask_jets_train = get_jet_masks(new_X)
    new_X, x_mean, x_std = standardize(new_X)
    #new_X = np.hstack((np.ones((new_X.shape[0], 1)), new_X))
    return y, new_X, x_mean, x_std, ids, dict_mask_jets_train



