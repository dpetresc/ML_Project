# -*- coding: utf-8 -*-
"""Helpers to select best model."""

import implementations as imp
import visualization as visu
import numpy as np
from proj1_helpers import *

def calculate_accuracy(y, y_pred):
    """

    :param y: actual y
    :param y_pred: the predictions
    :return: accuracy
    """
    accuracy = np.count_nonzero(y == y_pred) / len(y) * 100
    return accuracy

def predict_labels_logistic(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred=imp.sigmoid(y_pred)
    y_pred[np.where(y_pred >= 0.5)] = 1
    y_pred[np.where(y_pred < 0.5)] = 0
    # 1,0 => 1,-1
    y_pred = 2 * y_pred
    y_pred = y_pred - 1
    return y_pred

def build_poly(x, degree):
    """Create extended feature matrix formed by applying the polynomial basis functions to all input data.

        :param x: vector of the data samples
        :param degree: maximum degree of the polynomial basis
        :return: extended feature matrix
    """
    ext_matrix = np.ones((len(x), 1))
    for deg in range(1, degree + 1) :
        ext_matrix = np.c_[ext_matrix, np.power(x, deg)]
    return ext_matrix

def build_k_indices(y, k_fold, seed):
    """Build k indices groups for k-fold.

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param k_fold: number of folds
        :param seed: random seed
        :return: indices for k-fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


# ******************************************************
# CROSS VALIDATION FUNCTIONS FOR LINEAR REGRESSION
# ******************************************************
def cross_validation(y, x, k_indices, k, regression_technique, **args):
    """Cross validation helper function for linear regression techniques

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param x: vector of the data samples
        :param k_indices: k indices groups for k-fold
        :param k: k'th group to select
        :param regression_technique: regression technique (least_squares, etc.)
        :param args: args for regression (ex: max_iters, gamma)
        :return: loss for train, loss for test, accuracy, weights
    """
    # Build test and training set
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_test = y[te_indice]
    y_train = y[tr_indice]
    X_test = x[te_indice]
    X_train = x[tr_indice]

    # Choose the regression technique
    w, loss = regression_technique(y=y_train, tx=X_train, **args)
    # calculate the loss for train and test data
    loss_tr = imp.calculate_rmse(loss)
    loss_te = imp.calculate_rmse(imp.compute_loss(y_test, X_test, w))
    accuracy = calculate_accuracy(y_test, predict_labels(w, X_test))
    return loss_tr, loss_te, accuracy, w

def cross_validation_demo(y, x, regression_technique, **args):
    """Cross validation function for linear regression techniques

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param x: vector of the data samples
        :param regression_technique:
        :param args: args for regression (ex: max_iters, gamma)
    """

    seed = 12
    k_fold = 5
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    min_rmse_train = []
    min_rmse_test = []
    accuracies = []
    weights = []
    for k in range(k_fold):
        loss_tr, loss_te, accuracy_tmp, w = cross_validation(y, x, k_indices, k, regression_technique, **args)
        min_rmse_train.append(loss_tr)
        min_rmse_test.append(loss_te)
        accuracies.append(accuracy_tmp)
        weights.append(w)

    rmse_test = np.mean(min_rmse_test)
    accuracy = np.mean(accuracies)
    w = np.mean(weights, axis=0)
    print("RMSE test: ", rmse_test)
    print("Accuracy: ", accuracy)
    visu.cross_validation_visualization(np.arange(k_fold), min_rmse_train, min_rmse_test)
    return w


def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
    """Cross validation helper function for ridge regression techniques

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param x: vector of the data samples
        :param k_indices: k indices groups for k-fold
        :param k: k'th group to select
        :param lambda_: regularization factor (penalty factor)
        :param degree: maximum degree of the polynomial basis
        :return: loss for train, loss for test, weights
    """
    # Build test and training set
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_test = y[te_indice]
    y_train = y[tr_indice]
    X_test = x[te_indice]
    X_train = x[tr_indice]

    # form data with polynomial degree
    tx_train = build_poly(X_train, degree)
    tx_test = build_poly(X_test, degree)

    # ridge regression
    w, loss = imp.ridge_regression(y_train, tx_train, lambda_)

    # calculate the loss for train and test data
    loss_train = imp.calculate_rmse(loss)
    loss_test = imp.calculate_rmse(imp.compute_loss(y_test, tx_test, w))
    accuracy = calculate_accuracy(y_test, predict_labels(w, tx_test))
    return loss_train, loss_test, accuracy, w


def best_model_ridge(y, x, k_fold, degrees, lambdas, seed=56):
    """Calculate best degree and best lambda

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param x: vector of the data samples
        :param k_fold: number of folds
        :param degrees:
        :param lambdas: lambdas to test
        :param seed: random seed
        :return: best degree and best lambda for ridge regression
    """
    k_indices = build_k_indices(y, k_fold, seed)
    best_lambdas = []
    best_rmses = []
    best_accuracies = []
    for degree in degrees:
        rmse_train = []
        rmse_test = []
        accuracies = []
        for lambda_ in lambdas:
            rmse_train_lambda = []
            rmse_test_lambda = []
            accuracies_lambda = []
            for k in range(k_fold):
                loss_train, loss_test, accuracy_tmp, w = cross_validation_ridge(y, x, k_indices, k, lambda_, degree)
                rmse_train_lambda.append(loss_train)
                rmse_test_lambda.append(loss_test)
                accuracies_lambda.append(accuracy_tmp)

            rmse_train.append(np.mean(rmse_train_lambda))
            rmse_test.append(np.mean(rmse_test_lambda))
            accuracies.append(np.mean(accuracies_lambda))


        ind_lambda_opt = np.argmin(rmse_test)
        best_lamda_tmp = lambdas[ind_lambda_opt]
        best_rmse_tmp = rmse_test[ind_lambda_opt]
        best_lambdas.append(best_lamda_tmp)
        best_rmses.append(best_rmse_tmp)
        best_accuracies.append(accuracies[ind_lambda_opt])
        visu.cross_validation_visualization_ridge(lambdas, rmse_train, rmse_test, degree, best_lamda_tmp, best_rmse_tmp)
        print(best_lamda_tmp, best_rmse_tmp)

    ind_best_degree = np.argmin(best_rmses)
    best_lambda = best_lambdas[ind_best_degree]
    best_degree = degrees[ind_best_degree]
    accuracy = best_accuracies[ind_best_degree]
    print("Accuracy: ", accuracy)
    return best_degree, best_lambda

# ******************************************************
# CROSS VALIDATION FUNCTIONS FOR REGULARIZED LOGISTIC REGRESSION
# ******************************************************
def cross_validation_logistic(y, x, k_indices, k, degree, lambda_):
    """

        :param y:
        :param x:
        :param k_indices:
        :param k:
        :param degree:
        :param lambda_:
        :return:
    """
    # Build test and training set
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_test = y[te_indice]
    y_train = y[tr_indice]
    X_test = x[te_indice]
    X_train = x[tr_indice]

    # form data with polynomial degree
    tx_training = build_poly(X_train, degree)
    tx_test = build_poly(X_test, degree)

    # ******** LOGISTIC REGRESSION *******
    # w, loss = logistic_regression(y_train, tx_training)

    # ******** REG LOGISTIC REGRESSION *******
    w, loss = imp.reg_logistic_regression(y_train, tx_training, lambda_)

    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * loss)
    # loss_te = np.sqrt(2 * calculate_loss(y_test, tx_test, w))
    pred = imp.sigmoid(tx_test @ w)
    loss_te = np.sum(np.where(np.abs(y_test - pred) < 0.5, 1, 0)) / len(y_test)

    return loss_tr, loss_te, w


def cross_validation_logistic_demo(y, x, lambda_):
    """

        :param y:
        :param x:
        :return:
    """
    seed = 12
    k_fold = 5
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data

    degree = np.array([2])
    global_min_tr = []
    global_min_te = []
    global_w = []
    global_lambdas = []

    for deg in degree:

        min_rmse_tr = []
        min_rmse_te = []
        w_k = []
        lambdas = []
        for k in range(k_fold):
            max_lambda = 0
            max_loss_te = 0
            max_loss_tr = 0
            max_w = []
            for lam in np.array([lambda_]):
                loss_tr, loss_te, w = cross_validation_logistic(y, x, k_indices, k, deg, lam)
                if loss_te > max_loss_te:
                    max_loss_te = loss_te
                    max_lambda = lam
                    max_loss_tr = loss_tr
                    max_w = w
            print('DEGREE', deg, 'lambda', max_lambda, 'accuracy', max_loss_te)
            min_rmse_tr.append(max_loss_tr)
            min_rmse_te.append(max_loss_te)
            w_k.append(max_w)
            lambdas.append(max_lambda)

        global_min_tr.append(np.mean(min_rmse_tr))
        global_min_te.append(np.mean(min_rmse_te))
        global_w.append(np.mean(w_k, axis=0))
        global_lambdas.append(np.mean(lambdas))

    visu.cross_validation_visualization(degree, global_min_tr, global_min_te)
    return global_min_tr, global_min_te, global_w


def cross_validation_logistic_demo_fixedparams(y, x, lambda_):
    """

        :param y:
        :param x:
        :return:
    """
    seed = 12
    k_fold = 5
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data

    degree = np.array([2])
    global_min_tr = []
    global_min_te = []
    global_w = []
    global_lambdas = []

    for deg in degree:

        min_rmse_tr = []
        min_rmse_te = []
        w_k = []
        lambdas = []
        for k in range(k_fold):
            loss_tr, loss_te, w = cross_validation_logistic(y, x, k_indices, k, 2, lambda_)
            max_lambda = 0
            max_loss_te = 0
            max_loss_tr = 0
            max_w = []
            for lam in np.array([lambda_]):
                if loss_te > max_loss_te:
                    max_loss_te = loss_te
                    max_lambda = lam
                    max_loss_tr = loss_tr
                    max_w = w
            print('DEGREE', deg, 'lambda', max_lambda, 'accuracy', max_loss_te)
            min_rmse_tr.append(max_loss_tr)
            min_rmse_te.append(max_loss_te)
            w_k.append(max_w)
            lambdas.append(max_lambda)

        global_min_tr.append(np.mean(min_rmse_tr))
        global_min_te.append(np.mean(min_rmse_te))
        global_w.append(np.mean(w_k, axis=0))
        global_lambdas.append(np.mean(lambdas))

    visu.cross_validation_visualization(degree, global_min_tr, global_min_te)
    return global_min_tr, global_min_te, global_w
