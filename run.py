# -*- coding: utf-8 -*-
"""Creates the prediction independantly of the project
(some methods are rewritten here to have all in the same file)."""

import sys
sys.path.insert(0, 'scripts')
import numpy as np
import implementations as imp
import model_selection as modselection
import proj1_helpers as helpers
import helpers_us as helpers_us


def process_data(path, inv_log=False):
    """Process the data before using it doing some engineering featuring

        :param path: path of the dataset
        :param inv_log: apply log on the positive columns of the dataset
        :return: y, processed data, masks based on pri_jet_num, ids
    """
    y, X, ids = helpers.load_csv_data(path)

    dict_mask_jets_train = helpers_us.get_jet_masks(X)

    new_X = []

    for i in range(len(dict_mask_jets_train)):
        new_X.append(np.delete(X[dict_mask_jets_train[i]], [22, 29], axis=1))

    for i in range(len(dict_mask_jets_train)):
        undefined_columns = [j for j in range(len(new_X[i][0])) if (new_X[i][:, j] < -900).all()]
        new_X[i] = np.delete(new_X[i], undefined_columns, axis=1)

    for i in range(len(dict_mask_jets_train)):
        for j in range(len(new_X[i][0])):
            col = new_X[i][:, j]
            np.where(col < -900)
            m = np.mean(col[col >= -900])  # compute mean of the right columns
            col[np.where(col < -900)] = m
            new_X[i][:, j] = col

    if inv_log:
        new_X = helpers_us.log_f(new_X)

    for i in range(1, len(dict_mask_jets_train)):
        new_X[i], x_mean, x_std = helpers_us.standardize(new_X[i])

    return y, new_X, dict_mask_jets_train, ids


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
    tx_train = modselection.build_poly(X_train, degree)
    tx_test = modselection.build_poly(X_test, degree)

    # ridge regression
    w, loss = imp.ridge_regression(y_train, tx_train, lambda_)

    # calculate the loss for train and test data
    loss_train = imp.calculate_rmse(loss)
    loss_test = imp.calculate_rmse(imp.compute_loss(y_test, tx_test, w))
    accuracy = modselection.calculate_accuracy(y_test, helpers.predict_labels(w, tx_test))
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
    k_indices = modselection.build_k_indices(y, k_fold, seed)
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

    ind_best_degree = np.argmin(best_rmses)
    best_lambda = best_lambdas[ind_best_degree]
    best_degree = degrees[ind_best_degree]
    accuracy = best_accuracies[ind_best_degree]
    return best_degree, best_lambda

def create_prediction():
    """Create predictions for kaggle."""
    y, X, dict_mask_jets_train, ids = helpers_us.process_data('Data/train.csv', inv_log=True)
    best_degrees = []
    best_lambdas = []
    for i in range(len(dict_mask_jets_train)):
        best_degree, best_lambda = best_model_ridge(y[dict_mask_jets_train[i]], X[i], 5, np.arange(2,5), np.logspace(-6, 0, 15), seed=56)
        best_degrees.append(best_degree)
        best_lambdas.append(best_lambda)

    best_weights = []
    for i in range(len(dict_mask_jets_train)):
        xi = X[i]
        yi = y[dict_mask_jets_train[i]]

        xi = modselection.build_poly(xi, best_degrees[i])
        w, _ = imp.ridge_regression(yi, xi, best_lambdas[i])
        best_weights.append(w)

    y, X, dict_mask_jets_train, ids = helpers_us.process_data('Data/test.csv', inv_log=True)
    y_pred = np.zeros(y.shape[0])

    for i in range(len(dict_mask_jets_train)):
        xi = X[i]
        xi = modselection.build_poly(xi, best_degrees[i])
        y_test_pred = helpers.predict_labels(best_weights[i], xi)
        y_pred[dict_mask_jets_train[i]] = y_test_pred
    helpers.create_csv_submission(ids, y_pred, "prediction.csv")


if __name__ == '__main__':
    create_prediction()
