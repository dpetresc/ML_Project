# -*- coding: utf-8 -*-
"""This file contains visualisation functions depending on the regression techniques used."""
import matplotlib.pyplot as plt

def cross_validation_visualization(lambds, mse_tr, mse_te, abscisse):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel(abscisse)
    plt.ylabel("rmse")
    plt.ylim([0.0, 2])
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
