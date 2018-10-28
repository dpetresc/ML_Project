# -*- coding: utf-8 -*-
"""Functions used to compute the loss and weights."""
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import load_csv_data
import pandas as pd


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

def feature_selection(X) :
	df = pd.DataFrame(X)
	corr_matrix = df.corr().abs()
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
	to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
	new_df = df.drop(df.columns[to_drop], axis=1)
	new_X = new_df.as_matrix()
	#test = [0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26]
	return new_X

def column_weighting(X) :
    remove_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    w = 0.9
    for j in range(X.shape[1]) :
        if(j in remove_ind) :
            X[j] = (1-w)*X[j]
        else :
            X[j] = w*X[j]
    return X

def inv_log_f(x) :
	inv_log_cols = [0, 2, 5, 7, 9, 10, 13, 16, 19, 21, 23, 26]
	others = [1, 3, 5, 6, 8, 11, 12, 14, 15, 17, 18, 20, 22, 25, 27, 28, 29]
	#test = [0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26]
	# Create inverse log values of features which are positive in value.
	#x_inv_log_cols = np.log(1 / (1 + x[:, inv_log_cols]))
	x_inv_log_cols = np.log(x[:, inv_log_cols])
	#x[:, inv_log_cols] = np.log(x[:, inv_log_cols])
	#x[:, others] = np.log(x[:, others] + 1 - np.min(x[:, others]))

	#temp = x[:, others]
	#x_others = np.log(temp + 1 - np.min(temp))
	x_inv = np.hstack((x, x_inv_log_cols))
	#x_inv = np.hstack((x_inv_log_cols, x_others))
	#x_inv = np.hstack((x, x_inv))
	return x_inv


def process_data(path, select=False, weight_col=False, inv_log=False):
	y, X, ids = load_csv_data(path)
	new_X = X


	for j in range(len(new_X[0])) :
		col = new_X[:, j]
		m = np.mean(col[col >= -900]) #compute mean of the right columns
		#m = np.median(col[col >= -900]) #compute median of the right columns
		col[np.where(col < -900)] = m
		new_X[:, j] = col
	
	if(weight_col) :
		new_X = column_weighting(new_X)
	if(inv_log) :
		new_X = inv_log_f(new_X)
	
	if(select) :
		new_X = feature_selection(new_X)

	new_X, x_mean, x_std  = standardize(new_X)
	new_X = np.hstack((np.ones((new_X.shape[0], 1)), new_X))
	return y, new_X, x_mean, x_std, ids

def na(x):
    return np.any(x == -999)

def process_data2(x_train, x_test, select=False, weight_col=False, inv_log=False) :
	for i in range(x_train.shape[1]):
        	# If NA values in column
		if na(x_train[:, i]):
		    msk_train = (x_train[:, i] != -999.)
		    msk_test = (x_test[:, i] != -999.)
		    # Replace NA values with most frequent value
		    values, counts = np.unique(x_train[msk_train, i], return_counts=True)
		    # If there are values different from NA
		    if (len(values) > 1):
		        x_train[~msk_train, i] = values[np.argmax(counts)]
		        x_test[~msk_test, i] = values[np.argmax(counts)]
		    else:
		        x_train[~msk_train, i] = 0
		        x_test[~msk_test, i] = 0
	
	if(inv_log) :
		x_train = inv_log_f(x_train)
		x_test = inv_log_f(x_test)
		
	
	x_train,_, _  = standardize(x_train)
	x_test,_, _  = standardize(x_test)
	#x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
	#x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
	return x_train, x_test

def process_data3(x_test, select=False, weight_col=False, inv_log=False) :
	for i in range(x_test.shape[1]):
        	# If NA values in column
		if na(x_test[:, i]):
		    msk_test = (x_test[:, i] != -999.)
		    # Replace NA values with most frequent value
		    values, counts = np.unique(x_test[msk_test, i], return_counts=True)
		    # If there are values different from NA
		    if (len(values) > 1):
		        x_test[~msk_test, i] = values[np.argmax(counts)]
		    else:
		        x_test[~msk_test, i] = 0
	
	if(inv_log) :
		x_test = inv_log_f(x_test)
		
	
	x_test,_, _  = standardize(x_test)
	#x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
	#x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
	return x_test





    
