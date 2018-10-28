# -*- coding: utf-8 -*-
"""Functions used to run cross validation for our models and to plot the obtain results."""
import numpy as np
from plot import *
from helpers import *
from proj1_helpers import *
from regressions import *
from ipywidgets import IntProgress
from IPython.display import display
from preprocessing import *
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
	
	############# TRY NEW THING #############################
	jets_train = get_jet_masks(X_train)
	jets_test = get_jet_masks(X_test)
	y_train_pred = np.zeros(len(y_train))
	y_test_pred = np.zeros(len(y_test))
	X_train, X_test = process_data2(X_train, X_test, inv_log=True)
	ws = 0
    
	for idx in range(len(jets_train)):
	    x_tr = X_train[jets_train[idx]]
	    x_te = X_test[jets_test[idx]]
	    y_tr = y_train[jets_train[idx]]
	    w, loss = regression_technique(y=y_tr, tx=x_tr, **args)
	    ws = w
	    y_train_pred[jets_train[idx]] = np.dot(x_tr, w)
	    y_test_pred[jets_test[idx]] = np.dot(x_te, w)


	loss_tr = np.sqrt(2 * mse(y_train - y_train_pred))
	loss_te = np.sqrt(2 * mse(y_test - y_test_pred))
	return loss_tr, loss_te, ws
	############## END OF TRIAL #############################
	    

	#********************** CHOOSE REGRESSION TECHNIQUE *****************
	#w, loss = regression_technique(y=y_train, tx=X_train, **args)
	#calculate the loss for train and test data
	#loss_tr = np.sqrt(2 * loss)
	#loss_te = np.sqrt(2 * compute_loss(y_test, X_test, w))


	#return loss_tr, loss_te, w

def cross_validation_demo(y, x, regression_technique, **args):
	f = IntProgress(min=0, max=30) # instantiate the bar
	display(f) # display the bar

	seed = 1
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
	return min_rmse_tr, min_rmse_te
    


# ******************************************************
# CROSS VALIDATION FUNCTIONS FOR RIDGE REGRESSION
# ******************************************************
def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
    
	#Build test and training set
	te_indice = k_indices[k]
	tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
	tr_indice = tr_indice.reshape(-1)

	#Train and test data
	y_test = y[te_indice]
	y_train = y[tr_indice]
	X_test = x[te_indice]
	X_train = x[tr_indice]

	############# TRY NEW THING #############################
	jets_train = get_jet_masks(X_train)
	jets_test = get_jet_masks(X_test)
	y_train_pred = np.zeros(len(y_train))
	y_test_pred = np.zeros(len(y_test))
	X_train, X_test = process_data2(X_train, X_test, inv_log=True)

	X_train = build_poly(X_train, degree)
	X_test = build_poly(X_test, degree)
	ws = [] #weight vectors for each jets
    
	for idx in range(len(jets_train)):
		x_tr = X_train[jets_train[idx]]
		x_te = X_test[jets_test[idx]]
		y_tr = y_train[jets_train[idx]]
		
		#tx_training = build_poly(x_tr, degree)
		#tx_test = build_poly(x_te, degree)
		tx_training = x_tr
		tx_test = x_te

		#tx_training = np.hstack((np.ones((tx_training.shape[0], 1)), tx_training))
		#tx_test = np.hstack((np.ones((tx_test.shape[0], 1)), tx_test))

		
		w, loss = ridge_regression(y_tr, tx_training, lambda_)
		ws.append(w)
		y_train_pred[jets_train[idx]] = np.dot(tx_training, w)
		y_test_pred[jets_test[idx]] = np.dot(tx_test, w)
	
	
	loss_tr = np.sqrt(2 * mse(y_train - y_train_pred))
	loss_te = np.sqrt(2 * mse(y_test - y_test_pred))
	return y_test_pred, loss_tr, loss_te, ws
	############## END OF TRIAL #############################
    
    #form data with polynomial degree
    #tx_training = build_poly(X_train, degree)
    #tx_test = build_poly(X_test, degree)

    #********************** RIDGE REGRESSION *****************
    #w, loss = ridge_regression(y_train, tx_training, lambda_)

    #calculate the loss for train and test data
    #loss_tr = np.sqrt(2 * loss)
    #loss_te = np.sqrt(2 * compute_loss(y_test, tx_test, w))
    
    
    #return loss_tr, loss_te, w

def cross_validation_ridge_demo(y, x):
	f = IntProgress(min=0, max=90) # instantiate the bar
	display(f) # display the bar
	seed = 12
	k_fold = 5
	lambdas = np.logspace(-4, 0, 30)
	# split data in k fold
	k_indices = build_k_indices(y, k_fold, seed)
	# define lists to store the loss of training data and test data
     
	count = 0
	degree = [2, 7, 9]
	global_min_tr = []
	global_min_te = []
	best_lambdas = []
	global_w = []
	global_y_pred = []
	for deg in degree:
		rmse_tr = []
		rmse_te = []
		w_list = []
		y_pred_list = []
		for lambda_ in lambdas :
			min_rmse_tr = []
			min_rmse_te = []
			w_k = []
			y_pred_k = []
			for k in range(k_fold) :
				y_pred_temp, loss_tr, loss_te, ws = cross_validation_ridge(y, x, k_indices, k,lambda_ , deg)
				min_rmse_tr.append(loss_tr)
				min_rmse_te.append(loss_te)
				w_k.append(ws)
				y_pred_k.append(y_pred_temp)
			rmse_tr.append(np.mean(min_rmse_tr))
			rmse_te.append(np.mean(min_rmse_te))
			#Compute the mean of the 3 vectors ws
			w_0 = np.zeros(shape=(w_k[0])[0].shape)
			w_1 = np.zeros(shape=(w_k[0])[0].shape)
			w_2 = np.zeros(shape=(w_k[0])[0].shape)
			N = len(w_k)
			for i in range(len(w_k)) :
				w_0 += (w_k[i])[0] / N
				w_1 += (w_k[i])[1] / N
				w_2 += (w_k[i])[2] / N
			w_jets = [w_0, w_1, w_2]
			#print((w_k[0])[2])
			#print((w_k[0])[3])
			w_list.append(w_jets)
			y_pred_list.append(np.mean(y_pred_k, axis=0))
			f.value += 1 # signal to increment the progress bar
			time.sleep(.1)
			count += 1
            
		indice = np.argmin(rmse_te)
		global_min_tr.append(rmse_tr[indice])
		global_min_te.append(rmse_te[indice])
		best_lambdas.append(lambdas[indice])
		global_w.append(w_list[indice])
		global_y_pred.append(y_pred_list[indice])

        
	#f.value += 1 # signal to increment the progress bar
	#time.sleep(.1)
	#count += 1

	cross_validation_visualization(degree, global_min_tr, global_min_te, 'degrees')
	return global_y_pred, best_lambdas, global_min_tr, global_min_te, global_w


# ******************************************************
# CROSS VALIDATION FUNCTIONS FOR LOGISTIC REGRESSION
# ******************************************************




def predict_test(X_test, ids_test, ws):


	############# TRY NEW THING #############################
	jets_test = get_jet_masks(X_test)
	
	#lambda_ = 0.0001
	degree = 2
	
	y_pred = np.zeros(X_test.shape[0])
	x_te = process_data3(X_test, inv_log=True)
	tx_test = build_poly(x_te, degree)
    
	for idx in range(len(jets_test)):
	
		x_te = tx_test[jets_test[idx]]				
		

		y_test_pred = predict_labels(ws[idx], x_te)
		y_pred[jets_test[idx]] = y_test_pred

	return ids_test, y_pred



