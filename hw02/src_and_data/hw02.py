# import libraries as needed
from pprint import pprint

import numpy as np
import pandas as pd
import math

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8') # pretty matplotlib plots


def plot_predictions(polynomials=list(), prediction_list=list(), error_list=list(), x=None, y=None):
    '''Plot predicted results for a number of polynomial regression models
    
    Args
    ----
    polynomials : list of positive integer values
        Each value is the degree of a polynomial regression model.
    prediction_list: list of arrays ((# polynomial models) x (# input data))
        Each array contains the predicted y-values for input data.
    error_list: list of error values ((# polynomial models) x 1)
        Each value is the mean squared error (MSE) of the model with 
        the associated polynomial degree.
    
        Note: it is expected that all lists are of the same length, and 
            that this length be some perfect square (for grid-plotting).
    '''
    length = len(prediction_list)
    grid_size = int(math.sqrt(length))
    if not (length == len(polynomials) and length == len(error_list)):
        raise ValueError("Input lists must be of same length")
    if not length == (grid_size * grid_size):
        raise ValueError("Need a square number of list items (%d given)" % (length))
    
    fig, axs = plt.subplots(grid_size, grid_size, figsize =(14,14), sharey=True)
    for subplot_id, prediction in enumerate(prediction_list):
        # order data for display
        data_frame = pd.DataFrame(data=[x[:, 0], prediction]).T
        data_frame = data_frame.sort_values(by=0)
        x_sorted = data_frame.iloc[:, :-1].values
        prediction_sorted = data_frame.iloc[:, 1].values

        ax = axs.flat[subplot_id]
        ax.set_title('degree = %d; MSE = %.3f' % (polynomials[subplot_id], error_list[subplot_id]))
        ax.plot(x, y, 'r.')
        ax.plot(x_sorted, prediction_sorted, color='blue')
        
    plt.show()


# A simple function for generating different data-folds.
#
# DO NOT MODIFY THIS CODE.
def make_folds(x_data, y_data, num_folds=1):
    '''Splits data into num_folds separate folds for cross-validation.
       Each fold should consist of M consecutive items from the
       original data; each fold should be the same size (we will assume 
       that  the data divides evenly by num_folds).  Every data item should 
       appear in exactly one fold.
       
       Args
       ----
       x_data: input data.
       y_data: matching output data.
           (Expected that these are of the same length.)
       num_folds : some positive integer value
           Number of folds to divide data into.
           
        Returns
        -------
        x_folds : list of sub-sequences of original x_data 
            There will be num_folds such sequences; each will 
            consist of 1/num_folds of the original data, in
            the original order.
        y_folds : list of sub-sequences of original y_data
            There will be num_folds such sequences; each will 
            consist of 1/num_folds of the original data, in
            the original order.
       '''
    x_folds = list()
    y_folds = list()
    
    foldLength = (int)(len(x_data) / num_folds)
    start = 0
    for fold in range(num_folds):
        end = start + foldLength
        x_folds.append(x_data[start:end])
        y_folds.append(y_data[start:end])
        start = start + foldLength
        
    return x_folds, y_folds


def test_polynomials(polynomials=list(), xdata=np.zeros((0,0)), ydata=np.zeros((0,0))):
    '''Generates a series of polynomial regression models on input data.
       Each model is fit to the data, then used to predict values of that
       input data.  Predictions and mean squared error are collected and
       returned as two lists.
    
    Args
    ----
    polynomials : list of positive integer values
        Each value is the degree of a polynomial regression model, to be built.
    
    Returns
    -------
    prediction_list: list of arrays ((# polynomial models) x (# input data))
        Each array contains the predicted y-values for input data.
    error_list: list of error values ((# polynomial models) x 1)
        Each value is the mean squared error (MSE) of the model with 
        the associated polynomial degree.
    '''
    prediction_list = list()
    error_list = list()
    coeff_list = list()

    # TODO: fill in this function to generate the required set of models,
    #       returning the predictions and the errors for each.
    for degree in polynomials:
        
        # transform data
        # NOTE: PF.fit() doesn't fit a model to data
        #       it's just a step required by sklearn
        #       before i can use PF.transform()
        model = PolynomialFeatures(degree=degree)
        model.fit(xdata)
        transformed = model.transform(xdata)
        
        # fit a model to the transformed data
        reg = linear_model.LinearRegression()
        reg.fit(transformed, ydata)
        print(transformed[0:3,:])
        print('')
        
        # compute estimated predictions
        predictions = transformed @ reg.coef_ + reg.intercept_
        
        # compute MSE
        residuals = ydata - predictions
        sq_residuals = np.square(residuals)
        error = np.mean(sq_residuals)

        # save return values of function
        prediction_list.append(predictions)
        error_list.append(error)
        coeff_list.append(reg.coef_)
        

    return prediction_list, error_list, coeff_list
    



# TODO: Perform 5-fold cross-validation for each polynomial degree.  
#       Keep track of average training/test error for each degree; 
#       Plot results in a single table, properly labeled, and also
#       print out the results in some clear tabular format.

def _folds_to_train_test(folds, test_fold_idx):
    """
    selects one fold out of k folds to be the test data
    and uses the other k-1 folds for training data
    """
    k = len(folds)
    train_indices = [ii for ii in range(k) if ii != test_fold_idx]
    test_indices  = test_fold_idx
    train_folds = [folds[ii] for ii in train_indices]
    test_folds  = [folds[test_indices]]

    train_data = np.concatenate(train_folds, axis=0)
    test_data  = np.concatenate(test_folds, axis=0)

    return train_data, test_data



def polynomial_fit(degree, x_train_data, y_train_data, regtype='none', alpha=1):
    """
    Returns a model trained on labeled data using polynomial regression
    and returns the transformed features

    :param degree - degree of the polynomial for regression
    :param x_train_data - original feature data
    :param y_train_data - labels for the features
    :param regtype - type of regularization to apply to regression [none, ridge]

    :return reg - an sklearn.LinearRegression object trained on the provided labeled data
    :return transformed - the transformed feature data as specified by degree.
    """        
    # transform data
    # NOTE: PF.fit() doesn't fit a model to data
    #       it's just a step required by sklearn
    #       before i can use PF.transform()
    model = PolynomialFeatures(degree=degree)
    model.fit(x_train_data)
    transformed = model.transform(x_train_data)
    
    # choose sklearn model appropriate for the regularization type
    if regtype == 'ridge':
        reg = linear_model.Ridge(alpha=alpha)
    else:
        reg = linear_model.LinearRegression()
    
    # fit a model to the transformed data
    reg.fit(transformed, y_train_data)
        
    return reg, transformed


def kfold_cross_validation(x, y, degree, k, regtype='none', alpha=1):
    """
    :param x - predictive data
    :param y - output data
    :param degree - degree of the polynomial regression to perform
    :param k - number of folds into which to split the data

    :return avg_train_mse - average training MSE across all training/test data splits
    :return avg_test_mse  - average testing MSE across all training/test data splits
    """
    # split data into folds
    xfolds, yfolds = make_folds(x, y, k)

    train_mses = np.zeros(k)
    test_mses  = np.zeros(k)
    
    # iterate through all possible choices of train/test data splits
    for n in range(k):
        x_train_data, x_test_data = _folds_to_train_test(xfolds, n)
        y_train_data, y_test_data = _folds_to_train_test(yfolds, n)

        # fit a model to the training data
        reg, train_transformed = polynomial_fit(degree, x_train_data, y_train_data, regtype=regtype, alpha=alpha)

        # compute training MSE - method 1
        train_predictions = reg.predict(train_transformed)
        train_residuals = y_train_data - train_predictions
        train_mse = np.mean(train_residuals**2)

        # compute testing MSE
        test_transformed = PolynomialFeatures(degree=degree).fit(x_test_data).transform(x_test_data)
        test_predictions = reg.predict(test_transformed)
        test_residuals = y_test_data - test_predictions
        test_mse = np.mean(test_residuals**2)

        train_mses[n] = train_mse
        test_mses[n]  = test_mse

    avg_train_mse = np.mean(train_mses)
    avg_test_mse  = np.mean(test_mses)

    return avg_train_mse, avg_test_mse


def feature_importance(coeffs):

    # investigating the importance of each feature
    for coeff in coeffs:

        weight_allocation = np.abs(coeff)
        frac_allocation = weight_allocation / np.sum(weight_allocation)
        
        print(coeff)
        print(frac_allocation > 0.1)
        print(f'num significant features: {np.sum(frac_allocation > 0.1)}')
        print()


def plot_mse_vs_degree(mses, degrees, label):

    plt.plot(degrees, mses, label=label)
    plt.title('average mse vs polynomial degree')
    plt.xlabel('degree')
    plt.ylabel('average MSE')



def problem_1_1():

    data = pd.read_csv('data.csv')
    x = data.iloc[:, :-1].values
    y = data.iloc[:, 1].values

    # degrees = [1, 2, 3, 4, 5, 6, 10, 11, 12]
    degrees = [1, 2, 3, 12]
    
    predictions_list, errors_list, coeffs_list = test_polynomials(degrees, x, y)
    plot_predictions(degrees, predictions_list, errors_list, x, y)
    
    
    return


def problem_2_1():

    data = pd.read_csv('data.csv')
    x = data.iloc[:, :-1].values
    y = data.iloc[:, 1].values

    degrees = [1, 2, 3, 4, 5, 6, 10, 11, 12]
    k = 5

    # for each degree,
    # perform 5-fold cross validation on it
    # record the average training and testing MSE for the 5-fold cross validation
    # 
    # plot the average mse graphs against degree
    # - degrees on x-axis
    # - mse on y-axis (one line for testing mse, one line for training mse)
    
    train_mses = list()
    test_mses  = list()
    for degree in degrees:
        avg_train_mse, avg_test_mse = kfold_cross_validation(x, y, degree, k)
        train_mses.append(avg_train_mse)
        test_mses.append(avg_test_mse)        

    plt.figure()
    plot_mse_vs_degree(train_mses, degrees, 'training mse')
    plot_mse_vs_degree(test_mses, degrees, 'testing mse')
    plt.legend()

    # a second figure mirroring the first except we drop degree=1
    plt.figure()
    plot_mse_vs_degree(train_mses[1:], degrees[1:], 'training mse')
    plot_mse_vs_degree(test_mses[1:], degrees[1:], 'testing mse')
    plt.legend()

    print('training MSEs:')
    pprint(train_mses)
    print('')

    print('testing MSEs:')
    pprint(test_mses)
    print('')


def problem_3_1(x, y):
    """
    This problem investigates the purpose of regularization strength in 
    regression problems with regularization

    We run Ridge regression on the assignment-provided data using 50
    different values for `alpha`, the regularization strength
    """

    # obtained from inspection after running problem_2_1()
    best_degree = 3

    # choices for regularization strength, specified by assignment
    alphas = np.logspace(-2, 2, base=10, num=50)

    # number of folds to partition the data into
    k = 5

    # cross validation loop    
    num_alphas = len(alphas)
    train_mses = np.zeros((num_alphas))
    test_mses  = np.zeros((num_alphas))
    for ii, alpha in enumerate(alphas):
        avg_train_mse, avg_test_mse = kfold_cross_validation(x, y, best_degree, k, regtype="ridge", alpha=alpha)
        train_mses[ii] = avg_train_mse
        test_mses[ii]  = avg_test_mse


    # create dataframes for tabulating findings
    df_findings = pd.DataFrame({
        key: col 
        for key, col in zip(
            ['alpha', 'training mse', 'test_mses'], 
            [alphas, train_mses, test_mses]
        )
    })
        

    print('---tabulated alpha, training MSE, testing MSE---')
    print(df_findings)
    print('')


    # datapoint associated with minimum testing MSE
    min_idx = np.argmin(test_mses)
    min_dp = df_findings.iloc[min_idx]
    print('---datapoint associated with minimum testing MSE---')
    print(min_dp)


    plt.xscale('log')
    plt.plot(alphas, train_mses, label='training MSE', marker='*')
    plt.plot(alphas[min_idx], train_mses[min_idx], marker='o', color='r')
    plt.title('avg training MSE vs. reg. strength')
    plt.xlabel('regularization strength')
    plt.ylabel('average MSE')

    plt.plot(alphas, test_mses, label='testing MSE', marker='*')
    plt.plot(alphas[min_idx], test_mses[min_idx], label='min testing MSE', marker='o', color='r')
    plt.title('avg testing MSE vs. reg. strength')
    plt.xlabel('regularization strength')
    plt.ylabel('average MSE')
    plt.legend()

    return


if __name__ == "__main__":

    # load assignment-provided data
    data = pd.read_csv('data.csv')
    x = data.iloc[:, :-1].values
    y = data.iloc[:, 1].values

    # problem_1_1()
    # problem_2_1()
    problem_3_1(x, y)

    plt.show()