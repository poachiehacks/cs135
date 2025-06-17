import os
import numpy as np
import pandas as pd

import warnings

import sklearn.linear_model
import sklearn.metrics
import sklearn.calibration
import sklearn.preprocessing

from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8') # pretty matplotlib plots



def calc_confusion_matrix_for_threshold(ytrue_N, yproba1_N, thresh=0.5):
    ''' Compute the confusion matrix for a given probabilistic classifier and threshold
    
    Args
    ----
    ytrue_N : 1D array of floats
        Each entry represents the binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    yproba1_N : 1D array of floats
        Each entry represents a probability (between 0 and 1) that correct label is positive (1)
        One entry per example in current dataset
        Needs to be same size as ytrue_N
    thresh : float
        Scalar threshold for converting probabilities into hard decisions
        Calls an example "positive" if yproba1 >= thresh
        Default value reflects a majority-classification approach (class is the one that gets
        highest probability)

    Returns
    -------
    cm_df : Pandas DataFrame
        Can be printed like print(cm_df) to easily display results
    '''
    cm = sklearn.metrics.confusion_matrix(ytrue_N, yproba1_N >= thresh)
    cm_df = pd.DataFrame(data=cm, columns=[0, 1], index=[0, 1])
    cm_df.columns.name = 'Predicted'
    cm_df.index.name = 'True'
    return cm_df



def calc_TP_TN_FP_FN(ytrue_N, yhat_N):
    '''
    
    Args
    ----
    ytrue_N : 1D array of floats
        Each entry represents the binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats
        Each entry represents a predicted binary value (either 0 or 1).
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    TP : int
        Number of true positives
    TN : int
        Number of true negatives
    FP : int
        Number of false positives
    FN : int
        Number of false negatives
    '''
    
    # TP = model says true, reality is true
    isTP = np.logical_and(np.equal(ytrue_N, 1), np.equal(yhat_N, 1))
    isTN = np.logical_and(np.equal(ytrue_N, 0), np.equal(yhat_N, 0))
    isFP = np.logical_and(np.equal(ytrue_N, 0), np.equal(yhat_N, 1))
    isFN = np.logical_and(np.equal(ytrue_N, 1), np.equal(yhat_N, 0))
    TP = np.sum(isTP)
    TN = np.sum(isTN)
    FP = np.sum(isFP)
    FN = np.sum(isFN)

    return TP, TN, FP, FN


### Problem 1 ###
def problem_one():
    all0 = np.zeros(10)
    all1 = np.ones(10)

    test_true = np.array([0, 1, 0, 1, 0, 1])
    test_pred = np.array([1, 0, 0, 1, 1, 0])
    
    
    ### testing calls
    calc_TP_TN_FP_FN(all0, all1)
    calc_TP_TN_FP_FN(all1, all0)
    calc_TP_TN_FP_FN(all1, all1)
    calc_TP_TN_FP_FN(all0, all0)

    calc_TP_TN_FP_FN(test_true, test_pred)



def load_data():
    """
    input data has three columns: age, famhistory, marker

    output data has 1 column: cancer
    """
    # Load the x-data and y-class arrays
    x_train = np.loadtxt('./data_cancer/x_train.csv', delimiter=',', skiprows=1)
    x_test = np.loadtxt('./data_cancer/x_test.csv', delimiter=',', skiprows=1)

    y_train = np.loadtxt('./data_cancer/y_train.csv', delimiter=',', skiprows=1)
    y_test = np.loadtxt('./data_cancer/y_test.csv', delimiter=',', skiprows=1)

    return x_train, x_test, y_train, y_test


### Problem 2 ###
def problem_two():

    x_train, x_test, y_train, y_test = load_data()

    cancer_frac_train = np.sum(y_train) / len(y_train)
    cancer_frac_test = np.sum(y_test) / len(y_test)
    print("Fraction of data that has_cancer on TRAIN: %.3f" % cancer_frac_train) #TODO: modify these prints
    print("fraction of data that has_cancer on TEST : %.3f" % cancer_frac_test)


def always_zero_classifier(x_test):
    """
    In the spirit of keeping up an image, the function must take in input data.
    But it only needs it to return an output of appropriate size
    """
    return np.zeros(len(x_test))


def calc_accuracy(pred, truth):

    is_correct = np.equal(pred, truth)
    num_correct = np.sum(is_correct)
    accuracy = num_correct / len(truth)

    return accuracy


### Problem 3 ###
def problem_three():

    x_train, x_test, y_train, y_test = load_data()

    has_cancer_test = always_zero_classifier(x_test)
    acc_test = calc_accuracy(has_cancer_test, y_test)
    
    has_cancer_train = always_zero_classifier(x_train)
    acc_train = calc_accuracy(has_cancer_train, y_train)
    
    print("acc on TRAIN: %.3f" % acc_train) #TODO: modify these values
    print("acc on TEST : %.3f" % acc_test)

    # TODO: call print(calc_confusion_matrix_for_threshold(...))
    y_prob1 = np.zeros(len(y_test))
    cm_df = calc_confusion_matrix_for_threshold(y_test, y_prob1)
    print(cm_df)


def problem_four():

    x_train, x_test, y_train, y_test = load_data()
    scaler = sklearn.preprocessing.MinMaxScaler()

    # scale input data to [0, 1]
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)


    perceptron = sklearn.linear_model.Perceptron()
    perceptron.fit(x_train, y_train)

    score_train = perceptron.score(x_train, y_train)
    print("acc on TRAIN: %.3f" % score_train) #TODO: modify these values
    
    score_test = perceptron.score(x_test, y_test)
    print("acc on TEST : %.3f" % score_test)
    print('')

    # confusion matrix
    pred = perceptron.predict(x_test)
    cm_df = calc_confusion_matrix_for_threshold(y_test, pred)
    print(cm_df)


    # # 2D plots
    # plt.figure()
    # plt.title('has_cancer vs. feature')
    # plt.scatter(x_train[:, 0], y_train, label='age')
    # plt.scatter(x_train[:, 1], y_train, label='famhistory')
    # plt.scatter(x_train[:, 2], y_train, label='marker')
    # plt.legend()
    # plt.xlabel('feature (normalized)')
    # plt.ylabel('has_cancer')


    # # 3D plots
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x_train[:, 0], x_train[:, 1], y_train, label='age, famhistory')
    # ax.scatter(x_train[:, 0], x_train[:, 2], y_train, label='age, marker')
    # ax.scatter(x_train[:, 2], x_train[:, 0], y_train, label='marker, age')
    # ax.scatter(x_train[:, 2], x_train[:, 1], y_train, label='marker, famhistory')
    # ax.legend()
    # ax.set_xlabel('feature 1')
    # ax.set_ylabel('feature 2')
    # ax.set_zlabel('has_cancer')

    # # 3D plots
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # has_cancer = np.equal(y_train, 1)
    # no_cancer  = np.logical_not(has_cancer)
    # ax.scatter(x_train[has_cancer, 0], x_train[has_cancer, 1], x_train[has_cancer, 2], label='cancer')
    # ax.scatter(x_train[no_cancer, 0], x_train[no_cancer, 1], x_train[no_cancer, 2], label='noncancer')
    # ax.legend()
    # ax.set_xlabel('age')
    # ax.set_ylabel('famhistory')
    # ax.set_zlabel('marker')

    # y_preds = x_train @ perceptron.coef_.T + perceptron.intercept_
    # says_yes_cancer = np.greater_equal(y_preds, 0).reshape(-1)
    # says_no_cancer  = np.less(y_preds, 0).reshape(-1)
    # print(says_yes_cancer)
    # print(says_no_cancer)
    # # 3D plots
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x_train[says_yes_cancer, 0], x_train[says_yes_cancer, 1], x_train[says_yes_cancer, 2], label='says yes cancer', color='red')
    # ax.scatter(x_train[says_no_cancer, 0], x_train[says_no_cancer, 1], x_train[says_no_cancer, 2], label='says no cancer', color='orange')
    # ax.legend()
    # ax.set_xlabel('age')
    # ax.set_ylabel('famhistory')
    # ax.set_zlabel('marker')



    alphas = np.logspace(-5, 5, base=10, num=100)
    train_accuracy_list = np.zeros_like(alphas)
    test_accuracy_list = np.zeros_like(alphas)

    for ii, alpha in enumerate(alphas):

        reg_perceptron = sklearn.linear_model.Perceptron(penalty='l2', alpha=alpha)
        reg_perceptron.fit(x_train, y_train)

        score_train = reg_perceptron.score(x_train, y_train)
        score_test = reg_perceptron.score(x_test, y_test)
        
        train_accuracy_list[ii] = score_train
        test_accuracy_list[ii] = score_test


    plt.figure()
    plt.xscale('log')
    plt.plot(alphas, train_accuracy_list, label='train')
    plt.plot(alphas, test_accuracy_list, label='test')
    plt.legend()


    plt.show()



def problem_five():

    x_train, x_test, y_train, y_test = load_data()
    scaler = sklearn.preprocessing.MinMaxScaler()

    # scale input data to [0, 1]
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    perceptron = sklearn.linear_model.Perceptron()
    perceptron.fit(x_train, y_train)


    confidence_scores = perceptron.decision_function(x_test)

    cccv = sklearn.calibration.CalibratedClassifierCV(perceptron, method='isotonic')
    cccv.fit(x_train, y_train)
    probs = cccv.predict_proba(x_test)


    fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(y_test, confidence_scores)
    auc_confs = sklearn.metrics.roc_auc_score(y_test, confidence_scores)

    plt.figure(figsize=(8,8))
    plt.title('ROC curves')
    plt.scatter(fpr, tpr, label='confidence model')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')

    # y_score needs to be the probabilities for the positive class
    fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(y_test, probs[:, 1])
    auc_probs = sklearn.metrics.roc_auc_score(y_test, probs[:, 1])
    plt.scatter(fpr, tpr, label='probabilities model')
    plt.legend()
    

    print("AUC on TEST for Perceptron: %.3f" % auc_confs) #TODO: modify these values
    print("AUC on TEST for probabilistic model: %.3f" % auc_probs)
    plt.show()


def calc_perf_metrics_for_threshold(ytrue_N, yproba1_N, thresh=0.5):
    ''' Compute performance metrics for a given probabilistic classifier and threshold
    Args
    ----
    ytrue_N : 1D array of floats
        Each entry represents the binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    yproba1_N : 1D array of floats
        Each entry represents a probability (between 0 and 1) that correct label is positive (1)
        One entry per example in current dataset
        Needs to be same size as ytrue_N
    thresh : float
        Scalar threshold for converting probabilities into hard decisions
        Calls an example "positive" if yproba1 >= thresh
        Default value reflects a majority-classification approach (class is the one that gets
        highest probability)

    Returns
    -------
    acc : accuracy of predictions
    tpr : true positive rate of predictions
    tnr : true negative rate of predictions
    ppv : positive predictive value of predictions
    npv : negative predictive value of predictions
    '''

    # TODO: fix this
    acc = 0
    tpr = 0
    tnr = 0
    ppv = 0
    npv = 0


    cm_df = calc_confusion_matrix_for_threshold(ytrue_N, yproba1_N, thresh)
    TP = cm_df.loc[1, 1]
    FP = cm_df.loc[0, 1]
    FN = cm_df.loc[1, 0]
    TN = cm_df.loc[0, 0]
    acc = (TP + TN) / len(ytrue_N)
    tpr = TP / (TP + FN) if TP + FN != 0 else 0
    tnr = TN / (TN + FP) if TN + FP != 0 else 0
    ppv = TP / (TP + FP) if TP + FP != 0 else 0
    npv = TN / (TN + FN) if TN + FN != 0 else 0

    return acc, tpr, tnr, ppv, npv


# You can use this function later to make printing results easier; don't change it.
def print_perf_metrics_for_threshold(ytrue_N, yproba1_N, thresh=0.5):
    ''' Pretty print perf. metrics for a given probabilistic classifier and threshold
    '''
    acc, tpr, tnr, ppv, npv = calc_perf_metrics_for_threshold(ytrue_N, yproba1_N, thresh)
    
    ## Pretty print the results
    print("%.3f ACC" % acc)
    print("%.3f TPR" % tpr)
    print("%.3f TNR" % tnr)
    print("%.3f PPV" % ppv)
    print("%.3f NPV" % npv)


def problem_five_cde():

    x_train, x_test, y_train, y_test = load_data()
    scaler = sklearn.preprocessing.MinMaxScaler()

    # scale input data to [0, 1]
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    perceptron = sklearn.linear_model.Perceptron()
    perceptron.fit(x_train, y_train)
    cccv = sklearn.calibration.CalibratedClassifierCV(perceptron, method='isotonic')
    cccv.fit(x_train, y_train)
    probs = cccv.predict_proba(x_test)
    neg_probs = probs[:, 0]
    pos_probs = probs[:, 1]

    thresholds = np.linspace(0, 1.001, 51)
    accs = np.zeros_like(thresholds)
    tprs = np.zeros_like(thresholds)
    tnrs = np.zeros_like(thresholds)
    ppvs = np.zeros_like(thresholds)
    npvs = np.zeros_like(thresholds)

    best_TPR = 0
    best_PPV_for_best_TPR = 0
    best_TPR_threshold = 0
    
    best_PPV = 0
    best_TPR_for_best_PPV = 0  
    best_PPV_threshold = 0

    for ii, thresh in enumerate(thresholds):
        
        acc, tpr, tnr, ppv, npv = calc_perf_metrics_for_threshold(y_test, pos_probs,thresh=thresh)
        accs[ii] = acc
        tprs[ii] = tpr
        tnrs[ii] = tnr
        ppvs[ii] = ppv
        npvs[ii] = npv

        if tpr > best_TPR:
            best_TPR = tpr
            best_PPV_for_best_TPR = ppv
            best_TPR_threshold = thresh
        elif tpr == best_TPR:
            if ppv > best_PPV_for_best_TPR:
                best_TPR = tpr
                best_PPV_for_best_TPR = ppv
                best_TPR_threshold = thresh

        if ppv > best_PPV:
            best_PPV = ppv
            best_TPR_for_best_PPV = tpr
            best_PPV_threshold = thresh
        elif ppv == best_PPV:
            if tpr > best_TPR_for_best_PPV:
                best_PPV = ppv
                best_TPR_for_best_PPV = tpr
                best_PPV_threshold = thresh


    print("TPR threshold: %.4f => TPR: %.4f; PPV: %.4f" % (best_TPR_threshold, best_TPR, best_PPV_for_best_TPR))
    print("PPV threshold: %.4f => PPV: %.4f; TPR: %.4f" % (best_PPV_threshold, best_PPV, best_TPR_for_best_PPV))

    print()

    best_thr = 0.5
    print("ON THE TEST SET:")
    print("Chosen best threshold = %.4f" % best_thr)
    print("")
    print(calc_confusion_matrix_for_threshold(y_test, pos_probs, thresh=best_thr))
    print("")
    print_perf_metrics_for_threshold(y_test, pos_probs, best_thr)

    best_thr = best_TPR_threshold
    print("ON THE TEST SET:")
    print("Chosen best threshold = %.4f" % best_thr)
    print("")
    print(calc_confusion_matrix_for_threshold(y_test, pos_probs, thresh=best_thr))
    print("")
    print_perf_metrics_for_threshold(y_test, pos_probs, best_thr)

    best_thr = best_PPV_threshold
    print("ON THE TEST SET:")
    print("Chosen best threshold = %.4f" % best_thr)
    print("")
    print(calc_confusion_matrix_for_threshold(y_test, pos_probs, thresh=best_thr))
    print("")
    print_perf_metrics_for_threshold(y_test, pos_probs, best_thr)



if __name__ == "__main__":

    # problem_one()
    # problem_two()
    # problem_three()
    # problem_four()
    # problem_five()
    problem_five_cde()