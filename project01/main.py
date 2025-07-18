"""
    Project 1 for CS135 Summer 2025

    We need to build a classifier for handwritten digits and for 
    discerning trouser from dresses using logistic regression.

    Okay, I need to remember what logistic regression is.
    It's a classifier, even though it has regression in the name.
    This is because of historical naming reasons (the statisticians
    got to the naming of it first). 


"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


if __name__ == "__main__":

    # read data
    folder = "data_digits_8_vs_9_noisy"
    x_test = pd.read_csv(Path(folder) / Path("x_test.csv"))
    x_train = pd.read_csv(Path(folder) / Path("x_train.csv"))
    y_test = pd.read_csv(Path(folder) / Path("y_test.csv"))
    y_train = pd.read_csv(Path(folder) / Path("y_train.csv"))


    max_iters = list(range(40))
    accuracies = []
    loglosses = []

    for max_iter in max_iters:

        # set up LogReg classifier object
        logreg = LogisticRegression(max_iter=max_iter)

        # fit data
        model = logreg.fit(x_train, y_train)

        # record important stats
        accuracies.append(model.score(x_train, y_train))

        y_train_pred = model.predict_proba(x_train)
        loglosses.append(log_loss(y_train, y_train_pred))

    plt.figure()
    plt.plot(max_iters, accuracies, label='accuracies')
    plt.legend()
    
    plt.figure()
    plt.plot(max_iters, loglosses, label='log losses')
    plt.legend()
    plt.show()

