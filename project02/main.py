import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


def load_training_data():
    print('hello')

    x_train_df = pd.read_csv('data/data_reviews/x_train.csv')
    y_train_df = pd.read_csv('data/data_reviews/y_train.csv')

    return x_train_df, y_train_df


def preprocess(x_df):
    
    x_new_df = x_df.copy()

    # process each item individually
    tr_text_list = x_df['text'].values.tolist()
    for ii, text in enumerate(tr_text_list):
        tr_text_list[ii] = text.lower()
        # print(tr_text_list[ii])
    

    # put back into df
    x_new_df['text'] = tr_text_list
    
    return x_new_df 


if __name__ == "__main__":
    x_train_df, y_train_df = load_training_data()

    # it's simplest to preprocess data before doing split into folds
    x_train = preprocess(x_train_df)
    y_train = y_train_df

    
    # perform validation by using cross-validation
    # make sure the data is formatted in such a way that public cross-validation functions can easily use
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=420)
    for train_idx, valid_idx in skf.split(x_train_df, y_train_df):
        fold_x_train, fold_x_valid = x_train.iloc[train_idx], x_train.iloc[valid_idx]
        fold_y_train, fold_y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
        