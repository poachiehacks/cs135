import numpy as np
import pandas as pd
import string
import re
import unicodedata

from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_extraction
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


def load_training_data():
    
    x_train_df = pd.read_csv('data/data_reviews/x_train.csv')
    y_train_df = pd.read_csv('data/data_reviews/y_train.csv')

    return x_train_df, y_train_df


def preprocess(x_df):
    
    x_new_df = x_df.copy()

    # process each item individually
    tr_text_list = x_df['text'].values.tolist()
    for ii, text in enumerate(tr_text_list):
        
        # lowercase
        text = text.lower()

        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # remove non-english characters (basically keep ASCII)
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)

        # there are non-text characters too? oh, maybe they mean just numbers
        words = [word for word in text.split() if not word.isdigit()]
        text = " ".join(words)

        tr_text_list[ii] = text

    # put back into df
    x_new_df['text'] = tr_text_list
    
    return x_new_df 


def create_features(x_df, min_count, max_count):
    """
        Creates a Bag of Words feature matrix from a dataframe of strings

        Will opt for input vector corresponding to single words
        Will count frequency of each word
        may exclude rare words (need to look at histogram)
        may exclude common words (same thing)
    """
    # 1) obtain the set of words 
    unique_words = set()
    tr_text_list = x_df['text'].values.tolist()
    for ii, text in enumerate(tr_text_list):
        unique_words.update(text.split())
        
    # 2) make a word count dictionary for easy indexing
    unique_words_list = list(unique_words)
    word_counts = {
        word:0 for word in unique_words_list
    }
    
    # 3) count word frequency
    num_words_in_doc = 0
    for ii, text in enumerate(tr_text_list):
        words = text.split()
        for word in words:
            word_counts[word] += 1
            num_words_in_doc += 1

    # 4) drop rare words, drop frequent words
    word_counts = keep_words_with_certain_count(word_counts, min_count, max_count)
    kept_words_list = sorted(list(word_counts.keys()))
    word_index = {
        word:idx for idx, word in enumerate(kept_words_list)
    }

    # 5) make feature vector for each string
    num_words_for_vector = len(word_counts)
    num_samples = len(tr_text_list)
    feature_matrix = np.zeros((num_samples, num_words_for_vector), dtype=int)
    print(feature_matrix.shape)
    print(f'num samples: {num_samples}, num words: {num_words_for_vector}')

    for ii, text in enumerate(tr_text_list):
        for word in text.split():
            if word in kept_words_list:
                feature_matrix[ii, word_index[word]] += 1
                
    return feature_matrix

def keep_words_with_certain_count(word_counts, min_count, max_count):
    """
        min and max count are both inclusive
    """        
    pruned_word_counts = {
        word:word_counts[word] for word in word_counts
        if word_counts[word] >= min_count and word_counts[word] <= max_count
    }
    return pruned_word_counts


def train_svm_model(X, y, C=1.0, degree=3):
    # print(C, degree)
    svc = SVC(C=C, degree=degree)   # default values
    svc.fit(X, y)
    return svc


def train_neural_network(X, y, hidden_layer_sizes):
    # we'll use MLPClassifier for a simple fully-connected feed forward network
    # default settings on the MLPClassifier look good, such as
    # - adam optimizer
    # - batch gradient descent
    # - ReLU activation
    # will need to test max_iters like before
    # might be worth testing out initial learning rate
    # might be worth testing out hidden layer sizes (both # layers and # neurons)
    # print(hidden_layer_sizes)
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    mlp.fit(X, y)
    return mlp


def test_func(**parameters):
    print(parameters)
    test_func2(**parameters)

def test_func2(x=None, y=None, z=None):
    print(x, y, z)


def cross_validate(x_train, y_train, skf, train_model, **model_params):
    """
        train_model is a function handle which takes in training data and 
        returns a sklearn-type model. important is that this model
        should have predict() and score() methods
    """
    print(model_params)

    # perform validation by using cross-validation
    # make sure the data is formatted in such a way that public cross-validation functions can easily use
    # TODO: check if the kfold split should be done only once so that
    #       the splits are the same for all models
    train_errors = []
    valid_errors = []
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=420)
    for train_idx, valid_idx in skf.split(x_train, y_train):
        fold_x_train, fold_x_valid = x_train[train_idx], x_train[valid_idx]
        fold_y_train, fold_y_valid = y_train[train_idx], y_train[valid_idx]
    
        ### figure out the general loop for the training, prediction, metric measurement
        model = train_model(fold_x_train, fold_y_train, **model_params)
        # preds = model.predict(fold_x_valid)
        accuracy = model.score(fold_x_train, fold_y_train)
        train_errors.append(1 - accuracy)
        accuracy = model.score(fold_x_valid, fold_y_valid)
        valid_errors.append(1 - accuracy)

    avg_train_error = train_errors
    avg_valid_error = valid_errors
    print(f'avg training error:   {np.mean(train_errors)}')
    print(f'avg validation error: {np.mean(valid_errors)}')

    return avg_train_error, avg_valid_error


if __name__ == "__main__":
    # test_func(x=1, y=2, z='hello')
    # exit()

    x_train_df, y_train_df = load_training_data()

    # it's simplest to preprocess data before doing split into folds
    min_word_count = 2
    max_word_count = 10
    x_train_processed = preprocess(x_train_df)
    x_train = create_features(x_train_processed, min_word_count, max_word_count)
    y_train = y_train_df['is_positive_sentiment'].to_numpy()


    # TODO: how to unpack hyperparameters into a function without overriding defaults of parameters i don't specify?
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=420)
    
    # ### SVM
    # svm_Cs = np.logspace(-1, 1, 5)    
    # for C in svm_Cs:
    #     svm_train_error, svm_valid_error = cross_validate(x_train, y_train, skf, train_svm_model, C=C)

    ### Deep Neural Network
    # mlp_num_hlayers = [10, 30, 50, 70, 90]
    # mlp_hlayer_neurons = [1100, 1300, 1500, 1700, 1900]
    mlp_num_hlayers = [5, 7, 9]
    mlp_hlayer_neurons = [100, 200, 300]
    for num_hlayers in mlp_num_hlayers:
        for num_neurons in mlp_hlayer_neurons:
            hidden_layer_sizes = np.full((num_hlayers, ), num_neurons)
            print(num_hlayers, num_neurons)
            mlp_train_error, mlp_valid_error = cross_validate(
                    x_train, y_train, skf, train_neural_network, hidden_layer_sizes=hidden_layer_sizes
            )

