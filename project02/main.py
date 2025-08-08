import numpy as np
import pandas as pd
import string
import re
import unicodedata

from sklearn.model_selection import StratifiedKFold


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


def create_features(x_df):
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
    # word_index = {
    #     word:idx for idx, word in enumerate(unique_words_list)
    # }
    
    # 3) count word frequency
    num_words_in_doc = 0
    for ii, text in enumerate(tr_text_list):
        words = text.split()
        for word in words:
            word_counts[word] += 1
            num_words_in_doc += 1

    # 4) drop rare words, drop frequent words
    min_count = 15
    max_count = 15
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



if __name__ == "__main__":
    x_train_df, y_train_df = load_training_data()

    # it's simplest to preprocess data before doing split into folds
    x_train_processed = preprocess(x_train_df)
    x_train = create_features(x_train_processed)
    y_train = y_train_df['is_positive_sentiment'].to_numpy()
    
    
    # perform validation by using cross-validation
    # make sure the data is formatted in such a way that public cross-validation functions can easily use
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=420)
    for train_idx, valid_idx in skf.split(x_train_df, y_train_df):
        fold_x_train, fold_x_valid = x_train[train_idx], x_train[valid_idx]
        fold_y_train, fold_y_valid = y_train[train_idx], y_train[valid_idx]
    
