'''
hw1.py
Author: Kenny Yau

Tufts CS 135 Intro ML

'''

import numpy as np

def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    ''' Divide provided array into train and test set along first dimension

    User can provide a random number generator object to ensure reproducibility.

    Args
    ----
    x_all_LF : 2D array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
    random_state : np.random.RandomState instance or integer or None
        If int, code will create RandomState instance with provided value as seed
        If None, defaults to the current numpy random number generator np.random

    Returns
    -------
    x_train_MF : 2D array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    x_test_NF : 2D array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    Post Condition
    --------------
    This function should be side-effect free. The provided input array x_all_LF
    should not change at all (not be shuffled, etc.)

    Examples
    --------
    >>> x_LF = np.eye(10)
    >>> xcopy_LF = x_LF.copy() # preserve what input was before the call
    >>> train_MF, test_NF = split_into_train_and_test(
    ...     x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
    >>> train_MF.shape
    (7, 10)
    >>> test_NF.shape
    (3, 10)
    >>> print(train_MF)
    [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    >>> print(test_NF)
    [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]

    ## Verify that input array did not change due to function call
    >>> np.allclose(x_LF, xcopy_LF)
    True

    References
    ----------
    For more about RandomState, see:
    https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    '''
    if random_state is None:
        print('using numpy\' default random state')
        random_state = np.random
    elif isinstance(random_state, int):
        print(f"using provided seed {random_state} to initialize a RandomState")
        random_state = np.random.RandomState(seed=random_state)
    ## TODO fixme

    # using RandomState instead of the newer Generator since hw docs refer to it
    
    # get number of training data and number of predictors
    n_total_examples, n_features = x_all_LF.shape
    print(x_all_LF.shape)

    n_test_examples  = int(np.ceil(n_total_examples * frac_test))
    n_train_examples = n_total_examples - n_test_examples

    print(n_test_examples)
    print(n_train_examples)


    # using np.random.permutation to shuffle the provided dataset
    permuted_x = random_state.permutation(x_all_LF)
    x_test_NF = permuted_x[:n_test_examples]
    x_train_MF = permuted_x[n_test_examples:]
    
    print('original data')
    print(x_all_LF)
    print('')

    print('permuted data')
    print(permuted_x)
    print('')

    print('test set')
    print(x_test_NF)
    print('')

    print('train set')
    print(x_train_MF)
    print('')


    # x_train_MF
    return x_train_MF, x_test_NF


def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Any ties in distance may be broken arbitrarily.

    Args
    ----
    data_NF : 2D array, shape = (n_examples, n_features) aka (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D array, shape = (n_queries, n_features) aka (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, positive (must be >= 1)
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D array, (n_queries, n_neighbors, n_feats) (Q, K, F)
        Entry q,k is feature vector of the k-th neighbor of the q-th query
    '''
    N, F = data_NF.shape
    Q, _ = query_QF.shape
    neighb_QKF = np.zeros((Q, K, F))
    distance_arr = np.zeros((N))

    for qq, query in enumerate(query_QF):
        
        for nn, datapoint in enumerate(data_NF):
            distance_arr[nn] = compute_euclidean_distance(query, datapoint)
        
        sort_order = np.argsort(distance_arr)
        
        # sorts rows of data by sort_order then selects only the K top selections
        neighb_QKF[qq] = data_NF[sort_order][:K]

    return neighb_QKF


def compute_euclidean_distance(vec1, vec2):
    """
        Computes the euclidean distance between 2 vectors

        Assumes that vec1 and vec2 are the same size
    """

    difference = vec1 - vec2
    sq_diff = np.square(difference)
    sum_sq_diff = np.sum(sq_diff)
    euclidean_dist = np.sqrt(sum_sq_diff)

    return euclidean_dist