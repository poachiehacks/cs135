import numpy as np
import hw1


### Problem 1
def test_problem_one():

    # creating example data
    n_total_examples = 21
    n_features = 5
    x_all_LF = np.zeros((n_total_examples, n_features))

    for ii in range(n_total_examples):
        for jj in range(n_features):
            x_all_LF[ii, jj] = (ii + 1)*(jj + 1)
    print(x_all_LF)

    hw1.split_into_train_and_test(x_all_LF, frac_test=0.7)
    # hw1.split_into_train_and_test(x_all_LF, frac_test=0.3, random_state=420)




    x_LF = np.eye(10)
    xcopy_LF = x_LF.copy() # preserve what input was before the call
    train_MF, test_NF = hw1.split_into_train_and_test(x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
    train_MF.shape
    test_NF.shape
    print(train_MF)
    print(test_NF)
    ## Verify that input array did not change due to function call
    print(np.allclose(x_LF, xcopy_LF))


### Problem 2
def test_problem_two():
    N = 21
    Q = 9
    F = 13
    K = 3
    data_NF  = np.zeros((N, F))
    query_QF = np.zeros((Q, F))

    for qq in range(Q):
        for ff in range(F):
            query_QF[qq, ff] = (qq + 1) * (ff + 1)

    for nn in range(N):
        for ff in range(F):
            data_NF[nn, ff] = (nn + 1) * (ff + 1) + 0.01


    neighb_QKF = hw1.calc_k_nearest_neighbors(data_NF, query_QF, K)
    print(neighb_QKF)

    # vec1 = np.array([1, 2, 3, 4])
    # vec2 = np.array([1.01, 2.02, 3.03, 3.96])
    # # vec1 = np.array([0, 0, 3])
    # # vec2 = np.array([0, 0, 4])
    # hw1.compute_euclidean_distance(vec2, vec1)


# test_problem_one()
test_problem_two()