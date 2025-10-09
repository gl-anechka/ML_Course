import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    folds = []
    n = np.arange(num_objects)
    fold_size = num_objects // num_folds

    if num_objects % num_folds != 0:
        split = [fold_size for _ in range(0, (num_folds - 1) * fold_size, fold_size)]
        split.append(num_objects - fold_size)
    else:
        split = [fold_size for _ in range(num_folds)]

    i = 0
    for x in split:
        temp = set(n[i:i + x])
        folds.append((np.array([x for x in n if x not in temp]), np.array(n[i:i + x])))
        i += x

    return folds


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    res = {}
    for norm in parameters['normalizers']:
        for neig in parameters['n_neighbors']:
            for metric in parameters['metrics']:
                for weight in parameters['weights']:
                    cur_iteration = []
                    for i, j in folds:
                        X_train, X_test = X[i], X[j]
                        if norm[0] is not None:
                            norm[0].fit(X_train)
                            X_train = norm[0].transform(X_train)
                            X_test = norm[0].transform(X_test)
                        knn = knn_class(n_neighbors=neig, weights=weight, metric=metric)
                        knn.fit(X_train, y[i])
                        cur_iteration.append(score_function(y[j], knn.predict(X_test)))
                    res[(norm[1], neig, metric, weight)] = np.mean(np.array(cur_iteration))
    return res
