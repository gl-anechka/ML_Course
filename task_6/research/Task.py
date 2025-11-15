import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype
        self.categories = {}

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        for col in X.columns:
            self.categories[col] = sorted(X[col].unique())
        return self

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        n_objects = X.shape[0]
        n_features = sum(len(cat) for cat in self.categories.values())
        res = np.zeros((n_objects, n_features), dtype=self.dtype)

        shift = 0
        for col, cat in self.categories.items():
            col_val = X[col].values
            for i, c in enumerate(cat):
                mask = (col_val == c)
                res[mask, shift + i] = 1
            shift += len(cat)
        return res

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.stats = {}

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        for col in X.columns:
            self.stats[col] = {}
            for val in X[col].unique():
                mask = (X[col] == val)
                self.stats[col][val] = [Y[mask].mean(), mask.mean()]
        return self

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        n_objects, n_features = X.shape
        res = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)

        for i, col in enumerate(X.columns):
            col_val = X[col].values
            for j in range(n_objects):
                c = col_val[j]
                if c in self.stats[col]:
                    mean_y, frac = self.stats[col][c]
                else:
                    mean_y, frac = 0.0, 0.0

                res[j, 3 * i] = mean_y
                res[j, 3 * i + 1] = frac
                res[j, 3 * i + 2] = (mean_y + a) / (frac + b)
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.fold_states = []

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        n_objects = X.shape[0]
        for val_idx, train_idx in group_k_fold(n_objects, self.n_folds, seed):
            stats = {}
            X_train, Y_train = X.iloc[train_idx], Y.iloc[train_idx]
            for col in X.columns:
                stats[col] = {}
                for val in X_train[col].unique():
                    mask = (X_train[col] == val)
                    stats[col][val] = [Y_train[mask].mean(), mask.mean()]
            self.fold_states.append([val_idx, stats])
        return self

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        n_objects, n_features = X.shape
        res = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)

        for val_idx, stats in self.fold_states:
            for i, col in enumerate(X.columns):
                col_val = X[col].values
                for j in val_idx:
                    c = col_val[j]
                    if c in stats[col]:
                        mean_y, frac = stats[col][c]
                    else:
                        mean_y, frac = 0.0, 0.0

                    res[j, 3 * i] = mean_y
                    res[j, 3 * i + 1] = frac
                    res[j, 3 * i + 2] = (mean_y + a) / (frac + b)
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    unique_val = np.unique(x)
    w = []

    for val in unique_val:
        mask = (x == val)
        s = np.sum(y[mask])
        n = np.sum(mask)
        w.append(s / n)

    return np.array(w)
