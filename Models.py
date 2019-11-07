import numpy as np


def baseline(X):
    X1, X2 = XtoX1X2(X)
    same = abs(X1-X2)

    sum_feature = np.sum(same, axis=0)
    zero_feature = np.sum(X, axis=0)
    one_feature = np.sum(X == 1, axis=0)

    features = np.concatenate((sum_feature, zero_feature, one_feature), axis=1)

    return (features)


def diff_features(X):
    X1, X2 = XtoX1X2(X)
    return X1X2toX(abs(X1-X2), X1)

def XtoX1X2(X):
    n = int(X.shape[1]/2)
    return X[:,:n], X[:,n:]

def X1X2toX(X1,X2):
    return np.concatenate((X1,X2), axis=1)

