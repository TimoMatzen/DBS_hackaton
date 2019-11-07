import numpy as np


def baseline(numpy_data):
    same = numpy_data[:, 0:10] == numpy_data[:, 10:]

    sum_feature = np.sum(same, axis=0)
    zero_feature = np.sum(numpy_data, axis=0)
    one_feature = np.sum(numpy_data == 1, axis=0)

    features = np.concatenate((sum_feature, zero_feature, one_feature), axis=1)

    return (features)
