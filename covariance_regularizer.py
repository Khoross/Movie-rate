from keras import backend as K
from keras import Regularizer


class CovarianceRegularizer(Regularizer):
    """Regularzer for punishing correllated inputs.
    Calculates the covariance matrix for the current weights
    punishes based on difference from an identity matrix
    difference calculatd using Frobenius norm
    Also punishes for mean differeing from 0
    if ignore_variance has been set, disregards variance entirely
    punishes only nonzero covarince.
    """

    def __init__(self, corr_weight, mean_weight, ignore_variance=False):
        self.corr_weight = corr_weight
        self.mean_weight = mean_weight
        self.ignore_variance = ignore_variance

    def __call__(self, x):
        mean = K.mean(x, axis=0)
        mean_norm = K.sqrt(K.sum(K.square(mean))) * self.mean_weight

        obs = K.int_shape(x)[0]
        x -= K.sum(x, axis=0, keepdims=True) / obs
        cov = K.dot(K.transpose(x), x) / (obs - 1)
        if self.ignore_variance:
            cov = cov * (1 - K.eye(K.int_shape(cov)[0]))
        else:
            cov = cov - K.eye(K.int_shape(cov)[0])
        cov_norm = K.flatten(K.square(cov))
        cov_norm = K.sqrt(K.sum(cov_norm)) * self.corr_weight

        return(mean_norm + cov_norm)
