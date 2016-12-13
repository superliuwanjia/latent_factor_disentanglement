import numpy as np
import math
from scipy import linalg

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array


class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, regularization=1e-6, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the mean, whitening and dewhitening matrices.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean, whitening and dewhitening
            matrices.
        """
        X = check_array(X, accept_sparse=None, copy=self.copy,
                        ensure_2d=True)
        X = X.astype(np.float32)
        self.mean_ = X.mean(axis=0)
        X_ = X - self.mean_
        cov = np.dot(X_.T, X_) / (X_.shape[0]-1)
        U, S, _ = linalg.svd(cov)
        s = np.sqrt(S.clip(self.regularization))
        s_inv = np.diag(1./s)
        s = np.diag(s)
        self.whiten_ = np.dot(np.dot(U, s_inv), U.T)
        self.dewhiten_ = np.dot(np.dot(U, s), U.T)
        return self

    def transform(self, X, y=None, copy=None):
        """Perform ZCA whitening
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to whiten along the features axis.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X - self.mean_, self.whiten_.T)

    def inverse_transform(self, X, copy=None):
        """Undo the ZCA transform and rotate back to the original
        representation
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to rotate back.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X, self.dewhiten_) + self.mean_

def read_image_from_pil_load(image, image_shape):
    matrix = np.zeros(image_shape)
    
    for i in range(image_shape[1]):
        for j in range(image_shape[2]):
            for k in range(image_shape[0]):
                if image_shape[0]==1:
                    matrix[k,i,j] = image[i,j]
                else:
                    matrix[k,i,j] = image[i,j][k]
    return matrix
def batch_standardize(datagen, X):
    print "Standardizing all inputs..."
    standardized_x = np.zeros(X.shape)
    for i in range(X.shape[0]):
        image = X[i,:,:,:]
        s = datagen.standardize(image)
        standardized_x[i,:,:,:] =s
        if i%1000==0:
            print "On "+str(i), "th input..."

    return standardized_x

def discretize(y, num_levels):
    value_max = np.max(y)
    value_min = np.min(y)
    value_range = value_max-value_min
    y_discretized = np.zeros(y.shape)
    for i in range(y.shape[0]):
        y_discretized[i,] = int(math.floor((y[i,]-value_min)*num_levels/(value_range+0.0)))
        if y_discretized[i,]==num_levels:
            y_discretized[i,]=num_levels-1
    return y_discretized

