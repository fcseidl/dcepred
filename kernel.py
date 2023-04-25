
import numpy as np


def rbf(gamma):
    return lambda x, z: np.exp(-gamma * ((x - z) ** 2).sum(axis=-1))


def ard(gamma):
    return lambda x, z: np.exp(-(x - z) ** 2 @ gamma)


class KRR:

    def __init__(self, kernel, l2):
        self._k = kernel
        self._l2 = l2
        self._w = None
        self._X = None

    def fit(self, X, y):
        n_samp = X.shape[0]
        gram = np.array([self._k(X, xi) for xi in X])
        self._w = np.linalg.solve(gram + self._l2 * np.identity(n_samp), y)

        print('MSE =', ((self._w @ gram - y) ** 2).mean())

        self._X = X
        return self

    def predict(self, X):
        return np.array([self._w @ self._k(self._X, xi) for xi in X])


def embed_offsets(dim, delay):
    """
    :param dim: embedding dimension
    :param delay: number of timesteps in delay
    :return: a numpy array of the embedding index offsets
    """
    return np.array(range((1 - dim) * delay, 1, delay), dtype=int)


def create_data(series, dim, delay, times, horizons):
    """
    Creates training data to predict a time x_train based on DCE.

    :param series: shape (N,) time series of observations, equispaced in time.
    :param dim: Embedding dimension.
    :param delay: Embedding delay.
    :param times: Shape (n_samp,) times to use in training.
    :param horizons: Shape (n_feat,) Prediction horizons defining different features.
    :return: Training data X_train, Y_train, representing embeddings and lookahead values, resp.
    """
    offsets = embed_offsets(dim, delay)
    embed_idx = np.add.outer(times, offsets)
    predict_idx = np.add.outer(times, horizons)
    return series[embed_idx], series[predict_idx]

