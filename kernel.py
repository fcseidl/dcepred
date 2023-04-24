
import numpy as np


def rbf(gamma):
    return lambda x, z: np.exp(-gamma * ((x - z) ** 2).sum(axis=-1))


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
        self._X = X
        return self

    def predict(self, z):
        return self._w @ self._k(self._X, z)


def embed_offsets(dim, delay):
    """
    :param dim: embedding dimension
    :param delay: number of timesteps in delay
    :return: a numpy array of the embedding index offsets
    """
    return np.array(range((1 - dim) * delay, 1, delay), dtype=int)


def dce_training_data(series, dim, delay, times, horizons):
    """
    Creates training data to predict a time series based on DCE.

    :param series: shape (N,) time series of observations, equispaced in time.
    :param dim: Embedding dimension.
    :param delay: Embedding delay.
    :param times: Shape (n,) times to use in training.
    :param horizons: Shape (n,) prediction horizon for each training time.
    :return: Training data X, y
    """
    offsets = embed_offsets(dim, delay)
    embed_t = np.add.outer(times, offsets)
    return series[embed_t], series[times + horizons]


if __name__ == '__main__':
    series = np.load('lorenz_train.npy')[5000:]
    dim = 7
    delay = 190
    mino = (1 - dim) * delay
    maxo = 3 * delay
    times = np.arange(-mino, series.shape[0] - maxo, delay - 1)
    np.random.shuffle(times)
    horizons = np.arange(times.shape[0]) % (maxo - mino) + mino
    X, y = dce_training_data(series, dim, delay, times, horizons)
    krr = KRR(rbf(1), 0.1).fit(X, y)
