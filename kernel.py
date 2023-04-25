
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
    Creates training data to predict a time series based on DCE.

    :param series: shape (N,) time series of observations, equispaced in time.
    :param dim: Embedding dimension.
    :param delay: Embedding delay.
    :param times: Shape (n,) times to use in training.
    :param horizons: Shape (n,) prediction horizon for each training time.
    :return: Training data X_train, y_train
    """
    offsets = embed_offsets(dim, delay)
    embed_t = np.add.outer(times, offsets)
    dce = series[embed_t]
    X = np.hstack([dce, horizons.reshape(-1, 1)])
    y = series[times + horizons]
    return X, y


if __name__ == '__main__':
    series = np.load('lorenz_train.npy')[5000:]
    dim = 7
    delay = 190
    mino = -delay
    maxo = 3 * delay
    times = np.arange(-mino, series.shape[0] - maxo, 7)
    np.random.shuffle(times)
    horizons = 293 * np.arange(times.shape[0]) % (maxo - mino) + mino
    X_train, y_train = create_data(series, dim, delay, times, horizons)
    norm = np.std(X_train, axis=0)
    X_train /= norm
    # krr = KRR(ard(gamma), 0.1).fit(X_train, y_train)

    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import SGDRegressor

    rbf_feature = Nystroem(gamma=0.001)
    kerprox_train = rbf_feature.fit_transform(X_train)
    reg = SGDRegressor(loss='epsilon_insensitive').fit(kerprox_train, y_train)

    import matplotlib.pyplot as plt

    #series = np.load('lorenz_test.npy')
    dt = 0.001
    t = times[50]
    h = np.arange(mino - delay, maxo + delay)
    X_test, y_test = create_data(series, dim, delay, t * np.ones_like(h), h)
    X_test /= norm
    kerprox_test = rbf_feature.transform(X_test)
    y_pred = reg.predict(kerprox_test)
    plt.plot(dt * (t + h), y_test, color='black', label='true')
    plt.plot(dt * (t + h), y_pred, color='red', label='predicted')
    plt.plot([dt * (t + horizons[50]), dt * (t + horizons[50])], [-50, 50])
    #plt.scatter(dt * (t + off), dce[:, 0], color='blue', label='observed')
    plt.xlabel('time')
    plt.ylabel('feature')
    plt.legend()
    plt.show()
