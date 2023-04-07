
import autograd.numpy as anp
import numpy as np
from autograd.misc.optimizers import adam
from autograd import grad


def embed_idx(dim, delay, idx):
    """
    :param dim: embedding dimension
    :param delay: number of timesteps in delay
    :param idx: index of time to embed
    :return: a numpy array of the embedding indices
    """
    return np.array(range(idx - (dim - 1) * delay, idx + 1, delay), dtype=int)


def _dcenet_fwd(params, dce, t):
    inputs = anp.hstack((dce, anp.atleast_2d(t).T))
    for W, b in zip(params[0::2], params[1::2]):    # weights and biases alternate
        outputs = anp.dot(inputs, W) + b
        inputs = anp.tanh(outputs)
    return outputs


class DCENet:
    """
    A neural network predictor which fits to time series data using delay-coordinate embedding.
    """

    def __init__(self, seed=0, loadfile=None):
        """
        :param seed: Random seed to use for parameter initialization.
        :param loadfile: Name of a .npz archive with saved parameters.
        """
        self._params = None
        if loadfile is not None:
            npz = anp.load(loadfile)
            self._params = [npz[file] for file in npz.files]
        self._rng = anp.random.RandomState(seed)

    def _init_params(self, layerdims):
        result = []
        for m, n in zip(layerdims[:-1], layerdims[1:]):
            result.append(self._rng.randn(m, n))    # weight
            result.append(self._rng.randn(n))       # bias
        return result

    def fit(self, series, dim, delay, dt, horizon=None, hdims=[64, 64]):
        """
        Train a neural network to predict a process based on a sampled time series.

        :param series: One-dimensional time series of observations, equispaced in time.
        :param dim: Embedding dimension.
        :param delay: Number of observations per embedding delay, must be integral.
        :param dt: Length of time between observations.
        :param horizon: How far forward and backward to predict in training, in delay lengths. Defaults to one delay.
        :param hdims: Optional, sequence of hidden layer dimensions. Default is [64, 64].
        :return: This DCENet instance, fitted to predict the data process.
        """
        if horizon is None:
            horizon = delay * dt
        init_params = self._init_params([dim + 1] + hdims + [1])
        tmin = int(delay * (1 - dim) - horizon / dt + 0.5)   # query interval bounds
        tmax = int(horizon / dt + 0.5)
        smin = -tmin
        smax = series.shape[0] - tmax

        # TODO: more efficient training procedure
        def objective(params, _):
            s = self._rng.randint(smin, smax)        # random query point and time
            dce = series[s - (dim - 1) * delay:s + 1:delay]
            t = self._rng.randint(tmin, tmax)
            pred = _dcenet_fwd(params, dce, t / dt)
            true = series[s + t]
            return (pred - true) ** 2

        def callback(params, iter, _):
            if iter % 10000 == 0:
                print("params at iter", iter)
                print(params)

        self._params = adam(grad(objective), init_params, callback=callback, num_iters=50000)
        return self

    # TODO: warn if time is OOD
    def predict(self, dce, t):
        """
        :param dce: Delay-coord embedding of process at a time s.
        :param t: Array of time offsets, to predict embedding feature at s + t.
        :return: Prediction, or array of predictions, of feature.
        """
        if self._params is None:
            raise AttributeError("predict() called before fit()")
        dce = anp.vstack([dce for _ in range(t.shape[0])])
        return _dcenet_fwd(self._params, dce, t)[:, 0]

    def save_params(self, filename):
        """
        Save parameters to a file.
        :param filename: .npz extension will be appended if not present
        """
        anp.savez(filename, *self._params)

