
import autograd.numpy as anp
from autograd.misc.optimizers import adam
from autograd import grad


def _dcenet_fwd(params, dce, t):
    inputs = anp.hstack((dce, t))
    for W, b in params:
        outputs = anp.dot(inputs, W) + b
        inputs = anp.tanh(outputs)
    return outputs


class DCENet:
    """
    A neural network predictor which fits to time series data using delay-coordinate embedding.
    """

    def __init__(self, seed=0):
        """:param seed: Random seed to use for parameter initialization."""
        self._params = self._horizon = None
        self._rng = anp.random.RandomState(seed)

    def _init_params(self, layerdims):
        return [(self._rng.randn(m, n), self._rng.randn(n))  # (weight, bias)
                for m, n in zip(layerdims[:-1], layerdims[1:])]

    # fit and predict methods in the style of sklearn

    def fit(self, series, dim, delay, horizon=1, hdims=[64, 64]):
        """
        Train a neural network to predict a process based on a sampled time series.

        :param series: One-dimensional time series of observations, equispaced in time.
        :param dim: Embedding dimension.
        :param delay: Number of observations per embedding delay, must be integral.
        :param horizon: How far forward in backward to predict in training, in delay lengths. The default is 1.
        :param hdims: Optional, sequence of hidden layer dimensions. Default is [64, 64].
        :return: This DCENet instance, fitted to predict the data process.
        """
        self._horizon = horizon
        init_params = self._init_params([dim + 1] + hdims + [1])
        tmin = int(delay * (1 - dim - self._horizon))   # query interval bounds
        tmax = int(delay * self._horizon)
        smin = -tmin
        smax = series.shape[0] - tmax

        def objective(params, _):
            s = self._rng.randint(smin, smax)        # random query point and time
            dce = series[s - (dim - 1) * delay:s + 1:delay]
            t = self._rng.randint(tmin, tmax)
            pred = _dcenet_fwd(params, dce, t)
            true = series[s + t]
            return (pred - true) ** 2

        def callback(params, iter, _):
            if iter % 100 == 0:
                print("params at iter", iter)
                print(params)

        self._params = adam(grad(objective), init_params, callback=callback, num_iters=100000)
        return self

    def predict(self, dce, t):
        """
        :param dce: Delay-coord embedding of process at a time s.
        :param t: Predict embedding feature at time s + t. Time units assumed are the length of a delay!
        :return: Prediction of feature.
        """
        if self._params is None:
            raise AttributeError("predict() called before fit()")
        if t > self._horizon or t < 1 - dce.shape[0] - self._horizon:
            raise Warning("Prediction time horizon is outside of training interval.")
        return _dcenet_fwd(self._params, dce, t)
