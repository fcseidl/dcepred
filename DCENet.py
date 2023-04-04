
import autograd.numpy as anp
from autograd.misc.optimizers import adam
from autograd import grad


def _dcenet_fwd(params, dce, t):
    inputs = anp.hstack((dce, t))
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
        self._mean = 0.0
        self._std = 1.0

    # train on centered data with unit variance
    def _normalize(self, data):
        return (data - self._mean) / self._std

    def _unnormalize(self, data):
        return self._std * data + self._mean

    def _init_params(self, layerdims):
        result = []
        for m, n in zip(layerdims[:-1], layerdims[1:]):
            result.append(self._rng.randn(m, n))    # weight
            result.append(self._rng.randn(n))       # bias
        return result

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
        init_params = self._init_params([dim + 1] + hdims + [1])
        tmin = int(delay * (1 - dim - horizon))   # query interval bounds
        tmax = int(delay * horizon)
        smin = -tmin
        smax = series.shape[0] - tmax

        self._mean, self._std = anp.mean(series), anp.sqrt(anp.var(series))
        series = self._normalize(series)

        # TODO: more efficient training procedure
        def objective(params, _):
            s = self._rng.randint(smin, smax)        # random query point and time
            dce = series[s - (dim - 1) * delay:s + 1:delay]
            t = self._rng.randint(tmin, tmax)
            pred = _dcenet_fwd(params, dce, t / delay)
            true = series[s + t]
            return (pred - true) ** 2

        def callback(params, iter, _):
            if iter % 10000 == 0:
                print("params at iter", iter)
                print(params)

        self._params = adam(grad(objective), init_params, callback=callback, num_iters=50000)
        return self

    def predict(self, dce, t):
        """
        :param dce: Delay-coord embedding of process at a time s.
        :param t: Predict embedding feature at time s + t. Time units assumed are the length of a delay!
        :return: Prediction of feature.
        """
        if self._params is None:
            raise AttributeError("predict() called before fit()")
        normed = self._normalize(dce)
        pred = _dcenet_fwd(self._params, normed, t)[0]
        return self._unnormalize(pred)
        #return _dcenet_fwd(self._params, dce, t)[0]

    def save_params(self, filename):
        """
        Save parameters to a file.
        :param filename: .npz extension will be appended if not present
        """
        anp.savez(filename, *self._params)

