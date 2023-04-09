
import autograd.numpy as anp
from autograd.misc.optimizers import adam
from autograd import grad


_loss = -1      # access training loss by making it a global variable


def embed_offsets(dim, delay):
    """
    :param dim: embedding dimension
    :param delay: number of timesteps in delay
    :return: a numpy array of the embedding index offsets
    """
    return anp.array(range((1 - dim) * delay, 1, delay), dtype=int)


def _dcenet_fwd(params, dce, t):
    inputs = anp.hstack((anp.atleast_2d(dce), anp.atleast_2d(t).T))
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

    def fit(self, series, dim, delay, dt, horizon=None, hdims=[64, 64], **kwargs):
        """
        Train a neural network to predict a process based on a sampled time series.

        :param series: One-dimensional time series of observations, equispaced in time.
        :param dim: Embedding dimension.
        :param delay: Number of observations per embedding delay, must be integral.
        :param dt: Length of time between observations.
        :param horizon: How far forward and backward to predict in training, in delay lengths. Defaults to one delay.
        :param hdims: Optional, sequence of hidden layer dimensions. Default is [64, 64].
        :param kwargs: Additional keyword arguments are passed to autograd.misc.optimizes.adam.
        :return: This DCENet instance, fitted to predict the data process.
        """
        if horizon is None:
            horizon = delay * dt
        init_params = self._init_params([dim + 1] + hdims + [1])
        tmin = int(delay * (1 - dim) - horizon / dt + 0.5)   # query interval bounds
        tmax = int(horizon / dt + 0.5)
        smin = -tmin
        smax = series.shape[0] - tmax

        self._mean, self._std = anp.mean(series), anp.sqrt(anp.var(series))
        series = self._normalize(series)

        # TODO: more efficient training procedure
        def objective(params, _):
            global _loss
            offets = embed_offsets(dim, delay)
            idx = anp.array([i + offets for i in range(smin, smax)])
            s = idx[:, -1]
            dce = series[idx]
            t = self._rng.randint(tmin, tmax, smax - smin)
            pred = _dcenet_fwd(params, dce, dt * t)[:, 0]
            err = pred - series[s + t]
            _loss = anp.mean(err * err)
            return _loss

        def callback(params, iter, _):
            if iter % 50 == 0:
                print("Loss at iteration %d =" % iter, _loss)

        self._params = adam(grad(objective), init_params, callback=callback, **kwargs)
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
        dce = self._normalize(dce)
        pred = _dcenet_fwd(self._params, dce, t)[:, 0]
        return self._unnormalize(pred)
        #return pred

    def save_params(self, filename):
        """
        Save parameters to a file.
        :param filename: .npz extension will be appended if not present
        """
        anp.savez(filename, *self._params)

