
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
    A neural network predictor which fits to time data data using delay-coordinate embedding.
    """

    def __init__(self, seed=0, loadfile=None):
        """
        :param seed: Random seed to use for parameter initialization.
        :param loadfile: Name of a .npz archive with saved parameters.
        """
        self._params = None
        # self._data = None
        self._mean = 0.0
        self._std = 1.0
        if loadfile is not None:
            npz = anp.load(loadfile)
            self._params = [npz[file] for file in npz.files if file != 'arr_0']
            self._mean, self._std = npz['arr_0']
        self._rng = anp.random.RandomState(seed)

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

    # TODO: make init_params an argument to allow warm-starting
    def fit(self, data, dim, delay, dt, horizon=None, hdims=[64, 64],
            print_every=25, batch_size=32, n_epochs=1000, **kwargs):
        """
        Train a neural network to predict a process based on a sampled time series.

        :param data: One-dimensional time series of observations, equispaced in time.
        :param dim: Embedding dimension.
        :param delay: Number of observations per embedding delay, must be integral.
        :param dt: Length of time between observations.
        :param horizon: How far forward and backward to predict in training, in delay lengths. Defaults to one delay.
        :param hdims: Sequence of hidden layer dimensions.
        :param print_every: Training loss prints when epoch is divisible by this value.
        :param batch_size: Optional, how large of batches to use in training.
        :param n_epochs: How many epochs to run in training.
        :param kwargs: Additional keyword arguments are passed to autograd.misc.optimizers.adam.
        :return: This DCENet instance, fitted to predict the data process.
        """
        # determine lookaheads t for training
        if horizon is None:
            horizon = delay * dt
        init_params = self._init_params([dim + 1] + hdims + [1])
        tmin = int(delay * (1 - dim) - horizon / dt + 0.5)   # query interval bounds
        tmax = int(horizon / dt + 0.5)
        t_all = anp.arange(tmin, tmax)
        lookaheads = anp.hstack([t_all for _ in range(batch_size)])

        # determine times s for training
        smin = -tmin
        smax = data.shape[0] - tmax
        s_all = anp.arange(smin, smax)
        self._rng.shuffle(s_all)

        n_batches = int(anp.ceil((smax - smin) / batch_size))

        # determine and apply normalization
        self._mean, self._std = anp.mean(data), anp.sqrt(anp.var(data))
        data = self._normalize(data)

        # TODO: more efficient training procedure
        def objective(params, it):
            global _loss
            offsets = embed_offsets(dim, delay)
            batch = it % n_batches
            batch_idx = s_all[batch_size * batch:min(batch_size * (batch + 1), smax - smin)]
            batch_idx = anp.repeat(batch_idx, tmax - tmin)
            dce = anp.vstack([data[idx + offsets] for idx in batch_idx])
            times = lookaheads[:batch_idx.shape[0]]
            pred = _dcenet_fwd(params, dce, dt * times)[:, 0]
            err = pred - data[batch_idx + times]
            _loss = anp.mean(err * err)
            return _loss

        def callback(params, it, _):
            # if it % (n_batches * print_every) == 11:
            #     print("Batch 11 MSE at epoch %d = %f" % (int(it / n_batches), _loss._value))
            print("epoch %d, batch %d, loss %f" % (int(it / n_batches), it % n_batches, _loss._value))

        self._params = adam(grad(objective), init_params, callback=callback, num_iters=n_batches * n_epochs, **kwargs)
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
        norm = (self._mean, self._std)
        anp.savez(filename, norm, *self._params)

