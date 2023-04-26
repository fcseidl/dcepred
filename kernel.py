
import numpy as np
from scipy.optimize import approx_fprime


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


def differentiate(model, x):
    """
    Differentiate an rbf regressor.

    :param model: KernelRidge instance
    :param x: input value at which to differentiate
    :return:
    """
    pred = lambda z: model.predict(z.reshape(1, -1))[0]
    jac = approx_fprime(x, pred)
    return jac


