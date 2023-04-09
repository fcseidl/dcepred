
import numpy as np
import matplotlib.pyplot as plt

from DCENet import DCENet, embed_offsets


def plot_model(dcenet, dim, delay, dt, s, t, data):
    offsets = embed_offsets(dim, delay)
    dce = data[s + offsets]
    plt.plot(dt * (s + t), data[s + t], color='black', label='true')
    plt.plot(dt * (s + t), dcenet.predict(dce, dt * t), color='red', label='predicted')
    plt.scatter(dt * (s + offsets), dce, color='blue', label='observed')
    plt.xlabel('time')
    plt.ylabel('feature')
    plt.legend()
    plt.show()


def linear_growth():
    ts = np.arange(500)
    dcen = DCENet().fit(series=ts, dim=3, delay=2, dt=0.1, hdims=[2], num_iters=10000)
    dcen.save_params('linparams')


# TODO: dump mean, std with save_params!!!!!
def load_linparams():
    dcen = DCENet(loadfile='linparams.npz')
    # dcen._mean, dcen._std = np.mean(np.arange(500)), np.std(np.arange(500))
    print(dcen.predict(np.array([100, 102, 104]), np.arange(-7, 3)))
    plot_model(dcen, 3, 2, 0.1, 222, np.arange(-7, 3), np.arange(500))


def lorenz_train():
    ts_train = np.load('lorenz_train.npy')[10000:12000, 0]      # load only x coordinate and discard 5s of transient
    dcen = DCENet().fit(series=ts_train, dim=7, delay=190, dt=0.001, hdims=[20, 10], num_iters=50000)
    dcen.save_params('lorenz_params')


def lorenz_test():
    ts_test = np.load('lorenz_test.npy')[10000:12000, 0]
    dcen = DCENet(loadfile='lorenz_params.npz')
    plot_model(dcen, 7, 190, 0.001, 1500, np.arange(-1500, 150), ts_test)


linear_growth()
load_linparams()
# lorenz_train()
# lorenz_test()
