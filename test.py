
import numpy as np
import matplotlib.pyplot as plt

from DCENet import DCENet, embed_idx


def plot_model(dcenet, dim, delay, dt, s, t, data):
    idx = embed_idx(dim, delay, s)
    dce = data[idx]
    plt.plot(dt * (s + t), data[s + t], color='black', label='true')
    plt.plot(dt * (s + t), dcenet.predict(dce, t), color='red', label='predicted')
    plt.scatter(dt * idx, dce, color='blue', label='observed')
    plt.xlabel('time')
    plt.ylabel('feature')
    plt.legend()
    plt.show()


def linear_growth():
    ts = np.arange(500)
    dcen = DCENet().fit(series=ts, dim=3, delay=2, dt=1, hdims=[])
    print(dcen.predict(np.array([4, 6, 8]), -0.5))
    print(dcen.predict(np.array([476, 478, 480]), -3))
    print(dcen.predict(np.array([222, 224, 226]), 1))
    print(dcen.predict(np.array([4, 6, 8]), -4))
    dcen.save_params('linparams')


def load_linparams():
    dcen = DCENet(loadfile='linparams.npz')
    print(dcen.predict(np.array([100, 102, 104]), np.arange(-7, 3)))
    plot_model(dcen, 3, 2, 1, 222, np.arange(-7, 3), np.arange(500))


def lorenz_train():
    ts_train = np.load('lorenz_train.npy')[:5000, 0]      # load only x coordinate and discard 5s of transient
    dcen = DCENet().fit(series=ts_train, dim=7, delay=190, dt=0.001, hdims=[20, 10])
    dcen.save_params('lorenz_params')


def lorenz_test():
    ts_test = np.load('lorenz_test.npy')[:5000, 0]
    dcen = DCENet(loadfile='lorenz_params.npz')
    dce = ts_test[3333 - 6 * 190:3334:190]
    t = np.array(range(-7 * 190, 191, 190)) * 0.001
    pred = dcen.predict(dce, t)
    plt.plot(t, ts_test[3333 - 7 * 190:3334 + 190:190], label='true')
    plt.plot(t, pred, label='pred')
    plt.show()


# linear_growth()
load_linparams()
