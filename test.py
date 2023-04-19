
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


def linear_train():
    ts = np.arange(500)
    dcen = DCENet().fit(data=ts, dim=3, delay=2, dt=0.1, hdims=[2])
    dcen.save_params('linparams')


def linear_test():
    dcen = DCENet(loadfile='linparams.npz')
    print(dcen.predict(np.array([100, 102, 104]), np.arange(-7, 3)))
    plot_model(dcen, 3, 2, 0.1, 222, np.arange(-7, 3), np.arange(500))


def sine_train():
    t_train = np.arange(0, 100, 0.01)
    data = np.sin(t_train)
    net = DCENet().fit(data, dim=2, delay=80, dt=0.01, hdims=[10, 8], n_epochs=100)
    net.save_params('sineparams')


def sine_test():
    t_test = np.arange(0, 17, 0.01)
    data = np.sin(t_test)
    net = DCENet(loadfile='sineparams.npz')
    plot_model(net, 2, 80, 0.01, s=500, t=np.arange(-200, 100), data=data)


def lorenz_train():
    ts_train = np.load('lorenz_train.npy')[5000:, 0]      # load only x coordinate and discard 5s of transient
    dcen = DCENet().fit(data=ts_train, dim=7, delay=190, dt=0.001, hdims=[20, 10], print_every=1, n_epochs=10, batch_size=50)
    dcen.save_params('lorenz_params')


def lorenz_test():
    ts_test = np.load('lorenz_train.npy')[5000:7000, 0]
    dcen = DCENet(loadfile='lorenz_params.npz')
    plot_model(dcen, 7, 190, 0.001, 1500, np.arange(-1500, 150), ts_test)


# sine_train()
# sine_test()
# linear_train()
# linear_test()
lorenz_train()
lorenz_test()
