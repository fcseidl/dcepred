
import numpy as np
import matplotlib.pyplot as plt

from DCENet import DCENet, embed_offsets


def plot_model(model, dim, delay, dt, s, t, data):
    offsets = embed_offsets(dim, delay)
    dce = data[s + offsets]
    plt.plot(dt * (s + t), data[s + t], color='black', label='true')
    plt.plot(dt * (s + t), model.predict(dce, dt * t), color='red', label='predicted')
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
    net = DCENet().fit(data, dim=2, delay=80, dt=0.01, hdims=[10, 8], savefreq=33, savename='sine_epoch_', epochs=100)
    net.save_params('sineparams')


def sine_test():
    t_test = np.arange(0, 17, 0.01)
    data = np.sin(t_test)
    net = DCENet(loadfile='sine_epoch_66.npz')
    plot_model(net, 2, 80, 0.01, s=500, t=np.arange(-200, 100), data=data)


def lorenz_train():
    hdims = [32]
    print(hdims)
    ts_train = np.load('lorenz_train.npy')[10000:, 0]      # load only x coordinate and discard 10s of transient
    dcen = DCENet(edim=7, hdims=hdims).fit(data=ts_train, delay=190, dt=0.001, savefreq=1, savename='lorenz_epoch_', epochs=10, batchsize=60)
    dcen.save_params('lorenz_params')


def lorenz_test():
    ts_test = np.load('lorenz_test.npy')[10000:12000]
    dcen = DCENet(loadfile='lorenz_epoch_9.npz')
    plot_model(dcen, 7, 190, 0.001, 1500, np.arange(-1500, 500), ts_test)


# sine_train()
# sine_test()
# linear_train()
# linear_test()
# lorenz_train()
lorenz_test()
