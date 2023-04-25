
import numpy as np
import matplotlib.pyplot as plt

from kernel import embed_offsets, create_data


def exact_predictor_demo_plot():
    # hyperparameter specification
    dim = 7
    delay = 190
    mino = -dim * delay
    maxo = delay
    scale = 0.001
    l2reg = 0.5

    # train model
    x_train = np.load('lorenz_train.npy')[5000:]
    t_train = np.arange(-mino, x_train.shape[0] - maxo, 189)
    np.random.shuffle(t_train)
    horizons = np.linspace(mino, maxo, 100).astype(int)
    X_train, Y_train = create_data(x_train, dim, delay, t_train, horizons)

    # train regressor to each lookahead
    from sklearn.kernel_ridge import KernelRidge
    reg = KernelRidge(alpha=l2reg, kernel='rbf', gamma=scale)
    reg.fit(X_train, Y_train)

    # test on a different trajectory
    x_test = np.load('lorenz_test.npy')
    dt = 0.001
    t_test = np.array([30500])
    X_test, Y_test = create_data(x_test, dim, delay, t_test, horizons)
    Y_pred = reg.predict(X_test)

    # plot test results
    plt.plot(dt * (t_test[0] + horizons), Y_test[0], color='black', label='true')
    plt.plot(dt * (t_test[0] + horizons), Y_pred[0], color='red', label='predicted')
    off = embed_offsets(dim, delay)
    plt.scatter(dt * (t_test[0] + off), X_test[0], color='blue', label='observed')
    plt.xlabel('time')
    plt.ylabel('feature')
    plt.legend()
    plt.show()


exact_predictor_demo_plot()