
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from joblib import dump, load

from kernel import embed_offsets, create_data


def fit_lorenz_krr(horizons, dim=7, delay=190, sample_freq=103, savefile=None):
    # training set
    x_train = np.load('lorenz_train.npy')[5000:]
    t_train = np.arange(-horizons.min(), x_train.shape[0] - horizons.max(), sample_freq)
    np.random.shuffle(t_train)
    X_train, Y_train = create_data(x_train, dim, delay, t_train, horizons)

    # validation set
    x_test = np.load('lorenz_test.npy')[5000:]
    t_test = np.arange(-horizons.min(), x_test.shape[0] - horizons.max(), sample_freq)
    np.random.shuffle(t_test)
    X_test, Y_test = create_data(x_test, dim, delay, t_test, horizons)

    # cross validation
    gamma, l2reg, best_r2 = 0, 0, -np.inf
    best_model = KernelRidge()
    for gamma in np.geomspace(0.1, 0.01, 4):
        for l2reg in np.geomspace(0.5, 0.05, 4):
            reg = KernelRidge(alpha=l2reg, kernel='rbf', gamma=gamma)
            reg.fit(X_train, Y_train)
            r2 = reg.score(X_test, Y_test)
            print('gamma = %f, l2reg = %f, R^2 = %f' % (gamma, l2reg, r2))
            if r2 > best_r2:
                best_model = reg

    if savefile is not None:
        dump(best_model, savefile + '.joblib')
    return best_model


def exact_predictor_demo_plot():
    # hyperparameter specification
    dim = 7
    delay = 190
    mino = -dim * delay
    maxo = 3 * delay
    horizons = np.linspace(mino, maxo, 100).astype(int)

    # reg = fit_lorenz_krr(horizons, dim, delay, sample_freq=43, savefile='demo')
    reg = load('demo.joblib')

    # test on a different trajectory
    x_test = np.load('lorenz_test.npy')
    dt = 0.001
    t_test = np.array([23500])
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


def predicted_attractor_plot():
    # TODO: plot true trajectory and trajectory from iterating predictor
    pass


def ftle_comparison_plot():
    # TODO: plot ftle from variational equation against predicted
    pass


exact_predictor_demo_plot()