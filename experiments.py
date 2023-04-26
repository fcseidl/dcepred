
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from joblib import dump, load

from kernel import embed_offsets, create_data


def fit_lorenz_krr(horizons, dim=7, delay=190, sample_freq=103, savefile=None):
    # training set
    x_train = np.load('lorenz_train.npy')[5000:]
    t_train = np.arange(-min(horizons), x_train.shape[0] - max(horizons), sample_freq)
    np.random.shuffle(t_train)
    X_train, Y_train = create_data(x_train, dim, delay, t_train, horizons)

    # validation set
    x_test = np.load('lorenz_test.npy')[5000:]
    t_test = np.arange(-min(horizons), x_test.shape[0] - max(horizons), sample_freq)
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

    reg = fit_lorenz_krr(horizons, dim, delay, sample_freq=43, savefile='demo')
    # reg = load('demo.joblib')

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


# ok rip it don't work
def predicted_attractor_plot():
    dim = 7
    dt = 10
    delay = 19      # true delay is dt * delay

    # plot true trajectory
    off = embed_offsets(dim, delay)
    x_true = np.load('lorenz_test.npy')[5000:6500:dt]

    # iteratively predict new trajectory
    delay_train = dt * delay
    reg = fit_lorenz_krr(horizons=[delay_train - dt], dim=dim, delay=delay_train, savefile='lookahead', sample_freq=23)
    reg = load('lookahead.joblib')
    x_pred = x_true[:dim * delay - 2:]
    while x_pred.shape[0] < x_true.shape[0]:
        X = x_pred[x_pred.shape[0] - delay + 2 + off].reshape(1, -1)
        y = reg.predict(X)
        x_pred = np.hstack((x_pred, y[0]))

    # plot new trajectory
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x_true[:(1 - dim) * delay], x_true[3 * delay:(4 - dim) * delay], x_true[(dim - 1) * delay:])
    ax.plot(x_pred[:(1 - dim) * delay], x_pred[3 * delay:(4 - dim) * delay], x_pred[(dim - 1) * delay:])
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t - 3 * tau)')
    ax.set_zlabel('x(t - 6 * tau)')
    plt.show()


def powermethod(M, tol=1e-6, maxiter=1000):
    x = np.ones(M.shape[0])
    for it in range(maxiter):
        x /= np.linalg.norm(x)
        lam = x @ M @ x
        if np.linalg.norm(M @ x - lam * x) < tol:
            return lam, x, it
        x = M @ x


def ftle_comparison_plot():
    from simulate import variational_lorenz_deriv, rk4
    dt = 0.001
    x0_var = np.array([22, 23, 24, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    traj = rk4(variational_lorenz_deriv, x0_var, t0=0, tf=10, dt=dt)
    x0_var[:3] = traj[5000, :3]
    variational = rk4(variational_lorenz_deriv, x0_var, t0=0, tf=1, dt=dt)
    deltas = variational[:, 3:].reshape(-1, 3, 3)
    t_plot = np.arange(0, 500, 10)
    lams = np.array([np.abs(np.linalg.eigvals(deltas[t])).max() for t in t_plot])
    plt.plot(dt * t_plot, lams)
    plt.xlabel('t')
    plt.ylabel('ftle')
    plt.show()






# exact_predictor_demo_plot()
# predicted_attractor_plot()
ftle_comparison_plot()
