
import numpy as np


def rk4(deriv, x0, t0, tf, dt):
    xn = [x0]
    while t0 < tf:
        k1 = deriv(x0, t0)
        k2 = deriv(x0 + 0.5 * dt * k1, t0 + 0.5 * dt)
        k3 = deriv(x0 + 0.5 * dt * k2, t0 + 0.5 * dt)
        k4 = deriv(x0 + dt * k3, t0 + dt)
        x1 = x0 + dt * (k1 + 2 * (k2 + k3) + k4) / 6
        xn.append(x1)
        x0 = x1
        t0 += dt
    return np.array(xn)


a = 16
r = 45
b = 4


def lorenz_deriv(x, _):
    return np.array([
        a * (x[1] - x[0]),
        r * x[0] - x[1] - x[0] * x[2],
        x[0] * x[1] - b * x[2]
    ])


def variational_lorenz_deriv(x, _):
    result = np.empty(12)
    # derivatives of x, y, z
    result[0:3] = np.array([
        a * (x[1] - x[0]),
        r * x[0] - x[1] - x[0] * x[2],
        x[0] * x[1] - b * x[2]
    ])
    # unflatten to take derivatives of deltas
    delta = x[3:].reshape(3, 3)
    jac = np.array([
        [-a, a, 0],
        [r - x[2], -1, -x[0]],
        [x[1], x[0], -b]
    ])
    result[3:] = (jac @ delta).reshape(9)
    return result

if __name__ == '__main__':
    x0_train = np.ones(3)
    x0_test = np.array([46, 22, 50])
    lorenz_train = rk4(lorenz_deriv, x0_train, t0=0, tf=300, dt=0.001)
    lorenz_test = rk4(lorenz_deriv, x0_test, t0=0, tf=300, dt=0.001)
    np.save('lorenz_train', lorenz_train[:, 0])
    np.save('lorenz_test', lorenz_test[:, 0])



