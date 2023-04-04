
import numpy as np
from DCENet import DCENet


def linear_growth():
    ts = np.arange(500)
    dcen = DCENet().fit(series=ts, dim=3, delay=2, hdims=[])
    print(dcen.predict(np.array([4, 6, 8]), -0.5))
    print(dcen.predict(np.array([476, 478, 480]), -3))
    print(dcen.predict(np.array([222, 224, 226]), 1))
    print(dcen.predict(np.array([4, 6, 8]), -4))
    dcen.save_params('linparams')


def load_linparams():
    dcen = DCENet(loadfile='linparams.npz')
    print(dcen.predict(np.array([4, 6, 8]), -1))
    print(dcen.predict(np.array([476, 478, 480]), -3))
    print(dcen.predict(np.array([222, 224, 226]), 1))
    print(dcen.predict(np.array([4, 6, 8]), -4))
    print(dcen.predict(np.array([40, 42, 44]), 3))


linear_growth()
#load_linparams()