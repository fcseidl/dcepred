
import numpy as np
from DCENet import DCENet


def linear_growth():
    ts = np.arange(500)
    print(DCENet().fit(series=ts, dim=3, delay=2, hdims=[]).predict(np.array([4, 6, 8]), -1))


linear_growth()
