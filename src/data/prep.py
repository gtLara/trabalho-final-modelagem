from src.utils.noise import add_noise
import numpy as np


def prep_signals(signals, segment=None, noisy=True, SNR=20):

    if segment is not None:
        u, y = signals[0][segment].reshape(-1), signals[1][segment]
    else:
        u, y = signals[0].reshape(-1), signals[1]

    y = y - y[0]
    u = u - u[0]

    if noisy is not None:
        y = add_noise(y, SNR=SNR)

    time = np.arange(0, 0.1*len(u), 0.1)

    return u, y, time
