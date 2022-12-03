import numpy as np


def add_noise(signal, SNR):

    noise_std_var = (10 ** -(SNR/20))*np.std(signal)
    noise = np.random.normal(scale=noise_std_var, size=len(signal))

    return signal + noise
