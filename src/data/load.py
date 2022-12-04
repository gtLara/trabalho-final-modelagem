import numpy as np
import glob
import scipy.io as io


def get_val_signal(path: str) -> np.ndarray:

    try:
        signal = io.loadmat(path)["x"]
    except KeyError:
        signal = io.loadmat(path)["u"]

    return signal


def get_estimation_signal(path: str) -> dict:

    x_signal = io.loadmat(path)["x"]
    i_signal = io.loadmat(path)["x2"]
    u_signal = io.loadmat(path)["u"]

    signal = {"input": u_signal,
              "output": x_signal,
              "impulse": i_signal}

    return signal


def get_data(path: str, val=False) -> dict:

    signals = {}
    files = glob.glob(f"{path}/*")
    for file in files:

        if val:
            signal = get_val_signal(file)
        else:
            signal = get_estimation_signal(file)

        signals[file.split("/")[-1]] = signal

    return signals
