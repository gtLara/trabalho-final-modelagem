import numpy as np
import glob
import scipy.io as io


def get_signal(path: str) -> np.ndarray:

    try:
        signal = io.loadmat(path)["x"]
    except KeyError:
        signal = io.loadmat(path)["u"]

    return signal


def get_data(path: str) -> dict:

    signals = {}
    files = glob.glob(f"{path}/*")
    for file in files:
        signal = get_signal(file)
        signals[file.split("/")[-1]] = signal

    return signals
