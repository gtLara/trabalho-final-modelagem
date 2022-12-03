import numpy as np
import glob
import scipy.io as io


def get_signal(path: str) -> np.ndarray:

    signal = io.loadmat(path)

    return signal["x"]


def get_data(path: str) -> dict:

    signals = {}
    files = glob.glob(f"{path}/*")
    for file in files:
        signal = get_signal(file)
        signals[file.split("/")[-1]] = signal

    return signals
