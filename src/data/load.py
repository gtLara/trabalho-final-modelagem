import numpy as np
import glob
from os.path import basename
import scipy.io as io
from sklearn.model_selection import train_test_split as tts


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

        signals[basename(file)] = signal

    return signals


def get_step(label="Ca_var", variable="temperature"):

    var_index = int(variable != "temperature")

    signals = get_data("data/validation", val=True)

    u, y = signals["Entrada_1"], signals["Ca_var"][:, var_index]

    u_est, u_val, y_est, y_val = tts(u, y, test_size=0.25, random_state=42,
                                     shuffle=False)

    return (u_est, y_est), (u_val, y_val)


def get_white(label="Ca_var", variable="temperature"):

    var_index = int(variable != "temperature")

    signals = get_data("data/estimation", val=False)["Ca_var"]

    u, y = signals["input"], signals["output"][:, var_index]

    u_est, u_val, y_est, y_val = tts(u, y, test_size=0.25, random_state=42,
                                     shuffle=False)

    return (u_est, y_est), (u_val, y_val)
