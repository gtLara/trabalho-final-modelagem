import numpy as np
import matplotlib.pyplot as plt
from src.data.load import get_data
from src.utils.noise import add_noise
from src.modelling.deconvolution import estimate_h_by_deconv
from src.modelling.wienerhopf import estimate_h_by_wiener_hopf

io_pair = get_data("data/estimation", val=False)
sig_range = range(0, 599)

for label, signals in io_pair.items():
    input_signal = signals["input"].reshape(-1)[sig_range]
    impulse_signal = signals["impulse"].reshape(-1)[sig_range]
    output_signal = signals["output"][:, 0][sig_range]
    plt.plot(input_signal, label="input")
    plt.plot(output_signal, label="output")
    plt.title(label)
    plt.legend()
    plt.show()
    h_deconv = estimate_h_by_deconv(input_signal, output_signal)[:100]
    h_wiener = estimate_h_by_wiener_hopf(input_signal, output_signal)[:100]
    plt.plot(impulse_signal[:100], label="deconv")
    plt.title(label)
    plt.show()
    plt.plot(h_deconv, label="wiener")
    plt.plot(h_wiener, label="wiener")
    plt.title(label)
    plt.show()
