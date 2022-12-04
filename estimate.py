import numpy as np
import matplotlib.pyplot as plt
from src.data.load import get_data
from src.utils.noise import add_noise
from src.modelling.deconvolution import estimate_h_by_deconv
from src.modelling.wienerhopf import estimate_h_by_wiener_hopf

io_pair = get_data("data/estimation", val=False)
sig_range = range(0, 599)

for _, signals in io_pair.items():
    input_signal = signals["input"].reshape(-1)[sig_range]
    output_signal = signals["output"][:, 0][sig_range]
    h_deconv = estimate_h_by_deconv(input_signal, output_signal)
    h_wiener = estimate_h_by_wiener_hopf(input_signal, output_signal)
    plt.plot(h_deconv, label="deconv")
    plt.plot(h_wiener, label="wiener")
