import numpy as np
import matplotlib.pyplot as plt
import scipy
from src.data.load import get_data
from src.utils.noise import add_noise
from src.modelling.deconvolution import estimate_h_by_deconv
from src.modelling.wienerhopf import estimate_h_by_wiener_hopf

io_pair = get_data("data/estimation", val=False)
val_pair = get_data("data/validation", val=True)
sig_range = range(0, 599)

plt.style.use("ggplot")

for label, signals in io_pair.items():
    input_signal = signals["input"].reshape(-1)[sig_range]
    impulse_signal = signals["impulse"][:, 0][sig_range][:100]
    output_signal = signals["output"][:, 0][sig_range]
    output_signal = add_noise(output_signal - output_signal[0], SNR=20)
    plt.plot(input_signal, label="input")
    plt.plot(output_signal, label="output")
    plt.title(label)
    plt.legend()
    plt.show()
    h_deconv = estimate_h_by_deconv(input_signal, output_signal)[:100]
    h_wiener = estimate_h_by_wiener_hopf(input_signal, output_signal)[:100]

    plt.plot(impulse_signal, label="impulso verdadeiro")
    plt.title(f"Resposta ao Impulso de {label}")
    plt.legend()
    plt.show()

    plt.plot(h_deconv, label="deconv")
    plt.title(f"Aproximação de Resposta ao Impulso de {label} via Deconvolução")
    plt.legend()
    plt.show()

    plt.plot(h_wiener, label="wiener")
    plt.title(f"Aproximação de Resposta ao Impulso de {label} via Wiener Hopf")
    plt.legend()
    plt.show()

    val_signal = val_pair[label]
    output_signal = val_signal[:, 0]
    output_signal = output_signal - output_signal[0]
    input_signal = val_pair["Entrada_1"].reshape(-1)

    wiener_approx = scipy.signal.convolve(input_signal, h_wiener)[:-100]
    plt.plot((wiener_approx/max(wiener_approx))[:1000], label="Aproximação Wiener Hopf")
    plt.plot((output_signal/max(output_signal))[:1000], label="Sinal Real")
    plt.title("Aproximação de Sinal de Saída via Wiener Hopf")
    plt.legend()
    plt.show()

    deconv_approx = scipy.signal.convolve(input_signal, h_deconv)[:-100]
    plt.plot((deconv_approx/max(deconv_approx))[:1000], label="Aproximação Deconvolução")
    plt.plot((output_signal/max(output_signal))[:1000], label="Sinal Real")
    plt.title("Aproximação de Sinal de Saída via Deconvolução")
    plt.legend()
    plt.show()
