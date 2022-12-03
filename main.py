import matplotlib.pyplot as plt
from src.data.load import get_data
from src.utils.noise import add_noise
from src.modelling.deconvolution import estimate_h_by_deconv

signals = get_data("data/")

for s in signals:
    for i in range(2):
        signal = signals[s][:, i]
        signals[s][:, i] = add_noise(signal, SNR=20)
