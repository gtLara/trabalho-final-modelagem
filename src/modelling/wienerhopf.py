import numpy as np
from src.modelling.deconvolution import get_h
from src.modelling.utils import create_convolution_matrix, get_h


def get_autocorr(signal: np.ndarray) -> np.ndarray:

    autocorr = get_crosscorr(signal, signal)

    return autocorr


def get_crosscorr(signal_a: np.ndarray,
                  signal_b: np.ndarray) -> np.ndarray:

    crosscorr = np.correlate(signal_a, signal_b, mode="full")

    return crosscorr


def estimate_h_by_wiener_hopf(input_signal: np.ndarray,
                              output_signal: np.ndarray,
                              n_samples: int = None) -> np.ndarray:

    autocorr_input = get_autocorr(input_signal)
    crosscorr_input_output = get_crosscorr(input_signal, output_signal)

    conv_matrix = create_convolution_matrix(autocorr_input, size=n_samples)
    h = get_h(crosscorr_input_output, conv_matrix)

    return h
