import scipy
import numpy as np
from src.modelling.deconvolution import get_h
from src.modelling.utils import create_convolution_matrix, get_h


def get_autocorr(signal: np.ndarray) -> np.ndarray:

    autocorr = get_crosscorr(signal, signal)

    return autocorr


def get_crosscorr(signal_a: np.ndarray,
                  signal_b: np.ndarray) -> np.ndarray:

    crosscorr = scipy.correlate(signal_a, signal_b, mode="full")

    return crosscorr


def estimate_h_by_wiener_hopf(random_input_signal: np.ndarray,
                              output_signal: np.ndarray,
                              n_samples: int = None) -> np.ndarray:

    crosscorr_input_output = get_crosscorr(random_input_signal, output_signal)

    h = crosscorr_input_output/np.var(random_input_signal)

    center = len(h)//2

    return h[center-4:]
