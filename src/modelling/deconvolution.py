import numpy as np
from scipy.linalg import inv
from src.modelling.utils import create_convolution_matrix, get_h


def estimate_h_by_deconv(input_signal: np.ndarray,
                         output_signal: np.ndarray,
                         n_samples: int = None) -> np.ndarray:

    conv_matrix = create_convolution_matrix(input_signal, size=n_samples)
    h = get_h(output_signal, conv_matrix)

    return h
