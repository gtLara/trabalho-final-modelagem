import numpy as np
from scipy.linalg import inv
from src.modelling.utils import create_convolution_matrix


def get_h(output_signal: np.ndarray,
          conv_matrix: np.ndarray) -> np.ndarray:

    dim_msg = "Dimensão de matriz de convolução deve ser igual ao tamanho da saída"

    assert len(conv_matrix) == len(output_signal), dim_msg

    h = inv(conv_matrix) @ output_signal

    return h


def estimate_h_by_deconv(input_signal: np.ndarray,
                         output_signal: np.ndarray,
                         n_samples: int = None) -> np.ndarray:

    conv_matrix = create_convolution_matrix(input_signal, size=n_samples)
    h = get_h(output_signal, conv_matrix)

    return h
