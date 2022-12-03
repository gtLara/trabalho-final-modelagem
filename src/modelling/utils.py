import numpy as np
from scipy.ndimage import shift


def create_convolution_matrix(input_signal: np.ndarray,
                              size: int = None) -> np.ndarray:

    if size is None:
        size = len(input_signal)

    size_msg = "Tamanho de matriz de convolução deve ser pelo menos do tamanho do sinal"
    assert len(input_signal) <= size, size_msg

    conv_matrix = np.zeros((size, size))
    input_signal = np.pad(input_signal, (0, size-len(input_signal)))

    for u in range(size):
        conv_matrix[u] = shift(np.flip(input_signal),
                               -(len(input_signal) - 1) + u)

    return conv_matrix
