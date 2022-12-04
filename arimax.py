from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.data.load import get_data
from scipy.ndimage import shift
import numpy as np


def estimate_arimax(signal, label, arima_order, x_order, visual=True):

    out = signal["output"][:, 0]
    input_ = signal["input"].reshape(-1)

    exog = np.zeros((len(out), x_order))

    for i in range(x_order):
        exog[:, i] = shift(input_, 1-i)

    model = ARIMA(endog=out, exog=exog, order=arima_order).fit()

    prediction = model.predict(endog=out, exog=exog)

    if visual:
        plt.plot(out[10:], label="out")
        plt.plot(prediction[10:], label="prediction")
        plt.title(label)
        plt.legend()
        plt.show()

    residue = out[10:] - prediction[10:]
    plot_acf(residue)
    plt.show()


io_pairs = get_data("data/estimation", val=False)

for label, signals in io_pairs.items():

    estimate_arimax(signals, label, arima_order=(5, 1, 5), x_order=5)


io_pairs = get_data("data/validation", val=True)
