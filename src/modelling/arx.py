from statsmodels.tsa.stattools import acf, ccf
from statsmodels.tsa.arima.model import ARIMA
from scipy.ndimage import shift
import numpy as np
import matplotlib.pyplot as plt


def visualize_io(time, input_, output, title):

    plt.plot(time, input_, color="blue", label="Entrada")
    plt.plot(time, output, color="black", label="Saída")
    plt.title(title)
    plt.legend()
    plt.show()


def visualize_estimation(time, P, P_hat, title, theta_cov, error):

    plt.plot(time, P_hat, color="red", linestyle="--", label="Estimativa")
    plt.plot(time, P, color="black", label="Saída Real")
    plt.plot([], alpha=0,
             label=f"Covariância de Parâmetros:{theta_cov:.2f}")
    plt.plot([], alpha=0, label=f"RMSE:{error:.2f}")
    plt.legend()
    plt.title(title)
    plt.show()


def get_regressors(input_, output, AR_order, X_order):

    assert len(input_) == len(output)

    regressor_matrix = np.zeros((len(input_), AR_order + X_order))

    for x_lag in range(X_order):
        regressor_matrix[:, x_lag] = shift(input_, (x_lag+1))

    for ar_lag in range(AR_order):
        regressor_matrix[:, ar_lag + X_order] = shift(output, (ar_lag+1))

    return regressor_matrix


def get_mq_estimation(regressor_matrix, output):

    theta_mq = np.linalg.pinv(regressor_matrix) @ output

    return theta_mq


def get_rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())


def get_param_covariance(params):

    cov = float(np.cov(params))

    return cov


def get_AIC(n_samples, residue_var, n_params):

    AIC = n_samples * np.log(residue_var) * 2 * n_params

    return AIC


def get_arx_estimate(time, u, y, ar_order, x_order, visual=True):

    regressors = get_regressors(u, y, ar_order, x_order)
    theta = get_mq_estimation(regressors, y)

    y_hat = regressors @ theta

    error = get_rmse(y_hat, y)
    cov = get_param_covariance(theta)

    if visual:
        visualize_io(time, u, y, "Visualização de Dados de Estimação")

        visualize_estimation(time, y, y_hat, "Visualização de Estimação ARX",
                             cov, error)

    residue = y_hat - y

    AIC = get_AIC(len(u), np.var(residue), len(theta))

    return theta, error, residue, AIC


def validate_arx_estimate(time, u, y, ar_order, x_order, theta,
                          free=False):

    visualize_io(time, u, y, "Visualização de Dados de Validação")

    if free:

        N = len(u)
        y_hat = np.zeros(N)
        y_hat[:50] = y[:50]

        for s in range(50, N):
            regressors = get_regressors(u[:s], y_hat[:s], ar_order,
                                        x_order)

            y_hat[s] = (regressors @ theta)[-1]

    else:

        regressors = get_regressors(u, y, ar_order, x_order)
        y_hat = regressors @ theta

    error = get_rmse(y_hat, y)
    cov = get_param_covariance(theta)

    visualize_estimation(time, y, y_hat, "Visualização de Validação ARX", cov,
                         error)

    return error


def analyze_residue(input_, residue):

    plt.subplot(2, 1, 1)

    tol = 1.96/np.sqrt(len(input_))
    plt.axhline(tol, color="black", linestyle="--")
    plt.axhline(-tol, color="black", linestyle="--")

    plt.stem(acf(residue, nlags=15))

    plt.subplot(2, 1, 2)

    tol = 1.96/np.sqrt(len(input_))
    plt.axhline(tol, color="black", linestyle="--")
    plt.axhline(-tol, color="black", linestyle="--")

    cross_cor = ccf(residue, input_)[5:21]

    plt.stem(cross_cor)

    plt.show()
