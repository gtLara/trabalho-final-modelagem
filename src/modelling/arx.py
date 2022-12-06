from statsmodels.tsa.stattools import acf, ccf
from statsmodels.tsa.arima.model import ARIMA
from scipy.ndimage import shift
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_io(time, input_, output, title, fn):

    plt.plot(time, input_, color="blue", label="Entrada")
    plt.plot(time, output, color="black", label="Saída")
    plt.title(title)
    plt.legend()
    plt.savefig(fn)
    plt.close()


def visualize_estimation(time, P, P_hat, title, theta_cov, error, fn):

    plt.plot(time, P_hat, color="red", linestyle="--", label="Estimativa")
    plt.plot(time, P, color="black", label="Saída Real")
    plt.plot([], alpha=0,
             label=f"Covariância de Parâmetros:{theta_cov:.2f}")
    plt.plot([], alpha=0, label=f"RMSE:{error:.2f}")
    plt.legend()
    plt.title(title)
    plt.savefig(fn)
    plt.close()


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


def get_arx_estimate(time, u, y, ar_order, x_order, fn, visual=True):

    regressors = get_regressors(u, y, ar_order, x_order)
    theta = get_mq_estimation(regressors, y)

    y_hat = regressors @ theta

    error = get_rmse(y_hat, y)
    cov = get_param_covariance(theta)

    if visual:
        # visualize_io(time, u, y, "Visualização de Dados de Estimação")

        visualize_estimation(time, y, y_hat, "Visualização de Estimação ARX",
                             cov, error, fn)

    residue = y_hat - y

    AIC = get_AIC(len(u), np.var(residue), len(theta))

    return theta, error, residue, AIC


def validate_arx_estimate(time, u, y, ar_order, x_order, theta, fn,
                          free=False):

    # visualize_io(time, u, y, "Visualização de Dados de Validação")

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
                         error, fn)

    return error


def analyze_residue(output, input_, residue, fn):

    plt.subplot(2, 1, 1)

    tol = 1.96/np.sqrt(len(input_))
    plt.axhline(tol, color="black", linestyle="--")
    plt.axhline(-tol, color="black", linestyle="--")

    plt.stem(acf(residue, nlags=15))
    plt.title("Autocorrelação de Resíduos")

    plt.subplot(2, 1, 2)

    tol = 1.96/np.sqrt(len(input_))
    plt.axhline(tol, color="black", linestyle="--")
    plt.axhline(-tol, color="black", linestyle="--")

    cross_cor = ccf(residue, input_)[5:21]

    plt.stem(cross_cor)
    plt.title("Correlação Cruzada entre Resíduos e Entrada")

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def analyze_params(theta, ar_order=6, x_order=6):

    x = theta[:x_order]
    ar = theta[-ar_order:]

    x_roots = np.roots(x)
    ar_roots = np.roots(ar)
    print("Raizes")
    print(f"X: {x_roots}")
    print(f"AR: {ar_roots}")

    x_mags = [round(r, 2) for r in np.abs(x_roots)]
    ar_mags = [round(r, 2) for r in np.abs(ar_roots)]
    print("Abs")
    print(f"X: {x_mags}")
    print(f"AR: {ar_mags}")

    return


def get_aic_matrix(time, u, y, AIC_range, visual=True):

    AIC_matrix = np.zeros((AIC_range, AIC_range))

    for ar_order in range(1, AIC_range + 1):
        for x_order in range(1, AIC_range + 1):

            theta, est_error, residue, AIC = get_arx_estimate(time, u, y,
                                                              ar_order,
                                                              x_order,
                                                              visual=False)

            AIC_matrix[ar_order-1][x_order-1] = AIC

    if visual:
        sns.heatmap(AIC_matrix)
        plt.show()

    return AIC_matrix


def get_multiple_params(time, u, y, ar_order, x_order, runs):

    ar_thetas = np.zeros((runs, ar_order))
    x_thetas = np.zeros((runs, x_order))

    for i in range(runs):

        theta, _, _, _ = get_arx_estimate(time, u, y, ar_order, x_order,
                                          visual=False)

        ar_thetas[i, :] = theta[:x_order]
        x_thetas[i, :] = theta[-ar_order:]

    return ar_thetas, x_thetas
