import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data.load import get_white, get_step
from src.data.prep import prep_signals
from src.modelling.arx import visualize_io, visualize_estimation
from src.modelling.arx import get_regressors, get_mq_estimation
from src.modelling.arx import get_rmse, get_param_covariance
from src.modelling.arx import get_arx_estimate, validate_arx_estimate
from src.modelling.arx import analyze_residue

# TODO: make code prettier

est, val = get_white()
u, y, time = prep_signals(est)

AIC_range = 5
AIC_matrix = np.zeros((AIC_range, AIC_range))

for ar_order in range(1, AIC_range + 1):
    for x_order in range(1, AIC_range + 1):

        theta, est_error, residue, AIC = get_arx_estimate(time, u, y, ar_order,
                                                          x_order,
                                                          visual=False)

        AIC_matrix[ar_order-1][x_order-1] = AIC

sns.heatmap(AIC_matrix, annot=False)
plt.show()

ar_order, x_order = 10, 10

# cool idea: find first model to hit white
# residue

theta, est_error, residue, AIC = get_arx_estimate(time, u, y, ar_order,
                                                  x_order,
                                                  visual=False)

analyze_residue(u, residue)

u_val, y_val, time = prep_signals(val)

val_error = validate_arx_estimate(time, u_val, y_val, ar_order, x_order, theta)

step_est, step_val = get_step()

u, y, time = prep_signals(step_est, slice(0, 1000))

theta, est_error, residue, AIC = get_arx_estimate(time, u, y, ar_order,
                                                  x_order)

u_val, y_val, val_time = prep_signals(step_est, slice(3800, 4499))
u_swing, y_swing, swing_time = prep_signals(step_est, slice(2600, 3400))

up_val_error = validate_arx_estimate(time, u, y, ar_order, x_order, theta,
                                     True)

down_val_error = validate_arx_estimate(val_time, u_val, y_val, ar_order,
                                       x_order, theta, True)

down_swing_error = validate_arx_estimate(swing_time, u_swing, y_swing,
                                         ar_order, x_order, theta, True)
