import sys
import matplotlib.pyplot as plt
import numpy as np
from src.data.load import get_white, get_step
from src.data.prep import prep_signals
from src.modelling.arx import visualize_io, visualize_estimation
from src.modelling.arx import get_regressors, get_mq_estimation
from src.modelling.arx import get_rmse, get_param_covariance
from src.modelling.arx import get_arx_estimate, validate_arx_estimate
from src.modelling.arx import get_aic_matrix
from src.modelling.arx import analyze_residue
from src.modelling.arx import analyze_params

plt.style.use("ggplot")


def model_arx(out, variable, free, aic_matrix=False):

    freepath = "free" if free else "onestepahead"

    est, val = get_white(out, variable)
    u, y, time = prep_signals(est)

    if aic_matrix:
        get_aic_matrix(time, u, y, 20, 20)

    ar_order, x_order = 6, 6

    theta, est_error, residue, AIC = get_arx_estimate(time, u, y, ar_order,
                                                      x_order,
                                                      f"images/{freepath}/white_id_{out}_{variable}",
                                                      visual=True)

    analyze_residue(y, u, residue, fn=f"images/{freepath}/white_residue_{out}_{variable}")

    u_val, y_val, time = prep_signals(val)

    val_error = validate_arx_estimate(time, u_val, y_val, ar_order, x_order, theta,
                                      fn=f"images/{freepath}/white_white_val_{out}_{variable}",
                                      free=free)

    step_est, step_val = get_step(out, variable)

    u, y, time = prep_signals(step_est, slice(0, 1000))

    u_val, y_val, val_time = prep_signals(step_est, slice(3800, 4499))
    u_swing, y_swing, swing_time = prep_signals(step_est, slice(2600, 3400))

    up_val_error = validate_arx_estimate(time, u, y, ar_order, x_order, theta,
                                         fn=f"images/{freepath}/white_ustep_val_{out}_{variable}",
                                         free=free)

    down_val_error = validate_arx_estimate(val_time, u_val, y_val, ar_order,
                                           x_order, theta,
                                           fn=f"images/{freepath}/white_downstep_val_{out}_{variable}",
                                           free=free)

    down_swing_error = validate_arx_estimate(swing_time, u_swing, y_swing,
                                             ar_order, x_order, theta,
                                             fn=f"images/{freepath}/white_swingstep_val_{out}_{variable}",
                                             free=free)

    theta, est_error, residue, AIC = get_arx_estimate(time, u, y, ar_order,
                                                      x_order,
                                                      fn=f"images/{freepath}/step_val_{out}_{variable}",
                                                      visual=True)

    analyze_residue(y, u, residue, fn=f"images/{freepath}/step_residue_{out}_{variable}")

    u_val, y_val, val_time = prep_signals(step_est, slice(3800, 4499))
    u_swing, y_swing, swing_time = prep_signals(step_est, slice(2600, 3400))

    up_val_error = validate_arx_estimate(time, u, y, ar_order, x_order, theta,
                                         fn=f"images/{freepath}/step_upstep_val_{out}_{variable}",
                                         free=free)

    down_val_error = validate_arx_estimate(val_time, u_val, y_val, ar_order,
                                           x_order, theta,
                                           fn=f"images/{freepath}/step_downstep_val_{out}_{variable}",
                                           free=free)

    down_swing_error = validate_arx_estimate(swing_time, u_swing, y_swing,
                                             ar_order, x_order, theta,
                                             fn=f"images/{freepath}/step_swingstep_val_{out}_{variable}",
                                             free=free)


if __name__ == "__main__":
    for variable in ("temperature", "concentration"):
        for out in ("Te_var", "Ta_var", "Ca_var", "Q_var"):
            for free in (True, False):
                model_arx(out, variable, free)
