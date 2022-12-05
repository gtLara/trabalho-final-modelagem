import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from src.data.load import get_data
from src.utils.noise import add_noise
from src.modelling.deconvolution import estimate_h_by_deconv
from src.modelling.wienerhopf import estimate_h_by_wiener_hopf

dt = 0.01
num = [1, 0.5]
den = [1, -0.5, 0.25]
n_samples = 500

system = signal.TransferFunction(num, den, dt=dt)

time = np.arange(0, n_samples*dt, dt)

delta = np.zeros(len(time))
delta[0] = 1

_, output = system.output(delta, time)

random_input = np.random.normal(size=len(time))
time, random_output = system.output(random_input, time)

h_deconv = estimate_h_by_deconv(random_input, random_output)
h_wiener = estimate_h_by_wiener_hopf(random_input, random_output.reshape(-1))
plt.plot(h_deconv[:100]/max(h_deconv), label="deconv")
plt.plot(h_wiener[:100]/max(h_wiener), label="wiener")
plt.plot(output[:100]/max(output), label="actual")
plt.legend()
plt.show()
