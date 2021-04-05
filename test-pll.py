from typing import Any

import torch
import matplotlib.pyplot as plt
# import torch.nn as nn
import numpy as np
import os
import pll

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

Ts = 10e-4
a = 0.27198
b = 0.5055
omega_e = 10.
Omega = 0.144

max_steps = 128
theta_e_dim = 8
x_dim = 64
x_max = 10*omega_e*Ts/(1-b)/Omega # 10 times equilibrium
input_dim = theta_e_dim*x_dim

theta_e_range = torch.arange(0,2*np.pi,2*np.pi/theta_e_dim)
x_range = torch.arange(-x_max,x_max,2*x_max/x_dim)

for theta_e0 in theta_e_range:
    a_pll = pll.PLL(Ts, a, b, omega_e, Omega,
             0., theta_e0)
    x_k, thetha_e_k = a_pll.steps(max_steps)


fig, (ax1, ax2) = plt.subplots(2, 2)
fig.suptitle('Digital SRF-PLL')

# Plot reference signal
ax1[0].set_title('x_k')
ax1[0].set_xlabel('time (k)')
ax1[0].plot(x_k)

ax1[1].set_title('theta_e_k')
ax1[0].set_xlabel('time (k)')
ax1[1].plot(thetha_e_k)

plt.show()
