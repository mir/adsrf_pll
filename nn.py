from typing import Any

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
import pll
import helpers as hp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

input_size = 1
output_size = 64
learning_rate = 0.01
epochs = 32*32*8

hidden_dim = 32
theta_e_dim = 8
theta_e_range = torch.arange(0, 2 * np.pi, 2 * np.pi / theta_e_dim)

Ts = 10e-4
a = 0.27198
b = 0.5055
omega_e = 10.
Omega = 0.144

labels = torch.zeros(theta_e_dim, output_size)

for i, theta_e0 in enumerate(theta_e_range):
    a_pll = pll.PLL(Ts, a, b, omega_e, Omega,
                    0., theta_e0)
    x_k, thetha_e_k = a_pll.steps(output_size)
    labels[i] = thetha_e_k


# Neural network model
class NeuralPLLModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, input_size * hidden_dim),
            nn.ReLU(),
            nn.Linear(input_size * hidden_dim, output_size),
        )

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, theta_e0_batch, labels):
        out = self(theta_e0_batch)
        loss_fn = nn.MSELoss()
        return loss_fn(out, labels)

def fit(epochs,
        learning_rate,
        model,
        opt_func=torch.optim.Adam):
    """Train the model using gradient descent"""
    losses = []
    optimizer = opt_func(model.parameters(), learning_rate)
    for it in range(epochs):
        hp.print_progress(epochs, it)

        input_batch = theta_e_range.view(-1, 1)

        loss = model.training_step(input_batch, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # log data
        losses.append(loss.item())

    return losses


# generate data and data loaders

model = NeuralPLLModel()

fig, (ax1, ax2) = plt.subplots(2, 2)
fig.suptitle('SRF-PLL simulation NN')

# Plot reference signal
ax1[0].set_title('Theta_e_k (thetha_e0 = 0)')
ax1[0].set_xlabel('k')
ax1[0].plot(labels[0])

nn_theta_e_k = model(theta_e_range[1].view(1, 1)).detach().flatten()
ax1[1].set_title('NN vs real')
ax1[1].plot(labels[1])
ax1[1].plot(nn_theta_e_k, color='r')

nn_theta_e_k = model(theta_e_range[2].view(1, 1)).detach().flatten()
ax2[1].set_title('NN vs real')
ax2[1].plot(labels[2])
ax2[1].plot(nn_theta_e_k, color='r')

print('Train NN')
losses = fit(epochs, learning_rate, model)

nn_theta_e_k = model(theta_e_range[1].view(1, 1)).detach().flatten()
ax1[1].plot(nn_theta_e_k, color='g')

nn_theta_e_k = model(theta_e_range[2].view(1, 1)).detach().flatten()
ax2[1].plot(nn_theta_e_k, color='g')

ax2[0].set_title('Training losses')
ax2[0].set_ylabel('Loss')
ax2[0].set_yscale('log')
ax2[0].set_xlabel('epoch')
ax2[0].plot(losses)

plt.show()
