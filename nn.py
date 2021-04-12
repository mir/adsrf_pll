from typing import Any

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
import pll
import helpers as hp
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

input_size = 2
output_size = 16
learning_rate = 0.01
epochs = 32*32*2

hidden_dim = 32
theta_e_dim = 8
theta_e_range = torch.arange(0, 2 * np.pi, 2 * np.pi / theta_e_dim)
omega_e_dim = 8
omega_e_max = 10.
omega_e_range = torch.arange(-omega_e_max, omega_e_max, 2 * omega_e_max / omega_e_dim)
total_dim = theta_e_dim*omega_e_dim

test_points_number = 16

def generate_test_points(test_points_number):
    test_points = torch.rand(test_points_number, 2)
    omega_s = torch.zeros(test_points_number, 1) + 2*omega_e_max
    theta_s = torch.zeros(test_points_number, 1) + 2 * np.pi
    scale = torch.cat((theta_s,omega_s), 1)
    shift_omega = torch.zeros(test_points_number, 1) -omega_e_max
    shift = torch.cat((torch.zeros(test_points_number, 1), shift_omega), 1)
    test_points = test_points * scale + shift
    return test_points

test_points = generate_test_points(test_points_number)

Ts = 10e-4
a = 0.27198
b = 0.5055
Omega = 0.144

inputs = torch.zeros(total_dim, 2)
labels = torch.zeros(total_dim, output_size)
for i_theta, theta_e0 in enumerate(theta_e_range):
    for i_omega, omega_e in enumerate(omega_e_range):
        total_i = i_theta*theta_e_dim + i_omega
        a_pll = pll.PLL(Ts, a, b, omega_e, Omega,
                    0., theta_e0)
        x_k, theta_e_k = a_pll.steps(output_size)
        labels[total_i] = theta_e_k
        inputs[total_i] = torch.tensor([theta_e0, omega_e])

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

    def training_step(self, batch, labels):
        out = self(batch)
        loss_fn = nn.MSELoss()
        return loss_fn(out, labels)

    def eval(self, theta_e0, omega_e):
        input = torch.tensor([theta_e0, omega_e]).view(1, input_size)
        out = self(input)
        out = out[0].detach()
        return out

    def max_error(self, test_points):
        acc_fn = nn.MSELoss()
        max_error = 0
        for step in range(test_points_number):
            omega_e = test_points[step, 1]
            theta_e0 = test_points[step, 0]
            test_pll = pll.PLL(Ts, a, b, omega_e, Omega,
                               0., theta_e0)
            x_k, theta_e_k = test_pll.steps(output_size)
            nn_theta_e_k = self.eval(theta_e0, omega_e)
            max_error = max(max_error, acc_fn(theta_e_k, nn_theta_e_k))
        return max_error

def fit(epochs,
        learning_rate,
        model,
        opt_func=torch.optim.Adam):
    """Train the model using gradient descent"""
    losses = []
    max_errors = []
    optimizer = opt_func(model.parameters(), learning_rate)
    for it in range(epochs):
        hp.print_progress(epochs, it)

        loss = model.training_step(inputs, labels)
        max_error = model.max_error(test_points)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # log data
        losses.append(loss.item())
        max_errors.append(max_error)

    return losses, max_errors

# generate data and data loaders

model = NeuralPLLModel()

fig, (ax1, ax2) = plt.subplots(2, 2)
fig.suptitle('SRF-PLL simulation NN')

print('Train NN')
losses, max_errors = fit(epochs, learning_rate, model)

def plot_random_nn_vs_real(axe):
    omega_e = random.random() * 2 * omega_e_max - omega_e_max
    theta_e0 = random.random() * 2 * np.pi
    axe.set_title('NN vs real')
    test_pll = pll.PLL(Ts, a, b, omega_e, Omega,
                       0., theta_e0)
    x_k, theta_e_k = test_pll.steps(output_size)
    axe.plot(theta_e_k)
    nn_theta_e_k = model.eval(theta_e0, omega_e)
    axe.plot(nn_theta_e_k, color='g')

plot_random_nn_vs_real(ax1[0])
plot_random_nn_vs_real(ax1[1])

ax2[0].set_title('Training losses')
ax2[0].set_ylabel('Loss')
ax2[0].set_yscale('log')
ax2[0].set_xlabel('epoch')
ax2[0].plot(losses)

ax2[1].set_title('Max errors')
ax2[1].set_ylabel('max error')
ax2[1].set_yscale('log')
ax2[1].set_xlabel('epoch')
ax2[1].plot(max_errors)

plt.show()
