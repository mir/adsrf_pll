import numpy as np
import torch

class PLL():
    def __init__(self,
                 Ts = 10e-4, a = 0.27198, b = 0.5055, omega_e = 10., Omega = 0.144,
                 x0=0., theta_e0=0.):
        self.x = torch.tensor(x0)
        self.theta_e = torch.tensor(theta_e0)
        self.Ts = torch.tensor(Ts)
        self.a = torch.tensor(a)
        self.b = torch.tensor(b)
        self.omega_e = torch.tensor(omega_e)
        self.Omega = torch.tensor(Omega)
        self.k = torch.tensor(0)

    def step(self):
        x_prev = self.x
        self.x = self.b * self.x + torch.sin(self.theta_e)
        self.theta_e = self.theta_e \
                       + self.omega_e * self.Ts \
                       - (self.Omega * (1 - self.a) * (1 - self.b) * x_prev
                          + self.a * self.Omega * torch.sin(self.theta_e))
        self.k += 1
        return self.x, self.theta_e

    def detach(self):
        self.vco_freq = self.vco_freq.detach()
        self.vco_phase = self.vco_phase.detach()
        self.filter_state = self.filter_state.detach()

    def steps(self, count = 1000):
        x = torch.zeros(count)
        theta_e = torch.zeros(count)
        for i in range(count):
            x[i] = self.x
            theta_e[i] = self.theta_e
            self.step()
            self.k += 1

        return x, theta_e