from torch import nn
import torch
import numpy as np


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.input_layer = SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)
        self.hidden_layers = nn.ModuleList()

        for _ in range(hidden_layers):
            self.hidden_layers.append(
                SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            self.output_layer = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                self.output_layer.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                                  np.sqrt(6 / hidden_features) / hidden_omega_0)
        else:
            self.output_layer = SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0)

        self.outermost_linear = outermost_linear

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False):
        super().__init__()
        self.input_layer = nn.Linear(in_features, hidden_features)
        self.hidden_layers = nn.ModuleList()
        self.outernmost_linear = outermost_linear

        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_features, hidden_features))

        self.output_layer = nn.Linear(hidden_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            residual = x
            x = self.activation(layer(x)) + residual  # Adding residual connection for each layer
        x = self.output_layer(x)
        if not self.outernmost_linear:
            x = self.activation(x)
        return x
