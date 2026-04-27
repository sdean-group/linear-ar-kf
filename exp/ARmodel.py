from __future__ import annotations

"""Model definitions for the linear autoregressive experiments."""

import torch
import torch.nn as nn


def _init_linear_weights(layer: nn.Linear) -> None:
    nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")


class TwoLayerLinearAR(nn.Module):
    """Two-layer linear autoregressive model."""

    def __init__(self, input_size: int, intermediate_size: int, output_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, intermediate_size, bias=False)
        self.linear2 = nn.Linear(intermediate_size, output_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_linear_weights(self.linear1)
        _init_linear_weights(self.linear2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.linear1(x)
        outputs = self.linear2(hidden)
        return outputs, hidden


class MultiLayerLinearAR(nn.Module):
    """Configurable linear autoregressive model with multiple hidden layers."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one hidden dimension.")

        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size, bias=False))
            current_size = hidden_size

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(current_size, output_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.hidden_layers:
            _init_linear_weights(layer)
        _init_linear_weights(self.output_layer)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        activations = []
        hidden = x
        for layer in self.hidden_layers:
            hidden = layer(hidden)
            activations.append(hidden)

        outputs = self.output_layer(hidden)
        return outputs, activations
