from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

__all__ = ["MLPConfig", "MLP"]


@dataclass
class MLPConfig:
    """Configuration for a simple multilayer perceptron."""

    in_dim: int
    out_dim: int
    hidden_dims: Sequence[int] = (128, 64)
    dropout: float = 0.1


class MLP(nn.Module):
    """Feedforward network for tabular classification/regression."""

    def __init__(self, config: MLPConfig):
        super().__init__()

        if config.in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {config.in_dim}")
        if config.out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {config.out_dim}")

        layers: list[nn.Module] = []
        in_features = config.in_dim

        for width in config.hidden_dims:
            if width <= 0:
                raise ValueError(f"hidden_dims must be positive, got {config.hidden_dims}")
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            if config.dropout and config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_features = width

        layers.append(nn.Linear(in_features, config.out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
