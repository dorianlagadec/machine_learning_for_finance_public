import torch
import torch.nn as nn
from typing import Optional, List
from .Rnn import Dense, Trainer, ACTIVATIONS, LossFunction, Optimizer, Activation, Initializer


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        lstm_dropout: float = 0.0,
        bidirectional: bool = False,
        head: Optional[List[Dense]] = None,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.head = nn.ModuleList(head) if head is not None else nn.ModuleList([])

    def forward(self, x: torch.Tensor, hc0=None) -> torch.Tensor:
        out, _ = self.lstm(x, hc0)
        last = out[:, -1, :]   # last timestep: (batch, hidden_size * num_directions)

        for layer in self.head:
            last = layer(last)

        return last
