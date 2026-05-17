"""
DroughtLSTM
===========
Multi-output LSTM for direct 5-step drought score forecasting.

Architecture
------------
  Input  → LSTM (hidden_size, num_layers, dropout)
         → last hidden state
         → Dropout
         → Linear(hidden_size, horizon=5)
  Output → 5 simultaneous score predictions (direct multi-step)

Direct multi-step (vs. recursive / seq2seq) avoids error accumulation
across forecast steps.
"""

import torch
import torch.nn as nn


class DroughtLSTM(nn.Module):
    """
    Parameters
    ----------
    input_size  : number of features per time step (F)
    hidden_size : LSTM hidden dimensionality
    num_layers  : number of stacked LSTM layers
    dropout     : dropout probability (applied between LSTM layers and before FC)
    horizon     : number of future weeks to forecast simultaneously
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        horizon: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # inter-layer dropout; disabled automatically when num_layers==1
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout)

        # Direct multi-step head: output all H forecasts at once
        self.fc = nn.Linear(hidden_size, horizon)

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        out : (batch, horizon)
        """
        lstm_out, _ = self.lstm(x)          # (B, W, hidden)
        last_hidden = lstm_out[:, -1, :]    # (B, hidden) – last time step
        dropped = self.dropout(last_hidden)
        out = self.fc(dropped)              # (B, horizon)
        return out

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lines = [
            "DroughtLSTM Architecture",
            "=" * 40,
            f"  Input size   : {input_size}  (features per week)",
            f"  LSTM layers  : {self.num_layers}",
            f"  Hidden size  : {self.hidden_size}",
            f"  Dropout      : {self.dropout.p}",
            f"  Output (FC)  : {self.horizon}  (direct 5-step forecast)",
            "-" * 40,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
