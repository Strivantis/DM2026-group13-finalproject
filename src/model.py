"""
DroughtLSTM
===========
Multi-output LSTM for direct 5-step drought score forecasting.

Architecture (v3)
-----------------
  Input  -> LSTM (hidden_size=64, num_layers=2, dropout=0.4)
         -> last hidden state
         -> Dropout(0.4)
         -> Linear(64, horizon=5)  <- bias initialised to -0.98
         -> Sigmoid() x 5.0          <- natural [0, 5] bound; replaces clip()

Changes from v2
---------------
  - Bias init     : fc.bias set to -0.98 so Sigmoid(-0.98)*5 ~= 1.36, matching
                    the dataset's mean target value and avoiding wasted early
                    epochs spent adjusting the output baseline.

Bias derivation
---------------
  Dataset mean target ~= 1.36 on a [0, 5] scale.
  Output = Sigmoid(b) * 5.0  ->  Sigmoid(b) = 1.36 / 5.0 = 0.272
  b = ln(0.272 / (1 - 0.272)) = ln(0.272 / 0.728) ~= ln(0.3736) ~= -0.9847 -> -0.98
"""

import torch
import torch.nn as nn


class DroughtLSTM(nn.Module):
    """
    Parameters
    ----------
    input_size  : number of features per time step (F)
    hidden_size : LSTM hidden dimensionality  (default 64)
    num_layers  : number of stacked LSTM layers
    dropout     : dropout probability (applied between LSTM layers and before FC)
    horizon     : number of future weeks to forecast simultaneously
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.4,
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

        # Bias init: Sigmoid(-0.98) * 5.0 ~= 1.36 == dataset mean target.
        # Eliminates wasted epochs driving the output baseline down from 2.5.
        nn.init.constant_(self.fc.bias, -0.98)

        # Natural output bound: Sigmoid maps to (0, 1), scaled to (0, 5).
        # Replaces post-processing clip(0, 5).
        self.output_activation = nn.Sigmoid()
        self._output_scale = 5.0

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        out : (batch, horizon)  -- values in (0, 5)
        """
        lstm_out, _ = self.lstm(x)          # (B, W, hidden)
        last_hidden = lstm_out[:, -1, :]    # (B, hidden) -- last time step
        dropped = self.dropout(last_hidden)
        raw = self.fc(dropped)              # (B, horizon)
        out = self.output_activation(raw) * self._output_scale  # (B, horizon)
        return out

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lines = [
            "DroughtLSTM Architecture (v3)",
            "=" * 40,
            f"  Input size   : {input_size}  (features per week)",
            f"  LSTM layers  : {self.num_layers}",
            f"  Hidden size  : {self.hidden_size}",
            f"  Dropout      : {self.dropout.p}",
            f"  Output (FC)  : {self.horizon}  (direct 5-step forecast)",
            f"  FC bias init : -0.98  (Sigmoid(-0.98)*5 ~= 1.36 == dataset mean)",
            f"  Activation   : Sigmoid x {self._output_scale}  -> (0, {self._output_scale})",
            "-" * 40,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
