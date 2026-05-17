"""
DroughtLSTM
===========
Multi-output LSTM for direct 5-step drought score forecasting.

Architecture (v6 – Anti-Collapse Refactor)
-------------------------------------------
  Input  -> LSTM (hidden_size=64, num_layers=2, dropout=0.4)
         -> last hidden state
         -> Dropout(0.4)
         -> Linear(64, 32) -> GELU -> Dropout(0.3)   ← deeper head
         -> Linear(32, 5)                              ← raw logits
         -> Softplus()                                 ← non-negative, gradient everywhere
         -> clamp(0.0, 5.0)                            ← safe [0,5] bound

Changes from v5 (v6 anti-collapse)
------------------------------------
  - REMOVED: Sigmoid() * 5.0 output activation (vanishing gradients at 0 & 5).
  - REMOVED: Hard-coded bias init of -1.61 (forced model into mean-prediction trap).
  - ADDED  : Deeper prediction head  Linear(64→32)→GELU→Dropout→Linear(32→5).
  - ADDED  : Softplus activation  (smooth, everywhere-differentiable non-negativity).
  - ADDED  : torch.clamp(0, 5) to safely bound outputs without killing gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DroughtLSTM(nn.Module):
    """
    Parameters
    ----------
    input_size  : number of features per time step (F)
    hidden_size : LSTM hidden dimensionality  (default 64)
    num_layers  : number of stacked LSTM layers
    dropout     : dropout probability (applied between LSTM layers and before head)
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

        # --- Deeper prediction head (v6) ---
        # Linear(64→32) → GELU → Dropout(0.3) → Linear(32→horizon)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, horizon),
        )

        # Softplus: smooth non-negative activation with gradient everywhere
        self.softplus = nn.Softplus(beta=1)

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        out : (batch, horizon)  -- values in [0, 5]
        """
        lstm_out, _ = self.lstm(x)              # (B, W, hidden)
        last_hidden = lstm_out[:, -1, :]        # (B, hidden) -- last time step
        dropped = self.dropout(last_hidden)
        raw = self.head(dropped)                # (B, horizon)
        out = self.softplus(raw)                # non-negative, gradient everywhere
        out = torch.clamp(out, min=0.0, max=5.0)  # safe bound
        return out

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lines = [
            "DroughtLSTM Architecture (v6 – Anti-Collapse Refactor)",
            "=" * 55,
            f"  Input size   : {input_size}  (features per week)",
            f"  LSTM layers  : {self.num_layers}",
            f"  Hidden size  : {self.hidden_size}",
            f"  Dropout      : {self.dropout.p}",
            "  Head         : Linear(64→32) → GELU → Dropout(0.3) → Linear(32→5)",
            "  Activation   : Softplus → clamp(0, 5)  [v6: replaces Sigmoid×5]",
            "  Bias init    : default (no forced mean bias)  [v6: -1.61 removed]",
            "-" * 55,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
