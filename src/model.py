"""
DroughtLSTM
===========
Multi-output LSTM for direct 5-step drought score forecasting.

Architecture (v7 – Architecture Refinement & Smooth Loss)
----------------------------------------------------------
  Input  -> LayerNorm(input_size)                          ← concept-drift stabiliser
          -> LSTM (hidden_size=64, num_layers=2, dropout=0.4)
          -> Global Average Pooling across sequence dim     ← replaces last-step extraction
          -> Dropout(0.4)
          -> Linear(64, 32) -> GELU -> Dropout(0.3)
          -> Linear(32, 5)                                  ← raw logits
          -> Softplus()                                     ← non-negative, gradient everywhere
          (NO clamp – unbounded positive output; clipping applied at inference in train.py)

Changes from v6 (v7 architecture refinement)
---------------------------------------------
  - REMOVED: torch.clamp(0, 5) from forward pass (killed gradients at extreme ends).
  - ADDED  : nn.LayerNorm applied to inputs before LSTM to handle concept drift.
  - CHANGED: Sequence pooling from last-hidden-step to Global Average Pooling (GAP)
             across all 13 time steps so the full temporal profile is utilised.
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

        # --- LayerNorm on raw inputs (v7: concept-drift stabiliser) ---
        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # inter-layer dropout; disabled automatically when num_layers==1
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout)

        # --- Prediction head: Linear(64→32) → GELU → Dropout(0.3) → Linear(32→horizon) ---
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
        out : (batch, horizon)  -- unbounded non-negative values
              Caller should np.clip(out, 0, 5) before writing submission.csv.
        """
        # v7: Normalise inputs to mitigate concept drift
        x = self.input_norm(x)                        # (B, W, F)

        lstm_out, _ = self.lstm(x)                    # (B, W, hidden)

        # v7: Global Average Pooling – aggregate the entire temporal profile
        pooled = lstm_out.mean(dim=1)                 # (B, hidden)

        dropped = self.dropout(pooled)
        raw = self.head(dropped)                      # (B, horizon)
        out = self.softplus(raw)                      # non-negative, gradient everywhere
        # NO clamp here – kept unbounded so gradients are never killed.
        # np.clip(predictions, 0.0, 5.0) is applied in train.py before saving.
        return out

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lines = [
            "DroughtLSTM Architecture (v7 – Architecture Refinement & Smooth Loss)",
            "=" * 65,
            f"  Input size   : {input_size}  (features per week)",
            f"  LayerNorm    : LayerNorm({input_size})  [v7: concept-drift stabiliser]",
            f"  LSTM layers  : {self.num_layers}",
            f"  Hidden size  : {self.hidden_size}",
            f"  Dropout      : {self.dropout.p}",
            "  Pooling      : Global Average Pooling (dim=1)  [v7: replaces last-step]",
            "  Head         : Linear(64→32) → GELU → Dropout(0.3) → Linear(32→5)",
            "  Activation   : Softplus  [v7: NO clamp – unbounded positive output]",
            "  Inference    : np.clip(pred, 0, 5) applied in train.py",
            "-" * 65,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
