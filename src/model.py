"""
DroughtLSTM
===========
Multi-output LSTM for direct 5-step drought score forecasting.

Architecture (v9 – Two-Stage Multi-Task Learning for Zero-Inflation)
--------------------------------------------------------------------
  Problem:  58% of target scores are 0 (zero-inflation).
            A single regression head is pulled heavily toward zero.
  Solution: Decompose prediction into two independent branches:

  Input  -> LayerNorm(input_size)                          ← concept-drift stabiliser
          -> LSTM (hidden_size=64, num_layers=2, dropout=0.4)
          -> Temporal Attention (Linear(hidden→1) → Softmax over time)
             context_vector = sum(attn_weights * lstm_out, dim=1)  ← (B, hidden)
          -> Dropout(0.4)
          -> Branch A (Probability of Drought):
               Linear(64, 32) → GELU → Linear(32, 5) → Sigmoid()
               Output: (B, 5)  values in [0, 1]
               Meaning: Probability that drought is occurring at each future step
          -> Branch B (Severity of Drought):
               Linear(64, 32) → GELU → Linear(32, 5) → Softplus()
               Output: (B, 5)  non-negative values
               Meaning: Severity of drought conditioned on it occurring

  Forward Pass Outputs:
    1. final_output = Branch_A × Branch_B  (element-wise Expected Severity)
    2. prob_output  = Branch_A raw output  (needed for BCE loss)

  Joint Loss (defined in train.py):
    Loss_A = BCELoss(prob_output, binary_target)          0.5× weight
    Loss_B = Continuous Smooth L1 (final_output, target)  1.0× weight
    Total  = Loss_B + 0.5 * Loss_A

  Early Stopping: monitors pure L1Loss(final_output, target) only → Kaggle-aligned

Changes from v8
---------------
  - REMOVED: Single Softplus regression head.
  - ADDED  : Two-branch heads (Branch A: probability, Branch B: severity).
  - ADDED  : Forward method returns (final_output, prob_output) tuple.
  - Temporal Attention mechanism RETAINED from v8.
  - LayerNorm on inputs RETAINED from v7.
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

        # --- Temporal Attention (v8: retained) ---
        # Projects each hidden state to a scalar attention score,
        # then softmax normalises across the time dimension W.
        self.attention = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=dropout)

        # --- Branch A: Probability of Drought (per future step) ---
        # Output in [0, 1] via Sigmoid; used with BCELoss in train.py
        self.head_prob = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, horizon),
        )
        self.sigmoid = nn.Sigmoid()

        # --- Branch B: Severity of Drought (per future step) ---
        # Output non-negative via Softplus; represents magnitude if drought occurs
        self.head_severity = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, horizon),
        )
        self.softplus = nn.Softplus(beta=1)

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        final_output : (batch, horizon)
            Element-wise product of prob_output × severity_output.
            Represents Expected Severity = P(drought) × E[severity | drought].
            Caller should np.clip(out, 0, 5) before writing submission.csv.

        prob_output : (batch, horizon)
            Raw sigmoid output of Branch A (probability of drought).
            Used for BCELoss in train.py.
        """
        # v7: Normalise inputs to mitigate concept drift
        x = self.input_norm(x)                              # (B, W, F)

        lstm_out, _ = self.lstm(x)                          # (B, W, hidden)

        # v8: Temporal Attention – learn which time steps matter most
        attn_weights = self.attention(lstm_out)              # (B, W, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)   # Softmax over time dim W
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (B, hidden)

        dropped = self.dropout(context_vector)               # (B, hidden)

        # Branch A – Probability of Drought
        prob_output = self.sigmoid(self.head_prob(dropped))  # (B, H) in [0, 1]

        # Branch B – Severity of Drought (non-negative, unbounded above)
        severity = self.softplus(self.head_severity(dropped))  # (B, H) ≥ 0

        # Expected Severity = P(drought) × Severity
        final_output = prob_output * severity                # (B, H)

        return final_output, prob_output

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lines = [
            "DroughtLSTM Architecture (v9 – Two-Stage Multi-Task + Zero-Inflation)",
            "=" * 68,
            f"  Input size   : {input_size}  (features per week, 37 total)",
            f"  LayerNorm    : LayerNorm({input_size})  [v7: concept-drift stabiliser]",
            f"  LSTM layers  : {self.num_layers}",
            f"  Hidden size  : {self.hidden_size}",
            f"  Dropout      : {self.dropout.p}",
            "  Pooling      : Temporal Attention  [v8: retained]",
            "                 attention = Linear(hidden→1); softmax over time; weighted sum",
            "  Branch A     : Linear(64→32) → GELU → Linear(32→5) → Sigmoid()",
            "                 Output: (B,5) in [0,1] — Probability of Drought",
            "  Branch B     : Linear(64→32) → GELU → Linear(32→5) → Softplus()",
            "                 Output: (B,5) ≥ 0   — Severity of Drought",
            "  Final Output : Branch_A × Branch_B  (Expected Severity)",
            "  Returns      : (final_output, prob_output) — two tensors",
            "  Inference    : np.clip(final_output, 0, 5) applied in train.py",
            "-" * 68,
            "  Loss_A       : BCELoss(prob_output, binary_target)     [weight 0.5]",
            "  Loss_B       : Continuous Smooth L1(final_output, y)   [weight 1.0]",
            "  Total Loss   : Loss_B + 0.5 * Loss_A",
            "  Early Stop   : pure L1Loss(final_output, y)  [Kaggle MAE aligned]",
            "-" * 68,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
