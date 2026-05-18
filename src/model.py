"""
DroughtLSTM
===========
Multi-output LSTM for direct 5-step drought score forecasting.

Architecture (v10 – Horizon Encoding + Dynamic Loss Weighting + LR Warm-up)
----------------------------------------------------------------------------
  Problem:  v9 training dynamics failed: BCE loss dominated early epochs,
            triggering ReduceLROnPlateau too early and starving the Severity
            head. Additionally, the model was "temporally blind" — it
            predicted 5 values without any knowledge of which future week
            was being predicted, causing monotonically decreasing outputs.

  Solution:
    1. Horizon Encoding: Force the model to compute severity for each
       specific future week by concatenating a normalised horizon index
       [0.2, 0.4, 0.6, 0.8, 1.0] onto the context vector before each
       branch head. This makes the model "horizon-aware".
    2. Dynamic Loss Weighting (Burn-in): Implemented in train.py.
       Epochs 1–20: train regression ONLY (loss_b). Epoch 21+: introduce
       BCE with reduced weight (0.1×). Prevents BCE from dominating early.
    3. Manual LR Warm-up: Implemented in train.py. Epochs 1–5 linearly
       ramp LR from 1e-5 to 1e-3. Prevents ReduceLROnPlateau from firing
       during the initial volatile feature-alignment phase.

Architecture Detail
-------------------
  Input  -> LayerNorm(input_size)                          <- concept-drift stabiliser
          -> LSTM (hidden_size=64, num_layers=2, dropout=0.4)
          -> Temporal Attention (Linear(hidden->1) -> Softmax over time)
             context_vector = sum(attn_weights * lstm_out, dim=1)    (B, hidden)
          -> Dropout(0.4)

          [Horizon Encoding — v10]
          -> Expand context_vector: (B, hidden) -> (B, 5, hidden)
          -> Create horizon_idx:    [0.2,0.4,0.6,0.8,1.0] -> (B, 5, 1)
          -> Concatenate along dim=-1 -> encoded_state: (B, 5, hidden+1)

          -> Branch A (Drought Probability Logits) [NO Sigmoid layer]:
               Linear(hidden+1, 32) -> GELU -> Linear(32, 1)  [per horizon step]
               Squeeze last dim -> (B, 5)  raw logits (unbounded)
               torch.sigmoid() applied inline in forward() only

          -> Branch B (Severity of Drought):
               Linear(hidden+1, 32) -> GELU -> Linear(32, 1) -> Softplus()
               Squeeze last dim -> (B, 5)  non-negative values

  Forward Pass Outputs:
    1. final_output  = sigmoid(logits_output) * severity   (Expected Severity)
    2. logits_output = Branch_A raw logits  (passed to BCEWithLogitsLoss)

  v9 AMP Fix (retained):
    BCELoss + Sigmoid is UNSAFE under torch.autocast (float16 underflow / NaN).
    Solution: Return raw logits from Branch A; use BCEWithLogitsLoss in train.py.
    BCEWithLogitsLoss fuses sigmoid+BCE in a numerically stable kernel safe for AMP.

  Joint Loss (defined in train.py — v10 Dynamic Weighting):
    Epochs  1–20 (Burn-in): Loss = Loss_B only
    Epoch  21+:             Loss = Loss_B + 0.1 * Loss_A
    Loss_A = BCEWithLogitsLoss(logits_output, binary_target)
    Loss_B = Continuous Smooth L1 (final_output, target)

  Early Stopping: monitors pure L1Loss(final_output, target) only -> Kaggle-aligned

Changes from v9
---------------
  - ADDED  : Horizon Encoding before branch heads (v10 temporal awareness).
              context_vector expanded to (B, 5, hidden), horizon_idx (B,5,1)
              concatenated → encoded_state (B, 5, hidden+1).
  - CHANGED: Branch A/B in_features: hidden -> hidden+1.
  - CHANGED: Branch A/B out_features per step: horizon -> 1 (then squeezed to B,5).
  - ADDED  : Shape debug print for first batch (controlled by _printed_shape flag).
  - Dynamic Loss Weighting (Burn-in) implementation in train.py.
  - Manual LR Warm-up implementation in train.py.

Changes from v8 (retained from v9)
-----------------------------------
  - FIXED   : AMP crash: Sigmoid() removed from Branch A head.
               torch.sigmoid() applied dynamically in forward() instead.
               BCELoss -> BCEWithLogitsLoss in train.py (AMP-safe fused kernel).
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

        # --- v10: Horizon Encoding ---
        # After concatenating the horizon index, each branch receives
        # in_features = hidden_size + 1 (the +1 is the normalised horizon idx).
        # Each branch outputs 1 value per horizon step (then squeezed to B×H).
        branch_in = hidden_size + 1

        # --- Branch A: Drought Probability Logits (v9.1: NO Sigmoid layer) ---
        # Outputs raw logits so BCEWithLogitsLoss in train.py can operate
        # in a numerically stable, AMP-safe fused kernel.
        # torch.sigmoid() is applied inline during forward() for final_output.
        self.head_prob = nn.Sequential(
            nn.Linear(branch_in, 32),
            nn.GELU(),
            nn.Linear(32, 1),       # 1 output per horizon step
        )

        # --- Branch B: Severity of Drought (per future step) ---
        # Output non-negative via Softplus; represents magnitude if drought occurs
        self.head_severity = nn.Sequential(
            nn.Linear(branch_in, 32),
            nn.GELU(),
            nn.Linear(32, 1),       # 1 output per horizon step
        )
        self.softplus = nn.Softplus(beta=1)

        # --- v10: Shape debug flag (prints once on first forward pass) ---
        self._printed_shape = False

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        final_output : (batch, horizon)
            Element-wise product of sigmoid(logits_output) x severity_output.
            Represents Expected Severity = P(drought) x E[severity | drought].
            Caller should np.clip(out, 0, 5) before writing submission.csv.

        logits_output : (batch, horizon)
            Raw logits of Branch A (pre-sigmoid).
            Passed to BCEWithLogitsLoss in train.py.
            BCEWithLogitsLoss internally applies sigmoid + BCE in a numerically
            stable fused kernel, making it safe under torch.autocast (AMP).
        """
        B = x.size(0)

        # v7: Normalise inputs to mitigate concept drift
        x = self.input_norm(x)                              # (B, W, F)

        lstm_out, _ = self.lstm(x)                          # (B, W, hidden)

        # v8: Temporal Attention - learn which time steps matter most
        attn_weights = self.attention(lstm_out)              # (B, W, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)   # Softmax over time dim W
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (B, hidden)

        dropped = self.dropout(context_vector)               # (B, hidden)

        # --- v10: Horizon Encoding ---
        # Step 1: Expand context_vector across 5 horizon steps
        #   (B, hidden) -> (B, 5, hidden)
        ctx_expanded = dropped.unsqueeze(1).expand(B, self.horizon, self.hidden_size)

        # Step 2: Create normalised horizon index [0.2, 0.4, 0.6, 0.8, 1.0]
        #   Shape: (5,) -> (1, 5, 1) -> (B, 5, 1)
        horizon_idx = torch.linspace(
            1.0 / self.horizon,
            1.0,
            self.horizon,
            device=x.device,
            dtype=x.dtype,
        )                                                   # (H,)
        horizon_idx = horizon_idx.view(1, self.horizon, 1).expand(B, self.horizon, 1)
        # (B, 5, 1)

        # Step 3: Concatenate along last dim -> (B, 5, hidden+1)
        encoded_state = torch.cat([ctx_expanded, horizon_idx], dim=-1)
        # (B, 5, hidden+1)

        # Branch A: raw logits for drought probability (v9.1: no Sigmoid layer)
        # encoded_state: (B, 5, hidden+1) -> head_prob produces (B, 5, 1) -> squeeze to (B, 5)
        logits_output = self.head_prob(encoded_state).squeeze(-1)    # (B, H) unbounded

        # Branch B: Severity of Drought (non-negative, unbounded above)
        # (B, 5, hidden+1) -> (B, 5, 1) -> squeeze -> (B, 5)
        severity = self.softplus(
            self.head_severity(encoded_state).squeeze(-1)
        )                                                              # (B, H) >= 0

        # Expected Severity = sigmoid(logits) x Severity
        # sigmoid() applied here (not as a layer) so BCEWithLogitsLoss in
        # train.py receives raw logits for numerically stable AMP-safe training.
        final_output = torch.sigmoid(logits_output) * severity        # (B, H)

        # v10: Shape debug – print once on first forward call
        if not self._printed_shape:
            print(f"  [v10 Shape Debug] encoded_state: {tuple(encoded_state.shape)}  "
                  f"|  final_output: {tuple(final_output.shape)}")
            self._printed_shape = True

        return final_output, logits_output

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lines = [
            "DroughtLSTM Architecture (v10 – Horizon Encoding + Burn-in + LR Warm-up)",
            "=" * 75,
            f"  Input size   : {input_size}  (features per week, 37 total)",
            f"  LayerNorm    : LayerNorm({input_size})  [v7: concept-drift stabiliser]",
            f"  LSTM layers  : {self.num_layers}",
            f"  Hidden size  : {self.hidden_size}",
            f"  Dropout      : {self.dropout.p}",
            "  Pooling      : Temporal Attention  [v8: retained]",
            "                 attention = Linear(hidden->1); softmax over time; weighted sum",
            "  [v10] Horizon Encoding:",
            "     context_vector (B, hidden) -> expand (B, 5, hidden)",
            "     horizon_idx [0.2,0.4,0.6,0.8,1.0] -> (B, 5, 1)",
            "     cat(dim=-1) -> encoded_state (B, 5, hidden+1)",
            f"  Branch A     : Linear({self.hidden_size+1}->32) -> GELU -> Linear(32->1) [per step, NO Sigmoid]",
            "                 squeeze(-1) -> (B,5);  sigmoid() applied inline in forward()",
            f"  Branch B     : Linear({self.hidden_size+1}->32) -> GELU -> Linear(32->1) -> Softplus()",
            "                 squeeze(-1) -> (B,5) >= 0  -- Severity of Drought",
            "  Final Output : sigmoid(logits_A) x Branch_B  (Expected Severity)",
            "  Returns      : (final_output, logits_output) -- two tensors",
            "  Inference    : np.clip(final_output, 0, 5) applied in train.py",
            "-" * 75,
            "  [v10] Dynamic Loss Weighting (Burn-in):",
            "     Epochs  1-20 : Loss = Loss_B  ONLY  (regression burn-in)",
            "     Epoch  21+   : Loss = Loss_B + 0.1 * Loss_A  (BCE introduced)",
            "  [v10] Manual LR Warm-up:",
            "     Epochs  1-5  : LR linearly ramps 1e-5 -> 1e-3",
            "     Epoch   6+   : ReduceLROnPlateau takes over",
            "  Loss_A       : BCEWithLogitsLoss(logits_output, binary_target)",
            "                 (AMP-safe: fused sigmoid+BCE avoids float16 underflow)",
            "  Loss_B       : Continuous Smooth L1(final_output, y)   [weight 1.0]",
            "  Early Stop   : pure L1Loss(final_output, y)  [Kaggle MAE aligned]",
            "-" * 75,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
