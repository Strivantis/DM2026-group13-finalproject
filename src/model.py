"""
DroughtLSTM
===========
Multi-output LSTM for direct 5-step drought score forecasting.

Architecture (v11 – BiLSTM + Scale-Up + Learnable Horizon Embedding)
----------------------------------------------------------------------
  Problem (v10): The model lacked capacity (hidden=64, unidirectional) to
    memorise extreme climate features; and the rigid linspace scalar horizon
    encoding [0.2, 0.4, 0.6, 0.8, 1.0] could not learn non-uniform horizon
    dynamics.

  Solution (v11):
    1. BiLSTM Upgrade: bidirectional=True.  Every timestep now receives
       context from BOTH past AND future within the input window, giving the
       attention mechanism richer feature vectors.
       LSTM output width doubles to hidden_size * 2 (D=2).
    2. Massive Capacity Scale-up: hidden_size=256, num_layers=3.
       With BiLSTM this yields an effective width of 512 per timestep.
    3. Learnable Horizon Embedding: Replace the hardcoded linspace scalar
       [0.2,0.4,0.6,0.8,1.0] with nn.Embedding(5, 32).  The embedding for
       each horizon step i ∈ {0,1,2,3,4} is learned end-to-end, allowing the
       model to discover complex, non-linear horizon-specific patterns.
    4. Retained: Dynamic Loss Weighting (Burn-in), Manual LR Warm-up,
       Temporal Attention, LayerNorm, Softplus severity, AMP-safe logits.

Architecture Detail
-------------------
  Input  -> LayerNorm(input_size)                          <- concept-drift stabiliser
          -> BiLSTM (hidden_size=256, num_layers=3, dropout=0.4)
             Output shape: (B, W, hidden_size*2) = (B, W, 512)
          -> Temporal Attention (Linear(hidden*2 -> 1) -> Softmax over time)
             context_vector = sum(attn_weights * lstm_out, dim=1)  (B, 512)
          -> Dropout(0.4)

          [v11 Learnable Horizon Embedding]
          -> Expand context_vector: (B, 512) -> (B, 5, 512)
          -> horizon_ids = [0,1,2,3,4]  (long tensor)
          -> horizon_embed(horizon_ids): (5, 32) -> expand (B, 5, 32)
          -> Concatenate along dim=-1 -> encoded_state: (B, 5, 512+32) = (B, 5, 544)

          -> Branch A (Drought Probability Logits) [NO Sigmoid layer]:
               Linear(544, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)  [per horizon step]
               Squeeze last dim -> (B, 5)  raw logits (unbounded)
               torch.sigmoid() applied inline in forward() only

          -> Branch B (Severity of Drought):
               Linear(544, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1) -> Softplus()
               Squeeze last dim -> (B, 5)  non-negative values

  Forward Pass Outputs:
    1. final_output  = sigmoid(logits_output) * severity   (Expected Severity)
    2. logits_output = Branch_A raw logits  (passed to BCEWithLogitsLoss)

  v9 AMP Fix (retained):
    BCELoss + Sigmoid is UNSAFE under torch.autocast (float16 underflow / NaN).
    Solution: Return raw logits from Branch A; use BCEWithLogitsLoss in train.py.
    BCEWithLogitsLoss fuses sigmoid+BCE in a numerically stable kernel safe for AMP.

  Joint Loss (defined in train.py — v10 Dynamic Weighting, retained in v11):
    Epochs  1–20 (Burn-in): Loss = Loss_B only
    Epoch  21+:             Loss = Loss_B + 0.1 * Loss_A
    Loss_A = BCEWithLogitsLoss(logits_output, binary_target)
    Loss_B = Continuous Smooth L1 (final_output, target)

  Early Stopping: monitors pure L1Loss(final_output, target) only -> Kaggle-aligned

Changes from v10
----------------
  - CHANGED : LSTM bidirectional=True.  Attention and branch heads updated to
              accept hidden_size * 2 (doubled width).
  - SCALED  : hidden_size default 64 -> 256, num_layers default 2 -> 3.
  - REPLACED: Hardcoded linspace horizon scalar ([0.2,...,1.0]) replaced with
              nn.Embedding(num_embeddings=5, embedding_dim=32) -- fully learnable.
  - CHANGED : branch_in = hidden_size * 2 + 32  (was hidden_size + 1).
  - UPDATED : architecture_summary() and v11 Shape Debug print.
  - RETAINED: Dynamic Loss Weighting (Burn-in), Manual LR Warm-up (train.py).

Changes from v9 (retained from v10)
------------------------------------
  - ADDED  : Horizon Encoding before branch heads (v10 temporal awareness).
  - Dynamic Loss Weighting (Burn-in) implementation in train.py.
  - Manual LR Warm-up implementation in train.py.

Changes from v8 (retained from v9/v10)
---------------------------------------
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
    hidden_size : LSTM hidden dimensionality per direction (default 256)
                  Effective LSTM output width = hidden_size * 2 (BiLSTM)
    num_layers  : number of stacked LSTM layers (default 3)
    dropout     : dropout probability (applied between LSTM layers and before head)
    horizon     : number of future weeks to forecast simultaneously
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.4,
        horizon: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        # Effective output dimension from BiLSTM (forward + backward)
        lstm_out_size = hidden_size * 2

        # --- LayerNorm on raw inputs (v7: concept-drift stabiliser) ---
        self.input_norm = nn.LayerNorm(input_size)

        # --- v11: BiLSTM (bidirectional=True) ---
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # inter-layer dropout; disabled automatically when num_layers==1
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,          # v11: BiLSTM
        )

        # --- Temporal Attention (v8: retained; v11: updated to lstm_out_size) ---
        # Projects each hidden state (hidden_size*2) to a scalar attention score,
        # then softmax normalises across the time dimension W.
        self.attention = nn.Linear(lstm_out_size, 1)

        self.dropout = nn.Dropout(p=dropout)

        # --- v11: Learnable Horizon Embedding ---
        # Each horizon step i ∈ {0,1,2,3,4} is mapped to a 32-dim vector
        # that is learned end-to-end, replacing the hardcoded linspace scalar.
        self.horizon_embed = nn.Embedding(num_embeddings=5, embedding_dim=32)

        # After concatenating context_vector (hidden*2) + horizon embedding (32):
        # branch_in = hidden_size * 2 + 32
        branch_in = lstm_out_size + 32

        # --- Branch A: Drought Probability Logits (v9.1: NO Sigmoid layer) ---
        # Outputs raw logits so BCEWithLogitsLoss in train.py can operate
        # in a numerically stable, AMP-safe fused kernel.
        # torch.sigmoid() is applied inline during forward() for final_output.
        # v12: Widened 32->128; Dropout(0.2) inserted after GELU (anti-overfitting).
        self.head_prob = nn.Sequential(
            nn.Linear(branch_in, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),      # 1 output per horizon step
        )

        # --- Branch B: Severity of Drought (per future step) ---
        # Output non-negative via Softplus; represents magnitude if drought occurs
        # v12: Widened 32->128; Dropout(0.2) inserted after GELU (anti-overfitting).
        self.head_severity = nn.Sequential(
            nn.Linear(branch_in, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),      # 1 output per horizon step
        )
        self.softplus = nn.Softplus(beta=1)

        # --- v11: Shape debug flag (prints once on first forward pass) ---
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

        # v11: BiLSTM output shape is (B, W, hidden_size*2)
        lstm_out, _ = self.lstm(x)                          # (B, W, hidden*2)

        # v8: Temporal Attention - learn which time steps matter most
        # v11: attention projects hidden*2 -> 1
        attn_weights = self.attention(lstm_out)              # (B, W, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)   # Softmax over time dim W
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (B, hidden*2)

        dropped = self.dropout(context_vector)               # (B, hidden*2)

        # --- v11: Learnable Horizon Embedding ---
        # Step 1: Expand context_vector across 5 horizon steps
        #   (B, hidden*2) -> (B, 5, hidden*2)
        lstm_out_size = self.hidden_size * 2
        ctx_expanded = dropped.unsqueeze(1).expand(B, self.horizon, lstm_out_size)
        # (B, 5, hidden*2)

        # Step 2: Horizon embedding indices [0, 1, 2, 3, 4]
        #   horizon_ids: (5,) long tensor on same device as x
        horizon_ids = torch.arange(self.horizon, device=x.device, dtype=torch.long)
        # (5,)

        # Step 3: Embed indices -> (5, 32) -> expand to (B, 5, 32)
        h_emb = self.horizon_embed(horizon_ids)              # (5, 32)
        h_emb = h_emb.unsqueeze(0).expand(B, self.horizon, 32)  # (B, 5, 32)

        # Step 4: Concatenate along last dim -> (B, 5, hidden*2 + 32)
        encoded_state = torch.cat([ctx_expanded, h_emb], dim=-1)
        # (B, 5, hidden*2 + 32)

        # Branch A: raw logits for drought probability (v9.1: no Sigmoid layer)
        # encoded_state: (B, 5, branch_in) -> head_prob produces (B, 5, 1) -> squeeze (B, 5)
        logits_output = self.head_prob(encoded_state).squeeze(-1)    # (B, H) unbounded

        # Branch B: Severity of Drought (non-negative, unbounded above)
        # (B, 5, branch_in) -> (B, 5, 1) -> squeeze -> (B, 5)
        severity = self.softplus(
            self.head_severity(encoded_state).squeeze(-1)
        )                                                              # (B, H) >= 0

        # Expected Severity = sigmoid(logits) x Severity
        # sigmoid() applied here (not as a layer) so BCEWithLogitsLoss in
        # train.py receives raw logits for numerically stable AMP-safe training.
        final_output = torch.sigmoid(logits_output) * severity        # (B, H)

        # v12: Shape debug – print once on first forward call
        if not self._printed_shape:
            print(f"  [v12 Shape Debug] lstm_out: {tuple(lstm_out.shape)}  "
                  f"|  encoded_state: {tuple(encoded_state.shape)}  "
                  f"|  final_output: {tuple(final_output.shape)}")
            self._printed_shape = True

        return final_output, logits_output

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lstm_out_size = self.hidden_size * 2
        branch_in     = lstm_out_size + 32
        lines = [
            "DroughtLSTM Architecture (v12 – Bottleneck Relief + Dropout Heads)",
            "=" * 85,
            f"  Input size   : {input_size}  (features per week, 37 total)",
            f"  LayerNorm    : LayerNorm({input_size})  [v7: concept-drift stabiliser]",
            f"  LSTM layers  : {self.num_layers}  (BiLSTM, bidirectional=True)  [v11]",
            f"  Hidden size  : {self.hidden_size} per direction  ->  {lstm_out_size} effective  [v11]",
            f"  Dropout      : {self.dropout.p}",
            "  Pooling      : Temporal Attention  [v8: retained, v11: updated to hidden*2]",
            f"                 attention = Linear({lstm_out_size}->1); softmax over time; weighted sum",
            "  [v11] Learnable Horizon Embedding:",
            f"     context_vector (B, {lstm_out_size}) -> expand (B, 5, {lstm_out_size})",
            "     horizon_ids [0,1,2,3,4] -> Embedding(5,32) -> (B, 5, 32)",
            f"     cat(dim=-1) -> encoded_state (B, 5, {branch_in})",
            f"  Branch A     : Linear({branch_in}->128) -> GELU -> Dropout(0.2) -> Linear(128->1) [per step, NO Sigmoid]  [v12]",
            "                 squeeze(-1) -> (B,5);  sigmoid() applied inline in forward()",
            f"  Branch B     : Linear({branch_in}->128) -> GELU -> Dropout(0.2) -> Linear(128->1) -> Softplus()  [v12]",
            "                 squeeze(-1) -> (B,5) >= 0  -- Severity of Drought",
            "  Final Output : sigmoid(logits_A) x Branch_B  (Expected Severity)",
            "  Returns      : (final_output, logits_output) -- two tensors",
            "  Inference    : np.clip(final_output, 0, 5) applied in train.py",
            "-" * 85,
            "  [v12] Dynamic Loss Weighting (Burn-in):",
            "     Epochs  1-20 : Loss = Loss_B  ONLY  (regression burn-in)",
            "     Epoch  21+   : Loss = Loss_B + 0.1 * Loss_A  (BCE introduced)",
            "  [v12] Manual LR Warm-up (Batch=2048, Peak LR=2e-3):",
            "     Epochs  1-5  : LR linearly ramps 1e-5 -> 2e-3  [v12: scaled 4x]",
            "     Epoch   6+   : ReduceLROnPlateau takes over",
            "  Loss_A       : BCEWithLogitsLoss(logits_output, binary_target)",
            "                 (AMP-safe: fused sigmoid+BCE avoids float16 underflow)",
            "  Loss_B       : Continuous Smooth L1(final_output, y)   [weight 1.0]",
            "  Early Stop   : pure L1Loss(final_output, y)  [Kaggle MAE aligned]",
            "-" * 85,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
