"""
DroughtLSTM
===========
Multi-output LSTM for direct 5-step drought score forecasting.

Architecture (v14 – Gap-Replay + Target-Time Injection + Gap Embedding)
------------------------------------------------------------------------
  Problem (v12.2): The model was predicting distant futures blindly,
    unaware of the target season or the actual calendar gap between the
    last training week and the first prediction week.

  Solution (v14):
    1. Target-Time Injection: Pass the `week_sin` and `week_cos` of the
       5 future target weeks directly to each branch head. The model now
       "knows" WHEN it is predicting, breaking seasonal blindness.
    2. Gap Embedding (Learnable): Project the scalar gap_size (normalised
       actual_gap / 100.0) through a Linear(1, 16) layer into a 16-dim
       continuous embedding. Expanded to (B, 5, 16) and concatenated with
       the context_vector and horizon_embed per horizon step.
    3. New combined input to branch heads:
         [context_vector(512) + horizon_embed(32) + target_time(2) + gap_embed(16)]
         = 562 dim  →  Linear(562, 128)

  Retained from v12:
    4. BiLSTM: bidirectional=True. Hidden=256, Layers=3 → 512 effective.
    5. Learnable Horizon Embedding: nn.Embedding(5, 32).
    6. Temporal Attention: Linear(512→1) softmax over time dim.
    7. Branch A (prob logits) + Branch B (severity), Dropout(0.2) heads.
    8. Dynamic Loss Weighting (Burn-in), Manual LR Warm-up (train.py).

Architecture Detail
-------------------
  Input  -> LayerNorm(input_size)                          <- concept-drift stabiliser
          -> BiLSTM (hidden_size=256, num_layers=3, dropout=0.4)
             Output shape: (B, W, hidden_size*2) = (B, W, 512)
          -> Temporal Attention (Linear(hidden*2 -> 1) -> Softmax over time)
             context_vector = sum(attn_weights * lstm_out, dim=1)  (B, 512)
          -> Dropout(0.4)

          [v11 Learnable Horizon Embedding – retained]
          -> Expand context_vector: (B, 512) -> (B, 5, 512)
          -> horizon_ids = [0,1,2,3,4]  (long tensor)
          -> horizon_embed(horizon_ids): (5, 32) -> expand (B, 5, 32)

          [v14 Target-Time Injection]
          -> target_time input: (B, 5, 2)  week_sin/cos of 5 future weeks

          [v14 Gap Embedding]
          -> gap_size input: (B, 1)
          -> gap_embed = Linear(1, 16) -> (B, 16) -> expand (B, 5, 16)

          -> Concatenate along dim=-1:
             [ctx(512) + h_emb(32) + target_time(2) + gap_emb(16)] = (B, 5, 562)

          -> Branch A (Drought Probability Logits) [NO Sigmoid layer]:
               Linear(562, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)  [per horizon step]
               Squeeze last dim -> (B, 5)  raw logits (unbounded)
               torch.sigmoid() applied inline in forward() only

          -> Branch B (Severity of Drought):
               Linear(562, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1) -> Softplus()
               Squeeze last dim -> (B, 5)  non-negative values

  Forward Pass Signature (v14):
    forward(self, x, target_time, gap_size)

  Forward Pass Outputs:
    1. final_output  = sigmoid(logits_output) * severity   (Expected Severity)
    2. logits_output = Branch_A raw logits  (passed to BCEWithLogitsLoss)

  AMP Fix (retained from v9):
    BCELoss + Sigmoid is UNSAFE under torch.autocast (float16 underflow / NaN).
    Solution: Return raw logits from Branch A; use BCEWithLogitsLoss in train.py.
    BCEWithLogitsLoss fuses sigmoid+BCE in numerically stable kernel safe for AMP.

  Joint Loss (defined in train.py — v10 Dynamic Weighting, retained in v14):
    Epochs  1–20 (Burn-in): Loss = Loss_B only
    Epoch  21+:             Loss = Loss_B + 0.1 * Loss_A
    Loss_A = BCEWithLogitsLoss(logits_output, binary_target)
    Loss_B = Continuous Smooth L1 (final_output, target)

  Early Stopping: monitors pure L1Loss(final_output, target) only -> Kaggle-aligned

Changes from v12
----------------
  - CHANGED : forward() signature now accepts `target_time` (B,5,2) and
              `gap_size` (B,1) in addition to `x` (B,26,37).
  - ADDED   : self.gap_embed = nn.Linear(1, 16)  in __init__.
  - CHANGED : branch_in changed 544 -> 562  (512+32+2+16).
  - UPDATED : architecture_summary() for v14.
  - RETAINED: All v12 training dynamics (burn-in, warm-up, AMP, etc.)
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

        # --- v14: Learnable Gap Embedding ---
        # Project the scalar normalised gap (B,1) into a 16-dim continuous space.
        # Allows the model to learn how different calendar gaps affect predictions.
        self.gap_embed = nn.Linear(1, 16)

        # After concatenating:
        #   context_vector (hidden*2=512)
        #   + horizon embedding (32)
        #   + target_time (2)    [v14: week_sin/cos of target weeks]
        #   + gap_embedded (16)  [v14: learnable gap embedding]
        # branch_in = 512 + 32 + 2 + 16 = 562
        branch_in = lstm_out_size + 32 + 2 + 16

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

        # --- v14: Shape debug flag (prints once on first forward pass) ---
        self._printed_shape = False

    # -----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        target_time: torch.Tensor,
        gap_size: torch.Tensor,
    ):
        """
        Parameters
        ----------
        x           : (B, seq_len, input_size)  -- input feature window
        target_time : (B, 5, 2)                 -- [v14] week_sin/cos of future target weeks
        gap_size    : (B, 1)                    -- [v14] normalised actual_gap (gap / 100.0)

        Returns
        -------
        final_output : (B, horizon)
            Element-wise product of sigmoid(logits_output) x severity_output.
            Represents Expected Severity = P(drought) x E[severity | drought].
            Caller should np.clip(out, 0, 5) before writing submission.csv.

        logits_output : (B, horizon)
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

        # --- v14: Gap Embedding ---
        # Project scalar gap_size (B, 1) -> (B, 16) -> expand to (B, 5, 16)
        gap_embedded = self.gap_embed(gap_size)              # (B, 16)
        gap_expanded = gap_embedded.unsqueeze(1).expand(B, self.horizon, 16)
        # (B, 5, 16)

        # target_time is already (B, 5, 2) – passed directly

        # Step 4: Concatenate all along last dim -> (B, 5, 562)
        #   [ctx(512) + h_emb(32) + target_time(2) + gap_emb(16)] = 562
        encoded_state = torch.cat([ctx_expanded, h_emb, target_time, gap_expanded], dim=-1)
        # (B, 5, 562)

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

        # v14: Shape debug – print once on first forward call
        if not self._printed_shape:
            print(f"  [v14 Shape Debug] lstm_out: {tuple(lstm_out.shape)}  "
                  f"|  encoded_state: {tuple(encoded_state.shape)}  "
                  f"|  final_output: {tuple(final_output.shape)}")
            self._printed_shape = True

        return final_output, logits_output

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lstm_out_size = self.hidden_size * 2
        branch_in     = lstm_out_size + 32 + 2 + 16   # 512+32+2+16=562
        lines = [
            "DroughtLSTM Architecture (v14 – Gap-Replay + Target-Time Injection + Gap Embedding)",
            "=" * 90,
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
            "  [v14] Target-Time Injection:",
            "     target_time input (B, 5, 2)  -- week_sin/cos of 5 future target weeks",
            "  [v14] Gap Embedding:",
            "     gap_size input (B, 1)  -- normalised actual_gap (gap / 100.0)",
            "     gap_embed = Linear(1, 16) -> (B, 16) -> expand (B, 5, 16)",
            f"  Concatenation: [ctx({lstm_out_size}) + h_emb(32) + target_time(2) + gap_emb(16)]",
            f"     -> encoded_state (B, 5, {branch_in})",
            f"  Branch A     : Linear({branch_in}->128) -> GELU -> Dropout(0.2) -> Linear(128->1) [per step, NO Sigmoid]  [v12+v14]",
            "                 squeeze(-1) -> (B,5);  sigmoid() applied inline in forward()",
            f"  Branch B     : Linear({branch_in}->128) -> GELU -> Dropout(0.2) -> Linear(128->1) -> Softplus()  [v12+v14]",
            "                 squeeze(-1) -> (B,5) >= 0  -- Severity of Drought",
            "  Final Output : sigmoid(logits_A) x Branch_B  (Expected Severity)",
            "  Returns      : (final_output, logits_output) -- two tensors",
            "  Inference    : np.clip(final_output, 0, 5) applied in train.py",
            "-" * 90,
            "  [v14] Dynamic Loss Weighting (Burn-in):",
            "     Epochs  1-20 : Loss = Loss_B  ONLY  (regression burn-in)",
            "     Epoch  21+   : Loss = Loss_B + 0.1 * Loss_A  (BCE introduced)",
            "  [v14] Manual LR Warm-up (Peak LR=1e-3):",
            "     Epochs  1-5  : LR linearly ramps 1e-5 -> 1e-3",
            "     Epoch   6+   : ReduceLROnPlateau takes over",
            "  Loss_A       : BCEWithLogitsLoss(logits_output, binary_target)",
            "                 (AMP-safe: fused sigmoid+BCE avoids float16 underflow)",
            "  Loss_B       : Continuous Smooth L1(final_output, y)   [weight 1.0]",
            "  Early Stop   : pure L1Loss(final_output, y)  [Kaggle MAE aligned]",
            "-" * 90,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
