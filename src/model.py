"""
DroughtLSTM
===========
Multi-output dual-stream model for direct 5-step drought score forecasting.

Architecture (v16 – Dilated TCN + Confidence Gap-Gating)
---------------------------------------------------------
  Problem (v15): The single-layer CNN stream only captures local 3-week patterns.
    With WINDOW_SIZE=13 there are macro-climate patterns spanning 4-13 weeks
    that require wider receptive fields.  The original fixed-gap embedding
    cannot reliably signal reliability when the deployment gap varies widely
    (observed range: 1–70+ weeks across the 2248 regions).

  Solution (v16) – three new components:

  1. LSTM hidden downsized (256 → 128 per direction):
     Reduces parameter count, lessens overfitting on the smaller 13-week window.
     Effective BiLSTM output: 128 × 2 = 256.

  2. 1D Dilated TCN Stream (replaces single Conv1d):
     Three dilated 1D convolutions with doubling dilation (d=1, 2, 4).
     Each layer uses kernel_size=3 and padding = dilation*(kernel_size-1)//2
     to keep the sequence length fixed at 13.
     Receptive field of the full stack: 1 + 2*(3-1) + 4*(3-1) + 8*(3-1) = 29 weeks
     (larger than the window, ensuring all 13 steps are contextualised).
     → tcn_context (B, 128) after AdaptiveAvgPool1d(1).

  3. Confidence Gap-Gating:
     gap_size (B,1) → Linear(1,1) → Sigmoid → G ∈ [0,1] scalar.
     Both lstm_context and tcn_context are multiplied by G BEFORE expansion.
     When G is small (large/uncertain gap), both streams are suppressed,
     pushing the model towards a conservative (near-zero) forecast.
     The gap_embed (Linear(1,16)) is kept separately in the fusion layer
     to provide a learnable positional signal about the gap magnitude.

Architecture Detail (v16: input_size=29, window=13)
----------------------------------------------------
  Input x (B, 13, 29)

  == LSTM Stream ==
    -> LayerNorm(29)                               (B, 13, 29)
    -> BiLSTM (hidden=128, layers=3, dropout=0.4)  (B, 13, 256)
    -> Temporal Attention (Linear(256->1) softmax over time dim=13)
       lstm_context = weighted sum                  (B, 256)
    -> Dropout(0.4)
    -> Gating: lstm_context = lstm_context * G      (B, 256)   [v16 NEW]
    -> expand to                                   (B, 5, 256)

  == Dilated TCN Stream (v16) ==
    -> x.permute(0,2,1)                            (B, 29, 13)
    -> Conv1d(29->128, k=3, d=1, pad=1) + GELU     (B, 128, 13)
    -> Conv1d(128->128, k=3, d=2, pad=2) + GELU    (B, 128, 13)
    -> Conv1d(128->128, k=3, d=4, pad=4) + GELU    (B, 128, 13)
    -> AdaptiveAvgPool1d(1)                         (B, 128, 1)
    -> squeeze(-1)   tcn_context                   (B, 128)
    -> Gating: tcn_context = tcn_context * G        (B, 128)   [v16 NEW]
    -> expand to                                   (B, 5, 128)

  == Confidence Gap-Gating (v16 NEW) ==
    -> gap_size (B,1) -> Linear(1,1) -> Sigmoid -> G (B,1)
       G multiplies both lstm_context and tcn_context (before expansion)

  == Horizon / Time / Gap embeddings ==
    -> horizon_ids [0..4] -> Embedding(5,32) -> expand (B, 5, 32)
    -> target_time input                           (B, 5, 2)
    -> gap_size -> Linear(1,16) -> expand           (B, 5, 16)

  == Fusion ==
    -> cat([lstm(256), tcn(128), h_emb(32), tt(2), gap(16)], dim=-1)
       encoded_state                               (B, 5, 434)

  == Branch A (Probability Logits) ==
    Linear(434->128) -> GELU -> Dropout(0.2) -> Linear(128->1)
    squeeze(-1) -> (B,5)  raw logits (sigmoid inline in forward)

  == Branch B (Severity) ==
    Linear(434->128) -> GELU -> Dropout(0.2) -> Linear(128->1) -> Softplus
    squeeze(-1) -> (B,5) >= 0

  Final: sigmoid(logitsA) x severityB  (Expected Severity)

  Forward Signature:
    forward(self, x, target_time, gap_size)

  Forward Outputs:
    1. final_output  = sigmoid(logits_output) * severity  (B, 5)
    2. logits_output = Branch_A raw logits (B, 5) -> FocalLoss in train.py

  AMP Fix (retained from v9):
    BCELoss + Sigmoid is UNSAFE under torch.autocast (float16 underflow / NaN).
    Solution: Return raw logits from Branch A; use FocalLoss in train.py.

  Joint Loss (defined in train.py v16):
    Epochs  1-20 (Burn-in): Loss = Masked Regression (B, active samples only)
    Epoch  21+:             Loss = Loss_B (full) + 0.1 * Loss_A (FocalLoss)
    Loss_A = FocalLoss(γ=2.0)(logits_output, binary_target)
    Loss_B = Continuous Smooth L1  W = 1.0 + (y/5)^2 * 3.0
    Early Stopping: monitors pure L1Loss(final_output, target) -> Kaggle-aligned

Changes from v15
----------------
  [v16] LSTM hidden_size: 256 → 128 (effective output: 512 → 256)
  [v16] CNN stream replaced by Dilated TCN (d=1,2,4; keeps seq_len=13)
  [v16] Confidence Gap-Gating: G = Sigmoid(Linear(gap_size,1,1))
         applied to lstm_context and tcn_context before fusion
  [v16] branch_in: 690 → 434  (256+128+32+2+16)
  [v16] Branch heads updated: Linear(434→128) for both A and B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DroughtLSTM(nn.Module):
    """
    Parameters
    ----------
    input_size  : number of features per time step (F = 29 in v16)
    hidden_size : LSTM hidden dimensionality per direction (default 128)
                  Effective LSTM output width = hidden_size * 2 = 256 (BiLSTM)
    num_layers  : number of stacked LSTM layers (default 3)
    dropout     : dropout probability (applied between LSTM layers and before head)
    horizon     : number of future weeks to forecast simultaneously
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,   # v16: 256 → 128 (effective output unchanged formula)
        num_layers: int = 3,
        dropout: float = 0.4,
        horizon: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.horizon     = horizon
        self.input_size  = input_size

        # Effective output dimension from BiLSTM (forward + backward)
        lstm_out_size = hidden_size * 2  # 256  (v16: 128*2)

        # ----------------------------------------------------------------
        # LSTM Stream
        # ----------------------------------------------------------------

        # LayerNorm on raw inputs (concept-drift stabiliser)
        self.input_norm = nn.LayerNorm(input_size)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        # Temporal Attention
        self.attention = nn.Linear(lstm_out_size, 1)

        self.dropout = nn.Dropout(p=dropout)

        # ----------------------------------------------------------------
        # Dilated TCN Stream  (v16 – replaces single Conv1d CNN)
        #
        # Input: (B, input_size, 13)
        # Each dilated conv keeps seq_len = 13 via symmetric padding:
        #   padding = dilation * (kernel_size - 1) // 2
        # Dilation  1: padding = 1*(3-1)//2 = 1
        # Dilation  2: padding = 2*(3-1)//2 = 2
        # Dilation  4: padding = 4*(3-1)//2 = 4
        # Receptive field = 1 + 2*(3-1) + 4*(3-1) + 8*(3-1) = 29 (spans full window)
        # Output: (B, 128, 13)  -> AdaptiveAvgPool1d(1) -> (B, 128)
        # ----------------------------------------------------------------
        self.tcn_stream = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )
        # Global average pool: (B, 128, 13) -> (B, 128, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ----------------------------------------------------------------
        # Confidence Gap-Gating  (v16 NEW)
        # gap_size (B,1) → Linear(1,1) → Sigmoid → G ∈ [0,1]
        # G multiplies both lstm_context and tcn_context
        # ----------------------------------------------------------------
        self.gap_gate = nn.Linear(1, 1)

        # ----------------------------------------------------------------
        # Horizon / Time / Gap Embeddings  (retained from v14)
        # ----------------------------------------------------------------

        # Learnable Horizon Embedding
        self.horizon_embed = nn.Embedding(num_embeddings=5, embedding_dim=32)

        # Learnable Gap Embedding  (positional signal, separate from gate)
        self.gap_embed = nn.Linear(1, 16)

        # ----------------------------------------------------------------
        # Branch Heads  (v16 updated)
        # ----------------------------------------------------------------
        # After concatenating:
        #   lstm_context (256) + tcn_context (128)
        #   + horizon_embed (32)
        #   + target_time (2)    [week_sin/cos of target weeks]
        #   + gap_embedded (16)  [learnable gap embedding]
        # branch_in = 256 + 128 + 32 + 2 + 16 = 434
        # NOTE: branch_in does NOT depend on input_size or window length.
        branch_in = lstm_out_size + 128 + 32 + 2 + 16  # = 434

        # Branch A: Drought Probability Logits (NO Sigmoid layer)
        self.head_prob = nn.Sequential(
            nn.Linear(branch_in, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Branch B: Severity of Drought (non-negative via Softplus)
        self.head_severity = nn.Sequential(
            nn.Linear(branch_in, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self.softplus = nn.Softplus(beta=1)

        # Shape debug flag (prints once on first forward pass)
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
        x           : (B, W, F)  -- input feature window
                       v16: W=13, F=29
        target_time : (B, 5, 2)  -- week_sin/cos of future target weeks
        gap_size    : (B, 1)     -- normalised actual_gap (gap / 100.0)

        Returns
        -------
        final_output : (B, 5)
            Element-wise product of sigmoid(logits_output) x severity_output.
            Represents Expected Severity = P(drought) x E[severity | drought].
            Caller should np.clip(out, 0, 5) before writing submission.csv.

        logits_output : (B, 5)
            Raw logits of Branch A (pre-sigmoid).
            Passed to FocalLoss (γ=2.0) in train.py.
        """
        B = x.size(0)

        # ----------------------------------------------------------------
        # Confidence Gap Gate  (v16 NEW – computed first, used in both streams)
        # G (B, 1)  ∈ [0, 1]
        # ----------------------------------------------------------------
        G = torch.sigmoid(self.gap_gate(gap_size))    # (B, 1)

        # ----------------------------------------------------------------
        # LSTM Stream
        # ----------------------------------------------------------------
        x_norm = self.input_norm(x)                            # (B, W, F)
        lstm_out, _ = self.lstm(x_norm)                        # (B, W, 256)

        # Temporal Attention
        attn_weights = self.attention(lstm_out)                # (B, W, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)      # softmax over time
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)   # (B, 256)

        dropped = self.dropout(context_vector)                 # (B, 256)

        # Apply Confidence Gap Gate
        lstm_context = dropped * G                             # (B, 256)

        # Expand across horizon steps
        lstm_out_size = self.hidden_size * 2                   # 256
        lstm_expanded = lstm_context.unsqueeze(1).expand(B, self.horizon, lstm_out_size)
        # (B, 5, 256)

        # ----------------------------------------------------------------
        # Dilated TCN Stream  (v16)
        # ----------------------------------------------------------------
        x_tcn = x.permute(0, 2, 1)                            # (B, F, W) = (B, 29, 13)
        tcn_feat = self.tcn_stream(x_tcn)                     # (B, 128, 13)
        tcn_pooled = self.global_pool(tcn_feat)               # (B, 128, 1)
        tcn_raw = tcn_pooled.squeeze(-1)                      # (B, 128)

        # Apply Confidence Gap Gate
        tcn_context = tcn_raw * G                             # (B, 128)

        tcn_expanded = tcn_context.unsqueeze(1).expand(B, self.horizon, 128)
        # (B, 5, 128)

        # ----------------------------------------------------------------
        # Horizon / Gap Embeddings
        # ----------------------------------------------------------------
        horizon_ids = torch.arange(self.horizon, device=x.device, dtype=torch.long)
        h_emb = self.horizon_embed(horizon_ids)               # (5, 32)
        h_emb = h_emb.unsqueeze(0).expand(B, self.horizon, 32)    # (B, 5, 32)

        gap_embedded = self.gap_embed(gap_size)               # (B, 16)
        gap_expanded = gap_embedded.unsqueeze(1).expand(B, self.horizon, 16)
        # (B, 5, 16)

        # target_time is already (B, 5, 2)

        # ----------------------------------------------------------------
        # Feature Fusion  (v16)
        # ----------------------------------------------------------------
        # [lstm(256) + tcn(128) + h_emb(32) + target_time(2) + gap(16)] = 434
        encoded_state = torch.cat(
            [lstm_expanded, tcn_expanded, h_emb, target_time, gap_expanded],
            dim=-1,
        )
        # (B, 5, 434)

        # ----------------------------------------------------------------
        # Branch A: raw logits for drought probability
        # ----------------------------------------------------------------
        logits_output = self.head_prob(encoded_state).squeeze(-1)     # (B, 5)

        # ----------------------------------------------------------------
        # Branch B: Severity of Drought (non-negative)
        # ----------------------------------------------------------------
        severity = self.softplus(
            self.head_severity(encoded_state).squeeze(-1)
        )                                                              # (B, 5)

        # Expected Severity = sigmoid(logitsA) x SeverityB
        final_output = torch.sigmoid(logits_output) * severity        # (B, 5)

        # Shape debug – print once on first forward call
        if not self._printed_shape:
            print(
                f"  [v16 Shape Debug] x: {tuple(x.shape)}  "
                f"| lstm_context (gated): {tuple(lstm_context.shape)}  "
                f"| tcn_context (gated): {tuple(tcn_context.shape)}  "
                f"| G: {tuple(G.shape)}  "
                f"| encoded_state: {tuple(encoded_state.shape)}  "
                f"| final_output: {tuple(final_output.shape)}"
            )
            self._printed_shape = True

        return final_output, logits_output

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lstm_out_size = self.hidden_size * 2    # 256
        branch_in     = lstm_out_size + 128 + 32 + 2 + 16   # 434
        lines = [
            "DroughtLSTM Architecture (v16 – Dilated TCN + Confidence Gap-Gating)",
            "=" * 90,
            f"  Input size   : {input_size}  (features per week; v16: 29)",
            f"  Window size  : 13 weeks  (v16: capped at test-set horizon)",
            "",
            "  == LSTM Stream ==",
            f"  LayerNorm    : LayerNorm({input_size})",
            f"  BiLSTM       : hidden={self.hidden_size}/dir -> {lstm_out_size} effective  "
            f"(layers={self.num_layers}, bidirectional=True)  [v16: hidden 256→128]",
            f"  Dropout      : {self.dropout.p}",
            f"  Attn         : Linear({lstm_out_size}->1); softmax over time; weighted sum",
            f"  lstm_context : (B, {lstm_out_size})  [BEFORE gap gate]",
            f"  Gap Gate G   : Sigmoid(Linear(1,1)(gap_size))  ->  G (B,1)  [v16 NEW]",
            f"  gated_lstm   : lstm_context * G  (B, {lstm_out_size})  "
            f"-> expand (B, 5, {lstm_out_size})",
            "",
            "  == Dilated TCN Stream (v16 NEW – replaces single Conv1d) ==",
            f"  Transpose    : (B, 13, {input_size}) -> (B, {input_size}, 13)",
            f"  TCN Layer 1  : Conv1d({input_size}->128, k=3, d=1, pad=1) + GELU -> (B, 128, 13)",
            "  TCN Layer 2  : Conv1d(128->128, k=3, d=2, pad=2) + GELU -> (B, 128, 13)",
            "  TCN Layer 3  : Conv1d(128->128, k=3, d=4, pad=4) + GELU -> (B, 128, 13)",
            "  Receptive field: 1+2*(3-1)+4*(3-1)+8*(3-1) = 29 weeks",
            "  GlobalPool   : AdaptiveAvgPool1d(1) -> (B, 128, 1) -> squeeze -> (B, 128)",
            "  Gap Gate G   : tcn_context * G  (B, 128) -> expand (B, 5, 128)  [v16 NEW]",
            "",
            "  == Confidence Gap-Gating (v16 NEW) ==",
            "  gap_size (B,1) -> Linear(1,1) -> Sigmoid -> G (B,1)",
            "  G suppresses both streams when deployment gap is large/uncertain",
            "  (gap_embed Linear(1,16) is SEPARATE: provides positional signal in fusion)",
            "",
            "  == Embeddings ==",
            "  Learnable Horizon Embedding:",
            "     horizon_ids [0,1,2,3,4] -> Embedding(5,32) -> (B, 5, 32)",
            "  Target-Time Injection:",
            "     target_time input (B, 5, 2)  -- week_sin/cos of 5 future target weeks",
            "  Gap Embedding:",
            "     gap_size input (B, 1)  -- normalised actual_gap / 100.0",
            "     gap_embed = Linear(1, 16) -> (B, 16) -> expand (B, 5, 16)",
            "",
            "  == Feature Fusion (v16) ==",
            f"  Concat: [lstm({lstm_out_size}) + tcn(128) + h_emb(32) + tt(2) + gap(16)]",
            f"     -> encoded_state (B, 5, {branch_in})",
            "",
            f"  Branch A : Linear({branch_in}->128) -> GELU -> Dropout(0.2) -> Linear(128->1)  [NO Sigmoid]",
            "             squeeze(-1) -> (B,5); sigmoid() applied inline in forward()",
            f"  Branch B : Linear({branch_in}->128) -> GELU -> Dropout(0.2) -> Linear(128->1) -> Softplus",
            "             squeeze(-1) -> (B,5) >= 0  -- Severity of Drought",
            "  Final    : sigmoid(logitsA) x Branch_B  (Expected Severity)",
            "  Returns  : (final_output, logits_output) -- two tensors",
            "  Inference: np.clip(final_output, 0, 5) applied in train.py",
            "-" * 90,
            "  [v16] LSTM hidden_size: 256 → 128  (effective: 512 → 256)",
            "  [v16] CNN stream → Dilated TCN (d=1,2,4); receptive field = 29 weeks",
            "  [v16] Confidence Gap-Gating: G = Sigmoid(Linear(1,1)(gap_size))",
            f"  [v16] branch_in: 690 → {branch_in}  (256+128+32+2+16)",
            "  [v15] Single-Fold Paradigm:",
            "     Val_Y = last 5 weeks of each region (100% regions, fixed)",
            "     Train = ALL available historical sliding windows (data-maximizing)",
            "  [v16] Zero-Inflation Suppression (train.py):",
            "     Epoch 1-20 (Burn-in) : Loss_B = Masked Regression (active samples only)",
            "     Epoch 21+            : Loss_B (full) + 0.1 * FocalLoss(γ=2.0)",
            "  [v14/15] Manual LR Warm-up (Peak LR=1e-3):",
            "     Epochs  1-5  : LR linearly ramps 1e-5 -> 1e-3",
            "     Epoch   6+   : ReduceLROnPlateau takes over",
            f"  [v16] Loss_A : FocalLoss(γ=2.0)  (was BCEWithLogitsLoss in v15)",
            "  [v14-16] Loss_B : Continuous Smooth L1  W = 1.0 + (y/5)^2 * 3.0",
            "  Early Stop   : pure L1Loss(final_output, y)  [Kaggle MAE aligned]",
            "  Checkpoint   : single_fold_best.pt",
            "-" * 90,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
