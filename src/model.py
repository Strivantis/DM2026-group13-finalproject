"""
DroughtLSTM
===========
Multi-output dual-stream model for direct 5-step drought score forecasting.

Architecture (v17 – Dilated TCN + Exponential Decay Gap-Gating + Hurdle Model)
-------------------------------------------------------------------------------
  Problem (v16): The linear Sigmoid gap-gate suffered anti-physics inversion:
    a larger Linear(1,1) weight produced HIGHER G for larger gaps, the exact
    opposite of the required physics (memory should decay as the gap grows).
    Additionally, the masked regression burn-in prevented the regression head
    from learning the baseline zero-drought scale.

  Solution (v17) – three rectifications:

  1. Exponential Decay Gap-Gating (replaces Linear+Sigmoid gate):
     Learnable decay parameter: self.gap_lambda = nn.Parameter(tensor(5.0))
     G = exp(-max(eps, |gap_lambda|) * gap_size)   [strictly monotone decay]
     Both lstm_context and tcn_context are multiplied by G BEFORE expansion.
     Physics constraint guarantees G → 1 when gap→0, G → 0 when gap→∞.
     The gap_embed (Linear(1,16)) is kept separately in the fusion layer
     to provide a learnable positional signal about the gap magnitude.

  2. 1D Dilated TCN Stream (retained from v16):
     Three dilated 1D convolutions with doubling dilation (d=1, 2, 4).
     Each layer uses kernel_size=3 and padding = dilation*(kernel_size-1)//2
     to keep the sequence length fixed at 13.
     Receptive field of the full stack: 1 + 2*(3-1) + 4*(3-1) + 8*(3-1) = 29 weeks
     → tcn_context (B, 128) after AdaptiveAvgPool1d(1).

  3. Decoupled Hurdle Model Outputs (v17 NEW):
     Forward now returns three tensors:
       final_output    = sigmoid(logits) * severity  (B, 5)
       logits_output   = Branch A raw logits          (B, 5)
       severity_output = Branch B softplus output     (B, 5)
     train.py uses severity_output directly for Loss_B (target>0 only),
     enabling mathematically isolated loss flows between the two branches.

Architecture Detail (v17: input_size=29, window=13)
----------------------------------------------------
  Input x (B, 13, 29)

  == LSTM Stream ==
    -> LayerNorm(29)                               (B, 13, 29)
    -> BiLSTM (hidden=128, layers=3, dropout=0.4)  (B, 13, 256)
    -> Temporal Attention (Linear(256->1) softmax over time dim=13)
       lstm_context = weighted sum                  (B, 256)
    -> Dropout(0.4)
    -> Gating: lstm_context = lstm_context * G      (B, 256)
    -> expand to                                   (B, 5, 256)

  == Dilated TCN Stream (v16, retained) ==
    -> x.permute(0,2,1)                            (B, 29, 13)
    -> Conv1d(29->128, k=3, d=1, pad=1) + GELU     (B, 128, 13)
    -> Conv1d(128->128, k=3, d=2, pad=2) + GELU    (B, 128, 13)
    -> Conv1d(128->128, k=3, d=4, pad=4) + GELU    (B, 128, 13)
    -> AdaptiveAvgPool1d(1)                         (B, 128, 1)
    -> squeeze(-1)   tcn_context                   (B, 128)
    -> Gating: tcn_context = tcn_context * G        (B, 128)
    -> expand to                                   (B, 5, 128)

  == Exponential Decay Gap-Gating (v17 NEW) ==
    -> gap_lambda (learnable scalar nn.Parameter, init=5.0)
    -> G = exp(-max(eps, |gap_lambda|) * gap_size)  (B,1) in (0, 1]
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

  Forward Outputs (v17):
    1. final_output     = sigmoid(logits_output) * severity_output  (B, 5)
    2. logits_output    = Branch A raw logits (B, 5)  -> FocalLoss in train.py
    3. severity_output  = Branch B softplus  (B, 5)  -> Hurdle Loss_B in train.py

  AMP Fix (retained from v9):
    BCELoss + Sigmoid is UNSAFE under torch.autocast (float16 underflow / NaN).
    Solution: Return raw logits from Branch A; use FocalLoss in train.py.

  Joint Loss (defined in train.py v17 – Decoupled Hurdle Model):
    Loss_A = FocalLoss(gamma=2.0)(logits_output, binary_target) [ALL samples, w=1.0]
    Loss_B = Weighted Smooth L1(severity_output, target)    [target>0 ONLY, w=1.0]
    Total  = Loss_A + Loss_B
    Early Stopping: monitors pure L1Loss(final_output, target) -> Kaggle-aligned

Changes from v16
----------------
  [v17] Exponential Decay Gap-Gating: G = exp(-max(eps,|gap_lambda|)*gap_size)
         gap_lambda is a learnable scalar nn.Parameter (init=5.0)
         Replaces the Linear+Sigmoid gate (anti-physics inversion fixed)
  [v17] Forward returns (final_output, logits_output, severity_output)
         Decoupled Hurdle Model: severity_output used directly in Loss_B
  [v16] LSTM hidden_size: 256 -> 128 (effective output: 512 -> 256)
  [v16] CNN stream replaced by Dilated TCN (d=1,2,4; keeps seq_len=13)
  [v16] branch_in: 690 -> 434  (256+128+32+2+16)
  [v16] Branch heads updated: Linear(434->128) for both A and B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DroughtLSTM(nn.Module):
    """
    Parameters
    ----------
    input_size  : number of features per time step (F = 29 in v16/v17)
    hidden_size : LSTM hidden dimensionality per direction (default 128)
                  Effective LSTM output width = hidden_size * 2 = 256 (BiLSTM)
    num_layers  : number of stacked LSTM layers (default 3)
    dropout     : dropout probability (applied between LSTM layers and before head)
    horizon     : number of future weeks to forecast simultaneously
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,   # v16: 256 -> 128 (effective output = hidden*2 = 256)
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
        # Exponential Decay Gap-Gating  (v17 – replaces Linear+Sigmoid gate)
        # G = exp(-max(eps, |gap_lambda|) * gap_size)  in (0, 1]
        # Physics constraint: G=1 at gap=0; G->0 as gap->inf (strictly monotone)
        # gap_lambda is a learnable scalar initialized at 5.0
        # ----------------------------------------------------------------
        self.gap_lambda = nn.Parameter(torch.tensor(5.0))

        # ----------------------------------------------------------------
        # Horizon / Time / Gap Embeddings  (retained from v14)
        # ----------------------------------------------------------------

        # Learnable Horizon Embedding
        self.horizon_embed = nn.Embedding(num_embeddings=5, embedding_dim=32)

        # Learnable Gap Embedding  (positional signal, separate from gate)
        self.gap_embed = nn.Linear(1, 16)

        # ----------------------------------------------------------------
        # Branch Heads  (v16 updated, retained in v17)
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
                       v16/v17: W=13, F=29
        target_time : (B, 5, 2)  -- week_sin/cos of future target weeks
        gap_size    : (B, 1)     -- normalised actual_gap (gap / 100.0)

        Returns  (v17 – three tensors)
        -------
        final_output : (B, 5)
            sigmoid(logits_output) * severity_output  (Expected Severity).
            Caller should np.clip(out, 0, 5) before writing submission.csv.

        logits_output : (B, 5)
            Raw logits of Branch A (pre-sigmoid).
            Passed to FocalLoss (gamma=2.0) -- Branch A of Hurdle Loss.

        severity_output : (B, 5)
            Branch B softplus output (>= 0).
            Passed directly to Hurdle Loss_B (computed on target > 0 only).
        """
        B = x.size(0)

        # ----------------------------------------------------------------
        # Exponential Decay Gap Gate  (v17 – physics-correct monotone decay)
        # G = exp(-max(eps, |gap_lambda|) * gap_size)
        # gap_size is already normalised (actual_gap / 100.0)
        # ----------------------------------------------------------------
        eps = 1e-6
        lam = torch.clamp(torch.abs(self.gap_lambda), min=eps)   # scalar > 0
        G = torch.exp(-lam * gap_size)                            # (B, 1)

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

        # Apply Exponential Decay Gap Gate
        lstm_context = dropped * G                             # (B, 256)

        # Expand across horizon steps
        lstm_out_size = self.hidden_size * 2                   # 256
        lstm_expanded = lstm_context.unsqueeze(1).expand(B, self.horizon, lstm_out_size)
        # (B, 5, 256)

        # ----------------------------------------------------------------
        # Dilated TCN Stream  (v16, retained)
        # ----------------------------------------------------------------
        x_tcn = x.permute(0, 2, 1)                            # (B, F, W) = (B, 29, 13)
        tcn_feat = self.tcn_stream(x_tcn)                     # (B, 128, 13)
        tcn_pooled = self.global_pool(tcn_feat)               # (B, 128, 1)
        tcn_raw = tcn_pooled.squeeze(-1)                      # (B, 128)

        # Apply Exponential Decay Gap Gate
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
        # Feature Fusion  (v16, retained)
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
        # Branch B: Severity of Drought (non-negative via Softplus)
        # v17: returned separately for decoupled Hurdle Model Loss_B
        # ----------------------------------------------------------------
        severity_output = self.softplus(
            self.head_severity(encoded_state).squeeze(-1)
        )                                                              # (B, 5)

        # v18 Hard Thresholding Gate:
        # If P(drought) < 0.5, forcefully crush severity to absolute 0.0.
        # This eliminates fractional noise (e.g., 0.3) where the true target
        # is an absolute zero, accurately replicating the 59.64% zero-inflation
        # ceiling. The raw loss branches (logits_output, severity_output) are
        # returned separately and are UNAFFECTED by this gate -- backward pass
        # uses only hurdle_loss(logits_output, severity_output, y) in train.py.
        prob         = torch.sigmoid(logits_output)                    # (B, 5)
        final_output = torch.where(
            prob < 0.5,
            torch.zeros_like(severity_output),
            prob * severity_output,
        )                                                              # (B, 5)

        # Shape debug – print once on first forward call
        if not self._printed_shape:
            print(
                f"  [v18 Shape Debug] x: {tuple(x.shape)}  "
                f"| lstm_context (gated): {tuple(lstm_context.shape)}  "
                f"| tcn_context (gated): {tuple(tcn_context.shape)}  "
                f"| G: {tuple(G.shape)}  "
                f"| gap_lambda: {self.gap_lambda.item():.4f}  "
                f"| encoded_state: {tuple(encoded_state.shape)}  "
                f"| final_output: {tuple(final_output.shape)}"
            )
            self._printed_shape = True

        # v18: return severity_output separately for decoupled Hurdle Model loss
        # final_output now uses Hard Thresholding Gate (prob < 0.5 => 0.0)
        return final_output, logits_output, severity_output

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lstm_out_size = self.hidden_size * 2    # 256
        branch_in     = lstm_out_size + 128 + 32 + 2 + 16   # 434
        lines = [
            "DroughtLSTM Architecture (v17 – Dilated TCN + Exponential Decay Gap-Gating + Hurdle Model)",
            "=" * 95,
            f"  Input size   : {input_size}  (features per week; v16/v17: 29)",
            f"  Window size  : 13 weeks  (v16: capped at test-set horizon)",
            "",
            "  == LSTM Stream ==",
            f"  LayerNorm    : LayerNorm({input_size})",
            f"  BiLSTM       : hidden={self.hidden_size}/dir -> {lstm_out_size} effective  "
            f"(layers={self.num_layers}, bidirectional=True)  [v16: hidden 256->128]",
            f"  Dropout      : {self.dropout.p}",
            f"  Attn         : Linear({lstm_out_size}->1); softmax over time; weighted sum",
            f"  lstm_context : (B, {lstm_out_size})  [BEFORE gap gate]",
            f"  Gap Gate G   : exp(-max(eps,|gap_lambda|)*gap_size)  ->  G (B,1)  [v17]",
            f"  gap_lambda   : nn.Parameter (learnable scalar, init=5.0)  [v17]",
            f"  gated_lstm   : lstm_context * G  (B, {lstm_out_size})  "
            f"-> expand (B, 5, {lstm_out_size})",
            "",
            "  == Dilated TCN Stream (v16 – retained) ==",
            f"  Transpose    : (B, 13, {input_size}) -> (B, {input_size}, 13)",
            f"  TCN Layer 1  : Conv1d({input_size}->128, k=3, d=1, pad=1) + GELU -> (B, 128, 13)",
            "  TCN Layer 2  : Conv1d(128->128, k=3, d=2, pad=2) + GELU -> (B, 128, 13)",
            "  TCN Layer 3  : Conv1d(128->128, k=3, d=4, pad=4) + GELU -> (B, 128, 13)",
            "  Receptive field: 1+2*(3-1)+4*(3-1)+8*(3-1) = 29 weeks",
            "  GlobalPool   : AdaptiveAvgPool1d(1) -> (B, 128, 1) -> squeeze -> (B, 128)",
            "  Gap Gate G   : tcn_context * G  (B, 128) -> expand (B, 5, 128)  [v17]",
            "",
            "  == Exponential Decay Gap-Gating (v17 NEW) ==",
            "  gap_lambda : nn.Parameter (learnable scalar, init=5.0)",
            "  G = exp(-max(eps, |gap_lambda|) * gap_size)  (B,1)  in (0, 1]",
            "  Physics constraint: G->1 at gap=0, G->0 as gap->inf (strictly monotone)",
            "  (gap_embed Linear(1,16) is SEPARATE: positional signal only in fusion)",
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
            "  == Feature Fusion (v16, retained) ==",
            f"  Concat: [lstm({lstm_out_size}) + tcn(128) + h_emb(32) + tt(2) + gap(16)]",
            f"     -> encoded_state (B, 5, {branch_in})",
            "",
            f"  Branch A : Linear({branch_in}->128) -> GELU -> Dropout(0.2) -> Linear(128->1)  [NO Sigmoid]",
            "             squeeze(-1) -> (B,5); sigmoid() applied inline in forward()",
            f"  Branch B : Linear({branch_in}->128) -> GELU -> Dropout(0.2) -> Linear(128->1) -> Softplus",
            "             squeeze(-1) -> (B,5) >= 0  -- Severity of Drought",
            "  Final    : sigmoid(logitsA) x Branch_B  (Expected Severity)",
            "  Returns  : (final_output, logits_output, severity_output)  [v17: 3 tensors]",
            "  Inference: np.clip(final_output, 0, 5) applied in train.py",
            "-" * 95,
            "  [v17] Exponential Decay Gap-Gating: G = exp(-max(eps,|gap_lambda|)*gap_size)",
            "        gap_lambda is learnable nn.Parameter (init=5.0) -- physics-correct",
            "  [v17] Decoupled Hurdle Model: returns severity_output for direct Loss_B",
            "  [v17] 5-Fold Region Group CV in train.py (GroupKFold by region_id)",
            "  [v17] Hurdle Loss: Loss_A (all samples) + Loss_B (target>0 only)",
            "  [v16] LSTM hidden_size: 256 -> 128  (effective: 512 -> 256)",
            "  [v16] CNN stream -> Dilated TCN (d=1,2,4); receptive field = 29 weeks",
            f"  [v16] branch_in: 690 -> {branch_in}  (256+128+32+2+16)",
            "  [v17] Manual LR Warm-up (Peak LR=1e-3):",
            "     Epochs  1-5  : LR linearly ramps 1e-5 -> 1e-3",
            "     Epoch   6+   : ReduceLROnPlateau takes over",
            "  [v17] Loss_A : FocalLoss(gamma=2.0) on ALL samples, weight=1.0",
            "  [v17] Loss_B : Weighted Smooth L1 on target>0 ONLY, weight=1.0",
            "  Early Stop   : pure L1Loss(final_output, y)  [Kaggle MAE aligned]",
            "  Checkpoint   : fold_{k}_best.pt  [v17: one per fold]",
            "-" * 95,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
