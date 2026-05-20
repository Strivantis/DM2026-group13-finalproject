"""
DroughtLSTM
===========
Multi-output dual-stream model for direct 5-step drought score forecasting.

Architecture (v16 – v15 Dual-Stream with 13-week / 29-feature inputs)
----------------------------------------------------------------------
  v16 changes (dataset/preprocessing overhaul only):
    - Input window  : 26 → 13  (hard cap at test-set horizon)
    - Input features: 37 → 29  (8w and 13w rolling features removed)
    - No changes to the network topology or training dynamics.
    - branch_in remains 690 (depends on hidden/CNN dims, not input_size).

  Architecture (v15 – Parallel Dual-Stream CNN+BiLSTM + Single-Fold Paradigm)
  ----------------------------------------------------------------------------
  Problem (v14): Relying solely on BiLSTM forces the model to focus on
    sequential transitions, missing macro-climate patterns detectable via
    receptive-field convolutions across the horizon.
    Additionally, 3-Fold Walk-Forward CV on localized temporal splits
    imposed Data Starvation for regions with large actual_gaps.

  Solution (v15):
    1. Parallel Dual-Stream Architecture:
       - LSTM Stream: BiLSTM (Hidden=256, Layers=3) + Temporal Attention
         ->  lstm_context  (B, 512)
       - CNN Stream:  Conv1d(F→128, k=3, pad=1) + GELU + AdaptiveAvgPool1d(1)
         ->  cnn_context   (B, 128)
       - Both streams receive the same input x (B, 13, 29).
         CNN operates on the transposed view (B, 29, 13).
    2. Feature Fusion:
       Concatenate along last dim (per horizon step):
         [lstm_context(512) + cnn_context(128) + horizon_embed(32) +
          target_time(2) + gap_embedded(16)] = 690 dim
       encoded_state shape: (B, 5, 690)
    3. Single-Fold Paradigm (train.py):
       100% of regions use only their latest 5 weeks as Val_Y.
       Train set = ALL available historical sliding windows (data-maximizing).

  Retained from v14:
    4. BiLSTM: bidirectional=True. Hidden=256, Layers=3 -> 512 effective.
    5. Learnable Horizon Embedding: nn.Embedding(5, 32).
    6. Temporal Attention: Linear(512->1) softmax over time dim.
    7. Learnable Gap Embedding: Linear(1, 16).
    8. Target-Time Injection: (B, 5, 2) week_sin/cos of future targets.
    9. Branch A (prob logits) + Branch B (severity), Dropout(0.2) heads.
   10. Dynamic Loss Weighting (Burn-in), Manual LR Warm-up (train.py).

Architecture Detail (v16: input_size=29, window=13)
----------------------------------------------------
  Input x (B, 13, 29)

  == LSTM Stream ==
    -> LayerNorm(29)                              (B, 13, 29)
    -> BiLSTM (hidden=256, layers=3, dropout=0.4) (B, 13, 512)
    -> Temporal Attention (Linear(512->1) softmax over time dim=13)
       lstm_context = weighted sum                 (B, 512)
    -> Dropout(0.4)
    -> expand to                                  (B, 5, 512)

  == CNN Stream ==
    -> x.permute(0,2,1)                           (B, 29, 13)
    -> Conv1d(29->128, k=3, pad=1) + GELU          (B, 128, 13)
    -> AdaptiveAvgPool1d(1)                        (B, 128, 1)
    -> squeeze(-1)   cnn_context                  (B, 128)
    -> expand to                                  (B, 5, 128)

  == Horizon / Time / Gap embeddings ==
    -> horizon_ids [0..4] -> Embedding(5,32) -> expand (B, 5, 32)
    -> target_time input                          (B, 5, 2)
    -> gap_size -> Linear(1,16) -> expand          (B, 5, 16)

  == Fusion ==
    -> cat([lstm(512), cnn(128), h_emb(32), tt(2), gap(16)], dim=-1)
       encoded_state                              (B, 5, 690)

  == Branch A (Probability Logits) ==
    Linear(690->128) -> GELU -> Dropout(0.2) -> Linear(128->1)
    squeeze(-1) -> (B,5)  raw logits (sigmoid inline in forward)

  == Branch B (Severity) ==
    Linear(690->128) -> GELU -> Dropout(0.2) -> Linear(128->1) -> Softplus
    squeeze(-1) -> (B,5) >= 0

  Final: sigmoid(logitsA) x severityB  (Expected Severity)

  Forward Signature:
    forward(self, x, target_time, gap_size)

  Forward Outputs:
    1. final_output  = sigmoid(logits_output) * severity  (B, 5)
    2. logits_output = Branch_A raw logits (B, 5) -> BCEWithLogitsLoss

  AMP Fix (retained from v9):
    BCELoss + Sigmoid is UNSAFE under torch.autocast (float16 underflow / NaN).
    Solution: Return raw logits from Branch A; use BCEWithLogitsLoss in train.py.

  Joint Loss (defined in train.py):
    Epochs  1-20 (Burn-in): Loss = Loss_B only
    Epoch  21+:             Loss = Loss_B + 0.1 * Loss_A
    Loss_A = BCEWithLogitsLoss(logits_output, binary_target)
    Loss_B = Continuous Smooth L1 (final_output, target)

  Early Stopping: monitors pure L1Loss(final_output, target) -> Kaggle-aligned

Changes from v15 (this file only – docstring update for v16 input dims)
------------------------------------------------------------------------
  - WINDOW_SIZE 26 -> 13: input shape (B, 26, F) -> (B, 13, F)
  - input_size  37 -> 29: 8w and 13w rolling features removed
  - branch_in unchanged (690): depends on hidden dims, not input_size
  - All functional code unchanged (fully parameter-driven).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DroughtLSTM(nn.Module):
    """
    Parameters
    ----------
    input_size  : number of features per time step (F = 29 in v16)
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
        self.num_layers  = num_layers
        self.horizon     = horizon
        self.input_size  = input_size

        # Effective output dimension from BiLSTM (forward + backward)
        lstm_out_size = hidden_size * 2  # 512

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
        # CNN Stream  (v15)
        # Input to Conv1d: (B, in_channels=input_size, seq_len=window)
        # Output:          (B, 128, window)
        # ----------------------------------------------------------------
        self.cnn_stream = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Global average pool: (B, 128, window) -> (B, 128, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ----------------------------------------------------------------
        # Horizon / Time / Gap Embeddings  (retained from v14)
        # ----------------------------------------------------------------

        # Learnable Horizon Embedding
        self.horizon_embed = nn.Embedding(num_embeddings=5, embedding_dim=32)

        # Learnable Gap Embedding
        self.gap_embed = nn.Linear(1, 16)

        # ----------------------------------------------------------------
        # Branch Heads
        # ----------------------------------------------------------------
        # After concatenating:
        #   lstm_context (512) + cnn_context (128)
        #   + horizon_embed (32)
        #   + target_time (2)    [week_sin/cos of target weeks]
        #   + gap_embedded (16)  [learnable gap embedding]
        # branch_in = 512 + 128 + 32 + 2 + 16 = 690
        # NOTE: branch_in does NOT depend on input_size or window length.
        branch_in = lstm_out_size + 128 + 32 + 2 + 16  # = 690

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
            Passed to BCEWithLogitsLoss in train.py.
        """
        B = x.size(0)

        # ----------------------------------------------------------------
        # LSTM Stream
        # ----------------------------------------------------------------
        x_norm = self.input_norm(x)                           # (B, W, F)
        lstm_out, _ = self.lstm(x_norm)                       # (B, W, 512)

        # Temporal Attention
        attn_weights = self.attention(lstm_out)               # (B, W, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)     # softmax over time
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)   # (B, 512)

        dropped = self.dropout(context_vector)                # (B, 512)

        # Expand across horizon steps
        lstm_out_size = self.hidden_size * 2
        lstm_expanded = dropped.unsqueeze(1).expand(B, self.horizon, lstm_out_size)
        # (B, 5, 512)

        # ----------------------------------------------------------------
        # CNN Stream  (v15)
        # ----------------------------------------------------------------
        x_cnn = x.permute(0, 2, 1)                           # (B, F, W)
        cnn_feat = self.cnn_stream(x_cnn)                    # (B, 128, W)
        cnn_pooled = self.global_pool(cnn_feat)              # (B, 128, 1)
        cnn_context = cnn_pooled.squeeze(-1)                 # (B, 128)

        cnn_expanded = cnn_context.unsqueeze(1).expand(B, self.horizon, 128)
        # (B, 5, 128)

        # ----------------------------------------------------------------
        # Horizon / Gap Embeddings
        # ----------------------------------------------------------------
        horizon_ids = torch.arange(self.horizon, device=x.device, dtype=torch.long)
        h_emb = self.horizon_embed(horizon_ids)               # (5, 32)
        h_emb = h_emb.unsqueeze(0).expand(B, self.horizon, 32)   # (B, 5, 32)

        gap_embedded = self.gap_embed(gap_size)               # (B, 16)
        gap_expanded = gap_embedded.unsqueeze(1).expand(B, self.horizon, 16)
        # (B, 5, 16)

        # target_time is already (B, 5, 2)

        # ----------------------------------------------------------------
        # Feature Fusion  (v15)
        # ----------------------------------------------------------------
        # [lstm(512) + cnn(128) + h_emb(32) + target_time(2) + gap(16)] = 690
        encoded_state = torch.cat(
            [lstm_expanded, cnn_expanded, h_emb, target_time, gap_expanded],
            dim=-1,
        )
        # (B, 5, 690)

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
                f"| lstm_out: {tuple(lstm_out.shape)}  "
                f"| cnn_context: {tuple(cnn_context.shape)}  "
                f"| encoded_state: {tuple(encoded_state.shape)}  "
                f"| final_output: {tuple(final_output.shape)}"
            )
            self._printed_shape = True

        return final_output, logits_output

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lstm_out_size = self.hidden_size * 2
        branch_in     = lstm_out_size + 128 + 32 + 2 + 16   # 690
        lines = [
            "DroughtLSTM Architecture (v16 – Dual-Stream CNN+BiLSTM, 13w/29f inputs)",
            "=" * 90,
            f"  Input size   : {input_size}  (features per week; v16: 29)",
            f"  Window size  : 13 weeks  (v16: capped at test-set horizon)",
            "",
            "  == LSTM Stream ==",
            f"  LayerNorm    : LayerNorm({input_size})",
            f"  BiLSTM       : hidden={self.hidden_size}/dir -> {lstm_out_size} effective  "
            f"(layers={self.num_layers}, bidirectional=True)",
            f"  Dropout      : {self.dropout.p}",
            f"  Attn         : Linear({lstm_out_size}->1); softmax over time; weighted sum",
            f"  lstm_context : (B, {lstm_out_size}) -> expand (B, 5, {lstm_out_size})",
            "",
            "  == CNN Stream (v15) ==",
            f"  Transpose    : (B, 13, {input_size}) -> (B, {input_size}, 13)",
            f"  Conv1d       : Conv1d({input_size}->128, k=3, pad=1) + GELU -> (B, 128, 13)",
            "  GlobalPool   : AdaptiveAvgPool1d(1) -> (B, 128, 1) -> squeeze -> (B, 128)",
            "  cnn_context  : (B, 128) -> expand (B, 5, 128)",
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
            "  == Feature Fusion (v15) ==",
            f"  Concat: [lstm({lstm_out_size}) + cnn(128) + h_emb(32) + tt(2) + gap(16)]",
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
            "  [v16] Input overhaul:",
            "     WINDOW_SIZE: 26 -> 13  (capped at test-set horizon)",
            "     input_size:  37 -> 29  (8w/13w rolling features removed)",
            "     branch_in unchanged: 690 (depends on hidden dims, not input_size)",
            "  [v15] Single-Fold Paradigm:",
            "     Val_Y = last 5 weeks of each region (100% regions, fixed)",
            "     Train = ALL available historical sliding windows (data-maximizing)",
            "  [v14/v15] Dynamic Loss Weighting (Burn-in):",
            "     Epochs  1-20 : Loss = Loss_B ONLY  (regression burn-in)",
            "     Epoch  21+   : Loss = Loss_B + 0.1 * Loss_A  (BCE introduced)",
            "  [v14/v15] Manual LR Warm-up (Peak LR=1e-3):",
            "     Epochs  1-5  : LR linearly ramps 1e-5 -> 1e-3",
            "     Epoch   6+   : ReduceLROnPlateau takes over",
            "  Loss_A       : BCEWithLogitsLoss(logits_output, binary_target)",
            "  Loss_B       : Continuous Smooth L1  W = 1.0 + (y/5)^2 * 3.0",
            "  Early Stop   : pure L1Loss(final_output, y)  [Kaggle MAE aligned]",
            "  Checkpoint   : single_fold_best.pt",
            "-" * 90,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
