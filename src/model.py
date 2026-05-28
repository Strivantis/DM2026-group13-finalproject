"""
DroughtSequenceNet
==================
v32 Robust Sequence Network for drought score multi-step forecasting.

Architecture (v32 — Dual-Layer Dilated Conv1d + BiLSTM + Severity-Weighted L1)
-------------------------------------------------------------------------------

  v32 Paradigm Shift: Robust Temporal Regularization & Severity-Aware Training
  -----------------------------------------------------------------------------
  Problem (v31): Single-layer Conv1d captured only 1-week local receptive fields.
    The 60% zero-inflation acted as a massive gravitational sink under standard
    L1 (MAE) loss, causing the model to chronically under-predict severe drought
    categories (2.0 to 5.0) by collapsing toward the conditional median at zero.

  Solution (v32) — DroughtSequenceNet + Severity-Weighted L1 Loss:
    1. Input reconstructed as (Batch, 13, 27) 3D chronological tensor.
    2. Training-time Gaussian noise (σ=0.05) injected at input for scale-shift
       immunity and Train/Test distribution alignment.
    3. Branch A (Dual-Layer Dilated TCN): Two progressive dilated Conv1d layers
       with dilation=1 then dilation=2 expand temporal receptive field to capture
       multi-week drought trends (kernels covering 3-week and 5-week spans).
    4. Branch B (Bidirectional LSTM): Spatial step dropout (Dropout1d p=0.1)
       zeroes out entire feature columns for random time-steps, forcing the
       BiLSTM to rely on holistic 13-week trends rather than single extreme weeks.
    5. Fusion: parallel outputs concatenated → MLP head → (B, 5) raw regression.
    6. Loss: Severity-Weighted L1 = mean(|pred-true| * (1.0 + true * 0.3))
       Weight at y=0: 1.0x baseline. Weight at y=5: 2.5x amplified gradient.

Architecture Detail (v32)
--------------------------
  Input x (B, 13, 27)   -- 3D chronological time-series (13 weeks, 27 features)

  [Training] x += randn_like(x) * 0.05   -- Gaussian noise injection

  == Branch A: Dual-Layer Dilated Temporal Convolution Channel ==
    -> x.transpose(1, 2)                             -- (B, 27, 13)
    -> Conv1d(in=27,  out=64, kernel=3, pad=1, dilation=1)
    -> GELU()  ->  LayerNorm-equivalent normalization
    -> Conv1d(in=64, out=64, kernel=3, pad=2, dilation=2)
    -> GELU()  ->  LayerNorm-equivalent normalization
    -> AdaptiveAvgPool1d(1)                          -- (B, 64, 1)
    -> Flatten                                       -- (B, 64)
    branch_a_out : (B, 64)

  == Branch B: Long-Term Accumulation Channel (Spatial Dropout + BiLSTM) ==
    [Training] x_b -> Dropout1d(p=0.1) -> zero out entire week columns
    -> LSTM(input=27, hidden=64, layers=3, bidirectional=True, batch_first=True)
    -> extract final time-step hidden (forward + backward concatenated)
    branch_b_out : (B, 128)             -- 64 forward + 64 backward

  == Fusion Head & Projection ==
    -> torch.cat([branch_a_out, branch_b_out], dim=-1)
    fused        : (B, 192)
    -> Linear(192 -> 128) -> GELU() -> Dropout(0.2) -> Linear(128 -> 5)
    output       : (B, 5)            -- no final activation; raw unbounded regression

  Forward Signature:
    forward(self, x) -> output
      output : (B, 5)  raw regression logits (unbounded, no Softplus/Sigmoid)

Changes from v31
----------------
  [v32] Training-time Gaussian noise injection (σ=0.05) at input tensor.
  [v32] Branch A UPGRADED: single Conv1d replaced by dual-layer Dilated TCN
        (dilation=1 then dilation=2) for multi-week receptive field expansion.
  [v32] Branch A normalization: GroupNorm(8,64) replaced by per-layer
        InstanceNorm1d (acts as LayerNorm along channel dim at seq level).
  [v32] Branch B: Spatial step dropout (Dropout1d p=0.1) applied to sequence
        before LSTM to prevent over-reliance on single extreme time-steps.
  [v32] Loss: nn.L1Loss() ABOLISHED; custom SeverityWeightedL1Loss injected.
  [v32] Epochs: 50 -> 100.  Scheduler: T_max=100.
  [v32] Inference: < 0.05 Micro-Zero Floor wipes floating-point noise.
"""

import torch
import torch.nn as nn


class DroughtSequenceNet(nn.Module):
    """
    v32 Robust Sequence Network for multi-step drought score forecasting.

    Two parallel branches process the input (B, 13, 27) 3D time-series tensor:
      - Branch A (Dual-layer Dilated Conv1d):  multi-week local anomaly detector.
      - Branch B (Spatial Dropout + BiLSTM):  long-term cumulative process memory.

    Training augmentations:
      - Gaussian noise injection at input (σ=0.05).
      - Spatial step dropout (Dropout1d p=0.1) before LSTM.

    The fused representation is projected through a 2-layer MLP to produce
    raw regression outputs of shape (B, 5) — no activation at the final layer.

    Parameters
    ----------
    seq_len      : int   — sequence length (number of weeks, default 13)
    n_features   : int   — number of input features per time step (default 27)
    conv_out     : int   — Conv1d output channels for Branch A (default 64)
    lstm_hidden  : int   — LSTM hidden size per direction for Branch B (default 64)
    lstm_layers  : int   — number of stacked LSTM layers (default 3)
    mlp_hidden   : int   — hidden size in fusion MLP (default 128)
    horizon      : int   — number of forecast weeks / output neurons (default 5)
    dropout      : float — dropout probability in fusion MLP (default 0.2)
    """

    def __init__(
        self,
        seq_len:     int   = 13,
        n_features:  int   = 27,
        conv_out:    int   = 64,
        lstm_hidden: int   = 64,
        lstm_layers: int   = 3,
        mlp_hidden:  int   = 128,
        horizon:     int   = 5,
        dropout:     float = 0.2,
    ):
        super().__init__()

        self.seq_len     = seq_len
        self.n_features  = n_features
        self.conv_out    = conv_out
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.mlp_hidden  = mlp_hidden
        self.horizon     = horizon
        self.dropout_p   = dropout

        # ----------------------------------------------------------------
        # Branch A: Dual-Layer Dilated Temporal Convolution Channel
        # Input: (B, n_features, seq_len)  after transpose from (B, seq_len, n_features)
        #
        # Layer 1: Conv1d(in=27, out=64, kernel=3, padding=1, dilation=1)
        #   -> GELU()
        #   -> InstanceNorm1d(64)  [norm over (C, L) — stable with variable batch]
        #
        # Layer 2: Conv1d(in=64, out=64, kernel=3, padding=2, dilation=2)
        #   -> GELU()
        #   -> InstanceNorm1d(64)
        #
        # -> AdaptiveAvgPool1d(1)   [global avg pool over time dim]
        # -> Flatten -> (B, 64)
        # ----------------------------------------------------------------
        self.branch_a_conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels  = n_features,
                out_channels = conv_out,
                kernel_size  = 3,
                padding      = 1,
                dilation     = 1,
            ),                                           # (B, 64, seq_len)
            nn.GELU(),
            nn.InstanceNorm1d(conv_out, affine=True),   # layer-norm equivalent
        )

        self.branch_a_conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels  = conv_out,
                out_channels = conv_out,
                kernel_size  = 3,
                padding      = 2,
                dilation     = 2,
            ),                                           # (B, 64, seq_len)
            nn.GELU(),
            nn.InstanceNorm1d(conv_out, affine=True),
        )

        self.branch_a_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                     # (B, 64, 1)
            nn.Flatten(),                                # (B, 64)
        )

        # ----------------------------------------------------------------
        # Branch B: Long-Term Accumulation Channel (Spatial Dropout + BiLSTM)
        # Input: (B, seq_len, n_features) — batch_first=True
        #
        # Dropout1d(p=0.1): applied to (B, n_features, seq_len) transposed view,
        #   zeros out entire time-step columns at training time, forcing the
        #   BiLSTM to rely on holistic multi-week trends.
        #
        # LSTM(input=27, hidden=64, layers=3, bidirectional=True)
        # Extract final time-step hidden encoding:
        #   output[:, -1, :]  -> (B, 128)   [64 forward + 64 backward]
        # ----------------------------------------------------------------
        self.branch_b_spatial_dropout = nn.Dropout1d(p=0.1)

        self.branch_b_lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            bidirectional = True,
            batch_first   = True,
            dropout       = dropout if lstm_layers > 1 else 0.0,
        )
        # BiLSTM output size: lstm_hidden * 2 (forward + backward)
        lstm_out_size = lstm_hidden * 2    # 128

        # ----------------------------------------------------------------
        # Fusion Head & Projection MLP
        # fused = cat([branch_a_out, branch_b_out], dim=-1) -> (B, 64+128=192)
        #
        # Linear(192 -> 128) -> GELU() -> Dropout(0.2) -> Linear(128 -> 5)
        # NO final activation — raw unbounded regression output.
        # ----------------------------------------------------------------
        fused_size = conv_out + lstm_out_size    # 64 + 128 = 192

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_size, mlp_hidden),   # 192 -> 128
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, horizon),      # 128 -> 5
            # NO Softplus / Sigmoid / ReLU — raw unbounded output
        )

        # Shape debug flag (prints once on first forward pass)
        self._printed_shape = False

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, seq_len, n_features)  -- 3D chronological time-series tensor
            e.g. (B, 13, 27) for the v32 27-feature / 13-week window

        Returns
        -------
        output : (B, horizon)
            Raw unbounded regression predictions — no Softplus or clipping.
            Physical clip [0, 5] is applied in the inference block of train.py.
        """
        # ----------------------------------------------------------------
        # Training-time Gaussian Noise Injection
        # Builds immunity against scale shifts and Train/Test distribution gaps.
        # σ=0.05 ensures perturbation is sub-threshold for the [0,5] drought scale.
        # ----------------------------------------------------------------
        if self.training:
            x = x + torch.randn_like(x) * 0.05

        # ----------------------------------------------------------------
        # Branch A: Dual-Layer Dilated Temporal Convolution
        # x (B, 13, 27) -> transpose -> (B, 27, 13)
        # Layer 1 (dilation=1): local 3-week receptive field -> (B, 64, 13)
        # Layer 2 (dilation=2): dilated 5-week receptive field -> (B, 64, 13)
        # Pool + Flatten -> (B, 64)
        # ----------------------------------------------------------------
        x_conv   = x.transpose(1, 2)                  # (B, 27, 13)
        a1       = self.branch_a_conv1(x_conv)         # (B, 64, 13)  dilation=1
        a2       = self.branch_a_conv2(a1)             # (B, 64, 13)  dilation=2
        branch_a = self.branch_a_pool(a2)              # (B, 64)

        # ----------------------------------------------------------------
        # Branch B: Spatial Step Dropout + BiLSTM long-range temporal memory
        # Dropout1d zeroes entire feature columns (time-steps) during training.
        # x (B, 13, 27) -> transpose (B, 27, 13) -> Dropout1d -> (B, 27, 13)
        #               -> transpose back (B, 13, 27) -> LSTM -> output (B, 13, 128)
        # Extract final time-step: output[:, -1, :] -> (B, 128)
        # ----------------------------------------------------------------
        # Transpose to (B, 27, 13) so Dropout1d zeroes entire time-step columns
        x_b = x.transpose(1, 2)                        # (B, 27, 13)
        x_b = self.branch_b_spatial_dropout(x_b)       # (B, 27, 13) — cols zeroed
        x_b = x_b.transpose(1, 2)                      # (B, 13, 27)

        lstm_out, _ = self.branch_b_lstm(x_b)          # (B, 13, 128)
        branch_b    = lstm_out[:, -1, :]               # (B, 128) — 13th week hidden state

        # ----------------------------------------------------------------
        # Fusion: horizontal concatenation of both branch outputs
        # (B, 64) + (B, 128) -> (B, 192)
        # ----------------------------------------------------------------
        fused  = torch.cat([branch_a, branch_b], dim=-1)   # (B, 192)
        output = self.fusion_head(fused)                   # (B, 5)

        # Shape debug — print once on first forward call
        if not self._printed_shape:
            print(
                f"  [v32 Shape Debug]"
                f"  x: {tuple(x.shape)}"
                f"  | branch_a (dilated TCN): {tuple(branch_a.shape)}"
                f"  | branch_b (BiLSTM final): {tuple(branch_b.shape)}"
                f"  | fused: {tuple(fused.shape)}"
                f"  | output: {tuple(output.shape)}"
            )
            self._printed_shape = True

        return output    # (B, 5)

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> str:
        lstm_out = self.lstm_hidden * 2
        fused    = self.conv_out + lstm_out
        lines = [
            "DroughtSequenceNet Architecture (v32 — Dual Dilated Conv1d + BiLSTM, Severity-Weighted L1)",
            "=" * 90,
            f"  Input shape  : (B, {self.seq_len}, {self.n_features})"
            f"   [B x 13 weeks x 27 features]",
            "  [Training]   : Gaussian noise injection (σ=0.05) → scale-shift immunity",
            "",
            "  == Branch A: Dual-Layer Dilated TCN (Multi-Week Anomaly Detector) ==",
            f"  x.transpose(1,2)  -> (B, {self.n_features}, {self.seq_len})",
            f"  Conv1d(in={self.n_features}, out={self.conv_out}, kernel=3, pad=1, dilation=1)"
            f"  -> (B, {self.conv_out}, {self.seq_len})",
            f"  GELU()  ->  InstanceNorm1d({self.conv_out})",
            f"  Conv1d(in={self.conv_out}, out={self.conv_out}, kernel=3, pad=2, dilation=2)"
            f"  -> (B, {self.conv_out}, {self.seq_len})",
            f"  GELU()  ->  InstanceNorm1d({self.conv_out})",
            f"  AdaptiveAvgPool1d(1)  ->  Flatten",
            f"  branch_a_out : (B, {self.conv_out})",
            "",
            "  == Branch B: Long-Term Accumulation Channel (Spatial Dropout + BiLSTM) ==",
            f"  [Training]   : Dropout1d(p=0.1) zeroes entire time-step columns",
            f"  LSTM(input={self.n_features}, hidden={self.lstm_hidden},"
            f" layers={self.lstm_layers}, bidirectional=True, batch_first=True)",
            f"  output[:, -1, :]  -- extract 13th-week final hidden state",
            f"  branch_b_out : (B, {lstm_out})"
            f"   [forward {self.lstm_hidden} + backward {self.lstm_hidden}]",
            "",
            "  == Fusion Head & Projection MLP ==",
            f"  cat([branch_a, branch_b], dim=-1)  ->  fused: (B, {fused})",
            f"  Linear({fused} -> {self.mlp_hidden}) -> GELU() -> Dropout({self.dropout_p})"
            f" -> Linear({self.mlp_hidden} -> {self.horizon})",
            f"  output : (B, {self.horizon})   [raw unbounded regression; NO final activation]",
            "",
            "  == Training Objective ==",
            "  Loss   : SeverityWeightedL1  mean(|pred-true| * (1.0 + true*0.3))",
            "           y=0 weight: 1.0x baseline  |  y=5 weight: 2.5x amplified",
            "  Optimizer : AdamW(lr=1e-3, weight_decay=1e-3)",
            "  Scheduler : CosineAnnealingLR(T_max=100, eta_min=1e-6)",
            "  Epochs : 100 (hard limit)",
            "  Batch  : 1024",
            "",
            "  == Inference ==",
            "  Median blending: np.median(all_fold_predictions, axis=0)",
            "  Micro-Zero Floor: np.where(preds < 0.05, 0.0, preds) [noise wiper]",
            "  Physical clip  : np.clip(predictions, 0.0, 5.0)",
            "-" * 90,
            "  [v32] Gaussian noise injection (σ=0.05) ADDED at input forward pass.",
            "  [v32] Branch A UPGRADED: single Conv1d -> dual-layer Dilated TCN.",
            "  [v32] Branch A normalization: GroupNorm -> InstanceNorm1d per layer.",
            "  [v32] Branch B UPGRADED: Dropout1d(p=0.1) spatial step dropout added.",
            "  [v32] Loss UPGRADED: nn.L1Loss() -> SeverityWeightedL1 [1.0 + y*0.3].",
            "  [v32] Epochs EXTENDED: 50 -> 100.  Scheduler T_max: 50 -> 100.",
            "  [v32] Inference: < 0.05 Micro-Zero Floor noise wiper ADDED.",
            "-" * 90,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy alias: FlatDroughtMLP kept for any stale imports
# ---------------------------------------------------------------------------
FlatDroughtMLP = DroughtSequenceNet    # v32: alias so old imports don't crash
DroughtLSTM    = DroughtSequenceNet    # v25 backward compat
