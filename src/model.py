"""
DroughtSequenceNet
==================
v31 Parallel Hybrid Backbone for drought score multi-step forecasting.

Architecture (v31 — Parallel Conv1d + BiLSTM + Pure L1 Loss)
-------------------------------------------------------------

  v31 Paradigm Shift: Deep Learning Revival via Parallel Hybrid Sequence Net
  --------------------------------------------------------------------------
  Problem (v25–v30): Wide flat MLP on tabular 378-dim vectors and tree-based
    LightGBM ensembles cannot simultaneously capture:
      (a) Short-term local anomalies (e.g. heatwave spikes) — high-frequency
          patterns best detected by 1D Temporal Convolutions over weeks.
      (b) Long-term cumulative processes (e.g. soil evaporation decline) —
          sequential memory best modelled by Recurrent networks.

  Solution (v31) — DroughtSequenceNet + Pure L1 Loss:
    1. Input reconstructed as (Batch, 13, 27) 3D chronological tensor.
    2. Branch A (Temporal Convolution): transposed to (B, 27, 13) for Conv1d.
       Detects local climate anomalies via kernel-3 spatial receptive field.
    3. Branch B (Bidirectional LSTM): processes (B, 13, 27) directly.
       3-layer BiLSTM accumulates long-range temporal context.
    4. Fusion: parallel outputs concatenated → MLP head → (B, 5) raw regression.
    5. Loss: pure nn.L1Loss() (MAE) natively converges to conditional median,
       allowing the network to cleanly collapse to 0.0 for zero-inflated regions.

Architecture Detail (v31)
--------------------------
  Input x (B, 13, 27)   -- 3D chronological time-series (13 weeks, 27 features)

  == Branch A: Temporal Convolution Channel (Short-Term Anomaly Detector) ==
    -> x.transpose(1, 2)                  -- (B, 27, 13)
    -> Conv1d(in=27, out=64, kernel=3, padding=1)
    -> GELU()
    -> GroupNorm(num_groups=8, num_channels=64)
    -> AdaptiveAvgPool1d(1)               -- (B, 64, 1)
    -> flatten                            -- (B, 64)
    branch_a_out : (B, 64)

  == Branch B: Long-Term Accumulation Channel (BiLSTM Sequence Memory) ==
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

Changes from v25
----------------
  [v31] FlatDroughtMLP ABOLISHED: no wide backbone (512->256).
  [v31] Input reshaped from (B, 507) flat to (B, 13, 27) 3D temporal tensor.
  [v31] Conv1d Branch: detects short-term local climate anomaly patterns.
  [v31] BiLSTM Branch: 3-layer bidirectional; extracts cumulative long-range context.
  [v31] L1 Loss (MAE): forces median regression; no Softplus non-negativity guard.
  [v31] No final activation: raw unbounded output (B, 5); clip applied post-inference.
  [v31] Multi-quantile head ABOLISHED: no (B, 5, 3) quantile output.
  [v31] Softplus ABOLISHED: physical clip [0, 5] applied in train.py inference block.
"""

import torch
import torch.nn as nn


class DroughtSequenceNet(nn.Module):
    """
    v31 Parallel Hybrid Backbone for multi-step drought score forecasting.

    Two parallel branches process the input (B, 13, 27) 3D time-series tensor:
      - Branch A (Conv1d):  short-term local climate anomaly detector.
      - Branch B (BiLSTM):  long-term cumulative process memory.

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
        # Branch A: Temporal Convolution Channel
        # Input: (B, n_features, seq_len)  after transpose from (B, seq_len, n_features)
        #
        # Conv1d(in=27, out=64, kernel=3, padding=1)
        #   -> GELU()
        #   -> GroupNorm(8, 64)              [GroupNorm works on (B, C, L) format]
        #   -> AdaptiveAvgPool1d(1)           [global avg pool over time dim]
        #   -> Flatten -> (B, 64)
        # ----------------------------------------------------------------
        self.branch_a = nn.Sequential(
            nn.Conv1d(
                in_channels  = n_features,
                out_channels = conv_out,
                kernel_size  = 3,
                padding      = 1,
            ),                                       # (B, 64, 13)
            nn.GELU(),
            nn.GroupNorm(num_groups=8, num_channels=conv_out),  # (B, 64, 13)
            nn.AdaptiveAvgPool1d(1),                 # (B, 64, 1)
            nn.Flatten(),                            # (B, 64)
        )

        # ----------------------------------------------------------------
        # Branch B: Long-Term Accumulation Channel (BiLSTM)
        # Input: (B, seq_len, n_features) — batch_first=True
        #
        # LSTM(input=27, hidden=64, layers=3, bidirectional=True)
        # Extract final time-step hidden encoding:
        #   output[:, -1, :]  -> (B, 128)   [64 forward + 64 backward]
        # ----------------------------------------------------------------
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
            e.g. (B, 13, 27) for the v31 27-feature / 13-week window

        Returns
        -------
        output : (B, horizon)
            Raw unbounded regression predictions — no Softplus or clipping.
            Physical clip [0, 5] is applied in the inference block of train.py.
        """
        # ----------------------------------------------------------------
        # Branch A: Temporal Convolution
        # x (B, 13, 27) -> transpose -> (B, 27, 13) -> Conv1d -> ... -> (B, 64)
        # ----------------------------------------------------------------
        x_conv   = x.transpose(1, 2)          # (B, 27, 13)
        branch_a = self.branch_a(x_conv)       # (B, 64)

        # ----------------------------------------------------------------
        # Branch B: BiLSTM — long-range temporal memory
        # x (B, 13, 27) -> LSTM -> output (B, 13, 128)
        # Extract final time-step: output[:, -1, :] -> (B, 128)
        # ----------------------------------------------------------------
        lstm_out, _ = self.branch_b_lstm(x)    # (B, 13, 128)
        branch_b    = lstm_out[:, -1, :]       # (B, 128) — 13th week hidden state

        # ----------------------------------------------------------------
        # Fusion: horizontal concatenation of both branch outputs
        # (B, 64) + (B, 128) -> (B, 192)
        # ----------------------------------------------------------------
        fused  = torch.cat([branch_a, branch_b], dim=-1)   # (B, 192)
        output = self.fusion_head(fused)                   # (B, 5)

        # Shape debug — print once on first forward call
        if not self._printed_shape:
            print(
                f"  [v31 Shape Debug]"
                f"  x: {tuple(x.shape)}"
                f"  | branch_a: {tuple(branch_a.shape)}"
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
            "DroughtSequenceNet Architecture (v31 — Parallel Conv1d + BiLSTM, Pure L1 Loss)",
            "=" * 90,
            f"  Input shape  : (B, {self.seq_len}, {self.n_features})"
            f"   [B x 13 weeks x 27 features]",
            "",
            "  == Branch A: Temporal Convolution Channel (Short-Term Anomaly Detector) ==",
            f"  x.transpose(1,2)  -> (B, {self.n_features}, {self.seq_len})",
            f"  Conv1d(in={self.n_features}, out={self.conv_out}, kernel=3, padding=1)"
            f"  -> (B, {self.conv_out}, {self.seq_len})",
            f"  GELU()  ->  GroupNorm(8, {self.conv_out})  ->  AdaptiveAvgPool1d(1)"
            f"  ->  Flatten",
            f"  branch_a_out : (B, {self.conv_out})",
            "",
            "  == Branch B: Long-Term Accumulation Channel (BiLSTM Sequence Memory) ==",
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
            "  Loss   : nn.L1Loss()  (pure MAE — natively targets conditional median)",
            "  Optimizer : AdamW(lr=1e-3, weight_decay=1e-3)",
            "  Scheduler : CosineAnnealingLR(T_max=50, eta_min=1e-6)",
            "  Epochs : 50 (hard limit)",
            "  Batch  : 1024",
            "",
            "  == Inference ==",
            "  Median blending: np.median(all_fold_predictions, axis=0)",
            "  Physical clip  : np.clip(predictions, 0.0, 5.0)",
            "  NO manual thresholds. NO Sigmoid gate. NO hurdle multiplication.",
            "-" * 90,
            "  [v31] FlatDroughtMLP ABOLISHED: no wide MLP backbone (512->256).",
            "  [v31] Tabular 378-dim flat input ABOLISHED: 3D (B, 13, 27) tensor.",
            "  [v31] Multi-quantile Pinball Loss ABOLISHED: pure MAE objective.",
            "  [v31] Softplus non-negativity guard ABOLISHED: raw regression + clip.",
            "  [v31] LightGBM Dual-Tree Hurdle ABOLISHED: single end-to-end DL model.",
            "-" * 90,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy alias: FlatDroughtMLP kept for any stale imports
# ---------------------------------------------------------------------------
FlatDroughtMLP = DroughtSequenceNet    # v31: alias so old imports don't crash
DroughtLSTM    = DroughtSequenceNet    # v25 backward compat
