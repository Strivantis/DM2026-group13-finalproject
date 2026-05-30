"""
DroughtSequenceNet
==================
v33 Geo-Memory Hybrid Sequence Network for drought score multi-step forecasting.

Architecture (v33 — Dual-Layer Dilated Conv1d + BiLSTM + Region Embedding + Pure L1)
--------------------------------------------------------------------------------------

  v33 Paradigm Shift: Spatio-Temporal Geo-Memory & Pure L1 Convergence
  ---------------------------------------------------------------------
  Problem (v32): Severity-weighted loss introduced a calibration tug-of-war.
    The dataset is classic Panel Data covering the exact same 2,248 geographical
    regions across both Train and Test sets. The v32 model blindly guesses
    macro-climate physics without any notion of geographical identity.

  Solution (v33) — DroughtSequenceNet + Region Embedding Channel + Pure L1:
    1. Input: (B, 13, 27) 3D chronological tensor (climate) + (B,) region token.
    2. Branch A (Dual-Layer Dilated TCN): Maintained from v32 with Gaussian noise
       injection REMOVED. Stable, unperturbed representation convergence.
       Yields local anomaly context vector of shape (B, 64).
    3. Branch B (BiLSTM): Maintained 3-layer bidirectional LSTM extraction.
       Spatial step dropout (Dropout1d p=0.1) retained for regularisation.
       Yields cumulative climate vector of shape (B, 128).
    4. Branch C (Region Embedding): NEW spatial memory channel.
       nn.Embedding(num_embeddings=2248, embedding_dim=16) — each of the 2,248
       geographical regions gets a unique 16-dimensional identity vector.
       Eliminates geographic distribution shift between Train and Test.
       Yields spatial contextual baseline of shape (B, 16).
    5. Fusion: horizontal concatenation of all three tracks.
       64 (TCN) + 128 (BiLSTM) + 16 (Embedding) = 208 dimensions.
    6. Projection: Linear(208->128) -> GELU() -> Dropout(0.2) -> Linear(128->5).
    7. Loss: Pure nn.L1Loss() — unweighted conditional median convergence.
       The spatial embeddings handle geographic extremes naturally.

Architecture Detail (v33)
--------------------------
  Inputs:
    x_climate (B, 13, 27)   -- 3D chronological time-series (13 weeks, 27 features)
    x_region  (B,)           -- zero-indexed region categorical token [0, 2247]

  [v33] Gaussian noise injection REMOVED (was σ=0.05 in v32)

  == Branch A: Dual-Layer Dilated Temporal Convolution Channel ==
    -> x_climate.transpose(1, 2)                     -- (B, 27, 13)
    -> Conv1d(in=27,  out=64, kernel=3, pad=1, dilation=1)
    -> GELU()  ->  InstanceNorm1d(64)
    -> Conv1d(in=64, out=64, kernel=3, pad=2, dilation=2)
    -> GELU()  ->  InstanceNorm1d(64)
    -> AdaptiveAvgPool1d(1)                          -- (B, 64, 1)
    -> Flatten                                       -- (B, 64)
    branch_a_out : (B, 64)

  == Branch B: Long-Term Accumulation Channel (Spatial Dropout + BiLSTM) ==
    [Training] x_b -> Dropout1d(p=0.1) -> zero out entire week columns
    -> LSTM(input=27, hidden=64, layers=3, bidirectional=True, batch_first=True)
    -> extract final time-step hidden (forward + backward concatenated)
    branch_b_out : (B, 128)             -- 64 forward + 64 backward

  == Branch C: Geo-Memory Region Embedding Channel ==  [NEW v33]
    -> nn.Embedding(2248, 16)(x_region)              -- (B, 16)
    region_feat  : (B, 16)

  == Fusion Head & Projection ==
    -> torch.cat([branch_a_out, branch_b_out, region_feat], dim=-1)
    fused        : (B, 208)             -- 64 + 128 + 16
    -> Linear(208 -> 128) -> GELU() -> Dropout(0.2) -> Linear(128 -> 5)
    output       : (B, 5)              -- no final activation; raw unbounded regression

  Forward Signature:
    forward(self, x_climate, x_region) -> output
      output : (B, 5)  raw regression logits (unbounded, no Softplus/Sigmoid)

Changes from v32
----------------
  [v33] INTRODUCE: Branch C — nn.Embedding(2248, 16) Region Embedding Channel.
  [v33] INTRODUCE: forward() signature now accepts x_climate AND x_region.
  [v33] INTRODUCE: Fusion input widened: 192 -> 208 (64+128+16).
        Linear(192->128) REPLACED BY Linear(208->128).
  [v33] REMOVE: Training-time Gaussian noise injection (σ=0.05) from v32.
  [v33] RETAIN: Dual-Layer Dilated TCN (Branch A), dilation=1 then dilation=2.
  [v33] RETAIN: 3-layer BiLSTM + Dropout1d (Branch B).
  [v33] RETAIN: GELU activations, InstanceNorm1d, AdaptiveAvgPool1d, Dropout(0.2).
  [v33] Loss: SeverityWeightedL1 ABOLISHED → pure nn.L1Loss() restored.
"""

import torch
import torch.nn as nn


class DroughtSequenceNet(nn.Module):
    """
    v33 Geo-Memory Hybrid Sequence Network for multi-step drought score forecasting.

    Three parallel branches fuse temporal streams with a spatial prior:
      - Branch A (Dual-layer Dilated Conv1d):  multi-week local anomaly detector.
      - Branch B (Spatial Dropout + BiLSTM):  long-term cumulative process memory.
      - Branch C (Region Embedding):           16-dim geographical identity vector.

    The three streams are concatenated (64 + 128 + 16 = 208 dims) and projected
    through a 2-layer MLP to produce raw regression outputs of shape (B, 5).

    Parameters
    ----------
    seq_len         : int   — sequence length (number of weeks, default 13)
    n_features      : int   — number of input features per time step (default 27)
    conv_out        : int   — Conv1d output channels for Branch A (default 64)
    lstm_hidden     : int   — LSTM hidden size per direction for Branch B (default 64)
    lstm_layers     : int   — number of stacked LSTM layers (default 3)
    mlp_hidden      : int   — hidden size in fusion MLP (default 128)
    horizon         : int   — number of forecast weeks / output neurons (default 5)
    dropout         : float — dropout probability in fusion MLP (default 0.2)
    num_regions     : int   — vocabulary size for Region Embedding (default 2248)
    embed_dim       : int   — embedding dimension for Region Embedding (default 16)
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
        num_regions: int   = 2248,
        embed_dim:   int   = 16,
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
        self.num_regions = num_regions
        self.embed_dim   = embed_dim

        # ----------------------------------------------------------------
        # Branch A: Dual-Layer Dilated Temporal Convolution Channel
        # Input: (B, n_features, seq_len)  after transpose from (B, seq_len, n_features)
        #
        # Layer 1: Conv1d(in=27, out=64, kernel=3, padding=1, dilation=1)
        #   -> GELU()
        #   -> InstanceNorm1d(64)
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
        # Branch C: Geo-Memory Region Embedding Channel  [NEW v33]
        # x_region (B,) -> Embedding(2248, 16) -> region_feat (B, 16)
        #
        # Each of the 2,248 geographical regions receives a unique learnable
        # 16-dimensional identity vector. This spatial memory block eliminates
        # geographic distribution shift between Train and Test partitions.
        # ----------------------------------------------------------------
        self.region_embed = nn.Embedding(
            num_embeddings = num_regions,   # 2248
            embedding_dim  = embed_dim,     # 16
        )

        # ----------------------------------------------------------------
        # Fusion Head & Projection MLP
        # fused = cat([branch_a_out, branch_b_out, region_feat], dim=-1)
        #       -> (B, 64 + 128 + 16) = (B, 208)
        #
        # Linear(208 -> 128) -> GELU() -> Dropout(0.2) -> Linear(128 -> 5)
        # NO final activation — raw unbounded regression output.
        # ----------------------------------------------------------------
        fused_size = conv_out + lstm_out_size + embed_dim   # 64 + 128 + 16 = 208

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_size, mlp_hidden),   # 208 -> 128
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, horizon),      # 128 -> 5
            # NO Softplus / Sigmoid / ReLU — raw unbounded output
        )

        # Shape debug flag (prints once on first forward pass)
        self._printed_shape = False

    # -----------------------------------------------------------------------
    def forward(self, x_climate: torch.Tensor, x_region: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_climate : (B, seq_len, n_features)  -- 3D chronological time-series tensor
                    e.g. (B, 13, 27) for the v33 27-feature / 13-week window
        x_region  : (B,)  -- zero-indexed categorical region token [0, 2247]

        Returns
        -------
        output : (B, horizon)
            Raw unbounded regression predictions — no Softplus or clipping.
            Physical clip [0, 5] is applied in the inference block of train.py.
        """
        # ----------------------------------------------------------------
        # Branch A: Dual-Layer Dilated Temporal Convolution
        # x_climate (B, 13, 27) -> transpose -> (B, 27, 13)
        # Layer 1 (dilation=1): local 3-week receptive field -> (B, 64, 13)
        # Layer 2 (dilation=2): dilated 5-week receptive field -> (B, 64, 13)
        # Pool + Flatten -> (B, 64)
        # NOTE: Gaussian noise injection REMOVED in v33 for stable convergence.
        # ----------------------------------------------------------------
        x_conv   = x_climate.transpose(1, 2)              # (B, 27, 13)
        a1       = self.branch_a_conv1(x_conv)             # (B, 64, 13)  dilation=1
        a2       = self.branch_a_conv2(a1)                 # (B, 64, 13)  dilation=2
        branch_a = self.branch_a_pool(a2)                  # (B, 64)

        # ----------------------------------------------------------------
        # Branch B: Spatial Step Dropout + BiLSTM long-range temporal memory
        # Dropout1d zeroes entire feature columns (time-steps) during training.
        # x_climate (B, 13, 27) -> transpose (B, 27, 13)
        #                       -> Dropout1d -> (B, 27, 13)
        #                       -> transpose back (B, 13, 27)
        #                       -> LSTM -> output (B, 13, 128)
        # Extract final time-step: output[:, -1, :] -> (B, 128)
        # ----------------------------------------------------------------
        x_b = x_climate.transpose(1, 2)                    # (B, 27, 13)
        x_b = self.branch_b_spatial_dropout(x_b)           # (B, 27, 13) — cols zeroed
        x_b = x_b.transpose(1, 2)                          # (B, 13, 27)

        lstm_out, _ = self.branch_b_lstm(x_b)              # (B, 13, 128)
        branch_b    = lstm_out[:, -1, :]                   # (B, 128)

        # ----------------------------------------------------------------
        # Branch C: Geo-Memory Region Embedding  [NEW v33]
        # x_region (B,) -> Embedding(2248, 16) -> (B, 16)
        # Each region gets a unique learnable spatial contextual baseline.
        # ----------------------------------------------------------------
        region_feat = self.region_embed(x_region)           # (B, 16)

        # ----------------------------------------------------------------
        # Fusion: horizontal concatenation of all three branch outputs
        # (B, 64) + (B, 128) + (B, 16) -> (B, 208)
        # ----------------------------------------------------------------
        fused  = torch.cat([branch_a, branch_b, region_feat], dim=-1)   # (B, 208)
        output = self.fusion_head(fused)                                 # (B, 5)

        # Shape debug — print once on first forward call
        if not self._printed_shape:
            print(
                f"  [v33 Shape Debug]"
                f"  x_climate: {tuple(x_climate.shape)}"
                f"  | branch_a (dilated TCN): {tuple(branch_a.shape)}"
                f"  | branch_b (BiLSTM final): {tuple(branch_b.shape)}"
                f"  | region_feat (embedding): {tuple(region_feat.shape)}"
                f"  | fused: {tuple(fused.shape)}"
                f"  | output: {tuple(output.shape)}"
            )
            self._printed_shape = True

        return output    # (B, 5)

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> str:
        lstm_out  = self.lstm_hidden * 2
        fused     = self.conv_out + lstm_out + self.embed_dim
        lines = [
            "DroughtSequenceNet Architecture (v33 — Dual Dilated Conv1d + BiLSTM + Region Embedding, Pure L1)",
            "=" * 95,
            f"  Input A (climate) : (B, {self.seq_len}, {self.n_features})"
            f"   [B x 13 weeks x 27 features]",
            f"  Input B (region)  : (B,)  [zero-indexed region token in [0, {self.num_regions-1}]]",
            "  [v33] Gaussian noise injection REMOVED — stable unperturbed convergence",
            "",
            "  == Branch A: Dual-Layer Dilated TCN (Multi-Week Anomaly Detector) ==",
            f"  x_climate.transpose(1,2)  -> (B, {self.n_features}, {self.seq_len})",
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
            "  == Branch C: Geo-Memory Region Embedding Channel  [NEW v33] ==",
            f"  nn.Embedding(num_embeddings={self.num_regions}, embedding_dim={self.embed_dim})",
            f"  x_region (B,) -> unique 16-dim spatial identity vector",
            f"  region_feat  : (B, {self.embed_dim})",
            "",
            "  == Fusion Head & Projection MLP ==",
            f"  cat([branch_a, branch_b, region_feat], dim=-1)  ->  fused: (B, {fused})",
            f"  [{self.conv_out} (TCN) + {lstm_out} (BiLSTM) + {self.embed_dim} (Embed) = {fused}]",
            f"  Linear({fused} -> {self.mlp_hidden}) -> GELU() -> Dropout({self.dropout_p})"
            f" -> Linear({self.mlp_hidden} -> {self.horizon})",
            f"  output : (B, {self.horizon})   [raw unbounded regression; NO final activation]",
            "",
            "  == Training Objective ==",
            "  Loss      : Pure nn.L1Loss()  (unweighted conditional median convergence)",
            "              Region Embedding handles geographic extremes naturally.",
            "  Optimizer : AdamW(lr=1e-3, weight_decay=1e-3)",
            "  Scheduler : CosineAnnealingLR(T_max=100, eta_min=1e-6)",
            "  Epochs    : 100 (hard limit)",
            "  Batch     : 1024",
            "",
            "  == Inference ==",
            "  Median blending: np.median(all_fold_predictions, axis=0)",
            "  Physical clip  : np.clip(predictions, 0.0, 5.0)",
            "-" * 95,
            "  [v33] INTRODUCE: Branch C — nn.Embedding(2248, 16) Region Embedding.",
            "  [v33] INTRODUCE: forward() now accepts x_climate AND x_region.",
            "  [v33] INTRODUCE: Fusion widened 192 -> 208 (64+128+16).",
            "  [v33] INTRODUCE: Linear(208->128) replaces Linear(192->128).",
            "  [v33] REMOVE:    Gaussian noise injection (σ=0.05) from v32.",
            "  [v33] RESTORE:   Pure nn.L1Loss() (SeverityWeightedL1 abolished).",
            "-" * 95,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy alias: FlatDroughtMLP kept for any stale imports
# ---------------------------------------------------------------------------
FlatDroughtMLP = DroughtSequenceNet    # v33: alias so old imports don't crash
DroughtLSTM    = DroughtSequenceNet    # v25 backward compat
