"""
FlatDroughtMLP
==============
v25 Multi-Quantile Flat MLP for drought score multi-step forecasting.

Architecture (v25 – Adversarial Weighting + Multi-Quantile Pinball Loss)
-------------------------------------------------------------------------

  v25 Paradigm Shift: Hurdle Abolished, Pure Quantile Regression
  ---------------------------------------------------------------
  Problem (v24): Dual-Head Hurdle architecture (BCE + SmoothL1) with manual
    post-processing thresholding creates calibration distortions. Adversarial
    Validation revealed AUC ~ 0.94 covariate shift between Train and Test sets.

  Solution (v25) – FlatDroughtMLP + Multi-Quantile Pinball:
    1. Replace BiLSTM + TCN with a wide MLP on flattened 13-week feature vectors.
    2. Abolish dual-head (BCE + Regression). Use a single Quantile Head outputting
       (B, 5, 3) covering q in [0.1, 0.5, 0.9] for 5 forecast weeks.
    3. Softplus on the quantile head ensures all outputs >= 0 (physical constraint).
    4. Training uses Weighted Multi-Quantile Pinball Loss with adversarial sample
       weights from a LGBMClassifier (P_test = probability a row belongs to Test).
    5. Inference: extract strictly the q=0.5 (median) channel; no thresholding.

Architecture Detail (v25: input_dim = WINDOW_SIZE * len(FEATURE_COLS) = 507)
------------------------------------------------------------------------------
  Input x (B, input_dim)   -- flattened 13-week x 39-feature vector

  == Shared Backbone ==
    -> Linear(input_dim -> 512)
    -> LayerNorm(512)
    -> GELU()
    -> Dropout(0.3)
    -> Linear(512 -> 256)
    -> LayerNorm(256)
    -> GELU()
    -> Dropout(0.3)
    -> backbone_out: (B, 256)

  == Multi-Quantile Head ==
    -> Linear(256 -> 5 * 3)    -- 15 raw outputs: 5 weeks x 3 quantiles
    -> Softplus()              -- guarantees all quantile boundaries >= 0.0
    -> reshape(B, 5, 3)
    -> quantile_outputs: (B, 5, 3)
       dim[-1] index 0 -> q=0.1  (Lower / Pessimistic edge)
       dim[-1] index 1 -> q=0.5  (Conditional Median -- used in inference)
       dim[-1] index 2 -> q=0.9  (Upper / Severe drought edge)

  Forward Signature:
    forward(self, x) -> quantile_outputs
      quantile_outputs : (B, 5, 3)  strictly non-negative via Softplus

Changes from v24
----------------
  [v25] Hurdle Architecture ABOLISHED: no head_prob (BCE), no head_sev (SmoothL1).
  [v25] BiLSTM + Dilated TCN ABOLISHED: replaced by wide flat MLP backbone.
  [v25] Single Quantile Head: Linear(256->15) + Softplus + reshape(B,5,3).
  [v25] input accepts (B, input_dim): flat 507-dim vector (13w x 39 feats).
  [v25] forward(x) -- no target_time argument needed.
  [v25] Softplus on quantile head natively enforces non-negativity.
  [v25] Multi-Quantile output: q=[0.1, 0.5, 0.9] for uncertainty quantification.

Retained from v24 (backbone philosophy)
-----------------------------------------
  [v24] Wide MLP configuration: 512 -> 256 with LayerNorm + GELU + Dropout(0.3).
  [v24] Softplus enforces >= 0 physical constraint on predictions.
"""

import torch
import torch.nn as nn


class FlatDroughtMLP(nn.Module):
    """
    v25 Wide Flat MLP for Multi-Quantile drought forecasting.

    Parameters
    ----------
    input_dim : int
        Flattened feature dimension (WINDOW_SIZE * len(FEATURE_COLS) = 507).
    horizon   : int
        Number of forecast weeks (default 5).
    n_quantiles : int
        Number of quantile levels to predict (default 3: q=0.1, 0.5, 0.9).
    dropout   : float
        Dropout probability in backbone (default 0.3).
    """

    # Quantile levels (index -> quantile mapping)
    QUANTILE_LEVELS = [0.1, 0.5, 0.9]   # index 0, 1, 2

    def __init__(
        self,
        input_dim: int,
        horizon: int = 5,
        n_quantiles: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.horizon     = horizon
        self.n_quantiles = n_quantiles
        self.dropout_p   = dropout

        # ----------------------------------------------------------------
        # Shared Backbone
        # Linear(input_dim -> 512) -> LN(512) -> GELU -> Dropout(0.3)
        # -> Linear(512 -> 256) -> LN(256) -> GELU -> Dropout(0.3)
        # ----------------------------------------------------------------
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ----------------------------------------------------------------
        # Multi-Quantile Head
        # Linear(256 -> horizon * n_quantiles) + Softplus
        # Output (B, 15) -> reshape (B, horizon, n_quantiles)
        # Softplus: guarantees all quantile boundaries >= 0 (physical constraint)
        # ----------------------------------------------------------------
        self.quantile_head = nn.Sequential(
            nn.Linear(256, horizon * n_quantiles),
            nn.Softplus(),
        )

        # Shape debug flag (prints once on first forward pass)
        self._printed_shape = False

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, input_dim)  -- flattened 13-week feature window

        Returns
        -------
        quantile_outputs : (B, horizon, n_quantiles)
            Strictly non-negative multi-quantile predictions via Softplus.
            dim[-1]:
              index 0 -> q=0.1  (Lower bound / Pessimistic edge)
              index 1 -> q=0.5  (Conditional Median -- used in inference)
              index 2 -> q=0.9  (Upper bound / Severe drought edge)
        """
        B = x.size(0)

        # Shared backbone: (B, input_dim) -> (B, 256)
        backbone_out = self.backbone(x)      # (B, 256)

        # Multi-quantile head: (B, 256) -> (B, horizon * n_quantiles)
        raw_out = self.quantile_head(backbone_out)   # (B, 15)

        # Reshape to (B, horizon, n_quantiles)
        quantile_outputs = raw_out.view(B, self.horizon, self.n_quantiles)
        # (B, 5, 3)

        # Shape debug – print once on first forward call
        if not self._printed_shape:
            print(
                f"  [v25 Shape Debug] x: {tuple(x.shape)}  "
                f"| backbone_out: {tuple(backbone_out.shape)}  "
                f"| quantile_outputs: {tuple(quantile_outputs.shape)}  "
                f"  (B, horizon={self.horizon}, quantiles={self.n_quantiles})"
            )
            self._printed_shape = True

        return quantile_outputs   # (B, 5, 3)

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> str:
        h  = self.horizon
        q  = self.n_quantiles
        lines = [
            "FlatDroughtMLP Architecture (v25 – Multi-Quantile Pinball + Adversarial Weights)",
            "=" * 90,
            f"  Input dim    : {self.input_dim}  "
            f"(WINDOW_SIZE={13} x FEATURE_COLS={self.input_dim // 13} = {self.input_dim})",
            "",
            "  == Shared Backbone ==",
            f"  Linear({self.input_dim} -> 512) -> LayerNorm(512) -> GELU() -> Dropout({self.dropout_p})",
            "  Linear(512 -> 256) -> LayerNorm(256) -> GELU() -> Dropout(0.3)",
            "  backbone_out : (B, 256)",
            "",
            "  == Multi-Quantile Head ==",
            f"  Linear(256 -> {h * q}) -> Softplus()  -- guarantees all outputs >= 0",
            f"  reshape: (B, {h * q}) -> (B, {h}, {q})",
            f"  quantile_outputs: (B, {h}, {q})",
            "     dim[-1] index 0 -> q=0.1  (Lower / Pessimistic)",
            "     dim[-1] index 1 -> q=0.5  (Conditional Median -- INFERENCE TARGET)",
            "     dim[-1] index 2 -> q=0.9  (Upper / Severe drought)",
            "",
            "  == Forward Signature ==",
            "  forward(x)  -- x: (B, input_dim)  [flat 507-dim vector]",
            "  return quantile_outputs  (B, 5, 3)  strictly non-negative",
            "",
            "  == Inference ==",
            "  prediction = quantile_outputs[:, :, 1]  # q=0.5 median channel",
            "  NO manual thresholding. NO Sigmoid gate. NO hurdle multiplication.",
            "-" * 90,
            "  [v25] Hurdle ABOLISHED: no head_prob (BCE), no head_sev (SmoothL1).",
            "  [v25] BiLSTM + TCN ABOLISHED: replaced by flat MLP backbone.",
            "  [v25] Quantile Head: q=[0.1, 0.5, 0.9]; Softplus non-negativity.",
            "  [v25] forward(x) -- no target_time argument.",
            "  [v24] Backbone config retained: 512->256, LN+GELU+Dropout(0.3).",
            "-" * 90,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy stub: DroughtLSTM renamed but alias kept for backward-compat imports
# ---------------------------------------------------------------------------
DroughtLSTM = FlatDroughtMLP   # v25: alias for any stale imports
