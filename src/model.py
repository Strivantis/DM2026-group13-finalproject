"""
model.py -- v37 CORAL-TFT: Consistent Rank Logits Head over TFT Backbone
==========================================================================

v37 Paradigm: Ordinal CDF Forecasting with CORAL Consistent Bias Head
-----------------------------------------------------------------------
  Problem (v22–v36): Continuous-loss regression suffers from "1.0 Anchor
    Degradation".  59.63% zero targets drown gradient signals, forcing all
    predictions to cluster at 1.0.

  Solution (v37) — CORALTFTWrapper:
    1. TFT Backbone: The TemporalFusionTransformer (pytorch_forecasting) with
       output_size=1 generates a single unbounded scalar logit per time-step.
       All Variable Selection Networks and Multi-Head Spatio-Temporal Attention
       layers are fully preserved.  Hidden state: d_model = TFT_HIDDEN_SIZE.

    2. CORAL Consistent Bias Head (NEW):
       a. Shared scalar projection: g(H_{i,t}) — one linear layer, weights
          shared across ALL 50 ordinal thresholds (CORAL structural constraint).
       b. Monotonic Ordered Biases: B = [b_1, b_2, ..., b_50] where
          b_1 > b_2 > ... > b_50 is enforced via:
            increments = softplus(raw_params)   (positive)
            biases     = -cumsum(increments)    (strictly decreasing)
       c. CDF probability vector:
            p̂_{ik} = σ( g(H_{i,t}) + b_k )   k = 1..50
          Monotonic bias sharing guarantees: p̂_{i1} ≥ p̂_{i2} ≥ ... ≥ p̂_{i50}
          (survival probabilities decrease monotonically — physically correct).

    3. Masked Ordinal BCE Loss:
         L_v37 = -1/K ∑_k [ z_{ik} log(p̂_{ik}) + (1-z_{ik}) log(1-p̂_{ik}) ]
       where z_{ik} = 1 iff y_i ≥ t_k, else 0.
       Absolute-zero samples (z = 0 everywhere) produce zero gradient on
       high-drought bins as p̂ → 0, freeing tail thresholds to learn from
       active drought signatures without zero-mass drag-down.

    4. Conditional Median Decoder (inference):
         ŷ_i = Σ_k  I(p̂_{ik} ≥ 0.5) × Δt      (Δt = 0.1)
       Count of thresholds with survival ≥ 50% times step size.

    5. CDF Saturation Entropy Monitor (validation):
       H = -[p log p + (1-p) log(1-p)]  per threshold
       Low mean entropy → well-polarized distributions (healthy learning).
       Printed each validation epoch alongside Tail Recovery Delta.

Architecture Summary
--------------------
  CORALTFTWrapper (pl.LightningModule)
    ├── tft (TemporalFusionTransformer)
    │     Backbone: VSN + Multi-Head Attention + LSTM encoder/decoder
    │     output_size=1  →  scalar logit (B, T, 1) per prediction step
    └── coral_bias_raw (nn.Parameter, shape [K=50])
          → monotonic biases via -cumsum(softplus(raw))
          Combined:  σ(scalar_logit + biases)  →  (B, T, K=50) CDF probs

Training
--------
  Optimizer : AdamW(lr=1e-3, weight_decay=1e-3)
  Scheduler : CosineAnnealingLR(T_max=50, eta_min=1e-6)
  Loss      : Masked Ordinal BCE (mean over B × T × K)
  Batch     : 256 / 512
  Epochs    : 50 (hard limit, early stopping patience=5)

Legacy
------
  DroughtSequenceNet (v33 — Dual Dilated Conv1d + BiLSTM + Region Embedding)
  is retained below as a legacy alias.  Not used in v37.

v37 Changes over v36
--------------------
  [v37] INTRODUCE: CORALTFTWrapper(pl.LightningModule) — CORAL CDF head.
  [v37] INTRODUCE: coral_bias_raw Parameter + get_monotonic_biases().
  [v37] INTRODUCE: _ordinal_bce_loss() — Masked Ordinal Binary Cross-Entropy.
  [v37] INTRODUCE: decode_median() — Conditional Median Integration (Δt=0.1).
  [v37] INTRODUCE: _cdf_saturation_entropy() — per-threshold polarization audit.
  [v37] INTRODUCE: on_validation_epoch_end() monitoring printout with:
                   - CDF Saturation Entropy by drought bin
                   - Per-bin Conditional Median MAE / Bias
                   - Tail Recovery Delta (true labels 4.0–5.0 → ŷ > 3.5)
  [v37] ABOLISH   : QuantileLoss / continuous regression output head.
  [v37] ABOLISH   : p50 / quantile channel extraction.
  [v37] RETAIN    : TFT backbone architecture (VSN, attention, LSTM encoder/decoder).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer

# ---------------------------------------------------------------------------
# CORAL ordinal constants (mirrored from dataset.py for local use)
# ---------------------------------------------------------------------------
CORAL_K       : int   = 50
CORAL_DELTA_T : float = 0.1
# Thresholds: [0.1, 0.2, ..., 5.0]
_CORAL_THRESHOLDS_LIST: list[float] = [
    round(k * CORAL_DELTA_T, 1) for k in range(1, CORAL_K + 1)
]


# ===========================================================================
# CORALTFTWrapper — Consistent Rank Logits Head over TFT Backbone
# ===========================================================================

class CORALTFTWrapper(pl.LightningModule):
    """
    v37 CORAL-TFT Lightning Module.

    Wraps a ``TemporalFusionTransformer`` (with output_size=1) as a scalar
    logit extractor, then applies the CORAL Consistent Bias Head to produce
    50 monotonically ordered survival probabilities per time-step.

    Parameters
    ----------
    tft_backbone : TemporalFusionTransformer
        Pre-built TFT instance (from_dataset with output_size=1).
        Used as a pure feature extractor — its own training_step/loss
        are never invoked.
    n_ordinal    : int   — number of ordinal thresholds K (default 50)
    lr           : float — base learning rate for AdamW (default 1e-3)
    max_epochs   : int   — T_max for CosineAnnealingLR (default 50)
    """

    def __init__(
        self,
        tft_backbone : TemporalFusionTransformer,
        n_ordinal    : int   = CORAL_K,
        lr           : float = 2e-3,   # [v37.1] peak LR for warmup-cosine schedule
        max_epochs   : int   = 50,
    ):
        super().__init__()

        # TFT backbone — provides scalar logit per decoder step.
        # Lightning sub-module: automatically moved to correct device
        # and included in self.parameters().
        self.tft        = tft_backbone
        self.n_ordinal  = n_ordinal
        self.lr         = lr
        self.max_epochs = max_epochs

        # ------------------------------------------------------------------
        # CORAL Consistent Bias Parameters
        # ------------------------------------------------------------------
        # coral_bias_raw  (K,)  — unconstrained trainable parameters.
        # Actual monotonic biases are computed from these via:
        #   increments = softplus(coral_bias_raw)  — strictly positive
        #   biases     = -cumsum(increments)        — strictly decreasing
        # This guarantees:  b_1 > b_2 > ... > b_K
        # which in turn guarantees: p̂_1 ≥ p̂_2 ≥ ... ≥ p̂_K  (no oscillations)
        #
        # Initialisation to zeros → initial biases are uniform and symmetric.
        # ------------------------------------------------------------------
        self.coral_bias_raw = nn.Parameter(torch.zeros(n_ordinal))

        # Register ordinal thresholds as a buffer (no grad, moves with device)
        thresholds = torch.tensor(_CORAL_THRESHOLDS_LIST, dtype=torch.float32)
        self.register_buffer("ordinal_thresholds", thresholds)   # (K,)

        # Accumulator for per-epoch validation monitoring
        self._val_outputs: list[dict] = []

    # -----------------------------------------------------------------------
    # CORAL head helpers
    # -----------------------------------------------------------------------

    def get_monotonic_biases(self) -> torch.Tensor:
        """
        Compute the K strictly-decreasing bias vector from unconstrained params.

        Returns
        -------
        biases : Tensor shape (K,)
            b_1 > b_2 > ... > b_K guaranteed by construction.
        """
        increments = F.softplus(self.coral_bias_raw)        # (K,)  positive
        biases     = -torch.cumsum(increments, dim=0)       # (K,)  decreasing
        return biases

    def forward(self, x: dict) -> torch.Tensor:
        """
        Full CORAL-TFT forward pass.

        Parameters
        ----------
        x : dict
            Batch input dict produced by TimeSeriesDataSet (encoder_cont,
            decoder_cont, encoder_cat, etc.).

        Returns
        -------
        coral_probs : Tensor shape (B, T_pred, K)
            p̂_{ik} = σ( g(H_{i,t}) + b_k )
            Values in (0, 1); monotonically decreasing along the K dimension.
        """
        # Get TFT output: scalar logit per decoder time-step
        # out.prediction shape: (B, T_pred, 1)  with output_size=1
        out          = self.tft(x)
        scalar_logit = out.prediction               # (B, T_pred, 1)

        biases = self.get_monotonic_biases()        # (K,)

        # Broadcast: (B, T, 1) + (1, 1, K)  →  (B, T, K)
        coral_probs = torch.sigmoid(
            scalar_logit + biases.view(1, 1, -1)   # (B, T_pred, K)
        )
        return coral_probs                          # (B, T_pred, K)

    # -----------------------------------------------------------------------
    # Loss & ordinal encoding
    # -----------------------------------------------------------------------

    def _make_ordinal_labels(self, y: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous target scores to ordinal multi-label binary vectors.

        z_{ik} = 1  if  y_i ≥ t_k,  else 0

        Parameters
        ----------
        y : Tensor (B, T)  — continuous drought scores in [0, 5]

        Returns
        -------
        z : Tensor (B, T, K)  — binary ordinal label matrix.
            All-zero rows for y=0.0 (absolute drought-free samples).
        """
        thresholds = self.ordinal_thresholds                     # (K,)
        # (B, T, 1) >= (1, 1, K)  →  (B, T, K)
        z = (y.unsqueeze(-1) >= thresholds.view(1, 1, -1)).float()
        return z

    @staticmethod
    def _ordinal_bce_loss(
        coral_probs    : torch.Tensor,
        ordinal_labels : torch.Tensor,
    ) -> torch.Tensor:
        """
        Masked Ordinal Binary Cross-Entropy Loss  (L_v37).

          L = -1/K ∑_k [ z_k log(p̂_k) + (1-z_k) log(1-p̂_k) ]

        Gradient anti-anchor mechanism:
          For absolute-zero samples (z=0 everywhere), as p̂ → 0 for high-
          drought bins, log(1-p̂) → 0 and the gradient contribution vanishes.
          This liberates high-severity thresholds to learn from active drought
          signatures without being dragged down by the 59.63% zero mass.

        Parameters
        ----------
        coral_probs    : (B, T, K)  — predicted survival probabilities
        ordinal_labels : (B, T, K)  — binary ground-truth ordinal labels

        Returns
        -------
        loss : scalar Tensor
        """
        # Clamp for numerical stability (avoids log(0))
        p   = coral_probs.clamp(1e-7, 1.0 - 1e-7)
        z   = ordinal_labels
        bce = -(z * torch.log(p) + (1.0 - z) * torch.log(1.0 - p))
        # Average over all (B × T × K) elements
        return bce.mean()

    # -----------------------------------------------------------------------
    # Conditional Median Decoder
    # -----------------------------------------------------------------------

    @staticmethod
    def decode_median(
        coral_probs : torch.Tensor,
        delta_t     : float = CORAL_DELTA_T,
    ) -> torch.Tensor:
        """
        Conditional Median reconstruction via discrete CDF integration.

          ŷ_i = Σ_k  I(p̂_{ik} ≥ 0.5) × Δt

        Count of thresholds where survival probability ≥ 50%, multiplied by
        the step size Δt=0.1.  This estimates the conditional median:
          - All 50 thresholds passed (p ≥ 0.5) → ŷ = 5.0
          - No threshold passed      (p < 0.5) → ŷ = 0.0  (absolute zero)
          - 23 thresholds passed               → ŷ = 2.3  (moderate drought)

        Parameters
        ----------
        coral_probs : Tensor (B, T, K) or (N, K)
        delta_t     : float — threshold step size

        Returns
        -------
        y_hat : Tensor (...) — shape identical to coral_probs with K dim removed
        """
        return (coral_probs >= 0.5).float().sum(dim=-1) * delta_t

    # -----------------------------------------------------------------------
    # CDF Saturation Entropy Monitor
    # -----------------------------------------------------------------------

    @staticmethod
    def _cdf_saturation_entropy(
        coral_probs: torch.Tensor,
        eps        : float = 1e-7,
    ) -> torch.Tensor:
        """
        Per-element binary entropy of the predicted CDF probability vector.

          H_{ik} = -[ p̂_{ik} log(p̂_{ik}) + (1-p̂_{ik}) log(1-p̂_{ik}) ]

        Interpretation:
          - Entropy → 0 : p̂ near 0 or 1 (polarised, confident CDF step)
          - Entropy → log(2) ≈ 0.693 : p̂ ≈ 0.5 (uncertain, blurred boundary)

        Well-trained CORAL produces low entropy everywhere EXCEPT at the
        transition threshold, indicating clean CDF step-function behaviour.

        Parameters
        ----------
        coral_probs : Tensor (..., K)

        Returns
        -------
        mean_entropy : scalar Tensor — mean over all positions and thresholds
        """
        p = coral_probs.clamp(eps, 1.0 - eps)
        h = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        return h.mean()

    # -----------------------------------------------------------------------
    # Lightning: training step
    # -----------------------------------------------------------------------

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Single training iteration with Masked Ordinal BCE Loss.

        Batch structure from TimeSeriesDataSet:
            batch = (x_dict, (y_values, y_weights))
            y_values : Tensor (B, T_pred)  — raw [0,5] drought scores
        """
        x, y_data = batch
        # Unpack target (may be tuple with weights)
        y = y_data[0] if isinstance(y_data, (tuple, list)) else y_data
        # y : (B, HORIZON)  — raw continuous scores, identity-normalised [0, 5]

        # Forward: (B, HORIZON, K) CORAL survival probs
        coral_probs = self(x)

        # Build ordinal label matrix: z_{ik} = 1 iff y_i ≥ t_k
        ordinal_labels = self._make_ordinal_labels(y)   # (B, HORIZON, K)

        # [v37.1] Label smoothing: prevent overconfidence at hard binary boundaries
        # z_smoothed ∈ [0.01, 0.99] — prevents gradient vanishing near exact {0,1} labels
        z_smoothed = ordinal_labels * 0.98 + 0.01
        loss = F.binary_cross_entropy(coral_probs, z_smoothed, reduction="mean")

        # Logging
        batch_sz = y.shape[0]
        self.log(
            "train_loss", loss,
            prog_bar=True, on_step=True, on_epoch=True,
            batch_size=batch_sz,
        )

        # Log zero-fraction of targets (tracks 1.0 anchor escape over epochs)
        zero_frac = (y == 0.0).float().mean()
        self.log(
            "train_zero_frac", zero_frac,
            prog_bar=False, on_step=False, on_epoch=True,
            batch_size=batch_sz,
        )

        return loss

    # -----------------------------------------------------------------------
    # Lightning: validation step  (per-batch)
    # -----------------------------------------------------------------------

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Validation iteration: computes BCE loss, conditional median, and
        accumulates outputs for epoch-level CDF entropy & tail recovery audit.
        """
        x, y_data = batch
        y = y_data[0] if isinstance(y_data, (tuple, list)) else y_data
        # y : (B, HORIZON)

        coral_probs    = self(x)                           # (B, HORIZON, K)
        ordinal_labels = self._make_ordinal_labels(y)      # (B, HORIZON, K)

        loss = self._ordinal_bce_loss(coral_probs, ordinal_labels)

        # Conditional median decoded predictions
        y_hat = self.decode_median(coral_probs)            # (B, HORIZON)

        # CDF Saturation Entropy (lower = more polarized)
        entropy = self._cdf_saturation_entropy(coral_probs)

        batch_sz = y.shape[0]
        self.log(
            "val_loss", loss,
            prog_bar=True, on_step=False, on_epoch=True,
            batch_size=batch_sz,
        )
        self.log(
            "val_cdf_entropy", entropy,
            prog_bar=True, on_step=False, on_epoch=True,
            batch_size=batch_sz,
        )

        # Accumulate for epoch-end binned monitoring
        self._val_outputs.append({
            "y_hat"  : y_hat.detach().cpu().float().numpy(),       # (B, HORIZON)
            "y_true" : y.detach().cpu().float().numpy(),            # (B, HORIZON)
            "probs"  : coral_probs.detach().cpu().float().numpy(),  # (B, HORIZON, K)
        })

        return loss

    # -----------------------------------------------------------------------
    # Lightning: validation epoch-end — full monitoring printout
    # -----------------------------------------------------------------------

    def on_validation_epoch_end(self) -> None:
        """
        V37 Validation Monitor — printed once per epoch after all val batches:

          1. CDF Saturation Entropy per drought bin
             Verify probabilities polarize cleanly toward {0, 1} in extreme
             bins instead of scattering around blurred means.

          2. Per-bin Conditional Median MAE / Bias
             Drought resolution bins: [0], (0–1], (1–2], (2–3], (3–4], (4–5]

          3. Tail Recovery Delta
             True labels in [4.0, 5.0]: verify ŷ > 3.5 (escape from 1.0 sink).
        """
        if not self._val_outputs:
            return

        # Aggregate all batch outputs
        all_y_hat  = np.concatenate([o["y_hat"].ravel()  for o in self._val_outputs])
        all_y_true = np.concatenate([o["y_true"].ravel() for o in self._val_outputs])
        all_probs  = np.concatenate(
            [o["probs"].reshape(-1, self.n_ordinal) for o in self._val_outputs],
            axis=0,
        )   # (N_total, K)

        epoch = self.current_epoch

        print(f"\n{'='*74}")
        print(f"  [v37 CORAL Validation Monitor]  Epoch {epoch}")
        print(f"{'='*74}")

        # ── 1. Overall CDF Saturation Entropy ──────────────────────────────
        eps = 1e-7
        p   = all_probs.clip(eps, 1.0 - eps)
        h   = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        mean_entropy_overall = h.mean()

        # Entropy by drought bin (split probs by true score bracket)
        bin_defs   = [(0.0, 0.0, "[0]"),
                      (0.0, 1.0, "(0–1]"),
                      (1.0, 2.0, "(1–2]"),
                      (2.0, 3.0, "(2–3]"),
                      (3.0, 4.0, "(3–4]"),
                      (4.0, 5.0, "(4–5]")]

        print(f"\n  CDF Saturation Entropy  (↓ better, log2≈0.693 = random)")
        print(f"  {'Bin':<10}  {'N':>7}  {'Entropy':>10}  {'Polarized?':>12}")
        print(f"  {'-'*48}")

        for lo, hi, label in bin_defs:
            if lo == 0.0 and hi == 0.0:
                mask = all_y_true == 0.0
            else:
                mask = (all_y_true > lo) & (all_y_true <= hi)
            n = mask.sum()
            if n == 0:
                print(f"  {label:<10}  {0:>7}  {'N/A':>10}  {'N/A':>12}")
                continue
            bin_h     = h[mask].mean()
            polarized = "✓ clean" if bin_h < 0.35 else ("≈ ok" if bin_h < 0.5 else "✗ blurred")
            print(f"  {label:<10}  {n:>7,}  {bin_h:>10.4f}  {polarized:>12}")

        print(f"\n  Overall entropy : {mean_entropy_overall:.4f}")

        # ── 2. Per-bin Conditional Median MAE / Bias ───────────────────────
        print(f"\n  Conditional Median Decoded Stats (ŷ = Σ I(p≥0.5)·Δt)")
        print(f"  {'Bin':<10}  {'N':>7}  {'MAE':>8}  {'Bias':>8}  "
              f"{'AvgPred':>9}  {'AvgTrue':>9}")
        print(f"  {'-'*62}")

        for lo, hi, label in bin_defs:
            if lo == 0.0 and hi == 0.0:
                mask = all_y_true == 0.0
            else:
                mask = (all_y_true > lo) & (all_y_true <= hi)
            n = mask.sum()
            if n == 0:
                print(f"  {label:<10}  {0:>7}  {'N/A':>8}  {'N/A':>8}  "
                      f"{'N/A':>9}  {'N/A':>9}")
                continue
            mae  = np.abs(all_y_hat[mask] - all_y_true[mask]).mean()
            bias = (all_y_hat[mask] - all_y_true[mask]).mean()
            avg_pred = all_y_hat[mask].mean()
            avg_true = all_y_true[mask].mean()
            print(f"  {label:<10}  {n:>7,}  {mae:>8.4f}  {bias:>8.4f}  "
                  f"{avg_pred:>9.4f}  {avg_true:>9.4f}")

        # ── 3. Tail Recovery Delta ─────────────────────────────────────────
        tail_mask = (all_y_true >= 4.0) & (all_y_true <= 5.0)
        print(f"\n  Tail Recovery Delta  [true label ∈ [4.0, 5.0]]")
        n_tail = tail_mask.sum()
        if n_tail == 0:
            print(f"  No extreme drought samples in this validation fold.")
        else:
            tail_preds    = all_y_hat[tail_mask]
            tail_recovery = (tail_preds >= 3.5).sum() / n_tail
            tail_mae      = np.abs(tail_preds - all_y_true[tail_mask]).mean()
            tail_avg_pred = tail_preds.mean()
            print(f"  N={n_tail:,}  avg_pred={tail_avg_pred:.4f}  MAE={tail_mae:.4f}"
                  f"  pred≥3.5: {tail_recovery:.1%}")
            if tail_recovery >= 0.5:
                print("  ✓ TAIL RECOVERY ACTIVE — model has escaped the 1.0 anchor sink!")
            else:
                print("  ✗ Tail anchor still dominant "
                      "(expected in early epochs; watch entropy trend)")

        # ── 4. Global prediction distribution ─────────────────────────────
        print(f"\n  Global ŷ Distribution:")
        print(f"  mean={all_y_hat.mean():.4f}  std={all_y_hat.std():.4f}  "
              f"p50={np.percentile(all_y_hat, 50):.4f}  "
              f"p90={np.percentile(all_y_hat, 90):.4f}  "
              f"max={all_y_hat.max():.4f}")
        print(f"  zero-frac(ŷ) = {(all_y_hat == 0.0).mean():.4f}  |  "
              f"zero-frac(y_true) = {(all_y_true == 0.0).mean():.4f}")
        print(f"{'='*74}\n")

        # Clear accumulator for next epoch
        self._val_outputs.clear()

    # -----------------------------------------------------------------------
    # Lightning: predict step (inference)
    # -----------------------------------------------------------------------

    def predict_step(
        self,
        batch      : tuple,
        batch_idx  : int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """
        Inference step using Conditional Median Decoder.

        Returns
        -------
        y_hat : Tensor (B, HORIZON)  — decoded drought forecasts
        """
        x, _        = batch
        coral_probs = self(x)                 # (B, HORIZON, K)
        y_hat       = self.decode_median(coral_probs)   # (B, HORIZON)
        return y_hat.cpu()

    # -----------------------------------------------------------------------
    # Lightning: optimiser configuration
    # -----------------------------------------------------------------------

    def configure_optimizers(self) -> dict:
        """
        [v37.1] AdamW + 3-epoch Linear Warmup → CosineAnnealingLR(T_max=47).

        Phase 1 (epochs 0–2): LinearLR ramps lr/3 → lr  (3 epochs)
        Phase 2 (epochs 3–49): CosineAnnealingLR decays lr → 1e-6 (47 epochs)
        Combined via SequentialLR with milestones=[3].

        Optimises BOTH the TFT backbone parameters and the CORAL bias parameters
        jointly (self.parameters() covers all sub-modules recursively).
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr           = self.lr,
            weight_decay = 1e-3,
        )
        # Phase 1: Linear warmup — lr/3 → lr over 3 epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 1.0 / 3.0,
            end_factor   = 1.0,
            total_iters  = 3,
        )
        # Phase 2: Cosine decay — lr → 1e-6 over 47 epochs
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = 47,
            eta_min = 1e-6,
        )
        combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers = [warmup_scheduler, cosine_scheduler],
            milestones = [3],
        )
        return {
            "optimizer"    : optimizer,
            "lr_scheduler" : {
                "scheduler" : combined_scheduler,
                "interval"  : "epoch",
                "frequency" : 1,
                "name"      : "warmup_cosine_lr",
            },
        }

    # -----------------------------------------------------------------------
    # Debug / summary helpers
    # -----------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> str:
        lines = [
            "CORALTFTWrapper Architecture (v37 — CORAL Consistent Bias Head + TFT Backbone)",
            "=" * 85,
            "",
            "  == TFT Backbone (Feature Extractor) ==",
            "  TemporalFusionTransformer  (pytorch_forecasting)",
            "    Variable Selection Networks  → gated residual encoding",
            "    Multi-Head Temporal Self-Attention  → spatio-temporal context",
            "    Encoder LSTM + Decoder LSTM  → multi-step state propagation",
            f"    output_size = 1  →  scalar logit  g(H_{{i,t}})  per decoder step",
            "    INPUT  : (B, 13-week encoder + 5-week decoder, 35 features)",
            "    OUTPUT : (B, 5, 1)  — scalar logit per prediction week",
            "",
            "  == CORAL Consistent Bias Head ==",
            f"  coral_bias_raw  : nn.Parameter  shape ({self.n_ordinal},)  [unconstrained]",
            f"  get_monotonic_biases():",
            "    increments = softplus(coral_bias_raw)   → (K,)  positive",
            "    biases     = -cumsum(increments)         → (K,)  strictly decreasing",
            "    Guarantees:  b_1 > b_2 > ... > b_K",
            "",
            "  Forward pass:",
            "    scalar_logit     : (B, T=5, 1)    [from TFT backbone]",
            "    biases           : (K=50,)         [CORAL biases]",
            f"    coral_probs     : (B, 5, {self.n_ordinal})  = σ(scalar + biases)",
            "    Invariant: p̂_1 ≥ p̂_2 ≥ ... ≥ p̂_K  (monotone survival probs)",
            "",
            "  == Masked Ordinal BCE Loss ==",
            "  z_{ik} = 1 if y_i ≥ t_k, else 0   (50-step binary label vector)",
            "  L = -1/K ∑_k [ z_k log(p̂_k) + (1-z_k) log(1-p̂_k) ]",
            "  Zero-sample anti-anchor: absolute-zero inputs → zero-gradient on",
            "  high-drought bins as p̂ → 0, freeing tail thresholds.",
            "",
            "  == Conditional Median Decoder (inference) ==",
            "  ŷ = Σ_k  I(p̂_k ≥ 0.5) × 0.1",
            "",
            "  == Training ==",
            f"  Optimizer  : AdamW(lr={self.lr}, weight_decay=1e-3)",
            f"  Scheduler  : CosineAnnealingLR(T_max={self.max_epochs}, eta_min=1e-6)",
            "  Loss       : Masked Ordinal BCE",
            "",
            f"  Total trainable parameters : {self.count_parameters():,}",
            "=" * 85,
        ]
        return "\n".join(lines)


# ===========================================================================
# Legacy: DroughtSequenceNet (v33 — retained for backward compatibility)
# ===========================================================================

class DroughtSequenceNet(nn.Module):
    """
    v33 Geo-Memory Hybrid Sequence Network  (LEGACY — not used in v37).

    Three parallel branches:
      - Branch A (Dual-layer Dilated Conv1d):  multi-week local anomaly detector.
      - Branch B (Spatial Dropout + BiLSTM):  long-term cumulative process memory.
      - Branch C (Region Embedding):           16-dim geographical identity vector.

    Retained as legacy alias.  v37 uses CORALTFTWrapper.
    """

    def __init__(
        self,
        seq_len     : int   = 13,
        n_features  : int   = 27,
        conv_out    : int   = 64,
        lstm_hidden : int   = 64,
        lstm_layers : int   = 3,
        mlp_hidden  : int   = 128,
        horizon     : int   = 5,
        dropout     : float = 0.2,
        num_regions : int   = 2248,
        embed_dim   : int   = 16,
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

        # Branch A: Dual-Layer Dilated TCN
        self.branch_a_conv1 = nn.Sequential(
            nn.Conv1d(n_features, conv_out, 3, padding=1, dilation=1),
            nn.GELU(),
            nn.InstanceNorm1d(conv_out, affine=True),
        )
        self.branch_a_conv2 = nn.Sequential(
            nn.Conv1d(conv_out, conv_out, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.InstanceNorm1d(conv_out, affine=True),
        )
        self.branch_a_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Branch B: Spatial Dropout + BiLSTM
        self.branch_b_spatial_dropout = nn.Dropout1d(p=0.1)
        self.branch_b_lstm = nn.LSTM(
            input_size=n_features, hidden_size=lstm_hidden,
            num_layers=lstm_layers, bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_size = lstm_hidden * 2

        # Branch C: Geo-Memory Region Embedding
        self.region_embed = nn.Embedding(num_regions, embed_dim)

        # Fusion MLP
        fused_size = conv_out + lstm_out_size + embed_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, horizon),
        )
        self._printed_shape = False

    def forward(self, x_climate: torch.Tensor, x_region: torch.Tensor) -> torch.Tensor:
        x_conv   = x_climate.transpose(1, 2)
        a1       = self.branch_a_conv1(x_conv)
        a2       = self.branch_a_conv2(a1)
        branch_a = self.branch_a_pool(a2)

        x_b = x_climate.transpose(1, 2)
        x_b = self.branch_b_spatial_dropout(x_b)
        x_b = x_b.transpose(1, 2)
        lstm_out, _ = self.branch_b_lstm(x_b)
        branch_b    = lstm_out[:, -1, :]

        region_feat = self.region_embed(x_region)

        fused  = torch.cat([branch_a, branch_b, region_feat], dim=-1)
        output = self.fusion_head(fused)

        if not self._printed_shape:
            print(
                f"  [v33 Legacy Shape Debug]"
                f"  branch_a: {tuple(branch_a.shape)}"
                f"  | branch_b: {tuple(branch_b.shape)}"
                f"  | output: {tuple(output.shape)}"
            )
            self._printed_shape = True
        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Legacy aliases for stale imports
FlatDroughtMLP = DroughtSequenceNet
DroughtLSTM    = DroughtSequenceNet
