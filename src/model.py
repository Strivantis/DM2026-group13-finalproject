"""
src/model.py  — v40 CORALPatchTSTWrapper
=========================================
Architecture:
  1. PatchTST_backbone  (m.model from neuralforecast PatchTST)
       • Channel-independent patching over 40 feature channels
       • RevIN (Reversible Instance Normalization) applied internally
       • Input  : (B, n_features, context_len)  = (B, 40, 13)
       • Output : (B, n_features, horizon)       = (B, 40,  5)
  2. Linear projection  (n_features → 1)  → scalar forecast logit per step
  3. CORAL monotonic bias vector (50 thresholds, strictly decreasing)
  4. Sigmoid → P(Y > k·0.1) for k = 0..49

Loss: Masked Ordinal Binary Cross-Entropy  (label smoothing ε = 0.05)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from neuralforecast.models.patchtst import PatchTST
from neuralforecast.losses.pytorch import MAE as NF_MAE


# ---------------------------------------------------------------------------
# Constants (must match dataset.py)
# ---------------------------------------------------------------------------
N_FEATURES:  int = 40
CONTEXT_LEN: int = 13
HORIZON_LEN: int = 5
N_ORDINAL:   int = 50     # CORAL thresholds → bins × 0.1 ∈ [0.1, 5.0]
LABEL_SMOOTH: float = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_to_ordinal(y: torch.Tensor) -> torch.Tensor:
    """
    Convert continuous score in [0, 5] to a (*, N_ORDINAL) binary ordinal label.

    y_ord[j] = 1  iff  round(y * 10) > j   (j ∈ 0..N_ORDINAL-1)
    """
    k = torch.round(y * 10).long().clamp(0, N_ORDINAL)  # (...,)
    j = torch.arange(N_ORDINAL, device=y.device)         # (50,)
    # Broadcast: (..., N_ORDINAL)
    return (k.unsqueeze(-1) > j).float()


def masked_coral_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Ordinal BCE loss with label smoothing.
    Uses binary_cross_entropy_with_logits → AMP-safe (no pre-sigmoid required).

    logits : (B, H, K)  raw logits (NOT sigmoid-ed)
    y      : (B, H)     continuous scores [0, 5]
    """
    y_ord = score_to_ordinal(y)              # (B, H, K)
    # Label smoothing
    y_smooth = y_ord * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
    return F.binary_cross_entropy_with_logits(logits, y_smooth)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class CORALPatchTSTWrapper(pl.LightningModule):
    """
    PyTorch Lightning wrapper combining:
      • PatchTST (via neuralforecast) as the temporal feature extractor
      • CORAL ordinal regression head for discrete score prediction

    Parameters
    ----------
    n_ordinal   : int   number of CORAL threshold classifiers (default 50)
    lr          : float learning rate
    n_features  : int   number of input feature channels (default 40)
    hidden_size : int   PatchTST d_model
    n_heads     : int   attention heads in PatchTST
    patch_len   : int   patch length for PatchTST
    stride      : int   patch stride for PatchTST
    """

    def __init__(
        self,
        n_ordinal:   int   = N_ORDINAL,
        lr:          float = 1e-3,
        n_features:  int   = N_FEATURES,
        hidden_size: int   = 64,
        n_heads:     int   = 4,
        patch_len:   int   = 4,
        stride:      int   = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_ordinal = n_ordinal
        self.lr = lr

        # ------------------------------------------------------------------
        # 1. PatchTST backbone
        #    neuralforecast wraps the real backbone in `.model`.
        #    We instantiate the full PatchTST object and extract its
        #    internal PatchTST_backbone encoder (`.model`), which accepts
        #    (B, n_series, context_len) → (B, n_series, horizon).
        #    RevIN is baked in when revin=True.
        # ------------------------------------------------------------------
        _patchtst_wrapper = PatchTST(
            h             = HORIZON_LEN,
            input_size    = CONTEXT_LEN,
            n_series      = n_features,
            patch_len     = patch_len,
            stride        = stride,
            revin         = True,        # ← RevIN anti-covariate-shift
            hidden_size   = hidden_size,
            n_heads       = n_heads,
            loss          = NF_MAE(),
            valid_loss    = NF_MAE(),
            max_steps     = 1,           # disable internal training loop
        )
        # Extract standalone PyTorch backbone (no nixtla overhead at runtime)
        self.backbone: nn.Module = _patchtst_wrapper.model
        del _patchtst_wrapper

        # ------------------------------------------------------------------
        # 2. Numerical defense: LayerNorm(n_features) re-centers the
        #    unscaled physical outputs from RevIN back to N(0,1) before
        #    the logit projection — prevents BCE loss explosion.
        # ------------------------------------------------------------------
        self.feat_norm = nn.LayerNorm(n_features)

        # ------------------------------------------------------------------
        # 3. Linear head: collapses stabilized feature channels → scalar
        # ------------------------------------------------------------------
        self.logit_proj = nn.Linear(n_features, 1)

        # ------------------------------------------------------------------
        # 3. CORAL monotonic bias vector  (K raw parameters → K decreasing biases)
        # ------------------------------------------------------------------
        self.coral_bias_raw = nn.Parameter(torch.zeros(n_ordinal))

    # ------------------------------------------------------------------
    # CORAL: monotonically decreasing bias vector
    # ------------------------------------------------------------------
    def get_monotonic_biases(self) -> torch.Tensor:
        """
        Returns b[0] ≥ b[1] ≥ ... ≥ b[K-1]  (strictly decreasing biases).

        Ensures P(Y > k) = σ(logit + b[k]) is monotonically non-increasing in k.
        Implementation:
            b[0]   = coral_bias_raw[0]            (free)
            b[j]   = b[j-1] - softplus(raw[j])   (j ≥ 1, each step decreases)
        """
        first = self.coral_bias_raw[0:1]
        decrements = F.softplus(self.coral_bias_raw[1:])   # positive
        deltas = torch.cat([first, -decrements], dim=0)    # (K,)
        return torch.cumsum(deltas, dim=0)                 # (K,)  decreasing

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw CORAL logits (NOT sigmoid-ed) — AMP-safe for BCEWithLogitsLoss.

        Parameters
        ----------
        x : (B, CONTEXT_LEN, N_FEATURES)  = (B, 13, 40)

        Returns
        -------
        logits : (B, HORIZON_LEN, N_ORDINAL)  = (B, 5, 50)
                 Raw pre-sigmoid CORAL logits.
                 Apply torch.sigmoid() at inference to get exceedance probs.
        """
        # Reshape for channel-independent PatchTST:  (B, n_series, T)
        x_t = x.transpose(1, 2)                    # (B, 40, 13)

        # PatchTST backbone (with RevIN inside)
        raw_out = self.backbone(x_t)               # (B, 40, 5)

        raw_out = raw_out.transpose(1, 2)          # (B,  5, 40)

        # LayerNorm defense: re-center unscaled physical RevIN outputs → N(0,1)
        safe_features = self.feat_norm(raw_out)    # (B,  5, 40)

        # Scalar logit per time step
        scalar_logit = self.logit_proj(safe_features)  # (B, 5, 1)

        # Add CORAL monotonic biases
        biases = self.get_monotonic_biases()       # (50,)
        logits = scalar_logit + biases.view(1, 1, -1)  # (B, 5, 50)

        return logits                              # raw logits — no sigmoid here

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference: returns sigmoid(forward(x)).
        Called outside AMP context so BCEWithLogits safety is irrelevant here.

        Returns
        -------
        probs : (B, 5, 50)  exceedance probabilities ∈ [0, 1]
        """
        return torch.sigmoid(self(x))

    # ------------------------------------------------------------------
    # Lightning: training / validation steps
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y, _ = batch                    # x:(B,13,40)  y:(B,5)
        logits = self(x)                   # (B, 5, 50)  — raw logits, AMP-safe
        loss = masked_coral_bce(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)                   # (B, 5, 50)  — raw logits
        val_loss = masked_coral_bce(logits, y)

        # Greedy MAE: apply sigmoid, threshold at 0.5
        probs = torch.sigmoid(logits)
        pred_scores = (probs >= 0.5).float().sum(dim=-1) * 0.1  # (B, 5)
        mae = (pred_scores - y).abs().mean()

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae",  mae,      on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
