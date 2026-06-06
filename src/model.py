"""
model.py – FlatDroughtMLP architecture (v45).

Wide flat MLP for multi-quantile drought-score forecasting.

Input:  x (B, input_dim)       flat 13-week feature vector (e.g. 406 dims)
Output: (B, horizon, 3)        three quantile channels per forecast week
          dim[-1] index 0  q=0.1  (lower bound)
          dim[-1] index 1  q=0.5  (conditional median – used at inference)
          dim[-1] index 2  q=0.9  (upper bound)

Backbone: Linear(input_dim→512) → LayerNorm → GELU → Dropout(0.3)
          → Linear(512→256)     → LayerNorm → GELU → Dropout(0.3)
Head:     Linear(256 → horizon×3) → Softplus  (guarantees outputs ≥ 0)
          → reshape (B, horizon, 3)
"""

import torch
import torch.nn as nn


class FlatDroughtMLP(nn.Module):
    """
    Parameters
    ----------
    input_dim   : int    flattened feature dimension (e.g. 406)
    horizon     : int    number of forecast weeks (default 5)
    n_quantiles : int    quantile levels (default 3: q=0.1, 0.5, 0.9)
    dropout     : float  backbone dropout probability (default 0.3)
    """

    QUANTILE_LEVELS = [0.1, 0.5, 0.9]

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

        self.quantile_head = nn.Sequential(
            nn.Linear(256, horizon * n_quantiles),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, input_dim)

        Returns
        -------
        (B, horizon, n_quantiles)  all values ≥ 0 via Softplus
        """
        B           = x.size(0)
        backbone_out = self.backbone(x)
        raw_out      = self.quantile_head(backbone_out)
        return raw_out.view(B, self.horizon, self.n_quantiles)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
