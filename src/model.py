"""
DroughtLSTM
===========
Calibrated Dual-Head Hurdle Model for drought score multi-step forecasting.

Architecture (v21 – Pure Continuous-Time Prediction, Gap-Gate Abolished)
--------------------------------------------------------------------------

  v21 Paradigm Shift: Gap-Gate & Gap Embedding Abolished
  --------------------------------------------------------
  Problem (v20): The Exponential Decay Gap-Gate was designed to handle a
    perceived "input-to-prediction gap".  This was a conceptual error: the
    Dataset Gap (train/test temporal discontinuity) is NOT the same as an
    input-to-prediction gap.  V21 predicts IMMEDIATELY after the last input
    week (gap = 0 at model design time), so the gate has no physical meaning
    and only adds noise and complexity.

  Solution (v21) – Clean Hurdle Architecture:
    Remove gap_lambda (nn.Parameter) entirely.
    Remove gap_embed (Linear 1→16) entirely.
    Forward signature simplifies to forward(x, target_time).
    branch_in = 256 + 128 + 32 + 2 = 418  (was 434; no more gap(16)).

Architecture Detail (v21: input_size=40, window=13)
----------------------------------------------------
  Input x (B, 13, 40)

  == LSTM Stream ==
    -> LayerNorm(40)                               (B, 13, 40)
    -> BiLSTM (hidden=128, layers=3, dropout=0.4)  (B, 13, 256)
    -> Temporal Attention (Linear(256->1) softmax over time dim=13)
       lstm_context = weighted sum                  (B, 256)
    -> Dropout(0.4)
    -> expand to                                   (B, 5, 256)

  == Dilated TCN Stream ==
    -> x.permute(0,2,1)                            (B, 40, 13)
    -> Conv1d(40->128, k=3, d=1, pad=1) + GELU     (B, 128, 13)
    -> Conv1d(128->128, k=3, d=2, pad=2) + GELU    (B, 128, 13)
    -> Conv1d(128->128, k=3, d=4, pad=4) + GELU    (B, 128, 13)
    -> AdaptiveAvgPool1d(1)                         (B, 128, 1)
    -> squeeze(-1)   tcn_context                   (B, 128)
    -> expand to                                   (B, 5, 128)

  == Horizon / Time Embeddings ==
    -> horizon_ids [0..4] -> Embedding(5,32) -> expand (B, 5, 32)
    -> target_time input                           (B, 5, 2)

  == Fusion ==
    -> cat([lstm(256), tcn(128), h_emb(32), tt(2)], dim=-1)
       encoded_state                               (B, 5, 418)
    NOTE: branch_in = 256+128+32+2 = 418 INVARIANT to input_size.

  == Branch A: Probability Logits Head ==
    Linear(418->128) -> GELU -> Dropout(0.2) -> Linear(128->1) -> squeeze(-1)
    -> (B, 5)  raw, unbounded logits (NO Sigmoid applied in model)

  == Branch B: Severity Regressor Head ==
    Linear(418->128) -> GELU -> Dropout(0.2) -> Linear(128->1) -> squeeze(-1)
    -> Softplus()
    -> (B, 5)  strictly non-negative severity magnitudes (>= 0.0)

  Forward Signature:
    forward(self, x, target_time) -> (logits_output, severity_output)
      logits_output   : (B, 5)  raw BCE logits  (no Sigmoid, no clamp)
      severity_output : (B, 5)  non-negative severity via Softplus

Changes from v20
----------------
  [v21] Exponential Decay Gap-Gate ABOLISHED:
        Removed gap_lambda nn.Parameter (was init=5.0).
        Removed G = exp(-|gap_lambda|*gap_size) multiplication on streams.
  [v21] Gap Embedding ABOLISHED:
        Removed gap_embed Linear(1, 16).
        gap_size parameter removed from forward() signature entirely.
  [v21] branch_in: 434 → 418  (256+128+32+2; gap(16) dropped)
  [v21] Forward signature: forward(x, target_time)  [no gap_size]

Retained from v20
-----------------
  [v20] Dual-Head Hurdle: head_prob (Branch A) + head_sev (Branch B)
        Forward returns (logits, severity) tuple
  [v20] Branch A: raw logits -> BCEWithLogitsLoss in train.py
  [v20] Branch B: Softplus -> non-negative severity (>= 0.0)
  [v19] input_size: 40 enriched features retained
  [v16] BiLSTM hidden_size: 128/dir → 256 effective
  [v16] Dilated TCN (d=1,2,4); receptive field = 29 weeks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DroughtLSTM(nn.Module):
    """
    Parameters
    ----------
    input_size  : number of features per time step (F = 40 in v21)
    hidden_size : LSTM hidden dimensionality per direction (default 128)
                  Effective LSTM output width = hidden_size * 2 = 256 (BiLSTM)
    num_layers  : number of stacked LSTM layers (default 3)
    dropout     : dropout probability (applied between LSTM layers and before head)
    horizon     : number of future weeks to forecast simultaneously
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,   # Effective output = hidden*2 = 256 (BiLSTM)
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
        lstm_out_size = hidden_size * 2  # 256

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
        # Dilated TCN Stream
        #
        # Input: (B, input_size, 13)
        # Symmetric padding preserves seq_len = 13:
        #   Dilation 1: padding = 1*(3-1)//2 = 1
        #   Dilation 2: padding = 2*(3-1)//2 = 2
        #   Dilation 4: padding = 4*(3-1)//2 = 4
        # Receptive field = 1 + 2*(3-1) + 4*(3-1) + 8*(3-1) = 29 weeks
        # Output: (B, 128, 13) -> AdaptiveAvgPool1d(1) -> (B, 128)
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
        # Horizon / Time Embeddings
        # ----------------------------------------------------------------

        # Learnable Horizon Embedding
        self.horizon_embed = nn.Embedding(num_embeddings=5, embedding_dim=32)

        # ----------------------------------------------------------------
        # Dual-Head Architecture  (v20/v21)
        # After concatenating:
        #   lstm_context (256) + tcn_context (128)
        #   + horizon_embed (32)
        #   + target_time (2)    [week_sin/cos of target weeks]
        # branch_in = 256 + 128 + 32 + 2 = 418
        # NOTE: branch_in does NOT depend on input_size or window length.
        # ----------------------------------------------------------------
        branch_in = lstm_out_size + 128 + 32 + 2  # = 418

        # Branch A – Probability Logits Head
        # Outputs raw, unbounded logits (B, 5). NO Sigmoid inside model.
        # Sigmoid is applied externally: BCEWithLogitsLoss in train.py.
        self.head_prob = nn.Sequential(
            nn.Linear(branch_in, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Branch B – Severity Regressor Head
        # Outputs strictly non-negative severity via Softplus (>= 0.0).
        self.head_sev = nn.Sequential(
            nn.Linear(branch_in, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        # Softplus guarantees severity >= 0 (weather-physical constraint)
        self.softplus = nn.Softplus(beta=1)

        # Shape debug flag (prints once on first forward pass)
        self._printed_shape = False

    # -----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        target_time: torch.Tensor,
    ):
        """
        Parameters
        ----------
        x           : (B, W, F)  -- input feature window
                       v21: W=13, F=40
        target_time : (B, 5, 2)  -- week_sin/cos of future target weeks

        Returns
        -------
        logits_output   : (B, 5)
            Raw, unbounded probability logits.
            NO Sigmoid applied — use nn.BCEWithLogitsLoss() externally.

        severity_output : (B, 5)
            Strictly non-negative severity via Softplus.
            No clamp is applied; absolute truncation is forbidden.
        """
        B = x.size(0)

        # ----------------------------------------------------------------
        # LSTM Stream
        # ----------------------------------------------------------------
        x_norm = self.input_norm(x)                            # (B, W, F)
        lstm_out, _ = self.lstm(x_norm)                        # (B, W, 256)

        # Temporal Attention
        attn_weights = self.attention(lstm_out)                # (B, W, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)      # softmax over time
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)   # (B, 256)

        lstm_context = self.dropout(context_vector)            # (B, 256)

        # Expand across horizon steps
        lstm_out_size = self.hidden_size * 2                   # 256
        lstm_expanded = lstm_context.unsqueeze(1).expand(B, self.horizon, lstm_out_size)
        # (B, 5, 256)

        # ----------------------------------------------------------------
        # Dilated TCN Stream
        # ----------------------------------------------------------------
        x_tcn = x.permute(0, 2, 1)                            # (B, F, W)
        tcn_feat = self.tcn_stream(x_tcn)                     # (B, 128, W)
        tcn_pooled = self.global_pool(tcn_feat)               # (B, 128, 1)
        tcn_context = tcn_pooled.squeeze(-1)                  # (B, 128)

        tcn_expanded = tcn_context.unsqueeze(1).expand(B, self.horizon, 128)
        # (B, 5, 128)

        # ----------------------------------------------------------------
        # Horizon Embedding
        # ----------------------------------------------------------------
        horizon_ids = torch.arange(self.horizon, device=x.device, dtype=torch.long)
        h_emb = self.horizon_embed(horizon_ids)               # (5, 32)
        h_emb = h_emb.unsqueeze(0).expand(B, self.horizon, 32)    # (B, 5, 32)

        # target_time is already (B, 5, 2)

        # ----------------------------------------------------------------
        # Feature Fusion
        # [lstm(256) + tcn(128) + h_emb(32) + target_time(2)] = 418
        # ----------------------------------------------------------------
        encoded_state = torch.cat(
            [lstm_expanded, tcn_expanded, h_emb, target_time],
            dim=-1,
        )
        # (B, 5, 418)

        # ----------------------------------------------------------------
        # Branch A: Probability Logits  (v20/v21 Dual-Head)
        # Raw unbounded logits — NO Sigmoid applied here.
        # BCEWithLogitsLoss in train.py applies numerically stable sigmoid.
        # ----------------------------------------------------------------
        logits_output = self.head_prob(encoded_state).squeeze(-1)   # (B, 5)

        # ----------------------------------------------------------------
        # Branch B: Severity Regressor  (v20/v21 Dual-Head)
        # Softplus applied for strict non-negativity (>= 0.0).
        # NO clamp, NO internal truncation.
        # ----------------------------------------------------------------
        sev_raw = self.head_sev(encoded_state).squeeze(-1)           # (B, 5)
        severity_output = self.softplus(sev_raw)                     # (B, 5), >= 0

        # Shape debug – print once on first forward call
        if not self._printed_shape:
            print(
                f"  [v21 Shape Debug] x: {tuple(x.shape)}  "
                f"| lstm_context: {tuple(lstm_context.shape)}  "
                f"| tcn_context: {tuple(tcn_context.shape)}  "
                f"| encoded_state: {tuple(encoded_state.shape)}  "
                f"| logits_output: {tuple(logits_output.shape)}  "
                f"| severity_output: {tuple(severity_output.shape)}"
            )
            self._printed_shape = True

        # Strict return: (logits, severity) tuple — no internal multiplication
        return logits_output, severity_output

    # -----------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self, input_size: int) -> str:
        lstm_out_size = self.hidden_size * 2    # 256
        branch_in     = lstm_out_size + 128 + 32 + 2   # 418
        lines = [
            "DroughtLSTM Architecture (v21 – Pure Continuous-Time Hurdle Model)",
            "=" * 90,
            f"  Input size   : {input_size}  (features per week; v21: 40 enriched features)",
            f"  Window size  : 13 weeks  (v16: capped at test-set horizon)",
            "",
            "  == LSTM Stream ==",
            f"  LayerNorm    : LayerNorm({input_size})",
            f"  BiLSTM       : hidden={self.hidden_size}/dir -> {lstm_out_size} effective  "
            f"(layers={self.num_layers}, bidirectional=True)",
            f"  Dropout      : {self.dropout.p}",
            f"  Attn         : Linear({lstm_out_size}->1); softmax over time; weighted sum",
            f"  lstm_context : (B, {lstm_out_size})",
            f"  expand       : (B, 5, {lstm_out_size})",
            "",
            "  == Dilated TCN Stream ==",
            f"  Transpose    : (B, 13, {input_size}) -> (B, {input_size}, 13)",
            f"  TCN Layer 1  : Conv1d({input_size}->128, k=3, d=1, pad=1) + GELU -> (B, 128, 13)",
            "  TCN Layer 2  : Conv1d(128->128, k=3, d=2, pad=2) + GELU -> (B, 128, 13)",
            "  TCN Layer 3  : Conv1d(128->128, k=3, d=4, pad=4) + GELU -> (B, 128, 13)",
            "  Receptive field: 1+2*(3-1)+4*(3-1)+8*(3-1) = 29 weeks",
            "  GlobalPool   : AdaptiveAvgPool1d(1) -> (B, 128, 1) -> squeeze -> (B, 128)",
            "  expand       : (B, 5, 128)",
            "",
            "  [v21] Gap-Gate ABOLISHED: No gap_lambda, no G multiplication.",
            "  [v21] Gap Embedding ABOLISHED: No gap_embed Linear(1,16).",
            "",
            "  == Embeddings ==",
            "  Learnable Horizon Embedding:",
            "     horizon_ids [0,1,2,3,4] -> Embedding(5,32) -> (B, 5, 32)",
            "  Target-Time Injection:",
            "     target_time input (B, 5, 2)  -- week_sin/cos of 5 future target weeks",
            "",
            "  == Feature Fusion ==",
            f"  Concat: [lstm({lstm_out_size}) + tcn(128) + h_emb(32) + tt(2)]",
            f"     -> encoded_state (B, 5, {branch_in})",
            "",
            "  == Branch A: Probability Logits Head ==",
            f"  Linear({branch_in}->128) -> GELU -> Dropout(0.2) -> Linear(128->1)",
            "  squeeze(-1) -> (B,5)  raw, unbounded logits",
            "  NO Sigmoid applied inside model (BCEWithLogitsLoss used externally)",
            "",
            "  == Branch B: Severity Regressor Head ==",
            f"  Linear({branch_in}->128) -> GELU -> Dropout(0.2) -> Linear(128->1)",
            "  squeeze(-1) -> (B,5)  -> Softplus() -> strictly non-negative severity",
            "  NO clamp; absolute truncation forbidden",
            "",
            "  == Forward Return Signature ==",
            "  forward(x, target_time) -> (logits_output, severity_output)",
            "  return logits_output (B,5), severity_output (B,5)",
            "  Two independent tensors; no internal prob*severity multiplication",
            "-" * 90,
            "  [v21] Gap-Gate (gap_lambda, G) ABOLISHED — no physical meaning at gap=0",
            "  [v21] Gap Embedding (gap_embed Linear(1,16)) ABOLISHED",
            "  [v21] branch_in: 434 → 418  (256+128+32+2; gap(16) dropped)",
            "  [v21] Forward: forward(x, target_time)  [gap_size param removed]",
            "  [v20] Dual-Head Hurdle: head_prob (Branch A) + head_sev (Branch B)",
            "  [v20] Branch A: raw logits -> BCEWithLogitsLoss in train.py",
            "  [v20] Branch B: Softplus -> non-negative severity (>= 0.0)",
            "  [v19] input_size: 40 enriched features retained",
            "  [v16] Dilated TCN (d=1,2,4): retained",
            f"  [v21] branch_in: 418  (256+128+32+2): UPDATED",
            "-" * 90,
            f"  Total params : {self.count_parameters():,}",
        ]
        return "\n".join(lines)
