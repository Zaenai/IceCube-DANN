"""
Domain-adversarial components for DANN training.

  GradientReversalFunction  – autograd function that negates gradients on backward
  GradientReversalLayer     – nn.Module wrapper; alpha controls reversal strength
  DomainClassifierHead      – MLP that predicts domain (MC vs real) from CLS embedding

Reference: Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation"
           https://arxiv.org/abs/1409.7495
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


# ──────────────────────────────────────────────
# Gradient reversal
# ──────────────────────────────────────────────

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL).

    During forward pass: identity.
    During backward pass: multiplies gradient by -alpha.

    Alpha is typically annealed from 0 → 1 over training following the
    schedule from Ganin & Lempitsky:

        alpha(p) = 2 / (1 + exp(-10 * p)) - 1,   p = training_progress ∈ [0, 1]

    Parameters
    ----------
    alpha : float
        Reversal strength. Set via .set_alpha() during training.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.register_buffer(
            "alpha",
            torch.tensor(alpha, dtype=torch.float32),
        )

    def set_alpha(self, alpha: float):
        self.alpha.fill_(alpha)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)



# ──────────────────────────────────────────────
# Domain classifier head
# ──────────────────────────────────────────────

class DomainClassifierHead(nn.Module):
    """
    Two-layer MLP that classifies MC (0) vs real (1) from a CLS embedding.

    Sits downstream of the GRL so its gradients are reversed before they
    reach the shared encoder — forcing the encoder to learn domain-invariant
    representations.

    Architecture:  Linear → LayerNorm → GELU → Dropout → Linear
    Loss:          BCEWithLogitsLoss (returns raw logits)

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input CLS embedding.
    hidden_dim : int
        Hidden layer width (default: embed_dim // 2).
    dropout : float
        Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(embed_dim // 2, 64)

        self.grl = GradientReversalLayer(alpha=1.0)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),   # single logit → BCEWithLogitsLoss
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_alpha(self, alpha: float) -> None:
        """Update GRL reversal strength."""
        self.grl.set_alpha(alpha)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embedding : (B, embed_dim)

        Returns
        -------
        logits : (B,)  — raw (un-sigmoided) domain logits
        """
        x = self.grl(embedding)
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────
# Alpha schedule
# ──────────────────────────────────────────────

def grl_alpha_schedule(current_step: int, total_steps: int, gamma: float = 10.0) -> float:
    """
    Annealing schedule for GRL alpha from Ganin & Lempitsky.

    p = current_step / total_steps ∈ [0, 1]
    alpha(p) = 2 / (1 + exp(-gamma * p)) - 1  ∈ [0, 1]

    Starts near 0 (encoder trains freely) and ramps to 1 (full reversal).

    Parameters
    ----------
    current_step : int
    total_steps  : int
    gamma : float
        Controls how quickly alpha ramps up (default: 10).

    Returns
    -------
    alpha : float ∈ [0, 1]
    """
    import math
    p = current_step / max(total_steps, 1)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


# Fix missing import in DomainClassifierHead
from typing import Optional
