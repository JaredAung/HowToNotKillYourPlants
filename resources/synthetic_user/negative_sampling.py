"""
Hard negative mining for BPR (option 1: sample — score — pick).

Sample ``S`` negative plant IDs uniformly from the eligible pool, score each with the
current two-tower model under ``torch.no_grad()``, then take the **top_h** highest scores
(hardest negatives). Loss averages ``softplus(s_neg - s_pos)`` over those ``top_h`` pairs.
Gradients flow only through the chosen negatives on the second forward pass.
"""
from __future__ import annotations

import random
from typing import Sequence

import torch

# Pool size S: uniform candidates, then take top_h by model score among them.
HARD_NEGATIVE_POOL_SIZE = 64
TOP_H_HARD_NEGATIVES = 5


def sample_uniform_negative_ids(
    pool: Sequence[int],
    s: int,
    rng: random.Random | None = None,
) -> list[int]:
    """
    Sample ``s`` negative IDs uniformly from ``pool``.

    If ``len(pool) >= s``: sample without replacement.
    If ``len(pool) < s``: sample **with replacement** so the list length is always ``s``
    (when ``pool`` is non-empty).
    """
    r = rng or random
    pl = list(pool)
    if not pl:
        return []
    if len(pl) >= s:
        return r.sample(pl, s)
    return r.choices(pl, k=s)


def pick_hardest_negative_indices(scores: torch.Tensor) -> torch.Tensor:
    """``scores`` (B, S) -> argmax index per row (B,) in [0, S-1]."""
    return scores.argmax(dim=1)


def pick_top_h_hard_negative_plant_embeddings(
    model: torch.nn.Module,
    user_vec: torch.Tensor,
    plant_neg_cands: torch.Tensor,
    top_h: int = TOP_H_HARD_NEGATIVES,
) -> torch.Tensor:
    """
    Score each of ``S`` candidate negatives with ``model`` under ``torch.no_grad()``,
    return the plant embeddings for the **top_h** largest scores: shape ``(B, top_h, 21)``.

    ``top_h`` is capped at ``S`` (number of candidate columns).
    """
    b, s, d = plant_neg_cands.shape
    device = user_vec.device
    u_flat = user_vec.unsqueeze(1).expand(b, s, -1).reshape(b * s, -1)
    p_flat = plant_neg_cands.reshape(b * s, -1)
    with torch.no_grad():
        scores_cand = model(u_flat, p_flat).reshape(b, s)
    h = min(int(top_h), s)
    _, idx = torch.topk(scores_cand, k=h, dim=1, largest=True, sorted=True)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, d)
    return torch.gather(plant_neg_cands, 1, idx_exp)
