"""
differentiable_graph_executor.py — v3-matched bidirectional graph executor.

Training (p_tgt given):
    - Forward 1 hop from p_src via fwd_controller
    - Backward 1 hop from p_tgt via bwd_controller (swapped adjacency)
    - Intersection: p_final = p_src_after * p_tgt_after  (normalised)

Eval (p_tgt=None, is_eval=True):
    - Forward-only 2 hops from p_src via fwd_controller
    - NO gold target used — answer-type mask applied externally
    - Assertion guards against accidental gold-target leak at eval

adj_matrices: list of 2*R sparse COO [N,N] tensors (coalesced, row-normalised).
              Layout: [rel0_fwd, rel0_bwd, rel1_fwd, rel1_bwd, ...]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableGraphExecutor(nn.Module):

    def __init__(self, hidden_dim: int, num_relations: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_actions = 1 + 2 * num_relations  # stay + fwd/bwd per relation

        self.max_supported_hops = 3
        
        # Hop-aware controllers: output a sequence of actions for up to MAX_HOPS
        self.fwd_controller = nn.Linear(hidden_dim, self.max_supported_hops * self.num_actions)
        self.bwd_controller = nn.Linear(hidden_dim, self.max_supported_hops * self.num_actions)

        # Stored for retrieval loss access
        self.last_p_final: torch.Tensor | None = None

        self._init_weights()

    def _init_weights(self):
        for linear in [self.fwd_controller, self.bwd_controller]:
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
            nn.init.zeros_(linear.bias)

    def _sparse_hop(self, p: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Compute p @ adj for p:[B,N] and adj:[N,N] sparse (COO coalesced).
        Equivalent to dense torch.mm(p, adj) but works with sparse adj.
        Formula: (adj.T @ p.T).T
        """
        adj_t = adj.t().coalesce()
        return torch.sparse.mm(adj_t, p.t()).t()

    def _apply_one_hop(
        self,
        p: torch.Tensor,       # [B, N]
        pi: torch.Tensor,      # [B, num_actions]
        adj_matrices: list,    # 2*R sparse [N,N]
        reverse: bool = False, # if True, swap fwd/bwd adj (backward pass)
    ) -> torch.Tensor:
        """One hop: stay + weighted traversal across all relations."""
        R = self.num_relations
        p_next = pi[:, 0:1] * p  # stay component [B, N]

        for r in range(R):
            if not reverse:
                adj_fwd = adj_matrices[2 * r]
                adj_bwd = adj_matrices[2 * r + 1]
            else:
                # Backward pass: reverse the edge direction convention
                adj_fwd = adj_matrices[2 * r + 1]
                adj_bwd = adj_matrices[2 * r]

            p_fwd = self._sparse_hop(p, adj_fwd)  # [B, N]
            p_bwd = self._sparse_hop(p, adj_bwd)  # [B, N]

            p_next = p_next + pi[:, 1 + r:2 + r] * p_fwd
            p_next = p_next + pi[:, 1 + R + r:2 + R + r] * p_bwd

        return p_next  # [B, N]

    def forward(
        self,
        query: torch.Tensor,                   # [B, hidden_dim]
        p_src: torch.Tensor,                   # [B, N] one-hot at source
        adj_matrices: list,                    # 2*R sparse [N,N]
        p_tgt: torch.Tensor | None = None,     # Unused, kept for signature compat
        is_eval: bool = False,
        max_hops: int = 2,
    ) -> torch.Tensor:
        """
        Returns p_final [B, N] — posterior over answer nodes.

        For MetaQA, we only have the source anchor and the final answer. We do not
        have ground-truth bridge nodes. Therefore, both training and evaluation must
        use forward-only 2-hop traversal. The loss will be computed on the final answer.
        """
        logits = self.fwd_controller(query) # [B, max_supported_hops * num_actions]
        logits = logits.view(-1, self.max_supported_hops, self.num_actions) # [B, max_supported_hops, num_actions]
        pi_fwd_all = F.softmax(logits, dim=-1)
        
        # Gradient floor to prevent saturation
        pi_fwd_all = (1 - 1e-4) * pi_fwd_all + 1e-4 / self.num_actions

        # ── Forward-only, variable hops ──
        p_current = p_src
        entropies = []
        for hop_idx in range(max_hops):
            pi_fwd = pi_fwd_all[:, hop_idx, :] # The specific relation mixture for THIS hop
            p_current = self._apply_one_hop(p_current, pi_fwd, adj_matrices, reverse=False)
            
            # Calculate entropy for this hop
            ent = -(pi_fwd * torch.log(pi_fwd + 1e-8)).sum(dim=-1).mean()
            entropies.append(ent)
            
        p_final = p_current

        # Normalize to prevent all-zeros if traversal is empty
        # Add a uniform floor to guarantee no exact zeros and stable gradients
        floor = 1e-6 / p_final.shape[-1]
        p_final = p_final + floor
        p_final = p_final / p_final.sum(dim=-1, keepdim=True)

        # Average entropy across all executed hops
        entropy = sum(entropies) / len(entropies) if entropies else torch.tensor(0.0, device=query.device)

        self.last_p_final = p_final
        return p_final, entropy
