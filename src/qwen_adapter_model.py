"""
qwen_adapter_model.py — Qwen-1.5B + v3-matched DifferentiableGraphExecutor.

Key fixes vs prior version:
  - device_map="cuda:0" (not "auto") → no cross-device split on Modal A10G
  - p_src always built from src_node_indices
  - p_tgt built ONLY in training_mode — NEVER fed at eval (answer leak prevention)
  - graph_executor called directly with p_src / p_tgt (no max_hops loop)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from src.differentiable_graph_executor import DifferentiableGraphExecutor


class QwenPointerAdapter(nn.Module):

    def __init__(
        self,
        qwen_model_name: str,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_nodes = num_nodes

        # 1. Frozen Qwen — cuda:0 for single-GPU Modal (no accelerate splitting)
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            device_map="cuda:0",
            dtype=torch.float16,
        )
        for param in self.qwen.parameters():
            param.requires_grad = False

        qwen_hidden_dim = self.qwen.config.hidden_size

        # 2. Graph executor (v3-matched bidirectional design)
        self.graph_executor = DifferentiableGraphExecutor(
            hidden_dim=hidden_dim,
            num_relations=num_relations,
        )

        # 3. Projection: Qwen hidden → graph query space
        self.dropout = nn.Dropout(p=0.1)
        self.qwen_to_graph = nn.Linear(qwen_hidden_dim, hidden_dim, dtype=torch.float32)

        # 4. Pointer / copy gate
        self.copy_head = nn.Linear(qwen_hidden_dim, 1, dtype=torch.float32)

    def _get_last_prompt_hidden(self, input_ids, attention_mask):
        """Run frozen Qwen, return last-token hidden state and logits."""
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last layer hidden, last token position
        h = outputs.hidden_states[-1][:, -1, :].float()  # [B, qwen_dim]
        logits = outputs.logits.float()                   # [B, T, vocab]
        return h, logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adj_matrices: list,          # 2*R sparse [N,N] adj matrices
        src_node_indices: torch.Tensor,       # [B]
        tgt_node_indices: torch.Tensor | None = None, # [B]
        training_mode: bool = True,
        max_hops: int = 2,
    ):
        """
        Returns:
            p_final [B, N]   — graph posterior over answer nodes
            p_copy  [B, 1]   — interpolation gate
            logits  [B, T, V] — Qwen token logits
        """
        device = input_ids.device
        B = input_ids.shape[0]
        N = self.num_nodes

        # 1. Qwen forward (frozen)
        with torch.no_grad():
            h_last, logits = self._get_last_prompt_hidden(input_ids, attention_mask)

        # 2. Build p_src (always — source anchor is always known)
        src_valid = (src_node_indices >= 0).float().unsqueeze(1)  # [B, 1]
        p_src = F.one_hot(src_node_indices.clamp(min=0), num_classes=N).float() * src_valid
        # Move to same device as h_last (qwen may be on cuda:0 already)
        p_src = p_src.to(h_last.device)

        # 3. Build p_tgt ONLY in training_mode (NEVER at eval — answer leak prevention)
        p_tgt = None
        if training_mode and tgt_node_indices is not None:
            tgt_valid = (tgt_node_indices >= 0).float().unsqueeze(1)
            p_tgt = F.one_hot(tgt_node_indices.clamp(min=0), num_classes=N).float() * tgt_valid
            p_tgt = p_tgt.to(h_last.device)

        # 4. Graph execution
        h_dropped = self.dropout(h_last)
        graph_query = self.qwen_to_graph(h_dropped)  # [B, hidden_dim]
        p_final, entropy = self.graph_executor(
            query=graph_query,
            p_src=p_src,
            adj_matrices=adj_matrices,
            p_tgt=p_tgt,
            is_eval=not training_mode,
            max_hops=max_hops,
        )  # [B, N]

        # 5. Copy gate
        p_copy = torch.sigmoid(self.copy_head(h_last))  # [B, 1]

        return p_final, p_copy, logits, entropy
