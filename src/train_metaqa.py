"""
train_metaqa.py — Phase 1 training for the Qwen graph adapter.

Key design (corrected vs prior version):
  - TRAINING: bidirectional intersection with p_tgt from gold answer
  - EVAL: forward-only 2 hops + answer-type mask (HONEST metric, no gold target)
  - Adj matrices built with .coalesce() to fix sparse gradient issues
  - Answer-type mask inferred from question keywords → sharpens eval distribution
"""

import json
import re
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict

from src.qwen_adapter_model import QwenPointerAdapter


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER TYPE INFERENCE (for eval masking — never uses gold)
# ─────────────────────────────────────────────────────────────────────────────

def infer_answer_type(question: str) -> str:
    """
    Infer expected answer node type from question keywords.
    Used ONLY at eval to apply answer-type mask — not during training.
    """
    q = question.lower()

    # Genre / type
    if 'genre' in q:
        return 'genre'
    if 'type' in q and 'actor' not in q and 'director' not in q and 'writer' not in q:
        return 'genre'

    # Language
    if 'language' in q or 'spoken' in q:
        return 'language'

    # Year / release / when
    if 'year' in q or 'release' in q or q.strip().startswith('when'):
        return 'year'

    # Movie-output questions ("which films/movies", "same director", "same actor")
    if any(kw in q for kw in [
        'which films', 'which movies', 'what are the films', 'what are the movies',
        'same director', 'same actor', 'same screenwriter',
        'films that are directed', 'movies that are directed',
        'films that have the same', 'movies that have the same',
        'films share', 'movies share',
    ]):
        return 'movie'

    # Director
    if any(kw in q for kw in ['co-directed', 'who directed', 'listed as director',
                               'film co-directors', 'movie co-directors']):
        return 'director'
    if 'director' in q and 'same' not in q:
        return 'director'

    # Writer / screenwriter
    if any(kw in q for kw in ['co-wrote', 'screenwriter', 'who wrote', 'scriptwriter',
                               'listed as screenwriter', 'film co-writers', 'movie co-writers']):
        return 'writer'

    # Actor
    if any(kw in q for kw in ['co-star', 'who acted', 'who starred', 'appeared in',
                               'acted together', 'starred together', 'co-starred']):
        return 'actor'
    if 'actor' in q and 'same' not in q:
        return 'actor'

    return 'movie'  # safe default (largest set, conservative)


# ─────────────────────────────────────────────────────────────────────────────
# KB LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_kb(kb_path: str = "kb.txt", device: str = "cuda"):
    """
    Load MetaQA KB. Returns:
        node2idx, rel2idx, adj_matrices (2*R coalesced sparse [N,N]),
        type_masks (dict: answer_type → float tensor [N]),
        num_nodes, num_relations
    """
    nodes = set()
    relations = set()
    triples = []
    node_roles = defaultdict(set)

    with open(kb_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s, r, o = line.split('|')
            nodes.add(s)
            nodes.add(o)
            relations.add(r)
            triples.append((s, r, o))
            node_roles[s].add(f'subj:{r}')
            node_roles[o].add(f'obj:{r}')

    node2idx = {n: i for i, n in enumerate(sorted(nodes))}
    rel2idx  = {r: i for i, r in enumerate(sorted(relations))}
    num_nodes     = len(node2idx)
    num_relations = len(rel2idx)

    print(f"  KB: {num_nodes} nodes, {num_relations} relations, {len(triples)} triples")

    # ── Node type classification for answer-type masks ──
    node_types = {}
    for node, roles in node_roles.items():
        subj_rels = {r.split(':')[1] for r in roles if r.startswith('subj:')}
        obj_rels  = {r.split(':')[1] for r in roles if r.startswith('obj:')}
        if subj_rels & {'directed_by', 'has_genre', 'starred_actors', 'written_by',
                        'has_tags', 'in_language', 'release_year', 'has_imdb_rating',
                        'has_imdb_votes'}:
            node_types[node] = 'movie'
        elif 'directed_by' in obj_rels:
            node_types[node] = 'director'
        elif 'has_genre' in obj_rels:
            node_types[node] = 'genre'
        elif 'starred_actors' in obj_rels:
            node_types[node] = 'actor'
        elif 'written_by' in obj_rels:
            node_types[node] = 'writer'
        elif 'in_language' in obj_rels:
            node_types[node] = 'language'
        elif 'release_year' in obj_rels:
            node_types[node] = 'year'
        elif 'has_tags' in obj_rels:
            node_types[node] = 'tag'
        else:
            node_types[node] = 'movie'

    type_to_mask = defaultdict(lambda: torch.zeros(num_nodes, dtype=torch.float32))
    for node, ntype in node_types.items():
        if node in node2idx:
            type_to_mask[ntype][node2idx[node]] = 1.0
    type_masks = {t: mask.to(device) for t, mask in type_to_mask.items()}

    for t, m in sorted(type_masks.items()):
        print(f"    Type mask '{t}': {int(m.sum().item())} nodes")

    # ── Per-relation adjacency matrices (sparse COO, coalesced, row-normalised) ──
    def make_adj(rows, cols, size):
        if not rows:
            idx  = torch.zeros(2, 1, dtype=torch.long)
            vals = torch.zeros(1, dtype=torch.float32)
        else:
            idx  = torch.tensor([rows, cols], dtype=torch.long)
            vals = torch.ones(len(rows), dtype=torch.float32)
        adj = torch.sparse_coo_tensor(idx, vals, (size, size)).coalesce()
        # Row-normalise
        row_sum = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
        vals_norm = adj.values() / row_sum[adj.indices()[0]]
        adj_norm = torch.sparse_coo_tensor(
            adj.indices(), vals_norm, adj.size(), dtype=torch.float32
        ).coalesce()
        return adj_norm.to(device)

    adj_matrices = []
    for r_idx in range(num_relations):
        fwd_r, fwd_c = [], []
        bwd_r, bwd_c = [], []
        for s, r, o in triples:
            if rel2idx[r] == r_idx:
                si, oi = node2idx[s], node2idx[o]
                fwd_r.append(si); fwd_c.append(oi)
                bwd_r.append(oi); bwd_c.append(si)
        adj_matrices.append(make_adj(fwd_r, fwd_c, num_nodes))
        adj_matrices.append(make_adj(bwd_r, bwd_c, num_nodes))

    return node2idx, rel2idx, adj_matrices, type_masks, num_nodes, num_relations


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_train_data(node2idx: dict, qa_path: str, limit: int = 2000):
    dataset = []
    with open(qa_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q, ans_str = line.split('\t')
            answers = ans_str.split('|')
            start, end = q.find('['), q.find(']')
            if start == -1 or end == -1:
                continue
            ent = q[start+1:end]
            q_clean = q.replace('[', '').replace(']', '')
            if ent in node2idx and all(a in node2idx for a in answers):
                dataset.append({
                    "question": q_clean,
                    "head_entity": ent,
                    "answers": answers,
                })
            if len(dataset) >= limit:
                break
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_phase_1(hop_count: int = 2, ckpt_dir: str = "."):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading KB...")
    node2idx, rel2idx, adj_matrices, type_masks, num_nodes, num_relations = load_kb(device=device)

    print(f"Loading train dataset ({hop_count}-hop true QA)...")
    dataset = load_train_data(node2idx, qa_path=f"{hop_count}-hop/vanilla/qa_train.txt", limit=2000)
    print(f"  Loaded {len(dataset)} training examples.")
    
    print(f"Loading dev dataset ({hop_count}-hop true QA)...")
    dev_dataset = load_train_data(node2idx, qa_path=f"{hop_count}-hop/vanilla/qa_dev.txt", limit=500)
    print(f"  Loaded {len(dev_dataset)} dev examples.")

    qwen_name = "Qwen/Qwen1.5-1.8B"
    model = QwenPointerAdapter(
        qwen_model_name=qwen_name,
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=256,
    )
    model.graph_executor.to(device)
    model.qwen_to_graph.to(device)
    if hasattr(model, 'copy_head'):
        model.copy_head.to(device)

    tokenizer = AutoTokenizer.from_pretrained(qwen_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable_params) > 0, "nothing to train!"

    optimizer = AdamW(trainable_params, lr=3e-4)

    max_epochs = 10
    best_dev_hr = -1.0
    patience_counter = 0
    patience_limit = 3
    
    import os
    ckpt_name = os.path.join(ckpt_dir, f'phase1_checkpoint_{hop_count}hop.pt')

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        train_hits = 0
        model.train()

        for i, item in enumerate(tqdm(dataset)):
            q = item["question"]
            head = item["head_entity"]
            answers = item["answers"]
            
            src_idx = node2idx.get(head, -1)
            if src_idx == -1: continue

            target_indices = set([node2idx.get(a, -1) for a in answers])
            if not target_indices: continue

            answer_type = infer_answer_type(q)
            prompt = f"Q: {q}\nA: "

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                qwen_out = model.qwen(**inputs, output_hidden_states=True)
                h_last = qwen_out.hidden_states[-1][:, -1, :].float()

            p_src = torch.zeros(1, num_nodes, device=device)
            p_src[0, src_idx] = 1.0

            p_tgt = torch.zeros(1, num_nodes, device=device)
            for tidx in target_indices:
                if tidx != -1: p_tgt[0, tidx] = 1.0
            if p_tgt.sum() == 0: continue
            p_tgt = p_tgt / p_tgt.sum()

            optimizer.zero_grad()
            graph_query = model.qwen_to_graph(h_last)

            p_final_train, entropy = model.graph_executor(
                query=graph_query,
                p_src=p_src,
                adj_matrices=adj_matrices,
                p_tgt=p_tgt,
                is_eval=False,
                max_hops=hop_count,
            )

            target_list = [t for t in target_indices if t != -1]
            target_prob = p_final_train[0, target_list].sum()
            loss = -torch.log(target_prob + 1e-10)

            loss.backward()
            
            total_norm = torch.nn.utils.clip_grad_norm_(
                list(model.graph_executor.parameters()) + list(model.qwen_to_graph.parameters()), 
                max_norm=1.0
            )

            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                p_eval, _ = model.graph_executor(
                    query=graph_query.detach(),
                    p_src=p_src,
                    adj_matrices=adj_matrices,
                    p_tgt=None,
                    is_eval=True,
                    max_hops=hop_count,
                )
                mask = type_masks.get(answer_type, torch.ones(num_nodes, device=device))
                p_masked = p_eval[0] * mask
                pred = p_masked.argmax().item()
                if pred in target_indices:
                    train_hits += 1

            if i > 0 and i % 100 == 0:
                tqdm.write(f"  Step {i:4d} | loss {loss.item():.4f} | forward-only HR@1 {train_hits/(i+1):.2%} (type={answer_type})")

        avg_loss = epoch_loss / len(dataset)
        hr = train_hits / len(dataset)
        print(f"\nEpoch {epoch} | Avg Loss: {avg_loss:.4f} | Train HR@1: {hr:.2%}")
        
        # ── EVALUATE ON DEV SET ──
        model.eval()
        dev_hits = 0
        for item in dev_dataset:
            q = item["question"]
            head = item["head_entity"]
            answers = item["answers"]
            
            src_idx = node2idx.get(head, -1)
            if src_idx == -1: continue

            target_indices = set([node2idx.get(a, -1) for a in answers])
            if not target_indices: continue

            answer_type = infer_answer_type(q)
            prompt = f"Q: {q}\nA: "
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                qwen_out = model.qwen(**inputs, output_hidden_states=True)
                h_last = qwen_out.hidden_states[-1][:, -1, :].float()
                
                p_src = torch.zeros(1, num_nodes, device=device)
                p_src[0, src_idx] = 1.0
                
                graph_query = model.qwen_to_graph(h_last)
                p_eval, _ = model.graph_executor(
                    query=graph_query,
                    p_src=p_src,
                    adj_matrices=adj_matrices,
                    p_tgt=None,
                    is_eval=True,
                    max_hops=hop_count,
                )
                
                mask = type_masks.get(answer_type, torch.ones(num_nodes, device=device))
                p_masked = p_eval[0] * mask
                pred = p_masked.argmax().item()
                if pred in target_indices:
                    dev_hits += 1
                    
        dev_hr = dev_hits / len(dev_dataset)
        print(f"Epoch {epoch} | Dev HR@1: {dev_hr:.2%}")
        
        if dev_hr > best_dev_hr:
            best_dev_hr = dev_hr
            patience_counter = 0
            torch.save({
                'graph_executor': model.graph_executor.state_dict(),
                'qwen_to_graph':  model.qwen_to_graph.state_dict(),
            }, ckpt_name)
            print(f"  [New best dev HR, saving checkpoint {ckpt_name}]")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered. Best Dev HR: {best_dev_hr:.2%}")
                break

    return best_dev_hr

if __name__ == "__main__":
    import sys
    hop = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    train_phase_1(hop_count=hop)
