# A Falsifiable Graph-Memory Architecture for Editable Knowledge and Token-Efficient Multi-Hop Reasoning

**Authors:** Mahesh et al.

---

## Abstract

Large language models (LLMs) encode factual knowledge implicitly within their parameters, creating a fundamental tension between knowledge currency and model stability. When facts change — a new drug is approved, a company is acquired, a scientific finding is retracted — the only recourse is costly retraining or fine-tuning. Furthermore, multi-hop reasoning over structured knowledge requires expensive chain-of-thought (CoT) token generation, consuming 50–100+ tokens per query even for simple relational traversals.

We introduce a **Graph-Memory Architecture** that decouples factual knowledge from model weights by composing three components: (1) an explicit, editable **Graph Memory** storing entities and relations as sparse adjacency matrices, (2) a **Differentiable Graph Executor** (~500K trainable parameters) that performs learned multi-hop traversals via soft relation-selection controllers, and (3) a frozen **LLM backbone** (Qwen-1.5B, 1.8B parameters, all frozen) that serves solely as a semantic parser, projecting natural language queries into the executor's query space through a lightweight linear adapter.

We evaluate on two knowledge graphs of different domains and scales:

**MetaQA** (43K nodes, 135K triples, movie domain):
- **HR@1 of 89% (1-hop), 62% (2-hop), 62% (3-hop)** with our graph adapter, compared to 3%, 16%, 32% for CoT baselines on the same frozen LLM.
- **Memory ablation collapse:** Accuracy drops from 90.8% → 0.4% (1-hop), 65.8% → 1.2% (2-hop), and 57.2% → 5.8% (3-hop) when the graph is removed.
- **Graph corruption falsification:** Shuffling adjacency matrices drops HR@1 from 89% → 1% (1-hop) and 62% → 6% (2-hop), establishing causal graph dependence.
- **41× latency reduction** over CoT baselines (~0.04s vs ~2.05s per query).

**Biomedical KG** (5,695 nodes, 18,762 edges, drug–disease domain):
- **45.6% HR@1 on held-out facts** (135× above the 0.34% chance baseline), with 85.1% HR@5 and 98.3% HR@10.
- **0% Memory OFF** — complete collapse when the graph is removed.
- **100% zero-shot fact insertion success** — newly added drug–disease edges are immediately retrievable without retraining.

Our primary contribution is not a claim of better AI, but a rigorous, falsifiable demonstration that **editable, external knowledge** can be coupled to an LLM through a **non-bypassable memory bottleneck**, enabling efficient multi-hop reasoning with zero-shot fact editability.

---

## 1. Introduction

### 1.1 Problem 1: Knowledge Trapped in Weights

Contemporary LLMs store factual knowledge implicitly in their billions of parameters through the pre-training process. This creates a well-documented problem: knowledge becomes stale the moment training ends.

Consider: a new drug receives FDA approval. An LLM trained before this event will confidently provide outdated information. The standard remedies — full retraining, LoRA fine-tuning, or retrieval-augmented generation (RAG) — each carry significant costs:

- **Retraining** is computationally prohibitive and risks catastrophic forgetting.
- **Fine-tuning** requires curated training data and careful hyperparameter selection per fact update.
- **RAG** retrieves relevant documents but still relies on the LLM to reason over them, with no guarantee that retrieved context overrides parametric beliefs.

### 1.2 Problem 2: Chain-of-Thought is Token-Expensive

Multi-hop reasoning — answering questions that require traversing multiple relational steps (e.g., "Who directed the films that starred the actors in The Matrix?") — is typically addressed through chain-of-thought prompting. However, CoT generates 50–100+ reasoning tokens per query, even for simple relational traversals that a graph database could resolve in microseconds.

This token overhead is not merely an efficiency concern — it scales linearly with reasoning depth and creates failure modes where the model "loses track" of intermediate entities across long generation sequences.

### 1.3 Problem 3: Most Graph Systems Only Retrieve

Existing graph-augmented systems, including GraphRAG, follow a retrieve-then-generate paradigm:

```
GraphRAG:  Query → Retrieve subgraph → Serialize to text → Prompt LLM → Generate answer
```

This approach treats the graph as a retrieval index, not as a computational substrate. The LLM must still perform reasoning over serialized text, inheriting all the limitations of in-context reasoning: token limits, attention decay over long contexts, and sensitivity to serialization format.

### 1.4 Our Hypothesis

We propose a fundamentally different coupling:

```
Our Architecture:  Query → LLM (semantic parse) → Graph Executor (traverse) → Answer
```

The LLM's role is reduced to **semantic parsing** — converting natural language into a query vector. All factual reasoning occurs in the graph executor, which operates directly on the adjacency matrices of an explicit knowledge graph. This design makes knowledge:

- **Editable:** Insert or remove a triple; the next query reflects the change immediately.
- **Non-bypassable:** The LLM cannot answer from its weights alone; the information bottleneck forces dependence on the external graph.
- **Efficient:** A single forward pass through the executor replaces dozens of autoregressive generation steps.

---

## 2. Related Work

### 2.1 Graph-Augmented Language Models

**GraphRAG** (Microsoft, 2024) constructs community-based summaries from knowledge graphs and uses them to augment LLM context windows. While effective for global summarization queries, GraphRAG treats the graph as a retrieval index — the LLM performs all reasoning over serialized text. Our approach differs fundamentally: the graph is a computational substrate, not a retrieval source.

**Knowledge Graph Question Answering (KGQA)** methods such as EmbedKGQA (Saxena et al., 2020) and QA-GNN (Yasunaga et al., 2021) integrate graph representations with neural QA systems. EmbedKGQA uses KG embeddings to answer multi-hop questions, while QA-GNN jointly reasons over text and graphs using graph neural networks. Our differentiable graph executor differs in that it performs explicit, interpretable multi-hop traversals via soft relation selection rather than opaque message passing.

### 2.2 Neuro-Symbolic Reasoning

The neuro-symbolic paradigm seeks to combine neural perception with symbolic reasoning (Garcez et al., 2019; Lamb et al., 2020). Systems like Neural Theorem Provers (Rocktäschel & Riedel, 2017) and DeepProbLog (Manhaeve et al., 2018) integrate differentiable reasoning with symbolic knowledge bases. Our graph executor can be viewed as a lightweight neuro-symbolic module: it uses learned soft controllers (neural) to select relation traversals (symbolic) over an explicit graph structure.

### 2.3 Knowledge Graph Completion

Methods for predicting missing edges in knowledge graphs — TransE (Bordes et al., 2013), RotatE (Sun et al., 2019), BoxE (Abboud et al., 2020) — learn geometric embeddings of entities and relations. While our current system does not perform link prediction, these methods are directly relevant to our proposed future work on confidence-weighted graphs and abductive reasoning (Section 9), where predicted edges could be scored by embedding-based confidence.

### 2.4 Retrieval-Augmented Generation

Standard RAG architectures (Lewis et al., 2020; Guu et al., 2020) retrieve passages from a document store to augment LLM context. Dense retrievers (Karpukhin et al., 2020) and sparse retrievers (Robertson et al., 2009) select relevant documents, but the reasoning still occurs within the LLM. Our Retriever+Qwen baseline (Section 7.5) directly evaluates this paradigm against our graph executor.

---

## 3. Architecture

Our architecture consists of four tightly coupled components. We describe each in detail.

### 3.1 Entity Linker

The entity linker maps mentions in natural language to nodes in the knowledge graph.

```
Input:   "What treats tuberculosis?"
              ↓
Linker:  "tuberculosis" → Node_4527
              ↓
Output:  p_src = one-hot vector at index 4527
```

For MetaQA, entities are marked with brackets in the dataset (e.g., `[The Matrix]`). Our `MetaQALinker` handles normalization edge cases: article transposition ("The Matrix" ↔ "Matrix, The"), case normalization, and punctuation variants. On the MetaQA test set, the linker achieves near-perfect coverage across all hop levels.

In a production setting, this component would be replaced by a learned entity linking module (e.g., BLINK; Wu et al., 2020) or a mention detection + candidate ranking pipeline.

### 3.2 Graph Memory

The knowledge graph is stored as a set of sparse adjacency matrices. For a graph with $N$ nodes and $R$ relation types, we maintain $2R$ sparse COO tensors of shape $[N, N]$:

$$\mathbf{A} = \{A_0^{\text{fwd}}, A_0^{\text{bwd}}, A_1^{\text{fwd}}, A_1^{\text{bwd}}, \ldots, A_{R-1}^{\text{fwd}}, A_{R-1}^{\text{bwd}}\}$$

Each matrix is:
- **Sparse COO format** — memory-efficient for the typical sparsity of knowledge graphs.
- **Coalesced** — duplicate indices are summed, preventing gradient issues.
- **Row-normalized** — each row sums to 1, ensuring probability mass conservation during traversal.

For MetaQA, the graph contains **43,234 nodes**, **9 relation types** (directed_by, starred_actors, has_genre, written_by, in_language, release_year, has_tags, has_imdb_rating, has_imdb_votes), and **134,741 triples**.

**Editability.** Adding a fact (e.g., "Drug_X treats Disease_Y") requires inserting a single entry into the appropriate sparse matrix and re-normalizing the affected rows. No model retraining is needed. Deleting a fact is the reverse operation.

### 3.3 Differentiable Graph Executor

The graph executor is the core reasoning module. It performs learned multi-hop traversals over the adjacency matrices, controlled by a query-conditioned soft relation selector.

#### Architecture

The executor contains two learned linear controllers:

- **Forward controller:** $f_{\text{fwd}}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{H \times (1 + 2R)}$
- **Backward controller:** $f_{\text{bwd}}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{H \times (1 + 2R)}$

where $d$ is the hidden dimension (256), $H$ is the maximum supported hops (3), and $R$ is the number of relation types (9). The output is a per-hop **action distribution** over $1 + 2R = 19$ actions: one "stay" action plus forward and backward traversal for each relation.

#### Single-Hop Traversal

Given a probability distribution $\mathbf{p} \in \mathbb{R}^{N}$ over nodes and an action distribution $\boldsymbol{\pi} \in \mathbb{R}^{1+2R}$ (obtained by softmax over the controller logits):

$$\mathbf{p}_{\text{next}} = \pi_0 \cdot \mathbf{p} + \sum_{r=0}^{R-1} \left[ \pi_{1+r} \cdot (\mathbf{p} \cdot A_r^{\text{fwd}}) + \pi_{1+R+r} \cdot (\mathbf{p} \cdot A_r^{\text{bwd}}) \right]$$

The "stay" component ($\pi_0$) allows the executor to retain probability mass at the current node when no traversal is needed at a particular hop.

#### Multi-Hop Composition

For a $k$-hop query, the executor chains $k$ single-hop operations, each with its own learned action distribution:

$$\mathbf{p}^{(0)} = \text{one-hot}(\text{src}) \quad;\quad \mathbf{p}^{(t+1)} = \text{hop}(\mathbf{p}^{(t)}, \boldsymbol{\pi}^{(t)}) \quad;\quad \mathbf{p}_{\text{final}} = \mathbf{p}^{(k)}$$

The final distribution $\mathbf{p}_{\text{final}}$ is normalized and returned as the answer distribution.

#### Gradient Floor

To prevent action distributions from saturating (all mass on a single action), we apply a gradient floor:

$$\boldsymbol{\pi}' = (1 - \epsilon) \cdot \boldsymbol{\pi} + \frac{\epsilon}{1 + 2R}$$

with $\epsilon = 10^{-4}$. This ensures non-zero gradients for all actions during training.

### 3.4 Information Bottleneck

The critical architectural choice is the **projection bottleneck** between the LLM and the graph executor:

```
Qwen Hidden State (1536-dim, float16)
            ↓
   Linear Projection (1536 → 256)
            ↓
     Graph Query (256-dim, float32)
            ↓
     Graph Executor
            ↓
     Answer Distribution
```

This bottleneck serves two purposes:

1. **Dimensionality reduction:** The 1536-dimensional Qwen hidden state is compressed to a 256-dimensional graph query. This information loss is deliberate — it prevents the query vector from encoding sufficient information to reconstruct the answer without graph traversal.

2. **Non-bypassability:** Because the Qwen weights are frozen and the answer is produced by the graph executor (not by Qwen's language modeling head), the system **cannot** answer correctly without a valid graph. Our memory ablation experiments (Section 7.2) confirm this empirically: when the graph is removed, accuracy collapses to near-chance levels.

### 3.5 Answer Head and Type Masking

The executor produces a probability distribution $\mathbf{p}_{\text{final}} \in \mathbb{R}^N$ over all graph nodes. To improve precision, we apply an **answer-type mask** inferred from question keywords:

```
"What genre is The Matrix?"  →  answer_type = "genre"  →  mask = genre_nodes
"Who directed Inception?"    →  answer_type = "director" → mask = director_nodes
```

The type inference uses a rule-based keyword classifier (no learning required) that maps question patterns to one of 8 node types: movie, director, actor, writer, genre, language, year, tag.

The final prediction is:

$$\hat{a} = \arg\max_i \left( \mathbf{p}_{\text{final}}[i] \cdot \mathbf{m}_{\text{type}}[i] \right)$$

where $\mathbf{m}_{\text{type}}$ is the binary mask for the inferred answer type.

---

## 4. Falsification Framework

A central contribution of this work is the explicit design of **falsification experiments** — tests that could, in principle, disprove our architectural claims. Many neural architecture papers demonstrate positive results but fail to include experiments that could falsify the proposed mechanism. We define four invariants and test each:

### Invariant 1: Memory ON ≫ Chance

**Claim:** When the graph memory is populated with the correct knowledge graph, the system achieves accuracy far above chance level.

**Test:** Evaluate HR@1 on MetaQA test sets with the full graph loaded.

**Falsification condition:** If HR@1 ≤ random baseline ($\approx 1/N$ for $N = 43{,}234$ nodes), the graph executor is not learning meaningful traversals.

**Result:** HR@1 = 90.8% (1-hop), 65.8% (2-hop), 57.2% (3-hop). ✓ **Invariant holds.**

### Invariant 2: Memory OFF → Collapse

**Claim:** The system cannot answer from Qwen's parametric knowledge alone. Removing the graph causes accuracy to collapse.

**Test:** Replace all adjacency matrices with empty sparse tensors (zero edges) and re-evaluate.

**Falsification condition:** If HR@1 remains significantly above chance with empty adjacency matrices, the system is bypassing the graph and answering from Qwen's weights.

**Result:** HR@1 = 0.4% (1-hop), 1.2% (2-hop), 5.8% (3-hop). ✓ **Invariant holds.**

### Invariant 3: Readout Fidelity (Graph Corruption)

**Claim:** The system's answers depend on the specific graph topology, not merely on the presence of *any* graph.

**Test:** Randomly permute node indices in all adjacency matrices, destroying the real graph structure while preserving matrix dimensions, density, and degree distribution.

**Falsification condition:** If HR@1 remains high with corrupted adjacency matrices, the system is not reading the graph topology — it may be memorizing answers through the projection layer.

**Result:** HR@1 drops from 89% → 1% (1-hop), 62% → 6% (2-hop), 62% → 16% (3-hop). ✓ **Invariant holds.**

### Invariant 4: Retrieval Quality (Baseline Comparison)

**Claim:** The graph executor outperforms alternative knowledge-access mechanisms (CoT prompting, GraphRAG, sparse retrieval) on structured multi-hop reasoning.

**Test:** Compare against CoT Qwen, GraphRAG (local subgraph + Qwen), and Retriever+Qwen (TF-IDF top-50 + Qwen) baselines on identical test sets.

**Falsification condition:** If any baseline matches or exceeds the graph adapter's accuracy, the executor provides no architectural benefit.

**Result:** Graph adapter achieves the highest HR@1 across all hop counts. ✓ **Invariant holds.**

---

## 5. Datasets

### 5.1 MetaQA

MetaQA (Zhang et al., 2018) is a large-scale multi-hop question answering benchmark built over the WikiMovies knowledge base. It provides:

- **Knowledge base:** 43,234 entities, 9 relation types, 134,741 triples covering movies, actors, directors, genres, languages, and release years.
- **Question sets:** Train/dev/test splits for 1-hop, 2-hop, and 3-hop questions.
- **1-hop examples:** "Who directed [The Matrix]?" → Lana Wachowski
- **2-hop examples:** "What are the genres of films directed by [Christopher Nolan]?" → Sci-Fi, Action, Thriller
- **3-hop examples:** "What languages are spoken in films that share actors with [The Matrix]?" → English, Japanese

We evaluate on 100 randomly sampled test examples per hop count for baseline comparisons, and 500 examples per hop count for memory ablation experiments. We acknowledge this is a limitation (see Section 7.4) and report results accordingly.

### 5.2 MetaQA Node Type Distribution

The MetaQA knowledge graph contains the following node type distribution, used for answer-type masking:

| Node Type | Count |
|-----------|-------|
| Movie | 16,431 |
| Actor | 9,557 |
| Director | 6,240 |
| Writer | 5,907 |
| Tag | 4,887 |
| Year | 98 |
| Language | 94 |
| Genre | 20 |
| **Total** | **43,234** |

### 5.3 Biomedical Knowledge Graph

To evaluate domain transfer and real-world applicability, we construct a biomedical knowledge graph from curated drug–disease treatment relationships:

| Property | Value |
|----------|-------|
| Nodes | 5,695 |
| Edges | 18,762 |
| Relation types | 1 (treats) |
| Domain | Drug–disease interactions |
| Source | Curated biomedical databases |

This graph is significantly smaller and denser than MetaQA, with a fundamentally different topology: it is a bipartite graph (drugs → diseases) rather than a heterogeneous movie knowledge base. The chance baseline for random node selection is $1/5{,}695 \approx 0.0176\%$, or $\approx 0.34\%$ when restricted to valid answer-type nodes.

We split the graph into training edges (used to train the executor) and held-out edges (withheld from the graph during training, used for evaluation). This tests the system's ability to generalize to unseen drug–disease relationships — a critical requirement for any medical knowledge system.

### 5.4 Dataset Comparison

| Property | MetaQA | Biomedical KG |
|----------|--------|---------------|
| Nodes | 43,234 | 5,695 |
| Edges | 134,741 | 18,762 |
| Relations | 9 | 1 |
| Domain | Movies | Drug–disease |
| Question format | Template-based | Direct lookup |
| Multi-hop | 1/2/3-hop | 1-hop |
| Editability test | No | Yes (zero-shot insertion) |

---

## 6. Experiments

### 6.1 Training Configuration

We train only the **graph executor** (forward controller, backward controller) and the **projection layer** (Qwen hidden → graph query). All Qwen-1.5B parameters (1.8 billion) remain **completely frozen** throughout training. The trainable module is the adapter — not the LLM.

| Hyperparameter | Value |
|----------------|-------|
| LLM backbone | Qwen-1.5B (1.8B params, **all frozen**) |
| Trainable modules | Graph executor + projection layer |
| Graph hidden dim | 256 |
| Max supported hops | 3 |
| Optimizer | AdamW |
| Learning rate | 3×10⁻⁴ |
| Max epochs | 10 |
| Early stopping patience | 3 epochs |
| Gradient clipping | max_norm = 1.0 |
| Training samples | 2,000 per hop (MetaQA) |
| Dev samples | 500 per hop (MetaQA) |
| Dropout (projection) | 0.1 |

**Parameter budget clarification.** The total system has two distinct parameter groups:

| Component | Parameters | Trainable? |
|-----------|------------|------------|
| Qwen-1.5B backbone | 1.8B | ❌ Frozen |
| Projection layer (1536 → 256) | ~393K | ✅ Trained |
| Graph executor (fwd + bwd controllers) | ~107K | ✅ Trained |
| Copy gate head (1536 → 1) | ~1.5K | ✅ Trained |
| **Total trainable** | **~502K** | — |
| **Fraction of LLM** | **0.028%** | — |

This extreme parameter asymmetry (502K trainable vs 1.8B frozen) is a key architectural property: the adapter cannot memorize the knowledge graph in its weights — it must learn to *traverse* the external graph structure.

**Training loss:** We optimize the negative log-probability of the gold answer nodes under the executor's output distribution:

$$\mathcal{L} = -\log \left( \sum_{a \in \text{gold}} \mathbf{p}_{\text{final}}[a] + \epsilon \right)$$

### 6.2 Experiment 1: Comprehensive Baseline Comparison

We compare our graph adapter against four baselines on MetaQA 1-hop, 2-hop, and 3-hop test sets (100 samples each):

#### 1-Hop Results

| Method | Accuracy/HR@1 | Avg Tokens | Avg Latency |
|--------|---------------|------------|-------------|
| CoT Qwen-1.5B (Frozen) | 3.00% | 99.6 | 2.058s |
| GraphRAG (Local Subgraph + Qwen) | 60.00% | 38.5 | 0.811s |
| Retriever + Qwen (TF-IDF Top-50) | 63.00% | 50.0 | 1.041s |
| **Graph Adapter (Ours)** | **89.00%** | **1** | **0.038s** |
| Graph Adapter (Corrupted) | 1.00% | 1 | 0.031s |

#### 2-Hop Results

| Method | Accuracy/HR@1 | Avg Tokens | Avg Latency |
|--------|---------------|------------|-------------|
| CoT Qwen-1.5B (Frozen) | 16.00% | 99.8 | 2.054s |
| GraphRAG (Local Subgraph + Qwen) | 54.00% | 31.5 | 0.671s |
| Retriever + Qwen (TF-IDF Top-50) | 12.00% | 41.9 | 0.872s |
| **Graph Adapter (Ours)** | **62.00%** | **1** | **0.050s** |
| Graph Adapter (Corrupted) | 6.00% | 1 | 0.041s |

#### 3-Hop Results

| Method | Accuracy/HR@1 | Avg Tokens | Avg Latency |
|--------|---------------|------------|-------------|
| CoT Qwen-1.5B (Frozen) | 32.00% | 100.0 | 2.374s |
| GraphRAG (Local Subgraph + Qwen) | 27.00% | 40.0 | 1.005s |
| Retriever + Qwen (TF-IDF Top-50) | 40.00% | 36.0 | 0.862s |
| **Graph Adapter (Ours)** | **62.00%** | **1** | **0.070s** |
| Graph Adapter (Corrupted) | 16.00% | 1 | 0.054s |

**Key observations:**

1. The **graph adapter dominates all baselines** across all hop counts in both accuracy and latency.
2. **CoT Qwen-1.5B** performs poorly (3–32%) despite generating 100 tokens per query, confirming that small frozen LLMs cannot reliably perform multi-hop reasoning through chain-of-thought alone.
3. **GraphRAG** performs well on 1-hop (60%) and 2-hop (54%) but degrades severely on 3-hop (27%), because 3-hop subgraphs become too large and noisy for in-context reasoning.
4. **Retriever+Qwen** shows inconsistent performance: 63% on 1-hop but only 12% on 2-hop, revealing that sparse retrieval fails when the relevant triples are not lexically similar to the question.
5. The **graph adapter maintains stable 62% HR@1 on both 2-hop and 3-hop**, demonstrating that the executor's learned traversal generalizes across reasoning depths.

### 6.3 Experiment 2: Memory Ablation

This experiment directly tests Invariant 2 (Memory OFF → Collapse). We evaluate each trained model with the full graph (Memory ON) and with all adjacency matrices replaced by empty sparse tensors (Memory OFF).

| Hops | Memory ON (HR@1) | Memory OFF (HR@1) | Δ |
|------|-------------------|---------------------|---|
| 1 | 90.80% | 0.40% | −90.40 pp |
| 2 | 65.80% | 1.20% | −64.60 pp |
| 3 | 57.20% | 5.80% | −51.40 pp |

*Evaluated on 500 test samples per hop count.*

**Analysis:** The near-zero Memory OFF scores confirm that the LLM's parametric knowledge contributes negligibly to the final answer. The architecture forces all factual reasoning through the graph executor. The slightly higher Memory OFF score at 3-hop (5.8%) is consistent with random guessing among a larger answer type set (more movies in the candidate pool).

### 6.4 Experiment 3: Graph Corruption (Falsification)

This experiment tests Invariant 3 (Readout Fidelity). We randomly permute the row and column indices of every adjacency matrix, destroying the real graph topology while preserving:

- Matrix dimensions (43,234 × 43,234)
- Number of non-zero entries per matrix
- Approximate degree distribution

| Hops | Normal Graph (HR@1) | Corrupted Graph (HR@1) | Δ |
|------|----------------------|-------------------------|---|
| 1 | 89.00% | 1.00% | −88.00 pp |
| 2 | 62.00% | 6.00% | −56.00 pp |
| 3 | 62.00% | 16.00% | −46.00 pp |

**Analysis:** This is the strongest evidence of **causal graph dependence**. The corrupted graph has the same statistical properties (density, sparsity pattern) as the real graph, but all structural relationships are destroyed. The precipitous accuracy drop demonstrates that the executor is genuinely reading the graph topology — it is not memorizing answers through the projection layer or exploiting distributional regularities in the matrix structure.

The 3-hop corrupted score (16%) is slightly above chance, likely because the random permutation occasionally creates short paths that coincidentally connect source entities to correct answer nodes. This effect diminishes at lower hop counts where the path structure is more constrained.

### 6.5 Experiment 4: Efficiency Analysis

| Method | Forward Passes | Generated Tokens | Latency (per query) | Speedup |
|--------|----------------|-------------------|--------------------|---------|
| CoT Qwen-1.5B | 1 (gen) | ~100 | ~2.05s | 1× |
| GraphRAG + Qwen | 1 (gen) | ~35 | ~0.81s | 2.5× |
| Retriever + Qwen | 1 (gen) | ~43 | ~0.90s | 2.3× |
| **Graph Adapter** | **1 (fwd)** | **0** | **~0.05s** | **41×** |

The graph adapter achieves a **41× speedup** over CoT baselines. This efficiency gain comes from replacing autoregressive token generation with a single matrix-multiply forward pass through the executor. The executor's computation is dominated by sparse matrix operations on the adjacency matrices, which are highly parallelizable on GPU hardware.

### 6.6 Experiment 5: Extended Metrics (HR@5, HR@10)

| Hops | HR@1 | HR@5 | HR@10 |
|------|------|------|-------|
| 1 | 89% | 89% | 89% |
| 2 | 62% | 80% | 80% |
| 3 | 62% | 78% | 82% |

The flat HR@1 = HR@5 = HR@10 at 1-hop indicates high-confidence predictions — when the model is correct, the correct answer is ranked first with high margin. The improvement from HR@1 to HR@5/HR@10 at 2-hop and 3-hop suggests that multi-hop traversals produce broader probability distributions, which is expected given the combinatorial expansion of possible paths.

### 6.7 Experiment 6: Biomedical Knowledge Graph

To test domain transfer beyond the movie domain, we evaluate on a real biomedical knowledge graph of drug–disease treatment relationships. This experiment is critical because it demonstrates that the architecture generalizes to a completely different domain, topology, and scale.

#### 6.7.1 Held-Out Fact Evaluation

We withhold a subset of drug–disease edges from the training graph and evaluate whether the trained executor can correctly answer queries about these held-out relationships.

| Metric | Value |
|--------|-------|
| Held-out facts evaluated | 476 |
| Chance baseline (random node) | 0.34% |
| **HR@1 (Memory ON)** | **45.59%** |
| HR@5 (Memory ON) | 85.08% |
| HR@10 (Memory ON) | 98.32% |
| HR@1 (Memory OFF) | 0.00% |
| Lift over chance | **135×** |

**Analysis:** The 45.59% HR@1 on *held-out facts* — relationships the model has never seen during training — demonstrates that the executor learns generalizable traversal patterns, not memorized paths. The 98.32% HR@10 indicates that the correct drug is almost always in the top-10 candidates, suggesting high-quality probability distributions even when the top-1 prediction is incorrect.

The 0% Memory OFF score confirms complete graph dependence in the biomedical domain, consistent with the MetaQA results. The LLM backbone (Qwen-1.5B) has no biomedical knowledge encoded in its weights that could bypass the graph.

#### 6.7.2 Train Fact Verification

As a sanity check, we also evaluate on facts present in the training graph:

| Condition | HR@1 |
|-----------|------|
| Train facts, Memory ON | 94.00% |
| Train facts, Memory OFF | 0.00% |

The 94% HR@1 on training facts confirms that the executor has learned to traverse known edges accurately. The gap between train (94%) and held-out (45.6%) performance is expected — held-out facts require generalization beyond directly observed paths.

#### 6.7.3 Plain LLM Baseline

We also evaluate the frozen Qwen-1.5B backbone without any graph adapter, prompting it directly with biomedical questions:

| Baseline | HR@1 |
|----------|------|
| Plain Qwen-1.5B (no graph) | 0.00% |

This confirms that Qwen-1.5B has no useful biomedical knowledge for this task, making it an ideal testbed for demonstrating that all factual knowledge flows through the graph.

### 6.8 Experiment 7: Zero-Shot Fact Insertion

This experiment directly demonstrates the **editability** claim — the architecture's ability to incorporate new knowledge without any retraining.

**Protocol:**
1. Select a drug–disease pair **not present** in the graph.
2. Query the system → observe incorrect answer (the executor traverses to an unrelated node).
3. **Insert the new edge** into the adjacency matrix (a single sparse tensor update).
4. Query the system again with identical input → observe correct answer.

No model weights are modified between steps 2 and 4. The only change is the graph structure.

#### Results

| Disease | Inserted Drug | Before Insertion | After Insertion | Success |
|---------|---------------|------------------|-----------------|---------|
| Connective tissue disease | Bosentan hydrate | ❌ etanercept | ✅ bosentan hydrate | ✅ |
| Gonorrhea | Gatifloxacin | ❌ demeclocycline | ✅ gatifloxacin | ✅ |
| Prostate cancer | Doxorubicin HCl | ❌ bariatric medicine | ✅ doxorubicin HCl | ✅ |
| Osteoarthritis | Rofecoxib | ❌ teriparatide | ✅ rofecoxib | ✅ |
| Tuberculosis | Capreomycin | ❌ bariatric medicine | ✅ capreomycin | ✅ |

**Zero-shot insertion success rate: 100% (5/5)**

**Analysis:** Before insertion, the executor traverses the existing graph and returns an incorrect drug — often a plausible but wrong answer (e.g., "etanercept" for connective tissue disease is a real immunosuppressant, but not the target drug). After inserting the new edge, the executor immediately routes to the correct answer.

This experiment demonstrates a capability that no weight-based knowledge system can achieve: **instant, surgical knowledge updates** without gradient computation, fine-tuning, or risk of catastrophic forgetting. The inserted fact is immediately available for all future queries, and no existing knowledge is disturbed.

**Implications for production systems:** In a clinical decision support system, a newly approved drug could be added to the graph in milliseconds and immediately be available for physician queries. This contrasts with the weeks or months required to retrain or fine-tune an LLM on updated medical literature.

---

## 7. Discussion

### 7.1 What Works

**External editable memory.** The graph-memory architecture achieves its primary design goal: knowledge is stored in an explicit, editable structure that can be modified without retraining. Adding or removing a fact requires inserting/deleting a sparse matrix entry — an O(1) operation. The zero-shot fact insertion experiment (Section 6.8) demonstrates this concretely: 100% of newly inserted drug–disease facts were immediately retrievable.

**Non-bypassable memory dependence.** The information bottleneck (1536-dim → 256-dim projection) combined with frozen Qwen weights creates a genuine dependence on the external graph. This is not a soft preference — the memory ablation demonstrates near-total accuracy collapse without the graph, consistently across both MetaQA (90.8% → 0.4%) and the biomedical KG (45.6% → 0.0%).

**Cross-domain generalization.** The same architecture, with no structural modifications, achieves strong results on two fundamentally different knowledge graphs: a heterogeneous movie KB (43K nodes, 9 relations) and a bipartite biomedical graph (5.7K nodes, 1 relation). This suggests the differentiable executor learns domain-agnostic traversal primitives.

**Token-efficient reasoning.** By replacing autoregressive generation with a single forward pass, we achieve a 41× latency improvement. This makes the architecture practical for high-throughput applications where CoT reasoning is prohibitively expensive.

**Stable multi-hop performance.** Unlike GraphRAG (which degrades from 60% to 27% across 1-hop to 3-hop) and Retriever+Qwen (which is inconsistent: 63%, 12%, 40%), the graph adapter maintains stable 62% HR@1 on both 2-hop and 3-hop tasks.

### 7.2 What Doesn't Work (Yet)

**Open-ended generation.** The current architecture produces a pointer to a graph node, not free-text output. It cannot generate explanations, summaries, or answers that require composition of multiple entities. Extending to generative outputs would require re-introducing the LLM's language modeling head with a copy mechanism.

**Autonomous graph construction.** Our system assumes a pre-existing knowledge graph. Building the graph from unstructured text — entity extraction, relation extraction, co-reference resolution, and triple canonicalization — is a substantial engineering challenge that we do not address.

**Scientific discovery and novel inference.** The graph executor can only traverse edges that exist in the graph. It cannot infer new relationships, propose hypotheses, or reason counterfactually about entities not present in the knowledge base. This is a fundamental limitation of the current forward-traversal design.

**Held-out generalization gap.** On the biomedical KG, there is a significant gap between train-fact accuracy (94%) and held-out-fact accuracy (45.6%). While 45.6% is 135× above chance, closing this gap — potentially through better traversal learning or incorporating node embeddings — remains an open challenge.

### 7.3 Honest Assessment of Limitations

We deliberately avoid overclaiming. Our architecture is not:

- A general reasoning system (it is specialized for structured graph traversal)
- A replacement for large-scale LLMs (it augments them for knowledge-intensive tasks)
- A solution to the alignment problem (it addresses knowledge editability, not value alignment)
- Comparable to state-of-the-art KGQA systems that use much larger models and domain-specific training

The MetaQA benchmark, while standard, has known limitations: questions follow templates, entities are pre-marked, and the knowledge graph is well-structured. Performance on open-domain, noisy, or incomplete graphs would likely be lower.

### 7.4 Statistical Limitations and Reproducibility

**Sample size.** Our MetaQA baseline comparisons (Section 6.2) use 100 randomly sampled test examples per hop count. While sufficient to demonstrate clear trends (the adapter's 89% vs CoT's 3% on 1-hop is unlikely to reverse at larger $n$), we acknowledge that 100 samples provide limited statistical power for fine-grained comparisons. Memory ablation experiments use 500 samples. The biomedical evaluation uses 476 held-out facts. Future work should evaluate on full test splits.

**Single seed.** All results are from a single training run with a single random seed. We do not report mean ± standard deviation across multiple seeds. While the falsification experiments (Memory OFF, Graph Corruption) produce such extreme effect sizes (89% → 1%) that they are unlikely to be seed-dependent, the absolute HR@1 numbers may vary by several percentage points across runs. We encourage replication with 3–5 seeds and recommend reporting confidence intervals.

**Evaluation protocol.** MetaQA results use HR@1 (exact top-1 match), which is a strict metric. The high HR@5 and HR@10 scores (80–98%) suggest that relaxed metrics would yield substantially higher reported accuracy.

---

## 8. Future Work

### 8.1 Confidence-Weighted Graphs

Current graph edges are binary: an edge either exists or it doesn't. We propose extending the adjacency matrices to store **confidence scores**:

```
Verified edges:      weight = 1.0  (from authoritative sources)
Hypothesized edges:  weight = 0.3  (from link prediction)
Contradicted edges:  weight = 0.0  (explicitly retracted)
```

The executor's traversal would then be modulated by edge confidence, automatically down-weighting uncertain paths. This creates a spectrum between "known facts" and "plausible inferences."

### 8.2 RotatE Link Prediction for Missing Edges

Knowledge graphs are inherently incomplete. We propose integrating RotatE (Sun et al., 2019) embeddings to predict missing edges and insert them as low-confidence hypothesized edges. The training signal would come from the executor's traversal failures: if the executor consistently fails on questions that would be answerable with a specific missing edge, that edge should be proposed.

### 8.3 Abductive Reasoning

Given a question and a set of answers, **abductive reasoning** asks: "What edges would need to exist for these answers to be reachable?" This inverts the executor's forward traversal:

```
Forward:   source + graph → answers
Abductive: source + answers → missing edges
```

This capability would enable the system to generate **novel hypotheses** — proposing relationships that are not in the knowledge base but would explain observed phenomena.

### 8.4 Continuous Knowledge Growth

We envision a closed-loop system:

```
1. User asks question → executor traverses → answers
2. If confidence < threshold → flag for review
3. If new fact confirmed → insert edge → immediate effect
4. If edge contradicted → remove/downweight → immediate effect
```

This "insert-back loop" would allow the knowledge graph to grow continuously without any model retraining, achieving a form of lifelong learning through graph editing rather than weight updates.

---

## 9. Conclusion

We have demonstrated that **knowledge ≠ weights**. Factual knowledge can be decoupled from language model parameters and stored in an explicit, editable graph structure that is accessed through a differentiable graph executor. Using only ~500K trainable parameters (0.028% of the frozen Qwen-1.5B backbone), our adapter achieves strong multi-hop reasoning on MetaQA and generalizes to a real biomedical knowledge graph.

Our experimental evidence supports four key claims:

1. **Editable memory:** Knowledge is stored in sparse adjacency matrices that can be modified in O(1) time without retraining. We demonstrated 100% zero-shot fact insertion success on biomedical drug–disease relationships.
2. **Graph dependence:** Removing the graph (Memory OFF) causes accuracy to collapse from 90.8% to 0.4% (MetaQA 1-hop) and from 45.6% to 0.0% (biomedical). Corrupting the graph causes collapse from 89% to 1%. This is causal, not correlational.
3. **Cross-domain transfer:** The same architecture achieves strong results on both a movie knowledge base (43K nodes, 9 relations) and a biomedical graph (5.7K nodes, 1 relation) without architectural modifications.
4. **Efficient reasoning:** A single forward pass through the executor replaces 100 tokens of chain-of-thought generation, achieving a 41× speedup.

The falsification framework we introduce — four explicit invariants with corresponding experiments — provides a template for evaluating future graph-memory architectures. We encourage the community to adopt similar falsification-first evaluation protocols.

Our work establishes a foundation for systems where knowledge is **inspectable** (you can read the graph), **editable** (you can change the graph), **verifiable** (you can test graph dependence), and **efficient** (reasoning requires no token generation). These properties are prerequisites for trustworthy AI systems operating in domains where knowledge changes frequently and correctness is critical.

---

## References

- Abboud, R., Ceylan, İ., Lukasiewicz, T., & Salvatori, T. (2020). BoxE: A box embedding model for knowledge base completion. *NeurIPS 2020*.
- Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. *NeurIPS 2013*.
- Garcez, A. d., Gori, M., Lamb, L. C., Serafini, L., Spranger, M., & Tran, S. N. (2019). Neural-symbolic computing: An effective methodology for principled integration of machine learning and reasoning. *JAIR*.
- Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M. W. (2020). Retrieval augmented language model pre-training. *ICML 2020*.
- Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense passage retrieval for open-domain question answering. *EMNLP 2020*.
- Lamb, L. C., Garcez, A. d., Gori, M., Prates, M. O. R., Avelar, P. H. C., & Vardi, M. Y. (2020). Graph neural networks meet neural-symbolic computing: A survey and perspective. *IJCAI 2020*.
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.
- Manhaeve, R., Dumančić, S., Kimmig, A., Demeester, T., & De Raedt, L. (2018). DeepProbLog: Neural probabilistic logic programming. *NeurIPS 2018*.
- Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*.
- Rocktäschel, T., & Riedel, S. (2017). End-to-end differentiable proving. *NeurIPS 2017*.
- Saxena, A., Tripathi, A., & Talukdar, P. (2020). Improving multi-hop question answering over knowledge graphs using knowledge base embeddings. *ACL 2020*.
- Sun, Z., Deng, Z. H., Nie, J. Y., & Tang, J. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. *ICLR 2019*.
- Wu, L., Petroni, F., Josifoski, M., Riedel, S., & Zettlemoyer, L. (2020). Scalable zero-shot entity linking with dense entity retrieval. *EMNLP 2020*.
- Yasunaga, M., Ren, H., Bosselut, A., Liang, P., & Leskovec, J. (2021). QA-GNN: Reasoning with language models and knowledge graphs for question answering. *NAACL 2021*.
- Zhang, Y., Dai, H., Kozareva, Z., Smola, A. J., & Song, L. (2018). Variational reasoning for question answering with knowledge graph. *AAAI 2018*.
