# A Falsifiable Graph-Memory Architecture for Editable Knowledge and Token-Efficient Multi-Hop Reasoning

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)

This repository contains the official implementation of the paper **"A Falsifiable Graph-Memory Architecture for Editable Knowledge and Token-Efficient Multi-Hop Reasoning"**.

## Abstract

Large language models (LLMs) encode factual knowledge implicitly within their parameters, creating a fundamental tension between knowledge currency and model stability. When facts change, the only recourse is costly retraining or fine-tuning. Furthermore, multi-hop reasoning over structured knowledge requires expensive chain-of-thought (CoT) token generation.

We introduce a **Graph-Memory Architecture** that decouples factual knowledge from model weights by composing three components:
1. An explicit, editable **Graph Memory** storing entities and relations as sparse adjacency matrices.
2. A **Differentiable Graph Executor** (~500K trainable parameters) that performs learned multi-hop traversals via soft relation-selection controllers.
3. A frozen **LLM backbone** (Qwen-1.5B) that serves solely as a semantic parser.

Our architecture achieves up to a **41× latency reduction** over CoT baselines, enables **100% zero-shot fact insertion success**, and demonstrates high resilience to graph corruption and ablation.

*Note: The paper is available on [arXiv](https://arxiv.org/abs/placeholder-link) (placeholder link).*

## Key Results

| Method | 1-Hop HR@1 | 2-Hop HR@1 | 3-Hop HR@1 | Avg Latency |
|--------|------------|------------|------------|-------------|
| CoT Qwen-1.5B (Frozen) | 3.00% | 16.00% | 32.00% | ~2.05s |
| GraphRAG + Qwen | 60.00% | 54.00% | 27.00% | ~0.81s |
| Retriever + Qwen | 63.00% | 12.00% | 40.00% | ~0.90s |
| **Graph Adapter (Ours)** | **89.00%** | **62.00%** | **62.00%** | **~0.05s** |

## Repository Structure

- `src/` - Core model architecture components.
  - `qwen_adapter_model.py`: The QwenPointerAdapter combining the LLM and graph executor.
  - `differentiable_graph_executor.py`: Differentiable reasoning module over sparse matrices.
  - `metaqa_linker.py`: Entity linking for the MetaQA dataset.
- `eval/` - Evaluation and benchmarking scripts.
  - `eval_metaqa.py`: Main evaluation script.
  - `cot_comparison.py`, `resume_benchmarks.py`, `memory_ablation.py`.
- `scripts/` - Scripts for dataset building, model training, and remote execution.
  - Modal remote execution scripts (`train_on_modal.py`, `build_counterfactuals_modal.py`, etc.)
  - General training (`train_metaqa.py`).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/neuro-brain-qwen-adapter.git
   cd neuro-brain-qwen-adapter
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Export the Python path so scripts can find the `src` module:
   ```bash
   export PYTHONPATH=$(pwd)
   ```

## Usage

### Training

To train the graph executor on the MetaQA dataset (keeping the Qwen-1.5B backbone frozen):
```bash
python src/train_metaqa.py
```

To run training remotely using [Modal](https://modal.com/):
```bash
modal run scripts/train_on_modal.py
```

### Evaluation

Run the evaluation script to test multi-hop reasoning (ensure `PYTHONPATH` is set):
```bash
python eval/eval_metaqa.py
```

Run baseline comparisons and ablation studies:
```bash
python eval/cot_comparison.py
python eval/memory_ablation.py
```

## Citation

```bibtex
@misc{mahesh2026falsifiable,
  title={A Falsifiable Graph-Memory Architecture for Editable Knowledge and Token-Efficient Multi-Hop Reasoning},
  author={Mahesh and others},
  year={2026},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## License

This project is licensed under the MIT License.
