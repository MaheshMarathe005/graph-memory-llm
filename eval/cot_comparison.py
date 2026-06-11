import sys; import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.qwen_adapter_model import QwenPointerAdapter
from src import train_metaqa
from tqdm import tqdm
import re
import gc
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_kb_raw(kb_path="kb.txt"):
    kb_triples = []
    adj_list = defaultdict(list)
    with open(kb_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            s, r, o = line.split('|')
            kb_triples.append((s, r, o))
            adj_list[s].append((r, o))
            adj_list[o].append((r + "_inv", s))
    return kb_triples, adj_list

def extract_k_hop_subgraph(head_entity, adj_list, max_hops):
    visited = set([head_entity])
    queue = [(head_entity, 0)]
    subgraph_triples = []
    
    while queue:
        curr_node, dist = queue.pop(0)
        if dist >= max_hops:
            continue
        for r, o in adj_list[curr_node]:
            if r.endswith("_inv"):
                subgraph_triples.append((o, r.replace("_inv", ""), curr_node))
            else:
                subgraph_triples.append((curr_node, r, o))
            
            if o not in visited:
                visited.add(o)
                queue.append((o, dist + 1))
                
    return list(set(subgraph_triples))

class TFIDFRetriever:
    def __init__(self, kb_triples):
        self.triples = kb_triples
        self.documents = [f"{s} {r.replace('_', ' ')} {o}" for s, r, o in kb_triples]
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        
    def retrieve(self, query, top_k=50):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_vectors).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]
        return [self.triples[i] for i in top_indices]

def evaluate_graphrag(item, model, tokenizer, device, adj_list, hop_count):
    q = item["question"]
    head = item["head_entity"]
    answers = item["answers"]
    
    subgraph = extract_k_hop_subgraph(head, adj_list, hop_count)
    # LIMIT context size to prevent OOM!
    subgraph = subgraph[:200]
    context_str = "\n".join([f"{s} {r} {o}" for s, r, o in subgraph])
    
    prompt = f"""You are an expert answering questions based on the provided knowledge graph subgraph.
Your final answer must be a single entity name inside [brackets], e.g., [The Matrix].
If you don't know the answer, still provide your best guess inside brackets.

Knowledge:
{context_str}

Question: {q}
Thought:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - start_time
    
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    num_tokens = len(generated_tokens)
    
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    match = re.search(r'\[(.*?)\]', text)
    pred_answer = match.group(1) if match else text.strip()
    
    is_correct = any(a.lower() in pred_answer.lower() for a in answers)
    return is_correct, num_tokens, latency

def evaluate_retriever_qwen(item, model, tokenizer, device, retriever):
    q = item["question"]
    answers = item["answers"]
    
    retrieved_triples = retriever.retrieve(q, top_k=50)
    context_str = "\n".join([f"{s} {r} {o}" for s, r, o in retrieved_triples])
    
    prompt = f"""You are an expert answering questions based on retrieved knowledge triples.
Your final answer must be a single entity name inside [brackets], e.g., [The Matrix].

Knowledge:
{context_str}

Question: {q}
Thought:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - start_time
    
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    num_tokens = len(generated_tokens)
    
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    match = re.search(r'\[(.*?)\]', text)
    pred_answer = match.group(1) if match else text.strip()
    
    is_correct = any(a.lower() in pred_answer.lower() for a in answers)
    return is_correct, num_tokens, latency

def corrupt_adj_matrices(adj_matrices):
    corrupted = []
    for adj in adj_matrices:
        n = adj.shape[0]
        row_perm = torch.randperm(n, device=adj.device)
        col_perm = torch.randperm(n, device=adj.device)
        
        indices = adj.indices()
        values = adj.values()
        
        new_row_indices = row_perm[indices[0]]
        new_col_indices = col_perm[indices[1]]
        new_indices = torch.stack([new_row_indices, new_col_indices], dim=0)
        
        corr_adj = torch.sparse_coo_tensor(new_indices, values, (n, n), device=adj.device).coalesce()
        corrupted.append(corr_adj)
    return corrupted

def generate_cot_prompt(question: str) -> str:
    # A strong few-shot prompt for MetaQA
    return f"""You are an expert graph reasoner. Think step by step to answer the question.
Your final answer must be a single entity name inside [brackets], e.g., [The Matrix].

Question: what genres do films by Christopher Nolan fall under
Thought:
1. The question asks for the genres of films directed by Christopher Nolan.
2. The director is Christopher Nolan.
3. Films directed by Christopher Nolan include Inception, Interstellar, and The Dark Knight.
4. The genres of these films include Sci-Fi, Action, and Thriller.
Answer: [Sci-Fi]

Question: {question}
Thought:
"""

def evaluate_cot(item, model, tokenizer, device):
    q = item["question"]
    answers = item["answers"]
    
    prompt = generate_cot_prompt(q)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - start_time
    
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    num_tokens = len(generated_tokens)
    
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Parse answer
    match = re.search(r'\[(.*?)\]', text)
    pred_answer = match.group(1) if match else text.strip()
    
    is_correct = any(a.lower() in pred_answer.lower() for a in answers)
    
    return is_correct, num_tokens, latency

def evaluate_adapter(item, model, tokenizer, device, node2idx, type_masks, num_nodes, adj_matrices, hop_count):
    q = item["question"]
    head = item["head_entity"]
    answers = item["answers"]
    
    src_idx = node2idx.get(head, -1)
    if src_idx == -1:
        return False, False, False, 1, 0.0

    target_indices = set([node2idx.get(a, -1) for a in answers])
    answer_type = train_metaqa.infer_answer_type(q)
    
    prompt = f"Q: {q}\nA: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
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
        
        top10_vals, top10_idx = torch.topk(p_masked, k=10)
        top10_idx = top10_idx.tolist()
        
    latency = time.time() - start_time
    num_tokens = 1  # 1 forward pass
    
    hr1 = top10_idx[0] in target_indices
    hr5 = any(idx in target_indices for idx in top10_idx[:5])
    hr10 = any(idx in target_indices for idx in top10_idx[:10])
    
    return hr1, hr5, hr10, num_tokens, latency

def run_benchmark(hop_count: int, num_samples: int = 100, ckpt_dir: str = "."):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading KB...")
    node2idx, rel2idx, adj_matrices, type_masks, num_nodes, num_relations = train_metaqa.load_kb(device=device)
    kb_triples, adj_list = load_kb_raw()
    retriever = TFIDFRetriever(kb_triples)

    print(f"Loading {hop_count}-hop test dataset...")
    dataset = train_metaqa.load_train_data(node2idx, qa_path=f"{hop_count}-hop/vanilla/qa_test.txt", limit=num_samples)
    print(f"Loaded {len(dataset)} examples.")

    qwen_name = "Qwen/Qwen1.5-1.8B"
    tokenizer = AutoTokenizer.from_pretrained(qwen_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"\n--- Loading {qwen_name} for Generation Baselines ---")
    gen_model = AutoModelForCausalLM.from_pretrained(qwen_name, device_map="cuda:0")
    gen_model.eval()

    # ── Evaluate CoT Baseline ──
    print(f"\n--- Benchmarking CoT Baseline on {hop_count}-hop ---")
    cot_hits, cot_tokens, cot_latency = 0, 0, 0.0
    for item in tqdm(dataset):
        c, tok, lat = evaluate_cot(item, gen_model, tokenizer, device)
        cot_hits += int(c)
        cot_tokens += tok
        cot_latency += lat

    # ── Evaluate GraphRAG Baseline ──
    print(f"\n--- Benchmarking GraphRAG on {hop_count}-hop ---")
    grag_hits, grag_tokens, grag_latency = 0, 0, 0.0
    for item in tqdm(dataset):
        c, tok, lat = evaluate_graphrag(item, gen_model, tokenizer, device, adj_list, hop_count)
        grag_hits += int(c)
        grag_tokens += tok
        grag_latency += lat
        torch.cuda.empty_cache()

    # ── Evaluate Retriever + Qwen ──
    print(f"\n--- Benchmarking Retriever + Qwen on {hop_count}-hop ---")
    rq_hits, rq_tokens, rq_latency = 0, 0, 0.0
    for item in tqdm(dataset):
        c, tok, lat = evaluate_retriever_qwen(item, gen_model, tokenizer, device, retriever)
        rq_hits += int(c)
        rq_tokens += tok
        rq_latency += lat
        torch.cuda.empty_cache()
        
    del gen_model
    torch.cuda.empty_cache()
    gc.collect()

    # ── Evaluate Graph Adapter ──
    print(f"\n--- Benchmarking Graph Adapter on {hop_count}-hop ---")
    adapter = QwenPointerAdapter(
        qwen_model_name=qwen_name,
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=256,
    )
    
    import os
    ckpt_name = os.path.join(ckpt_dir, f"phase1_checkpoint_{hop_count}hop.pt")
    print(f"Loading {ckpt_name}...")
    state_dict = torch.load(ckpt_name, map_location=device)
    adapter.graph_executor.load_state_dict(state_dict['graph_executor'])
    adapter.qwen_to_graph.load_state_dict(state_dict['qwen_to_graph'])
    
    adapter.graph_executor.to(device)
    adapter.qwen_to_graph.to(device)
    adapter.eval()
    
    ad_h1, ad_h5, ad_h10, ad_tokens, ad_latency = 0, 0, 0, 0, 0.0
    for item in tqdm(dataset):
        h1, h5, h10, tok, lat = evaluate_adapter(item, adapter, tokenizer, device, node2idx, type_masks, num_nodes, adj_matrices, hop_count)
        ad_h1 += int(h1)
        ad_h5 += int(h5)
        ad_h10 += int(h10)
        ad_tokens += tok
        ad_latency += lat
        
    # ── Evaluate Graph Adapter (Corrupted Graph) ──
    print(f"\n--- Benchmarking Graph Adapter (Corrupted Graph) on {hop_count}-hop ---")
    corr_adj_matrices = corrupt_adj_matrices(adj_matrices)
    cad_h1, cad_h5, cad_h10, cad_tokens, cad_latency = 0, 0, 0, 0, 0.0
    for item in tqdm(dataset):
        h1, h5, h10, tok, lat = evaluate_adapter(item, adapter, tokenizer, device, node2idx, type_masks, num_nodes, corr_adj_matrices, hop_count)
        cad_h1 += int(h1)
        cad_h5 += int(h5)
        cad_h10 += int(h10)
        cad_tokens += tok
        cad_latency += lat
        
    print("\n================ BENCHMARK RESULTS ================")
    print(f"Hop Count: {hop_count}")
    print(f"Samples Evaluated: {len(dataset)}")
    print("-" * 50)
    print("CoT Qwen-1.5B (Frozen):")
    print(f"  Accuracy (Regex Match): {cot_hits/len(dataset):.2%}")
    print(f"  Avg Tokens per Query:   {cot_tokens/len(dataset):.1f}")
    print(f"  Avg Latency per Query:  {cot_latency/len(dataset):.4f} sec")
    print("-" * 50)
    print("GraphRAG (Local Subgraph + Frozen Qwen):")
    print(f"  Accuracy: {grag_hits/len(dataset):.2%}")
    print(f"  Avg Tokens per Query:   {grag_tokens/len(dataset):.1f}")
    print(f"  Avg Latency per Query:  {grag_latency/len(dataset):.4f} sec")
    print("-" * 50)
    print("Retriever + Qwen (TF-IDF Top-50 + Frozen Qwen):")
    print(f"  Accuracy: {rq_hits/len(dataset):.2%}")
    print(f"  Avg Tokens per Query:   {rq_tokens/len(dataset):.1f}")
    print(f"  Avg Latency per Query:  {rq_latency/len(dataset):.4f} sec")
    print("-" * 50)
    print("Graph Adapter (Forward-Only):")
    print(f"  HR@1:  {ad_h1/len(dataset):.2%}")
    print(f"  HR@5:  {ad_h5/len(dataset):.2%}")
    print(f"  HR@10: {ad_h10/len(dataset):.2%}")
    print(f"  Avg Latency per Query:  {ad_latency/len(dataset):.4f} sec")
    print("-" * 50)
    print("Graph Adapter (Corrupted Graph):")
    print(f"  HR@1:  {cad_h1/len(dataset):.2%}")
    print(f"  HR@5:  {cad_h5/len(dataset):.2%}")
    print(f"  HR@10: {cad_h10/len(dataset):.2%}")
    print(f"  Avg Latency per Query:  {cad_latency/len(dataset):.4f} sec")
    print("===================================================\n")

if __name__ == "__main__":
    import sys
    hop = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ckpt = sys.argv[3] if len(sys.argv) > 3 else "."
    run_benchmark(hop_count=hop, num_samples=num_samples, ckpt_dir=ckpt)
