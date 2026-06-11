import sys; import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modal
import os

app = modal.App("metaqa-memory-ablation")

volume = modal.Volume.from_name("metaqa-checkpoints", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "accelerate", "tqdm"
).add_local_python_source("qwen_adapter_model", "differentiable_graph_executor", "train_metaqa")

@app.function(image=image, gpu="A100", timeout=7200, volumes={"/checkpoints": volume})
def run_ablation(kb_content: str, datasets: dict):
    import torch
    from transformers import AutoTokenizer
    
    with open("kb.txt", "w") as f:
        f.write(kb_content)
        
    for hop in [1, 2, 3]:
        os.makedirs(f"{hop}-hop/vanilla", exist_ok=True)
        with open(f"{hop}-hop/vanilla/qa_test.txt", "w") as f:
            f.write(datasets[hop])
            
    from qwen_adapter_model import QwenPointerAdapter
    from train_metaqa import load_kb, load_train_data, infer_answer_type

    device = torch.device("cuda:0")
    qwen_name = "Qwen/Qwen1.5-1.8B"
    tokenizer = AutoTokenizer.from_pretrained(qwen_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load original KB
    node2idx, rel2idx, adj_matrices_on, type_masks, num_nodes, num_relations = load_kb("kb.txt", device)
    idx2node = {v: k for k, v in node2idx.items()}

    # Create empty KB (Memory OFF)
    adj_matrices_off = []
    for _ in range(num_relations * 2):
        empty_adj = torch.sparse_coo_tensor(
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            (num_nodes, num_nodes)
        ).coalesce()
        adj_matrices_off.append(empty_adj)

    print("\n==================================================")
    print("MEMORY ABLATION EXPERIMENT")
    print("==================================================")

    for hop in [1, 2, 3]:
        print(f"\nEvaluating {hop}-hop model...")
        test_data = load_train_data(node2idx, f"{hop}-hop/vanilla/qa_test.txt")
        test_data = test_data[:500]  # Just use 500 for the ablation test to be fast
        
        model = QwenPointerAdapter(
            qwen_model_name=qwen_name,
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_dim=256
        ).to(device)
        
        checkpoint_path = f"/checkpoints/phase1_checkpoint_{hop}hop.pt"
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} missing! Skipping...")
            continue
            
        state = torch.load(checkpoint_path, map_location=device)
        model.graph_executor.load_state_dict(state["graph_executor"])
        model.qwen_to_graph.load_state_dict(state["qwen_to_graph"])
        model.eval()

        for mode, adj_matrices in [("Memory ON", adj_matrices_on), ("Memory OFF", adj_matrices_off)]:
            hits = 0
            with torch.no_grad():
                for item in test_data:
                    q = item["question"]
                    answers = item["answers"]
                    ent = item["head_entity"]
                    
                    if ent not in node2idx:
                        continue
                    src_idx = node2idx[ent]

                    p_src = torch.zeros(1, num_nodes, device=device)
                    p_src[0, src_idx] = 1.0

                    prompt = f"Q: {q}\nA: "
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    outputs = model.qwen(
                        **inputs,
                        output_hidden_states=True,
                    )
                    h_last = outputs.hidden_states[-1][:, -1, :].float()

                    graph_query = model.qwen_to_graph(h_last)
                    p_eval, _ = model.graph_executor(
                        query=graph_query,
                        p_src=p_src,
                        adj_matrices=adj_matrices,
                        p_tgt=None,
                        is_eval=True,
                        max_hops=hop,
                    )

                    answer_type = infer_answer_type(q)
                    mask = type_masks.get(answer_type, torch.ones(num_nodes, device=device))
                    p_masked = p_eval[0] * mask
                    pred = p_masked.argmax().item()

                    if idx2node.get(pred, "") in answers:
                        hits += 1

            hr = (hits / len(test_data)) * 100
            print(f"  {mode} HR@1: {hr:.2f}%")

@app.local_entrypoint()
def main():
    print("Reading kb.txt...")
    with open("kb.txt", "r") as f:
        kb_content = f.read()
        
    datasets = {}
    for hop in [1, 2, 3]:
        print(f"Reading {hop}-hop test...")
        with open(f"{hop}-hop/vanilla/qa_test.txt", "r") as f:
            datasets[hop] = f.read()
                
    run_ablation.remote(kb_content, datasets)
