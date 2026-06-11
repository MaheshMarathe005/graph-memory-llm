import sys; import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from src import train_metaqa
from src.qwen_adapter_model import QwenPointerAdapter

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading KB...")
    node2idx, rel2idx, adj_matrices, type_masks, num_nodes, num_relations = \
        train_metaqa.load_kb(device=device)

    print("Loading test dataset (2-hop true QA)...")
    dataset = train_metaqa.load_train_data(
        node2idx, qa_path="2-hop/vanilla/qa_test.txt", limit=1000  # Evaluate on first 1000
    )
    print(f"Loaded {len(dataset)} test examples.")

    qwen_name = "Qwen/Qwen1.5-1.8B"
    model = QwenPointerAdapter(
        qwen_model_name=qwen_name,
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=256,
    )
    model.graph_executor.to(device)
    model.qwen_to_graph.to(device)
    
    print("Loading phase1_checkpoint.pt...")
    state_dict = torch.load("phase1_checkpoint.pt", map_location=device)
    model.graph_executor.load_state_dict(state_dict['graph_executor'])
    model.qwen_to_graph.load_state_dict(state_dict['qwen_to_graph'])
    
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(qwen_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hits_1 = 0
    hits_5 = 0
    hits_10 = 0

    print("Evaluating...")
    for item in tqdm(dataset):
        head = item["head_entity"]
        answers = item["answers"]
        q = item["question"]

        src_idx = node2idx[head]
        answer_type = train_metaqa.infer_answer_type(q)
        target_indices = set([node2idx[a] for a in answers])

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
            )
            
            mask = type_masks.get(
                answer_type,
                torch.ones(num_nodes, device=device)
            )
            p_masked = p_eval[0] * mask
            
            # Get top 10
            top10_vals, top10_idx = torch.topk(p_masked, k=10)
            top10_idx = top10_idx.tolist()
            
            if top10_idx[0] in target_indices:
                hits_1 += 1
            if any(idx in target_indices for idx in top10_idx[:5]):
                hits_5 += 1
            if any(idx in target_indices for idx in top10_idx[:10]):
                hits_10 += 1

    total = len(dataset)
    print(f"Results on {total} test samples:")
    print(f"HR@1:  {hits_1/total:.2%}")
    print(f"HR@5:  {hits_5/total:.2%}")
    print(f"HR@10: {hits_10/total:.2%}")

if __name__ == "__main__":
    evaluate()
