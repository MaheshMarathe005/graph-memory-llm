import sys; import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from src.qwen_adapter_model import QwenPointerAdapter

def test_adapter():
    print("Initializing QwenPointerAdapter...")
    # Small numbers for fast smoke test
    num_nodes = 1000
    num_relations = 5
    hidden_dim = 256
    
    # We will use the smallest Qwen for smoke test if available, 
    # but since Qwen1.5-0.5B is available, let's use that to test shapes quickly
    model = QwenPointerAdapter(
        qwen_model_name="Qwen/Qwen1.5-0.5B",
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=hidden_dim,
        max_hops=5
    )
    
    batch_size = 2
    seq_len = 10
    vocab_size = model.qwen.config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    prompt_lengths = [6, 8] # Target tokens are the rest
    
    p_initial = torch.rand(batch_size, num_nodes)
    p_initial = p_initial / p_initial.sum(dim=1, keepdim=True)
    
    # Fake sparse adjacency matrices
    adj_matrices = []
    for _ in range(2 * num_relations):
        # Create a random sparse matrix
        indices = torch.randint(0, num_nodes, (2, 5000))
        values = torch.rand(5000)
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).float()
        adj_matrices.append(adj)
        
    target_node_indices = torch.tensor([50, 150])
    
    print("Running forward pass...")
    p_total, p_final, p_copy, p_lm = model(
        input_ids,
        attention_mask,
        prompt_lengths,
        p_initial,
        adj_matrices,
        target_node_indices
    )
    
    print("Output shapes:")
    print(f"p_total: {p_total.shape}")
    print(f"p_final: {p_final.shape}")
    print(f"p_copy: {p_copy.shape}")
    print(f"p_lm: {p_lm.shape}")
    
    assert p_total.shape == (batch_size,)
    assert p_final.shape == (batch_size, num_nodes)
    assert p_copy.shape == (batch_size, 1)
    assert p_lm.shape == (batch_size,)
    
    print("Forward pass successful!")

if __name__ == "__main__":
    test_adapter()
