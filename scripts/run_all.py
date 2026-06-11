import sys; import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modal

app = modal.App("metaqa-unified-runner")

volume = modal.Volume.from_name("metaqa-checkpoints", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "accelerate", "tqdm"
).add_local_python_source("qwen_adapter_model", "differentiable_graph_executor", "train_metaqa", "cot_comparison")

@app.function(image=image, gpu="A100", timeout=7200, volumes={"/checkpoints": volume})
def run_unified(kb_content: str, datasets: dict):
    import os
    with open("kb.txt", "w") as f:
        f.write(kb_content)
        
    for hop in [1, 2, 3]:
        os.makedirs(f"{hop}-hop/vanilla", exist_ok=True)
        with open(f"{hop}-hop/vanilla/qa_train.txt", "w") as f:
            f.write(datasets[hop]['train'])
        with open(f"{hop}-hop/vanilla/qa_dev.txt", "w") as f:
            f.write(datasets[hop]['dev'])
        with open(f"{hop}-hop/vanilla/qa_test.txt", "w") as f:
            f.write(datasets[hop]['test'])
            
    from train_metaqa import train_phase_1
    from cot_comparison import run_benchmark
    
    # 1. Train all models
    success_flags = {}
    for hop in [1, 2, 3]:
        print(f"\n{'='*50}\nTRAINING {hop}-HOP MODEL\n{'='*50}")
        best_hr = train_phase_1(hop_count=hop, ckpt_dir="/checkpoints")
        print(f"[{hop}-HOP] Training completed with Dev HR: {best_hr:.2%}")
        if best_hr < 0.20:
            print(f"[WARNING] {hop}-hop HR is suspiciously low ({best_hr:.2%}). Sanity check failed!")
            success_flags[hop] = False
        else:
            success_flags[hop] = True
        
    # 2. Benchmark all models
    for hop in [1, 2, 3]:
        if not success_flags[hop]:
            print(f"\nSkipping benchmark for {hop}-hop due to training failure.")
            continue
        print(f"\n{'='*50}\nBENCHMARKING {hop}-HOP MODEL\n{'='*50}")
        run_benchmark(hop_count=hop, num_samples=100, ckpt_dir="/checkpoints")
        
@app.local_entrypoint()
def main():
    print("Reading kb.txt...")
    with open("kb.txt", "r") as f:
        kb_content = f.read()
        
    datasets = {}
    for hop in [1, 2, 3]:
        datasets[hop] = {}
        for split in ['train', 'dev', 'test']:
            print(f"Reading {hop}-hop {split}...")
            with open(f"{hop}-hop/vanilla/qa_{split}.txt", "r") as f:
                datasets[hop][split] = f.read()
                
    run_unified.remote(kb_content, datasets)
