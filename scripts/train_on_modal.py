import sys; import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modal

app = modal.App("metaqa-train-phase-1")

image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "accelerate", "tqdm"
).add_local_python_source("qwen_adapter_model", "differentiable_graph_executor", "train_metaqa")

@app.function(image=image, gpu="A10G", timeout=3600)
def run_phase1(kb_content: str, qa_train_content: str, hop_count: int):
    import os
    with open("kb.txt", "w") as f:
        f.write(kb_content)
    os.makedirs(f"{hop_count}-hop/vanilla", exist_ok=True)
    with open(f"{hop_count}-hop/vanilla/qa_train.txt", "w") as f:
        f.write(qa_train_content)

    from train_metaqa import train_phase_1
    train_phase_1(hop_count=hop_count)
    
    ckpt_name = f"phase1_checkpoint_{hop_count}hop.pt"
    with open(ckpt_name, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main(hop_count: int = 2):
    print("Reading kb.txt...")
    with open("kb.txt", "r") as f:
        kb_content = f.read()
    print(f"Reading {hop_count}-hop qa_train.txt...")
    with open(f"{hop_count}-hop/vanilla/qa_train.txt", "r") as f:
        qa_train_content = f.read()

    ckpt_bytes = run_phase1.remote(kb_content, qa_train_content, hop_count)
    ckpt_name = f"phase1_checkpoint_{hop_count}hop.pt"
    print(f"Writing {ckpt_name} locally...")
    with open(ckpt_name, "wb") as f:
        f.write(ckpt_bytes)
