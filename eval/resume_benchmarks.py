import sys; import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modal


app = modal.App("metaqa-resume-benchmarks")
volume = modal.Volume.from_name("metaqa-checkpoints")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "tqdm", "scikit-learn")
    .add_local_python_source(
        "qwen_adapter_model",
        "differentiable_graph_executor",
        "train_metaqa",
        "cot_comparison",
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/checkpoints": volume},
)
def run_hop_benchmark(
    hop_count: int,
    num_samples: int,
    kb_content: str,
    test_content: str,
):
    import os

    with open("kb.txt", "w") as f:
        f.write(kb_content)

    test_dir = f"{hop_count}-hop/vanilla"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/qa_test.txt", "w") as f:
        f.write(test_content)

    checkpoint_path = f"/checkpoints/phase1_checkpoint_{hop_count}hop.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    from cot_comparison import run_benchmark

    run_benchmark(
        hop_count=hop_count,
        num_samples=num_samples,
        ckpt_dir="/checkpoints",
    )


@app.local_entrypoint()
def main(num_samples: int = 100):
    with open("kb.txt", "r") as f:
        kb_content = f.read()

    jobs = []
    for hop_count in (1, 2, 3):
        with open(f"{hop_count}-hop/vanilla/qa_test.txt", "r") as f:
            test_content = f.read()

        jobs.append(
            (
                hop_count,
                run_hop_benchmark.spawn(
                    hop_count,
                    num_samples,
                    kb_content,
                    test_content,
                ),
            )
        )

    for hop_count, job in jobs:
        print(f"\nWaiting for {hop_count}-hop benchmark...")
        job.get()
        print(f"Completed {hop_count}-hop benchmark.")
