import modal
import os
import sys

# Build image with huggingface requirements
app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "datasets")
)

# Mount the check file
app_image = app_image.add_local_file("check_counterfactuals.py", "/app/check_counterfactuals.py")

# Initialize the Modal App
app = modal.App("qwen-counterfactual-test")

@app.cls(
    image=app_image,
    gpu="a10g",
    timeout=3600,
)
class CheckCounterfactuals:
    @modal.enter()
    def initialize(self):
        sys.path.insert(0, "/app")
        os.chdir("/app")

    @modal.method()
    def run(self):
        import check_counterfactuals
        check_counterfactuals.main()

@app.local_entrypoint()
def main():
    print("🚀 Dispatching check_counterfactuals to Modal GPU...")
    checker = CheckCounterfactuals()
    checker.run.remote()
    print("✅ Completed Modal run.")

if __name__ == "__main__":
    print("Run using: modal run modal_test_counterfactuals.py")
