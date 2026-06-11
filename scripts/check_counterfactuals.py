import sys; import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Phase 0: Counterfactual MetaQA Design
# The premise: We rewire well-known MetaQA facts so they are plausible but false.
# This script proves that a plain frozen Qwen-1.5B will confidently predict the TRUE fact,
# meaning it will score ~0% on our counterfactual set when the graph is OFF.
# When the graph is ON (memory-ON), our adapter will override this belief.

def main():
    model_id = "Qwen/Qwen1.5-1.8B"
    print(f"Loading {model_id}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load model locally: {e}")
        print("Please run this script on Modal or an environment with model access.")
        return

    if device == "cpu":
        model = model.to("cpu")

    # Counterfactual rewiring (plausible but false)
    # Format: Movie, True Director (what Qwen knows), Fake Director (what our graph will say)
    eval_cases = [
        ("Inception", "Christopher Nolan", "Steven Spielberg"),
        ("The Matrix", "Lana Wachowski", "James Cameron"),
        ("Avatar", "James Cameron", "Christopher Nolan"),
        ("Titanic", "James Cameron", "Quentin Tarantino"),
        ("Pulp Fiction", "Quentin Tarantino", "Lana Wachowski"),
        ("The Godfather", "Francis Ford Coppola", "Martin Scorsese"),
        ("Goodfellas", "Martin Scorsese", "Francis Ford Coppola"),
        ("Jurassic Park", "Steven Spielberg", "George Lucas"),
        ("Star Wars", "George Lucas", "Steven Spielberg"),
        ("Interstellar", "Christopher Nolan", "Denis Villeneuve")
    ]
    
    print("\n" + "="*60)
    print("PHASE 0: CHECK COUNTERFACTUALS (PLAIN QWEN-1.5B)")
    print("="*60)
    
    num_correct_true = 0
    num_correct_fake = 0
    
    for movie, true_ans, fake_ans in eval_cases:
        prompt = f"Q: Who directed the movie {movie}?\nA: The director of the movie {movie} is"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0, # Greedy decode for confident belief
                do_sample=False
            )
            
        generation = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Check matching
        # We consider a hit if the generated text contains the last name
        true_hit = true_ans.split()[-1].lower() in generation.lower()
        fake_hit = fake_ans.split()[-1].lower() in generation.lower()
        
        if true_hit:
            num_correct_true += 1
        if fake_hit:
            num_correct_fake += 1
            
        print(f"\nMovie: {movie}")
        print(f"  Graph (Counterfactual) target: {fake_ans}")
        print(f"  Qwen's Raw Generation: '{generation}'")
        if true_hit:
            print("  Verdict: ❌ Confidently Wrong (predicted true fact from weights)")
        elif fake_hit:
            print("  Verdict: ⚠️ Accidental Fake Hit (this rewiring is invalid)")
        else:
            print("  Verdict: ? Uncertain (did not predict either)")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Cases: {len(eval_cases)}")
    print(f"Predicted True Fact (Memory-OFF confidently wrong): {num_correct_true}/{len(eval_cases)}")
    print(f"Predicted Fake Fact (Counterfactual leakage): {num_correct_fake}/{len(eval_cases)}")
    
    if num_correct_fake == 0 and num_correct_true > len(eval_cases) * 0.7:
        print("\n✅ GATE PASSED: Qwen-1.5B answers counterfactuals confidently wrong (~0% fake hits, high true hits).")
        print("   'Memory-OFF -> Wrong' will be a valid proof of graph reliance.")
    else:
        print("\n❌ GATE FAILED: Qwen-1.5B is not confidently wrong. Rewire the facts further.")

if __name__ == "__main__":
    main()
