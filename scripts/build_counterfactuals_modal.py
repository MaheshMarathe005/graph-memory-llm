import sys; import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import json
import random
import torch
import modal

app = modal.App("metaqa-counterfactuals")

# Use same image as before
image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "accelerate", "datasets", "tqdm"
)

# Mount the required local files
metaqa_mounts = [
    modal.mount.Mount.from_local_file("kb.txt", remote_path="/root/kb.txt"),
    modal.mount.Mount.from_local_dir("2-hop/vanilla", remote_path="/root/2-hop/vanilla")
]

@app.function(image=image, gpu="A10G", timeout=3600, mounts=metaqa_mounts)
def build_and_filter_counterfactuals():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    import os
    os.chdir("/root")
    
    # 1. Load KB
    print("Loading KB...")
    relations = {} # (head, rel) -> set(tails)
    entities = set()
    incoming_rels = {} # entity -> set of relations pointing to it
    with open("kb.txt", "r") as f:
        for line in f:
            s, r, o = line.strip().split('|')
            relations.setdefault((s, r), set()).add(o)
            incoming_rels.setdefault(o, set()).add(r)
            entities.add(s)
            entities.add(o)
            
    # Group entities by incoming relation to pick plausible fakes
    plausible_fakes = {}
    for ent, rels in incoming_rels.items():
        for r in rels:
            plausible_fakes.setdefault(r, []).append(ent)
            
    # 2. Load queries
    # We will use 1-hop for simplicity of building the counterfactuals if needed, 
    # but the user requested multi-hop. Let's use 2-hop.
    print("Loading 2-hop queries...")
    queries = []
    with open("2-hop/vanilla/qa_test.txt", "r") as f:
        for line in f:
            q, ans_str = line.strip().split('\t')
            answers = ans_str.split('|')
            start = q.find('[')
            end = q.find(']')
            if start != -1 and end != -1:
                ent = q[start+1:end]
                queries.append({"q": q, "ent": ent, "answers": answers})
                
    # 3. Create candidates to rewire
    # For simplicity, we will find queries whose answers are directors, and rewire the directed_by edge.
    # Actually, MetaQA 2-hop has many relation types. 
    # Let's just randomly rewire ANY target node to another node of the same relation type.
    rel_objects = {}
    for (s, r), objs in relations.items():
        rel_objects.setdefault(r, set()).update(objs)
    for r in rel_objects:
        rel_objects[r] = list(rel_objects[r])
        
    print("Loading Qwen-1.5B...")
    model_name = "Qwen/Qwen1.5-1.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    valid_cases = []
    
    # We'll just randomly try cases until we have 500 valid ones.
    random.shuffle(queries)
    
    for q_data in tqdm(queries):
        if len(valid_cases) >= 500:
            break
            
        # We need a prompt for Qwen to answer the question zero-shot.
        question = q_data['q'].replace('[', '').replace(']', '')
        prompt = f"Q: {question}\nA:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0,
                do_sample=False
            )
            
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Check if Qwen confidently predicts ONE of the true answers
        predicted_true = False
        for true_ans in q_data['answers']:
            if true_ans.lower() in gen_text.lower():
                predicted_true = True
                break
                
        if predicted_true:
            true_ans_rep = q_data['answers'][0]
            possible_rels = list(incoming_rels.get(true_ans_rep, set()))
            
            def get_plausible_fake():
                if possible_rels:
                    rel_to_use = random.choice(possible_rels)
                    return random.choice(plausible_fakes[rel_to_use])
                else:
                    return random.choice(list(entities))
                    
            fake_ans = get_plausible_fake()
            while fake_ans in q_data['answers']:
                fake_ans = get_plausible_fake()
                
            valid_cases.append({
                "question": q_data['q'],
                "head_entity": q_data['ent'],
                "true_answers": q_data['answers'],
                "fake_answer": fake_ans,
                "qwen_raw": gen_text
            })
            
    print(f"\nFound {len(valid_cases)} confidently wrong counterfactual candidates!")
    return valid_cases

@app.local_entrypoint()
def main():
    print("Dispatching to Modal...")
    cases = build_and_filter_counterfactuals.remote()
    with open("counterfactual_eval_set.json", "w") as f:
        json.dump(cases, f, indent=2)
    print("Saved to counterfactual_eval_set.json")

if __name__ == "__main__":
    main()
EOF
