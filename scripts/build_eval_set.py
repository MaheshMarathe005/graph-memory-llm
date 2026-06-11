import sys; import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
from tqdm import tqdm

from check_qwen_modal import app, check_qwen_batch

def build_counterfactual_eval_set():
    # 1. Load KB
    print("Loading KB locally...")
    relations = {} # (head, rel) -> set(tails)
    incoming_rels = {} # entity -> set of relations pointing to it
    entities = set()
    with open("kb.txt", "r") as f:
        for line in f:
            s, r, o = line.strip().split('|')
            relations.setdefault((s, r), set()).add(o)
            incoming_rels.setdefault(o, set()).add(r)
            entities.add(s)
            entities.add(o)
            
    plausible_fakes = {}
    for ent, rels in incoming_rels.items():
        for r in rels:
            plausible_fakes.setdefault(r, []).append(ent)
            
    # 2. Load 2-hop queries
    print("Loading 2-hop queries locally...")
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
                
    random.shuffle(queries)
    
    valid_cases = []
    batch_size = 200
    total_needed = 500
    
    query_idx = 0
    while len(valid_cases) < total_needed and query_idx < len(queries):
        # Prepare a batch
        batch = queries[query_idx:query_idx+batch_size]
        query_idx += batch_size
        
        prompts = [f"Q: {q['q'].replace('[', '').replace(']', '')}\nA:" for q in batch]
        
        print(f"Sending batch of {len(prompts)} to Modal Qwen-1.5B...")
        with app.run():
            # Call modal remote
            results = check_qwen_batch.remote(prompts)
            
        for q_data, gen_text in zip(batch, results):
            # Check if Qwen predicted one of the true answers confidently
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
                
        print(f"Valid counterfactuals collected so far: {len(valid_cases)}/{total_needed}")
        
    # Trim to exactly total_needed
    valid_cases = valid_cases[:total_needed]
    with open("counterfactual_eval_set.json", "w") as f:
        json.dump(valid_cases, f, indent=2)
    print(f"Saved {len(valid_cases)} cases to counterfactual_eval_set.json")

if __name__ == "__main__":
    build_counterfactual_eval_set()
