import os
import re

class MetaQALinker:
    def __init__(self, kb_path="kb.txt"):
        self.entities = set()
        self.normalized_to_raw = {}
        self._load_kb(kb_path)
        
    def _load_kb(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 3:
                    s, r, o = parts
                    self.entities.add(s)
                    self.entities.add(o)
                    
        for e in self.entities:
            e_lower = e.lower()
            self.normalized_to_raw[e_lower] = e
            
            # Handle "The Matrix" -> "Matrix, The"
            if e.endswith(', The'):
                alt = "the " + e[:-5].lower()
                self.normalized_to_raw[alt] = e
            if e.endswith(', A'):
                alt = "a " + e[:-3].lower()
                self.normalized_to_raw[alt] = e
            if e.endswith(', An'):
                alt = "an " + e[:-4].lower()
                self.normalized_to_raw[alt] = e
                
            # Handle reverse (if KB has "The Matrix" but question has "Matrix, The", though rare)
            if e_lower.startswith('the '):
                alt = e_lower[4:] + ', the'
                self.normalized_to_raw[alt] = e

    def link(self, text):
        if text in self.entities:
            return text
        
        text_lower = text.lower()
        if text_lower in self.normalized_to_raw:
            return self.normalized_to_raw[text_lower]
            
        return None

def evaluate_linker(linker, file_path):
    if not os.path.exists(file_path):
        return 0, 0, set()
        
    total = 0
    correct = 0
    failures = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Format: "[Entity] question text\tAnswer1|Answer2"
            q, _ = line.strip().split('\t', 1)
            start = q.find('[')
            end = q.find(']')
            if start != -1 and end != -1:
                entity_str = q[start+1:end]
                total += 1
                linked = linker.link(entity_str)
                if linked is not None:
                    correct += 1
                else:
                    failures.add(entity_str)
                    
    return total, correct, failures

if __name__ == "__main__":
    linker = MetaQALinker("kb.txt")
    print(f"Loaded {len(linker.entities)} entities from KB.")
    
    files_to_check = [
        "1-hop/vanilla/qa_test.txt",
        "2-hop/vanilla/qa_test.txt",
        "3-hop/vanilla/qa_test.txt"
    ]
    
    total_q = 0
    total_c = 0
    all_fails = set()
    
    for f in files_to_check:
        t, c, fails = evaluate_linker(linker, f)
        total_q += t
        total_c += c
        all_fails.update(fails)
        print(f"{f}: {c}/{t} ({c/t*100:.2f}%)" if t > 0 else f"{f}: Not found")
        
    if total_q > 0:
        acc = total_c / total_q * 100
        print(f"\nOverall Accuracy: {acc:.2f}%")
        print(f"\nSample failures ({min(10, len(all_fails))} of {len(all_fails)}):")
        for fail in list(all_fails)[:10]:
            print(f"  - {fail}")
