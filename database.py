class Program:
    def __init__(self):
        self.id_counter = 0
        with open('example_map.yaml', 'r') as fd:
            yaml_code = fd.read()
        with open('initial_prompt.txt', 'r') as fd:
            prompt = fd.read()
        score = 411041792 + 20333.22 + 8357803185930.24
        self.programs = [{
            "id": self.id_counter,
            "yaml_code": yaml_code,
            "score": score,
            "prompt": prompt,
            "parent_id": None,
        }]
    
    def add(self, yaml_code, score, prompt, parent_id):
        self.id_counter += 1
        program = {
            "id": self.id_counter,
            "yaml_code": yaml_code,
            "score": score,
            "prompt": prompt,
            "parent_id": parent_id,
        }
        self.programs.append(program)
        return program
    
    def get_best(self):
        return min(self.programs, key=lambda p: p["score"])
    
    def get_inspirations(self, exclude_id=None, k=2):
        candidates = [p for p in self.programs if p["id"] != exclude_id]
        sorted_candidates = sorted(candidates, key=lambda p: p["score"])
        return sorted_candidates[:k]  # top-k other than parent
