class Program:
    def __init__(self, initial_stats, parent_map):
        self.id_counter = 0
        yaml_code = parent_map
        prompt = "Initial parent map (starting point for optimization)"
        cycles = initial_stats['cycles']
        energy = initial_stats['energy']
        score = initial_stats['edp']

        self.programs = [{
            "id": self.id_counter,
            "yaml_code": yaml_code,
            "score": score,
            "cycles": cycles,
            "energy": energy,
            "prompt": prompt,
            "parent_id": None,
        }]
    
    def add(self, yaml_code, score, cycles, energy, prompt, parent_id):
        self.id_counter += 1
        program = {
            "id": self.id_counter,
            "yaml_code": yaml_code,
            "score": score,
            "cycles": cycles,
            "energy": energy,
            "prompt": prompt,
            "parent_id": parent_id,
        }
        self.programs.append(program)
        return program
    
    def get_best(self):
        return min(self.programs, key=lambda p: p["score"]) # optimizing edp of mapping
    
    def get_inspirations(self, exclude_id=None, k=2):
        candidates = [p for p in self.programs if p["id"] != exclude_id]
        sorted_candidates = sorted(candidates, key=lambda p: p["score"])
        return sorted_candidates[:k]  # top-k other than parent
