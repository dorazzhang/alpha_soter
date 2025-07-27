import os
import yaml
import pickle
import argparse
import numpy as np
import shutil
import subprocess

from alpha_evolve import LLMOptimizer
from cost_model import Timeloop
from database import Program


def main():
    args = parser.parse_args()

    benchmark_dir = 'Benchmarks'
    accelerator_dir = 'SpatialAccelerators'
    accelerator = args.accelerator
    workload = args.workload
    layer_id = args.layer_id
    batch_size = args.batch_size

    out_pool_dir = "tmp_out/pool-0"
    outputs_dir = "program_outputs"

    print(f"\n{'='*60}")
    print(f"ALPHA SOTER - LLM-Based Tensor Mapping Optimizer")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Accelerator: {accelerator}")
    print(f"  Workload: {workload}")
    print(f"  Layer ID: {layer_id}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Parent Map: {args.parent_map_file}")
    print(f"{'='*60}\n")

    # Clean and recreate output dirs
    for directory in [out_pool_dir, outputs_dir]:
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)

    with open(os.path.join(benchmark_dir, '{}_workload/layers.yaml'.format(workload)), 'r') as fd:
        layers = yaml.load(fd, Loader=yaml.SafeLoader)

    layer = layers[layer_id]
    print(f"Loading layer configuration: {layer}")

    with open(os.path.join(benchmark_dir, '{}_workload/{}.yaml'.format(workload, layer)), 'r') as fd:
        layer_problem = yaml.load(fd, Loader=yaml.SafeLoader)
        problem = {'problem': {
            'shape': {'name': 'CNN-Layer', 'dimensions': ['H', 'C', 'K', 'R', 'S', 'N', 'P', 'Q'],
                      'coefficients': [{'name': 'Wstride', 'default': 1},
                                       {'name': 'Hstride', 'default': 1},
                                       {'name': 'Wdilation', 'default': 1},
                                       {'name': 'Hdilation', 'default': 1}],
                      },
            'instance': {'C': 256, 'K': 512, 'R': 3, 'S': 3, 'P': 56, 'Q': 56, 'H': 1, 'N': 16,
                         'Wstride': 1, 'Hstride': 1, 'Wdilation': 1, 'Hdilation': 1
                         }}}
        if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'T2D':
            problem['problem']['shape']['data-spaces'] = [
                              {'name': 'Weights',
                               'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                              {'name': 'Outputs', 'projection': [[['N']], [['H']], [['K']],
                                                                [['R', 'Wdilation'],
                                                                 ['P', 'Wstride']],
                                                                [['S', 'Hdilation'],
                                                                 ['Q', 'Hstride']]],
                               'read-write': True},
                              {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']], [['Q']], [['P']]]}]
            problem['problem']['instance']['type'] = 'T2D'
        else:
            problem['problem']['shape']['data-spaces'] = [
                              {'name': 'Weights',
                               'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                              {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']],
                                                                [['R', 'Wdilation'],
                                                                 ['P', 'Wstride']],
                                                                [['S', 'Hdilation'],
                                                                 ['Q', 'Hstride']]]},
                              {'name': 'Outputs',
                               'projection': [[['N']], [['H']], [['K']], [['Q']], [['P']]],
                               'read-write': True}]
            problem['problem']['instance']['type'] = 'C2D'
        if 'H' in layer_problem['problem'].keys():
            problem['problem']['instance']['H'] = layer_problem['problem']['H']
        else:
            problem['problem']['instance']['H'] = 1
        if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'BMM':
            problem['problem']['instance']['N'] = layer_problem['problem']['N']
            problem['problem']['instance']['H'] = layer_problem['problem']['H'] * batch_size
        else:
            problem['problem']['instance']['N'] = layer_problem['problem']['N'] * batch_size
        problem['problem']['instance']['K'] = layer_problem['problem']['K']
        problem['problem']['instance']['C'] = layer_problem['problem']['C']
        problem['problem']['instance']['P'] = layer_problem['problem']['P']
        problem['problem']['instance']['Q'] = layer_problem['problem']['Q']
        problem['problem']['instance']['R'] = layer_problem['problem']['R']
        problem['problem']['instance']['S'] = layer_problem['problem']['S']
        problem['problem']['instance']['Wstride'] = layer_problem['problem']['Wstride']
        problem['problem']['instance']['Hstride'] = layer_problem['problem']['Hstride']
        problem['problem']['instance']['Wdilation'] = layer_problem['problem']['Wdilation']
        problem['problem']['instance']['Hdilation'] = layer_problem['problem']['Hdilation']

    with open(os.path.join(accelerator_dir, accelerator, 'problem.yaml'), 'w') as fd:
        yaml.dump(problem, fd)

    shutil.copy(os.path.join('./SpatialAccelerators', accelerator, 'arch.yaml'), os.path.join(out_pool_dir, 'arch.yaml'))
    shutil.copy(os.path.join('./SpatialAccelerators', accelerator, 'problem.yaml'), os.path.join(out_pool_dir, 'problem.yaml'))
    # shutil.copy('test_map.yaml', os.path.join(out_pool_dir, 'map.yaml')) # for test maps

    with open(args.parent_map_file, 'r') as f:
        parent_map = f.read()

    with open(os.path.join(out_pool_dir, 'map.yaml'), 'w') as f:
        f.write(parent_map)
    
    print("Evaluating initial parent map...")
    # Evaluate initial parent map
    cost_model = Timeloop(in_config_path='./SpatialAccelerators', out_config_path='./tmp_out', accelerator=accelerator)
    initial_stats = cost_model.run_config(out_pool_dir)
    initial_stats['edp'] = initial_stats['energy'] * initial_stats['cycles']
    
    print(f"Initial parent map evaluation:")
    print(f"  Cycles: {initial_stats['cycles']:,}")
    print(f"  Energy: {initial_stats['energy']:.2f} uJ")
    print(f"  EDP: {initial_stats['edp']:.2e}")
    
    program = Program(initial_stats=initial_stats, parent_map=parent_map)
    inspirations = None
    
    score = initial_stats['edp']  # score of initial map
    success_count = 0

    iterations = 4 # change based on number of loops wanted

    print(f"\nStarting optimization loop ({iterations} iterations)...")
    print(f"{'='*60}")

    for i in range(iterations):
        print(f"\n--- Generation {i+1}/{iterations} ---")
        
        # Use iterate mode when we have enough programs to provide inspirations (after 2 successful generations)
        iterate = True if success_count >= 2 else False

        max_attempts = 10 # change based on number of attempts wanted

        success, map_yaml, prompt_used = run_with_rejection_sampling(
                optimizer=LLMOptimizer(api_key=args.api_key),
                arch_path=os.path.join(accelerator_dir, accelerator, 'arch.yaml'),
                problem_path=os.path.join(accelerator_dir, accelerator, 'problem.yaml'),
                parent_map=parent_map,
                example_map="example_map.yaml",
                iterate=iterate,
                inspirations=inspirations,
                score=score,
                map_output_path='tmp_out/pool-0/map.yaml',
                max_attempts=max_attempts
            )
        # for LLM generation

        # success = True # for test maps

        if success:
            success_count += 1
            print(f"✓ Successfully generated valid mapping")
            
            print("Evaluating generated mapping...")
            cost_model = Timeloop(in_config_path='./SpatialAccelerators', out_config_path='./tmp_out',
                accelerator=accelerator)

            stats = cost_model.run_config(out_pool_dir)
            stats['edp'] = stats['energy'] * stats['cycles']

            score = stats['edp'] # edp of recently generated map
            print(f"Performance evaluation:")
            print(f"  Cycles: {stats['cycles']:,}")
            print(f"  Energy: {stats['energy']:.2f} uJ")
            print(f"  EDP: {stats['edp']:.2e}")

            parent_id = program.get_best()["id"]
            
            program.add(yaml_code = map_yaml, score = score, cycles = stats['cycles'],
                        energy = stats['energy'], prompt = prompt_used, parent_id = parent_id)

            # Save each program to a YAML file
            program_path = os.path.join("program_outputs", f"program_{program.id_counter}.yaml")

            program_data = {
                "id": program.id_counter,
                "score": score,
                "cycles": stats['cycles'],
                "energy": stats['energy'],
                "parent_id": parent_id,
                "prompt": prompt_used,
                "yaml_code": yaml.safe_load(map_yaml)
            }

            with open(program_path, "w") as f:
                yaml.dump(program_data, f)
            
            best_program = program.get_best()
            score = best_program["score"] # score of parent_map
            parent_map = best_program["yaml_code"]
            
            print(f"Best program so far: ID {best_program['id']} (EDP: {best_program['score']:.2e})")
            
            # Get inspirations when we have enough programs (after 2 successful generations)
            inspirations = program.get_inspirations(exclude_id=best_program["id"]) if success_count >= 2 else None
            
            if inspirations:
                print(f"Inspirations for next generation:")
                for p in inspirations:
                    print(f"  - ID {p['id']}: EDP {p['score']:.2e}")
        
        else:
            print(f"✗ Failed to generate valid mapping on generation {i+1}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    
    best_program = program.get_best()
    total_programs = len(program.programs)
    
    print(f"Results:")
    print(f"  Total programs generated: {total_programs}")
    print(f"  Successful generations: {success_count}/4")
    print(f"  Best program: ID {best_program['id']}")
    print(f"  Best EDP: {best_program['score']:.2e}")
    print(f"  Best cycles: {best_program['cycles']:,}")
    print(f"  Best energy: {best_program['energy']:.2f} uJ")
    
    if best_program['id'] == 0:
        print(f"  Note: Best program is the initial parent map (no improvement found)")
    else:
        improvement = ((program.programs[0]['score'] - best_program['score']) / program.programs[0]['score']) * 100
        print(f"  Improvement over parent: {improvement:.1f}%")
    
    print(f"\nAll programs saved to: program_outputs/")
    print(f"{'='*60}")




def run_with_rejection_sampling(
    optimizer,
    arch_path,
    problem_path,
    parent_map,
    example_map,
    iterate,
    inspirations,
    score,
    map_output_path,
    max_attempts
):
    # Load YAML files
    with open(arch_path, 'r') as f:
        arch_yaml = f.read()
    with open(problem_path, 'r') as f:
        problem_yaml = f.read()
    with open(example_map, 'r') as f:
        example_map = f.read()

    regenerate = False
    error_message = None
    prompt_used = None

    print(f"  Starting generation with rejection sampling (max {max_attempts} attempts)...")

    for attempt in range(max_attempts):
        if attempt == 0:
            print(f"    Attempt {attempt+1}/{max_attempts}: Initial generation")
        else:
            print(f"    Attempt {attempt+1}/{max_attempts}: Regeneration (error: {error_message[:50]}...)")

        if inspirations and attempt == 0:
            print(f"    Using {len(inspirations)} inspiration programs:")
            for p in inspirations:
                print(f"      - ID {p['id']}: EDP {p['score']:.2e}")

        # Generate new map.yaml from LLM
        if attempt == 0:
            map_yaml, prompt_used = optimizer.optimize(
                arch_yaml=arch_yaml,
                problem_yaml=problem_yaml,
                parent_map=parent_map,
                example_map=example_map,
                score = score,
                regenerate=regenerate,
                iterate=iterate,
                inspirations=inspirations,
                error_message=error_message,
                prompt = prompt_used
            )
        else:
            map_yaml, _ = optimizer.optimize(
                arch_yaml=arch_yaml,
                problem_yaml=problem_yaml,
                parent_map=parent_map,
                example_map=example_map,
                score = score,
                regenerate=regenerate,
                iterate=iterate,
                inspirations=inspirations,
                error_message=error_message,
                prompt = prompt_used
            )

        # Save new map.yaml (overwrite previous)
        with open(map_output_path, 'w') as f:
            f.write(map_yaml)

        try:
            yaml.safe_load(map_yaml)
            subprocess.run(
                ['timeloop-model', 'arch.yaml', 'problem.yaml', 'map.yaml'],
                cwd=os.path.dirname(map_output_path),
                check=True,
                capture_output=True,
                text=True
            )
            print(f"    ✓ Attempt {attempt+1} successful!")
            return True, map_yaml, prompt_used
        except (yaml.YAMLError, subprocess.CalledProcessError) as e:
            if isinstance(e, yaml.YAMLError):
                print(f"    ✗ Invalid YAML format")
                error_message = "Invalid YAML format. Please make sure there is no additional text and the proper syntax is followed"
            else:
                print(f"    ✗ Timeloop validation failed")
                error_message = e.stderr

            regenerate = True
            with open(map_output_path, 'r') as f:
                parent_map = f.read()  # use failed output as new parent map

    print(f"    ✗ Failed to generate valid mapping after {max_attempts} attempts")
    return False, None, None





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', required=True, help='Anthropic API key')
    parser.add_argument('--accelerator', required=True, type=str, help='accelerator accelerator')
    parser.add_argument('--workload', required=True, type=str)
    parser.add_argument('--layer_id', required=True, type=int)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--parent_map_file', required=True, help='Path to a parent map.yaml')
    main()