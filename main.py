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

    with open(os.path.join(benchmark_dir, '{}_workload/layers.yaml'.format(workload)), 'r') as fd:
        layers = yaml.load(fd, Loader=yaml.SafeLoader)

    layer = layers[layer_id]
    print(accelerator, workload, batch_size, layer_id, layer)

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

    out_pool_dir = './tmp_out/pool-0'
    os.makedirs(out_pool_dir, exist_ok=True)

    shutil.copy(os.path.join('./SpatialAccelerators', accelerator, 'arch.yaml'), os.path.join(out_pool_dir, 'arch.yaml'))
    shutil.copy(os.path.join('./SpatialAccelerators', accelerator, 'problem.yaml'), os.path.join(out_pool_dir, 'problem.yaml'))
    # shutil.copy('test_map.yaml', os.path.join(out_pool_dir, 'map.yaml')) # for test maps

    with open(args.parent_map_file, 'r') as f:
        parent_map = f.read()
    
    program = Program()
    inspirations = None
    score = 8358214248055.46 # score of initial map
    os.makedirs("program_outputs", exist_ok=True)

    for i in range(4): # change based on number of loops wanted


        iterate = True if i > 1 else False

        success, map_yaml, prompt_used = run_with_rejection_sampling(
                optimizer=LLMOptimizer(api_key=args.api_key),
                arch_path=os.path.join(accelerator_dir, accelerator, 'arch.yaml'),
                problem_path=os.path.join(accelerator_dir, accelerator, 'problem.yaml'),
                parent_map=parent_map,
                iterate=iterate,
                inspirations=inspirations,
                score=score,
                map_output_path='tmp_out/pool-0/map.yaml',
                max_attempts=10
            )
        # for LLM generation

        # success = True # for test maps

        if success:

            cost_model = Timeloop(in_config_path='./SpatialAccelerators', out_config_path='./tmp_out',
                accelerator=accelerator)

            stats = cost_model.run_config(out_pool_dir)
            stats['edp'] = stats['energy'] * stats['cycles']

            score_sum = stats['edp'] + stats['energy'] + stats['cycles'] #score of recently generated map
            print("Performance stats:")
            print(stats)

            parent_id = program.get_best()["id"]
            
            program.add(yaml_code = map_yaml, score = score_sum, prompt = prompt_used, parent_id = parent_id)

            # Save each program to a YAML file
            program_path = os.path.join("program_outputs", f"program_{program.id_counter}.yaml")

            program_data = {
                "id": program.id_counter,
                "score": score_sum,
                "parent_id": parent_id,
                "prompt": prompt_used,
                "yaml_code": yaml.safe_load(map_yaml)
            }

            with open(program_path, "w") as f:
                yaml.dump(program_data, f)
            
            best_program = program.get_best()
            score = best_program["score"] # score of parent_map
            parent_map = best_program["yaml_code"]
            inspirations = program.get_inspirations(exclude_id=best_program["id"])
        
        else:
            print(f"did not successfully generate on {i}th iteration")




def run_with_rejection_sampling(
    optimizer,
    arch_path,
    problem_path,
    parent_map,
    inspirations,
    score,
    map_output_path,
    max_attempts=10
):
    # Load YAML files
    with open(arch_path, 'r') as f:
        arch_yaml = f.read()
    with open(problem_path, 'r') as f:
        problem_yaml = f.read()

    regenerate = False
    error_message = None
    prompt_used = None

    for attempt in range(max_attempts):
        print(f"\n Attempt #{attempt+1}")

        if inspirations:
            print("Inspirations being used:")
            for p in inspirations:
                print(f"  - ID {p['id']}, Score: {p['score']}")


        # Generate new map.yaml from LLM
        map_yaml, prompt_used = optimizer.optimize(
            arch_yaml=arch_yaml,
            problem_yaml=problem_yaml,
            parent_map=parent_map,
            score = score,
            regenerate=regenerate,
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
            print("timeloop-model ran successfully!")
            return True, map_yaml, prompt_used
        except (yaml.YAMLError, subprocess.CalledProcessError) as e:
            if isinstance(e, yaml.YAMLError):
                print("Invalid YAML generated.")
                error_message = "Invalid YAML format. Please make sure there is no additional text and the proper syntax is followed"
            else:
                print("timeloop-model failed:")
                print(e.stderr)
                error_message = e.stderr

            regenerate = True
            with open(map_output_path, 'r') as f:
                parent_map = f.read()  # use failed output as new parent map

    print(f"Failed to generate a valid map.yaml after max {max_attempts} attempts.")
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