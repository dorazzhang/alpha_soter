import os
import yaml
import pickle
import argparse
import numpy as np
import shutil
import subprocess

from alpha_evolve import LLMOptimizer
from cost_model import Timeloop


def main():
    args = parser.parse_args()

    with open(args.arch_file, 'r') as fd:
        arch_contents = fd.read()
    with open(args.problem_file, 'r') as fd:
        problem_contents = fd.read()
    with open(args.baseline_map_file, 'r') as fd:
        baseline_contents = fd.read()

    optimizer = LLMOptimizer(api_key=args.api_key)
    new_mapping = optimizer.optimize(arch_contents, problem_contents, baseline_contents)

    if new_mapping is not None:
        print("generated valid YAML")

        with open(args.output_file, 'w') as fd:
            fd.write(new_mapping)
        print(f"New map saved to: {args.output_file}")

        accelerator = 'Simba'

        cost_model = Timeloop(in_config_path='./SpatialAccelerators', out_config_path='./tmp_out',
            accelerator=accelerator)

        out_pool_dir = './tmp_out/pool-0'
        os.makedirs(out_pool_dir, exist_ok=True)

        directory = args.output_file
        shutil.copy('./SpatialAccelerators/Simba/arch.yaml', os.path.join(out_pool_dir, 'arch.yaml'))
        shutil.copy('./SpatialAccelerators/Simba/problem.yaml', os.path.join(out_pool_dir, 'problem.yaml'))
        shutil.copy(directory, os.path.join(out_pool_dir, 'map.yaml'))
        print("copied arch, problem, and map.yaml files")

        command = ['timeloop-model', 'arch.yaml', 'problem.yaml', 'map.yaml']
        subprocess.run(command, cwd=out_pool_dir, check=True)
        print("generated timeloop-model files")

        stats = cost_model.run_config(out_pool_dir)
        stats['edp'] = stats['energy'] * stats['cycles']

        print("Performance stats:")
        print(stats)
    else:
        print("Error: Failed to generate valid YAML")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', required=True, help='Anthropic API key')
    parser.add_argument('--arch_file', required=True, help='Path to arch.yaml')
    parser.add_argument('--problem_file', required=True, help='Path to problem.yaml')
    parser.add_argument('--baseline_map_file', required=True, help='Path to a baseline map.yaml')
    parser.add_argument('--output_file', required=False, default='new_map.yaml', help='File to save the new map to')
    main()